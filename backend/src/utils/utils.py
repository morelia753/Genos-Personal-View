# 标准库（内置模块）
import os
import random
import logging
import datetime
from typing import Optional, List, Dict, Union, Tuple

# 第三方库（pip 安装的包）
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModel
from safetensors.torch import load_file
from src.utils.dist import dist_print 
import json
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch


def setup_logging(
    output_base_dir: str,
    timestamp: Optional[str] = None,
    log_level: int = logging.INFO,
    log_filename: str = None,
) -> str:
    """
    配置日志系统：仅 rank 0 写入日志文件，所有 rank 输出到控制台。
    返回日志文件路径（所有进程都返回相同值）。
    """
    # 等待所有进程进入，确保 dist 初始化完成
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # 只有 rank 0 生成 timestamp
    if rank == 0:
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = ""

    # 广播 timestamp 给所有进程
    if dist.is_available() and dist.is_initialized():
        timestamp_list = [timestamp]
        dist.broadcast_object_list(timestamp_list, src=0)
        timestamp = timestamp_list[0]

    # 定义日志目录
    log_dir = os.path.join(output_base_dir, "logs")
    log_filepath = None

    # 获取 logger
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()  # 清除已有 handler

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )

    # 所有进程都添加控制台输出（可选）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 仅 rank 0 创建目录并添加文件 handler
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)

        if log_filename is None:
            log_filename = f"training_{timestamp}.log"
        else:
            log_filename = f"{log_filename}_{timestamp}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 广播 log_filepath 给所有进程（确保 callback 能拿到）
    if dist.is_available() and dist.is_initialized():
        log_filepath_list = [log_filepath]
        dist.broadcast_object_list(log_filepath_list, src=0)
        log_filepath = log_filepath_list[0]

    # 所有进程都设置日志级别
    logger.setLevel(log_level)

    # rank 0 打印日志路径
    if rank == 0:
        logger.info(f"✅ 日志系统初始化完成，日志文件: {log_filepath}")

    return log_filepath  # 所有进程都返回相同的路径

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# track mean处理逻辑（新增）
def _as_track_means_view(
    track_means: Union[float, int, torch.Tensor],
    target_shape: Tuple[int, int, int],
    name: str,
) -> torch.Tensor:
    """
    Convert track_means into a broadcastable tensor view for (B, L, C) targets/preds.

    Allowed:
      - scalar: float/int or 0-dim tensor
      - per-track global: (C,)  (only if you explicitly want it; no fallback)
      - per-sample: (B, C)     (this is what you want in personal genome training)

    Returns:
      tm_view shaped as (B, 1, C) to broadcast over sequence length L.

    Any mismatch -> raise RuntimeError (NO silent fallback).
    """
    B, L, C = target_shape

    # normalize to tensor
    if isinstance(track_means, (int, float)):
        tm = torch.tensor(track_means, dtype=torch.float32)
    elif isinstance(track_means, torch.Tensor):
        tm = track_means
    else:
        raise RuntimeError(f"[{name}] track_means must be float/int/torch.Tensor, got {type(track_means)}")

    if tm.ndim == 0:
        # scalar -> (1,1,1) broadcastable
        tm_view = tm.view(1, 1, 1)
        return tm_view

    if tm.ndim == 1:
        # (C,)
        if tm.numel() != C:
            raise RuntimeError(f"[{name}] track_means shape (C,) mismatch: got {tuple(tm.shape)} but C={C}")
        return tm.view(1, 1, C)

    if tm.ndim == 2:
        # (B, C)
        if tm.shape[0] != B or tm.shape[1] != C:
            raise RuntimeError(
                f"[{name}] track_means shape (B,C) mismatch: got {tuple(tm.shape)} but expected ({B},{C})"
            )
        return tm.view(B, 1, C)

    raise RuntimeError(f"[{name}] track_means has unsupported ndim={tm.ndim}, shape={tuple(tm.shape)}")



# 这里注意;这个函数仅用于推理！！！
def load_finetuned_model(
    model_class,
    model_path: str,
    ckpt_path: str,
    use_flash_attn: bool = False,
    trust_remote_code: bool = True,
    revision: str = "main",
    device: str = "auto",  # 修改默认值为 auto 以支持多卡
    torch_dtype: torch.dtype = torch.bfloat16, # 10B模型强烈建议默认用 bf16
    model_init_args: Optional[List] = None,          
    model_init_kwargs: Optional[Dict] = None,        
) -> torch.nn.Module:
    """
    修复版加载函数：
    1. 使用 Meta Device 初始化，避免内存/显存瞬间爆表。
    2. 使用 device_map="auto" 自动实现多 GPU 张量并行/模型并行。
    """
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch

    # 1. 加载配置
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        revision=revision
    )

    # 设置 Attention 实现
    if use_flash_attn:
        config._attn_implementation = "flash_attention_2"
    
    init_args = model_init_args or []
    init_kwargs = model_init_kwargs or {}

    # 2. 在 Meta Device 上创建空模型（不占显存）
    # init_empty_weights 会拦截所有的内存分配请求
    with init_empty_weights():
        base_model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
        model = model_class(base_model, *init_args, **init_kwargs)

    # 3. 分发模型到所有可用的 GPU
    # load_checkpoint_and_dispatch 会做以下几件事：
    #   a. 自动计算每张卡的显存，决定每一层放哪
    #   b. 逐个加载权重分片，避免 load_file 一次性占用几百G内存
    #   c. 自动处理多卡之间的数据流转
    model = load_checkpoint_and_dispatch(
        model,
        ckpt_path,
        device_map=device,
        # 这里的 dtype 很重要，必须与权重一致，否则会因自动转型导致显存翻倍
        dtype=torch_dtype,
        # no_split_module_classes 是为了防止一个 Transformer Block 被拆在两张卡上导致速度极慢
        # 请根据 GenOs 实际的层类名修改，通常是 "GenOsBlock" 或 "DecoderLayer"
        no_split_module_classes=["GenOsBlock", "LlamaDecoderLayer"], 
    )

    # 4. 设置为评估模式
    model.eval()

    return model