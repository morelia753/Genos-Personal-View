# 标准库（内置模块）
import os
import logging
import sys

# 第三方库（pip 安装的包）
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, barrier
import datetime



def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def dist_print(*args, **kwargs) -> None:
    """
    Print only from the main process (rank 0) in distributed training.
    Prevents duplicate outputs in multi-GPU settings.

    Args:
        *args: Arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    # 检查分布式训练是否已初始化
    if dist.is_available() and dist.is_initialized():
        # 只在 rank 0 进程打印
        if dist.get_rank() == 0:
            logging.info(*args, **kwargs)
            sys.stdout.flush()
    else:
        # 单卡运行时直接打印
        logging.info(*args, **kwargs)
        sys.stdout.flush()


def setup_distributed():
    """
    初始化分布式训练环境（DDP），自动判断是否为多卡模式。
    
    Returns:
        tuple: (local_rank, world_size, is_distributed)
            - local_rank (int): 当前进程的本地 GPU 编号（单卡为 0）
            - world_size (int): 总进程数（单卡为 1）
            - is_distributed (bool): 是否启用了分布式
    """
    # 检查是否已在分布式环境中
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if dist.is_initialized():
            # 已初始化，直接返回信息
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = dist.get_world_size()
            is_distributed = True
            logging.info(f"✅ [Distributed] 已初始化！rank={dist.get_rank()}, world_size={world_size}, local_rank={local_rank}")
        else:
            try:
                # 初始化进程组
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                init_process_group(backend=backend, init_method="env://",timeout=datetime.timedelta(hours=2))

                local_rank = int(os.environ["LOCAL_RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                is_distributed = True

                # 设置当前 GPU 设备
                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)

                # 同步所有进程
                barrier()

                logging.info(
                    f"✅ [Distributed] 初始化成功！"
                    f" rank={dist.get_rank()}, world_size={world_size}, local_rank={local_rank}"
                )
            except Exception as e:
                logging.error(f"❌ [Distributed] 初始化失败: {e}")
                raise
    else:
        # 单卡模式
        local_rank = 0
        world_size = 1
        is_distributed = False
        logging.info("✅ [Distributed] 未检测到分布式环境变量，使用单卡模式。")

    return local_rank, world_size, is_distributed



def setup_sync_batchnorm(model):
    """
    根据分布式状态，设置 SyncBatchNorm：
      - 单卡/未初始化：跳过
      - 多卡：构建 per-node group 并 convert
    """
    if not dist.is_initialized():
        dist_print("⚠️ SyncBatchNorm: 非分布式环境，跳过转换")
        return model

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    if world_size % gpus_per_node != 0:
        dist_print(f"⚠️ SyncBatchNorm: world_size={world_size} 不能被 {gpus_per_node} 整除，退化为 world group")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        dist_print(f"🔗 SyncBN 使用 world group (ranks=0..{world_size-1})")
        return model

    num_nodes = world_size // gpus_per_node
    bn_group = None
    bn_group_ranks = None

    # 构建每节点的进程组：[0..7], [8..15], ...
    for node_idx in range(num_nodes):
        ranks = list(range(node_idx * gpus_per_node, (node_idx + 1) * gpus_per_node))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            bn_group = group
            bn_group_ranks = ranks

    dist_print(f"🔗 SyncBatchNorm 使用 per-node group，当前 rank={rank} 所在组 ranks={bn_group_ranks}")

    # ⚡ 关键：convert + 指定 process_group
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=bn_group)
    return model



