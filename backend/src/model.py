# 标准库（内置模块）
from typing import Optional, Union, Dict, List
import math

# 第三方库（pip 安装的包）
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from src.utils.dist import dist_print
from src.losses import *
from src.utils.utils import _as_track_means_view


def targets_scaling_torch(
    targets: torch.Tensor,
    track_means: Union[float, torch.Tensor],
    apply_squashing: Union[bool, list, torch.Tensor] = True
) -> torch.Tensor:
    """
    Strict targets scaling:
      - targets: [B,L] / [B,L,C] / [B,C,L]
      - track_means: scalar / [C] / [B,C]
    NO silent fallback: any mismatch -> raise.
    """
    # normalize targets layout to [B, L, C]
    transposed = False
    squeezed_single_channel = False

    t = targets
    if t.ndim == 2:
        t = t.unsqueeze(-1)
        squeezed_single_channel = True

    # heuristic: [B,C,L] -> [B,L,C]
    if t.ndim == 3 and t.shape[1] <= 16 and t.shape[2] > 1:
        # only transpose if it looks like channels-first
        # (keep your original heuristic but remove dependence on tm.numel)
        t = t.transpose(1, 2)
        transposed = True

    if t.ndim != 3:
        raise RuntimeError(f"[targets_scaling_torch] targets must become 3D, got {tuple(t.shape)}")

    B, L, C = t.shape

    # build strict tm_view via helper (no tm.mean fallback)
    tm_view = _as_track_means_view(track_means, (B, L, C), name="targets_scaling_torch")
    tm_view = tm_view.to(device=t.device, dtype=t.dtype)

    if torch.any(tm_view == 0):
        raise RuntimeError("[targets_scaling_torch] track_means contains 0; cannot divide")

    scaled = t / tm_view  # broadcasting must succeed, otherwise error will raise

    def _squash(x):
        x_pow = x.pow(0.75)
        return torch.where(x_pow > 10.0, 2 * torch.sqrt(x_pow * 10.0) - 10.0, x_pow)

    if isinstance(apply_squashing, bool):
        if apply_squashing:
            scaled = _squash(scaled)
    else:
        if isinstance(apply_squashing, torch.Tensor):
            mask_list = [bool(x) for x in apply_squashing.to('cpu').tolist()]
        elif isinstance(apply_squashing, (list, tuple)):
            mask_list = [bool(x) for x in apply_squashing]
        else:
            mask_list = [True] * C
        if len(mask_list) != C:
            raise RuntimeError(f"[targets_scaling_torch] apply_squashing length {len(mask_list)} != C={C}")
        mask = torch.tensor(mask_list, dtype=torch.bool, device=scaled.device)
        mask_view = mask.view(1, 1, C)
        transformed = _squash(scaled)
        scaled = torch.where(mask_view, transformed, scaled)

    out = scaled
    if transposed:
        out = out.transpose(1, 2)
    if squeezed_single_channel:
        out = out.squeeze(-1)
    return out



def predictions_scaling_torch(
    predictions: torch.Tensor,
    track_means: Union[float, torch.Tensor],
    apply_squashing: Union[bool, list, torch.Tensor] = True
) -> torch.Tensor:
    """
    Strict inverse-scaling for predictions (no silent fallback).
    Inverse order: inv_piecewise -> inv_pow -> multiply track_means.
    """
    preds = predictions.clone()

    single_channel = (preds.ndim == 2)
    if single_channel:
        preds = preds.unsqueeze(-1)  # [B,L,1]

    if preds.ndim != 3:
        raise RuntimeError(f"[predictions_scaling_torch] predictions must be (B,L,C), got {tuple(preds.shape)}")

    B, L, C = preds.shape

    # strict tm_view
    tm_view = _as_track_means_view(track_means, (B, L, C), name="predictions_scaling_torch")
    tm_view = tm_view.to(device=preds.device, dtype=preds.dtype)

    # build per-channel squashing mask
    if isinstance(apply_squashing, bool):
        squashing_mask = [apply_squashing] * C
    elif isinstance(apply_squashing, (list, tuple)):
        squashing_mask = list(apply_squashing)
    elif isinstance(apply_squashing, torch.Tensor):
        squashing_mask = [bool(x) for x in apply_squashing.to('cpu').tolist()]
    else:
        squashing_mask = [True] * C

    if len(squashing_mask) != C:
        raise RuntimeError(f"[predictions_scaling_torch] apply_squashing length {len(squashing_mask)} != C={C}")

    mask = torch.tensor(squashing_mask, dtype=torch.bool, device=preds.device).view(1, 1, C)

    # inverse squashing (strict inverse of your forward squash):
    # forward: x_pow = x^0.75; if x_pow>10: y = 2*sqrt(x_pow*10)-10 else y=x_pow
    # inverse: if y>10: x_pow=((y+10)^2)/40 else x_pow=y; x = x_pow^(1/0.75)
    def _inv_piecewise(y):
        return torch.where(y > 10.0, ((y + 10.0) ** 2) / 40.0, y)

    inv_pow = 1.0 / 0.75
    x_pow = _inv_piecewise(preds)
    x_pow = torch.clamp(x_pow, min=0.0)
    x = x_pow.pow(inv_pow)

    preds = torch.where(mask, x, preds)

    # multiply by track means at the end
    preds = preds * tm_view  # must broadcast; otherwise raise by PyTorch

    if single_channel:
        preds = preds.squeeze(-1)

    preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    return preds


# ==========================================
# 1. 纯净的 Transformer 层 (单层逻辑)
# ==========================================
class PureTransformerLayer(nn.Module):
    """
    标准的 Pre-Norm Transformer 层。
    输入/输出形状保持为: [B, L, C] (Batch First)
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [B, L, C]
        
        # Self-Attention (Pre-Norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        
        # FFN (Pre-Norm)
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================
# 2. 封装的 Bottleneck 模块 (包含 PE 和 堆叠)
# ==========================================
class GenomicTransformerBottleneck(nn.Module):
    """
    自适应长度的 Transformer Bottleneck。
    不再需要手动指定 max_len，能够处理任意长度的输入。
    """
    def __init__(self, dim: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        
        if dim % num_heads != 0:
            raise ValueError(f"bottleneck_dim ({dim}) must be divisible by num_heads ({num_heads})")

        # 堆叠 Transformer Layers
        self.layers = nn.ModuleList([
            PureTransformerLayer(dim, num_heads, mlp_ratio=4.0, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.1)
        
        # 注册一个空的 buffer 用于缓存 PE，标记为 persistent=False (不保存到 state_dict 中，因为它是算出来的)
        self.register_buffer("pe_cache", torch.zeros(1, 0, dim), persistent=False)

    def _get_pos_encoding(self, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        动态获取位置编码。如果缓存够大直接切片，不够大则重新生成。
        """
        # 如果缓存为空 或 缓存长度不够，则重新生成
        if self.pe_cache.shape[1] < length:
            # 为了防止每次长度增加一点点就重新计算，可以适当多生成一点 (例如取 length 的 1.5 倍或向上取整)
            # 这里简单起见，直接生成当前需要的长度，或者你可以设置一个最小步长
            calc_len = max(length, 4096) # 至少生成 4096 长度，避免短序列频繁计算
            
            pe = torch.zeros(calc_len, self.dim, device=device, dtype=dtype)
            position = torch.arange(0, calc_len, dtype=dtype, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=dtype, device=device) * (-math.log(10000.0) / self.dim))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            if self.dim % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            
            # 更新缓存 [1, L_new, C]
            self.pe_cache = pe.unsqueeze(0)
            
        # 返回切片 [1, L, C]
        return self.pe_cache[:, :length, :]

    def forward(self, x: Tensor) -> Tensor:
        # Input x: [B, C, L]
        B, C, L = x.shape
        
        # 1. Transpose to [B, L, C]
        x = x.transpose(1, 2)
        
        # 2. Add Positional Encoding (动态获取)
        # 注意：这里我们传入 x 的 device 和 dtype 确保匹配
        pe = self._get_pos_encoding(L, x.device, x.dtype)
        x = x + pe
        
        x = self.dropout(x)
        
        # 3. Pass through stacked layers
        for layer in self.layers:
            x = layer(x)
            
        # 4. Transpose back to [B, C, L]
        x = x.transpose(1, 2)
        
        return x

class Conv1DBlock(nn.Module):
    """
    Enhanced 1D convolutional block with support for different downsampling methods.
    Use strided conv, max pooling or average pooling when downsample > 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        padding: Optional[int] = None,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        downsample: int = 1,
        downsample_method: str = 'conv',  # 默认：'conv', 可选：'maxpool', 'avgpool'
        upsample: int = 1      # Still uses interpolate for safety
    ):
        super().__init__()

        if downsample < 1 or upsample < 1:
            raise ValueError("downsample and upsample must be >= 1")
        if downsample > 1 and upsample > 1:
            raise ValueError("Cannot apply both downsampling and upsampling in the same block.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to allow symmetric padding.")
        if downsample_method not in ['conv', 'maxpool', 'avgpool']:
            raise ValueError("downsample_method must be 'conv', 'maxpool', or 'avgpool'")

        self.downsample_factor = downsample
        self.downsample_method = downsample_method
        self.upsample_factor = upsample

        # Calculate padding to preserve length after convolution
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2
        self.padding = padding

        # Build main conv layer (only for 'conv' method or when no downsampling)
        if downsample_method == 'conv' or downsample == 1:
            conv_layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=downsample,  # ✅ Learnable downsampling via stride
                padding=self.padding,
                dilation=dilation
            )
            layers = [conv_layer]
        else:
            # For pooling methods, use stride=1 in conv and separate pooling layer
            conv_layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=self.padding,
                dilation=dilation
            )
            layers = [conv_layer]
            
            # Add pooling layer
            if downsample_method == 'maxpool':
                self.downsample_pool = nn.MaxPool1d(kernel_size=downsample, stride=downsample)
            elif downsample_method == 'avgpool':
                self.downsample_pool = nn.AvgPool1d(kernel_size=downsample, stride=downsample)
            else:
                self.downsample_pool = None

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))

        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

        # Upsample: handled in forward via interpolate (safe)
        self.upsample_scale = upsample if upsample > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)

        # Apply additional downsampling if needed
        if hasattr(self, 'downsample_pool') and self.downsample_pool is not None:
            out = self.downsample_pool(out)

        if self.upsample_scale is not None:
            out = F.interpolate(out, scale_factor=self.upsample_scale, mode='nearest')

        return out
    
# ==========================================
# 3. 修改后的 UNet
# ==========================================

class func_genome_UNet(nn.Module):
    """
    功能性基因组信号的 U-Net 模型，用于特征提取。
    包含动态构建的编码器（Encoder）、瓶颈层（Bottleneck）和解码器（Decoder）。
    """

    def __init__(self, proj_dim, num_downsamples, bottleneck_dim, use_transformer_bottleneck=True):
        """
        初始化 U-Net 模型。

        参数:
            proj_dim (int): 输入特征的维度。
            num_downsamples (int): 下采样次数，建议 1 到 6 次，比如 2 或 4。
            bottleneck_dim (int): 瓶颈层的维度。
        """
        super(func_genome_UNet, self).__init__()
        assert 1 <= num_downsamples <= 6, "num_downsamples 必须在 1 到 6 之间"
        assert bottleneck_dim > proj_dim, "bottleneck_dim 必须大于 proj_dim"
        self.proj_dim = proj_dim
        self.num_downsamples = num_downsamples
        self.bottleneck_dim = bottleneck_dim

        # 自动计算每次下采样需要增加的维度
        self.dim_step = (bottleneck_dim - proj_dim) // num_downsamples

        # 动态构建编码器（Encoder）
        self.encoders = nn.ModuleList()
        in_channels = proj_dim
        for i in range(num_downsamples):
            out_channels = proj_dim + self.dim_step * (i + 1)
            self.encoders.append(Conv1DBlock(in_channels, out_channels, kernel_size=5, downsample=2))
            in_channels = out_channels

        # 瓶颈层（Bottleneck）
        self.bottleneck = nn.Sequential(
            Conv1DBlock(in_channels, bottleneck_dim, kernel_size=5, dilation=2),
            Conv1DBlock(bottleneck_dim, bottleneck_dim, kernel_size=5, dilation=4)
        )

        # ================= 改动部分 =================
        # 瓶颈层（Bottleneck）
        # 这里我们结合了 CNN 投影（调整维度）和 Transformer（提取长程依赖）
        
        if use_transformer_bottleneck:
            dist_print(f"✅ [Init] func_genome_UNet: Building TRANSFORMER Bottleneck with dim={bottleneck_dim}")
            self.bottleneck = nn.Sequential(
                # 1. 先用 CNN 将特征维度映射到 bottleneck_dim
                Conv1DBlock(in_channels, bottleneck_dim, kernel_size=3, padding=1),
                
                # 2. 插入 4 层 Transformer 堆叠
                GenomicTransformerBottleneck(
                    dim=bottleneck_dim, 
                    num_layers=4,    # ✅ 这里加了 4 层
                    num_heads=8      # 确保 bottleneck_dim 能被 8 整除
                ),
                
                # 3. 再接一个 CNN 整理特征 (可选，保持原有结构风格)
                Conv1DBlock(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1)
            )
        else:
            dist_print(f"⚠️ [Init] func_genome_UNet: Building CNN Bottleneck") # 确认没跑偏
            # 原有的纯 CNN 实现
            self.bottleneck = nn.Sequential(
                Conv1DBlock(in_channels, bottleneck_dim, kernel_size=5, dilation=2),
                Conv1DBlock(bottleneck_dim, bottleneck_dim, kernel_size=5, dilation=4)
            )
        # ===========================================

        # 动态构建解码器（Decoder）
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_downsamples):
            out_channels = proj_dim + self.dim_step * (num_downsamples - i - 1)
            self.upsamplers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            self.decoders.append(Conv1DBlock(out_channels * 2, out_channels, kernel_size=5))
            in_channels = out_channels

    def forward(self, x):
        """
        前向传播。

        参数:
            x (Tensor): 输入张量，形状为 [batch_size, proj_dim, sequence_length]。

        返回:
            Tensor: 输出张量，形状为 [batch_size, proj_dim, sequence_length]。
        """
        # 编码器（Encoder）
        skip_connections = []
        for encoder in self.encoders:
            skip_connections.append(x)
            x = encoder(x)

        # 瓶颈层（Bottleneck）
        x = self.bottleneck(x)  # 这里现在会经过 Transformer

        # 解码器（Decoder）与跳跃连接（Skip Connections）
        for i in range(self.num_downsamples):
            x = self.upsamplers[i](x)
            skip_connection = skip_connections[-(i + 1)]
            if x.size(-1) != skip_connection.size(-1):
                print(f"Upsampled size: {x.size(-1)}, Skip connection size: {skip_connection.size(-1)}")
                x = F.interpolate(x, size=skip_connection.size(-1), mode='nearest')
            x = self.decoders[i](torch.cat([x, skip_connection], dim=1))

        return x
    


class GenOmics(nn.Module):
    """
    GenoOmics: 基于 Genos 基因组大模型的多组学信号预测框架。
    核心功能:
        输入 DNA 序列，通过 Genos 基因组大模型提取深层特征，并结合 U-Net 网络捕获功能性基因组学信号。
        实现单碱基分辨率的转录组学（RNA-seq）和表观基因组学（ATAC-seq）信号轨迹的联合预测。
        用于解析基因调控机制，助力多组学数据的功能注释与机制研究。
    """
    
    # 新增，通道一致性检查（避免来自于label_meta_df和index_stat.json的通道来源不一致）
    def _sanity_check_channel_ordering(self):
        """
        Sanity assert: channel/head ordering consistency

        目的：防止 labels_meta_df 的通道顺序 与 index_stat['counts'] 的 head 分组/顺序不一致，
        导致 silent bug（比如 plus/minus 对调、head 切片错位，训练还能跑但学错任务）。
        """
        if "task_head" not in self.labels_meta_df.columns:
            raise RuntimeError(
                "[Sanity] labels_meta_df missing column 'task_head'. "
                "Expected bigWig_labels_meta.csv to contain task_head."
            )

        # labels_meta_df 的通道顺序（row order matters）
        labels_task_heads = self.labels_meta_df["task_head"].tolist()

        # index_stat 的 head 顺序 & 每个 head 的通道数
        counts = self.index_stat.get("counts", {})
        index_heads = list(counts.get("task_head", []))
        index_num_task_head = counts.get("num_task_head", None)  # dict: head_name -> num_tracks

        if not index_heads:
            raise RuntimeError(f"[Sanity] index_stat['counts']['task_head'] is empty: {index_heads}")

        if not isinstance(index_num_task_head, dict):
            raise RuntimeError(
                "[Sanity] index_stat['counts']['num_task_head'] must be a dict {head_name: num_tracks}. "
                f"Got: {type(index_num_task_head)} {index_num_task_head}"
            )

        # 1) 总通道数必须一致
        expected_total_tracks = 0
        for h in index_heads:
            if h not in index_num_task_head:
                raise RuntimeError(
                    "[Sanity] Head appears in counts.task_head but missing in counts.num_task_head mapping.\n"
                    f"  missing head: {h}\n"
                    f"  index_heads: {index_heads}\n"
                    f"  num_task_head keys: {list(index_num_task_head.keys())}"
                )
            n = int(index_num_task_head[h])
            if n <= 0:
                raise RuntimeError(f"[Sanity] num_task_head[{h}] must be >0, got {n}")
            expected_total_tracks += n

        if expected_total_tracks != len(labels_task_heads):
            raise RuntimeError(
                "[Sanity] Total number of tracks mismatch between index_stat and labels_meta_df.\n"
                f"  expected_total_tracks(sum num_task_head) = {expected_total_tracks}\n"
                f"  len(labels_meta_df rows)              = {len(labels_task_heads)}\n"
                f"  index_heads                           = {index_heads}\n"
                f"  index_num_task_head                   = {index_num_task_head}\n"
                f"  labels_meta task_head rows            = {labels_task_heads}"
            )

        # 2) labels_meta_df 必须按 index_heads 的顺序分组，且每组连续
        ptr = 0
        for h in index_heads:
            n = int(index_num_task_head[h])
            block = labels_task_heads[ptr: ptr + n]
            if len(block) != n:
                raise RuntimeError(
                    f"[Sanity] Internal slicing error: head={h}, expected n={n}, got block_len={len(block)}"
                )
            if any(x != h for x in block):
                raise RuntimeError(
                    "[Sanity] Track head ordering mismatch.\n"
                    f"  At head='{h}', expected next {n} rows in labels_meta_df['task_head'] to all be '{h}'.\n"
                    f"  But got block: {block}\n"
                    f"  index_heads: {index_heads}\n"
                    f"  index_num_task_head: {index_num_task_head}\n"
                    f"  full labels_meta task_head rows: {labels_task_heads}\n"
                    "Fix: ensure bigWig_labels_meta.csv row order matches index_stat['counts'] "
                    "or regenerate index_stat using the same labels_meta ordering."
                )
            ptr += n

        dist_print("[Sanity] Channel/head ordering check passed.")


    def __init__(self, base_model, 
                 index_stat,
                 labels_meta_df, 
                 loss_func: str = 'mse', 
                 proj_dim: int = 512, 
                 num_downsamples: int = 2, 
                 bottleneck_dim: int = 1024,
                 **kwargs):
        """
        初始化模型。

        参数:
            base_model: 预训练的 DNA 模型。
            loss_func (str): 训练时使用的损失函数。支持 'mse'、'poisson'、'tweedie'、'poisson-multinomial'。
            proj_dim (int): 投影层的维度。
            num_downsamples (int): U-Net 编码器中的下采样层数。
        """
        super().__init__()
        self.loss_func = loss_func
        self.index_stat = index_stat
        self.labels_meta_df = labels_meta_df
        self.task_head = list(self.index_stat['counts']['task_head'])
        self.num_task_head = self.index_stat['counts']['num_task_head']
        self.biosample_names = self.index_stat['counts']['biosample_names']

        # 检查通道定义一致性
        self._sanity_check_channel_ordering()

        
        self.apply_squashing = [
            not (name.startswith("ATAC") or name.startswith("atac")) 
            for name in self.labels_meta_df['target_file_name']
            ]

        # 数据缩放
        self.num_tracks = len(self.labels_meta_df['target_file_name'].to_list())
        self.track_means = torch.tensor(self.labels_meta_df['nonzero_mean'].to_list(), dtype=torch.float32)  # 每个轨迹的均值
        
        # 获取基础模型的隐藏层大小
        base_model_hidden_size = getattr(base_model.config, "hidden_size", None)
        if base_model_hidden_size is None:
            raise ValueError("无法从 `base_model` 中获取 `hidden_size`")
        
        # 特征提取：使用预训练的 DNA 模型作为嵌入器
        self.base = base_model

        # 嵌入投影层
        self.embedd_proj = Conv1DBlock(base_model_hidden_size, proj_dim, kernel_size=1)
        
        # 使用 genome_signal_UNet 作为编码器-解码器
        self.unet = func_genome_UNet(proj_dim=proj_dim, num_downsamples=num_downsamples, bottleneck_dim=bottleneck_dim)
        
        # 任务特定的输出头
        self.output_heads = nn.ModuleDict({
            head_name: nn.Conv1d(proj_dim, num_tracks, kernel_size=1) # 相当于线性层
            for head_name, num_tracks in self.num_task_head.items()
        })

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.zeros(self.num_tracks))


        # multinomial loss tuning parameters — only relevant when using poisson-multinomial
        if self.loss_func == 'poisson-multinomial':
            # defaults compatible with CLI defaults; allow kwargs to override
            mw = kwargs.pop('multinomial_weight', 5.0)
            mts = kwargs.pop('multinomial_term_scale', 0.1)
            try:
                self.multinomial_weight = float(mw)
            except Exception:
                self.multinomial_weight = 5.0
            try:
                self.multinomial_term_scale = float(mts)
            except Exception:
                self.multinomial_term_scale = 0.1
        else:
            # not used for other loss types
            self.multinomial_weight = None
            self.multinomial_term_scale = None

    def _compute_loss(self, logits, scaled_labels, pos_strand_mask=None, neg_strand_mask=None):
        """
        计算每个轨迹的损失并返回总 loss 以及按轨道的 loss dict。
        简洁实现：按 assay 分片计算并汇总。
        """
        # Build per-head offsets from `self.num_task_head` which maps head_name -> num_channels
        losses_by_head = {}
        head_offsets = {}
        cur = 0
        for head in self.task_head:
            n_ch = int(self.num_task_head.get(head, getattr(self, 'num_biosamples', 1)))
            head_offsets[head] = (cur, cur + n_ch)
            cur += n_ch

        for i, name in enumerate(self.task_head):
            if name not in head_offsets:
                raise KeyError(f"Head '{name}' not found in num_task_head mapping")
            s, e = head_offsets[name]
            pred = logits[..., s:e]
            targ = scaled_labels[..., s:e]

            if self.loss_func == 'mse':
                l = F.mse_loss(pred, targ)
            elif self.loss_func == 'poisson':
                l = poisson_loss(pred, targ)
            elif self.loss_func == 'tweedie':
                l = tweedie_loss(pred, targ, p=torch.tensor(1.2, device=pred.device, dtype=pred.dtype))
            elif self.loss_func == 'poisson-multinomial':
                # if name == "total_RNA-seq_+":
                #     add_gene_level_loss = True
                #     gene_mask = pos_strand_mask
                # if name == "total_RNA-seq_-":
                #     add_gene_level_loss = True
                #     gene_mask = neg_strand_mask
                # if name == "ATAC-seq_":
                #     add_gene_level_loss = False
                #     gene_mask = None
                # if name == "polyA_plus_RNA-seq_+":
                #     add_gene_level_loss = True
                #     gene_mask = pos_strand_mask
                # if name == "polyA_plus_RNA-seq_-":
                #     add_gene_level_loss = True
                #     gene_mask = neg_strand_mask
                
                # qi:仅修改为只支持RNASeq
                add_gene_level_loss = False
                gene_mask = None
                if name.endswith("_+"):
                    add_gene_level_loss = True
                    gene_mask = pos_strand_mask
                elif name.endswith("_-"):
                    add_gene_level_loss = True
                    gene_mask = neg_strand_mask
 
                total_l, track_level_l, track_level_poisson_term_l, track_level_multinomial_term_l, gene_level_l, gene_level_poisson_term_l, gene_level_multinomial_term_l  = poisson_multinomial_loss(
                    pred,
                    targ,
                    add_gene_level_loss=add_gene_level_loss,
                    gene_mask=gene_mask,
                    multinomial_term_scale=self.multinomial_term_scale,
                    multinomial_weight=self.multinomial_weight,
                )
                l = total_l
            else:
                raise ValueError(f"不支持的损失函数: {self.loss_func}")

            # weight per-head (can override for specific heads if desired)
            weight_h = 1/len(self.task_head) # uniform weighting by default
            # store breakdowns for poisson-multinomial (or scalar loss)
            if self.loss_func == 'poisson-multinomial':
                losses_by_head[name] = {
                    "total": weight_h * l,
                    "total_track": weight_h * track_level_l,
                    "total_track_poisson": weight_h * track_level_poisson_term_l,
                    "total_track_multinomial": weight_h * track_level_multinomial_term_l,
                    "total_gene": weight_h * gene_level_l,
                    "total_gene_poisson": weight_h * gene_level_poisson_term_l,
                    "total_gene_multinomial": weight_h * gene_level_multinomial_term_l,
                }
            else:
                losses_by_head[name] = {"total": weight_h * l}

            # metrics_by_head will be produced by _compute_metrics()

        # accumulate total loss (summing 'total' when breakdown dicts present)
        total_loss = None
        for v in losses_by_head.values():
            if isinstance(v, dict):
                t = v["total"]
            else:
                t = v
            if total_loss is None:
                total_loss = t
            else:
                total_loss = total_loss + t
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=logits.device)
        # metrics are computed separately after inverse-scaling
        return total_loss, losses_by_head

    def _compute_metrics(
        self,
        logits_orig: torch.Tensor,
        labels: torch.Tensor,
        pos_strand_mask: Optional[torch.Tensor] = None,
        neg_strand_mask: Optional[torch.Tensor] = None,
        compute_on_nonzero: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute per-head metrics on original-scale predictions and labels.

        Args:
            logits_orig: [B, L, num_tracks] predictions already inverse-scaled.
            labels: [B, L, num_tracks] original-scale labels.
            pos_strand_mask / neg_strand_mask: gene masks for LFC correlation.
            compute_on_nonzero: if True, compute KL/cosine only on positions where label>0

        Returns:
            metrics_by_head: { head_name: {metric_name: tensor} }
        """
        metrics_by_head: Dict[str, Dict[str, torch.Tensor]] = {}
        eps = 1e-12
        clamp_eps = 1e-12

        with torch.no_grad():
            # build head offsets (same logic as in _compute_loss)
            head_offsets = {}
            cur = 0
            for head in self.task_head:
                n = int(self.num_task_head.get(head, getattr(self, 'num_biosamples', 1)))
                head_offsets[head] = (cur, cur + n)
                cur += n

            for i, name in enumerate(self.task_head):
                if name not in head_offsets:
                    # skip unknown head for safety
                    continue
                s, e = head_offsets[name]
                pred = logits_orig[..., s:e]  # [B, L, tracks]
                targ = labels[..., s:e]       # [B, L, tracks]

                # ensure float tensors on same device
                pred = pred.to(dtype=targ.dtype, device=targ.device)

                # total_signal_MAE: mean abs error over sample & track totals
                true_totals = targ.sum(dim=1)    # [B, tracks]
                pred_totals = pred.sum(dim=1)    # [B, tracks]
                total_signal_MAE = torch.abs(pred_totals - true_totals).mean()

                # background region MAE: for each sample compute MAE over all
                # positions/tracks where true==0, then average that per-sample MAE across the batch.
                zero_mask = (targ == 0.0)  # [B, L, tracks]
                abs_err_pos = torch.abs(pred - targ) * zero_mask.to(pred.dtype)  # [B, L, tracks]
                # sum over positions and tracks to get per-sample total absolute error
                sum_abs_err_bg_per_sample = abs_err_pos.sum(dim=(1, 2))  # [B]
                counts_zero_per_sample = zero_mask.sum(dim=(1, 2)).to(pred.dtype)  # [B]
                per_sample_bg_mae = torch.where(
                    counts_zero_per_sample > 0,
                    sum_abs_err_bg_per_sample / (counts_zero_per_sample + eps),
                    torch.zeros_like(counts_zero_per_sample)
                )  # [B]
                # finally average across samples in the batch
                background_signal_MAE = per_sample_bg_mae.mean()

                # Distribution metrics (KL and cosine) computed on original-scale
                pred_c = torch.clamp(pred, min=clamp_eps)
                targ_c = torch.clamp(targ, min=clamp_eps)

                if compute_on_nonzero:
                    mask = (targ > 0.0)
                    counts = mask.sum(dim=1)  # [B, tracks]

                    # masked sums; where no nonzero positions exist, fall back to full vector
                    targ_masked = targ_c * mask.to(pred_c.dtype)
                    pred_masked = pred_c * mask.to(pred_c.dtype)

                    sum_t = targ_masked.sum(dim=1, keepdim=True)
                    sum_p = pred_masked.sum(dim=1, keepdim=True)

                    # fallback sums for zero-count cases
                    full_sum_t = targ_c.sum(dim=1, keepdim=True)
                    full_sum_p = pred_c.sum(dim=1, keepdim=True)

                    use_mask = (counts > 0).unsqueeze(1)  # [B,1,tracks]
                    sum_t = torch.where(use_mask, sum_t, full_sum_t)
                    sum_p = torch.where(use_mask, sum_p, full_sum_p)

                    p_true = targ_masked / (sum_t + eps)
                    p_pred = pred_masked / (sum_p + eps)
                else:
                    p_true = targ_c / (targ_c.sum(dim=1, keepdim=True) + eps)
                    p_pred = pred_c / (pred_c.sum(dim=1, keepdim=True) + eps)

                # KL and cosine over token dim -> mean across samples and tracks
                kl_per_sample_track = (p_true * (torch.log(p_true + eps) - torch.log(p_pred + eps))).sum(dim=1)
                distribution_KL = kl_per_sample_track.mean()

                cos_sim_per_sample_track = F.cosine_similarity(p_true, p_pred, dim=1, eps=1e-8)
                distribution_cosine_distance = (1.0 - cos_sim_per_sample_track).mean()

                # Non-zero region MAE and MSE: per-sample then averaged across batch
                nonzero_mask = (targ > 0.0)  # [B, L, tracks]
                abs_err_nonzero = torch.abs(pred - targ) * nonzero_mask.to(pred.dtype)  # [B, L, tracks]
                sum_abs_err_nonzero_per_sample = abs_err_nonzero.sum(dim=(1, 2))  # [B]
                counts_nonzero_per_sample = nonzero_mask.sum(dim=(1, 2)).to(pred.dtype)  # [B]
                per_sample_nonzero_mae = torch.where(
                    counts_nonzero_per_sample > 0,
                    sum_abs_err_nonzero_per_sample / (counts_nonzero_per_sample + eps),
                    torch.zeros_like(counts_nonzero_per_sample)
                )  # [B]
                mean_nonzero_MAE = per_sample_nonzero_mae.mean()

                sq_err_nonzero = ((pred - targ) ** 2) * nonzero_mask.to(pred.dtype)
                sum_sq_err_nonzero_per_sample = sq_err_nonzero.sum(dim=(1, 2))  # [B]
                per_sample_nonzero_mse = torch.where(
                    counts_nonzero_per_sample > 0,
                    sum_sq_err_nonzero_per_sample / (counts_nonzero_per_sample + eps),
                    torch.zeros_like(counts_nonzero_per_sample)
                )
                mean_nonzero_mse = per_sample_nonzero_mse.mean()

                # Track-level correlations on non-zero positions: per-sample per-track correlations
                track_pearson_vals = []
                track_spearman_vals = []
                track_log1p_pearson_vals = []
                Bbatch = pred.shape[0]
                Ctracks = pred.shape[2]
                for b in range(Bbatch):
                    for c in range(Ctracks):
                        mask_nc = (targ[b, :, c] > 0.0)
                        if mask_nc.sum() < 2:
                            continue
                        try:
                            x = pred[b, mask_nc, c].detach().cpu().numpy()
                            y = targ[b, mask_nc, c].detach().cpu().numpy()
                            # skip if x or y are constant (all elements equal) or contain NaN/inf

                            if x.size < 2 or y.size < 2:
                                continue
                            # skip if constant (all elements equal)
                            if np.allclose(x, x.flat[0]) or np.allclose(y, y.flat[0]):
                                continue
                            # skip if non-finite
                            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                                continue
                            # Pearson
                            r, _ = pearsonr(x, y)
                            if np.isfinite(r):
                                track_pearson_vals.append(float(max(min(r, 1.0), -1.0)))
                            # Spearman
                            sr, _ = spearmanr(x, y)
                            if np.isfinite(sr):
                                track_spearman_vals.append(float(max(min(sr, 1.0), -1.0)))
                            # log1p Pearson
                            lx = np.log1p(x)
                            ly = np.log1p(y)
                            if not (np.allclose(lx, lx.flat[0]) or np.allclose(ly, ly.flat[0])) and np.isfinite(lx).all() and np.isfinite(ly).all():
                                lr, _ = pearsonr(lx, ly)
                                if np.isfinite(lr):
                                    track_log1p_pearson_vals.append(float(max(min(lr, 1.0), -1.0)))
                        except Exception:
                             continue

                if len(track_pearson_vals) == 0:
                    mean_track_nonzero_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                else:
                    mean_track_nonzero_pearson = torch.tensor(np.mean(track_pearson_vals), device=pred.device, dtype=pred.dtype)

                if len(track_spearman_vals) == 0:
                    mean_track_nonzero_spearman = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                else:
                    mean_track_nonzero_spearman = torch.tensor(np.mean(track_spearman_vals), device=pred.device, dtype=pred.dtype)

                if len(track_log1p_pearson_vals) == 0:
                    mean_track_nonzero_log1p_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                else:
                    mean_track_nonzero_log1p_pearson = torch.tensor(np.mean(track_log1p_pearson_vals), device=pred.device, dtype=pred.dtype)

                # gene-level LFC Pearson correlation (same logic as before, on original-scale)
                # print("name: ")
                # print(name)
                # print("pos_strand_mask: ")
                # print(pos_strand_mask)

                if name.endswith("_+"):
                    gm = pos_strand_mask
                elif name.endswith("_-"):
                    gm = neg_strand_mask
                else:
                    gm = None

                # print("gm: ")
                # print(gm)
                if gm is None:
                    mean_gene_LFC_corr = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    # also set raw gene-level correlations to 0 when no genes
                    mean_gene_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    mean_gene_spearman = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    mean_gene_log1p_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                else:
                    # 【修改点 1】: 先给这三个变量赋默认值，防止后面因为数据为空跳过赋值逻辑
                    mean_gene_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    mean_gene_spearman = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    mean_gene_log1p_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    
                    mask = gm.to(device=pred.device)
                    if mask.dtype != torch.bool:
                        mask = mask.bool()

                    gene_pred_means = []
                    gene_true_means = []

                    for b in range(pred.shape[0]):
                        mask_b = mask[b]
                        if not mask_b.any():
                            continue
                        idxs = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)
                        if idxs.numel() == 0:
                            continue
                        idxs_cpu = idxs.to('cpu').numpy()
                        runs = []
                        start = idxs_cpu[0]
                        prev = start
                        for ii in idxs_cpu[1:]:
                            if ii == prev + 1:
                                prev = ii
                                continue
                            else:
                                runs.append((start, prev + 1))
                                start = ii
                                prev = ii
                        runs.append((start, prev + 1))

                        for (s_idx, e_idx) in runs:
                            seg_pred = pred[b, s_idx:e_idx, :]
                            seg_true = targ[b, s_idx:e_idx, :]
                            if seg_pred.numel() == 0 or seg_true.numel() == 0:
                                continue
                            mean_pred = seg_pred.mean(dim=0)
                            mean_true = seg_true.mean(dim=0)
                            gene_pred_means.append(mean_pred)
                            gene_true_means.append(mean_true)

                    if len(gene_pred_means) == 0:
                        mean_gene_LFC_corr = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    else:
                        gp = torch.stack(gene_pred_means, dim=0)
                        gt = torch.stack(gene_true_means, dim=0)
                        ref_gp = gp[:, 0].unsqueeze(1)
                        ref_gt = gt[:, 0].unsqueeze(1)
                        lfc_pred = torch.log2((gp + eps) / (ref_gp + eps))
                        lfc_true = torch.log2((gt + eps) / (ref_gt + eps))

                        G = lfc_pred.shape[0]
                        Cg = lfc_pred.shape[1]
                        if G < 1 or Cg <= 1:
                            mean_gene_LFC_corr = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                        else:
                            gene_rs = []
                            for g in range(G):
                                x = lfc_pred[g, :]
                                y = lfc_true[g, :]
                                if x.numel() == 0 or y.numel() == 0:
                                    continue
                                try:
                                    x_np = x.detach().cpu().numpy()
                                    y_np = y.detach().cpu().numpy()
                                    if (
                                        x_np.size < 2
                                        or y_np.size < 2
                                        or not (np.isfinite(x_np).all() and np.isfinite(y_np).all())
                                        or np.allclose(x_np, x_np.flat[0])
                                        or np.allclose(y_np, y_np.flat[0])
                                    ):
                                        continue
                                    r_val, _ = pearsonr(x_np, y_np)
                                    if not np.isfinite(r_val):
                                        continue
                                except Exception:
                                    continue
                                r_val = float(max(min(r_val, 1.0), -1.0))
                                gene_rs.append(torch.tensor(r_val, device=pred.device, dtype=pred.dtype))

                            if len(gene_rs) == 0:
                                mean_gene_LFC_corr = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                            else:
                                mean_gene_LFC_corr = torch.stack(gene_rs).mean()

                            # compute gene-level raw correlations (pearson, spearman, log1p pearson)
                            gene_pearson_vals = []
                            gene_spearman_vals = []
                            gene_log1p_pearson_vals = []
                            for g in range(gp.shape[0]):
                                x = gp[g, :].detach().cpu().numpy()
                                y = gt[g, :].detach().cpu().numpy()
                                try:
                                    if x.size < 2 or y.size < 2:
                                        continue
                                    if not (np.isfinite(x).all() and np.isfinite(y).all()):
                                        continue
                                    if np.allclose(x, x.flat[0]) or np.allclose(y, y.flat[0]):
                                        continue
                                    r, _ = pearsonr(x, y)
                                    if np.isfinite(r):
                                        gene_pearson_vals.append(float(max(min(r, 1.0), -1.0)))
                                    sr, _ = spearmanr(x, y)
                                    if np.isfinite(sr):
                                        gene_spearman_vals.append(float(max(min(sr, 1.0), -1.0)))
                                    lr, _ = pearsonr(np.log1p(x), np.log1p(y))
                                    if np.isfinite(lr):
                                        gene_log1p_pearson_vals.append(float(max(min(lr, 1.0), -1.0)))
                                except Exception:
                                    continue

                            if len(gene_pearson_vals) == 0:
                                mean_gene_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                            else:
                                mean_gene_pearson = torch.tensor(np.mean(gene_pearson_vals), device=pred.device, dtype=pred.dtype)

                            if len(gene_spearman_vals) == 0:
                                mean_gene_spearman = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                            else:
                                mean_gene_spearman = torch.tensor(np.mean(gene_spearman_vals), device=pred.device, dtype=pred.dtype)

                            if len(gene_log1p_pearson_vals) == 0:
                                mean_gene_log1p_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                            else:
                                mean_gene_log1p_pearson = torch.tensor(np.mean(gene_log1p_pearson_vals), device=pred.device, dtype=pred.dtype)

                per_head_metrics = {
                    "mean_total_absolute_error": total_signal_MAE,
                    "mean_background_mae": background_signal_MAE,
                    "mean_nonzero_mae": mean_nonzero_MAE,
                    "mean_nonzero_mse": mean_nonzero_mse,
                    "mean_track_nonzero_pearson": mean_track_nonzero_pearson,
                    "mean_track_nonzero_spearman": mean_track_nonzero_spearman,
                    "mean_track_nonzero_log1p_pearson": mean_track_nonzero_log1p_pearson,
                    "mean_dist_KL": distribution_KL,
                    "mean_dist_cosine_distance": distribution_cosine_distance,
                    "mean_gene_LFC(raw)_corr": mean_gene_LFC_corr,
                    "mean_gene_pearson": mean_gene_pearson,
                    "mean_gene_spearman": mean_gene_spearman,
                    "mean_gene_log1p_pearson": mean_gene_log1p_pearson,
                }

                metrics_by_head[name] = per_head_metrics

        return metrics_by_head


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        将 gradient_checkpointing_enable 调用转发给 base model
        """
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            # 也可以手动设置标志，确保 forward 中的逻辑能读到
            if hasattr(self.base, "config"):
                self.base.config.use_cache = False # 开启 GC 时必须关闭 kv cache

    def gradient_checkpointing_disable(self):
        """
        将 gradient_checkpointing_disable 调用转发给 base model
        """
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()
    
    def _encode_sequence(self, input_ids: Tensor) -> Tensor:
        """
        Encode input token ids into transformer sequence hidden states.

        This wraps embedding lookup, RoPE position embeddings and the
        per-layer traversal (with optional gradient checkpointing and
        frozen-layer no-grad fast path). Returns `sequence_hidden` of shape
        [B, L, H].
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # LoRA/PEFT 包装后，真实 backbone 在 get_base_model() 里；取不到就用当前 self.base
        base = self.base
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        
        # 1. 获取 Embedding（智能判断是否需要梯度）
        embed_layer = base.get_input_embeddings()
        embeddings_require_grad = any(p.requires_grad for p in embed_layer.parameters())

        if embeddings_require_grad:
            hidden_states = embed_layer(input_ids)
        else:
            with torch.no_grad():
                hidden_states = embed_layer(input_ids)

        # 2. 准备位置编码（RoPE）- 对于 Genos/Mixtral 模型必须
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # 获取 RoPE 模块
        rotary_emb = None
        if hasattr(base, 'model') and hasattr(base.model, 'rotary_emb'):
            rotary_emb = base.model.rotary_emb
        elif hasattr(base, 'rotary_emb'):
            rotary_emb = base.rotary_emb
        else:
            raise RuntimeError("⚠️ 无法找到 rotary_emb 模块，请检查模型结构")

        # 计算位置编码（不需要梯度，省显存）
        with torch.no_grad():
            position_embeddings = rotary_emb(hidden_states, position_ids)

        # 3. 手动遍历Transformer层，逐层应用checkpoint
        if hasattr(base, 'layers'):
            layers = base.layers
        elif hasattr(base, 'model') and hasattr(base.model, 'layers'):
            layers = base.model.layers
        else:
            raise RuntimeError("⚠️ 无法找到 逐层应用checkpoints，请检查模型结构")
        if layers is not None:
            gc_enabled = self.training and getattr(base, "gradient_checkpointing", False)

            for layer in layers:
                layer_requires_grad = any(p.requires_grad for p in layer.parameters())

                if not layer_requires_grad:
                    with torch.no_grad():
                        layer_output = layer(hidden_states, position_embeddings=position_embeddings)
                        hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

                elif gc_enabled:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            output = module(*inputs, position_embeddings=position_embeddings)
                            return output[0] if isinstance(output, tuple) else output
                        return custom_forward

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        use_reentrant=False
                    )

                else:
                    layer_output = layer(hidden_states, position_embeddings=position_embeddings)
                    hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # 4. 应用最终的 LayerNorm（不 checkpoint）
            if hasattr(base, 'norm'):
                sequence_hidden = base.norm(hidden_states)
            elif hasattr(base, 'model') and hasattr(base.model, 'norm'):
                sequence_hidden = base.model.norm(hidden_states)
            else:
                raise RuntimeError(
                    "[FATAL] cannot find final norm on backbone. "
                    f"type(base)={type(base)} has_norm={hasattr(base,'norm')} "
                    f"has_model={hasattr(base,'model')} has_model_norm={hasattr(getattr(base,'model',None),'norm')}"
                )


        return sequence_hidden

    
    # #个人基因组版本
    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        pos_strand_mask: Optional[Tensor] = None,
        neg_strand_mask: Optional[torch.Tensor] = None,
        sample_track_means: Optional[Tensor] = None,  # ✅ 新增：每个样本自己的 [B, num_tracks] 均值（你这里是 [B,2]）
    ) -> Dict[str, Optional[Tensor]]:
        """
        前向传播（个人基因组版本）：

        - 训练时：必须提供 sample_track_means（每个样本的 plus/minus mean），用它做 targets_scaling 和 predictions_scaling。
        - 推理时：labels=None，则 sample_track_means 可选；不给则回退到 self.track_means（global mean）。
        """
        # Encode tokens into transformer hidden states (embeddings + RoPE + layers)
        sequence_hidden = self._encode_sequence(input_ids)  # [B, L, H]

        # ✅ 严格正确性保护：训练（labels!=None）必须给 per-sample means
        if labels is not None and sample_track_means is None:
            raise ValueError(
                "Personal-genome training requires `sample_track_means` (shape [B, num_tracks]). "
                "You are giving labels but sample_track_means is None."
            )

        # ✅ 训练优先用 per-sample mean；推理可用 global mean
        track_means_used = sample_track_means if sample_track_means is not None else self.track_means

        # 转置为 [B, H, L] 以便 CNN 处理
        x = sequence_hidden.transpose(1, 2)  # [B, H, L]

        # 嵌入投影
        x = self.embedd_proj(x)  # [B, proj_dim, L]

        # 使用 UNet 进行特征提取
        x = self.unet(x)  # [B, proj_dim, L]

        # 每个轨迹的输出头，应用 softplus 激活（不在此处做缩放）
        head_outputs = []
        for i, name in enumerate(self.task_head):
            out = self.output_heads[name](x)  # [B, C_h, L]
            out = F.softplus(out)  # scale applied after concatenation
            head_outputs.append(out)

        logits = torch.cat(head_outputs, dim=1)  # [B, num_tracks, L]
        # 在所有 head 拼接后，按通道应用可学习缩放（self.scale 的长度应为总 track 数）
        scale_vec = F.softplus(self.scale).view(1, -1, 1)  # [1, num_tracks, 1]
        logits = logits * scale_vec

        # 转置为 [B, L, num_tracks] 以匹配下游代码
        logits = logits.transpose(1, 2)  # [B, L, num_tracks]

        # 计算损失（使用 scaled labels）
        loss = None
        per_head_losses = None
        per_head_metrics = None

        if labels is not None:
            scaled_labels = targets_scaling_torch(
                targets=labels,
                track_means=track_means_used,          # ✅ 关键改动 1：用 per-sample mean
                apply_squashing=self.apply_squashing
            )
            loss, per_head_losses = self._compute_loss(
                logits, scaled_labels, pos_strand_mask, neg_strand_mask
            )

        # 将预测值缩放回原始尺度（用于返回与用于metrics计算）
        logits = predictions_scaling_torch(
            predictions=logits,
            track_means=track_means_used,              # ✅ 关键改动 2：用同一套 per-sample mean
            apply_squashing=self.apply_squashing
        )

        # 计算 metrics（使用原尺度的 logits 与原始 labels）
        if labels is not None:
            per_head_metrics = self._compute_metrics(logits, labels, pos_strand_mask, neg_strand_mask)

        return {
            "loss": loss,
            "logits": logits,
            "per_head_losses": per_head_losses,
            "per_head_metrics": per_head_metrics
        }

    
    
    def predict(
        self,
        input_ids: Tensor,
        assay_names: Optional[Union[str, List[str]]] = None,
        biosample_names: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Run the model forward (no labels) and return selected logits.

        Returns a nested dict: { assay_name: { biosample_name: tensor[B, L, 1] } }.
        If assay_names or biosample_names is None, all available names are used.
        
        TODO: predict 方法还可以包装一下，指定输出层进行 forward。
        """
        # normalize inputs to lists
        if assay_names is None:
            assay_list = list(self.task_head)
        elif isinstance(assay_names, str):
            assay_list = [assay_names]
        else:
            assay_list = list(assay_names)

        if biosample_names is None:
            biosample_list = list(self.biosample_names)
        elif isinstance(biosample_names, str):
            biosample_list = [biosample_names]
        else:
            biosample_list = list(biosample_names)

        # validate
        for a in assay_list:
            if a not in self.task_head:
                raise KeyError(f"Assay '{a}' not found in task_head")

        # forward pass without computing loss
        with torch.no_grad():
            out = self.forward(input_ids=input_ids, labels=None)
        logits = out.get("logits")
        if logits is None:
            raise RuntimeError("Model forward did not return logits")
        # logits: [B, L, num_tracks]

        result: Dict[str, Dict[str, Tensor]] = {}
        for a in assay_list:
            result[a] = {}
            for b in biosample_list:
                matches = self.labels_meta_df[
                    (self.labels_meta_df['task_head'] == a) &
                    (self.labels_meta_df['biosample_name'] == b)
                ]
                if matches.shape[0] == 0:
                    # no matching channel -> set None (or skip)
                    result[a][b] = None
                    continue
                if matches.shape[0] > 1:
                    # ambiguous mapping; choose first and warn
                    dist_print(f"[predict] WARNING multiple channels found for ({a}, {b}) — using first match")
                idx = int(matches.index[0])
                # select single channel as [B, L, 1]
                sel = logits[..., idx:idx+1]
                result[a][b] = sel
 
        return result