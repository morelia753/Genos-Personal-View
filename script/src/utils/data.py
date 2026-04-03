import numpy as np
import pyBigWig
import os
from typing import Optional, Union, Any
import numbers

def load_bigwig_signal(
    bw: Union[str, Any],
    chromosome: str,
    start: int,
    end: int,
    max_length: Optional[int] = None,
    pad: bool = False,
) -> np.ndarray:
    # bw可以是 bigwig句柄 或者 bigwig file path
    if bw is None:
        raise ValueError("bw is None")
    if not chromosome:
        raise ValueError("chromosome is empty")
    if not isinstance(start, numbers.Integral) or not isinstance(end, numbers.Integral):
        raise TypeError(f"start/end must be int-like, got {type(start)} / {type(end)}")
    if start < 0 or end <= start:
        raise ValueError(f"invalid interval: start={start}, end={end}")
    if max_length is not None and (not isinstance(max_length, int) or max_length <= 0):
        raise ValueError(f"max_length must be positive int or None, got {max_length}")

    opened = False
    bw_handle = bw
    try:
        if isinstance(bw, str):
            if not os.path.exists(bw):
                raise FileNotFoundError(bw)
            bw_handle = pyBigWig.open(bw)
            opened = True

        chroms = bw_handle.chroms()
        if chromosome not in chroms:  
            raise KeyError(f"chromosome not found in bigWig: {chromosome}")
        if end > int(chroms[chromosome]):  
            raise ValueError(f"out of range: {chromosome} len={chroms[chromosome]} end={end}")

        vals = bw_handle.values(chromosome, start, end)
        if vals is None:  
            raise RuntimeError(f"pyBigWig.values returned None: {chromosome}:{start}-{end}")

        arr = np.asarray(vals, dtype=np.float32)
        if arr.shape[0] != (end - start):
            raise RuntimeError(f"length mismatch: expected {end-start}, got {arr.shape[0]}")

        arr = np.nan_to_num(arr, nan=0.0)   # 避免nan值错误 

        if max_length is None:
            return arr

        if arr.shape[0] > max_length:
            return arr[:max_length]
        if pad and arr.shape[0] < max_length:
            return np.pad(arr, (0, max_length - arr.shape[0]), mode="constant", constant_values=0.0) # 填充
        return arr
    finally:
        if opened:
            bw_handle.close()


def load_fasta_sequence(
    fasta: Any,
    chromosome: str,
    start: int,
    end: int,
    max_length: Optional[int] = None,
) -> str:
    """
    从 FASTA 文件中加载指定区域的序列
    
    Args:
        fasta (pyfaidx.Fasta): 已打开的 FASTA 对象
        chromosome (str): 染色体名称
        start (int): 起始位置
        end (int): 终止位置
        max_length (int, optional): 最大长度，仅用于截断（不进行填充）
    
    Returns:
        str: DNA 序列
    """
    if fasta is None:
        raise ValueError("fasta is None")
    if not chromosome:
        raise ValueError("chromosome is empty")

    if not isinstance(start, numbers.Integral) or not isinstance(end, numbers.Integral):
        raise TypeError("start/end must be int")
    if start < 0 or end <= start:
        raise ValueError(f"invalid interval: start={start}, end={end}")
    if max_length is not None and (not isinstance(max_length, int) or max_length <= 0):
        raise ValueError(f"max_length must be positive int or None, got {max_length}")

    # 只支持你原来那种 pyfaidx 风格：fasta[chr][start:end]
    seq = str(fasta[chromosome][start:end]).upper()  # 若不支持/chr 不存在/越界 -> 直接抛异常
    if max_length is not None and len(seq) > max_length:
        return seq[:max_length]
    return seq