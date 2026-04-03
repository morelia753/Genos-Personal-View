#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_user_region_online.py

独立的在线单区域推理脚本（部署/服务模式）。

设计目标：
- 输入：sample_id, fasta_path, chromosome, start, end
- 要求：end - start == window_size（默认 32768），且区间合法
- 不依赖用户真实 RNA-seq / bigWig / per-sample track mean
- 默认使用训练集 global track mean 作为部署期固定缩放值
- 返回 Python dict：
    {
        "sample_id": str,
        "fasta_path": str,
        "chromosome": str,
        "start": int,
        "end": int,
        "pred_plus": np.ndarray(shape=(32768,), dtype=np.float32),
        "pred_minus": np.ndarray(shape=(32768,), dtype=np.float32),
    }

注意：
- 核心函数是 predict_one_region(...)。
- CLI 仅用于手工调试/验证；CLI 不直接打印 32768 长数组，只打印摘要。


调用说明：
1. CLI调用：
cd /mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref

python predict_user_region_online.py \
  --sample_id "USER_DEMO_001" \
  --fasta_path "/你的fasta路径/USER_DEMO_001.hap1.fa" \
  --chrom "chr19" \
  --start 0 \
  --end 32768 \
  --index_stat_json "/mnt/data/index_stat_person100.json" \
  --bigWig_labels_meta "/mnt/data/bigWig_labels_meta.csv" \
  --base_model_path "/mnt/genos100-new/peixunban/qixianzhi/pretrained_model/Genos-1.2B" \
  --tokenizer_dir "/mnt/genos100-new/peixunban/qixianzhi/pretrained_model/Genos-1.2B" \
  --ckpt_model_safetensors "/你的checkpoint目录/model.safetensors" \
  --proj_dim 1024 \
  --num_downsamples 4 \
  --bottleneck_dim 1536 \
  --loss_func mse \
  --save_npz "/tmp/user_demo_chr19_0_32768.npz"

2. 作为模块被 import 调用：
from predict_user_region_online import build_model_for_online_inference
model_bundle = build_model_for_online_inference(
    index_stat_json="/mnt/data/index_stat_person100.json",
    bigwig_labels_meta_csv="/mnt/data/bigWig_labels_meta.csv",
    base_model_path="/mnt/genos100-new/peixunban/qixianzhi/pretrained_model/Genos-1.2B",
    tokenizer_dir="/mnt/genos100-new/peixunban/qixianzhi/pretrained_model/Genos-1.2B",
    ckpt_model_safetensors="/你的checkpoint目录/model.safetensors",
    proj_dim=1024,
    num_downsamples=4,
    bottleneck_dim=1536,
    loss_func="mse",
    use_flash_attn=False,   # 需要的话可改 True
    device="cuda",          # 或 "cpu"，也可以不写自动判断
)


#### CLI测试示例
root@820659eb57bf:/mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref# python predict_user_region_online.py \
  --sample_id "USER_DEMO_001" \
  --fasta_path "/mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref/deploy_test/input/HG00232.hap1.fa" \
  --chrom "chr19" \
  --start 0 \
  --end 32768 \
  --index_stat_json "/mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref/deploy_test/input/index_stat.json" \
  --bigWig_labels_meta "/mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref/deploy_test/input/bigWig_labels_meta.csv" \
  --base_model_path "/mnt/genos100-new/peixunban/qixianzhi/pretrained_model/Genos-1.2B" \
  --tokenizer_dir "/mnt/genos100-new/peixunban/qixianzhi/pretrained_model/Genos-1.2B" \
  --ckpt_model_safetensors "/mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref/deploy_test/checkpoints/checkpoint-34270/model.safetensors" \
  --proj_dim 1024 \
  --num_downsamples 4 \
  --bottleneck_dim 1536 \
  --loss_func mse \
  --save_npz "/mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref/deploy_test/output/user_demo_1_chr19_0_32768.npz"
[INFO] loading base model ...
`torch_dtype` is deprecated! Use `dtype` instead!
[INFO] loading full checkpoint ...
[INFO] online inference model ready | device=cuda | window_size=32768 | deploy_plus_mean=0.413584444785 | deploy_minus_mean=0.463594237637
[INFO] online inference done.
{
  "sample_id": "USER_DEMO_001",
  "fasta_path": "/mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref/deploy_test/input/HG00232.hap1.fa",
  "chromosome": "chr19",
  "start": 0,
  "end": 32768,
  "pred_plus_shape": [
    32768
  ],
  "pred_minus_shape": [
    32768
  ],
  "pred_plus_dtype": "float32",
  "pred_minus_dtype": "float32",
  "deploy_plus_mean": 0.4135844447852688,
  "deploy_minus_mean": 0.4635942376368138
}
[INFO] saved debug npz: /mnt/genos100-new/peixunban/qixianzhi/02.bidrectional_attention_research/GenOmic_git_personal/personal_GenOmics_factory_4ref/deploy_test/output/user_demo_1_chr19_0_32768.npz
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import pyfaidx

from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------------
# 让脚本在“仓库根目录直接运行”或“被 import 调用”时都能找到 src 包
# ----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from src.model import GenOmics
from src.utils.data import load_fasta_sequence

try:
    from safetensors import safe_open
except Exception:
    safe_open = None


# 100 人训练集 index_stat.json 中的 global_track_mean
# 来源：index_stat_person100.json -> inputs.global_track_mean
DEFAULT_DEPLOY_PLUS_MEAN = 0.4135844447852688
DEFAULT_DEPLOY_MINUS_MEAN = 0.4635942376368138


# ==========================================================================
# 基础工具函数
# ==========================================================================
def ensure_tools_or_die() -> None:
    if safe_open is None:
        raise RuntimeError("[FATAL] safetensors is not importable. Please `pip install safetensors`.")


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[FATAL] json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_region_or_die(fasta_path: str, chromosome: str, start: int, end: int, expected_window_size: int) -> int:
    """
    严格检查区间合法性；返回该染色体长度。
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"[FATAL] fasta not found: {fasta_path}")
    if not chromosome:
        raise ValueError("[FATAL] chromosome is empty")
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError(f"[FATAL] start/end must be int, got {type(start)} / {type(end)}")
    if start < 0:
        raise ValueError(f"[FATAL] start must be >= 0, got {start}")
    if end <= start:
        raise ValueError(f"[FATAL] end must be > start, got start={start}, end={end}")
    if (end - start) != expected_window_size:
        raise ValueError(
            f"[FATAL] region length mismatch: end-start={end-start}, expected window_size={expected_window_size}"
        )

    fasta = pyfaidx.Fasta(fasta_path)
    try:
        if chromosome not in fasta.keys():
            raise KeyError(
                f"[FATAL] chromosome {chromosome} not found in fasta: {fasta_path}. "
                f"Available examples: {list(fasta.keys())[:10]}"
            )
        chrom_len = len(fasta[chromosome])
        if end > chrom_len:
            raise ValueError(
                f"[FATAL] region out of range: {chromosome}:{start}-{end}, chromosome_length={chrom_len}"
            )
        return chrom_len
    finally:
        try:
            fasta.close()
        except Exception:
            pass


@torch.no_grad()
def strict_load_and_verify_full_checkpoint(
    model: torch.nn.Module,
    ckpt_model_safetensors: str,
    allow_missing_track_means_only: bool = True,
) -> None:
    """
    严格加载 full-SFT checkpoint。
    逻辑与你原来的 predict_to_bw_fullsft_mp.py 保持一致。
    """
    if not os.path.exists(ckpt_model_safetensors):
        raise FileNotFoundError(f"[FATAL] checkpoint model.safetensors not found: {ckpt_model_safetensors}")

    with safe_open(ckpt_model_safetensors, framework="pt", device="cpu") as f:
        ckpt_keys = list(f.keys())
        if len(ckpt_keys) == 0:
            raise RuntimeError("[FATAL] checkpoint model.safetensors has no tensors")
        to_load = {k: f.get_tensor(k) for k in ckpt_keys}

    model_sd = model.state_dict()
    missing_in_model = [k for k in to_load.keys() if k not in model_sd]
    if missing_in_model:
        raise KeyError(
            "[FATAL] some checkpoint keys do not exist in current model.state_dict(). "
            f"first20={missing_in_model[:20]}"
        )

    incompatible = model.load_state_dict(to_load, strict=False)

    if len(incompatible.unexpected_keys) > 0:
        raise RuntimeError(
            f"[FATAL] unexpected keys when loading full checkpoint: {incompatible.unexpected_keys[:50]}"
        )

    if len(incompatible.missing_keys) > 0:
        allowed_missing = {"track_means"} if allow_missing_track_means_only else set()
        actual_missing = set(incompatible.missing_keys)
        if actual_missing != allowed_missing:
            raise RuntimeError(
                "[FATAL] missing keys after loading full checkpoint are not allowed. "
                f"missing={sorted(actual_missing)} allowed={sorted(allowed_missing)}"
            )

    sd_after = model.state_dict()
    mismatches = 0
    for k, t_file in to_load.items():
        t_model = sd_after[k]
        t_file_cast = t_file.to(dtype=t_model.dtype)
        t_model_cpu = t_model.detach().cpu()

        if t_model_cpu.shape != t_file_cast.shape:
            raise RuntimeError(
                f"[FATAL] shape mismatch for {k}: model={tuple(t_model_cpu.shape)} file={tuple(t_file_cast.shape)}"
            )

        if not torch.equal(t_model_cpu, t_file_cast):
            mismatches += 1
            if mismatches <= 20:
                print(f"[MISMATCH_FULL] {k}", flush=True)

    if mismatches > 0:
        raise RuntimeError(f"[FATAL] Full checkpoint verification failed: mismatches={mismatches}")


# ==========================================================================
# 模型构建
# ==========================================================================
def _resolve_deploy_track_means(
    index_stat: Dict[str, Any],
    deploy_plus_mean: Optional[float],
    deploy_minus_mean: Optional[float],
) -> Tuple[float, float]:
    """
    优先级：
    1) 显式传入参数
    2) index_stat['inputs']['global_track_mean']
    3) 本文件内置默认常量（来自你上传的 100 人 index_stat）
    """
    if deploy_plus_mean is not None and deploy_minus_mean is not None:
        plus = float(deploy_plus_mean)
        minus = float(deploy_minus_mean)
    else:
        gtm = index_stat.get("inputs", {}).get("global_track_mean", None)
        if isinstance(gtm, dict) and ("plus" in gtm) and ("minus" in gtm):
            plus = float(gtm["plus"])
            minus = float(gtm["minus"])
        else:
            plus = float(DEFAULT_DEPLOY_PLUS_MEAN)
            minus = float(DEFAULT_DEPLOY_MINUS_MEAN)

    if not np.isfinite(plus) or not np.isfinite(minus):
        raise RuntimeError(
            f"[FATAL] deploy track means must be finite, got plus={plus}, minus={minus}"
        )
    if plus <= 0 or minus <= 0:
        raise RuntimeError(
            f"[FATAL] deploy track means must be > 0, got plus={plus}, minus={minus}"
        )
    return plus, minus


def build_model_for_online_inference(
    *,
    index_stat_json: str,
    bigwig_labels_meta_csv: str,
    base_model_path: str,
    tokenizer_dir: str,
    ckpt_model_safetensors: str,
    proj_dim: int,
    num_downsamples: int,
    bottleneck_dim: int,
    loss_func: str = "mse",
    use_flash_attn: bool = False,
    deploy_plus_mean: Optional[float] = None,
    deploy_minus_mean: Optional[float] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    构建在线推理所需的 model/tokenizer/metadata。

    返回：
        {
            "model": model,
            "tokenizer": tokenizer,
            "index_stat": index_stat,
            "labels_meta_df": labels_meta_df,
            "window_size": int,
            "deploy_plus_mean": float,
            "deploy_minus_mean": float,
            "plus_idx": int,
            "minus_idx": int,
            "device": str,
        }
    """
    ensure_tools_or_die()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    index_stat = load_json(index_stat_json)
    labels_meta_df = pd.read_csv(bigwig_labels_meta_csv)

    window_size = int(index_stat["inputs"]["window_size"])

    # 严格解析 plus / minus channel 顺序
    strands = labels_meta_df["strand"].tolist()
    if len(strands) != 2:
        raise RuntimeError(f"[FATAL] expected exactly 2 tracks (plus/minus), got {len(strands)}")
    try:
        plus_idx = strands.index("+")
        minus_idx = strands.index("-")
    except ValueError:
        raise RuntimeError(f"[FATAL] labels_meta strand must contain '+' and '-' but got: {strands}")

    rkwargs = dict(
        trust_remote_code=True,
        revision="main",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if use_flash_attn:
        rkwargs["attn_implementation"] = "flash_attention_2"

    print("[INFO] loading base model ...", flush=True)
    base_model = AutoModel.from_pretrained(base_model_path, **rkwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, revision="main")

    model = GenOmics(
        base_model,
        index_stat=index_stat,
        labels_meta_df=labels_meta_df,
        loss_func=loss_func,
        proj_dim=proj_dim,
        num_downsamples=num_downsamples,
        bottleneck_dim=bottleneck_dim,
    )

    print("[INFO] loading full checkpoint ...", flush=True)
    strict_load_and_verify_full_checkpoint(model, ckpt_model_safetensors)

    plus_mean, minus_mean = _resolve_deploy_track_means(index_stat, deploy_plus_mean, deploy_minus_mean)
    deploy_track_means = torch.tensor([plus_mean, minus_mean], dtype=torch.float32)

    with torch.no_grad():
        if model.track_means.shape != deploy_track_means.shape:
            raise RuntimeError(
                f"[FATAL] model.track_means shape mismatch: model={tuple(model.track_means.shape)} "
                f"deploy={tuple(deploy_track_means.shape)}"
            )
        model.track_means.copy_(deploy_track_means)

    model.to(device)
    model.eval()

    print(
        f"[INFO] online inference model ready | device={device} | window_size={window_size} | "
        f"deploy_plus_mean={plus_mean:.12f} | deploy_minus_mean={minus_mean:.12f}",
        flush=True,
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "index_stat": index_stat,
        "labels_meta_df": labels_meta_df,
        "window_size": window_size,
        "deploy_plus_mean": plus_mean,
        "deploy_minus_mean": minus_mean,
        "plus_idx": plus_idx,
        "minus_idx": minus_idx,
        "device": device,
    }


# ==========================================================================
# 单区域在线推理核心函数
# ==========================================================================
def predict_one_region(
    *,
    model_bundle: Dict[str, Any],
    sample_id: str,
    fasta_path: str,
    chromosome: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    在线单区域推理核心函数。

    参数：
        model_bundle: build_model_for_online_inference(...) 的返回结果
        sample_id: 业务侧传入的样本标识，仅用于回传，不参与任何索引映射
        fasta_path: 用户本机/服务可见路径
        chromosome/start/end: 区间，必须满足 end-start==window_size

    返回：
        {
            "sample_id": str,
            "fasta_path": str,
            "chromosome": str,
            "start": int,
            "end": int,
            "pred_plus": np.ndarray(shape=(32768,), dtype=np.float32),
            "pred_minus": np.ndarray(shape=(32768,), dtype=np.float32),
        }
    """
    if not isinstance(sample_id, str) or len(sample_id.strip()) == 0:
        raise ValueError(f"[FATAL] sample_id must be non-empty str, got {sample_id!r}")

    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    window_size = int(model_bundle["window_size"])
    plus_idx = int(model_bundle["plus_idx"])
    minus_idx = int(model_bundle["minus_idx"])
    device = model_bundle["device"]

    validate_region_or_die(
        fasta_path=fasta_path,
        chromosome=chromosome,
        start=int(start),
        end=int(end),
        expected_window_size=window_size,
    )

    fasta = pyfaidx.Fasta(fasta_path)
    try:
        seq = load_fasta_sequence(
            fasta=fasta,
            chromosome=chromosome,
            start=int(start),
            end=int(end),
            max_length=window_size,
        )
    finally:
        try:
            fasta.close()
        except Exception:
            pass

    if len(seq) != window_size:
        raise RuntimeError(
            f"[FATAL] loaded sequence length mismatch: len(seq)={len(seq)}, expected={window_size}"
        )

    encodings = tokenizer(
        seq,
        padding="max_length",
        max_length=window_size,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
        return_attention_mask=False,
    )
    input_ids = encodings["input_ids"].to(device)

    autocast_enabled = (str(device).startswith("cuda"))
    autocast_dtype = torch.bfloat16

    with torch.no_grad():
        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out = model(input_ids=input_ids, labels=None, sample_track_means=None)
        else:
            out = model(input_ids=input_ids, labels=None, sample_track_means=None)

    logits = out.get("logits", None)
    if logits is None:
        raise RuntimeError("[FATAL] model output logits is None")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != window_size or logits.shape[2] != 2:
        raise RuntimeError(
            f"[FATAL] unexpected logits shape: {tuple(logits.shape)}, expected (1, {window_size}, 2)"
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError("[FATAL] non-finite logits found in online inference output")

    pred = logits.detach().to(device="cpu", dtype=torch.float32).numpy()[0]  # [L, 2]
    pred_plus = pred[:, plus_idx].copy()
    pred_minus = pred[:, minus_idx].copy()

    return {
        "sample_id": str(sample_id),
        "fasta_path": str(fasta_path),
        "chromosome": str(chromosome),
        "start": int(start),
        "end": int(end),
        "pred_plus": pred_plus,
        "pred_minus": pred_minus,
    }


# ==========================================================================
# CLI（用于手工调试）
# ==========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online single-region inference for personal-genome RNA prediction")

    p.add_argument("--sample_id", required=True)
    p.add_argument("--fasta_path", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)

    p.add_argument("--index_stat_json", required=True)
    p.add_argument("--bigWig_labels_meta", required=True)

    p.add_argument("--base_model_path", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--ckpt_model_safetensors", required=True)

    p.add_argument("--proj_dim", type=int, required=True)
    p.add_argument("--num_downsamples", type=int, required=True)
    p.add_argument("--bottleneck_dim", type=int, required=True)
    p.add_argument("--loss_func", type=str, default="mse")
    p.add_argument("--use_flash_attn", action="store_true")

    p.add_argument("--deploy_plus_mean", type=float, default=None)
    p.add_argument("--deploy_minus_mean", type=float, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu; default auto-detect")

    p.add_argument(
        "--save_npz",
        type=str,
        default=None,
        help="Optional. Save output dict arrays to compressed npz for debugging.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 可预先加载的模型上下文，只需要加载一次
    model_bundle = build_model_for_online_inference(
        index_stat_json=args.index_stat_json,
        bigwig_labels_meta_csv=args.bigWig_labels_meta,
        base_model_path=args.base_model_path,
        tokenizer_dir=args.tokenizer_dir,
        ckpt_model_safetensors=args.ckpt_model_safetensors,
        proj_dim=args.proj_dim,
        num_downsamples=args.num_downsamples,
        bottleneck_dim=args.bottleneck_dim,
        loss_func=args.loss_func,
        use_flash_attn=args.use_flash_attn,
        deploy_plus_mean=args.deploy_plus_mean,
        deploy_minus_mean=args.deploy_minus_mean,
        device=args.device,
    )

    # 真实调用在线推理函数，短资源。可反复调用 predict_one_region(...) 来模拟多次在线推理请求。
    result = predict_one_region(
        model_bundle=model_bundle,
        sample_id=args.sample_id,
        fasta_path=args.fasta_path,
        chromosome=args.chrom,
        start=args.start,
        end=args.end,
    )

    # 这里需要使用的其实是            pred_plus=result["pred_plus"],
    #                                pred_minus=result["pred_minus"],
    print("[INFO] online inference done.", flush=True)
    print(
        json.dumps(
            {
                "sample_id": result["sample_id"],
                "fasta_path": result["fasta_path"],
                "chromosome": result["chromosome"],
                "start": result["start"],
                "end": result["end"],
                "pred_plus_shape": list(result["pred_plus"].shape),
                "pred_minus_shape": list(result["pred_minus"].shape),
                "pred_plus_dtype": str(result["pred_plus"].dtype),
                "pred_minus_dtype": str(result["pred_minus"].dtype),
                "deploy_plus_mean": model_bundle["deploy_plus_mean"],
                "deploy_minus_mean": model_bundle["deploy_minus_mean"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if args.save_npz is not None:
        out_dir = os.path.dirname(args.save_npz)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            args.save_npz,
            sample_id=np.array(result["sample_id"], dtype=object),
            fasta_path=np.array(result["fasta_path"], dtype=object),
            chromosome=np.array(result["chromosome"], dtype=object),
            start=np.array(result["start"], dtype=np.int64),
            end=np.array(result["end"], dtype=np.int64),
            pred_plus=result["pred_plus"],
            pred_minus=result["pred_minus"],
        )
        print(f"[INFO] saved debug npz: {args.save_npz}", flush=True)


if __name__ == "__main__":
    main()
