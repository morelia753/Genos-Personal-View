import gc
import gzip
import json
import math
import os
import re
import shutil
import time
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pyfaidx

_PREDICTOR = None  # model_bundle from build_model_for_online_inference


# =============================================================================
# Predictor lifecycle
# =============================================================================

def init_predictor(
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
):
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR

    from backend.predict_user_region_online import build_model_for_online_inference

    for p in [index_stat_json, bigwig_labels_meta_csv, base_model_path,
              tokenizer_dir, ckpt_model_safetensors]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path does not exist: {p}")

    _PREDICTOR = build_model_for_online_inference(
        index_stat_json=index_stat_json,
        bigwig_labels_meta_csv=bigwig_labels_meta_csv,
        base_model_path=base_model_path,
        tokenizer_dir=tokenizer_dir,
        ckpt_model_safetensors=ckpt_model_safetensors,
        proj_dim=proj_dim,
        num_downsamples=num_downsamples,
        bottleneck_dim=bottleneck_dim,
        loss_func=loss_func,
        use_flash_attn=use_flash_attn,
        deploy_plus_mean=deploy_plus_mean,
        deploy_minus_mean=deploy_minus_mean,
        device=device,
    )
    return _PREDICTOR


def require_predictor():
    if _PREDICTOR is None:
        raise RuntimeError("Predictor is not initialized")
    return _PREDICTOR


def release_predictor() -> bool:
    global _PREDICTOR
    predictor = _PREDICTOR
    _PREDICTOR = None
    if predictor is None:
        return False
    try:
        del predictor
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    return True


# =============================================================================
# Upload cache helpers  (mirrors Genos-Reg design)
# =============================================================================

def reset_upload_cache(cache_dir_abs: str):
    if os.path.isdir(cache_dir_abs):
        shutil.rmtree(cache_dir_abs, ignore_errors=True)
    os.makedirs(cache_dir_abs, exist_ok=True)


def cache_uploaded_file(src_path: str, cache_dir_abs: str) -> str:
    os.makedirs(cache_dir_abs, exist_ok=True)
    base_name = os.path.basename(src_path)
    stem, ext = os.path.splitext(base_name)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "uploaded"
    target_name = f"{safe_stem}_{int(time.time() * 1000)}{ext}"
    dst_path = os.path.join(cache_dir_abs, target_name)
    shutil.copy2(src_path, dst_path)
    try:
        os.chmod(dst_path, 0o644)
    except Exception:
        pass
    return dst_path


# =============================================================================
# Input normalisation helpers
# =============================================================================

def normalise_chrom(raw: str) -> str:
    """Accept '1', 'chr1', 'CHR1' — always return 'chr1' form."""
    s = str(raw).strip()
    if not s:
        raise ValueError("Chromosome cannot be empty")
    if not s.lower().startswith("chr"):
        s = "chr" + s
    return s


def parse_start(raw) -> int:
    try:
        v = int(str(raw).strip().replace(",", ""))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid start position: {raw!r}")
    if v < 0:
        raise ValueError(f"Start position must be >= 0, got {v}")
    return v


def get_chrom_length(fasta_path: str, chrom: str) -> int:
    """Return chromosome length from a FASTA file."""
    fa = pyfaidx.Fasta(fasta_path)
    try:
        if chrom not in fa.keys():
            raise KeyError(
                f"Chromosome '{chrom}' not found in FASTA. "
                f"Available (first 10): {list(fa.keys())[:10]}"
            )
        return len(fa[chrom])
    finally:
        try:
            fa.close()
        except Exception:
            pass


def compute_window(
    fasta_path: str,
    chrom: str,
    start: int,
    window_size: int,
) -> Tuple[int, int, Optional[str]]:
    """
    Return (actual_start, actual_end, warning_message).

    Rules:
    - Ideal window: [start, start + window_size)
    - If that fits entirely  -> return it, no warning.
    - If start + window_size > chrom_len:
        * If start >= chrom_len -> error (start is beyond chromosome end)
        * Otherwise -> reject and inform user of the last valid window
    """
    chrom_len = get_chrom_length(fasta_path, chrom)

    if start >= chrom_len:
        raise ValueError(
            f"Start position {start} is beyond the end of {chrom} "
            f"(chromosome length = {chrom_len:,}). Please choose a smaller start."
        )

    ideal_end = start + window_size
    if ideal_end <= chrom_len:
        return start, ideal_end, None

    # Not enough room: compute the last valid window and reject
    last_valid_start = max(0, chrom_len - window_size)
    last_valid_end = last_valid_start + window_size
    raise ValueError(
        f"Insufficient sequence: {chrom}:{start} to chromosome end is only "
        f"{chrom_len - start:,} bp, but the model requires {window_size:,} bp. "
        f"Suggested last valid window: {chrom}:{last_valid_start}-{last_valid_end}"
    )


# =============================================================================
# VCF helpers
# =============================================================================

def _open_vcf_text(vcf_path: str):
    """
    Open VCF as text, supporting both plain .vcf and gzip-compressed .vcf.gz/.bgz.
    """
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    # First trust extension; fallback to gzip magic-byte sniffing.
    if vcf_path.endswith((".gz", ".bgz", ".bgzf")):
        return gzip.open(vcf_path, "rt", encoding="utf-8", errors="replace")

    with open(vcf_path, "rb") as fb:
        magic = fb.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(vcf_path, "rt", encoding="utf-8", errors="replace")

    return open(vcf_path, "r", encoding="utf-8", errors="replace")


def parse_vcf_snps(
    vcf_path: str,
    chrom: str,
    start: int,
    end: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Parse VCF and return only SNPs within [start, end).

    Returns:
        (snp_list, total_variants_in_region, indel_count)

    Each SNP dict: {"pos": int (0-based), "ref": str, "alt": str}
    """
    snps = []
    total_in_region = 0
    indel_count = 0

    with _open_vcf_text(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue

            vcf_chrom = parts[0].strip()
            # normalise VCF chrom to match our convention
            if not vcf_chrom.lower().startswith("chr"):
                vcf_chrom = "chr" + vcf_chrom

            if vcf_chrom != chrom:
                continue

            try:
                pos_1based = int(parts[1])
            except ValueError:
                continue

            pos_0based = pos_1based - 1  # convert to 0-based
            if not (start <= pos_0based < end):
                continue

            ref = parts[3].strip().upper()
            alt = parts[4].strip().upper().split(",")[0]  # take first ALT allele

            total_in_region += 1

            # SNP: both REF and ALT are single bases
            if len(ref) == 1 and len(alt) == 1 and alt not in (".", "*", "<"):
                snps.append({"pos": pos_0based, "ref": ref, "alt": alt})
            else:
                indel_count += 1

    return snps, total_in_region, indel_count


def apply_snps_to_sequence(seq: str, start: int, snps: List[Dict[str, Any]]) -> str:
    """
    Apply a list of SNPs to a sequence string.

    seq    : the reference sequence for [start, start+len(seq))
    start  : genomic start coordinate (0-based) of seq
    snps   : list of {"pos": int (0-based genomic), "ref": str, "alt": str}
    """
    seq_list = list(seq)
    applied = 0
    skipped_ref_mismatch = 0

    for snp in snps:
        local_idx = snp["pos"] - start
        if not (0 <= local_idx < len(seq_list)):
            continue
        if seq_list[local_idx] != snp["ref"]:
            skipped_ref_mismatch += 1
            continue
        seq_list[local_idx] = snp["alt"]
        applied += 1

    return "".join(seq_list), applied, skipped_ref_mismatch


# =============================================================================
# signal_to_features  (same contract as Genos-Reg)
# =============================================================================

def signal_to_features(
    chrom: str,
    start: int,
    end: int,
    values: np.ndarray,
    max_points: int = 900,
) -> List[Dict]:
    n = len(values)
    if n == 0:
        return []
    step = max(1, (n + max_points - 1) // max_points)
    feats = []
    for i in range(0, n, step):
        j = min(i + step, n)
        s = start + i
        e = min(start + j, end)
        if e <= s:
            continue
        chunk = values[i:j]
        finite = chunk[np.isfinite(chunk)]
        mean_v = float(finite.mean()) if len(finite) > 0 else 0.0
        feats.append({"chr": chrom, "start": int(s), "end": int(e), "value": mean_v})
    return feats


# =============================================================================
# Core prediction runner
# =============================================================================

def run_prediction_fasta(
    *,
    predictor,
    fasta_path: str,
    chrom_raw: str,
    start_raw,
    window_size: int,
    max_points: int,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    FASTA mode: extract sequence from user-supplied FASTA and predict.
    """
    from backend.predict_user_region_online import predict_one_region

    chrom = normalise_chrom(chrom_raw)
    start = parse_start(start_raw)

    # Will raise ValueError with suggestion if window doesn't fit
    actual_start, actual_end, _ = compute_window(fasta_path, chrom, start, window_size)

    if sample_id is None:
        stem = os.path.splitext(os.path.basename(fasta_path))[0]
        sample_id = f"{stem}_{chrom}_{actual_start}_{actual_end}"

    t0 = time.time()
    result = predict_one_region(
        model_bundle=predictor,
        sample_id=sample_id,
        fasta_path=fasta_path,
        chromosome=chrom,
        start=actual_start,
        end=actual_end,
    )
    elapsed = time.time() - t0

    pred_plus: np.ndarray = result["pred_plus"]
    pred_minus: np.ndarray = result["pred_minus"]

    plus_features = signal_to_features(chrom, actual_start, actual_end, pred_plus, max_points)
    minus_features = signal_to_features(chrom, actual_start, actual_end, pred_minus, max_points)

    return {
        "ok": True,
        "mode": "fasta",
        "chrom": chrom,
        "start": actual_start,
        "end": actual_end,
        "sample_id": sample_id,
        "elapsed": elapsed,
        "plus_features": plus_features,
        "minus_features": minus_features,
        "snp_info": None,
    }


def run_prediction_vcf(
    *,
    predictor,
    hg38_fasta_path: str,
    vcf_path: str,
    chrom_raw: str,
    start_raw,
    window_size: int,
    max_points: int,
    cache_dir_abs: str,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    VCF mode: apply SNPs from VCF onto hg38, write a temp FASTA, then predict.
    """
    from backend.predict_user_region_online import predict_one_region
    from src.utils.data import load_fasta_sequence

    chrom = normalise_chrom(chrom_raw)
    start = parse_start(start_raw)

    actual_start, actual_end, _ = compute_window(hg38_fasta_path, chrom, start, window_size)

    # Extract reference sequence
    fa = pyfaidx.Fasta(hg38_fasta_path)
    try:
        ref_seq = load_fasta_sequence(fa, chrom, actual_start, actual_end, max_length=window_size)
    finally:
        try:
            fa.close()
        except Exception:
            pass

    # Parse VCF SNPs
    snps, total_in_region, indel_count = parse_vcf_snps(vcf_path, chrom, actual_start, actual_end)

    snp_info = {
        "total_variants_in_region": total_in_region,
        "indel_count": indel_count,
        "snp_count": len(snps),
        "warning": None,
    }

    if indel_count > 0:
        snp_info["warning"] = (
            f"{indel_count} indel variant(s) in this region were removed; "
            f"only {len(snps)} SNP(s) were applied."
        )

    # Apply SNPs (or use ref seq if none found)
    if len(snps) == 0:
        mutated_seq = ref_seq
        snp_info["warning"] = (
            (snp_info["warning"] or "") +
            " No SNPs found in this region — predicting on reference sequence."
        ).strip()
    else:
        mutated_seq, applied, skipped = apply_snps_to_sequence(ref_seq, actual_start, snps)
        snp_info["applied"] = applied
        snp_info["skipped_ref_mismatch"] = skipped

    # Write mutated sequence to a temporary FASTA in cache
    os.makedirs(cache_dir_abs, exist_ok=True)
    vcf_stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(os.path.basename(vcf_path))[0])
    tmp_fa_name = f"vcf_mutated_{vcf_stem}_{chrom}_{actual_start}_{actual_end}_{int(time.time()*1000)}.fa"
    tmp_fa_path = os.path.join(cache_dir_abs, tmp_fa_name)

    seq_id = f"{chrom}_{actual_start}_{actual_end}_vcf_mutated"
    with open(tmp_fa_path, "w") as fh:
        fh.write(f">{seq_id}\n")
        # Write in 60-char lines (standard FASTA)
        for i in range(0, len(mutated_seq), 60):
            fh.write(mutated_seq[i:i + 60] + "\n")
    try:
        os.chmod(tmp_fa_path, 0o644)
    except Exception:
        pass

    if sample_id is None:
        sample_id = f"{vcf_stem}_{chrom}_{actual_start}_{actual_end}"

    try:
        t0 = time.time()
        result = predict_one_region(
            model_bundle=predictor,
            sample_id=sample_id,
            fasta_path=tmp_fa_path,
            chromosome=seq_id,   # must match the >header in tmp FASTA
            start=0,
            end=window_size,
        )
        elapsed = time.time() - t0
    finally:
        # Clean up temp FASTA
        try:
            os.remove(tmp_fa_path)
            fai = tmp_fa_path + ".fai"
            if os.path.exists(fai):
                os.remove(fai)
        except Exception:
            pass

    pred_plus: np.ndarray = result["pred_plus"]
    pred_minus: np.ndarray = result["pred_minus"]

    plus_features = signal_to_features(chrom, actual_start, actual_end, pred_plus, max_points)
    minus_features = signal_to_features(chrom, actual_start, actual_end, pred_minus, max_points)

    return {
        "ok": True,
        "mode": "vcf",
        "chrom": chrom,
        "start": actual_start,
        "end": actual_end,
        "sample_id": sample_id,
        "elapsed": elapsed,
        "plus_features": plus_features,
        "minus_features": minus_features,
        "snp_info": snp_info,
    }
