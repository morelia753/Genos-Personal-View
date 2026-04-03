import os
import time
import gzip
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyBigWig
import pyfaidx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_model as safe_load_model
from scipy.ndimage import gaussian_filter1d


# ===== 工具函数（来自推理脚本并做适配） =====

def compute_track_mean(bigwig_path: str, chrom: str = "chr19") -> float:
    bw = pyBigWig.open(bigwig_path)
    total_sum = 0.0
    total_bases = 0
    try:
        if chrom in bw.chroms():
            intervals = bw.intervals(chrom)
            if intervals:
                for start, end, value in intervals:
                    if value != 0 and not np.isnan(value):
                        span = end - start
                        total_sum += float(value) * span
                        total_bases += span
    finally:
        bw.close()
    return total_sum / total_bases if total_bases > 0 else 1.0


def compute_track_mean_fast(bw_obj: pyBigWig.pyBigWig, bw_path: str, chrom: str = "chr19") -> float:
    try:
        chrom_len = bw_obj.chroms().get(chrom, None)
        if chrom_len is None:
            return 1.0
        v = bw_obj.stats(chrom, 0, chrom_len, type="mean", exact=True)[0]
        if v is None or np.isnan(v) or v <= 0:
            raise ValueError("invalid mean")
        return float(v)
    except Exception:
        try:
            v = compute_track_mean(bw_path, chrom=chrom)
            return float(v) if np.isfinite(v) and v > 0 else 1.0
        except Exception:
            return 1.0


def predictions_unscaling(logits: np.ndarray, track_means: np.ndarray) -> np.ndarray:
    preds = np.expm1(logits)
    preds = preds * track_means[:, None]
    preds = np.clip(preds, a_min=0.0, a_max=None)
    return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


def proc_signal(raw_vals: np.ndarray, target_len: int = 32000) -> np.ndarray:
    vals = np.nan_to_num(raw_vals, nan=0.0)
    if len(vals) > target_len:
        vals = vals[:target_len]
    elif len(vals) < target_len:
        vals = np.pad(vals, (0, target_len - len(vals)), mode='constant', constant_values=0.0)
    return vals


def gaussian_smooth(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return arr
    return gaussian_filter1d(arr, sigma=sigma, mode='nearest')


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.squeeze()
    if torch.is_tensor(x):
        t = x.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype in (torch.float16, torch.bfloat16):
            t = t.float()
        return t.squeeze().numpy()
    return np.asarray(x).squeeze()


# ===== 模型结构（从 inference_fragments.py 移入） =====

class ATAC_Encoder(nn.Module):
    def __init__(self, output_dim=4096):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(512, output_dim, kernel_size=1)

    def forward(self, atac_signal):
        x = atac_signal.unsqueeze(1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = self.conv5(x)
        return x


class InferenceMultiModalPredictor(nn.Module):
    def __init__(self, base_model, atac_encoder):
        super().__init__()
        self.base = base_model
        self.atac_encoder = atac_encoder

        self.fusion = nn.Sequential(
            nn.Conv1d(4096, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.enc_blocks = nn.ModuleList()
        in_ch = 512
        self.skip_channels = []
        for _ in range(7):
            out_ch = in_ch + 64
            self.skip_channels.append(in_ch)
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            )
            in_ch = out_ch

        bottleneck_ch = in_ch
        self.atac_bottleneck_proj = nn.Sequential(
            nn.Conv1d(4096, bottleneck_ch, kernel_size=1),
            nn.GELU(),
        )

        self.bottleneck_cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_ch,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        self.bottleneck_norm1 = nn.LayerNorm(bottleneck_ch)
        self.bottleneck_norm2 = nn.LayerNorm(bottleneck_ch)
        self.bottleneck_ffn = nn.Sequential(
            nn.Linear(bottleneck_ch, bottleneck_ch * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_ch * 2, bottleneck_ch),
            nn.Dropout(0.1)
        )

        self.dec_blocks = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        dec_in = bottleneck_ch
        for skip_ch in self.skip_channels[::-1]:
            self.dec_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(dec_in, dec_in, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(dec_in),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(dec_in, dec_in, kernel_size=3, padding=1),
                    nn.BatchNorm1d(dec_in),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            )
            dec_out = skip_ch
            self.skip_proj.append(
                nn.Sequential(
                    nn.Conv1d(dec_in + skip_ch, dec_out, kernel_size=1),
                    nn.BatchNorm1d(dec_out),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )
            dec_in = dec_out

        self.final_head = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 2, kernel_size=1)
        )

    def _resolve_layers(self):
        if hasattr(self.base, 'model') and hasattr(self.base.model, 'layers'):
            return self.base.model.layers
        if hasattr(self.base, 'encoder') and hasattr(self.base.encoder, 'layer'):
            return self.base.encoder.layer
        if hasattr(self.base, 'transformer') and hasattr(self.base.transformer, 'h'):
            return self.base.transformer.h
        if hasattr(self.base, 'layers'):
            return self.base.layers
        raise RuntimeError("无法识别 base model 的层结构")

    def _resolve_rotary(self):
        if hasattr(self.base, 'model') and hasattr(self.base.model, 'rotary_emb'):
            return self.base.model.rotary_emb
        if hasattr(self.base, 'rotary_emb'):
            return self.base.rotary_emb
        raise RuntimeError("无法找到 rotary_emb 模块")

    def forward(self, input_ids, atac_signal):
        device = input_ids.device
        B, L = input_ids.shape

        with torch.no_grad():
            inputs_embeds = self.base.get_input_embeddings()(input_ids)

        atac_highres = self.atac_encoder(atac_signal)
        atac_embeds = atac_highres.transpose(1, 2)

        position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        rotary_emb = self._resolve_rotary()
        cos, sin = rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        layers = self._resolve_layers()

        h = inputs_embeds
        with torch.no_grad():
            for layer in layers[:-1]:
                out = layer(h, position_embeddings=position_embeddings)
                h = out[0] if isinstance(out, tuple) else out

        fused_h = h + atac_embeds
        out = layers[-1](fused_h, position_embeddings=position_embeddings)
        dna_features = out[0] if isinstance(out, tuple) else out

        x = dna_features.transpose(1, 2)
        x = self.fusion(x)

        skip_connections = []
        for enc_block in self.enc_blocks:
            skip_connections.append(x)
            x = enc_block(x)
            x = F.max_pool1d(x, kernel_size=2)

        atac_bottleneck = F.adaptive_avg_pool1d(atac_highres, x.size(2))
        atac_bottleneck = self.atac_bottleneck_proj(atac_bottleneck).transpose(1, 2)
        dna_bottleneck = x.transpose(1, 2)

        attn_out, _ = self.bottleneck_cross_attn(
            query=dna_bottleneck,
            key=atac_bottleneck,
            value=atac_bottleneck
        )
        fused = self.bottleneck_norm1(dna_bottleneck + attn_out)
        fused = fused + self.bottleneck_ffn(fused)
        fused = self.bottleneck_norm2(fused)
        x = fused.transpose(1, 2)

        skip_connections = skip_connections[::-1]
        for i, (dec_block, proj_layer) in enumerate(zip(self.dec_blocks, self.skip_proj)):
            x = dec_block(x)
            skip = skip_connections[i]
            if x.size(2) != skip.size(2):
                x = F.interpolate(x, size=skip.size(2), mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = proj_layer(x)

        logits = self.final_head(x)
        return logits

class GenosRegPredictor:

    def __init__(self,
                 fasta_path: str,
                 base_model_path: str,
                 checkpoint_path: str,
                 biosample_configs: Optional[Dict[str, Dict[str, str]]] = None,
                 target_len: int = 32000,
                 device: Optional[str] = None,
                 track_mean_chrom: str = "chr19",
                 precompute_track_means: bool = False,
                 track_mean_cache_path: Optional[str] = None,
                 fixed_track_mean: float = 0.18,
                 hf_parallel_loading: bool = True,
                 hf_parallel_workers: int = 8,
                 local_files_only: bool = True):

        self.target_len = int(target_len)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_bf16 = self.device.type == "cuda"
        self.model_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        self.track_mean_chrom = track_mean_chrom
        self.track_mean_cache_path = track_mean_cache_path
        self.fixed_track_mean = float(fixed_track_mean)
        self.biosample_configs = biosample_configs if isinstance(biosample_configs, dict) else {}

        if hf_parallel_loading:
            os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
            os.environ["HF_PARALLEL_LOADING_WORKERS"] = str(max(1, int(hf_parallel_workers)))
            print(f"[Init] HF parallel loading enabled, workers={hf_parallel_workers}")

        print("[Init] Opening FASTA...")
        self.fasta = pyfaidx.Fasta(fasta_path)

        print("[Init] Loading base model...")
        load_kwargs = dict(
            trust_remote_code=True,
            revision="main",
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        if self.use_bf16:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            load_kwargs["dtype"] = torch.bfloat16

        self.base_model = AutoModel.from_pretrained(base_model_path, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side='right',
            local_files_only=local_files_only,
        )

        print("[Init] Building predictor model (on CPU)...")
        self.model = InferenceMultiModalPredictor(self.base_model, ATAC_Encoder(output_dim=4096))

        print("[Init] Loading checkpoint weights (CPU)...")
        try:
            safe_load_model(self.model, checkpoint_path, device="cpu")
        except TypeError:
            safe_load_model(self.model, checkpoint_path)

        print("[Init] Moving model to target device...")
        self.model = self.model.to(self.model_dtype).to(self.device)
        self.model.eval()

        self._bw_cache = {}
        self.track_means = {}

        if self.track_mean_cache_path and os.path.exists(self.track_mean_cache_path):
            try:
                cache_obj = json.load(open(self.track_mean_cache_path, "r"))
                if isinstance(cache_obj, dict):
                    self.track_means.update(cache_obj)
                print(f"[Init] Loaded track_mean cache from: {self.track_mean_cache_path}")
            except Exception as e:
                print(f"[Init] Failed to load track_mean cache: {e}")

        if precompute_track_means and self.biosample_configs:
            self._ensure_track_means(list(self.biosample_configs.keys()))
        else:
            print("[Init] Using fixed track_mean for single-ATAC mode, or lazy per-biosample means when configs are provided.")

        print("✅ Model loaded successfully.")
        print(f"Device: {self.device}, dtype: {self.model_dtype}")

    def _get_bw(self, bw_path: str):
        bw = self._bw_cache.get(bw_path)
        if bw is None:
            bw = pyBigWig.open(bw_path)
            self._bw_cache[bw_path] = bw
        return bw

    def _ensure_track_means(self, biosamples: List[str]):
        missing = [b for b in biosamples if b not in self.track_means]
        if not missing:
            return

        print(f"[track_mean] Computing for {len(missing)} biosample(s): {missing}")
        for b in missing:
            if b not in self.biosample_configs:
                continue
            cfg = self.biosample_configs[b]
            plus = compute_track_mean_fast(self._get_bw(cfg["rna_plus_path"]), cfg["rna_plus_path"], self.track_mean_chrom)
            minus = compute_track_mean_fast(self._get_bw(cfg["rna_minus_path"]), cfg["rna_minus_path"], self.track_mean_chrom)
            self.track_means[b] = {"plus": plus, "minus": minus}
            print(f"[track_mean] {b}: plus={plus:.6f}, minus={minus:.6f}")

        if self.track_mean_cache_path:
            try:
                with open(self.track_mean_cache_path, "w") as f:
                    json.dump(self.track_means, f, indent=2)
                print(f"[track_mean] Cache saved to: {self.track_mean_cache_path}")
            except Exception as e:
                print(f"[track_mean] Cache save failed: {e}")

    def _get_signal(self, bw_path: str, chrom: str, start: int, end: int):
        vals = np.array(self._get_bw(bw_path).values(chrom, start, end))
        return proc_signal(vals, target_len=self.target_len)

    def predict(self,
                chrom: str,
                start: int,
                end: Optional[int] = None,
                atac_path: Optional[str] = None,
                biosample_names: Optional[List[str]] = None):

        if end is None:
            end = start + self.target_len

        infer_start = int(start)
        infer_end = infer_start + self.target_len

        raw_seq = str(self.fasta[chrom][infer_start:infer_end]).upper()
        seq = (raw_seq + "N" * self.target_len)[:self.target_len]

        enc = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.target_len,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids = enc["input_ids"].squeeze(0).to(self.device)

        values = {
            "RNA-seq_+": {},
            "RNA-seq_-": {},
        }

        # Single ATAC path mode (used by backend service).
        if atac_path is not None:
            if not os.path.exists(atac_path):
                raise FileNotFoundError(f"ATAC path not found: {atac_path}")
            source_name = os.path.splitext(os.path.basename(atac_path))[0] or "input_atac"
            raw_atac = np.array(self._get_bw(atac_path).values(chrom, infer_start, infer_end))
            raw_atac = np.nan_to_num(raw_atac, nan=0.0)
            clip_hi = np.percentile(raw_atac, 99) if raw_atac.size else 0.0
            atac_vals = proc_signal(np.clip(raw_atac, 0, clip_hi), target_len=self.target_len)
            atac_tensor = torch.tensor(atac_vals, dtype=self.model_dtype, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(input_ids.unsqueeze(0), atac_tensor)
                pred = logits.squeeze(0).float().cpu().numpy()

            tm = np.array([self.fixed_track_mean, self.fixed_track_mean], dtype=np.float32)
            pred_unscaled = predictions_unscaling(pred, tm)
            values["RNA-seq_+"][source_name] = pred_unscaled[0]
            values["RNA-seq_-"][source_name] = pred_unscaled[1]

            return {
                "sequence": seq,
                "position": (chrom, infer_start, infer_end),
                "values": values,
            }

        selected = biosample_names or list(self.biosample_configs.keys())
        if not selected:
            raise ValueError("No biosample names provided and biosample_configs is empty. Provide atac_path for single-ATAC inference.")
        self._ensure_track_means(selected)

        st = time.time()
        for biosample in selected:
            if biosample not in self.biosample_configs:
                raise KeyError(f"Unknown biosample: {biosample}")
            cfg = self.biosample_configs[biosample]

            raw_atac = np.array(self._get_bw(cfg["atac_path"]).values(chrom, infer_start, infer_end))
            raw_atac = np.nan_to_num(raw_atac, nan=0.0)
            clip_hi = np.percentile(raw_atac, 99) if raw_atac.size else 0.0
            atac_vals = proc_signal(np.clip(raw_atac, 0, clip_hi), target_len=self.target_len)
            atac_tensor = torch.tensor(atac_vals, dtype=self.model_dtype, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(input_ids.unsqueeze(0), atac_tensor)
                pred = logits.squeeze(0).float().cpu().numpy()

            tm = np.array([
                self.track_means[biosample]["plus"],
                self.track_means[biosample]["minus"],
            ], dtype=np.float32)
            pred_unscaled = predictions_unscaling(pred, tm)

            values["RNA-seq_+"][biosample] = pred_unscaled[0]
            values["RNA-seq_-"][biosample] = pred_unscaled[1]

        print(f"Inference time: {time.time() - st:.2f} s for {len(selected)} biosample(s)")

        return {
            "sequence": seq,
            "position": (chrom, infer_start, infer_end),
            "values": values,
        }

    def close(self):
        try:
            self.fasta.close()
        except Exception:
            pass
        for bw in self._bw_cache.values():
            try:
                bw.close()
            except Exception:
                pass
        self._bw_cache.clear()


if __name__ == "__main__":
    # Smoke test for local debugging:
    # 1) instantiate GenosRegPredictor
    # 2) run single-ATAC predict once
    #
    # Required env vars:
    #   FASTA_PATH
    #   BASE_MODEL_PATH
    #   CHECKPOINT_PATH
    #   TEST_ATAC_PATH
    # Optional env vars:
    #   TEST_CHROM (default: chr19)
    #   TEST_START (default: 49496000)
    #   TARGET_LEN (default: 32000)
    #   FIXED_TRACK_MEAN (default: 0.18)
    #   LOCAL_FILES_ONLY (default: 1)
    def _load_dotenv() -> Optional[str]:
        # Search current directory and parents for a .env file.
        cur = os.path.abspath(os.getcwd())
        while True:
            candidate = os.path.join(cur, ".env")
            if os.path.isfile(candidate):
                loaded = 0
                with open(candidate, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                            loaded += 1
                print(f"[MainTest] Loaded .env: {candidate} (new vars: {loaded})")
                return candidate

            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

        print("[MainTest] No .env found while searching parent directories.")
        return None

    def _require_env(name: str) -> str:
        val = os.getenv(name, "").strip()
        if not val:
            raise RuntimeError(f"Missing required env var: {name}")
        return val

    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    predictor = None
    try:
        _load_dotenv()

        fasta_path = _require_env("FASTA_PATH")
        base_model_path = _require_env("BASE_MODEL_PATH")
        checkpoint_path = _require_env("CHECKPOINT_PATH")
        test_atac_path = _require_env("TEST_ATAC_PATH")

        test_chrom = os.getenv("TEST_CHROM", "chr19").strip()
        test_start = int(os.getenv("TEST_START", "49496000"))
        target_len = int(os.getenv("TARGET_LEN", "32000"))
        fixed_track_mean = float(os.getenv("FIXED_TRACK_MEAN", "0.18"))
        local_files_only = _env_bool("LOCAL_FILES_ONLY", True)

        print("[MainTest] Instantiating GenosRegPredictor...")
        predictor = GenosRegPredictor(
            fasta_path=fasta_path,
            base_model_path=base_model_path,
            checkpoint_path=checkpoint_path,
            target_len=target_len,
            fixed_track_mean=fixed_track_mean,
            local_files_only=local_files_only,
            precompute_track_means=False,
        )

        print("[MainTest] Running single-ATAC predict...")
        result = predictor.predict(
            chrom=test_chrom,
            start=test_start,
            atac_path=test_atac_path,
        )

        chrom, start, end = result["position"]
        plus_keys = list(result["values"]["RNA-seq_+"].keys())
        minus_keys = list(result["values"]["RNA-seq_-"].keys())

        print("[MainTest] Predict succeeded.")
        print(f"[MainTest] Position: {chrom}:{start}-{end}")
        print(f"[MainTest] Sequence length: {len(result['sequence'])}")
        print(f"[MainTest] RNA-seq_+ tracks: {plus_keys}")
        print(f"[MainTest] RNA-seq_- tracks: {minus_keys}")
        if plus_keys:
            arr = result["values"]["RNA-seq_+"][plus_keys[0]]
            print(f"[MainTest] First RNA-seq_+ track shape: {arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

    except Exception as exc:
        print(f"[MainTest] Failed: {exc}")
        raise
    finally:
        if predictor is not None:
            predictor.close()
            print("[MainTest] Predictor closed.")
