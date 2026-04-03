import json
import os
import sys
import time

from backend.igv_payload import build_default_prediction_reference
from frontend.config import LOCAL_FASTA_REL, LOCAL_FASTA_INDEX_REL

# DEFAULT_PRED_REFERENCE = build_default_prediction_reference(
#     local_fasta_rel=LOCAL_FASTA_REL,
#     local_fasta_index_rel=LOCAL_FASTA_INDEX_REL,
#     data_dir_abs=DATA_DIR_ABS,
# )

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR_ABS = str(FRONTEND_DIR / "data")
BACKEND_CACHE_DIR_ABS = str(Path(DATA_DIR_ABS) / "cache" / "backend_uploaded")


def _load_env_file(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


_load_env_file(ROOT_DIR / ".env")
_load_env_file(FRONTEND_DIR / ".env")

# Add root to sys.path so backend.* and src.* imports work
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Reuse central config constants
from frontend.config import (  # noqa: E402
    HG38_FASTA_PATH,
    INDEX_STAT_JSON,
    BIGWIG_LABELS_META_CSV,
    BASE_MODEL_PATH,
    TOKENIZER_DIR,
    CHECKPOINT_PATH,
    PROJ_DIM,
    NUM_DOWNSAMPLES,
    BOTTLENECK_DIM,
    TARGET_LEN,
    PREDICTION_MAX_POINTS,
    DEFAULT_GENOME,
    LOCAL_GTF_REL,
)

from backend.prediction_service import (  # noqa: E402
    init_predictor,
    require_predictor,
    release_predictor,
    reset_upload_cache,
    cache_uploaded_file,
    run_prediction_fasta,
    run_prediction_vcf,
)
from backend.igv_payload import (  # noqa: E402
    build_default_prediction_reference,
    build_prediction_payloads,
)

app = FastAPI(title="Genos-Personal-Server Backend API")

DEFAULT_PRED_REFERENCE = None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictFastaRequest(BaseModel):
    chrom: str
    start: int
    fasta_path: str          # stable cached path sent from frontend
    sample_id: Optional[str] = None
    genome: str = "hg38"


class PredictVcfRequest(BaseModel):
    chrom: str
    start: int
    vcf_path: str            # stable cached path sent from frontend
    sample_id: Optional[str] = None
    genome: str = "hg38"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

# @app.on_event("startup")
# def _startup():
#     reset_upload_cache(ATAC_CACHE_DIR_ABS)
#     t0 = time.time()
#     init_predictor(
#         index_stat_json=INDEX_STAT_JSON,
#         bigwig_labels_meta_csv=BIGWIG_LABELS_META_CSV,
#         base_model_path=BASE_MODEL_PATH,
#         tokenizer_dir=TOKENIZER_DIR,
#         ckpt_model_safetensors=CHECKPOINT_PATH,
#         proj_dim=PROJ_DIM,
#         num_downsamples=NUM_DOWNSAMPLES,
#         bottleneck_dim=BOTTLENECK_DIM,
#         loss_func="mse",
#         use_flash_attn=False,
#     )
#     print(f"[Backend] Predictor initialized in {time.time() - t0:.2f}s", flush=True)

@app.on_event("startup")
def _startup():
    global DEFAULT_PRED_REFERENCE
    DEFAULT_PRED_REFERENCE = build_default_prediction_reference(
        local_fasta_rel=LOCAL_FASTA_REL,
        local_fasta_index_rel=LOCAL_FASTA_INDEX_REL,
        data_dir_abs=DATA_DIR_ABS,
    )

    reset_upload_cache(BACKEND_CACHE_DIR_ABS)
    t0 = time.time()
    init_predictor(
        index_stat_json=INDEX_STAT_JSON,
        bigwig_labels_meta_csv=BIGWIG_LABELS_META_CSV,
        base_model_path=BASE_MODEL_PATH,
        tokenizer_dir=TOKENIZER_DIR,
        ckpt_model_safetensors=CHECKPOINT_PATH,
        proj_dim=PROJ_DIM,
        num_downsamples=NUM_DOWNSAMPLES,
        bottleneck_dim=BOTTLENECK_DIM,
        loss_func="mse",
        use_flash_attn=False,
    )
    print(f"[Backend] Predictor initialized in {time.time() - t0:.2f}s", flush=True)


@app.on_event("shutdown")
def _shutdown():
    released = release_predictor()
    if released:
        print("[Backend] Predictor released and CUDA cache cleaned", flush=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------------------
# File upload endpoint
# Gradio uploads files to its own temp dir; the frontend calls this endpoint
# to copy them into the stable backend cache before prediction.
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receive a file (FASTA or VCF) from the frontend and store it in the
    backend cache. Returns the stable path for subsequent predict calls.
    """
    try:
        os.makedirs(BACKEND_CACHE_DIR_ABS, exist_ok=True)
        import re, time as _time
        stem, ext = os.path.splitext(file.filename or "uploaded")
        safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "uploaded"
        dst_name = f"{safe_stem}_{int(_time.time() * 1000)}{ext}"
        dst_path = os.path.join(BACKEND_CACHE_DIR_ABS, dst_name)

        contents = await file.read()
        with open(dst_path, "wb") as fh:
            fh.write(contents)
        try:
            os.chmod(dst_path, 0o644)
        except Exception:
            pass

        return {"ok": True, "path": dst_path, "filename": file.filename}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Predict — FASTA mode
# ---------------------------------------------------------------------------

@app.post("/predict/fasta")
def predict_fasta(req: PredictFastaRequest):
    try:
        if req.genome != DEFAULT_GENOME:
            return {"ok": False, "error": f"Unsupported genome '{req.genome}'. Only '{DEFAULT_GENOME}' is supported."}

        predictor = require_predictor()
        result = run_prediction_fasta(
            predictor=predictor,
            fasta_path=req.fasta_path,
            chrom_raw=req.chrom,
            start_raw=req.start,
            window_size=TARGET_LEN,
            max_points=PREDICTION_MAX_POINTS,
            sample_id=req.sample_id,
        )

        payload = build_prediction_payloads(
            genome=req.genome,
            chrom=result["chrom"],
            user_start=result["start"],
            user_end=result["end"],
            plus_features=result["plus_features"],
            minus_features=result["minus_features"],
            data_dir_abs=DATA_DIR_ABS,
            local_gtf_rel=LOCAL_GTF_REL,
            track_name=result["sample_id"],
            default_pred_reference=DEFAULT_PRED_REFERENCE,  # 新增这一行
        )

        msg = (
            "<span style='color:#666;'>"
            f"✅ FASTA prediction done | sample={result['sample_id']} | "
            f"locus={result['chrom']}:{result['start']}-{result['end']} | "
            f"elapsed={result['elapsed']:.2f}s"
            "</span>"
        )
        return {
            "ok": True,
            "message": msg,
            "payload": json.dumps(payload, ensure_ascii=False),
            "snp_info": None,
        }

    except Exception as e:
        print(f"[predict/fasta] error: {e}", flush=True)
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Predict — VCF mode
# ---------------------------------------------------------------------------

@app.post("/predict/vcf")
def predict_vcf(req: PredictVcfRequest):
    try:
        if req.genome != DEFAULT_GENOME:
            return {"ok": False, "error": f"Unsupported genome '{req.genome}'. Only '{DEFAULT_GENOME}' is supported."}

        predictor = require_predictor()
        result = run_prediction_vcf(
            predictor=predictor,
            hg38_fasta_path=HG38_FASTA_PATH,
            vcf_path=req.vcf_path,
            chrom_raw=req.chrom,
            start_raw=req.start,
            window_size=TARGET_LEN,
            max_points=PREDICTION_MAX_POINTS,
            cache_dir_abs=BACKEND_CACHE_DIR_ABS,
            sample_id=req.sample_id,
        )

        payload = build_prediction_payloads(
            genome=req.genome,
            chrom=result["chrom"],
            user_start=result["start"],
            user_end=result["end"],
            plus_features=result["plus_features"],
            minus_features=result["minus_features"],
            data_dir_abs=DATA_DIR_ABS,
            local_gtf_rel=LOCAL_GTF_REL,
            track_name=result["sample_id"],
            default_pred_reference=DEFAULT_PRED_REFERENCE,  # 新增这一行
        )

        snp_info = result.get("snp_info") or {}
        snp_msg = ""
        if snp_info.get("warning"):
            snp_msg = f" | ⚠️ {snp_info['warning']}"

        msg = (
            "<span style='color:#666;'>"
            f"✅ VCF prediction done | sample={result['sample_id']} | "
            f"locus={result['chrom']}:{result['start']}-{result['end']} | "
            f"SNPs applied={snp_info.get('applied', 0)} | "
            f"elapsed={result['elapsed']:.2f}s"
            f"{snp_msg}"
            "</span>"
        )
        return {
            "ok": True,
            "message": msg,
            "payload": json.dumps(payload, ensure_ascii=False),
            "snp_info": snp_info,
        }

    except Exception as e:
        print(f"[predict/vcf] error: {e}", flush=True)
        return {"ok": False, "error": str(e)}