import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib import error, request

import gradio as gr
from fastapi import FastAPI
import uvicorn

try:
    from frontend.config import *
    from frontend.igv_payload import (
        build_default_prediction_reference,
    )
except ImportError:
    import sys
    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from config import *
    from igv_payload import (
        build_default_prediction_reference,
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_ABS = os.path.join(BASE_DIR, "data")
REFERENCE_DIR_ABS = os.path.join(DATA_DIR_ABS, "reference")
IGV_LOCAL_ABS = os.path.join(DATA_DIR_ABS, "vendor", "igv.min.js")
FRONTEND_UPLOAD_CACHE_DIR_ABS = os.path.join(DATA_DIR_ABS, "cache", "frontend_uploaded")
BACKEND_UPLOAD_CACHE_DIR_ABS = os.path.join(DATA_DIR_ABS, "cache", "backend_uploaded")


def _load_env_file(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


_load_env_file(os.path.join(Path(BASE_DIR).parent, ".env"))
_load_env_file(os.path.join(BASE_DIR, ".env"))

FRONTEND_HOST = os.getenv("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8010"))
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8011")


# ---------------------------------------------------------------------------
# IGV.js local vendor cache
# ---------------------------------------------------------------------------

def _ensure_local_igv_script() -> str:
    os.makedirs(os.path.dirname(IGV_LOCAL_ABS), exist_ok=True)
    if os.path.exists(IGV_LOCAL_ABS) and os.path.getsize(IGV_LOCAL_ABS) > 1024:
        from urllib.parse import quote
        return "/gradio_api/file=" + quote(IGV_LOCAL_ABS)
    for url in [
        "https://cdn.jsdelivr.net/npm/igv@3.7.0/dist/igv.min.js",
        "https://unpkg.com/igv@3.7.0/dist/igv.min.js",
    ]:
        try:
            with request.urlopen(url, timeout=20) as resp:
                data = resp.read()
            if data and len(data) > 1024:
                with open(IGV_LOCAL_ABS, "wb") as f:
                    f.write(data)
                from urllib.parse import quote
                return "/gradio_api/file=" + quote(IGV_LOCAL_ABS)
        except Exception:
            continue
    return ""


# ---------------------------------------------------------------------------
# Module-level init
# ---------------------------------------------------------------------------

DEFAULT_PRED_REFERENCE = build_default_prediction_reference(
    local_fasta_rel=LOCAL_FASTA_REL,
    local_fasta_index_rel=LOCAL_FASTA_INDEX_REL,
    data_dir_abs=DATA_DIR_ABS,
)
DEFAULT_PRED_REFERENCE_JSON = json.dumps(DEFAULT_PRED_REFERENCE, ensure_ascii=False)
IGV_LOCAL_URL = _ensure_local_igv_script()

HEAD_JS = build_head_js(
    default_pred_reference_json=DEFAULT_PRED_REFERENCE_JSON,
    igv_local_url=IGV_LOCAL_URL,
)


# ---------------------------------------------------------------------------
# Upload cache helpers
# ---------------------------------------------------------------------------

def _reset_upload_caches():
    for cache_dir in (FRONTEND_UPLOAD_CACHE_DIR_ABS, BACKEND_UPLOAD_CACHE_DIR_ABS):
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)


def _materialize_upload(uploaded_file) -> tuple[str | None, str | None]:
    """
    Copy a Gradio-uploaded file to the stable frontend cache.
    Returns (stable_path, error_message).
    """
    if uploaded_file is None:
        return None, None

    if isinstance(uploaded_file, str):
        src = uploaded_file
    elif isinstance(uploaded_file, dict):
        src = uploaded_file.get("path") or uploaded_file.get("name")
    else:
        src = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None)

    if not src or not os.path.exists(src):
        return None, "Uploaded file path is not yet available. Please wait and try again."

    # Already in our cache — reuse
    if str(src).startswith(FRONTEND_UPLOAD_CACHE_DIR_ABS) and os.path.exists(src):
        try:
            os.chmod(src, 0o644)
        except Exception:
            pass
        return src, None

    base_name = os.path.basename(src)
    stem, ext = os.path.splitext(base_name)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "uploaded"
    dst_name = f"{safe_stem}_{int(time.time() * 1000)}{ext}"
    dst_path = os.path.join(FRONTEND_UPLOAD_CACHE_DIR_ABS, dst_name)
    shutil.copy2(src, dst_path)
    try:
        os.chmod(dst_path, 0o644)
    except Exception:
        pass
    return dst_path, None


def _push_file_to_backend(local_path: str) -> tuple[str | None, str | None]:
    """
    POST the file to /upload on the backend so it gets a stable backend path.
    Returns (backend_path, error_message).
    """
    upload_url = f"{BACKEND_API_URL}/upload"
    try:
        with open(local_path, "rb") as fh:
            file_data = fh.read()
        filename = os.path.basename(local_path)
        print("filename ", filename)
        boundary = "----GradioFormBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = request.Request(
            url=upload_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if result.get("ok"):
            return result["path"], None
        return None, result.get("error", "Upload failed")
        
        print("2222222222222222222222222222222222222")
    except Exception as e:
        return None, f"Backend upload error: {e}"


# ---------------------------------------------------------------------------
# Shared: compute end position display
# ---------------------------------------------------------------------------

def _compute_end(start_val) -> str:
    try:
        start = int(str(start_val).strip().replace(",", ""))
        if start < 0:
            return "—"
        return f"{start + TARGET_LEN:,}"
    except (ValueError, TypeError):
        return "—"


# ---------------------------------------------------------------------------
# FASTA mode callbacks
# ---------------------------------------------------------------------------

def _fasta_update_end(start_val):
    return _compute_end(start_val)


def _run_fasta_prediction(fasta_file, chrom_raw: str, start_val):
    """Called when user clicks Predict (FASTA mode)."""

    # -- Validate inputs --
    if fasta_file is None:
        return (
            gr.update(value="<span style='color:#a33;'>❌ Please upload a FASTA file.</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    chrom = chrom_raw.strip() if chrom_raw else ""
    if not chrom:
        return (
            gr.update(value="<span style='color:#a33;'>❌ Please enter a chromosome number.</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    try:
        start = int(str(start_val).strip().replace(",", ""))
    except (ValueError, TypeError):
        return (
            gr.update(value="<span style='color:#a33;'>❌ Invalid start position.</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    # -- Materialise upload to stable path --
    stable_path, mat_err = _materialize_upload(fasta_file)

    print("stable_path ", stable_path)
    print("mat_err ", mat_err)
    if mat_err:
        return (
            gr.update(value=f"<span style='color:#a33;'>⏳ {mat_err}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    # -- Push to backend cache --
    backend_path, push_err = _push_file_to_backend(stable_path)
    print("backend_path ", backend_path)
    print("push_err ", push_err)
    if push_err:
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ {push_err}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    # -- Call backend predict/fasta --
    req_body = {
        "genome": DEFAULT_GENOME,
        "fasta_path": backend_path,
        "chrom": chrom,
        "start": start,
    }
    try:
        api_req = request.Request(
            url=f"{BACKEND_API_URL}/predict/fasta",
            data=json.dumps(req_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(api_req, timeout=1800) as resp:
            api_result = json.loads(resp.read().decode("utf-8"))
    except error.URLError as e:
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ Backend not reachable: {e}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )
    except Exception as e:
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ {e}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    if not api_result.get("ok"):
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ {api_result.get('error', 'Unknown error')}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    # --- 核心修复逻辑：解析嵌套的 payload JSON 字符串 ---
    try:
        # api_result["payload"] 是一个字符串，需要先转为字典
        payload_dict = json.loads(api_result["payload"])
        # 提取 plus 和 minus 轨迹数据，并转回字符串供 Gradio 渲染
        payload_plus = json.dumps(payload_dict["plus"])
        payload_minus = json.dumps(payload_dict["minus"])
    except (KeyError, json.JSONDecodeError, TypeError) as e:
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ Frontend Parse Error: {e}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    return (
        gr.update(value=api_result["message"], visible=True),
        gr.update(visible=True),   # plus IGV panel
        gr.update(visible=True),   # minus IGV panel
        payload_plus,
        payload_minus,
    )


# ---------------------------------------------------------------------------
# VCF mode callbacks
# ---------------------------------------------------------------------------

def _vcf_update_end(start_val):
    return _compute_end(start_val)


def _run_vcf_prediction(vcf_file, chrom_raw: str, start_val):
    """Called when user clicks Predict (VCF mode)."""

    print("vcf_file ", vcf_file)
    if vcf_file is None:
        return (
            gr.update(value="<span style='color:#a33;'>❌ Please upload a VCF file.</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    chrom = chrom_raw.strip() if chrom_raw else ""

    print("chrom ", chrom)
    if not chrom:
        return (
            gr.update(value="<span style='color:#a33;'>❌ Please enter a chromosome number.</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    try:
        start = int(str(start_val).strip().replace(",", ""))
        print("start ", start)
    except (ValueError, TypeError):
        return (
            gr.update(value="<span style='color:#a33;'>❌ Invalid start position.</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    stable_path, mat_err = _materialize_upload(vcf_file)
    print("stable_path ", stable_path)
    print("mat_err ", mat_err)
    if mat_err:
        return (
            gr.update(value=f"<span style='color:#a33;'>⏳ {mat_err}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    backend_path, push_err = _push_file_to_backend(stable_path)
    print("backend_path ", backend_path)
    print("push_err ", push_err)
    if push_err:
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ {push_err}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    req_body = {
        "genome": DEFAULT_GENOME,
        "vcf_path": backend_path,
        "chrom": chrom,
        "start": start,
    }
    try:
        api_req = request.Request(
            url=f"{BACKEND_API_URL}/predict/vcf",
            data=json.dumps(req_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(api_req, timeout=1800) as resp:
            api_result = json.loads(resp.read().decode("utf-8"))
        

    except error.URLError as e:
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ Backend not reachable: {e}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )
    except Exception as e:
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ {e}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    if not api_result.get("ok"):
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ {api_result.get('error', 'Unknown error')}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    
    # Append SNP warning to message if present
    message = api_result["message"]

    # --- 核心修复逻辑：解析嵌套的 payload JSON 字符串 ---
    try:
        payload_dict = json.loads(api_result["payload"])
        payload_plus = json.dumps(payload_dict["plus"])
        payload_minus = json.dumps(payload_dict["minus"])
    except (KeyError, json.JSONDecodeError, TypeError) as e:
        return (
            gr.update(value=f"<span style='color:#a33;'>❌ Frontend Parse Error: {e}</span>", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "", "",
        )

    snp_info = api_result.get("snp_info") or {}
    if snp_info.get("warning"):
        message += f"<br><span style='color:#e07b00;'>⚠️ {snp_info['warning']}</span>"

    return (
        gr.update(value=message, visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        # api_result["payload_plus"],
        # api_result["payload_minus"],
        payload_plus,
        payload_minus,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_interface():
    with gr.Blocks(
        title=APP_TITLE,
        css="footer {visibility: hidden}",
        head=HEAD_JS,
    ) as demo:

        gr.Markdown(APP_HEADER_MARKDOWN)

        # Shared hidden state for IGV payloads
        payload_plus_tb  = gr.Textbox(value="", visible=False)
        payload_minus_tb = gr.Textbox(value="", visible=False)

        # Shared status bar
        pred_status_md = gr.Markdown(visible=False)

        # ── Input panels (left = FASTA, right = VCF) ───────────────────────
        with gr.Row(equal_height=False):

            # ── FASTA panel ─────────────────────────────────────────────────
            with gr.Column():
                gr.Markdown("## 📄 FASTA Mode")
                gr.Markdown(
                    "Upload your personal genome FASTA file. "
                    "The model will extract the 32 k sequence starting at the position you specify."
                )
                fasta_file = gr.File(
                    label="Upload FASTA (.fa / .fasta)",
                    file_count="single",
                    file_types=[".fa", ".fasta"],
                    type="filepath",
                )
                with gr.Row():
                    fasta_chrom = gr.Textbox(
                        label="Chromosome",
                        placeholder="e.g. 19",
                        scale=1,
                    )
                    fasta_start = gr.Number(
                        label="Start",
                        precision=0,
                        scale=1,
                    )
                    fasta_end = gr.Textbox(
                        label="End (start + 32 k)",
                        value="—",
                        interactive=False,
                        scale=1,
                    )
                btn_fasta = gr.Button("Predict (FASTA)", variant="primary")

            # ── VCF panel ────────────────────────────────────────────────────
            with gr.Column():
                gr.Markdown("## 🧬 VCF Mode")
                gr.Markdown(
                    "Upload a VCF file. SNPs in the specified region will be applied to the "
                    "hg38 reference sequence before prediction. Indels are automatically removed."
                )
                vcf_file = gr.File(
                    label="Upload VCF (.vcf / .vcf.gz)",
                    file_count="single",
                    file_types=[".vcf", ".gz"],
                    type="filepath",
                )
                with gr.Row():
                    vcf_chrom = gr.Textbox(
                        label="Chromosome",
                        placeholder="e.g. 19",
                        scale=1,
                    )
                    vcf_start = gr.Number(
                        label="Start",
                        precision=0,
                        scale=1,
                    )
                    vcf_end = gr.Textbox(
                        label="End (start + 32 k)",
                        value="—",
                        interactive=False,
                        scale=1,
                    )
                btn_vcf = gr.Button("Predict (VCF)", variant="primary")

        # ── Status bar ─────────────────────────────────────────────────────
        pred_status_md = gr.Markdown(visible=False)

        # ── IGV panels (shared, shown after first prediction) ──────────────
        igv_plus_panel = gr.HTML(
            f"""
            <div style="margin-top:12px;">
                <div style="font-weight:500; margin-bottom:4px; color:var(--body-text-color);">
                    RNA-seq Prediction — Plus strand (+)
                </div>
                <div style="background:white; border-radius:8px; overflow:hidden;
                            border:1px solid #ddd;">
                    <div id="igv-plus-status"
                         style="padding:10px 16px; color:#666; font-style:italic;
                                border-bottom:1px solid #eee; display:none;">
                        Rendering plus strand...
                    </div>
                    <div id="igv-plus-container"
                         style="width:100%; height:{HEIGHT_STRAND_DEFAULT_CSS};"></div>
                </div>
            </div>
            """,
            visible=False,
        )

        igv_minus_panel = gr.HTML(
            f"""
            <div style="margin-top:12px;">
                <div style="font-weight:500; margin-bottom:4px; color:var(--body-text-color);">
                    RNA-seq Prediction — Minus strand (−)
                </div>
                <div style="background:white; border-radius:8px; overflow:hidden;
                            border:1px solid #ddd;">
                    <div id="igv-minus-status"
                         style="padding:10px 16px; color:#666; font-style:italic;
                                border-bottom:1px solid #eee; display:none;">
                        Rendering minus strand...
                    </div>
                    <div id="igv-minus-container"
                         style="width:100%; height:{HEIGHT_STRAND_DEFAULT_CSS};"></div>
                </div>
            </div>
            """,
            visible=False,
        )

        # ── Event wiring ────────────────────────────────────────────────────

        # Auto-fill end when start changes (FASTA)
        fasta_start.change(
            fn=_fasta_update_end,
            inputs=[fasta_start],
            outputs=[fasta_end],
            queue=False,
        )

        # Auto-fill end when start changes (VCF)
        vcf_start.change(
            fn=_vcf_update_end,
            inputs=[vcf_start],
            outputs=[vcf_end],
            queue=False,
        )

        # FASTA predict button
        fasta_evt = btn_fasta.click(
            fn=_run_fasta_prediction,
            inputs=[fasta_file, fasta_chrom, fasta_start],
            outputs=[pred_status_md, igv_plus_panel, igv_minus_panel,
                     payload_plus_tb, payload_minus_tb],
            queue=True,
        )
        # After Python callback, trigger JS to render both IGV instances
        fasta_evt.then(
            fn=None,
            inputs=[payload_plus_tb, payload_minus_tb],
            outputs=None,
            js="(plus, minus) => { window.loadPlusResult(plus); window.loadMinusResult(minus); }",
        )

        # VCF predict button
        vcf_evt = btn_vcf.click(
            fn=_run_vcf_prediction,
            inputs=[vcf_file, vcf_chrom, vcf_start],
            outputs=[pred_status_md, igv_plus_panel, igv_minus_panel,
                     payload_plus_tb, payload_minus_tb],
            queue=True,
        )
        vcf_evt.then(
            fn=None,
            inputs=[payload_plus_tb, payload_minus_tb],
            outputs=None,
            js="(plus, minus) => { window.loadPlusResult(plus); window.loadMinusResult(minus); }",
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _reset_upload_caches()
    demo = create_interface()
    demo.queue(default_concurrency_limit=QUEUE_CONCURRENCY_LIMIT)

    # Collect all external directories that Gradio's file server must expose
    allowed = [DATA_DIR_ABS]
    for abs_path in [HG38_FASTA_PATH, LOCAL_FASTA_REL, LOCAL_GTF_REL]:
        d = os.path.dirname(abs_path) if os.path.isabs(abs_path) else None
        if d and os.path.isdir(d) and d not in allowed:
            allowed.append(d)

    frontend_api = FastAPI(title="Genos-Personal-Server Frontend")
    frontend_api = gr.mount_gradio_app(
        frontend_api, demo, path="/", allowed_paths=allowed
    )
    uvicorn.run(frontend_api, host=FRONTEND_HOST, port=FRONTEND_PORT)