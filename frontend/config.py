import json
import os


# =============================================================================
# Environment variable helpers  (identical to Genos-Reg)
# =============================================================================

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


# =============================================================================
# App-level settings
# =============================================================================

APP_TITLE = os.getenv("APP_TITLE", "Genos-Personal-Server")
APP_HEADER_MARKDOWN = (
    "# 🧬 Genos-Personal-Server: Personal Genome RNA Expression Prediction"
)

DEFAULT_GENOME = os.getenv("DEFAULT_GENOME", "hg38")

# Default locus shown in the IGV panel before any prediction
DEFAULT_LOCUS = _env_str("DEFAULT_LOCUS", "chr19:0-32768")

# Concurrency
QUEUE_CONCURRENCY_LIMIT = _env_int("QUEUE_CONCURRENCY_LIMIT", 1)
SERVER_PORT = _env_int("SERVER_PORT", 8010)

# IGV downsampling — keep consistent with Genos-Reg
PREDICTION_MAX_POINTS = _env_int("PREDICTION_MAX_POINTS", 900)


# =============================================================================
# Model & inference settings
# =============================================================================

# Base model (Genos-1.2B)
BASE_MODEL_PATH = _env_str("BASE_MODEL_PATH", "")
TOKENIZER_DIR = _env_str(
    "TOKENIZER_DIR",
    "/mnt/genos100-new/peixunban/qixianzhi/pretrained_model/Genos-1.2B",
)
CHECKPOINT_PATH = _env_str("CHECKPOINT_PATH", "")

# Model structure hyperparameters — must match the trained checkpoint
PROJ_DIM = _env_int("PROJ_DIM", 1024)
NUM_DOWNSAMPLES = _env_int("NUM_DOWNSAMPLES", 4)
BOTTLENECK_DIM = _env_int("BOTTLENECK_DIM", 1536)

# Inference window — must equal the window the model was trained with
TARGET_LEN = _env_int("TARGET_LEN", 32768)

# Deploy track means (used for prediction unscaling)
# If not set in .env the values baked into predict_user_region_online.py are used
DEPLOY_PLUS_MEAN = _env_float("DEPLOY_PLUS_MEAN", 0.4135844447852688)
DEPLOY_MINUS_MEAN = _env_float("DEPLOY_MINUS_MEAN", 0.4635942376368138)


# =============================================================================
# Data / reference paths
# =============================================================================

# index_stat.json and bigWig_labels_meta.csv produced during training
INDEX_STAT_JSON = _env_str(
    "INDEX_STAT_JSON",
    "/mnt/genos100-new/peixunban/lijiongzhen/Genos-Personal-Server/script/deploy_test/input/index_stat.json",
)
BIGWIG_LABELS_META_CSV = _env_str(
    "BIGWIG_LABELS_META_CSV",
    "/mnt/genos100-new/peixunban/lijiongzhen/Genos-Personal-Server/script/deploy_test/input/bigWig_labels_meta.csv",
)

# hg38 reference FASTA used by VCF mode (absolute path)
HG38_FASTA_PATH = _env_str(
    "HG38_FASTA_PATH",
    "/mnt/zzbnew/peixunban/lijunyou/GM12878_batch/hg38_cleaned.fa",
)

# Paths used by IGV.js for sequence rendering (may be the same file as
# HG38_FASTA_PATH or a separately placed copy; supports absolute paths)
LOCAL_FASTA_REL = _env_str(
    "LOCAL_FASTA_REL",
    "/mnt/zzbnew/peixunban/lijunyou/GM12878_batch/hg38_cleaned.fa",
)
LOCAL_FASTA_INDEX_REL = _env_str(
    "LOCAL_FASTA_INDEX_REL",
    "/mnt/zzbnew/peixunban/lijunyou/GM12878_batch/hg38_cleaned.fa.fai",
)

# GTF gene annotation (absolute path supported)
LOCAL_GTF_REL = _env_str(
    "LOCAL_GTF_REL",
    "./reference/gencode.v49.annotation.gtf.bgz",
)


# =============================================================================
# IGV container adaptive-height constants  (same as Genos-Reg)
# =============================================================================

HEIGHT_MIN = 260
HEIGHT_MAX = 700
HEIGHT_RESERVE_BOTTOM = 28
HEIGHT_FALLBACK = 400
HEIGHT_HEADER_PADDING = 150
HEIGHT_DEFAULT_CSS = "clamp(260px, 45vh, 700px)"

# Each strand panel is shorter than the Genos-Reg panel because there is only
# one signal track per IGV instance (no ATAC heatmap rows).
HEIGHT_STRAND_DEFAULT_CSS = "clamp(220px, 35vh, 500px)"


# =============================================================================
# Model description (shown in the Evaluation / About tab)
# =============================================================================

MODEL_DESCRIPTION = """
**Genos-Personal** 是一个基于个人基因组序列的 RNA 表达预测模型。
输入一段 32k DNA 序列（来自用户自身基因组 FASTA 或经 VCF 变异替换后的 hg38 参考序列），
模型输出**单碱基分辨率**的正链（+）和负链（−）RNA 表达信号轨迹。

模型以 **Genos-1.2B** 为骨干，通过 U-Net + Transformer Bottleneck 结构对序列特征进行提取，
直接从 DNA 序列预测转录组信号，无需额外的表观基因组输入。

**两种输入模式：**
- **FASTA 模式**：上传个人基因组 FASTA 文件，指定染色体和起始位置，模型提取对应 32k 序列进行预测。
- **VCF 模式**：上传 VCF 变异文件，模型将 SNP 变异替换到 hg38 参考序列中，预测变异后的 RNA 表达轨迹。
"""


# =============================================================================
# JavaScript injected into the Gradio <head>
# Manages two independent IGV instances:
#   window._igvPlusBrowser  — top panel, (+) strand
#   window._igvMinusBrowser — bottom panel, (-) strand
# =============================================================================

def build_head_js(
    default_pred_reference_json: str,
    igv_local_url: str = "",
) -> str:
    return f"""
<script src="https://cdn.jsdelivr.net/npm/igv@3.7.0/dist/igv.min.js"></script>
<script>
window._igvPlusBrowser  = null;
window._igvMinusBrowser = null;
window._igvHeightBound  = false;
window._igvHeightTimer  = null;
window._igvLoadPromise  = null;
window._defaultPredReference = {default_pred_reference_json};

// ── IGV script loader (local vendor → jsdelivr → unpkg fallback) ──────────
function ensureIgvLoaded() {{
    if (typeof igv !== 'undefined') return Promise.resolve(true);
    if (window._igvLoadPromise) return window._igvLoadPromise;

    const scriptUrls = [
        {json.dumps(igv_local_url)},
        'https://cdn.jsdelivr.net/npm/igv@3.7.0/dist/igv.min.js',
        'https://unpkg.com/igv@3.7.0/dist/igv.min.js'
    ].filter(Boolean);

    window._igvLoadPromise = new Promise((resolve, reject) => {{
        let idx = 0;
        const tryNext = () => {{
            if (idx >= scriptUrls.length) {{
                reject(new Error('Unable to load igv.min.js'));
                return;
            }}
            const script = document.createElement('script');
            script.src = scriptUrls[idx++];
            script.async = true;
            script.onload = () => resolve(true);
            script.onerror = () => tryNext();
            document.head.appendChild(script);
        }};
        tryNext();
    }});
    return window._igvLoadPromise;
}}

// ── Adaptive height helpers ───────────────────────────────────────────────
function getIgvContentHeight(browser, fallback) {{
    if (browser && Array.isArray(browser.trackViews)) {{
        let h = 0;
        for (const tv of browser.trackViews) {{
            const domH = tv?.trackDiv?.getBoundingClientRect?.().height;
            h += domH || tv?.track?.height || tv?.track?.config?.height || 50;
        }}
        return Math.ceil(h + {HEIGHT_HEADER_PADDING});
    }}
    return fallback;
}}

function computeIgvHeight(container, browser, fallback) {{
    const minH = {HEIGHT_MIN};
    const maxH = {HEIGHT_MAX};
    const reserve = {HEIGHT_RESERVE_BOTTOM};
    const top = container.getBoundingClientRect().top;
    const byViewport = Math.max(minH, Math.floor(window.innerHeight - top - reserve));
    const byContent  = getIgvContentHeight(browser, fallback);
    return Math.max(minH, Math.min(byContent, maxH, byViewport));
}}

function applyHeights() {{
    const cPlus  = document.getElementById('igv-plus-container');
    const cMinus = document.getElementById('igv-minus-container');
    if (cPlus)  cPlus.style.height  = computeIgvHeight(cPlus,  window._igvPlusBrowser,  {HEIGHT_FALLBACK}) + 'px';
    if (cMinus) cMinus.style.height = computeIgvHeight(cMinus, window._igvMinusBrowser, {HEIGHT_FALLBACK}) + 'px';
}}

function scheduleRelayout(delay) {{
    if (window._igvHeightTimer) clearTimeout(window._igvHeightTimer);
    window._igvHeightTimer = setTimeout(() => {{
        applyHeights();
        if (window._igvPlusBrowser)  window._igvPlusBrowser.resize();
        if (window._igvMinusBrowser) window._igvMinusBrowser.resize();
    }}, delay || 60);
}}

function bindAdaptiveHeight() {{
    if (window._igvHeightBound) return;
    window._igvHeightBound = true;
    window.addEventListener('resize', () => scheduleRelayout(40));
    window.addEventListener('orientationchange', () => scheduleRelayout(40));
    const obs = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => scheduleRelayout(30)) : null;
    ['igv-plus-container', 'igv-minus-container'].forEach(id => {{
        const el = document.getElementById(id);
        if (el && obs) obs.observe(el);
    }});
}}

// ── Status helpers ────────────────────────────────────────────────────────
function setStatus(id, msg) {{
    const el = document.getElementById(id);
    if (!el) return;
    el.style.display = msg ? 'block' : 'none';
    el.innerHTML = msg || '';
}}

// ── Load a single strand result into its IGV container ───────────────────
async function _loadStrandResult(payloadJson, containerId, statusId, browserKey) {{
    if (!payloadJson) return;

    const container = document.getElementById(containerId);
    if (!container) return;

    if (typeof igv === 'undefined') {{
        setStatus(statusId, 'Loading IGV library...');
        try {{ await ensureIgvLoaded(); }} catch(e) {{
            setStatus(statusId, 'IGV load failed: ' + (e?.message || e));
            return;
        }}
    }}

    let cfg;
    try {{ cfg = JSON.parse(payloadJson); }}
    catch (e) {{ setStatus(statusId, 'Invalid payload JSON'); return; }}

    setStatus(statusId, 'Rendering tracks...');

    const options = {{
        locus: cfg.locus,
        showDefaultTracks: false,
        tracks: cfg.tracks || []
    }};
    if (cfg.reference) {{
        options.reference = cfg.reference;
    }} else if (window._defaultPredReference) {{
        options.reference = window._defaultPredReference;
    }} else {{
        options.genome = cfg.genome || 'hg38';
    }}

    const existing = window[browserKey];
    if (existing) {{
        const old = existing.trackViews.map(tv => tv.track).filter(t => t && t.type !== 'ruler');
        old.forEach(t => existing.removeTrack(t));
        existing.loadTrackList(cfg.tracks || [])
            .then(() => existing.search(cfg.locus))
            .then(() => {{ setStatus(statusId, ''); scheduleRelayout(60); }})
            .catch(e => setStatus(statusId, 'Render failed: ' + e));
        return;
    }}

    bindAdaptiveHeight();
    applyHeights();

    requestAnimationFrame(() => requestAnimationFrame(() => {{
        igv.createBrowser(container, options)
            .then(browser => {{
                window[browserKey] = browser;
                setStatus(statusId, '');
                scheduleRelayout(80);
                setTimeout(() => scheduleRelayout(20), 320);
                setTimeout(() => scheduleRelayout(20), 700);
            }})
            .catch(e => setStatus(statusId, 'Render failed: ' + e));
    }}));
}}

// ── Public API called from Gradio .then(js=...) ───────────────────────────
window.loadPlusResult = function(payloadJson) {{
    _loadStrandResult(payloadJson, 'igv-plus-container', 'igv-plus-status', '_igvPlusBrowser');
}};

window.loadMinusResult = function(payloadJson) {{
    _loadStrandResult(payloadJson, 'igv-minus-container', 'igv-minus-status', '_igvMinusBrowser');
}};

// Convenience: load both strands at once from a two-element array [plusJson, minusJson]
window.loadBothStrands = function(plusJson, minusJson) {{
    window.loadPlusResult(plusJson);
    window.loadMinusResult(minusJson);
}};
</script>
"""