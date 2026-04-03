import copy
import json
import os
from typing import List, Dict, Optional
from urllib.parse import quote


# =============================================================================
# URL helpers
# =============================================================================

def to_gradio_file_url(file_name: str, data_dir_abs: str) -> str:
    if os.path.isabs(file_name):
        abs_path = file_name
        return "/gradio_api/file=" + quote(abs_path)
    normalized = file_name[2:] if file_name.startswith("./") else file_name
    abs_path = os.path.join(data_dir_abs, normalized)
    return "/gradio_api/file=" + quote(abs_path)


def resolve_reference_urls(reference_cfg: dict, data_dir_abs: str) -> dict:
    resolved = copy.deepcopy(reference_cfg)
    for key in ("fastaURL", "indexURL", "cytobandURL"):
        value = resolved.get(key)
        if isinstance(value, str) and value and not value.startswith(
            ("http://", "https://", "/gradio_api/file=")
        ):
            resolved[key] = to_gradio_file_url(value, data_dir_abs)
    return resolved


def resolve_track_urls(track_list: list, data_dir_abs: str) -> list:
    resolved = copy.deepcopy(track_list)
    url_like_fields = ("url", "indexURL", "fastaURL", "cytobandURL")
    for track in resolved:
        for field in url_like_fields:
            value = track.get(field)
            if isinstance(value, str) and value and not value.startswith(
                ("http://", "https://", "/gradio_api/file=")
            ):
                track[field] = to_gradio_file_url(value, data_dir_abs)
        if "tracks" in track and isinstance(track["tracks"], list):
            track["tracks"] = resolve_track_urls(track["tracks"], data_dir_abs)
    return resolved


# =============================================================================
# GTF annotation track
# =============================================================================

def build_reference_annotation_track(
    data_dir_abs: str,
    local_gtf_rel: str,
) -> Optional[dict]:
    """Build a Ref-seq GTF annotation track. Supports absolute paths."""
    if os.path.isabs(local_gtf_rel):
        selected_abs = local_gtf_rel
    else:
        normalized = local_gtf_rel[2:] if local_gtf_rel.startswith("./") else local_gtf_rel
        selected_abs = os.path.join(data_dir_abs, normalized)

    if not os.path.exists(selected_abs):
        return None

    track = {
        "name": "Ref-seq",
        "type": "annotation",
        "format": "gtf",
        "url": to_gradio_file_url(selected_abs, data_dir_abs),
        "displayMode": "EXPANDED",
        "height": 120,
    }
    tbi_path = selected_abs + ".tbi"
    if os.path.exists(tbi_path):
        track["indexURL"] = to_gradio_file_url(tbi_path, data_dir_abs)
    return track


def _with_annotation_track(
    track_list: list,
    data_dir_abs: str,
    local_gtf_rel: str,
) -> list:
    """Append GTF annotation track if not already present."""
    tracks = list(track_list or [])
    has_annotation = any(
        isinstance(t, dict) and (
            t.get("type") == "annotation"
            or str(t.get("name", "")).lower() in {"ref-seq", "refseq"}
        )
        for t in tracks
    )
    if has_annotation:
        return tracks
    ann = build_reference_annotation_track(data_dir_abs, local_gtf_rel)
    if ann is not None:
        tracks.append(ann)
    return tracks


# =============================================================================
# Default prediction reference  (hg38 local)
# =============================================================================

def build_default_prediction_reference(
    local_fasta_rel: str,
    local_fasta_index_rel: str,
    data_dir_abs: str,
) -> dict:
    local_ref = {
        "id": "hg38-local",
        "name": "hg38 local",
        "fastaURL": local_fasta_rel,
        "indexURL": local_fasta_index_rel,
    }
    return resolve_reference_urls(local_ref, data_dir_abs)


# =============================================================================
# Single-strand payload builder
# =============================================================================

def build_strand_payload(
    genome: str,
    chrom: str,
    user_start: int,
    user_end: int,
    features: List[Dict],
    strand: str,
    track_name: str,
    track_color: str,
    data_dir_abs: str,
    local_gtf_rel: str,
    default_pred_reference: Optional[dict] = None,
) -> dict:
    """
    Build an IGV payload for a single strand (either + or -).

    Track layout:
      1. Predicted RNA-seq (strand)  — line, inline bedgraph features
      2. Ref-seq GTF annotation
    """
    strand_label = "+" if strand == "+" else "-"

    signal_track = {
        "type": "wig",
        "format": "bedgraph",
        "name": f"{track_name} ({strand_label})",
        "features": features,
        "color": track_color,
        "height": 100,
        "autoscale": True,
        "showDataRange": True,
        "showAxis": True,
        "graphType": "line",
        "fill": False,
    }

    tracks = [signal_track]
    tracks = _with_annotation_track(tracks, data_dir_abs, local_gtf_rel)

    payload = {
        "genome": genome,
        "locus": f"{chrom}:{user_start}-{user_end}",
        "tracks": tracks,
    }

    if default_pred_reference is not None:
        payload["reference"] = default_pred_reference

    return payload


# =============================================================================
# Convenience: build both strand payloads at once
# =============================================================================

def build_prediction_payloads(
    genome: str,
    chrom: str,
    user_start: int,
    user_end: int,
    plus_features: List[Dict],
    minus_features: List[Dict],
    data_dir_abs: str,
    local_gtf_rel: str,
    track_name: str = "Predicted RNA-seq",
    track_color_plus: str = "#d62728",
    track_color_minus: str = "#1f77b4",
    default_pred_reference: Optional[dict] = None,
) -> Dict[str, dict]:
    """
    Return {"plus": payload_dict, "minus": payload_dict}.

    The two payloads are independent — each renders in its own IGV instance:
      - Plus  IGV (top):    (+) signal track + GTF
      - Minus IGV (bottom): (-) signal track + GTF
    """
    payload_plus = build_strand_payload(
        genome=genome,
        chrom=chrom,
        user_start=user_start,
        user_end=user_end,
        features=plus_features,
        strand="+",
        track_name=track_name,
        track_color=track_color_plus,
        data_dir_abs=data_dir_abs,
        local_gtf_rel=local_gtf_rel,
        default_pred_reference=default_pred_reference,
    )

    payload_minus = build_strand_payload(
        genome=genome,
        chrom=chrom,
        user_start=user_start,
        user_end=user_end,
        features=minus_features,
        strand="-",
        track_name=track_name,
        track_color=track_color_minus,
        data_dir_abs=data_dir_abs,
        local_gtf_rel=local_gtf_rel,
        default_pred_reference=default_pred_reference,
    )

    return {"plus": payload_plus, "minus": payload_minus}


# =============================================================================
# JSON serialisation helper
# =============================================================================

def dumps_payload_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)