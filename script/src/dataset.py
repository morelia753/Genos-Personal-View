# (替换你工程里的 src/dataset.py 为本文件内容)

import os
import json
import torch
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import logging
from torch.utils.data import Dataset

from src.utils.data import load_bigwig_signal, load_fasta_sequence
from src.utils.dist import dist_print


class MultiTrackDataset(Dataset):
    def __init__(
        self,
        sequence_split_df,
        labels_meta_df,
        index_stat,
        tokenizer,
        max_length=32768,
        augment=False,
        annotation_file=None,
        fail_on_bw_open_error=True,
        load_labels=True,
    ):
        self.sequence_split_df = sequence_split_df.reset_index(drop=True)
        self.labels_meta_df = labels_meta_df.reset_index(drop=True)
        self.index_stat = index_stat
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

        # NEW: annotation + strict bw open behavior
        self.annotation_file = annotation_file
        self.fail_on_bw_open_error = bool(fail_on_bw_open_error)
        self.load_labels = load_labels

        dist_print(f"[MultiTrackDataset] max_length={max_length}, augment={augment}")

        # Pre-load annotation groups (for strand masks)
        # 推理(load_labels=False)不需要strand mask，禁止无意义解析GTF（否则会严重卡住/或直接报错）
        if self.load_labels:
            self.annotations_by_chrom = self._load_and_group_annotations()
        else:
            self.annotations_by_chrom = None


        # -------------------------
        # Caches (per worker process)
        # -------------------------
        # - FASTA handle cache is keyed by fasta_path (for multi-sample training)
        # - BigWig handle cache is keyed by full bw_path
        self._fasta_handles = {}         # fasta_path -> pyfaidx.Fasta
        self._chrom_sizes_cache = {}     # fasta_path -> {chrom: length}

        inputs = self.index_stat.get("inputs", {})
        if not isinstance(inputs.get("genome_fasta"), (list, tuple)):
            raise RuntimeError(
                "[MultiTrackDataset] This project only supports multi-sample index_stat.\n"
                "For reference genome, wrap it as N=1 multi-sample:\n"
                "  inputs.sample_id = ['REF']\n"
                "  inputs.genome_fasta = ['/path/ref.fa']\n"
                "  inputs.processed_rnaseq_bw_dir = [{'plus': '/path/plus.bw', 'minus': '/path/minus.bw', "
                "'plus_mean': xxx, 'minus_mean': yyy}]\n"
            )
 
        # Track order is ALWAYS defined by labels_meta_df row order
        # For personal mode, we infer which per-sample track to load (plus/minus) from the meta rows.

        self.nonzero_means = self.labels_meta_df["nonzero_mean"].tolist()

        # Expect: inputs.sample_id, inputs.genome_fasta, inputs.processed_rnaseq_bw_dir (list of dicts)
        sids = list(inputs.get("sample_id", []))
        fastas = list(inputs.get("genome_fasta", []))
        bw_list = list(inputs.get("processed_rnaseq_bw_dir", []))

        if not sids or not fastas or not bw_list:
            raise ValueError("index_stat['inputs'] is missing sample_id/genome_fasta/processed_rnaseq_bw_dir")

        if not (len(sids) == len(fastas) == len(bw_list)):
            raise ValueError(
                f"Length mismatch in index_stat['inputs']: "
                f"len(sample_id)={len(sids)}, len(genome_fasta)={len(fastas)}, len(processed_rnaseq_bw_dir)={len(bw_list)}"
            )

        self._sid2fasta = {str(sid): str(fp) for sid, fp in zip(sids, fastas)}
        self._sid2bw = {}
        self._sid2track_means = {}

        # 严格check processed_rnaseq_bw_dir entries
        for sid, rec in zip(sids, bw_list):
            sid = str(sid)
            if not isinstance(rec, dict):
                raise ValueError(f"processed_rnaseq_bw_dir entry for {sid} must be dict, got: {type(rec)}")

            # bw 路径必须存在 key
            if "plus" not in rec or "minus" not in rec:
                raise ValueError(f"processed_rnaseq_bw_dir entry for {sid} missing 'plus'/'minus' keys: {rec.keys()}")
            self._sid2bw[sid] = {"plus": str(rec["plus"]), "minus": str(rec["minus"])}

            # === STRICT: per-sample track mean 必须在 index_stat.json 里显式给出，缺失直接 fatal ===
            if "plus_mean" not in rec or "minus_mean" not in rec:
                raise RuntimeError(
                    f"[FATAL][track_means] missing plus_mean/minus_mean for sample_id={sid}. "
                    f"Keys={list(rec.keys())}. Global fallback is forbidden."
                )
            try:
                plus_mean = float(rec["plus_mean"])
                minus_mean = float(rec["minus_mean"])
            except Exception as e:
                raise RuntimeError(
                    f"[FATAL][track_means] plus_mean/minus_mean must be numeric for sample_id={sid}. "
                    f"Got plus_mean={rec.get('plus_mean')} minus_mean={rec.get('minus_mean')}"
                ) from e

            if not np.isfinite(plus_mean) or not np.isfinite(minus_mean):
                raise RuntimeError(
                    f"[FATAL][track_means] plus_mean/minus_mean is not finite for sample_id={sid}. "
                    f"plus_mean={plus_mean}, minus_mean={minus_mean}"
                )

            self._sid2track_means[sid] = {"plus": plus_mean, "minus": minus_mean}


        # In multi-sample mode, fasta/bw are chosen PER ROW via sample_id, so these are unused.
        self.fasta_path = None
        self.bigwig_rnaseq_dir = None

        # Infer track keys (plus/minus) in the SAME ORDER as labels_meta_df
        self._track_keys = []
        for _, mrow in self.labels_meta_df.iterrows():
            strand = str(mrow.get("strand", "")).strip()
            tname = str(mrow.get("target_file_name", "")).lower()
            if strand == "+" or tname.endswith("plus") or "plus" in tname:
                self._track_keys.append("plus")
            elif strand == "-" or tname.endswith("minus") or "minus" in tname:
                self._track_keys.append("minus")
            else:
                raise ValueError(
                    f"Cannot infer track key (plus/minus) from labels_meta row: "
                    f"strand={strand}, target_file_name={mrow.get('target_file_name')}"
                )

        if len(self._track_keys) != len(self.labels_meta_df):
            raise RuntimeError("Internal error: track key inference length mismatch")

        # Strong sanity for your intended setup: exactly 2 tracks (plus/minus)
        if set(self._track_keys) != {"plus", "minus"} or len(self._track_keys) != 2:
            logging.fatal(
                f"[MultiTrackDataset] multi-sample mode expects 2 tracks (plus/minus). "
                f"Got track_keys={self._track_keys}. Training may still run, but please verify your labels_meta_df."
            )
            raise RuntimeError("[MultiTrackDataset] multi-sample mode expects 2 tracks (plus/minus).")


        # We compute chrom sizes lazily per fasta_path when augmentation needs them.
        self.chrom_sizes = None

        self.print_data_info()

    def print_data_info(self):
        dist_print("[Dataset Info]")
        dist_print(f"FASTA path: {self.fasta_path}")
        dist_print(f"RNA_SEQ BigWig directory: {self.bigwig_rnaseq_dir}")
        dist_print(f"Number of targets: {len(self.labels_meta_df)}")
        dist_print(f"Sequence split rows: {len(self.sequence_split_df)}")


    def _load_and_group_annotations(self):
        """
        Load gene annotation intervals for strand masks.

        Priority:
        1) self.annotation_file (from --annotation_file)
        2) index_stat["inputs"]["gtf_annotations"] (legacy fallback, only if annotation_file not provided)

        Supported formats:
        - Preprocessed TSV with columns: chrom, start, end, strand (strand: 1 for +, -1 for -)
        - GTF/GTF.GZ (GENCODE): we parse 'gene' features into the same schema.
        """
        # 1) prefer CLI-provided annotation_file
        annotations_path = self.annotation_file

        if not annotations_path:
            raise RuntimeError(
                "[ANNOTATION] annotation_file is required for training (no silent all-false masks). "
                "Pass --annotation_file /path/to/gencode.gtf.gz"
            )

        annotations_path = str(annotations_path)
        if not os.path.exists(annotations_path):
            # If user explicitly passed it, this must be fatal
            raise RuntimeError(f"[ANNOTATION] annotation_file not found: {annotations_path}")

        def _gtf_iter_lines(path):
            import gzip
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt") as f:
                for line in f:
                    if not line or line.startswith("#"):
                        continue
                    yield line.rstrip("\n")

        # --- Case A: GTF / GTF.GZ ---
        if annotations_path.endswith(".gtf") or annotations_path.endswith(".gtf.gz"):
            rows = []
            for line in _gtf_iter_lines(annotations_path):
                parts = line.split("\t")
                if len(parts) < 9:
                    continue
                chrom, _, feature, start, end, _, strand, _, _ = parts[:9]
                if feature != "gene":
                    continue
                try:
                    s = int(start) - 1  # GTF is 1-based inclusive; convert to 0-based start
                    e = int(end)        # keep as half-open end
                except ValueError:
                    continue
                if strand == "+":
                    st = 1
                elif strand == "-":
                    st = -1
                else:
                    continue
                rows.append((chrom, s, e, st))

            if not rows:
                raise RuntimeError(f"[ANNOTATION] Parsed 0 gene intervals from GTF: {annotations_path}")

            df = pd.DataFrame(rows, columns=["chrom", "start", "end", "strand"])
        else:
            raise RuntimeError(f"[ANNOTATION] annotation_file style must be gtf or gtf.gz: {annotations_path}")

        grouped = {}
        for chrom, sub in df.groupby("chrom"):
            grouped[str(chrom)] = sub.reset_index(drop=True)
        dist_print(f"[ANNOTATION] Loaded annotations from {annotations_path}, chroms={len(grouped)}")
        return grouped



    def _create_strand_masks(self, chrom, start, end, max_length):
        if self.annotations_by_chrom is None:
            raise RuntimeError("[ANNOTATION] strand masks requested but annotations are not loaded (load_labels=False).")

        if chrom not in self.annotations_by_chrom:
            return np.zeros(max_length, dtype=np.bool_), np.zeros(max_length, dtype=np.bool_)

        ann = self.annotations_by_chrom[chrom]
        # interval overlap
        sub = ann[(ann["end"] > start) & (ann["start"] < end)]
        pos_mask = np.zeros(max_length, dtype=np.bool_)
        neg_mask = np.zeros(max_length, dtype=np.bool_)

        for _, r in sub.iterrows():
            s = max(int(r["start"]), start)
            e = min(int(r["end"]), end)
            if e <= s:
                continue
            rs = s - start
            re = e - start
            if int(r["strand"]) == 1:
                pos_mask[rs:re] = True
            else:
                neg_mask[rs:re] = True

        return pos_mask, neg_mask

    def _get_fasta(self, fasta_path: str):
        """Return cached or newly opened FASTA handle. Multi-sample only."""
        if fasta_path is None:
            raise ValueError("fasta_path must be provided (multi-sample only).")
        fasta_path = str(fasta_path)
        if fasta_path not in self._fasta_handles:
            self._fasta_handles[fasta_path] = pyfaidx.Fasta(fasta_path)
        return self._fasta_handles[fasta_path]


    def __len__(self):
        return len(self.sequence_split_df)

    def __getitem__(self, idx):
        row = self.sequence_split_df.iloc[idx].copy()  # copy to avoid modifying original

        # Resolve sample_id (personal-genome mode)
        if "sample_id" not in row.index:
            raise ValueError("sequence_split_df must contain 'sample_id' column in multi-sample mode")
        sample_id = str(row["sample_id"])

        if sample_id not in self._sid2fasta:
            raise KeyError(f"sample_id={sample_id} not found in index_stat['inputs']['sample_id']")
        fasta_path = self._sid2fasta[sample_id]
        fasta = self._get_fasta(fasta_path)

        # === Step 1: Random shift (ONLY for training; inference no augment) ===
        if self.load_labels and self.augment:
            chrom = str(row["chromosome"])
            orig_start, orig_end = int(row["start"]), int(row["end"])
            window_size = orig_end - orig_start
            try:
                chrom_len = len(fasta[chrom])
            except Exception as e:
                raise KeyError(f"Chromosome {chrom} not found in FASTA ({fasta_path})") from e

            shift = torch.randint(-1024, 1025, ()).item()
            new_center = (orig_start + orig_end) // 2 + shift
            new_start = max(0, min(new_center - window_size // 2, chrom_len - window_size))
            new_end = new_start + window_size
            row["start"], row["end"] = new_start, new_end

        # === Step 2: Load sequence ===
        seq = load_fasta_sequence(fasta, row["chromosome"], row["start"], row["end"], self.max_length)

        # === Step 3: Decide whether to reverse-complement ===
        # IMPORTANT: inference does NOT do RC, force do_rc=False when load_labels=False
        do_rc = (self.load_labels and self.augment and (torch.rand(()).item() < 0.5))

        if do_rc:
            complement_map = str.maketrans("ACGTN", "TGCAN")
            seq = seq.translate(complement_map)[::-1]

        # Tokenize (add_special_tokens=False MUST)
        encodings = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=False
        )
        input_ids = encodings["input_ids"].squeeze(0)

        # === Step 4: per-sample track means (ALWAYS required for your inference; comes from index_stat.json only) ===
        if sample_id not in self._sid2track_means:
            raise RuntimeError(f"[track_means] missing track means for sample_id={sample_id} (should be fatal)")

        means = []
        for tk in self._track_keys:
            # if RC happened (training only), swap plus/minus mean accordingly
            src = ("minus" if tk == "plus" else "plus") if do_rc else tk
            means.append(self._sid2track_means[sample_id][src])
        sample_track_means = torch.tensor(means, dtype=torch.float32)

        # === Inference path: do NOT return labels/masks at all (avoid None + avoid GTF dependence) ===
        if not self.load_labels:
            return {
                "position": (str(row["chromosome"]), int(row["start"]), int(row["end"])),
                "input_ids": input_ids,
                "sample_id": sample_id,
                "sample_track_means": sample_track_means,
            }

        # === Training path: load tracks + masks ===
        track_values = []

        if sample_id not in self._sid2bw:
            raise KeyError(f"sample_id={sample_id} not found in _sid2bw (index_stat inputs)")

        vals_by_key = {}
        for tk in self._track_keys:
            bw_path = str(self._sid2bw[sample_id][tk])

            vals = load_bigwig_signal(
                bw_path,
                str(row["chromosome"]),
                int(row["start"]),
                int(row["end"]),
                max_length=self.max_length,
                pad=True,
            )

            if do_rc:
                vals = np.flip(vals).copy()

            vals_by_key[tk] = vals

        # assemble in channel order
        for tk in self._track_keys:
            src = ("minus" if tk == "plus" else "plus") if do_rc else tk
            track_values.append(vals_by_key[src])

        labels = torch.stack([torch.tensor(tv, dtype=torch.float32) for tv in track_values], dim=-1)

        # strand masks (requires annotations_by_chrom already loaded in __init__)
        pos_mask_np, neg_mask_np = self._create_strand_masks(
            str(row["chromosome"]), int(row["start"]), int(row["end"]), self.max_length
        )
        pos_mask = torch.from_numpy(pos_mask_np).bool()
        neg_mask = torch.from_numpy(neg_mask_np).bool()

        if do_rc:
            pos_mask = torch.flip(pos_mask, dims=[0])
            neg_mask = torch.flip(neg_mask, dims=[0])
            pos_mask, neg_mask = neg_mask, pos_mask

        return {
            "position": (str(row["chromosome"]), int(row["start"]), int(row["end"])),
            "input_ids": input_ids,
            "labels": labels,
            "pos_strand_mask": pos_mask,
            "neg_strand_mask": neg_mask,
            "sample_id": sample_id,
            "sample_track_means": sample_track_means,
        }



    def close(self):
        """Close opened FASTA and BigWig handles and clear caches."""
        # Close FASTA handles
        for _, fa in list(self._fasta_handles.items()):
            try:
                if fa is not None:
                    fa.close()
            except Exception:
                pass
        self._fasta_handles.clear()

        # Clear auxiliary caches
        self._chrom_sizes_cache.clear()
