import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from src.utils.data import load_bigwig_signal
import gzip
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from typing import Dict, List, Optional, Tuple


def gaussian_smooth(signal, sigma):
    """
    Apply simple Gaussian smoothing using a normalized kernel.
    If sigma <= 0 or signal is empty, return original signal.
    """
    if sigma is None:
        return signal
    try:
        sigma = float(sigma)
    except Exception:
        return signal
    if sigma <= 0 or signal is None or len(signal) == 0:
        return signal
    # kernel size = odd integer, cover +/-3 sigma
    kernel_size = max(3, int(6 * sigma) | 1)  # ensure odd by forcing last bit 1
    half = kernel_size // 2
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    try:
        smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed
    except Exception:
        return signal

class DatasetViewer:
    """
    Helper to visualize dataset windows and optional gene annotations.

    Usage:
      viewer = DatasetViewer(dataset, annotation_path="gencode.v48.gff3.gz")
      viewer.plot_window(idx=0, smoothing_sigma=2.0)
    """
    def __init__(self, dataset, annotation_path=None, max_subplots=6,
                 gene_color_plus="tab:blue", gene_color_minus="tab:orange",
                 xtick_step=4000, dpi=100, signal_palette=None):
        self.dataset = dataset
        self.max_subplots = int(max_subplots)
        self.gene_color_plus = gene_color_plus
        self.gene_color_minus = gene_color_minus
        self.xtick_step = xtick_step
        self.dpi = dpi
        self.signal_palette = signal_palette  # if None, will use tab10/tab20 per-plot
        self.genes_by_chrom = {}
        self.exons_by_chrom = {}
        if annotation_path:
            self._load_gff(annotation_path)

    def _load_gff(self, path):
        """Load GFF/GTF (optionally gzipped). Store genes and exons per chromosome."""
        import gzip
        genes = {}
        exons = {}
        try:
            open_func = gzip.open if path.endswith('.gz') else open
            mode = 'rt' if path.endswith('.gz') else 'r'
            with open_func(path, mode) as fh:
                for line in fh:
                    if line.startswith('#') or not line.strip():
                        continue
                    cols = line.rstrip().split('\t')
                    if len(cols) < 9:
                        continue
                    chrom, src, feature, start, end, score, strand, phase, attrs = cols[:9]
                    try:
                        start_i = int(start); end_i = int(end)
                    except Exception:
                        continue
                    # extract a simple name (gene_name / Name / gene_id)
                    gene_name = ""
                    for key in ("gene_name=", "Name=", "gene_id="):
                        if key in attrs:
                            parts = [p for p in attrs.split(';') if p.strip().startswith(key)]
                            if parts:
                                gene_name = parts[0].split('=', 1)[-1].strip('"').strip()
                                break
                    if feature == "gene":
                        genes.setdefault(chrom, []).append((start_i, end_i, strand, gene_name))
                    elif feature == "exon":
                        exons.setdefault(chrom, []).append((start_i, end_i, strand, gene_name))
        except Exception as e:
            logging.warning(f"Failed to load annotation {path}: {e}")
        self.genes_by_chrom = genes
        self.exons_by_chrom = exons
        logging.info(f"Loaded annotation: chromosomes={list(self.genes_by_chrom.keys())}")

    def get_genes_in_interval(self, chrom, start, end):
        """Return (genes, exons) overlapping [start, end)."""
        genes = self.genes_by_chrom.get(chrom, [])
        exons = self.exons_by_chrom.get(chrom, [])
        genes_f = [(gs, ge, st, nm) for (gs, ge, st, nm) in genes if not (ge < start or gs >= end)]
        exons_f = [(es, ee, st, nm) for (es, ee, st, nm) in exons if not (ee < start or es >= end)]
        return genes_f, exons_f

    def plot_window(self, idx=None, max_subplots=None, assembly_filter=None, track_indices=None, smoothing_sigma=2,
                    window_start=None, window_end=None):
        """
        Plot selected tracks for one window and optional gene models.

        Args:
            idx (int): index into dataset.sequence_split_df; defaults to 0.
            max_subplots (int): max number of tracks to draw (default: viewer.max_subplots).
            assembly_filter (str or list): filter labels_meta_df by 'File assembly'.
            track_indices (list[int]): explicit indices into labels_meta_df to plot.
            smoothing_sigma (float): gaussian smoothing sigma (<=0 disables).
        Returns:
            (fig, axes) matplotlib objects or None on error.
        """
        from matplotlib.ticker import FuncFormatter
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        ds = self.dataset
        if idx is None:
            if len(ds.sequence_split_df) == 0:
                logging.warning("sequence_split_df is empty")
                return
            idx = 0
        try:
            row = ds.sequence_split_df.iloc[idx]
        except Exception as e:
            logging.error(f"Invalid idx for sequence_split_df: {e}")
            return

        meta_df = ds.labels_meta_df.copy()
        if assembly_filter is not None and 'File assembly' in meta_df.columns:
            if isinstance(assembly_filter, (list, tuple)):
                meta_df = meta_df[meta_df['File assembly'].isin(assembly_filter)]
            else:
                meta_df = meta_df[meta_df['File assembly'] == assembly_filter]

        if track_indices is not None:
            sel_df = meta_df.iloc[track_indices].reset_index(drop=True)
        else:
            nmax = self.max_subplots if max_subplots is None else int(max_subplots)
            sel_df = meta_df.reset_index(drop=True).iloc[:nmax]

        if sel_df.shape[0] == 0:
            logging.info("No tracks selected for plotting")
            return

        # load signals and titles
        signals = []
        titles = []
        for _, mrow in sel_df.iterrows():
            fname = mrow.get('target_file_name') or mrow.get('target_file') or None
            if fname is None:
                logging.warning("Missing target file name for a track; skipping")
                continue
            bigwig_dir = ds.bigwig_rnaseq_dir if 'rnaseq' in fname.lower() else ds.bigwig_atac_dir
            bw_path = os.path.join(bigwig_dir, fname)
            vals = load_bigwig_signal(bw_path, row["chromosome"], int(row["start"]), int(row["end"]), ds.max_length)
            vals = np.nan_to_num(vals, nan=0.0)
            signals.append(vals)
            output_type = mrow.get('output_type') or ""
            biosample_name = mrow.get('biosample_name') or ""
            assay = mrow.get('name') or ""
            strand = mrow.get('strand') or ""
            if assay or strand:
                titles.append(f"{output_type}: {biosample_name} ({strand})\n{assay}")
            else:
                titles.append(os.path.basename(fname))

        if len(signals) == 0:
            logging.info("No signals loaded for plotting")
            return

        n_tracks = len(signals)
        fig, axes = plt.subplots(n_tracks + 1, 1, figsize=(18, 1.5 * (n_tracks + 1)), sharex=True, dpi=self.dpi,
                                 gridspec_kw={'height_ratios': [1.0] + [1.0] * n_tracks})
        if n_tracks + 1 == 1:
            axes = [axes]
        ax_gene = axes[0]

        chrom = row["chromosome"]
        row_start = int(row["start"])  # original interval start loaded from files
        first_signal = signals[0]
        actual_len = len(first_signal)
        row_end = row_start + actual_len

        # If no window specified, keep previous behavior (show entire loaded interval)
        if window_start is None and window_end is None:
            start_display = row_start
            end_display = row_end
            display_positions = np.arange(start_display, end_display)
            signals_to_plot = signals
        else:
            # Fill missing window_start/window_end with loaded interval bounds if only one provided
            if window_start is None:
                window_start = row_start
            if window_end is None:
                window_end = row_end
            start_display = int(window_start)
            end_display = int(window_end)
            desired_len = end_display - start_display
            if desired_len <= 0:
                logging.error("Invalid window: window_end must be greater than window_start")
                return
            display_positions = np.arange(start_display, end_display)
            displayed_len = len(display_positions)

            # Slice/pad each signal so that the returned segment corresponds exactly to
            # [start_display, end_display). Regions outside the originally loaded interval
            # are filled with zeros.
            signals_to_plot = []
            for sig in signals:
                seg = np.zeros(displayed_len, dtype=sig.dtype)
                # source indices within the original loaded signal
                src_start = max(0, start_display - row_start)
                src_end = min(actual_len, end_display - row_start)
                if src_end > src_start:
                    # destination start index in the segment
                    dest_start = max(0, row_start - start_display)
                    seg[dest_start:dest_start + (src_end - src_start)] = sig[src_start:src_end]
                signals_to_plot.append(seg)

        # draw gene models if available
        genes, exons = self.get_genes_in_interval(chrom, start_display, end_display)
        if genes:
            genes_df = __import__("pandas").DataFrame(genes, columns=["start", "end", "strand", "name"]).sort_values("start").reset_index(drop=True)
            level_ends = []
            level_height_base = 0.3
            level_gap = 0.25
            max_levels = 8
            gene_levels = []
            for _, g in genes_df.iterrows():
                gs, ge = int(g["start"]), int(g["end"])
                placed = False
                for lvl in range(len(level_ends)):
                    if gs >= level_ends[lvl]:
                        level_ends[lvl] = ge
                        gene_levels.append(lvl)
                        placed = True
                        break
                if not placed and len(level_ends) < max_levels:
                    level_ends.append(ge)
                    gene_levels.append(len(level_ends) - 1)
                    placed = True
                if not placed:
                    earliest = int(np.argmin(level_ends))
                    level_ends[earliest] = ge
                    gene_levels.append(earliest)
            for (idx_g, rowg), lvl in zip(genes_df.iterrows(), gene_levels):
                gs, ge, strand, name = int(rowg["start"]), int(rowg["end"]), rowg["strand"], rowg["name"]
                y = level_height_base + lvl * level_gap
                color = self.gene_color_plus if strand == "+" else self.gene_color_minus
                ax_gene.plot([gs, ge], [y, y], color=color, lw=2.5, zorder=2, solid_capstyle='round')
                gene_length = ge - gs
                arrow_len = min(gene_length * 0.1, 2000)
                if strand == "+":
                    ax_gene.arrow(ge - arrow_len, y, arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                  fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                else:
                    ax_gene.arrow(gs + arrow_len, y, -arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                  fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                gene_exons = [e for e in exons if e[3] == name]
                for es, ee, st, nm in gene_exons:
                    es_d, ee_d = max(es, start_display), min(ee, end_display)
                    if ee_d > es_d and (ee_d - es_d) > 50:
                        rect = mpatches.Rectangle((es_d, y - 0.04), ee_d - es_d, 0.08, facecolor=color, alpha=0.9,
                                                  zorder=3, edgecolor='white', linewidth=0.5)
                        ax_gene.add_patch(rect)
                text_x = (gs + ge) / 2
                text_x = np.clip(text_x, start_display + 500, end_display - 500)
                ax_gene.text(text_x, y + 0.06, name if name else "Unknown", ha="center", va="bottom", fontsize=9, zorder=4,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
            gene_track_height = level_height_base + len(level_ends) * level_gap + 0.2
            ax_gene.set_ylim(0, gene_track_height)
            legend_elems = [
                Line2D([0], [0], color=self.gene_color_plus, lw=2, marker='>', markersize=8, label='Forward (+)'),
                Line2D([0], [0], color=self.gene_color_minus, lw=2, marker='<', markersize=8, label='Reverse (-)')
            ]
            ax_gene.legend(handles=legend_elems, loc='upper right', fontsize=9, framealpha=0.9)
        else:
            ax_gene.text(0.5, 0.5, "No genes in this region", ha="center", va="center", transform=ax_gene.transAxes,
                         fontsize=10, style='italic')
            ax_gene.set_ylim(0, 1)
        ax_gene.set_yticks([])
        ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha='right', va='center')

        # plot signals
        # choose palette: tab10/tab20 per number of tracks (scientific colors)
        cmap_name = 'tab10' if n_tracks <= 8 else 'tab20'
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(n_tracks+2)]
        for i, name in enumerate(titles):
            ax = axes[i + 1]
            sig = np.asarray(signals_to_plot[i])
            if smoothing_sigma and smoothing_sigma > 0:
                sig = gaussian_smooth(sig, sigma=float(smoothing_sigma))
            color = colors[i+2] if self.signal_palette is None else (self.signal_palette[i % len(self.signal_palette)])
            ax.plot(display_positions, sig, color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(display_positions, 0, sig, alpha=0.25, color=color)
            y_max = max(sig.max() * 1.15, 0.1) if len(sig) > 0 else 1
            ax.set_ylim(0, y_max)
            ax.set_yticks(np.linspace(0, y_max, 5))
            ax.tick_params(axis='y', labelsize=9)
            # move the per-track title into the y-label (publication-friendly, rotated 0)
            ax.set_ylabel(f"{name}", fontsize=10, rotation=0, ha='right', va='center')
            # do not use per-subplot title to reduce clutter
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        # x axis formatting
        # displayed length = number of positions in the x-axis
        displayed_len = len(display_positions)
        for ax in axes:
            ax.set_xlim(start_display, end_display)
            if self.xtick_step and displayed_len > self.xtick_step:
                xticks = np.arange(start_display, end_display, self.xtick_step)
                ax.set_xticks(xticks)
                if ax != axes[-1]:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=9)
        def fmt(x, pos):
            return f"{int(x):,}"
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))

        axes[-1].set_xlabel(f"Chromosome position; interval= {chrom}:{start_display:,}-{end_display:,} (length: {displayed_len:,} bp)", fontsize=11)
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        plt.subplots_adjust(hspace=0.12)
        plt.show()
        return fig, axes



def gaussian_smooth(arr, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(arr, sigma=sigma, mode='nearest')

def _to_numpy(x):
    """
    Robust conversion to numpy:
      - handles torch.Tensor on CUDA/CPU (detaches, moves to CPU, converts bfloat16/float16 -> float32)
      - handles numpy arrays
      - handles iterables element-wise (safe for lists of tensors)
    """
    try:
        import torch
    except Exception:
        torch = None

    # torch.Tensor -> numpy
    if torch is not None and isinstance(x, torch.Tensor):
        t = x.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if t.dtype in (getattr(torch, "bfloat16", None), torch.float16):
            t = t.to(torch.float32)
        return t.squeeze().numpy()

    # numpy array
    if isinstance(x, np.ndarray):
        return x.squeeze()

    # objects exposing .cpu()
    try:
        if hasattr(x, "cpu") and callable(x.cpu):
            y = x.cpu()
            if torch is not None and isinstance(y, torch.Tensor):
                t = y.detach()
                if t.dtype in (getattr(torch, "bfloat16", None), torch.float16):
                    t = t.to(torch.float32)
                return t.squeeze().numpy()
            return np.asarray(y).squeeze()
    except Exception:
        pass

    # iterable (element-wise conversion)
    try:
        if hasattr(x, "__iter__") and not isinstance(x, (str, bytes, bytearray)):
            items = list(x)
            if len(items) == 0:
                return np.array([]).squeeze()
            converted = []
            for it in items:
                try:
                    if torch is not None and isinstance(it, torch.Tensor):
                        t = it.detach()
                        if t.device.type != "cpu":
                            t = t.cpu()
                        if t.dtype in (getattr(torch, "bfloat16", None), torch.float16):
                            t = t.to(torch.float32)
                        converted.append(t.squeeze().numpy())
                    elif isinstance(it, np.ndarray):
                        converted.append(it.squeeze())
                    else:
                        converted.append(np.asarray(it))
                except Exception:
                    try:
                        converted.append(np.array(it))
                    except Exception:
                        converted.append(np.asarray(it))
            try:
                return np.asarray(converted).squeeze()
            except Exception:
                try:
                    return np.concatenate([np.atleast_1d(c) for c in converted]).squeeze()
                except Exception:
                    return np.array(converted).squeeze()
    except Exception:
        pass

    # fallback
    return np.array([x]).squeeze()


class ResultsViewer:
    """
    Visualize model prediction results with pre-loaded gene annotations.

    Mimics DatasetViewer behavior:
      - Load GFF once at init
      - Plot full-resolution signals (no binning)
      - Only supports small genomic windows (< 100 kb recommended)
    """
    def __init__(self,
                 annotation_path: Optional[str] = None,
                 signal_palette: Optional[List[str]] = None,
                 xtick_step: int = 4000,
                 dpi: int = 100,
                 max_region_length: int = 100_000):  # 新增安全限制
        self.signal_palette = signal_palette
        self.xtick_step = xtick_step
        self.dpi = dpi
        self.max_region_length = max_region_length
        self.genes_by_chrom = {}
        self.exons_by_chrom = {}
        if annotation_path:
            self._load_gff(annotation_path)

    def _load_gff(self, path):
        """Load GFF/GTF (optionally gzipped). Store genes and exons per chromosome."""
        genes = {}
        exons = {}
        try:
            open_func = gzip.open if path.endswith('.gz') else open
            mode = 'rt' if path.endswith('.gz') else 'r'
            with open_func(path, mode) as fh:
                for line in fh:
                    if line.startswith('#') or not line.strip():
                        continue
                    cols = line.rstrip().split('\t')
                    if len(cols) < 9:
                        continue
                    chrom, src, feature, start, end, score, strand, phase, attrs = cols[:9]
                    try:
                        start_i = int(start)
                        end_i = int(end)
                    except Exception:
                        continue
                    # extract a simple name (gene_name / Name / gene_id)
                    gene_name = ""
                    for key in ("gene_name=", "Name=", "gene_id="):
                        if key in attrs:
                            parts = [p for p in attrs.split(';') if p.strip().startswith(key)]
                            if parts:
                                gene_name = parts[0].split('=', 1)[-1].strip('"').strip()
                                break
                    if feature == "gene":
                        genes.setdefault(chrom, []).append((start_i, end_i, strand, gene_name))
                    elif feature == "exon":
                        exons.setdefault(chrom, []).append((start_i, end_i, strand, gene_name))
        except Exception as e:
            logging.warning(f"Failed to load annotation {path}: {e}")
        self.genes_by_chrom = genes
        self.exons_by_chrom = exons
        logging.info(f"Loaded annotation: chromosomes={list(self.genes_by_chrom.keys())}")

    def get_genes_in_interval(self, chrom, start, end):
        """Return (genes, exons) overlapping [start, end)."""
        genes = self.genes_by_chrom.get(chrom, [])
        exons = self.exons_by_chrom.get(chrom, [])
        genes_f = [(gs, ge, st, nm) for (gs, ge, st, nm) in genes if not (ge < start or gs >= end)]
        exons_f = [(es, ee, st, nm) for (es, ee, st, nm) in exons if not (ee < start or es >= end)]
        return genes_f, exons_f

    def plot(self,
             results: Dict,
             track_order: Optional[List[str]] = None,
             smoothing_sigma: float = 2.0,
             figsize: Optional[Tuple[float, float]] = None,
             show_legend: bool = True,
             gene_color_plus: str = "tab:blue",
             gene_color_minus: str = "tab:orange",
             window_start: Optional[int] = None,
             window_end: Optional[int] = None):
        """
        Plot results. Optional window_start/window_end specify absolute genomic coordinates
        to display; if not provided, use results['position'] interval (original behavior).
        """
        values = results.get("values", {})
        position = results.get("position", (None, None, None))
        chrom, start, end = position if len(position) == 3 else (None, None, None)

        if not values:
            raise ValueError("results['values'] is empty")
        if chrom is None or start is None or end is None:
            raise ValueError("results['position'] must be (chrom, start, end)")

        # determine display window
        orig_start = int(start)
        # original region length inferred from provided position
        orig_region_length = int(end) - int(start)

        if window_start is None and window_end is None:
            start_display = orig_start
            end_display = orig_start + orig_region_length
        else:
            start_display = int(window_start) if window_start is not None else orig_start
            end_display = int(window_end) if window_end is not None else (orig_start + orig_region_length)

        displayed_len = end_display - start_display
        if displayed_len <= 0:
            raise ValueError("Invalid window: window_end must be greater than window_start")

        if displayed_len > self.max_region_length:
            raise ValueError(
                f"Display window length ({displayed_len:,} bp) exceeds max allowed ({self.max_region_length:,} bp)."
            )

        # === 获取基因注释 ===
        genes, exons = self.get_genes_in_interval(chrom, start_display, end_display)

        # === 收集 tracks 和 biosamples ===
        track_names = list(values.keys())
        biosample_set = []
        for tn in track_names:
            for b in values[tn].keys():
                if b not in biosample_set:
                    biosample_set.append(b)
        biosamples = biosample_set

        if track_order:
            ordered = [t for t in track_order if t in track_names]
            ordered += [t for t in track_names if t not in ordered]
            track_names = ordered

        n_heads = len(track_names)
        n_bios = max(1, len(biosamples))
        total_signal_subplots = n_heads * n_bios
        total_subplots = 1 + total_signal_subplots

        default_fig_width = 18.0
        default_fig_height = 1.5 * total_subplots
        figsize = figsize or (default_fig_width, default_fig_height)

        fig, axes = plt.subplots(total_subplots, 1, figsize=figsize, sharex=True, dpi=self.dpi,
                                 gridspec_kw={'height_ratios': [0.8] + [1.0] * total_signal_subplots})
        if total_subplots == 1:
            axes = [axes]
        ax_gene = axes[0]

        # === 基因轨道 ===
        display_positions = np.arange(start_display, end_display)

        if not genes:
            ax_gene.text(0.5, 0.5, "No genes in this region", ha="center", va="center",
                         transform=ax_gene.transAxes, fontsize=10, style='italic')
            ax_gene.set_ylim(0, 1)
        else:
            import pandas as pd
            genes_df = pd.DataFrame(genes, columns=["start", "end", "strand", "name"]).sort_values("start").reset_index(drop=True)
            level_ends = []
            level_height_base = 0.3
            level_gap = 0.25
            max_levels = 8
            gene_levels = []

            for _, g in genes_df.iterrows():
                gs, ge = int(g["start"]), int(g["end"])
                placed = False
                for lvl in range(len(level_ends)):
                    if gs >= level_ends[lvl]:
                        level_ends[lvl] = ge
                        gene_levels.append(lvl)
                        placed = True
                        break
                if not placed and len(level_ends) < max_levels:
                    level_ends.append(ge)
                    gene_levels.append(len(level_ends) - 1)
                    placed = True
                if not placed:
                    earliest = int(np.argmin(level_ends))
                    level_ends[earliest] = ge
                    gene_levels.append(earliest)

            for (idx_g, rowg), lvl in zip(genes_df.iterrows(), gene_levels):
                gs, ge, strand, name = int(rowg["start"]), int(rowg["end"]), rowg["strand"], rowg["name"]
                y = level_height_base + lvl * level_gap
                color = gene_color_plus if strand == "+" else gene_color_minus
                ax_gene.plot([gs, ge], [y, y], color=color, lw=2.5, zorder=2, solid_capstyle='round')

                gene_length = ge - gs
                arrow_len = min(gene_length * 0.1, 2000)
                if strand == "+":
                    ax_gene.arrow(ge - arrow_len, y, arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                  fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)
                else:
                    ax_gene.arrow(gs + arrow_len, y, -arrow_len, 0, head_width=0.04, head_length=arrow_len * 0.3,
                                  fc=color, ec=color, linewidth=0, length_includes_head=True, zorder=3)

                gene_exons = [e for e in exons if e[3] == name]
                for es, ee, st, nm in gene_exons:
                    es_d, ee_d = max(es, start_display), min(ee, end_display)
                    if ee_d > es_d and (ee_d - es_d) > 50:
                        rect = patches.Rectangle((es_d, y - 0.04), ee_d - es_d, 0.08,
                                                facecolor=color, alpha=0.9, zorder=3,
                                                edgecolor='white', linewidth=0.5)
                        ax_gene.add_patch(rect)

                text_x = (gs + ge) / 2
                text_x = np.clip(text_x, start_display + 500, end_display - 500)
                ax_gene.text(text_x, y + 0.06, name if name else "Unknown", ha='center', va='bottom',
                             fontsize=9, zorder=4,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

            gene_track_height = level_height_base + len(level_ends) * level_gap + 0.2
            ax_gene.set_ylim(0, gene_track_height)

            legend_elems = [
                Line2D([0], [0], color=gene_color_plus, lw=2, marker='>', markersize=8, label='Forward (+)'),
                Line2D([0], [0], color=gene_color_minus, lw=2, marker='<', markersize=8, label='Reverse (-)')
            ]
            ax_gene.legend(handles=legend_elems, loc='upper right', fontsize=9, framealpha=0.9)

        ax_gene.set_yticks([])
        ax_gene.set_ylabel("Genes", fontsize=10, rotation=0, ha='right', va='center')

        # === 信号轨道 ===
        idx_ax = 1
        for head in track_names:
            track_dict = values.get(head, {})
            for b in biosamples:
                ax = axes[idx_ax]
                arr = track_dict.get(b, None)
                if arr is None:
                    ax.text(0.5, 0.5, f"No data for {head}/{b}", ha='center', va='center')
                    ax.set_yticks([])
                    idx_ax += 1
                    continue

                y_src = _to_numpy(arr).reshape(-1)
                # source covers [orig_start, orig_start + len(y_src))
                src_len = len(y_src)
                src_start = orig_start
                src_end = orig_start + src_len

                # Build displayed segment of length displayed_len
                seg = np.zeros(displayed_len, dtype=y_src.dtype)
                # compute overlap between source and display window
                overlap_start = max(start_display, src_start)
                overlap_end = min(end_display, src_end)
                if overlap_end > overlap_start:
                    src_slice_start = overlap_start - src_start
                    src_slice_end = overlap_end - src_start
                    dest_slice_start = overlap_start - start_display
                    seg[dest_slice_start:dest_slice_start + (src_slice_end - src_slice_start)] = y_src[src_slice_start:src_slice_end]

                y = seg

                if smoothing_sigma > 0:
                    y = gaussian_smooth(y, smoothing_sigma)

                # Color
                if self.signal_palette:
                    color = self.signal_palette[biosamples.index(b) % len(self.signal_palette)]
                else:
                    cmap = plt.get_cmap('tab10' if len(biosamples) <= 10 else 'tab20')
                    color = cmap((biosamples.index(b)+2) % cmap.N)

                ax.plot(display_positions, y, color=color, linewidth=1.2)
                ax.fill_between(display_positions, 0, y, color=color, alpha=0.14)

                ax.set_ylabel(f"{head}\n{b}", fontsize=9, rotation=0, ha='right', va='center')
                y_max = max(np.nanmax(y) * 1.15, 0.1) if y.size > 0 else 0.1
                ax.set_ylim(0, y_max)
                ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.4)
                idx_ax += 1

        # === X-axis ===
        for ax in axes:
            ax.set_xlim(start_display, end_display)
            if self.xtick_step and displayed_len > self.xtick_step:
                xticks = np.arange(start_display, end_display, self.xtick_step)
                ax.set_xticks(xticks)
                if ax != axes[-1]:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.tick_params(axis='x', labelsize=9)

        def fmt(x, pos): return f"{int(x):,}"
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt))

        axes[-1].set_xlabel(f"Chromosome position; interval= {chrom}:{start_display:,}-{end_display:,} ({displayed_len:,} bp)", fontsize=10)

        # === Legend ===
        if show_legend and biosamples:
            handles = []
            for i, b in enumerate(biosamples):
                if self.signal_palette:
                    c = self.signal_palette[i % len(self.signal_palette)]
                else:
                    cmap = plt.get_cmap('tab10' if len(biosamples) <= 10 else 'tab20')
                    c = cmap((i+2) % cmap.N)
                handles.append(Line2D([0], [0], color=c, lw=2))
            try:
                axes[1].legend(handles, biosamples, loc='upper right', fontsize=8)
            except:
                fig.legend(handles, biosamples, loc='upper right', fontsize=8)

        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
        plt.subplots_adjust(hspace=0.12)
        return fig, axes
