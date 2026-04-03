import torch
from typing import Optional
import logging
# Poisson Loss
def poisson_loss(preds, targets, eps=1e-7):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    poisson_nll = preds - targets * torch.log(preds + eps)
    return torch.mean(poisson_nll)

# Tweedie Loss（仅一个可学习参数 p）
def tweedie_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Tweedie 回归损失（负对数似然近似），适用于 1 < p < 2 （复合泊松-伽马）
    用于建模零膨胀连续正数数据（如 RNA-seq 覆盖度）
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    preds = preds + eps
    targets = targets + eps
    p_clipped = p.clamp(min=1.01, max=1.99)  # 安全边界

    term1 = -targets * torch.pow(preds, 1 - p_clipped) / (1 - p_clipped)
    term2 = torch.pow(preds, 2 - p_clipped) / (2 - p_clipped)

    loss = term1 + term2
    return torch.mean(loss)


def poisson_multinomial_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float = 1e-8,
    multinomial_resolution: Optional[int] = None,
    multinomial_term_scale: float = 0.1,
    multinomial_weight: float = 5,
    poisson_weight: float = 1,
    add_gene_level_loss: bool = False,
    gene_mask: Optional[torch.Tensor] = None,
    gene_level_weight: float = 1.0,
) -> torch.Tensor:
    """
    Poisson-Multinomial loss.

    Track-level: compute per-track (channel) Poisson + Multinomial over groups.
    Gene-level (optional, enabled by add_gene_level_loss and providing gene_mask):
      - gene_mask: bool tensor of shape [B, L], True for positions belonging to genes.
      - For each contiguous True run in gene_mask[b] we treat it as one gene.
      - For each gene we compute the mean value across positions for y_pred/y_true -> [C].
      - For each gene compute Poisson on the sum across tracks (s_pred_gene, s_true_gene)
        and Multinomial across tracks (distribution of gene-value over tracks).
      - Average gene losses across genes per sample, then across batch -> gene_level_total_mean.
    """
    y_true_eps = y_true + epsilon
    y_pred_eps = y_pred + epsilon

    B, L, C = y_pred.shape

    # determine grouping resolution
    if multinomial_resolution is None or multinomial_resolution <= 0 or multinomial_resolution >= L:
        res = L
    else:
        res = int(multinomial_resolution)

    groups = (L + res - 1) // res
    pad_len = groups * res - L

    # pad with small epsilon to avoid zero sums / div-by-zero
    if pad_len > 0:
        pad_pred = torch.full((B, pad_len, C), fill_value=epsilon, device=y_pred.device, dtype=y_pred.dtype)
        pad_true = torch.full((B, pad_len, C), fill_value=epsilon, device=y_true.device, dtype=y_true.dtype)
        y_pred_p = torch.cat([y_pred_eps, pad_pred], dim=1)
        y_true_p = torch.cat([y_true_eps, pad_true], dim=1)
    else:
        y_pred_p = y_pred_eps
        y_true_p = y_true_eps

    # grouped tensors: [B, groups, res, C]
    y_pred_g = y_pred_p.reshape(B, groups, res, C)
    y_true_g = y_true_p.reshape(B, groups, res, C)

    # totals per group: [B, groups, C]
    s_pred = y_pred_g.sum(dim=2)
    s_true = y_true_g.sum(dim=2)

    # track-level Poisson term
    track_level_poisson_term = s_pred - s_true * torch.log(s_pred + epsilon) + (s_true * torch.log(s_true + epsilon) - s_true)

    # track-level Multinomial probabilities and NLL
    p_pred = y_pred_g / (s_pred.unsqueeze(2) + epsilon)
    track_level_multinomial_term = -(y_true_g * torch.log(p_pred + epsilon)).sum(dim=2)  # [B, groups, C]

    # scale by group size (keep same convention as prior code)
    track_level_poisson_term = track_level_poisson_term / res
    track_level_multinomial_term = multinomial_term_scale * track_level_multinomial_term / res

    # apply explicit weights
    track_level_multinomial_term = multinomial_weight * track_level_multinomial_term
    track_level_poisson_term = poisson_weight * track_level_poisson_term

    # combine per (B, groups, C)
    track_level_loss_per_bgc = track_level_multinomial_term + track_level_poisson_term

    # average across groups -> [B, C], then across batch & channels -> scalar
    track_level_loss_per_bc = track_level_loss_per_bgc.mean(dim=1)  # [B, C]
    track_level_total_mean = track_level_loss_per_bc.mean()

    # prepare outputs for return_terms
    track_level_poisson_mean = track_level_poisson_term.mean()
    track_level_multinomial_mean = track_level_multinomial_term.mean()

    # gene-level loss (optional) - gene_mask expected shape [B, L], dtype=bool
    # single zero scalar reused for initialization and fallback
    zero_scalar = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
    gene_level_total_mean = zero_scalar
    gene_level_poisson_mean = zero_scalar
    gene_level_multinomial_mean = zero_scalar

    if add_gene_level_loss and gene_mask is not None:
        # ensure boolean mask on correct device
        mask = gene_mask.to(device=y_pred.device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        # y_pred_eps / y_true_eps shape: [B, L, C]
        y_pred_pos = y_pred_eps
        y_true_pos = y_true_eps

        per_sample_gene_losses = []
        per_sample_poisson_losses = []
        per_sample_multinomial_losses = []

        for b in range(B):
            mask_b = mask[b]  # [L]
            if not mask_b.any():
                # no genes in this sample -> record zeros
                per_sample_gene_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
                per_sample_poisson_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
                per_sample_multinomial_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
                continue
            idxs = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                per_sample_gene_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
                per_sample_poisson_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
                per_sample_multinomial_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
                continue
            # find contiguous runs in idxs
            idxs_cpu = idxs.to('cpu').numpy()
            runs = []
            start = idxs_cpu[0]
            prev = start
            for i in idxs_cpu[1:]:
                if i == prev + 1:
                    prev = i
                    continue
                else:
                    runs.append((start, prev + 1))  # end exclusive
                    start = i
                    prev = i
            runs.append((start, prev + 1))

            gene_losses = []
            poisson_vals_b = []
            multinomial_vals_b = []

            for (s_idx, e_idx) in runs:
                # compute mean per-track value over positions for this gene
                seg_pred = y_pred_pos[b, s_idx:e_idx, :]  # [seg_len, C]
                seg_true = y_true_pos[b, s_idx:e_idx, :]  # [seg_len, C]
                # safety: if seg_len==0 skip
                if seg_pred.numel() == 0 or seg_true.numel() == 0:
                    continue
                mean_pred = seg_pred.mean(dim=0)  # [C]
                mean_true = seg_true.mean(dim=0)  # [C]

                # totals across tracks (C) for this gene
                s_pred_gene = mean_pred.sum()  # scalar
                s_true_gene = mean_true.sum()  # scalar

                # gene-level poisson term (scalar)
                poisson_gene = s_pred_gene - s_true_gene * torch.log(s_pred_gene + epsilon) + (s_true_gene * torch.log(s_true_gene + epsilon) - s_true_gene)

                # gene-level multinomial across tracks
                p_pred_gene = mean_pred / (s_pred_gene + epsilon)  # [C]
                multinomial_gene = -(mean_true * torch.log(p_pred_gene + epsilon)).sum()  # scalar

                # apply scaling consistent with track-level convention
                # multinomial_gene = multinomial_term_scale * multinomial_gene
                multinomial_gene = multinomial_gene
                # weights
                multinomial_gene = multinomial_weight * multinomial_gene
                poisson_gene = poisson_weight * poisson_gene

                loss_gene = multinomial_gene + poisson_gene  # scalar

                gene_losses.append(loss_gene)
                poisson_vals_b.append(poisson_gene)
                multinomial_vals_b.append(multinomial_gene)

            # if no genes were collected for this sample, record zeros
            if len(gene_losses) == 0:
                per_sample_gene_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
                per_sample_poisson_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
                per_sample_multinomial_losses.append(torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype))
            else:
                # sum per-gene losses to obtain sample-level loss
                loss_gene_b = torch.stack(gene_losses).sum()
                poisson_b = torch.stack(poisson_vals_b).sum()
                multinomial_b = torch.stack(multinomial_vals_b).sum()
                per_sample_gene_losses.append(loss_gene_b)
                per_sample_poisson_losses.append(poisson_b)
                per_sample_multinomial_losses.append(multinomial_b)

        if len(per_sample_gene_losses) > 0:
            # average sample-level summed losses across batch
            gene_level_total_mean = torch.stack(per_sample_gene_losses).mean()
            gene_level_poisson_mean = torch.stack(per_sample_poisson_losses).mean() if len(per_sample_poisson_losses) > 0 else torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            gene_level_multinomial_mean = torch.stack(per_sample_multinomial_losses).mean() if len(per_sample_multinomial_losses) > 0 else torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        else:
            # no genes found in batch -> keep zeros
            gene_level_total_mean = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            gene_level_poisson_mean = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            gene_level_multinomial_mean = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

    # combine track + gene (if added) with explicit gene-level weight
    # gene_level_total_mean is initialized to a zero-tensor; apply weight only when requested
    if add_gene_level_loss:
        total_loss = track_level_total_mean + gene_level_weight * gene_level_total_mean
    else:
        total_loss = track_level_total_mean

    # normalize return shape: always return the same 7-tuple of tensors.
    # Return the computed gene-level tensors (they will be zero if not computed).
    gene_total_ret = gene_level_total_mean
    gene_poisson_ret = gene_level_poisson_mean
    gene_multinom_ret = gene_level_multinomial_mean

    result = (
        total_loss,
        track_level_total_mean,
        track_level_poisson_mean,
        track_level_multinomial_mean,
        gene_total_ret,
        gene_poisson_ret,
        gene_multinom_ret,
    )

    return result
