# 标准库（内置模块）
import numpy as np

# 第三方库（pip 安装的包）
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score, 
    average_precision_score,
    r2_score
)
from scipy.stats import pearsonr, spearmanr
from transformers import EvalPrediction
from src.utils.dist import dist_print

def evaluate_zero_inflated(y_true, y_pred, epsilon=1e-8):
    """
    计算零膨胀回归任务的评测指标（含非零区域的相关系数 和 样本级平均指标）

    参数:
    y_true: list of arrays 或 2D array, 形状为 [N, L]，每个样本长度 L
    y_pred: list of arrays 或 2D array, 形状为 [N, L]
    epsilon: 判断预测为0的阈值（预测值 < epsilon 视为预测0）

    返回:
    metrics: 包含所有指标的字典
    """
    # 转换输入为 list of arrays（确保结构清晰）
    if isinstance(y_true, np.ndarray) and y_true.ndim == 2:
        y_true = [y_true[i] for i in range(len(y_true))]
    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
        y_pred = [y_pred[i] for i in range(len(y_pred))]

    # 展平用于全局指标计算
    y_true_flatten = np.concatenate(y_true)
    y_pred_flatten = np.concatenate(y_pred)

    if len(y_true_flatten) != len(y_pred_flatten):
        raise ValueError("y_true_flatten 和 y_pred_flatten 长度必须一致")

    # --------------------------
    # 1. 零值判断与基础准备
    # --------------------------
    true_zero = (y_true_flatten == 0)
    true_nonzero = (y_true_flatten != 0)

    # --------------------------
    # 2. 整体误差指标（Global）
    # --------------------------
    mse = mean_squared_error(y_true_flatten, y_pred_flatten)
    mae = mean_absolute_error(y_true_flatten, y_pred_flatten)
    r2 = r2_score(y_true_flatten, y_pred_flatten)
    pearson, _ = pearsonr(y_true_flatten, y_pred_flatten)
    try:
        log1p_pearson, _ = pearsonr(np.log1p(y_true_flatten), np.log1p(y_pred_flatten))
    except:
        log1p_pearson = np.nan
    spearman, _ = spearmanr(y_true_flatten, y_pred_flatten)

    # --------------------------
    # 3. 零值识别指标
    # --------------------------
    try:
        auroc_zero = roc_auc_score(true_zero, -y_pred_flatten)
        auprc_zero = average_precision_score(true_zero, -y_pred_flatten)
        auprc_nonzero = average_precision_score(true_nonzero, y_pred_flatten)
    except ValueError as e:
        print(f"无法计算 AUROC/AUPRC: {e}")
        auroc_zero = np.nan
        auprc_zero = np.nan
        auprc_nonzero = np.nan

    # --------------------------
    # 4. 非零区域指标
    # --------------------------
    non_zero_mask = ~true_zero
    y_true_nonzero = y_true_flatten[non_zero_mask]
    y_pred_nonzero = y_pred_flatten[non_zero_mask]
    n_nonzero = len(y_true_nonzero)

    if n_nonzero < 2:
        non_zero_mse = np.nan
        non_zero_mae = np.nan
        non_zero_pearson = np.nan
        non_zero_spearman = np.nan
        non_zero_log1p_pearson = np.nan
    else:
        non_zero_mse = mean_squared_error(y_true_nonzero, y_pred_nonzero)
        non_zero_mae = mean_absolute_error(y_true_nonzero, y_pred_nonzero)
        non_zero_pearson, _ = pearsonr(y_true_nonzero, y_pred_nonzero)
        non_zero_spearman, _ = spearmanr(y_true_nonzero, y_pred_nonzero)
        try:
            non_zero_log1p_pearson, _ = pearsonr(np.log1p(y_true_nonzero), np.log1p(y_pred_nonzero))
        except:
            non_zero_log1p_pearson = np.nan

    # --------------------------
    # 5. 样本级平均指标 (Per-sample Mean)
    # --------------------------
    sample_mses = []
    sample_maes = []
    sample_r2s = []

    for yt, yp in zip(y_true, y_pred):
        if len(yt) < 2:
            continue

        # 基础指标
        sample_mses.append(mean_squared_error(yt, yp))
        sample_maes.append(mean_absolute_error(yt, yp))
        # sample_r2s.append(r2_score(yt, yp))


    # 取平均（注意：可能为空）
    sample_mean_mse = np.mean(sample_mses) if sample_mses else np.nan
    sample_mean_mae = np.mean(sample_maes) if sample_maes else np.nan

    # --------------------------
    # 6. 整理所有指标
    # --------------------------
    metrics = {
        # 整体指标（Global）
        "mse": round(float(mse), 6),
        "mae": round(float(mae), 6),
        "r2_score": round(float(r2), 6),
        "pearson": round(float(pearson), 6),
        "log1p_pearson": round(float(log1p_pearson), 6),
        "spearman": round(float(spearman), 6),

        # 零值识别指标
        "zero_auroc": round(float(auroc_zero), 6),
        "zero_auprc": round(float(auprc_zero), 6),
        "nonzero_auprc": round(float(auprc_nonzero), 6),

        # 非零区域指标
        "nonzero_mse": round(float(non_zero_mse), 6),
        "eval_nonzero_mae": round(float(non_zero_mae), 6),
        "nonzero_pearson": round(float(non_zero_pearson), 6),
        "nonzero_log1p_pearson": round(float(non_zero_log1p_pearson), 6),
        "nonzero_spearman": round(float(non_zero_spearman), 6),

        # 样本平均指标（Per-sample Mean）
        "sample_mean_mse": round(float(sample_mean_mse), 6),
        "sample_mean_mae": round(float(sample_mean_mae), 6),

        # 辅助信息
        "zero_ratio": round(float(np.mean(true_zero) * 100), 4),
    }

    return metrics


def compute_multimodal_metrics(eval_pred: EvalPrediction, val_chromosomes, tokenizer):
    """
    Eval 指标（按你的要求）：
    - 使用 N 条 32k window
    - 每条 window：对每个 track 分别算 Pearson
    - 对 N 条 window 的 Pearson 取平均，作为最终指标
    同时补充：log1p Pearson / MSE / MAE（同样按 window 先算再平均）
    """
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    # HF 有时会把 preds 包成 tuple（例如带额外输出）
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if isinstance(labels, (tuple, list)):
        labels = labels[0]

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    if preds.ndim != 3 or labels.ndim != 3:
        raise ValueError(
            f"[compute_multimodal_metrics] expected preds/labels to be 3D [N,L,C]. "
            f"Got preds={preds.shape}, labels={labels.shape}"
        )

    # 兼容可能出现的 [N,C,L]（尽量不猜：只做一个非常保守的 heuristic）
    def _to_nlc(x: np.ndarray) -> np.ndarray:
        # 若第二维很小、第三维很大，通常是 [N,C,L] -> 转成 [N,L,C]
        if x.shape[1] <= 16 and x.shape[2] > 16:
            return np.transpose(x, (0, 2, 1))
        return x

    preds = _to_nlc(preds)
    labels = _to_nlc(labels)

    if preds.shape != labels.shape:
        raise ValueError(
            f"[compute_multimodal_metrics] preds/labels shape mismatch after transpose: "
            f"preds={preds.shape}, labels={labels.shape}"
        )

    N, L, C = labels.shape

    # 向量化的 per-window Pearson：对每条 window 独立算相关系数
    def _pearson_per_window(y_true_2d: np.ndarray, y_pred_2d: np.ndarray) -> np.ndarray:
        # y_true_2d, y_pred_2d: [N, L]
        yt = y_true_2d.astype(np.float64, copy=False)
        yp = y_pred_2d.astype(np.float64, copy=False)

        yt = yt - yt.mean(axis=1, keepdims=True)
        yp = yp - yp.mean(axis=1, keepdims=True)

        num = np.sum(yt * yp, axis=1)
        den = np.sqrt(np.sum(yt * yt, axis=1) * np.sum(yp * yp, axis=1))

        r = np.full((yt.shape[0],), np.nan, dtype=np.float64)
        valid = den > 0
        r[valid] = num[valid] / den[valid]
        return r  # [N]

    pearson_by_track = []
    log1p_pearson_by_track = []
    mse_by_track = []
    mae_by_track = []

    # 逐 track 做（C 很小：你现在就是 2 通道）
    for c in range(C):
        y = labels[:, :, c]
        p = preds[:, :, c]

        r = _pearson_per_window(y, p)
        pearson_by_track.append(r)  # [N]

        # log1p Pearson（对负值做 clip，防止 log1p 出 nan；你的信号理论上应 >=0）
        y_log = np.log1p(np.clip(y, 0, None))
        p_log = np.log1p(np.clip(p, 0, None))
        r_log = _pearson_per_window(y_log, p_log)
        log1p_pearson_by_track.append(r_log)

        # 每 window 的 MSE/MAE（先 per-window，再平均）
        mse = np.mean((p - y) ** 2, axis=1)
        mae = np.mean(np.abs(p - y), axis=1)
        mse_by_track.append(mse)
        mae_by_track.append(mae)

    pearson_by_track = np.stack(pearson_by_track, axis=1)          # [N, C]
    log1p_pearson_by_track = np.stack(log1p_pearson_by_track, axis=1)  # [N, C]
    mse_by_track = np.stack(mse_by_track, axis=1)                  # [N, C]
    mae_by_track = np.stack(mae_by_track, axis=1)                  # [N, C]

    metrics = {
        "num_windows": int(N),

        # 你要的主指标：window-wise Pearson 平均（跨 window、跨 track）
        "pearson_mean": float(np.nanmean(pearson_by_track)),

        # 同时把每个通道单独报出来（避免 plus/minus 名字绑死）
        **{f"pearson_track{c}": float(np.nanmean(pearson_by_track[:, c])) for c in range(C)},

        # 额外指标（同样 window-wise 平均）
        "log1p_pearson_mean": float(np.nanmean(log1p_pearson_by_track)),
        **{f"log1p_pearson_track{c}": float(np.nanmean(log1p_pearson_by_track[:, c])) for c in range(C)},

        "mse_mean": float(np.nanmean(mse_by_track)),
        **{f"mse_track{c}": float(np.nanmean(mse_by_track[:, c])) for c in range(C)},

        "mae_mean": float(np.nanmean(mae_by_track)),
        **{f"mae_track{c}": float(np.nanmean(mae_by_track[:, c])) for c in range(C)},
    }

    return metrics



