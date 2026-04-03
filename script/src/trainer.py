# 标准库（内置模块）
from typing import Optional, Union, Dict, Any

import sys
import json
import os
import time

# 第三方库（pip 安装的包）
import torch
from torch.utils.data import Dataset, Subset, DataLoader, DistributedSampler, TensorDataset
from collections import defaultdict

# Hugging Face Transformers
from transformers import (
    Trainer,
    TrainerCallback
    )

# 从自定义仓库中导入模块
from src.utils.dist import dist_print



# 自定义训练器
class DistributedSamplerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(int(state.epoch))



class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chrom2id = {f"chr{i}": i for i in range(1,23)}
        self._accumulated_per_head_losses = defaultdict(list)
        self._accumulated_per_head_metrics = defaultdict(list)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
            pos_strand_mask=inputs["pos_strand_mask"],
            neg_strand_mask=inputs["neg_strand_mask"],
            sample_track_means=inputs.get("sample_track_means", None),
        )
        loss = outputs["loss"]
        per_head_losses = outputs.get("per_head_losses", {})
        per_head_metrics = outputs.get("per_head_metrics", {})

        extra = {"per_head_losses": per_head_losses, "per_head_metrics": per_head_metrics}
        return (loss, extra) if return_outputs else loss

    
    # 修复log，在log step不为1时也正确
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Strict logging: any unexpected content in per-head buffers raises immediately.
        """
        # per-head losses: mean
        for name, losses in self._accumulated_per_head_losses.items():
            if not losses:
                continue
            stacked = torch.stack([
                l if isinstance(l, torch.Tensor) else torch.tensor(l)
                for l in losses
            ])
            logs[f"loss_{name}"] = float(stacked.mean().item())

        # per-head metrics: mean
        for name, metrics in self._accumulated_per_head_metrics.items():
            if not metrics:
                continue
            stacked = torch.stack([
                m if isinstance(m, torch.Tensor) else torch.tensor(m)
                for m in metrics
            ])
            logs[f"metric_{name}"] = float(stacked.mean().item())

        self._accumulated_per_head_losses.clear()
        self._accumulated_per_head_metrics.clear()
        super().log(logs, *args, **kwargs)

        
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, extra = self.compute_loss(model, inputs, return_outputs=True)

        # 缩放总 loss（HF 原生逻辑）
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # 反向传播
        self.accelerator.backward(loss)

        # === 关键：累积分任务 loss（做相同缩放）===
        per_head_losses = extra.get("per_head_losses", {})
        world_size = 1
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = float(torch.distributed.get_world_size())
        except Exception:
            world_size = 1.0

        device_for_reduce = None
        try:
            device_for_reduce = next(model.parameters()).device
        except Exception:
            device_for_reduce = torch.device("cpu")

        for name, val in per_head_losses.items():
            # 若 val 是 dict（例如 poisson-multinomial 返回的 breakdown），则展开每一项
            if isinstance(val, dict):
                for subk, subv in val.items():
                    subname = f"{name}.{subk}"
                    # 标准化为 tensor
                    try:
                        if not isinstance(subv, torch.Tensor):
                            subv_t = torch.tensor(subv, dtype=torch.float32, device=device_for_reduce)
                        else:
                            subv_t = subv.detach().to(device_for_reduce)
                    except Exception:
                        try:
                            subv_t = torch.tensor(float(subv), dtype=torch.float32, device=device_for_reduce)
                        except Exception:
                            continue

                    # 多卡聚合（sum -> / world_size 即为平均）
                    try:
                        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
                            torch.distributed.all_reduce(subv_t, op=torch.distributed.ReduceOp.SUM)
                            subv_t = subv_t / world_size
                    except Exception:
                        pass

                    # HF 原生对总 loss 做的 mean()/grad_accum 缩放也应同样应用于 per-head
                    try:
                        if self.args.n_gpu > 1:
                            subv_t = subv_t.mean()
                    except Exception:
                        pass
                    if self.args.gradient_accumulation_steps > 1:
                        subv_t = subv_t / self.args.gradient_accumulation_steps

                    # 存为 CPU tensor（便于后续 stack & logging）
                    self._accumulated_per_head_losses[subname].append(subv_t.detach().cpu())
                # 已处理该 name 下的所有子项，继续下一个
                continue

            # 标准化为 tensor（非 dict 情况）
            try:
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32, device=device_for_reduce)
                else:
                    val = val.detach().to(device_for_reduce)
            except Exception:
                # fallback: convert to float -> tensor
                try:
                    val = torch.tensor(float(val), dtype=torch.float32, device=device_for_reduce)
                except Exception:
                    continue

            # 多卡聚合（sum -> / world_size 即为平均）
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
                    torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
                    val = val / world_size
            except Exception:
                # ignore reduce errors, keep local value
                pass

            # HF 原生对总 loss 做的 mean()/grad_accum 缩放也应同样应用于 per-head
            try:
                if self.args.n_gpu > 1:
                    val = val.mean()
            except Exception:
                pass
            if self.args.gradient_accumulation_steps > 1:
                val = val / self.args.gradient_accumulation_steps

            # 存为 CPU tensor（便于后续 stack & logging）
            self._accumulated_per_head_losses[name].append(val.detach().cpu())

        # === 关键：累积 per-head metrics（同 per_head_losses 的处理逻辑） ===
        per_head_metrics = extra.get("per_head_metrics", {})
        for name, val in per_head_metrics.items():
            # 若 val 是 dict，则展开每一项
            if isinstance(val, dict):
                for subk, subv in val.items():
                    subname = f"{name}.{subk}"
                    try:
                        if not isinstance(subv, torch.Tensor):
                            subv_t = torch.tensor(subv, dtype=torch.float32, device=device_for_reduce)
                        else:
                            subv_t = subv.detach().to(device_for_reduce)
                    except Exception:
                        try:
                            subv_t = torch.tensor(float(subv), dtype=torch.float32, device=device_for_reduce)
                        except Exception:
                            continue

                    try:
                        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
                            torch.distributed.all_reduce(subv_t, op=torch.distributed.ReduceOp.SUM)
                            subv_t = subv_t / world_size
                    except Exception:
                        pass

                    try:
                        if self.args.n_gpu > 1:
                            subv_t = subv_t.mean()
                    except Exception:
                        pass
                    if self.args.gradient_accumulation_steps > 1:
                        subv_t = subv_t / self.args.gradient_accumulation_steps

                    self._accumulated_per_head_metrics[subname].append(subv_t.detach().cpu())
                continue

            try:
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32, device=device_for_reduce)
                else:
                    val = val.detach().to(device_for_reduce)
            except Exception:
                try:
                    val = torch.tensor(float(val), dtype=torch.float32, device=device_for_reduce)
                except Exception:
                    continue

            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
                    torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
                    val = val / world_size
            except Exception:
                pass

            try:
                if self.args.n_gpu > 1:
                    val = val.mean()
            except Exception:
                pass
            if self.args.gradient_accumulation_steps > 1:
                val = val / self.args.gradient_accumulation_steps

            self._accumulated_per_head_metrics[name].append(val.detach().cpu())

        return loss.detach()
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=self._collate_fn, 
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            drop_last=self.args.dataloader_drop_last,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset or self.eval_dataset
        )

        sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            collate_fn=self._collate_fn,  
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            drop_last=False,
        )

    def _get_train_sampler(self):
        # Use the trainer's built-in method for creating DistributedSampler
        if self.train_dataset is None:
            return None
            
        # For newer versions of transformers, use the built-in method
        return DistributedSampler(
            self.train_dataset,
            shuffle=True,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset):
        return DistributedSampler(
            eval_dataset,
            shuffle=False,
        )


    def _collate_fn(self, batch):
        """
        Compact collate: stack tensors and return position metadata.
        """
        # stack tensors
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        pos_stand_mask = torch.stack([b["pos_strand_mask"] for b in batch])
        neg_strand_mask = torch.stack([b["neg_strand_mask"] for b in batch])

        # position metadata
        position = [b["position"] for b in batch]
        chroms = [p[0] for p in position]
        starts = [p[1] for p in position]
        ends = [p[2] for p in position]

        position_chrom = torch.tensor([self.chrom2id.get(c, 0) for c in chroms], dtype=torch.long)
        position_start = torch.tensor(starts, dtype=torch.long)
        position_end = torch.tensor(ends, dtype=torch.long)

        # personal-genome: per-sample track means (shape [B, 2]) + sample_id list
        sample_id = [b.get("sample_id") for b in batch]
        if all(b.get("sample_track_means") is not None for b in batch):
            sample_track_means = torch.stack([b["sample_track_means"] for b in batch])  # [B, 2]
        else:
            sample_track_means = None

        return {
            "input_ids": input_ids,
            "labels": labels,
            "sample_id": sample_id,  # 非 tensor，Trainer 不会搬到 GPU；仅用于 debug
            "sample_track_means": sample_track_means,  # tensor，会随 batch 一起搬到 GPU
            "position": position,
            "position_chrom": position_chrom,
            "position_start": position_start,
            "position_end": position_end,
            "pos_strand_mask": pos_stand_mask,
            "neg_strand_mask": neg_strand_mask,
        }
   

    
    
class LocalLoggerCallback(TrainerCallback):
    def __init__(self, log_file_path: str, metrics_file_path: Optional[str] = None):
        super().__init__()
        self.log_file_path = log_file_path
        self.metrics_file_path = metrics_file_path or log_file_path.replace(".log", "_metrics.jsonl")
        self._model_ref = None  # 缓存 model 引用

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return

        _step = state.global_step
        _epoch = state.epoch or 0

        # 直接从 logs 提取所有字段
        content = []
        for k, v in logs.items():
            if k in {"epoch", "step", "runtime"}:
                continue
            if isinstance(v, float):
                content.append(f"{k}: {v:.6f}")
            else:
                content.append(f"{k}: {v}")

        message = f"📌 TRAINER LOG | Step: {_step} | Epoch: {_epoch:.2f} | " + " | ".join(content)
        dist_print(message)

        # --- 写入结构化 eval 日志 ---
        if logs and ("eval_loss" in logs or any(k.startswith("eval_") for k in logs.keys())):
            try:
                log_entry = {
                    "step": _step,
                    "epoch": round(_epoch, 4),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **{
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in logs.items()
                    }
                }
                with open(self.metrics_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                dist_print(f"[LocalLoggerCallback] 写入 metrics 失败: {e}")