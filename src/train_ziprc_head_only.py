#!/usr/bin/env python3
"""
Fine-tune a model for ZIP (Zero-overhead Inference-time Prediction) with DDP support.

This variant implements the "frozen backbone + trainable output head" baseline:

- The transformer body is kept completely frozen.
- Only the LM head / output embedding layer is trainable.
- The head learns to predict the joint distribution
  P(reward_bin, tokens_remaining_bin) from the frozen hidden states.
- KL to a reference model is removed, since the backbone cannot drift.

Example usage:
    python train_ziprc_head_only.py --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --weights_path models/zip_head_only --data_path data/data.parquet
"""

from __future__ import annotations
import argparse, math, os, time, wandb, ast
from typing import List
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from ziprc_training_visualization import (
    visualize_predictions,
    log_prediction_distributions,
    log_joint_distribution_grid,
)


class ZIPDataset(Dataset):
    def __init__(self, table, max_length: int = 32_768, thinking_only: bool = False,
                 thinking_token_id: int = 151667, reward_values: List[float] | None = None,
                 label_column: str = "correct"):
        self.table = pq.read_table(table).to_pandas() if isinstance(table, str) else table
        cols = set(self.table.columns)
        # Decide which supervision column to use
        choice = (label_column or "correct").lower()
        if choice == "auto":
            if "correct" in cols:
                self.reward_column = "correct"
            elif "value" in cols:
                self.reward_column = "value"
            else:
                raise ValueError("Data must contain 'correct' or 'value' column.")
        elif choice == "correct":
            if "correct" in cols:
                self.reward_column = "correct"
            else:
                raise ValueError("Requested label_column='correct' but column 'correct' is not present in the dataset.")
        elif choice == "value":
            if "value" not in cols:
                raise ValueError("Requested label_column='value' but 'value' not found in data.")
            self.reward_column = "value"
        else:
            raise ValueError(f"Unknown label_column: {label_column}")

        # If training on correctness, enforce binary labels and coerce to {0.0, 1.0}
        if self.reward_column == "correct":
            uniq = set(self.table["correct"].dropna().unique().tolist())
            uniq_numeric = {float(x) for x in uniq}
            if not uniq_numeric.issubset({0.0, 1.0}):
                raise ValueError(
                    f"Column 'correct' must be binary in {{0,1}}/{{False,True}}; found values {sorted(uniq, key=str)}."
                )
            self.table["correct"] = self.table["correct"].astype(float)

        if thinking_only:
            orig_len = len(self.table)

            def _as_list_safe(x):
                if isinstance(x, (list, tuple)):
                    return list(x)
                if isinstance(x, str):
                    return list(ast.literal_eval(x))
                return list(x)

            # Filter for samples that contain the thinking token
            thinking_mask = self.table["input_ids"].apply(lambda ids: thinking_token_id in _as_list_safe(ids))
            self.table = self.table[thinking_mask].reset_index(drop=True)
            print(
                f"Filtered to thinking samples only: {len(self.table)}/{orig_len} "
                f"({len(self.table) / orig_len:.1%}) samples retained."
            )

        self.max_length = max_length

        # Define bins for tokens-to-completion
        # Two 256-token bins, then power-of-2 sized bins: [0-255], [256-511], [512-1023], [1024-2047],
        # [2048-4095], [4096-8191], [8192-16383], [16384-32767]
        self.length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        self.num_length_bins = len(self.length_bins) - 1  # 8 bins

        # Reward bin midpoints:
        #  - If explicitly provided, use them.
        #  - Else, default to binary [0,1] for 'correct', or 7 bins for 'value'.
        if reward_values is not None:
            self.reward_values = reward_values
        else:
            if self.reward_column == "correct":
                self.reward_values = [0.0, 1.0]
            else:
                self.reward_values = [0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0]
        self.num_reward_states = len(self.reward_values)
        self.num_bins = self.num_length_bins * self.num_reward_states

        # Derive value bin edges from midpoints so we truly have BINS
        # Interior edges are midpoints between consecutive midpoints; ends extrapolate by half-step
        if self.num_reward_states >= 2:
            edges = [0.0] * (self.num_reward_states + 1)
            for i in range(1, self.num_reward_states):
                edges[i] = 0.5 * (self.reward_values[i - 1] + self.reward_values[i])
            first_step = self.reward_values[1] - self.reward_values[0]
            last_step = self.reward_values[-1] - self.reward_values[-2]
            edges[0] = self.reward_values[0] - 0.5 * first_step
            edges[-1] = self.reward_values[-1] + 0.5 * last_step
        else:
            # Single state: make a wide bin around it
            edges = [self.reward_values[0] - 0.5, self.reward_values[0] + 0.5]
        # Clamp to [0,1] range
        edges[0] = max(0.0, edges[0])
        edges[-1] = min(1.0, edges[-1])
        self.value_bin_edges = edges

    def __len__(self):
        return len(self.table)

    def _get_bin_idx(self, tokens_to_completion, reward):
        # Find which length bin this falls into
        length_bin = 0
        for i in range(len(self.length_bins) - 1):
            if self.length_bins[i] <= tokens_to_completion < self.length_bins[i + 1]:
                length_bin = i
                break

        # Handle overflow: anything >= last bin boundary goes to the last bin
        if tokens_to_completion >= self.length_bins[-1]:
            length_bin = self.num_length_bins - 1

        # Map continuous value to bin via edges
        reward_state = 0
        for i in range(len(self.value_bin_edges) - 1):
            if self.value_bin_edges[i] <= reward < self.value_bin_edges[i + 1]:
                reward_state = i
                break
        # Handle edge case of reward == last edge
        if reward >= self.value_bin_edges[-1]:
            reward_state = self.num_reward_states - 1

        return length_bin + reward_state * self.num_length_bins

    def __getitem__(self, idx):
        row = self.table.iloc[idx]

        def _to_int_list(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return [int(t) for t in x]
            if isinstance(x, str):
                return [int(t) for t in ast.literal_eval(x)]
            return [int(t) for t in list(x)]

        ids_list = _to_int_list(row["input_ids"])
        ids = torch.tensor(ids_list, dtype=torch.long)[:-1][: self.max_length]
        lp_list = _to_int_list(row["label_positions"])
        label_positions = [p - 1 for p in lp_list if 0 <= p - 1 < len(ids)]

        # Compute bin labels for each position
        bin_labels = []
        total_length = len(ids)
        reward_value = float(row[self.reward_column])
        # Clamp to [0,1] to be safe
        reward_value = max(0.0, min(1.0, reward_value))
        for pos in label_positions:
            tokens_to_completion = total_length - pos - 1
            bin_idx = self._get_bin_idx(tokens_to_completion, reward_value)
            bin_labels.append(bin_idx)

        return {
            "input_ids": ids,
            "label_positions": label_positions,
            "bin_labels": bin_labels,
            "num_bins": self.num_bins,
        }

    @staticmethod
    def collate_fn(batch):
        max_len = max(s["input_ids"].size(0) for s in batch)
        return {
            "input_ids": torch.stack(
                [F.pad(s["input_ids"], (0, max_len - s["input_ids"].size(0))) for s in batch]
            ),
            "label_positions": [s["label_positions"] for s in batch],
            "bin_labels": [s["bin_labels"] for s in batch],
            "num_bins": batch[0]["num_bins"],
        }


def compute_loss(model, batch, distribution_token_id, num_bins):
    """Compute training loss for the joint distribution prediction.

    The model learns to predict P(reward, tokens_remaining) at each position,
    enabling computation of expected values during inference.

    In this baseline, the transformer body is frozen and only the LM head is trained.
    No KL loss is used.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    # Forward pass with hidden states only to avoid materializing full [B,S,V] logits
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # [B, S, E]

    # Access the underlying lm_head even if wrapped (DDP/compiled)
    tgt = model.module if hasattr(model, "module") else model
    if hasattr(tgt, "_orig_mod"):
        lm_module = tgt._orig_mod
    else:
        lm_module = tgt
    lm_head = lm_module.lm_head if hasattr(lm_module, "lm_head") else lm_module.get_output_embeddings()

    # ===== Vectorized path: flatten all label positions across the batch =====
    all_b, all_pos, all_labels = [], [], []
    for i, (pos_list, label_list) in enumerate(zip(batch["label_positions"], batch["bin_labels"])):
        for p, l in zip(pos_list, label_list):
            if 0 <= p < hidden_states.size(1):
                all_b.append(i)
                all_pos.append(p)
                all_labels.append(l)
    P = len(all_pos)
    if P == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero

    b_idx = torch.as_tensor(all_b, device=device, dtype=torch.long)
    s_idx = torch.as_tensor(all_pos, device=device, dtype=torch.long)
    labels_tensor = torch.as_tensor(all_labels, device=device, dtype=torch.long)  # [P]
    h_flat = hidden_states[b_idx, s_idx, :]  # [P, E]

    # Joint distribution prediction loss (fully vectorized over all P)
    weight_bins = lm_head.weight[distribution_token_id : distribution_token_id + num_bins]
    bias_bins = (
        lm_head.bias[distribution_token_id : distribution_token_id + num_bins]
        if getattr(lm_head, "bias", None) is not None
        else None
    )
    logits_bins = F.linear(h_flat, weight_bins, bias_bins)  # [P, num_bins]
    distribution_loss = F.cross_entropy(logits_bins, labels_tensor, reduction="mean")

    kl_loss = torch.tensor(0.0, device=device)
    total_loss = distribution_loss

    return total_loss, kl_loss, distribution_loss


def print_trainable(model):
    print("\n=== Trainable parameters AFTER wrapping ===")
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n:<60} {p.numel():>10} elements")
            total += p.numel()
    print(f"TOTAL trainable elements: {total}\n")


def freeze_backbone_keep_lm_head_trainable(model):
    """Freeze all parameters except the LM head / output embeddings."""
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    tgt = model
    if hasattr(tgt, "_orig_mod"):
        tgt = tgt._orig_mod

    # Unfreeze LM head or output embeddings
    if hasattr(tgt, "lm_head"):
        for p in tgt.lm_head.parameters():
            p.requires_grad = True
    else:
        head = tgt.get_output_embeddings()
        for p in head.parameters():
            p.requires_grad = True


def train(
    model,
    dataset,
    distribution_token_id,
    num_bins,
    weights_path,
    collate_fn,
    shuffle_data,
    seed,
    dtype,
    compile_mode,
    num_epochs,
    batch_size,
    gradient_accumulation_steps,
    learning_rate,
    min_learning_rate,
    warmup_ratio,
    weight_decay,
    beta_1,
    beta_2,
    grad_clip,
    wandb_project,
    dist_backend,
    max_steps=-1,
    visualization_freq=100,
):

    # Setup distributed training
    distributed = int(os.environ.get("RANK", -1)) != -1
    if distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world, master = 1, True

    torch.manual_seed(seed)

    # Setup data loader
    sampler = DistributedSampler(dataset, shuffle=shuffle_data) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle_data and sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
    )

    # Initialize wandb
    if master and wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"run_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
                "dist_backend": dist_backend,
                "distribution_token_id": distribution_token_id,
                "num_bins": num_bins,
                "num_reward_states": dataset.num_reward_states,
                "max_steps": max_steps,
                "frozen_backbone": True,
                "joint_distribution_bins": f"{dataset.num_length_bins} length × {dataset.num_reward_states} reward",
                "reward_values": dataset.reward_values,
            },
        )

    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[device])

    if master:
        print_trainable(model)

    if compile_mode in {"default", "reduce-overhead", "max-autotune"}:
        model = torch.compile(model, mode=compile_mode)
    model.train()

    # Setup optimizer and learning rate schedule
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(beta_1, beta_2),
        weight_decay=weight_decay,
    )
    scaler = torch.GradScaler(enabled=(dtype == "float16"))

    total_iters = (num_epochs * len(loader)) // gradient_accumulation_steps
    warmup_iters = int(warmup_ratio * total_iters) if total_iters > 0 else 0

    def lr_schedule(i):
        if warmup_iters > 0 and i < warmup_iters:
            return learning_rate * i / warmup_iters
        if total_iters <= warmup_iters:
            return min_learning_rate
        progress = (i - warmup_iters) / (total_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_learning_rate)

    # Training loop
    global_step = 0
    accum_losses = {"total": 0.0, "kl": 0.0, "distribution": 0.0}

    for epoch in range(num_epochs):
        if distributed:
            sampler.set_epoch(epoch)

        for it, batch in enumerate(loader):
            update = (it + 1) % gradient_accumulation_steps == 0
            if distributed and isinstance(model, DDP):
                model.require_backward_grad_sync = update

            # Forward pass
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=getattr(torch, dtype)):
                total_loss, kl_loss, distribution_loss = compute_loss(
                    model, batch, distribution_token_id, batch["num_bins"]
                )
                total_loss = total_loss / gradient_accumulation_steps
                kl_loss = kl_loss / gradient_accumulation_steps
                distribution_loss = distribution_loss / gradient_accumulation_steps

            scaler.scale(total_loss).backward()

            # Accumulate losses for logging
            loss_vals = [l.detach().clone() for l in [total_loss, kl_loss, distribution_loss]]
            if distributed:
                for lv in loss_vals:
                    dist.all_reduce(lv)
                    lv /= world

            for name, val in zip(["total", "kl", "distribution"], loss_vals):
                accum_losses[name] += val.item()

            if not update:
                continue

            # Optimizer step
            scaler.unscale_(optimizer)
            if grad_clip:
                # Clip only required grads to reduce memory pressure
                params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                clip_grad_norm_(params, grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            lr = lr_schedule(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Log metrics
            if master and wandb_project:
                wandb.log({f"train/{k}_loss": v for k, v in accum_losses.items()} | {"lr": lr, "step": global_step})
                accum_losses = {k: 0.0 for k in accum_losses}

            # Visualize predictions occasionally
            if master and global_step % visualization_freq == 0:
                print(f"\n{'=' * 80}")
                print(f"JOINT DISTRIBUTION VISUALIZATION - Step {global_step}")
                print(f"{'=' * 80}")

                # Get predictions for the current batch (first and last positions)
                position_to_probs = visualize_predictions(
                    model, batch, distribution_token_id, num_bins, dataset.length_bins, device
                )

                if not position_to_probs:
                    print("No valid predictions to visualize in this batch.")
                else:
                    for label, (pred_probs, gt_probs) in position_to_probs.items():
                        try:
                            pred_grid = np.array(pred_probs).reshape(
                                dataset.num_reward_states, dataset.num_length_bins
                            )
                            gt_grid = np.array(gt_probs).reshape(
                                dataset.num_reward_states, dataset.num_length_bins
                            )
                            log_joint_distribution_grid(
                                pred_grid,
                                dataset.length_bins,
                                dataset.num_length_bins,
                                dataset.num_reward_states,
                                dataset.reward_values,
                                title_prefix=f"Predicted ({label})",
                            )
                            log_joint_distribution_grid(
                                gt_grid,
                                dataset.length_bins,
                                dataset.num_length_bins,
                                dataset.num_reward_states,
                                dataset.reward_values,
                                title_prefix=f"Ground Truth ({label})",
                            )
                        except Exception:
                            log_prediction_distributions(
                                pred_probs,
                                gt_probs,
                                dataset.length_bins,
                                dataset.num_length_bins,
                                dataset.num_reward_states,
                                dataset.reward_values,
                            )

                print(f"{'=' * 80}\n")

            if max_steps > 0 and global_step >= max_steps:
                break

        if max_steps > 0 and global_step >= max_steps:
            break

    # Save model
    if master:
        tgt = model.module if hasattr(model, "module") else model
        if hasattr(tgt, "save_pretrained"):
            tgt.save_pretrained(weights_path)
        else:
            torch.save(tgt.state_dict(), weights_path)
        if wandb_project:
            wandb.finish()

    if distributed:
        dist.barrier()


def main_worker(local_rank, world_size, cfg):
    os.environ.update(
        WORLD_SIZE=str(world_size),
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    # Setup model
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device).gradient_checkpointing_enable()
    model.config.use_cache = False

    # Freeze everything except LM head (head-only baseline)
    freeze_backbone_keep_lm_head_trainable(model)

    # Load dataset to get num_bins
    dataset = ZIPDataset(
        cfg.data_path,
        max_length=cfg.max_length,
        thinking_only=cfg.thinking_only,
        thinking_token_id=cfg.thinking_token_id,
        reward_values=(cfg.reward_values if cfg.reward_values is not None else None),
        label_column=cfg.label_column,
    )
    num_bins = dataset.num_bins

    # Save tokenizer on first rank
    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        tokenizer.save_pretrained(cfg.weights_path)

    # Train model
    train(
        model,
        dataset,
        cfg.distribution_token_id,
        num_bins,
        cfg.weights_path,
        ZIPDataset.collate_fn,
        True,
        42,
        "bfloat16",
        cfg.compile_mode,
        cfg.num_epochs,
        cfg.batch_size,
        cfg.gradient_accumulation_steps,
        cfg.learning_rate,
        0.0,
        cfg.warmup_ratio,
        cfg.weight_decay,
        0.9,
        0.95,
        1.0,
        cfg.wandb_project,
        cfg.dist_backend,
        cfg.max_steps,
        cfg.visualization_freq,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--weights_path", default="models/zip_model")
    p.add_argument("--data_path", default="data/data.parquet")

    # Token ID for distribution prediction
    p.add_argument(
        "--distribution_token_id",
        type=int,
        default=151669,
        help="Starting token ID for joint distribution prediction P(reward, tokens_remaining)",
    )
    p.add_argument(
        "--thinking_only",
        action="store_true",
        help="Only train on samples that contain thinking token",
    )
    p.add_argument(
        "--thinking_token_id",
        type=int,
        default=151667,
        help="Token ID for detecting thinking/reasoning samples",
    )
    p.add_argument(
        "--reward_values",
        type=float,
        nargs="+",
        default=None,
        help="Midpoints for reward bins. If omitted: [0,1] for correctness; 7 bins for value.",
    )
    p.add_argument(
        "--label_column",
        choices=["auto", "correct", "value"],
        default="correct",
        help="Which column to use for reward supervision. 'auto' prefers 'correct' if present.",
    )

    # Training parameters
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=32_768)
    p.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If >0, stop training after this many optimizer steps",
    )
    p.add_argument(
        "--visualization_freq",
        type=int,
        default=100,
        help="Frequency (in steps) to display prediction visualizations",
    )

    # Model configuration
    p.add_argument(
        "--compile_mode",
        default="none",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument("--wandb_project", default="zip_training")
    p.add_argument(
        "--dist-backend",
        choices=["ddp"],
        default="ddp",
        help="Distributed training backend",
    )

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
