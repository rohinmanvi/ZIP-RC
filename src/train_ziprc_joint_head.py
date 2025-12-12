#!/usr/bin/env python3
"""
Fine-tune a model for ZIP (Zero-overhead Inference-time Prediction) with DDP support.

ZIP trains models to predict the joint distribution of:
1. Tokens to completion: binned distribution from current position (0-255, 256-511, 512-1023, ..., 16384-32767)
2. Reward: scalar reward value binned into discrete states (e.g., correctness: 0=incorrect, 1=correct)

The training uses a single cross-entropy loss over the joint distribution bins,
plus optional KL divergence from a reference model to maintain original capabilities.

This joint distribution approach enables:
- Richer probabilistic reasoning for pruning decisions during inference
- Computing expected rewards and tokens for individual samples
- Optimizing group objectives across multiple parallel samples

Example usage:
    # Binary correctness (default)
    python train_ziprc_joint_head.py --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                                     --weights_path models/zip_model --data_path data/data.parquet

    # Custom reward values (e.g., 5-level quality score)
    python train_ziprc_joint_head.py --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                                     --weights_path models/zip_model --data_path data/data.parquet \
                                     --reward_values 0.0 0.25 0.5 0.75 1.0
"""

from __future__ import annotations
import argparse, math, os, time, wandb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from ziprc_training_visualization import (
    visualize_predictions,
    log_prediction_distributions,
    log_joint_distribution_grid,
)
from ziprc_dataset import ZIPDataset

def _logsumexp_linear_seqchunk(h: torch.Tensor,
                               W: torch.Tensor,
                               b: torch.Tensor | None,
                               skip: tuple[int, int] | None = None,
                               seq_chunk: int = 512) -> torch.Tensor:
    """
    Exact logsumexp(h @ W^T (+b)) over the full vocab (excluding [skip[0]:skip[1]))
    computed in chunks of rows of `h` (sequence dimension), so we never form [P,V] at once.

    This keeps the computation in the autograd graph (exact gradients).
    """
    P = h.size(0)
    s0, s1 = skip if skip is not None else (0, 0)
    out_pieces = []
    for start in range(0, P, seq_chunk):
        end = min(start + seq_chunk, P)
        h_chunk = h[start:end]                                   # [p, E]
        logits = F.linear(h_chunk, W, b)                         # [p, V]
        if s1 > s0:
            logits[:, s0:s1] = -float("inf")                     # mask distribution bins
        # Compute in float32 for stability; keeps exact grads.
        out_pieces.append(torch.logsumexp(logits.to(torch.float32), dim=-1))  # [p]
        # logits tensor will be freed per-iteration; saved activations are per-chunk.
    return torch.cat(out_pieces, dim=0)                          # [P]


def compute_loss(model, batch, distribution_token_id, num_bins,
                 ref_model=None, kl_coefficient=1.0):
    """Compute training loss for the joint distribution prediction.
    
    The model learns to predict P(reward, tokens_remaining) at each position,
    enabling computation of expected values during inference.
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
                all_b.append(i); all_pos.append(p); all_labels.append(l)
    P = len(all_pos)
    if P == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero

    b_idx = torch.as_tensor(all_b, device=device, dtype=torch.long)
    s_idx = torch.as_tensor(all_pos, device=device, dtype=torch.long)
    labels_tensor = torch.as_tensor(all_labels, device=device, dtype=torch.long)  # [P]
    h_student_flat = hidden_states[b_idx, s_idx, :]  # [P, E]

    # ---- KL divergence: teacher sparse (topk + min_p), student FULL log-softmax ----
    kl_loss = torch.tensor(0.0, device=device)
    if ref_model:
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=input_ids, output_hidden_states=True)
            ref_hidden_states = ref_outputs.hidden_states[-1]  # [B, S, E]
            ref_tgt = ref_model.module if hasattr(ref_model, "module") else ref_model
            ref_lm_module = getattr(ref_tgt, "_orig_mod", ref_tgt)
            ref_lm_head = (ref_lm_module.lm_head if hasattr(ref_lm_module, "lm_head")
                           else ref_lm_module.get_output_embeddings())

        h_teacher_flat = ref_hidden_states[b_idx, s_idx, :]  # [P, E]

        # Teacher logits over full vocab (mask out joint-distribution bins)
        teacher_full_logits = F.linear(h_teacher_flat, ref_lm_head.weight, ref_lm_head.bias)  # [P, V]
        teacher_full_logits[:, distribution_token_id:distribution_token_id + num_bins] = -float("inf")

        # Keep top-k = 32
        K = 32
        teacher_topk_vals, teacher_topk_idx = torch.topk(teacher_full_logits, k=K, dim=-1)  # [P,K], [P,K]
        del teacher_full_logits

        # Teacher distro over top-K only (no min_p): renormalize across K
        t_logp = F.log_softmax(teacher_topk_vals, dim=-1)                          # [P,K]
        t_probs = t_logp.exp()                                                     # [P,K]

        # Student logits only on teacher’s sparse indices
        E = h_student_flat.size(-1)
        W_sel = lm_head.weight.index_select(0, teacher_topk_idx.reshape(-1)).view(P, K, E)
        s_topk_logits = torch.bmm(W_sel, h_student_flat.unsqueeze(-1)).squeeze(-1)  # [P, K]
        if getattr(lm_head, "bias", None) is not None:
            b_sel = lm_head.bias.index_select(0, teacher_topk_idx.reshape(-1)).view_as(s_topk_logits)
            s_topk_logits = s_topk_logits + b_sel

        # logZ: exact logsumexp over full vocab, excluding your distribution bins
        logZ = _logsumexp_linear_seqchunk(
            h_student_flat,
            lm_head.weight,
            (lm_head.bias if getattr(lm_head, "bias", None) is not None else None),
            skip=(distribution_token_id, distribution_token_id + num_bins),
            seq_chunk=512,
        )  # [P]

        # log Q_full at sparse indices (no [P,V] tensor)
        s_logq_sel = s_topk_logits - logZ.unsqueeze(1)  # [P,K]

        # KL(P_trunc || Q_full) over top-K indices
        kl_terms = t_probs * (t_logp - s_logq_sel)
        kl_loss = kl_terms.sum(dim=-1).mean()

    # ---- Joint distribution prediction loss (fully vectorized over all P) ----
    weight_bins = lm_head.weight[distribution_token_id:distribution_token_id + num_bins]
    bias_bins = (lm_head.bias[distribution_token_id:distribution_token_id + num_bins]
                 if getattr(lm_head, "bias", None) is not None else None)
    logits_bins = F.linear(h_student_flat, weight_bins, bias_bins)  # [P, num_bins]
    distribution_loss = F.cross_entropy(logits_bins, labels_tensor, reduction='mean')  # == sum/P
    
    total_loss = kl_coefficient * kl_loss + distribution_loss
    
    return total_loss, kl_loss, distribution_loss

def print_trainable(model):
    print("\n=== Trainable parameters AFTER wrapping ===")
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n:<60} {p.numel():>10} elements")
            total += p.numel()
    print(f"TOTAL trainable elements: {total}\n")

def train(model, dataset, distribution_token_id, num_bins, weights_path,
          collate_fn, shuffle_data, seed, dtype, compile_mode, num_epochs, batch_size,
          gradient_accumulation_steps, learning_rate, min_learning_rate, warmup_ratio,
          weight_decay, beta_1, beta_2, grad_clip, wandb_project, dist_backend,
          ref_model=None, kl_coefficient=1.0, max_steps=-1,
          visualization_freq=100):
    
    # Setup distributed training
    distributed = int(os.environ.get("RANK", -1)) != -1
    if distributed:
        dist.init_process_group("nccl")
        rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world, master = 1, True

    torch.manual_seed(seed)

    # Setup data loader
    sampler = DistributedSampler(dataset, shuffle=shuffle_data) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle_data and sampler is None),
                        sampler=sampler, collate_fn=collate_fn)

    # Initialize wandb
    if master and wandb_project:
        wandb.init(project=wandb_project, name=f"run_{int(time.time())}", config={
            "num_epochs": num_epochs, "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate, "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay, "compile_mode": compile_mode,
            "dist_backend": dist_backend, "distribution_token_id": distribution_token_id,
            "num_bins": num_bins, "num_reward_states": dataset.num_reward_states,
            "kl_coefficient": kl_coefficient, "max_steps": max_steps,
            "joint_distribution_bins": f"{dataset.num_length_bins} length × {dataset.num_reward_states} reward",
            "reward_values": dataset.reward_values
        })

    model = model.to(device)
    if ref_model: ref_model = ref_model.to(device).eval()
    
    if distributed:
        model = DDP(model, device_ids=[device])
    
    if master: print_trainable(model)
    
    if compile_mode in {"default", "reduce-overhead", "max-autotune"}:
        model = torch.compile(model, mode=compile_mode)
    model.train()

    # Setup optimizer and learning rate schedule
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], 
                      lr=learning_rate, betas=(beta_1, beta_2), weight_decay=0.0)
    scaler = torch.GradScaler(enabled=(dtype == "float16"))
    
    total_iters = (num_epochs * len(loader)) // gradient_accumulation_steps
    warmup_iters = int(warmup_ratio * total_iters)
    
    def lr_schedule(i):
        if i < warmup_iters:
            return learning_rate * i / warmup_iters
        progress = (i - warmup_iters) / (total_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_learning_rate)

    # Training loop
    global_step = 0
    accum_losses = {"total": 0.0, "kl": 0.0, "distribution": 0.0}
    
    for epoch in range(num_epochs):
        if distributed: sampler.set_epoch(epoch)
        
        for it, batch in enumerate(loader):
            update = (it + 1) % gradient_accumulation_steps == 0
            if distributed and isinstance(model, DDP):
                model.require_backward_grad_sync = update

            # Forward pass
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=getattr(torch, dtype)):
                losses = compute_loss(model, batch, distribution_token_id, batch["num_bins"],
                                    ref_model, kl_coefficient)
                total_loss, kl_loss, distribution_loss = [l / gradient_accumulation_steps for l in losses]
            
            scaler.scale(total_loss).backward()
            
            # Accumulate losses for logging
            loss_vals = [l.detach().clone() for l in [total_loss, kl_loss, distribution_loss]]
            if distributed:
                for lv in loss_vals:
                    dist.all_reduce(lv)
                    lv /= world
            
            for name, val in zip(["total", "kl", "distribution"], loss_vals):
                accum_losses[name] += val.item()

            if not update: continue

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
            for g in optimizer.param_groups: g["lr"] = lr
            
            # Log metrics
            if master and wandb_project:
                wandb.log({
                    f"train/{k}_loss": v for k, v in accum_losses.items()
                } | {"lr": lr, "step": global_step})
                accum_losses = {k: 0.0 for k in accum_losses}
            
            # Visualize predictions occasionally
            if master and global_step % visualization_freq == 0:
                print(f"\n{'='*80}")
                print(f"JOINT DISTRIBUTION VISUALIZATION - Step {global_step}")
                print(f"{'='*80}")
                
                # Get predictions for the current batch (first and last positions)
                position_to_probs = visualize_predictions(
                    model, batch, distribution_token_id, num_bins, 
                    dataset.length_bins, device
                )
                
                if not position_to_probs:
                    print("No valid predictions to visualize in this batch.")
                else:
                    for label, (pred_probs, gt_probs) in position_to_probs.items():
                        try:
                            import numpy as np
                            pred_grid = np.array(pred_probs).reshape(dataset.num_reward_states, dataset.num_length_bins)
                            gt_grid = np.array(gt_probs).reshape(dataset.num_reward_states, dataset.num_length_bins)
                            log_joint_distribution_grid(pred_grid, dataset.length_bins, dataset.num_length_bins, dataset.num_reward_states, dataset.reward_values, title_prefix=f"Predicted ({label})")
                            log_joint_distribution_grid(gt_grid, dataset.length_bins, dataset.num_length_bins, dataset.num_reward_states, dataset.reward_values, title_prefix=f"Ground Truth ({label})")
                        except Exception:
                            log_prediction_distributions(
                                pred_probs, gt_probs, dataset.length_bins, dataset.num_length_bins,
                                dataset.num_reward_states, dataset.reward_values
                            )
                
                print(f"{'='*80}\n")
            
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
        if wandb_project: wandb.finish()
    
    if distributed: dist.barrier()

def main_worker(local_rank, world_size, cfg):
    os.environ.update(
        WORLD_SIZE=str(world_size), RANK=str(local_rank), LOCAL_RANK=str(local_rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29500"
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
    
    # Setup reference model if needed
    ref_model = None
    if cfg.kl_coefficient > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    
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
    train(model, dataset, cfg.distribution_token_id, num_bins,
          cfg.weights_path, ZIPDataset.collate_fn, True, 42, "bfloat16", cfg.compile_mode,
          cfg.num_epochs, cfg.batch_size, cfg.gradient_accumulation_steps, cfg.learning_rate,
          0.0, cfg.warmup_ratio, cfg.weight_decay, 0.9, 0.95, 1.0, cfg.wandb_project,
          cfg.dist_backend, ref_model, cfg.kl_coefficient, cfg.max_steps,
          cfg.visualization_freq)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--weights_path", default="models/zip_model")
    p.add_argument("--data_path", default="data/data.parquet")
    
    # Token ID for distribution prediction
    p.add_argument("--distribution_token_id", type=int, default=151669, 
                   help="Starting token ID for joint distribution prediction P(reward, tokens_remaining)")
    p.add_argument("--thinking_only", action="store_true", 
                   help="Only train on samples that contain thinking token")
    p.add_argument("--thinking_token_id", type=int, default=151667,
                   help="Token ID for detecting thinking/reasoning samples")
    p.add_argument("--reward_values", type=float, nargs="+", default=None,
                   help="Midpoints for reward bins. If omitted: [0,1] for correctness; 7 bins for value.")
    p.add_argument("--label-column", choices=["auto","correct","value"], default="correct",
                   help="Which column to use for reward supervision. 'auto' prefers 'correct' if present.")
    
    # Training parameters
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=32_768)
    p.add_argument("--max_steps", type=int, default=-1, help="If >0, stop training after this many optimizer steps")
    p.add_argument("--visualization_freq", type=int, default=100, help="Frequency (in steps) to display prediction visualizations")

    
    # Model configuration
    p.add_argument("--compile_mode", default="none", choices=["none", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--wandb_project", default="zip_training")
    p.add_argument("--dist-backend", choices=["ddp"], default="ddp", help="Distributed training backend")
    
    # Loss coefficient
    p.add_argument("--kl_coefficient", type=float, default=10.0, help="Coefficient for KL divergence loss from reference model")
    
    return p.parse_args()

def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))

if __name__ == "__main__":
    main()