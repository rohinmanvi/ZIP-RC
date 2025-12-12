#!/usr/bin/env python3
"""
Label sequences with a scalar 'value' by running a ZIP joint-distribution model
(trained on correctness) and averaging the last-K positionwise expected rewards.

- Multi-GPU data-parallel via torch.multiprocessing + torch.distributed (no vLLM).
- Reads a parquet with at least:  input_ids, label_positions
- Writes a new parquet identical to input but with a 'value' column in [0,1].

Expected reward at position t:
  p_joint = softmax(logits_bins[t]) over num_bins (= num_length_bins * num_reward_states)
  p_reward = sum over length bins
  value_t = dot(p_reward, reward_values)   # reward_values e.g. [0.0, 1.0]
Row value:
  mean(value_t for the last K labeled positions), with K=512 by default.
"""

from __future__ import annotations
import argparse, ast, os, time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# Utilities
# ---------------------------

def _to_int_list(x) -> List[int]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(t) for t in x]
    if isinstance(x, str):
        # Stored as stringified list
        return [int(t) for t in ast.literal_eval(x)]
    # Fallback: iterable of scalars
    return [int(t) for t in list(x)]

class InferenceDataset(Dataset):
    """Mimics training-time preprocessing so label positions align exactly."""
    def __init__(self, df: pd.DataFrame, max_length: int = 32_768):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        if "input_ids" not in df.columns or "label_positions" not in df.columns:
            missing = {"input_ids", "label_positions"} - set(df.columns)
            raise ValueError(f"Missing required column(s): {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        ids_list = _to_int_list(row["input_ids"])
        # Match train_ziprc_joint_head.py: drop final token, truncate to max_length
        ids = torch.tensor(ids_list, dtype=torch.long)[:-1][: self.max_length]

        lp_list = _to_int_list(row["label_positions"])
        # Match train_ziprc_joint_head.py: shift positions by -1 and keep those within sequence
        label_positions = [p - 1 for p in lp_list if 0 <= p - 1 < len(ids)]
        return {"input_ids": ids, "label_positions": label_positions, "row_idx": int(idx)}

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(item["input_ids"].size(0) for item in batch) if batch else 0
        input_ids = torch.stack([
            F.pad(item["input_ids"], (0, max_len - item["input_ids"].size(0)))
            for item in batch
        ]) if batch else torch.empty(0, dtype=torch.long)
        return {
            "input_ids": input_ids,                                  # [B, S]
            "label_positions": [item["label_positions"] for item in batch],
            "row_idx": [item["row_idx"] for item in batch],
        }

# ---------------------------
# Core computation
# ---------------------------

def expected_values_last_k(
    hidden_states: torch.Tensor,                 # [B, S, E]
    label_positions: List[List[int]],
    lm_head,
    distribution_token_id: int,
    num_length_bins: int,
    reward_values: torch.Tensor,                 # [R] on device, dtype float32
    last_k: int = 512,
    chunk_size: int = 512
) -> List[float]:
    """
    Compute the average of the last-K expected rewards for each sequence in the batch.

    Returns: list of floats (len = batch size), each in [0,1] or None if no positions.
    """
    device = hidden_states.device
    dtype_logits = hidden_states.dtype

    # Determine number of bins and slice lm_head weights
    num_reward_states = int(reward_values.numel())
    num_bins = num_length_bins * num_reward_states

    # Access lm_head safely (also works if compiled/wrapped)
    tgt = lm_head
    weight = tgt.weight         # [V, E]
    bias = getattr(tgt, "bias", None)  # [V]

    # Pre-slice the joint-distribution rows for efficiency
    weight_bins = weight[distribution_token_id: distribution_token_id + num_bins]  # [num_bins, E]
    bias_bins = None if bias is None else bias[distribution_token_id: distribution_token_id + num_bins]  # [num_bins]

    batch_size = hidden_states.size(0)
    out_values: List[float] = []

    for i in range(batch_size):
        pos = [p for p in label_positions[i] if 0 <= p < hidden_states.size(1)]
        if not pos:
            out_values.append(float("nan"))
            continue

        pos.sort()
        pos = pos[-last_k:]  # last-K positions

        vals_i = []
        # Process positions in chunks to keep memory bounded
        for start in range(0, len(pos), chunk_size):
            chunk_pos = pos[start:start + chunk_size]  # [P]
            # [P, E]
            h = hidden_states[i, chunk_pos, :]
            # Compute logits only over the joint-distribution slice: [P, num_bins]
            chunk_logits = F.linear(h, weight_bins, bias_bins).to(dtype=torch.float32)
            # Convert to probabilities over the joint bins
            probs = F.softmax(chunk_logits, dim=-1)                     # [P, R*L]
            # Reshape to [P, R, L] and marginalize over length bins
            probs = probs.view(-1, num_reward_states, num_length_bins)  # [P, R, L]
            p_reward = probs.sum(dim=-1)                                # [P, R]
            # Expected correctness/value per position: [P]
            exp_chunk = torch.matmul(p_reward, reward_values)           # [P]
            vals_i.append(exp_chunk)

        vals_i = torch.cat(vals_i, dim=0) if vals_i else torch.empty(0, device=device)
        if vals_i.numel() == 0:
            out_values.append(float("nan"))
        else:
            out_values.append(float(vals_i.mean().item()))

    return out_values

# ---------------------------
# Distributed runner
# ---------------------------

def worker(rank: int, world_size: int, args):
    # Do NOT set MASTER_PORT per-rank; set it once in main()
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Select device for this rank and make it current
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        device_obj = device              # torch.device('cuda', index=rank)
        dev_index = device.index         # int
    else:
        device = torch.device("cpu")
        device_obj = None
        dev_index = None

    torch.backends.cuda.matmul.allow_tf32 = True

    # --- init PG with explicit device when NCCL (PyTorch >= 2.4 expects torch.device) ---
    init_kwargs = dict(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl" and torch.cuda.is_available():
        init_kwargs["device_id"] = device_obj  # torch.device, not int

    try:
        dist.init_process_group(**init_kwargs)
    except TypeError:
        # Older torch that doesn't accept device_id at all
        init_kwargs.pop("device_id", None)
        dist.init_process_group(**init_kwargs)

    # Helper that ensures NCCL barrier uses the same device as the rank
    def pg_barrier():
        if backend == "nccl" and torch.cuda.is_available():
            try:
                dist.barrier(device_ids=[dev_index])  # list[int] in PyTorch >= 2.4
            except TypeError:
                # Older torch: device_ids arg not supported
                dist.barrier()
        else:
            dist.barrier()

    if rank == 0:
        print("===============================================")
        print("ZIP: Label with joint-distribution (no vLLM)")
        print(f"  Model:   {args.model}")
        print(f"  Input:   {args.in_parquet}")
        print(f"  Output:  {args.out_parquet}")
        print(f"  GPUs:    {world_size}")
        print(f"  dist_token_id: {args.distribution_token_id}")
        print(f"  reward_values: {args.reward_values}")
        print(f"  num_length_bins: {args.num_length_bins}")
        print(f"  last_k:  {args.last_k}")
        print(f"  batch:   {args.batch_size}")
        print(f"  Device map: rank {rank} -> cuda:{dev_index}" if torch.cuda.is_available() else "CPU mode")
        print("===============================================", flush=True)

    # Load input table once per rank
    df = pq.read_table(args.in_parquet).to_pandas()
    df = df.reset_index(drop=True)

    dataset = InferenceDataset(df, max_length=args.max_length)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=InferenceDataset.collate_fn,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Load model (no vLLM), flash_attn2 if available; fall back gracefully
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    def _load_model(attn_impl: str | None):
        kw = dict(torch_dtype=torch_dtype)
        if attn_impl:
            kw["attn_implementation"] = attn_impl
        return AutoModelForCausalLM.from_pretrained(args.model, **kw).to(device).eval()

    try:
        model = _load_model("flash_attention_2")
    except Exception as e:
        if rank == 0:
            print(f"flash_attention_2 not available ({e}); falling back to default attention.", flush=True)
        model = _load_model(None)

    # Access lm_head robustly
    tgt = model.module if hasattr(model, "module") else model
    base = getattr(tgt, "_orig_mod", tgt)
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()

    # Prepare constants
    reward_values = torch.tensor(args.reward_values, dtype=torch.float32, device=device)
    num_reward_states = reward_values.numel()
    num_bins = int(args.num_length_bins * num_reward_states)

    assert args.distribution_token_id + num_bins <= lm_head.weight.size(0), \
        f"Distribution token slice exceeds vocab size. Check distribution_token_id / num_bins. " \
        f"Token range: {args.distribution_token_id} to {args.distribution_token_id + num_bins}, Vocab size: {lm_head.weight.size(0)}"

    # Inference loop
    shard_rows = []
    t0 = time.time()
    seen = 0

    with torch.inference_mode():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)  # [B, S]
            outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            h_last = outputs.hidden_states[-1]  # [B, S, E]

            vals = expected_values_last_k(
                hidden_states=h_last,
                label_positions=batch["label_positions"],
                lm_head=lm_head,
                distribution_token_id=args.distribution_token_id,
                num_length_bins=args.num_length_bins,
                reward_values=reward_values,
                last_k=args.last_k,
                chunk_size=args.pos_chunk_size,
            )

            for ridx, v in zip(batch["row_idx"], vals):
                # Keep NaN as None in parquet
                shard_rows.append((int(ridx), (None if (v != v) else float(max(0.0, min(1.0, v))))))

            seen += len(vals)
            if step % max(1, args.log_every) == 0:
                elapsed = time.time() - t0
                rate = seen / max(1e-9, elapsed)
                print(f"[rank {rank}] processed {seen} rows  ({rate:.1f} rows/s)", flush=True)

    # Clean up resources before barrier to reduce CUDA activity
    del loader
    del dataset
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Write this rank's partial results
    part_path = f"{args.out_parquet}.part{rank}"
    part_tbl = pa.Table.from_pandas(pd.DataFrame(shard_rows, columns=["index", "value"]))
    pq.write_table(part_tbl, part_path)

    pg_barrier()

    # Rank 0 merges
    if rank == 0:
        parts = [f"{args.out_parquet}.part{r}" for r in range(world_size)]
        mapping: Dict[int, float | None] = {}
        for p in parts:
            pt = pq.read_table(p).to_pandas()
            for i, v in zip(pt["index"].tolist(), pt["value"].tolist()):
                mapping[int(i)] = (None if pd.isna(v) else float(v))

        out_df = df.copy()
        out_df["value"] = [mapping.get(i, None) for i in range(len(out_df))]
        
        # Add some statistics if ground truth is available
        if "correct" in out_df.columns:
            valid_mask = out_df["value"].notna()
            if valid_mask.any():
                y_true = out_df.loc[valid_mask, "correct"].astype(float).values
                y_pred = out_df.loc[valid_mask, "value"].values
                # Simple correlation and accuracy at 0.5 threshold
                corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
                acc_at_half = np.mean((y_pred >= 0.5).astype(float) == y_true) * 100
                print(f"  Correlation with ground truth: {corr:.3f}")
                print(f"  Accuracy at threshold 0.5: {acc_at_half:.1f}%")
        
        pq.write_table(pa.Table.from_pandas(out_df), args.out_parquet)
        for p in parts:
            try:
                os.remove(p)
            except OSError:
                pass

        total = len(out_df)
        n_nan = sum(v is None for v in out_df["value"].tolist())
        print("===============================================")
        print(f"✓ Wrote labeled parquet: {args.out_parquet}")
        print(f"  rows: {total} | missing values: {n_nan}")
        print(f"  elapsed: {time.time() - t0:.1f}s")
        print("===============================================", flush=True)

    try:
        pg_barrier()
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass  # Ignore cleanup errors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained joint-distribution model (save_pretrained dir).")
    p.add_argument("--in-parquet", required=True, help="Input parquet to label.")
    p.add_argument("--out-parquet", required=True, help="Output parquet with added 'value' column.")
    p.add_argument("--distribution-token-id", type=int, default=151669)
    p.add_argument("--num-length-bins", type=int, default=8,
                   help="Length bins used during training: [0-255],[256-511],...,[16384-32767].")
    p.add_argument("--reward-values", type=float, nargs="+", default=[0.0, 1.0],
                   help="Reward state midpoints corresponding to the joint bins. For correctness use [0.0, 1.0].")
    p.add_argument("--last-k", type=int, default=512, help="Average of the last-K positionwise expectations.")
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--pos-chunk-size", type=int, default=512, help="Chunk size for per-position logits.")
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def main():
    args = parse_args()
    
    # Check if we're running under torchrun (it sets LOCAL_RANK)
    if "LOCAL_RANK" in os.environ:
        # torchrun mode - it handles process spawning and sets env vars
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        worker(rank, world_size, args)
    else:
        # mp.spawn mode - we handle spawning
        world_size = torch.cuda.device_count() or 1
        
        # Set rendezvous once for all ranks
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        
        if world_size == 1:
            # Run single-process path for convenience
            worker(rank=0, world_size=1, args=args)
        else:
            mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()