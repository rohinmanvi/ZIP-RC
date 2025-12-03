#!/usr/bin/env python3
"""
Stage 2: Label generated sequences with joint-distribution probabilities.

- Extract teacher hidden states at positions preceding each generated token.
- Project to a *contiguous* joint grid starting at --distribution-token-id.
- Convert to probabilities and apply *causal* sliding-window averaging of size W.
- Downsample at --update-interval steps (e.g., every 64 tokens).
- Store flattened per-interval probs into an NPZ sidecar and offsets into a Parquet.

Output:
  Parquet: original rows + columns:
    - joint_probs_num_steps, joint_probs_offset
    - interval_positions (list[int]) [optional compact copy per row]
    - reward_values (list[float]), length_bins (list[int]), num_reward_states, num_length_bins, distribution_token_id
  NPZ: 'indices', 'num_steps', 'offsets', 'probs' (float16), 'positions'
"""

from __future__ import annotations
import argparse, os, sys, time, math, traceback
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-parquet", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--distribution-token-id", type=int, default=151669)
    p.add_argument("--num-value-bins", type=int, default=7)
    p.add_argument("--value-min", type=float, default=0.0)
    p.add_argument("--value-max", type=float, default=1.0)
    p.add_argument("--length-bins", type=str, default="0,256,512,1024,2048,4096,8192,16384,32768",
                   help="Comma-separated boundaries; N+1 edges yield N length bins")
    p.add_argument("--window-size", type=int, default=64, help="Causal sliding window size (W)")
    p.add_argument("--update-interval", type=int, default=64, help="Emit smoothed probs every K tokens")
    p.add_argument("--dtype", choices=["bfloat16","float16"], default="bfloat16")
    p.add_argument("--dp-size", type=int, default=1)
    return p.parse_args()

def _clear_env():
    for k in ["RANK","WORLD_SIZE","LOCAL_RANK","MASTER_ADDR","MASTER_PORT",
              "VLLM_DP_RANK","VLLM_DP_SIZE","VLLM_DP_MASTER_IP","VLLM_DP_MASTER_PORT","VLLM_DP_LOCAL_RANK"]:
        os.environ.pop(k, None)

def _ensure_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try: return eval(x)
        except Exception: pass
    return list(x)

@torch.no_grad()
def _forward_hidden(model, input_ids: List[int], device):
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states[-1][0]  # [S, E]

def main():
    args = parse_args()
    _clear_env()

    df = pq.read_table(args.in_parquet).to_pandas()
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=(torch.bfloat16 if args.dtype=="bfloat16" else torch.float16),
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map=None
    )
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Build grid mapping: value-major then length (contiguous block)
    length_bins = [int(x) for x in args.length_bins.split(",")]
    num_length_bins = len(length_bins) - 1
    if args.num_value_bins < 1: raise ValueError("--num-value-bins must be >=1")
    if args.num_value_bins == 1:
        reward_values = [args.value_min]
    else:
        step = (args.value_max - args.value_min) / (args.num_value_bins - 1)
        reward_values = [args.value_min + i*step for i in range(args.num_value_bins)]
    num_reward_states = len(reward_values)
    num_bins = num_reward_states * num_length_bins

    # Slice lm_head rows for the contiguous token block
    tgt = model.module if hasattr(model, "module") else model
    if hasattr(tgt, "_orig_mod"):
        tgt = tgt._orig_mod
    lm_head = tgt.lm_head if hasattr(tgt, "lm_head") else tgt.get_output_embeddings()
    W = lm_head.weight[args.distribution_token_id : args.distribution_token_id + num_bins]  # [num_bins,E]
    
    # Safety assertion: verify slice size matches expected num_bins
    assert W.shape[0] == num_bins, \
        f"lm_head slice size ({W.shape[0]}) != expected num_bins ({num_bins}). " \
        f"Check distribution_token_id, num_value_bins, and length_bins."
    b = None
    if hasattr(lm_head, "bias") and lm_head.bias is not None:
        b = lm_head.bias[args.distribution_token_id : args.distribution_token_id + num_bins]

    probs_all = []   # flattened probs for all intervals across rows
    pos_all = []     # flattened token positions (ends) for all intervals
    offsets = []
    nsteps = []
    indices = []

    t0 = time.time()
    for ridx, row in df.iterrows():
        input_ids = _ensure_list(row.get("input_ids", []))
        label_positions = _ensure_list(row.get("label_positions", []))
        if not input_ids or not label_positions:
            offsets.append(len(probs_all)); nsteps.append(0); indices.append(ridx)
            continue

        H = _forward_hidden(model, input_ids, device)  # [S,E]
        pos_list = [p-1 for p in label_positions if 0 <= p-1 < len(input_ids)]
        if not pos_list:
            offsets.append(len(probs_all)); nsteps.append(0); indices.append(ridx)
            continue

        h = H[pos_list, :].to(W.dtype)  # [T,E], T=#generated tokens
        logits = torch.nn.functional.linear(h, W, b)  # [T,num_bins]
        # probs per generated token (before smoothing)
        with torch.inference_mode():
            token_probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()  # [T, B]
        T = token_probs.shape[0]

        # Emit at causal intervals, applying causal sliding average of window W.
        # Always include: first token (1), K, 2K, ..., and final T.
        K = max(1, int(args.update_interval))
        Wsz = max(1, int(args.window_size))

        ends = set()
        if T > 0:
            ends.add(1)                    # first position
            ends.update(range(K, T+1, K))  # K, 2K, ..., nK <= T
            ends.add(T)                    # last position
        ends_sorted = sorted(ends)

        steps = []
        for end in ends_sorted:
            start = max(0, end - Wsz)
            steps.append((start, end))

        row_probs = []
        row_positions = []
        for (s,e) in steps:
            smoothed = token_probs[s:e].mean(axis=0) if e>s else token_probs[e-1]
            row_probs.append(smoothed.astype(np.float16))
            row_positions.append(e)  # number of generated tokens seen at this interval

        if row_probs:
            probs_all.extend(row_probs)
            pos_all.extend(row_positions)
            offsets.append(len(probs_all) - len(row_probs))
            nsteps.append(len(row_probs))
            indices.append(ridx)
        else:
            offsets.append(len(probs_all)); nsteps.append(0); indices.append(ridx)

        if (ridx+1) % 50 == 0:
            print(f"[label] {ridx+1}/{len(df)} rows processed in {time.time()-t0:.1f}s", flush=True)

    # Pack NPZ
    probs_arr = np.stack(probs_all, axis=0) if probs_all else np.empty((0, num_bins), dtype=np.float16)
    positions_arr = np.array(pos_all, dtype=np.int32) if pos_all else np.empty((0,), dtype=np.int32)
    indices_arr = np.array(indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int64)
    nsteps_arr = np.array(nsteps, dtype=np.int32)

    npz_path = args.out_parquet.replace(".parquet", "_joint_probs.npz")
    np.savez_compressed(
        npz_path,
        indices=indices_arr,
        offsets=offsets_arr,
        num_steps=nsteps_arr,
        probs=probs_arr,
        positions=positions_arr,
    )
    print(f"✓ wrote NPZ {npz_path}  (rows with steps: {(nsteps_arr>0).sum()})")

    # Write updated parquet with metadata columns (repeat mapping constants for downstream)
    df2 = df.copy()
    df2["joint_probs_offset"] = offsets_arr
    df2["joint_probs_num_steps"] = nsteps_arr
    df2["reward_values"] = [reward_values] * len(df2)
    df2["length_bins"] = [length_bins] * len(df2)
    df2["num_reward_states"] = num_reward_states
    df2["num_length_bins"] = num_length_bins
    df2["distribution_token_id"] = args.distribution_token_id

    pq.write_table(pa.Table.from_pandas(df2), args.out_parquet, compression="zstd")
    print(f"✓ wrote labeled parquet {args.out_parquet} with {len(df2)} rows")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)