#!/usr/bin/env python3
"""
Stage 3: Offline pruning simulation.

- Reads the labeled Parquet + NPZ from Stage 2.
- For each prompt group, steps through intervals; at each interval:
    * compute E[value], E[tokens] per sample
    * **FIRST**: prune any unfinished sample whose current E[value] is below a configurable threshold
    * **THEN**: run two-prefix search over subsets **and a future prune bin option**
    * prune "now" those not in the chosen active set; optionally plan a
      single future prune for the kept set at a chosen length bin.
- Produces a results Parquet matching online inference schema, so you can run:
    python src/label_and_evaluate.py --data <out>

New fast baselines (do not read NPZ / joint distributions):
  • eval-mode=prune_after_tokens
      Truncate each sample at a fixed token budget (default 8192).
  • eval-mode=prune_after_half_finished
      For each group (prompt or try), find the time when ⌈n/2⌉ samples have
      finished (the median final length). Prune/truncate all others at that time.
"""

from __future__ import annotations
import argparse, os, sys, json, math, traceback, ast, logging, time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# ASCII viz: reuse the same utilities as the online runner
from visualization import (
    log_inference_distribution,
    log_before_after_pruning,
    log_max_distribution,
    log_max_input_distributions,
    log_joint_distribution_grid,
)

# Optional JIT for hotspots
try:
    from numba import njit  # type: ignore
    HAVE_NUMBA = True
except Exception:  # degrade gracefully if numba isn't present
    HAVE_NUMBA = False

# Numerical tolerance used for objective tie-breaking / no-harm checks
TOL = 1e-12
EPS_GEOM = 1e-12

# Runtime flag toggled in main() when --jit is provided
JIT_EXPECTED_MAX = False
JIT_EXPECTED_MIN = False

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # degrade gracefully if tqdm isn't present
    def tqdm(x, **k):  # type: ignore
        return x

def _fmt_float_tag(x: float) -> str:
    """Human+filesystem friendly float tag."""
    # Prefer scientific notation for very small values
    s = f"{x:.8g}"
    # Replace characters that are awkward in filenames
    return s.replace(".", "p").replace("-", "m")

def _fmt_int_tag(x: int | None) -> str:
    """Human+filesystem friendly integer tag."""
    if x is None:
        return "none"
    return str(int(x))

# --- helpers for multi-input handling ---
def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def _infer_mode_from_path(p: str) -> str:
    s = os.path.basename(p).lower()
    if ("non_thinking" in s) or ("nonthinking" in s) or ("no_think" in s):
        return "nonthinking"
    if ("thinking" in s) or ("reason" in s):
        return "thinking"
    return "mode0"

def parse_args():
    p = argparse.ArgumentParser()
    # Support one or more inputs; order must align between parquet and NPZ
    p.add_argument("--labeled-parquet", nargs="+", required=True,
                   help="One or more labeled parquet paths (thinking/non-thinking/etc).")
    p.add_argument("--joint-npz", nargs="*", default=None,
                   help="Optional: NPZ per labeled parquet. If omitted, each is derived as *_joint_probs.npz")
    p.add_argument("--mode-tags", nargs="*", default=None,
                   help="Optional: tag per labeled parquet (e.g., thinking nonthinking). If omitted, inferred from filenames.")
    p.add_argument("--out", default="results/offline_results.parquet")
    # ZIP costs (same defaults as inference.py)
    p.add_argument("--c-pt", type=float, default=1/2_000_000)
    p.add_argument("--c-seq", type=float, default=1/32_000)
    # Step alignment for evaluating prune decisions (match labeling update_interval)
    p.add_argument("--step-size", type=int, default=64,
                   help="Evaluate prune cut decisions only at multiples of this many tokens; always include first (1) and final step.")
    # Evaluation modes for baselines vs policy
    p.add_argument("--eval-mode", choices=["policy","no_prune","keep_random_n","prune_after_tokens","prune_after_half_finished"], default="policy",
                   help="Evaluation mode. 'policy' uses pruning policy; 'no_prune' keeps all; "
                        "'keep_random_n' keeps exactly N randomly chosen samples per prompt group. "
                        "'prune_after_tokens' truncates each sample at a fixed budget; "
                        "'prune_after_half_finished' truncates all unfinished once ⌈n/2⌉ have finished.")
    p.add_argument("--keep-random-n", type=int, default=None,
                   help="When --eval-mode=keep_random_n, keep exactly this many randomly chosen samples per prompt group (others pruned at t=0).")
    p.add_argument("--random-seed", type=int, default=0, help="Random seed for --eval-mode=keep_random_n.")
    p.add_argument("--prune-after-tokens", type=int, default=8192,
                   help="For --eval-mode=prune_after_tokens, truncate each sample to this many output tokens (default: 8192).")
    # Optional: override when multiple models present
    p.add_argument("--decode-model", type=str, default=None, help="Tokenizer for decoding truncated responses; defaults to df['model_id'][0]")
    # Instrumentation flags
    p.add_argument("--log-interval", type=int, default=100, help="How many rows between progress logs.")
    p.add_argument("--trace-jsonl", type=str, default=None, help="If set, write a per-row JSONL trace here.")
    p.add_argument("--profile", action="store_true", help="If set, run with cProfile and write stats next to --out.")
    p.add_argument("--sample-limit", type=int, default=None, help="Process at most this many rows (for fast debugging).")
    p.add_argument("--jit", action="store_true", help="JIT hotspot loops with numba if available.")
    # Grouping: optionally split each prompt's samples into repeated trials of size K
    p.add_argument("--samples-per-try", type=int, default=None,
                   help="If set and 'sample_id' exists, split each prompt into trials of this many samples."
                        " Grouping and objectives will be computed per-trial instead of per-prompt.")

    # --- ZIP-style offline logging & visualization flags ---
    p.add_argument("--zip-logs", action="store_true",
                   help="Emit ZIP-style logs that mirror the online runner (text; gated by group/step limits).")
    p.add_argument("--zip-logs-groups", type=int, default=0,
                   help="Log at most this many groups (0 disables).")
    p.add_argument("--zip-logs-sample-idx", type=int, default=-1,
                   help="Only emit per-sample lines/visualizations for this sample index (-1 = all samples).")
    p.add_argument("--viz-pruning", action="store_true",
                   help="With --zip-logs, print BEFORE/AFTER ASCII joint distributions for chosen action.")
    p.add_argument("--viz-max", action="store_true",
                   help="With --zip-logs, print Distribution of max (value & tokens) and the input marginals.")
    p.add_argument("--debug", action="store_true",
                   help="Emit [ACTION] lines comparable to the online runner's debug output.")
    # --- Built-in evaluation (no model calls) using prelabels from the input parquet ---
    p.add_argument("--compute-metrics", action="store_true",
                   help="After pruning, compute and print metrics using cached labels (no model calls).")
    p.add_argument("--use-consistency", action="store_true",
                   help="Enable consistency/majority-vote metrics (requires 'extracted_answer' column from prelabeling).")
    p.add_argument("--thinking-token", type=str, default="</think>",
                   help="String token that marks the end of the thinking section (for reporting only).")
    p.add_argument("--thinking-token-id", type=int, default=151667,
                   help="Token id that marks the thinking section (for reasoning preference).")
    # NEW:
    p.add_argument("--metrics-json-out", type=str, default=None,
                   help="Where to save a JSON with the computed metrics. "
                        "If omitted and --compute-metrics is set, a default path is derived from --out/c_pt/c_seq.")
    p.add_argument("--faulthandler-timeout", type=int, default=0,
                   help="If >0, dump stacks after N seconds (non-repeating). 0 disables.")
    p.add_argument("--mmap", action="store_true",
                   help="Use np.load(..., mmap_mode='r') for the joint NPZ to reduce RAM without changing results.")
    # --- Joint temperature scaling ---
    p.add_argument("--joint-temp", type=float, default=1.0,
                   help="Temperature applied to each per-step JOINT distribution p(value,length). "
                        "T<1 sharpens; T>1 smooths; 1.0 = no change.")
    # --- Joint nucleus filtering (top-p) ---
    p.add_argument(
        "--joint-top-p", type=float, default=1.0,
        help="Top-p (nucleus) filtering applied to each per-step JOINT distribution p(value,length). "
             "Keeps the smallest set of (value,length) cells whose cumulative mass ≥ p, then renormalizes. "
             "1.0 = no change. If set <1.0, supersedes --joint-temp."
    )
    # --- New: selection skill via geometric interpolation between E[avg] and E[max] ---
    p.add_argument(
        "--selection-geom-alpha", type=float, default=1.0,
        help="α∈[0,1] for geometric interpolation of the value term: "
             "0→E[avg value], 1→E[max value]. Default 1.0 (original behavior).")
    # --- New: value-threshold pruning (precedes group objective) ---
    p.add_argument(
        "--ev-threshold", type=float, default=0.0,
        help="Prune any UNFINISHED sample at a step if its current E[value] falls below this threshold. "
             "Set to a negative value to disable."
    )

    # --- New: adaptive tries (choose K at t=0) ---
    p.add_argument("--adaptive-tries", action="store_true",
                   help="Enable t=0 optimization of the number of initial active samples K per prompt, "
                        "then partition each prompt's samples into disjoint tries of size K*. "
                        "Only applies to --eval-mode=policy.")
    p.add_argument("--k-min", type=int, default=1,
                   help="Lower bound for the optimized initial K (default: 1).")
    p.add_argument("--k-max", type=int, default=8,
                   help="Optional upper bound for the optimized initial K (default: unlimited).")

    # --- New: baseline chunking for no_prune ---
    p.add_argument("--no-prune-try-size", type=int, default=None,
                   help="When --eval-mode=no_prune, chunk each prompt into tries of this size. "
                        "If omitted, use all samples in one try.")
    p.add_argument("--drop-incomplete-tries", action="store_true",
                   help="Drop the final partial try when its sample count is below the requested per-try size "
                        "(adaptive K* for policy, or the fixed size for no_prune / samples_per_try).")
    p.add_argument("--max-active", type=int, default=None,
                   help="Max #unfinished samples to continue per group/step (policy). None=unlimited.")
    p.add_argument("--switch-penalty", type=float, default=0.0,
                   help="Subtract this from the group objective whenever the ACTIVE set changes between steps (policy).")
    return p.parse_args()

def _reshape_prob(flat: np.ndarray, V: int, L: int) -> np.ndarray:
    # We use value-major order then length, matching stage 2.
    return flat.reshape(V, L)

def _expected_values(joint: np.ndarray, reward_vals: List[float], length_bins: List[int]) -> Tuple[float,float]:
    # joint: [V,L] probs
    marg_reward = joint.sum(axis=1)     # [V]
    marg_tokens = joint.sum(axis=0)     # [L]
    # value expectation
    e_val = float(np.dot(marg_reward, np.array(reward_vals, dtype=np.float64)))
    # tokens: use midpoints
    mids = np.array([(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(len(length_bins)-1)], dtype=np.float64)
    e_tok = float(np.dot(marg_tokens, mids))
    return e_val, e_tok

def _cdf(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr, axis=0)

# --- temperature scaling helpers ---
def _softmax_temperature_from_probs(j2d: np.ndarray, T: float) -> np.ndarray:
    """
    Apply temperature scaling to a joint probability table j2d (shape [V,L]).
    p' ∝ p^{1/T} = softmax(log p / T) with normalization over all V×L cells.
    Stable in log-space; no-op when T≈1. Returns float64 array.
    """
    T = float(T)
    if T <= 0:
        raise ValueError("--joint-temp must be > 0")
    if abs(T - 1.0) <= 1e-12:
        return j2d
    arr = j2d.astype(np.float64, copy=False)
    logp = np.log(np.clip(arr, 1e-300, 1.0)) / T
    m = float(np.max(logp))
    x = np.exp(logp - m)
    Z = float(x.sum())
    if not np.isfinite(Z) or Z <= 0:
        return j2d
    return (x / Z).astype(np.float64)

# --- nucleus (top-p) filtering over the 2D joint ---
def _nucleus_top_p_from_probs(j2d: np.ndarray, top_p: float) -> np.ndarray:
    """
    Deterministic top-p (nucleus) filtering on a joint probability table j2d (shape [V,L]).
    Flattens the table, sorts by descending probability, keeps the smallest prefix whose
    cumulative mass ≥ top_p, zeros the rest, and renormalizes the kept mass to 1.
    Edge cases:
      - top_p >= ~1.0: no-op (returns input unchanged)
      - top_p <= 0: keep only the single highest-probability cell
      - non-positive / non-finite total mass: returns input unchanged
    """
    tp = float(top_p)
    if not np.isfinite(tp) or tp >= 1.0 - 1e-12:
        return j2d
    arr = j2d.astype(np.float64, copy=False)
    flat = arr.ravel()
    total = float(flat.sum())
    if total <= 0.0 or not np.isfinite(total):
        return j2d
    probs = flat / total
    if tp <= 0.0:
        out = np.zeros_like(probs, dtype=np.float64)
        out[int(np.argmax(probs))] = 1.0
        return out.reshape(arr.shape)
    order = np.argsort(probs)[::-1]
    cdf = np.cumsum(probs[order])
    k = int(np.searchsorted(cdf, tp, side="left")) + 1
    keep = order[:k]
    out = np.zeros_like(probs, dtype=np.float64)
    kept_sum = float(probs[keep].sum())
    out[keep] = (probs[keep] / kept_sum) if kept_sum > 0.0 else 0.0
    return out.reshape(arr.shape)

# --- value aggregation helpers (selection skill) ---
def _geom_interpolate(avg_val: float, max_val: float, alpha: float, eps: float = EPS_GEOM) -> float:
    """
    Geometric interpolation between average and max value:
        out = (max(avg_val,0)+eps)^(1-alpha) * (max(max_val,0)+eps)^alpha
    Falls back to arithmetic blend if either term is negative.
    """
    a = float(avg_val)
    m = float(max_val)
    if a <= 0.0 or m <= 0.0:
        # Safe fallback for any negative domain (rare in practice)
        return (1.0 - alpha) * a + alpha * m
    return float(np.exp((1.0 - alpha) * np.log(a + eps) + alpha * np.log(m + eps)))

# --- helper utilities for ZIP-style offline logging ---

def _max_bin_probs_np(marginals: List[np.ndarray], bin_values: List[float]):
     """
     Compute the probability that the maximum falls in each bin (discrete order statistics)
     using the CDF product identity, and return (prob_per_bin, sorted_bin_values, sort_idx).
     """
     if not marginals:
         return [], list(bin_values), np.arange(len(bin_values))
     vals_np = np.asarray(bin_values, dtype=np.float64)
     idx = np.argsort(vals_np)
     vals_sorted = vals_np[idx]
     cdfs = [np.cumsum(np.asarray(m, dtype=np.float64)[idx]) for m in marginals]
     probs = []
     prev = 0.0
     for b in range(len(vals_sorted)):
         f_upper = 1.0
         for c in cdfs:
             f_upper *= float(c[b])
         p_bin = f_upper if b == 0 else (f_upper - prev)
         probs.append(float(p_bin))
         prev = f_upper
     return probs, vals_sorted.tolist(), idx
 
def _reorder_marginals(marginals: List[np.ndarray], order_idx: np.ndarray) -> List[np.ndarray]:
    """Reorder each marginal by the provided permutation (to match sorted bin order)."""
    return [np.asarray(m, dtype=np.float64)[order_idx] for m in marginals]

if HAVE_NUMBA:
    @njit(fastmath=True, cache=True)
    def _expected_max_jitted(cdfs: np.ndarray, vals_sorted: np.ndarray) -> float:
        # cdfs: [N, B], vals_sorted: [B]
        n = cdfs.shape[0]
        B = cdfs.shape[1]
        out = 0.0
        prev = 0.0
        for b in range(B):
            f_upper = 1.0
            for i in range(n):
                f_upper *= cdfs[i, b]
            if b == 0:
                p_bin = f_upper
            else:
                p_bin = f_upper - prev
            out += p_bin * vals_sorted[b]
            prev = f_upper
        return out

def _expected_max(marginals: List[np.ndarray], bin_values: List[float]) -> float:
    if not marginals:
        return 0.0
    # Sort bins ascending by value to integrate with CDF
    idx = np.argsort(np.array(bin_values, dtype=np.float64))
    vals_sorted = np.array(bin_values, dtype=np.float64)[idx]

    # Optional JIT fast path
    if JIT_EXPECTED_MAX and HAVE_NUMBA:
        cdfs_list = []
        for m in marginals:
            m_sorted = np.asarray(m, dtype=np.float64)[idx]
            cdfs_list.append(np.cumsum(m_sorted))
        cdfs_np = np.ascontiguousarray(np.stack(cdfs_list, axis=0))
        vals_sorted_np = np.ascontiguousarray(np.asarray(vals_sorted, dtype=np.float64))
        return float(_expected_max_jitted(cdfs_np, vals_sorted_np))

    # Fallback: pure NumPy/Python
    cdfs = [np.cumsum(m[idx]) for m in marginals]
    out = 0.0
    prev = 0.0
    for b in range(len(vals_sorted)):
        f_upper = 1.0
        for c in cdfs:
            f_upper *= c[b]
        if b == 0:
            p_bin = f_upper
        else:
            p_bin = f_upper - prev
        out += p_bin * vals_sorted[b]
        prev = f_upper
    return float(out)

if HAVE_NUMBA:
    @njit(fastmath=True, cache=True)
    def _expected_min_jitted(survs: np.ndarray, vals_sorted: np.ndarray) -> float:
        """
        JIT for expected minimum using inclusive survival products.
        survs: [N, B] with S_i[b] = P(X_i >= bin[b]) on sorted bins (ascending).
        """
        n = survs.shape[0]
        B = survs.shape[1]
        # prod_ge[b] = Π_i S_i[b]
        prod_ge = np.ones(B, dtype=np.float64)
        for i in range(n):
            for b in range(B):
                prod_ge[b] *= survs[i, b]
        out = 0.0
        for b in range(B):
            # P(min in bin b) = Π_i S_i[b] - Π_i S_i[b+1], with prod_ge[B] := 0
            next_prod = prod_ge[b+1] if (b+1) < B else 0.0
            p_bin = prod_ge[b] - next_prod
            out += p_bin * vals_sorted[b]
        return out

def _expected_min(marginals: List[np.ndarray], bin_values: List[float]) -> float:
    """
    Expected minimum of discrete variables defined on 'bin_values'
    given per-sample marginals over those bins.
    """
    if not marginals:
        return 0.0
    vals = np.asarray(bin_values, dtype=np.float64)
    idx = np.argsort(vals)  # ascending by value
    vals_sorted = vals[idx]
    # Inclusive survival per sample at each sorted bin
    survs_list = [np.cumsum(np.asarray(m, dtype=np.float64)[idx][::-1])[::-1] for m in marginals]
    if JIT_EXPECTED_MIN and HAVE_NUMBA:
        survs_np = np.ascontiguousarray(np.stack(survs_list, axis=0))
        return float(_expected_min_jitted(survs_np, np.ascontiguousarray(vals_sorted)))
    # Fallback pure NumPy (identical math)
    prod_ge = np.ones_like(vals_sorted, dtype=np.float64)
    for s in survs_list:
        prod_ge *= s
    p_bin = prod_ge.copy()
    if len(p_bin) > 1:
        p_bin[:-1] -= prod_ge[1:]
    return float(np.dot(p_bin, vals_sorted))


def _min_bin_probs_np(marginals: List[np.ndarray], bin_values: List[float]):
    """
    Analogue of _max_bin_probs_np, but for the MIN order statistic.
    Returns (prob_per_bin, sorted_bin_values, sort_idx).
    Useful for optional viz.
    """
    if not marginals:
        return [], list(bin_values), np.arange(len(bin_values))
    vals_np = np.asarray(bin_values, dtype=np.float64)
    idx = np.argsort(vals_np)  # ascending
    vals_sorted = vals_np[idx]
    # Inclusive survival per sample at each sorted bin
    survs = [np.cumsum(np.asarray(m, dtype=np.float64)[idx][::-1])[::-1] for m in marginals]
    # Product over samples at each bin
    prod_ge = np.ones_like(vals_sorted, dtype=np.float64)
    for s in survs:
        prod_ge *= s
    # Probability mass that the minimum falls in each bin
    probs = prod_ge.copy()
    if len(probs) > 1:
        probs[:-1] -= prod_ge[1:]
    return probs.tolist(), vals_sorted.tolist(), idx

def _modify_for_prune(dist: np.ndarray, prune_at_bin: int | None, reward_values: List[float]) -> np.ndarray:
    # Collapse mass from bins >= prune_at_bin to reward=0 at that bin.
    if prune_at_bin is None: return dist
    V, L = dist.shape
    mod = dist.copy()
    collapsed = mod[:, prune_at_bin:].sum()
    mod[:, prune_at_bin:] = 0.0
    if collapsed > 0:
        # find reward idx closest to 0.0 (or equal)
        vals = np.array(reward_values, dtype=np.float64)
        zero_idx = int(np.argmin(np.abs(vals - 0.0)))
        mod[zero_idx, prune_at_bin] += collapsed
    return mod

def _choose_action(joints: List[np.ndarray], pruned_flags: List[bool],
                   reward_values: List[float], length_bins: List[int], c_pt: float, c_seq: float,
                   unfinished_mask: np.ndarray, selection_geom_alpha: float,
                   norm_tokens_per_sample: float):
    """
    Two-prefix search over subsets S **and** a single future prune bin B.
    Choices per interval:
      • KEEP (active) with a planned future prune at bin B (or B=None for no future prune)
      • PRUNE now (collapse all mass to reward≈0 at length bin 0)
    Objective (unchanged semantics vs v3):
      G = E[max value] - c_pt * E[Σ remaining tokens] - c_seq * E[max remaining tokens],
    where the token terms are computed only over ACTIVE kept items (unfinished ∧ kept).
    Returns: (active_set, future_bin, best_group_obj)

    Tie-breaking policy:
      • Prefer the candidate with the larger KEPT set |S| when objectives tie within tolerance.
      • If both objective and |S| tie, keep the first encountered (implicit / stable order).
        (No explicit randomness is introduced.)
    """
    V, L = joints[0].shape
    n = len(joints)
    unpruned = [i for i, p in enumerate(pruned_flags) if not p]
    if not unpruned:
        return set(), None, -1e30

    # ---- constants & basic per-sample marginals ----
    reward_values_np = np.array(reward_values, dtype=np.float64)
    mids = [(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(L)]
    mids_np = np.array(mids, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(reward_values_np - 0.0)))
    pruned_arr = np.array(pruned_flags, dtype=bool)

    # Per-sample base marginals under KEEP-now (use current distribution)
    reward_margs = [j.sum(axis=1).astype(np.float64) for j in joints]  # [n] of [V]
    token_margs  = [j.sum(axis=0).astype(np.float64) for j in joints]  # [n] of [L]
    # Prefix tables over length for fast "prune @ bin" modification (KEEP with future B)
    token_prefix  = [np.concatenate([[0.0], np.cumsum(tm, dtype=np.float64)]) for tm in token_margs]  # [n] of [L+1]
    reward_prefix = [np.concatenate([np.zeros((V,1), dtype=np.float64), np.cumsum(j.astype(np.float64), axis=1)], axis=1)
                     for j in joints]  # [n] of [V, L+1]
    # Per-sample marginals for PRUNE-now (all mass to reward≈0 at length bin 0)
    reward_margs_prune = []
    token_margs_prune  = []
    for j in joints:
        tot = float(j.sum())
        rm0 = np.zeros(V, dtype=np.float64)
        rm0[zero_idx] = tot
        tm0 = np.zeros(L, dtype=np.float64)
        tm0[0] = tot
        reward_margs_prune.append(rm0)
        token_margs_prune.append(tm0)

    # Ordering keys (computed only on unpruned)
    e_vals = [float(np.dot(reward_margs[i], reward_values_np)) for i in range(n)]
    e_toks = [float(np.dot(token_margs[i], mids_np))           for i in range(n)]
    order_tokens = sorted(unpruned, key=lambda i: e_toks[i])      # ascending E[tokens]
    order_reward = sorted(unpruned, key=lambda i: -e_vals[i])     # descending E[value]

    # Precompute boolean prefix masks for fast unions S = prefix_t[k1] | prefix_r[k2]
    prefix_t = np.zeros((len(order_tokens)+1, n), dtype=bool)
    for k in range(1, len(order_tokens)+1):
        idxs = order_tokens[:k]
        prefix_t[k, idxs] = True
    prefix_r = np.zeros((len(order_reward)+1, n), dtype=bool)
    for k in range(1, len(order_reward)+1):
        idxs = order_reward[:k]
        prefix_r[k, idxs] = True

    def group_obj_from_marginals(marg_vals: List[np.ndarray],
                                 marg_tok: List[np.ndarray],
                                 keep_mask: np.ndarray,
                                 active_mask: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Objective components:
          • Value term uses KEPT samples only (finished or active; excludes pruned):
              E_value_term = GeomInterp(E[avg value], E[max value]; alpha=selection_geom_alpha)
            where E[avg value] is the mean of per-sample expectations among kept.
            E[max value] is the same order-statistic expectation as before.
          • E[max remaining tokens] is computed only over ACTIVE kept samples.
          • E[Σ remaining tokens] is the SUM of per-sample E[remaining tokens] over ACTIVE kept samples.
        Returns: (G, E_value_term, E_sum_remaining_tokens, E_max_remaining_tokens)
        """
        kept_idxs = np.nonzero(keep_mask)[0].tolist()
        if kept_idxs:
            # E[max value] (upper bound)
            emax_val = _expected_max([marg_vals[i] for i in kept_idxs], bin_values=reward_values)
            # E[avg value] (lower bound = random pick among kept)
            reward_values_np = np.asarray(reward_values, dtype=np.float64)
            e_vals_kept = [float(np.dot(marg_vals[i], reward_values_np)) for i in kept_idxs]
            eavg_val = float(sum(e_vals_kept) / len(e_vals_kept)) if e_vals_kept else 0.0
        else:
            emax_val = 0.0
            eavg_val = 0.0
        e_value_term = _geom_interpolate(eavg_val, emax_val, float(selection_geom_alpha))
        # active subset for token-side costs
        active_idxs = np.nonzero(active_mask)[0].tolist()
        if active_idxs:
            active_tok_margs = [marg_tok[i] for i in active_idxs]
            # E[max remaining tokens] over ACTIVE
            e_max_tok = _expected_max(marginals=active_tok_margs, bin_values=mids)
            # E[Σ remaining tokens] = sum_i E[remaining_i] over ACTIVE
            e_sum_rem = float(sum(float(np.dot(active_tok_margs[j], mids_np)) for j in range(len(active_tok_margs))))
        else:
            e_max_tok = 0.0
            e_sum_rem = 0.0
        denom = max(float(norm_tokens_per_sample), 1e-12)  # guard
        # Normalize token costs into "average-sample" units
        g = e_value_term - c_pt * (e_sum_rem / denom) - c_seq * (e_max_tok / denom)
        return g, e_value_term, e_sum_rem, e_max_tok

    best_obj = -1e30
    best_mask: np.ndarray | None = None
    best_B: int | None = None
    # Track the size of the KEPT set for tie-breaking
    best_keep_size: int = -1

    # Finished samples in this step must always be kept (they add value and cost nothing).
    finished_mask = (~np.asarray(unfinished_mask, dtype=bool)) & (~pruned_arr)

    Kt = len(order_tokens)
    Kr = len(order_reward)
    future_options = [None] + list(range(1, L))  # skip 0 (equivalent to prune-now)
    seen: set[tuple[int | None, bytes]] = set()

    for B in future_options:
        # Precompute per-sample modified marginals under KEEP with this B
        keep_rm: List[np.ndarray] = [None] * n  # type: ignore
        keep_tm: List[np.ndarray] = [None] * n  # type: ignore
        for i in range(n):
            if B is None:
                keep_rm[i] = reward_margs[i]
                keep_tm[i] = token_margs[i]
            else:
                # reward: prefix up to B + collapsed mass into reward≈0 at bin B
                total_mass = float(joints[i].sum())
                rmB = (reward_prefix[i][:, B].copy())
                collapsedB = total_mass - float(token_prefix[i][B])
                if collapsedB > 0:
                    rmB[zero_idx] += collapsedB
                tmB = np.zeros_like(token_margs[i])
                if B > 0:
                    tmB[:B] = token_margs[i][:B]
                tmB[B] = collapsedB
                keep_rm[i] = rmB
                keep_tm[i] = tmB

        for k1 in range(Kt + 1):
            mask_t = prefix_t[k1]
            for k2 in range(Kr + 1):
                # Candidates to KEEP now; also force-keep finished.
                S_mask = (mask_t | prefix_r[k2]) | finished_mask
                key = (B, S_mask.tobytes())
                if key in seen:
                    continue
                seen.add(key)

                keep_mask   = S_mask & (~pruned_arr)           # used for E[max value]
                active_mask = keep_mask & unfinished_mask      # used for token costs

                # Build marginals: kept => KEEP-with-B dist, else => PRUNE-now dist
                marg_vals = [keep_rm[i] if keep_mask[i] else reward_margs_prune[i] for i in range(n)]
                marg_tok  = [keep_tm[i] if keep_mask[i] else token_margs_prune[i]  for i in range(n)]

                g, _, _, _ = group_obj_from_marginals(marg_vals, marg_tok, keep_mask, active_mask)
                # Switching penalty (allocation-free): compare mask bytes
                if float(globals().get("SWITCH_PENALTY", 0.0)) and (globals().get("PREV_ACTIVE_MASK_BYTES") is not None) and (active_mask.tobytes() != globals()["PREV_ACTIVE_MASK_BYTES"]):
                    g -= float(globals()["SWITCH_PENALTY"])

                # --- Tie-breaking logic ---
                # Primary: maximize objective g (with tolerance).
                # Secondary: on ties, prefer LARGER kept sets (|S|).
                # If both tie, keep the earlier candidate (implicit stable order).
                curr_keep_size = int(keep_mask.sum())
                if g > best_obj + TOL:
                    best_obj = g
                    best_mask = S_mask.copy()
                    best_B = B
                    best_keep_size = curr_keep_size
                elif abs(g - best_obj) <= TOL and curr_keep_size > best_keep_size:
                    best_mask = S_mask.copy()
                    best_B = B
                    best_keep_size = curr_keep_size

    best_S = set(np.nonzero(best_mask)[0]) if best_mask is not None else set(unpruned)
    return best_S, best_B, best_obj

def _parse_token_ids(val):
    """Return a Python list[int] from list/tuple/ndarray/bytes/str/NaN values."""
    # Already a list/tuple/ndarray -> normalize to list[int]
    try:
        import numpy as np
        if isinstance(val, (list, tuple, np.ndarray)):
            return [int(x) for x in list(val)]
    except Exception:
        pass

    # Handle missing
    if val is None:
        return []

    # Convert to string (bytes -> utf-8, ignore errors), strip NULs
    if isinstance(val, bytes):
        s = val.decode("utf-8", errors="ignore")
    else:
        s = str(val)
    s = s.replace("\x00", "").strip()

    if s == "" or s.lower() in {"none", "nan"}:
        return []

    # Try JSON first, then literal_eval (safer than eval)
    try:
        parsed = json.loads(s)
    except Exception:
        parsed = ast.literal_eval(s)

    return [int(x) for x in list(parsed)]

# --- instrumentation helpers ---

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

class RunStats:
    """Accumulates per-row timing & pruning behavior, and prints periodic summaries."""
    def __init__(self, total_rows: int, log_every: int, jsonl_path: str | None):
        self.total = total_rows
        self.log_every = max(1, log_every)
        self.t0 = time.perf_counter()
        self.rows = 0
        self.tokens = 0
        self.pruned_tokens = 0
        self.recent_dt = deque(maxlen=self.log_every)
        self.recent_keep = deque(maxlen=self.log_every)
        self.recent_correct = deque(maxlen=self.log_every)
        self.phase_totals = defaultdict(float)
        self.total_correct = 0
        # Ensure directory exists if path provided
        self.jsonl = None
        if jsonl_path:
            try:
                os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
            except Exception:
                pass
            self.jsonl = open(jsonl_path, "w")

    def log_row(self, *, orig_len: int, pruned_len: int, phases: dict[str, float], correct: bool = False, extra: dict | None = None):
        dt = sum(phases.values())
        self.rows += 1
        self.tokens += orig_len
        self.pruned_tokens += pruned_len
        self.recent_dt.append(dt)
        keep = pruned_len / max(1, orig_len)
        self.recent_keep.append(keep)
        self.recent_correct.append(correct)
        if correct:
            self.total_correct += 1
        for k, v in phases.items():
            self.phase_totals[k] += v
        if self.jsonl:
            rec = {
                "row": self.rows,
                "orig_len": orig_len,
                "pruned_len": pruned_len,
                "keep_ratio": keep,
                "correct": correct,
                "phases": phases,
            }
            if extra:
                rec.update(extra)
            self.jsonl.write(json.dumps(rec) + "\n")

        if (self.rows % self.log_every) == 0 or self.rows == self.total:
            elapsed = time.perf_counter() - self.t0
            rows_per_s = self.rows / max(1e-9, elapsed)
            toks_per_s = self.tokens / max(1e-9, elapsed)
            try:
                import numpy as _np
                keep_recent = float(_np.mean(self.recent_keep)) if self.recent_keep else float("nan")
                dt_mean = float(_np.mean(self.recent_dt)) if self.recent_dt else float("nan")
                dt_p95  = float(_np.percentile(self.recent_dt, 95)) if self.recent_dt else float("nan")
                correct_recent = float(_np.mean(self.recent_correct)) if self.recent_correct else float("nan")
            except Exception:
                keep_recent = float("nan"); dt_mean = float("nan"); dt_p95 = float("nan"); correct_recent = float("nan")
            keep_global = self.pruned_tokens / max(1, self.tokens)
            correct_global = self.total_correct / max(1, self.rows)
            total_phase = sum(self.phase_totals.values()) or 1.0
            phase_share = " ".join(
                f"{k}={100.0*v/total_phase:.0f}%"
                for k, v in sorted(self.phase_totals.items(), key=lambda kv: kv[1], reverse=True)
            )

            logging.info(
                f"[{self.rows}/{self.total}] rows/s={rows_per_s:.2f}  "
                f"toks/s={toks_per_s:.0f}  "
                f"keep_recent={keep_recent:.1%}  keep_global={keep_global:.1%}  "
                f"correct_recent={correct_recent:.1%}  correct_global={correct_global:.1%}  "
                f"row_dt(mean)={dt_mean:.3f}s  row_dt(p95)={dt_p95:.3f}s  |  phases: {phase_share}"
            )

    def close(self):
        if self.jsonl:
            self.jsonl.close()

@contextmanager
def tick(phases: dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        phases[key] += (time.perf_counter() - t0)

def _expected_max_iid_from_cdf(cdf_sorted: np.ndarray, vals_sorted: np.ndarray, K: int) -> float:
    """
    Expected max for K i.i.d. discrete variables with inclusive CDF 'cdf_sorted' on ascending support 'vals_sorted'.
    P(max in bin b) = F(b)^K - F(b-1)^K (with F(-1) := 0).
    """
    if K <= 0:
        return 0.0
    cdf = np.clip(np.asarray(cdf_sorted, dtype=np.float64), 0.0, 1.0)
    vals = np.asarray(vals_sorted, dtype=np.float64)
    Fk = np.power(cdf, float(K))
    prev = np.concatenate(([0.0], Fk[:-1]))
    p_bin = Fk - prev
    return float(np.dot(p_bin, vals))


def _optimize_initial_k_and_future_bin(
    j0: np.ndarray,
    reward_values: List[float],
    length_bins: List[int],
    tnow0: int,
    c_pt: float,
    c_seq: float,
    alpha: float,
    K_available: int,
    k_min: int = 1,
    k_max: int | None = None,
    tie_prefer_smaller_k: bool = True,
) -> Tuple[int, int | None, float]:
    """
    Optimize (K, future_bin B) at t=0 when all samples share the same joint j0.
    Returns (best_K, best_B, best_objective).
    Tie-breaking favors SMALLER K on exact ties to preserve samples for more tries.
    """
    V, L = j0.shape
    rv = np.asarray(reward_values, dtype=np.float64)
    mids = np.asarray([(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(L)], dtype=np.float64)

    # Collapse-helper for planned future prune
    zero_idx = int(np.argmin(np.abs(rv - 0.0)))

    def rm_tm_after_B(B: int | None) -> Tuple[np.ndarray, np.ndarray]:
        if B is None:
            rm = j0.sum(axis=1).astype(np.float64)
            tm = j0.sum(axis=0).astype(np.float64)
            return rm, tm
        rm = j0[:, :B].sum(axis=1).astype(np.float64)
        collapsed = float(j0[:, B:].sum())
        if collapsed > 0:
            rm = rm.copy()
            rm[zero_idx] += collapsed
        tm = np.zeros(L, dtype=np.float64)
        if B > 0:
            tm[:B] = j0.sum(axis=0)[:B]
        tm[B] = collapsed
        return rm, tm

    # Denominator as used in the main loop at a step: tnow + E[remaining]
    e_rem_pre = float(np.dot(j0.sum(axis=0).astype(np.float64), mids))
    norm_denom = max(float(tnow0) + e_rem_pre, 1e-12)

    # Search space
    Kmax = K_available if (k_max is None) else min(K_available, int(k_max))
    Kmin = max(1, int(k_min))
    if Kmin > Kmax:
        Kmin, Kmax = 1, max(1, K_available)

    best_g = -1e30
    best_k = Kmin
    best_B: int | None = None

    # Pre-sort keys for the order-statistic formulas
    order_r = np.argsort(rv)
    vals_r_sorted = rv[order_r]
    order_t = np.arange(L)  # mids already ascending by construction
    mids_sorted = mids[order_t]

    for B in [None] + list(range(1, L)):
        rm_B, tm_B = rm_tm_after_B(B)
        eavg_val = float(np.dot(rm_B, rv))
        e_tok_single = float(np.dot(tm_B, mids))

        cdf_r = np.cumsum(rm_B[order_r])
        cdf_t = np.cumsum(tm_B[order_t])

        for K in range(Kmin, Kmax + 1):
            emax_val = _expected_max_iid_from_cdf(cdf_r, vals_r_sorted, K)
            e_val_term = _geom_interpolate(eavg_val, emax_val, float(alpha))
            e_sum_rem = K * e_tok_single
            e_max_tok = _expected_max_iid_from_cdf(cdf_t, mids_sorted, K)

            g = e_val_term - c_pt * (e_sum_rem / norm_denom) - c_seq * (e_max_tok / norm_denom)

            if g > best_g + TOL:
                best_g, best_k, best_B = g, K, B
            elif abs(g - best_g) <= TOL:
                # Tie-break: prefer smaller K to conserve samples for more tries.
                if tie_prefer_smaller_k and K < best_k:
                    best_k, best_B = K, B
                elif K == best_k:
                    # Prefer None over a bin if both tie
                    if (best_B is not None) and (B is None):
                        best_B = None

    # Clamp defensively to at least 1
    best_k = max(1, min(int(best_k), int(K_available)))
    return best_k, best_B, float(best_g)

# --- NEW: two-pool fast-adaptive helpers --------------------------------------

def _avg_joint(joints: List[np.ndarray] | None) -> np.ndarray | None:
    """Average a list of [V,L] joints (already calibrated) into a single [V,L] joint."""
    if not joints:
        return None
    acc = None
    count = 0
    for j in joints:
        if j is None or j.size == 0:
            continue
        if acc is None:
            acc = j.astype(np.float64, copy=True)
        else:
            acc += j
        count += 1
    if acc is None or count == 0:
        return None
    return (acc / float(count)).astype(np.float64)

def _rm_tm_after_B(j0: np.ndarray, reward_values: List[float], B: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return reward marginal and token marginal after planning a future prune at bin B (None → keep)."""
    V, L = j0.shape
    rv = np.asarray(reward_values, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(rv - 0.0)))
    if B is None:
        rm = j0.sum(axis=1).astype(np.float64)
        tm = j0.sum(axis=0).astype(np.float64)
        return rm, tm
    rm = j0[:, :B].sum(axis=1).astype(np.float64)
    collapsed = float(j0[:, B:].sum())
    if collapsed > 0.0:
        rm = rm.copy()
        rm[zero_idx] += collapsed
    tm = np.zeros(L, dtype=np.float64)
    if B > 0:
        tm[:B] = j0.sum(axis=0)[:B]
    tm[B] = collapsed
    return rm, tm

def _score_iid_candidate(
    j0: np.ndarray,
    reward_values: List[float],
    length_bins: List[int],
    *, K: int, B: int | None,
    c_pt: float, c_seq: float, alpha: float,
    norm_denom: float
) -> tuple[float, float, float, float]:
    """
    Score G for IID pool with fixed (K, B) and a given normalization denominator.
    Returns (G, E_value_term, E_sum_remaining_tokens, E_max_remaining_tokens).
    """
    if K <= 0:
        return -1e30, 0.0, 0.0, 0.0
    rv = np.asarray(reward_values, dtype=np.float64)
    mids = np.asarray([(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(j0.shape[1])], dtype=np.float64)

    rm_B, tm_B = _rm_tm_after_B(j0, reward_values, B)
    eavg_val = float(np.dot(rm_B, rv))
    # Expected max(value) for K IID
    order_r = np.argsort(rv)
    emax_val = _expected_max_iid_from_cdf(np.cumsum(rm_B[order_r]), rv[order_r], K)
    e_value_term = _geom_interpolate(eavg_val, emax_val, float(alpha))

    # Token costs (remaining)
    order_t = np.arange(len(mids))  # already ascending
    e_tok_single = float(np.dot(tm_B, mids))
    e_sum_rem = K * e_tok_single
    e_max_tok = _expected_max_iid_from_cdf(np.cumsum(tm_B[order_t]), mids[order_t], K)

    denom = max(float(norm_denom), 1e-12)
    G = e_value_term - c_pt * (e_sum_rem / denom) - c_seq * (e_max_tok / denom)
    return G, e_value_term, e_sum_rem, e_max_tok

def _best_B_for_fixed_K(
    j0: np.ndarray,
    reward_values: List[float],
    length_bins: List[int],
    *, K: int, c_pt: float, c_seq: float, alpha: float,
    norm_denom: float
) -> tuple[int | None, float]:
    """Search B∈{None,1..L-1} for fixed K, return (B*, G*)."""
    V, L = j0.shape
    best_B, best_G = None, -1e30
    for B in [None] + list(range(1, L)):
        G, *_ = _score_iid_candidate(j0, reward_values, length_bins, K=K, B=B,
                                     c_pt=c_pt, c_seq=c_seq, alpha=alpha, norm_denom=norm_denom)
        if G > best_G + TOL:
            best_B, best_G = B, G
    return best_B, best_G

def _norm_denom_from_j0(j0: np.ndarray, tnow_avg: float, length_bins: List[int]) -> float:
    """tnow + E[remaining] based on a representative joint (pre-B), matching denominator semantics."""
    mids = np.asarray([(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(j0.shape[1])], dtype=np.float64)
    e_rem = float(np.dot(j0.sum(axis=0).astype(np.float64), mids))
    return float(tnow_avg) + e_rem

def _optimize_initial_k_two_pool(
    j_th: np.ndarray | None, j_nt: np.ndarray | None,
    tnow_th_avg: float | None, tnow_nt_avg: float | None,
    n_th: int, n_nt: int,
    reward_values: List[float], length_bins: List[int],
    *, c_pt: float, c_seq: float, alpha: float, k_min: int, k_max: int | None
) -> tuple[str, int, int | None, float, float]:
    """
    Return (mode*, K*, B*, G*, norm_denom_ref).
    Policy:
      • If thinking exists: compare 'thinking-only' (optimize K_t) vs 'nonthinking-only with K=1';
        normalize both with the THINKING denominator.
      • If thinking absent: optimize nonthinking K freely and normalize on nonthinking.
    """
    best_mode, best_K, best_B, best_G = "nonthinking", 1, None, -1e30
    norm_ref = 1.0

    has_th = (j_th is not None and n_th > 0)
    has_nt = (j_nt is not None and n_nt > 0)

    if not has_th and not has_nt:
        return best_mode, best_K, best_B, best_G, norm_ref

    if has_th:
        norm_ref = _norm_denom_from_j0(j_th, float(tnow_th_avg or 0.0), length_bins)
        Kmax_t = n_th if (k_max is None) else min(n_th, int(k_max))
        Kmin_t = max(1, int(k_min))
        best_G_th, best_K_th, best_B_th = -1e30, Kmin_t, None
        V, L = j_th.shape
        for B in [None] + list(range(1, L)):
            rm_B, tm_B = _rm_tm_after_B(j_th, reward_values, B)
            rv = np.asarray(reward_values, dtype=np.float64)
            mids = np.asarray([(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(L)], dtype=np.float64)
            order_r = np.argsort(rv); cdf_r = np.cumsum(rm_B[order_r])
            order_t = np.arange(L);  cdf_t = np.cumsum(tm_B[order_t])
            eavg_val = float(np.dot(rm_B, rv))
            for K in range(Kmin_t, Kmax_t + 1):
                emax_val = _expected_max_iid_from_cdf(cdf_r, rv[order_r], K)
                e_value_term = _geom_interpolate(eavg_val, emax_val, float(alpha))
                e_sum_rem = K * float(np.dot(tm_B, mids))
                e_max_tok = _expected_max_iid_from_cdf(cdf_t, mids[order_t], K)
                G = e_value_term - c_pt*(e_sum_rem/max(norm_ref,1e-12)) - c_seq*(e_max_tok/max(norm_ref,1e-12))
                if G > best_G_th + TOL or (abs(G - best_G_th) <= TOL and K < best_K_th):
                    best_G_th, best_K_th, best_B_th = G, K, B

        best_G_nt1, best_B_nt1 = -1e30, None
        if has_nt:
            B_nt1, G_nt1 = _best_B_for_fixed_K(j_nt, reward_values, length_bins,
                                               K=1, c_pt=c_pt, c_seq=c_seq, alpha=alpha, norm_denom=norm_ref)
            best_B_nt1, best_G_nt1 = B_nt1, G_nt1

        if best_G_th >= best_G_nt1:
            return "thinking", int(best_K_th), best_B_th, float(best_G_th), float(norm_ref)
        else:
            return "nonthinking", 1, best_B_nt1, float(best_G_nt1), float(norm_ref)

    norm_ref = _norm_denom_from_j0(j_nt, float(tnow_nt_avg or 0.0), length_bins)
    Kmax_nt = n_nt if (k_max is None) else min(n_nt, int(k_max))
    Kmin_nt = max(1, int(k_min))
    best_G_nt, best_K_nt, best_B_nt = -1e30, Kmin_nt, None
    V, L = j_nt.shape
    for B in [None] + list(range(1, L)):
        rm_B, tm_B = _rm_tm_after_B(j_nt, reward_values, B)
        rv = np.asarray(reward_values, dtype=np.float64)
        mids = np.asarray([(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(L)], dtype=np.float64)
        order_r = np.argsort(rv); cdf_r = np.cumsum(rm_B[order_r])
        order_t = np.arange(L);  cdf_t = np.cumsum(tm_B[order_t])
        eavg_val = float(np.dot(rm_B, rv))
        for K in range(Kmin_nt, Kmax_nt + 1):
            emax_val = _expected_max_iid_from_cdf(cdf_r, rv[order_r], K)
            e_value_term = _geom_interpolate(eavg_val, emax_val, float(alpha))
            e_sum_rem = K * float(np.dot(tm_B, mids))
            e_max_tok = _expected_max_iid_from_cdf(cdf_t, mids[order_t], K)
            G = e_value_term - c_pt*(e_sum_rem/max(norm_ref,1e-12)) - c_seq*(e_max_tok/max(norm_ref,1e-12))
            if G > best_G_nt + TOL or (abs(G - best_G_nt) <= TOL and K < best_K_nt):
                best_G_nt, best_K_nt, best_B_nt = G, K, B
    return "nonthinking", int(best_K_nt), best_B_nt, float(best_G_nt), float(norm_ref)

def main():
    args = parse_args()
    # --- SIMPLE INTERVAL-CONTINUE MODE (replace adaptive tries + disable threshold prunes) ---
    args.adaptive_tries = False
    args.samples_per_try = 8
    args.drop_incomplete_tries = True
    args.ev_threshold = -1.0  # disable threshold-based pruning entirely
    labeled_paths = _as_list(args.labeled_parquet)
    npz_paths = _as_list(args.joint_npz)
    if not npz_paths:
        npz_paths = [lp.replace(".parquet", "_joint_probs.npz") for lp in labeled_paths]
    if len(npz_paths) != len(labeled_paths):
        raise ValueError("--joint-npz count must match --labeled-parquet count (or be omitted entirely).")
    mode_tags = _as_list(args.mode_tags)
    if mode_tags and len(mode_tags) != len(labeled_paths):
        raise ValueError("--mode-tags length must equal number of --labeled-parquet inputs.")
    if not mode_tags:
        mode_tags = [_infer_mode_from_path(p) for p in labeled_paths]

    # Fast baseline modes skip NPZ entirely.
    FAST_BASELINE = args.eval_mode in {"prune_after_tokens", "prune_after_half_finished"}

    # Load all parquets and NPZs, validate grids, and merge
    df_list = []
    probs_list = []
    positions_list = []
    step_offset = 0
    ref = None  # (reward_values, length_bins, V, L, B)
    model_ids = []

    for i, (lp, npz) in enumerate(zip(labeled_paths, npz_paths)):
        dfi = pq.read_table(lp).to_pandas()
        if "model_id" in dfi.columns:
            try:
                model_ids.append(str(dfi.iloc[0]["model_id"]))
            except Exception:
                pass

        rv = list(dfi.iloc[0]["reward_values"])
        lb = list(dfi.iloc[0]["length_bins"])
        Vi = int(dfi.iloc[0]["num_reward_states"])
        Li = int(dfi.iloc[0]["num_length_bins"])
        Bi = Vi * Li
        if ref is None:
            ref = (rv, lb, Vi, Li, Bi)
        else:
            r0, l0, V0, L0, B0 = ref
            if Vi != V0 or Li != L0 or Bi != B0 or list(rv) != list(r0) or list(lb) != list(l0):
                raise ValueError(
                    f"Incompatible grids across inputs. Expected V={V0},L={L0} and identical reward_values/length_bins; got V={Vi},L={Li} for {lp}."
                )

        # Always carry mode tag
        dfi["__mode__"] = mode_tags[i]
        df_list.append(dfi)

        # Only load NPZ / adjust offsets when not in fast baseline mode
        if not FAST_BASELINE:
            data_i = np.load(npz, mmap_mode=("r" if args.mmap else None))
            probs_i = data_i["probs"]
            positions_i = data_i["positions"]
            if probs_i.shape[1] != ref[4]:
                raise ValueError(f"NPZ {npz} probs second dim {probs_i.shape[1]} != V*L {ref[4]}")
            off_i = dfi["joint_probs_offset"].astype(np.int64).values + step_offset
            # Store shifted offsets back into the same dataframe we appended
            df_list[-1]["joint_probs_offset"] = off_i
            probs_list.append(probs_i)
            positions_list.append(positions_i)
            step_offset += probs_i.shape[0]

    df = pd.concat(df_list, ignore_index=True)
    if args.sample_limit is not None:
        df = df.iloc[: int(args.sample_limit)].copy()
    # Preserve a copy of the merged INPUTS before any try construction/drops (for metrics baselines)
    df_inputs_all = df.copy()

    reward_values, length_bins, V, L, B = ref
    # Instrumentation setup (do this once, after merge)
    setup_logging()
    # Configure JIT toggle and report status
    global JIT_EXPECTED_MAX, JIT_EXPECTED_MIN
    JIT_EXPECTED_MAX = bool(args.jit and HAVE_NUMBA)
    JIT_EXPECTED_MIN = bool(args.jit and HAVE_NUMBA)
    if args.jit and not HAVE_NUMBA:
        logging.warning("--jit requested but numba is not available; running without JIT.")
    else:
        logging.info(f"expected_max JIT: {'ON' if JIT_EXPECTED_MAX else 'OFF'}")
        if args.jit:
            logging.info(f"expected_min JIT: {'ON' if JIT_EXPECTED_MIN else 'OFF'}")
    # Report threshold pruning config
    if float(args.ev_threshold) < 0:
        logging.info("EV-threshold pruning: DISABLED")
    else:
        logging.info(f"EV-threshold pruning: threshold={float(args.ev_threshold):.6f}")
    # Report joint calibration (top-p takes precedence over temperature if both are set)
    if float(args.joint_top_p) < 1.0 - 1e-12:
        logging.info(f"Joint nucleus filtering enabled: top-p={float(args.joint_top_p):.3f}")
    elif abs(float(args.joint_temp) - 1.0) > 1e-12:
        logging.info(f"Joint temperature scaling enabled: T={float(args.joint_temp):.3f}")
    logging.info(f"Selection skill (geom alpha): {float(args.selection_geom_alpha):.3f}")

    # Set conservative defaults for BLAS threading if not provided (perf only; no semantic change)
    for _var, _default in (("OMP_NUM_THREADS","1"),("OPENBLAS_NUM_THREADS","1"),
                           ("MKL_NUM_THREADS","1"),("NUMEXPR_NUM_THREADS","1")):
        os.environ.setdefault(_var, _default)
    # Only materialize NPZ-derived arrays when not in fast baseline mode
    if not FAST_BASELINE:
        probs = np.concatenate(probs_list, axis=0) if len(probs_list) > 1 else probs_list[0]
        positions = np.concatenate(positions_list, axis=0) if len(positions_list) > 1 else positions_list[0]
        # Prepare per-row interval metadata (using the merged df/probs/positions)
        offsets = np.asarray(df["joint_probs_offset"].values, dtype=np.int64)
        num_steps = np.asarray(df["joint_probs_num_steps"].values, dtype=np.int32)
        # Safety assertion for merged NPZ tensor
        assert probs.shape[1] == B, \
            f"NPZ probs second dim ({probs.shape[1]}) != V*L ({B}). Make sure all inputs share the same grid."

    # Tokenizer for decoding truncated responses
    decode_model = args.decode_model or (df.iloc[0]["model_id"] if "model_id" in df.columns else "gpt2")
    tok = AutoTokenizer.from_pretrained(decode_model, trust_remote_code=True)

    # Prepare containers for final rows
    final_rows = []

    # RunStats will be initialized after try construction and optional filtering
    # to ensure totals reflect the processed rows only.

    # Optional one-shot traceback dump if process stalls (non-repeating)
    fh_armed = False
    try:
        import faulthandler
        if int(args.faulthandler_timeout or 0) > 0:
            faulthandler.dump_traceback_later(int(args.faulthandler_timeout), repeat=False)
            fh_armed = True
    except Exception:
        pass

    # Iterate by group (prompt or per-try subgroup)
    base_group_key = "prompt_idx" if "prompt_idx" in df.columns else "prompt"
    groupby_cols = [base_group_key]

    # -----------------------------
    # Try construction (either adaptive or fixed)
    # -----------------------------
    if args.adaptive_tries and args.eval_mode == "policy":
        # Two-pool fast adaptive K*: per-prompt choose thinking vs nonthinking IID pools; tries are homogeneous.
        dropped_leftover_rows_total = 0
        df["try_idx"] = -1
        df["group_id"] = None
        df["group_mode"] = None
        prompt_norm_denom_for: dict[object, float] = {}

        # Helper to get the first-step joint for a row
        def _first_step_joint_for_row(ridx: int):
            off = int(offsets[ridx])
            ns  = int(num_steps[ridx])
            if ns <= 0:
                # degenerate: all mass at reward≈0, length bin 0
                j = np.zeros((V, L), dtype=np.float64)
                zi = int(np.argmin(np.abs(np.asarray(reward_values, dtype=np.float64) - 0.0)))
                j[zi, 0] = 1.0
                tnow = 0
                return j, tnow
            j = _reshape_prob(probs[off].astype(np.float64), V, L)
            # Apply joint calibration (same precedence as main loop)
            if float(args.joint_top_p) < 1.0 - 1e-12:
                j = _nucleus_top_p_from_probs(j, float(args.joint_top_p))
            elif abs(float(args.joint_temp) - 1.0) > 1e-12:
                j = _softmax_temperature_from_probs(j, float(args.joint_temp))
            tnow = int(positions[off])
            return j, tnow

        for pid, grp in df.groupby(base_group_key):
            row_indices = list(grp.index)
            if not row_indices:
                continue
            # Build per-sample j0 and tnow across all samples in this prompt
            j0_list = []
            tnow_list = []
            for rid in row_indices:
                j0, t0 = _first_step_joint_for_row(rid)
                j0_list.append(j0)
                tnow_list.append(int(t0))

            # Split indices by mode
            idx_th = [rid for rid in row_indices if df.at[rid, "__mode__"] == "thinking"]
            idx_nt = [rid for rid in row_indices if df.at[rid, "__mode__"] == "nonthinking"]

            j_th_list = [j0_list[row_indices.index(r)] for r in idx_th] if idx_th else []
            j_nt_list = [j0_list[row_indices.index(r)] for r in idx_nt] if idx_nt else []
            t_th_list = [tnow_list[row_indices.index(r)] for r in idx_th] if idx_th else []
            t_nt_list = [tnow_list[row_indices.index(r)] for r in idx_nt] if idx_nt else []

            j_th = _avg_joint(j_th_list)
            j_nt = _avg_joint(j_nt_list)
            tnow_th_avg = float(np.mean(t_th_list)) if t_th_list else 0.0
            tnow_nt_avg = float(np.mean(t_nt_list)) if t_nt_list else 0.0

            mode_star, K_star, B_star, G_star, norm_ref = _optimize_initial_k_two_pool(
                j_th, j_nt, tnow_th_avg, tnow_nt_avg, len(idx_th), len(idx_nt),
                reward_values, length_bins,
                c_pt=float(args.c_pt), c_seq=float(args.c_seq), alpha=float(args.selection_geom_alpha),
                k_min=int(args.k_min), k_max=(None if args.k_max is None else int(args.k_max))
            )
            prompt_norm_denom_for[pid] = norm_ref

            # --- Per-prompt allocation logging (always at INFO) ---
            N_th, N_nt = len(idx_th), len(idx_nt)
            alloc_th = int(K_star) if (mode_star == "thinking") else 0
            alloc_nt = (1 if (mode_star != "thinking" and N_nt > 0) else 0)
            logging.info(
                f"[ALLOC] prompt={pid}  K*={K_star}  B*={B_star}  "
                f"alloc_thinking={alloc_th}  alloc_nonthinking={alloc_nt}  "
                f"available_thinking={N_th}  available_nonthinking={N_nt}"
            )

            assigned = set()

            # Choose the first try (full K* for the chosen mode; guaranteed by optimizer bounds)
            if mode_star == "thinking" and idx_th:
                chosen_first = idx_th[:K_star]
                remaining_same = idx_th[K_star:]
                other_mode_remaining = idx_nt
                chosen_mode = "thinking"
            elif idx_nt:
                chosen_first = idx_nt[:K_star]
                remaining_same = idx_nt[K_star:]
                other_mode_remaining = idx_th
                chosen_mode = "nonthinking"
            else:
                chosen_first, remaining_same, other_mode_remaining = [], [], []
                chosen_mode = None

            # Assign the first (full) try
            if chosen_first:
                df.loc[chosen_first, "try_idx"] = 0
                df.loc[chosen_first, "group_id"] = f"{pid}_t0"
                df.loc[chosen_first, "group_mode"] = chosen_mode
                assigned.update(chosen_first)

            # Assign ONLY full-size chunks of the same mode; drop leftovers < K*
            t = 1
            stride = max(1, int(K_star))
            full_count = (len(remaining_same) // stride) * stride
            for i in range(0, full_count, stride):
                idxs = remaining_same[i:i+stride]
                df.loc[idxs, "try_idx"] = t
                df.loc[idxs, "group_id"] = f"{pid}_t{t}"
                df.loc[idxs, "group_mode"] = chosen_mode
                assigned.update(idxs)
                t += 1

            # Strict policy: never mix modes in tries; no baseline non-thinking when thinking is chosen.

            # --- Tries summary (based on actual assignments) ---
            if assigned:
                sizes_for_prompt = (
                    df.loc[list(assigned), ["group_id", "group_mode"]]
                      .groupby(["group_id", "group_mode"])  # type: ignore[arg-type]
                      .size()
                      .reset_index(name="size")
                )
                th_tries = int((sizes_for_prompt["group_mode"] == "thinking").sum())
                nt_tries = int((sizes_for_prompt["group_mode"] == "nonthinking").sum())
                th_sizes = sorted(sizes_for_prompt.loc[sizes_for_prompt["group_mode"] == "thinking", "size"].unique().tolist())
                nt_sizes = sorted(sizes_for_prompt.loc[sizes_for_prompt["group_mode"] == "nonthinking", "size"].unique().tolist())
            else:
                th_tries = 0
                nt_tries = 0
                th_sizes = []
                nt_sizes = []

            logging.info(
                f"[TRIES] prompt={pid}  tries_thinking={th_tries}  tries_nonthinking={nt_tries}  "
                f"try_sizes_thinking={th_sizes}  try_sizes_nonthinking={nt_sizes}"
            )

            # Drop all rows in this prompt that did not get assigned to a full try
            unassigned = [rid for rid in row_indices if rid not in assigned]
            if unassigned:
                dropped_th = sum(1 for rid in unassigned if df.at[rid, "__mode__"] == "thinking")
                dropped_nt = sum(1 for rid in unassigned if df.at[rid, "__mode__"] == "nonthinking")
                logging.info(f"[SKIP-LEFTOVER] prompt={pid}  dropped_thinking={dropped_th}  dropped_nonthinking={dropped_nt}")
                dropped_leftover_rows_total += len(unassigned)
                df = df.drop(index=unassigned)

        groupby_cols = ["group_id"]

    elif args.samples_per_try is not None:
        # existing fixed-chunking (unchanged)
        try:
            sptry = int(args.samples_per_try)
        except Exception:
            sptry = None
        if sptry and sptry > 0:
            base_key = base_group_key
            if "sample_id" in df.columns:
                df["__rank_in_prompt__"] = (
                    df.groupby(base_key)["sample_id"].rank(method="first", ascending=True).astype(int) - 1
                )
            else:
                df["__rank_in_prompt__"] = df.groupby(base_key).cumcount()
            df["try_idx"] = (df["__rank_in_prompt__"] // sptry).astype(int)
            df["group_id"] = df[base_key].astype(str) + "_t" + df["try_idx"].astype(str)
            groupby_cols = ["group_id"]

    elif (args.eval_mode == "no_prune") and (args.no_prune_try_size is not None) and (int(args.no_prune_try_size) > 0):
        # For a fair no_prune baseline, allow chunking each prompt into fixed-size tries
        sptry = int(args.no_prune_try_size)
        base_key = base_group_key
        df["__rank_in_prompt__"] = df.groupby(base_key).cumcount()
        df["try_idx"] = (df["__rank_in_prompt__"] // sptry).astype(int)
        df["group_id"] = df[base_key].astype(str) + "_t" + df["try_idx"].astype(str)
        groupby_cols = ["group_id"]

    # --- Drop incomplete tries if requested ------------------------------------
    dropped_rows_incomplete = 0
    dropped_partial_tries = 0

    if args.drop_incomplete_tries and ("group_id" in df.columns):
        # Size of each try (group)
        sizes_by_group = df.groupby("group_id").size()

        # Map group_id -> (prompt, mode)
        if "group_mode" in df.columns:
            gid_meta = df[["group_id", base_group_key, "group_mode"]].drop_duplicates().set_index("group_id")
            # target size per (prompt, group_mode)
            target_size_for_group = sizes_by_group.groupby([gid_meta[base_group_key], gid_meta["group_mode"]]).transform("max")
        else:
            # Fallback to previous behavior
            gid_to_prompt = (
                df[["group_id", base_group_key]]
                .drop_duplicates()
                .set_index("group_id")[base_group_key]
            )
            target_size_for_group = sizes_by_group.groupby(gid_to_prompt).transform("max")

        mask_incomplete = sizes_by_group < target_size_for_group
        to_drop = sizes_by_group.index[mask_incomplete]
        if len(to_drop) > 0:
            dropped_rows_incomplete = int(sizes_by_group.loc[to_drop].sum())
            dropped_partial_tries = int(len(to_drop))
            df = df[~df["group_id"].isin(set(to_drop))].copy()
            logging.info(f"Dropped {dropped_partial_tries} partial tries ({dropped_rows_incomplete} rows) due to --drop-incomplete-tries")

    # Initialize RunStats after filtering so totals/denominators are accurate
    num_rows = len(df)
    stats = RunStats(total_rows=num_rows, log_every=args.log_interval, jsonl_path=args.trace_jsonl)

    has_try_col = "try_idx" in df.columns
    has_group_col = "group_id" in df.columns
    # ZIP-style logging counters
    logged_groups = 0
    for pid, grp in df.groupby(groupby_cols):
        # --- FAST BASELINES: prune decisions without NPZ / step simulation -------
        if 'FAST_BASELINE' in locals() and FAST_BASELINE:
            row_indices = list(grp.index)
            if not row_indices:
                continue

            pruned = [False] * len(row_indices)
            pruned_at_tokens = [None] * len(row_indices)
            exp_hist: List[List[float]] = [[] for _ in row_indices]
            exp_last: List[float | None] = [None] * len(row_indices)

            if args.eval_mode == "prune_after_tokens":
                cutoff = max(0, int(args.prune_after_tokens))
                for i, ridx in enumerate(row_indices):
                    full_len = int(df.loc[ridx].get("length", 0) or 0)
                    if full_len > cutoff:
                        pruned[i] = True
                        pruned_at_tokens[i] = cutoff
                    else:
                        pruned[i] = False
                        pruned_at_tokens[i] = None

            elif args.eval_mode == "prune_after_half_finished":
                n = len(row_indices)
                k = (n + 1) // 2  # ceil(n/2)
                lengths = [int(df.loc[ridx].get("length", 0) or 0) for ridx in row_indices]
                sorted_lens = sorted(lengths)
                median_finish = sorted_lens[k - 1] if k > 0 else 0
                T = int(median_finish)
                for i, ridx in enumerate(row_indices):
                    L_i = int(df.loc[ridx].get("length", 0) or 0)
                    if L_i > T:
                        pruned[i] = True
                        pruned_at_tokens[i] = T
                    else:
                        pruned[i] = False
                        pruned_at_tokens[i] = None

            # Build final rows (no joints / objectives in phases)
            for i, ridx in enumerate(row_indices):
                row = df.loc[ridx]
                out_ids_full = _parse_token_ids(row["output_token_ids"])
                prompt_ids = _parse_token_ids(row["prompt_token_ids"]) 
                phases: dict[str, float] = {"joint": 0.0, "objective": 0.0}

                decode_t0 = time.perf_counter()
                if pruned[i]:
                    keep = int(pruned_at_tokens[i] or 0)
                    out_ids = out_ids_full[:keep]
                    text = tok.decode(out_ids, skip_special_tokens=True).strip()
                    finished = False
                    length = len(out_ids)
                else:
                    out_ids = out_ids_full
                    text = row["response"]
                    finished = bool(row.get("finished", True))
                    length = int(row.get("length", len(out_ids_full)))
                phases["decode"] = time.perf_counter() - decode_t0

                exp_last[i] = None
                write_t0 = time.perf_counter()
                rec = {
                    "prompt_idx": int(row["prompt_idx"]),
                    "prompt": row["prompt"],
                    "answer": row.get("answer", None),
                    "response": text,
                    "length": length,
                    "original_length": int(len(out_ids_full)),
                    "finished": finished and (not pruned[i]),
                    "pruned": bool(pruned[i]),
                    "expected_reward": None,
                    "reward_values": row["reward_values"],
                    "expected_tokens": None,
                    "expected_reward_history": exp_hist[i],
                    "distribution_logits_history": [],
                    "prompt_token_ids": prompt_ids,
                    "output_token_ids": out_ids,
                    "correct": bool(row.get("correct", False)) if (finished and (not pruned[i])) else False,
                    "extracted_answer": (row.get("extracted_answer", None) if (finished and (not pruned[i])) else None),
                    "mode": (row.get("__mode__") if "__mode__" in df.columns else None),
                }
                if has_try_col:
                    rec["try_idx"] = int(row["try_idx"])
                if has_group_col:
                    rec["group_id"] = row["group_id"]
                final_rows.append(rec)
                phases["write"] = time.perf_counter() - write_t0

                orig_len = len(out_ids_full)
                pruned_len = len(out_ids)
                is_correct = rec["correct"]
                stats.log_row(orig_len=orig_len, pruned_len=pruned_len, phases=phases, correct=is_correct, extra=None)
            # Done with this group
            continue

        # Build per-sample per-step joints
        row_indices = list(grp.index)
        joints_per_sample: List[List[np.ndarray]] = []
        steps_per_sample: List[List[int]] = []
        group_joint_t0 = time.perf_counter()
        for ridx in row_indices:
            off = int(offsets[ridx])
            ns = int(num_steps[ridx])
            if ns == 0:
                joints_per_sample.append([])
                steps_per_sample.append([])
                continue
            flat = probs[off:off+ns]  # [ns, B]
            joints_all = []
            for f in flat:
                j = _reshape_prob(f.astype(np.float64), V, L)
                # Apply joint calibration in precedence order: 1) top-p (<1.0) else 2) temperature (!=1.0)
                if float(args.joint_top_p) < 1.0 - 1e-12:
                    j = _nucleus_top_p_from_probs(j, float(args.joint_top_p))
                elif abs(float(args.joint_temp) - 1.0) > 1e-12:
                    j = _softmax_temperature_from_probs(j, float(args.joint_temp))
                joints_all.append(j)
            step_tokens_all = positions[off:off+ns].tolist()

            # Step-align: keep t==1 and multiples of --step-size, always include first+last indices
            K = max(1, int(args.step_size))
            keep_idx = [i for i, t in enumerate(step_tokens_all) if (t == 1) or (t % K) == 0]
            if step_tokens_all:
                keep_idx.append(0)  # be robust if the first value isn't exactly 1
                keep_idx.append(len(step_tokens_all) - 1)  # always include final
                keep_idx = sorted(set(keep_idx))
            # Apply filtering
            joints = [joints_all[i] for i in keep_idx] if keep_idx else []
            step_tokens = [step_tokens_all[i] for i in keep_idx] if keep_idx else []

            joints_per_sample.append(joints)
            steps_per_sample.append(step_tokens)
        group_joint_dt = time.perf_counter() - group_joint_t0 if row_indices else 0.0

        # track per-sample progression (current step index) and histories
        adv_idx = [0] * len(row_indices)  # which step of this sample we are on
        # keep these names for compatibility but they are now unused/no-op
        pruned = [False] * len(row_indices)
        pruned_at_tokens = [None] * len(row_indices)
        exp_hist: List[List[float]] = [[] for _ in row_indices]
        exp_last: List[float | None] = [None] * len(row_indices)
        # track which samples we already printed a FINISH line for
        finished_reported: List[bool] = [False] * len(row_indices)

        # Baseline overrides: pre-decide pruning for 'no_prune' and 'keep_random_n'
        if args.eval_mode in {"no_prune", "keep_random_n"}:
            if args.eval_mode == "keep_random_n":
                N = max(0, int(args.keep_random_n or 0))
                # Derive a stable integer-ish seed per group
                try:
                    pid_int = int(pid)  # numeric group id
                except Exception:
                    pid_int = abs(hash(str(pid))) % (2**31)
                rng = np.random.default_rng(args.random_seed + pid_int)
                if N < len(row_indices):
                    keep_indices = set(rng.choice(len(row_indices), size=N, replace=False).tolist())
                else:
                    keep_indices = set(range(len(row_indices)))
                for si in range(len(row_indices)):
                    if si not in keep_indices:
                        pruned[si] = True
                        pruned_at_tokens[si] = 0
            # 'no_prune' keeps all (no action needed)

        # Simulate intervals (advance to the max available step index over samples)
        # Iterate until every sample reaches its last step or we run out of budget.
        total_iters = sum(max(0, len(s) - 1) for s in steps_per_sample)
        group_obj_t0 = time.perf_counter()
        # Count only iterations that actually advanced ≥1 sample.
        ticks_used = 0
        _do_zip_logs = bool(args.zip_logs and (logged_groups < int(args.zip_logs_groups)))
        _did_any_logging_for_group = False
        prev_S, prev_B = None, None  # remember last active set and planned future prune bin
        for step_idx in range(total_iters):
            # Early exit if all samples are finished.
            if all((len(steps_per_sample[si]) == 0) or (adv_idx[si] >= len(steps_per_sample[si]) - 1)
                   for si in range(len(row_indices))):
                break
            # Build current joint list: for samples without this step or already pruned, keep last available or degenerate?
            # We use last available prediction for those with fewer steps; and degenerate [pruned] for already pruned.
            current = []
            for si, joints in enumerate(joints_per_sample):
                # joint at this sample's current progress
                j = (joints[adv_idx[si]] if joints else np.full((V, L), 0.0, dtype=np.float64))
                # expectation for history (record current state)
                e_val, _ = _expected_values(j, reward_values, length_bins)
                exp_hist[si].append(e_val)
                current.append(j)

            # Build an "unfinished now" mask using observed tokens vs final length
            # tokens_now = tokens seen by this step for sample i
            row_lengths = [int(df.loc[ridx].get("length", 0) or 0) for ridx in row_indices]

            unfinished_mask_now = np.zeros(len(row_indices), dtype=bool)
            for si in range(len(row_indices)):
                # current observed tokens for this sample at its current progress
                tokens_now = (steps_per_sample[si][adv_idx[si]] if steps_per_sample[si] else 0)
                # unfinished iff observed tokens so far < final length
                unfinished_mask_now[si] = (tokens_now < row_lengths[si])

            # ---- Normalization denominator: estimated avg tokens/sample (pre-mod joints) ----
            # Use raw per-step joints (not the 'current' list which collapses already-pruned).
            mids = [(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(L)]
            mids_np = np.array(mids, dtype=np.float64)
            total_expected_final_tokens = 0.0
            n_group = len(row_indices)
            for si in range(n_group):
                # tokens_now for this sample at its current progress (aligned, independent of our actions)
                tnow = (steps_per_sample[si][adv_idx[si]] if steps_per_sample[si] else 0)
                # pre-mod joint at this current step
                if joints_per_sample[si]:
                    j_pre = joints_per_sample[si][adv_idx[si]]
                    tm_pre = j_pre.sum(axis=0).astype(np.float64)
                    e_rem_pre = float(np.dot(tm_pre, mids_np))
                else:
                    e_rem_pre = 0.0
                total_expected_final_tokens += float(tnow) + e_rem_pre
            norm_tokens_per_sample = (total_expected_final_tokens / max(1, n_group))
            # Use per-prompt reference normalization if available
            try:
                prompt_id_for_group = grp[base_group_key].iloc[0]
            except Exception:
                prompt_id_for_group = None
            norm_for_obj = float(prompt_norm_denom_for.get(prompt_id_for_group, norm_tokens_per_sample)) if 'prompt_norm_denom_for' in locals() else norm_tokens_per_sample

            # --- FINISH events (natural completion), logged once per sample ---
            # A sample "finishes" when we reach its last aligned step and it wasn't pruned.
            if _do_zip_logs:
                for si in range(len(current)):
                    if finished_reported[si] or pruned[si]:
                        continue
                    # last aligned step for this sample?
                    if (len(steps_per_sample[si]) > 0) and (adv_idx[si] == len(steps_per_sample[si]) - 1):
                        if args.zip_logs_sample_idx >= 0 and si != int(args.zip_logs_sample_idx):
                            continue
                        # final expected value from history we just appended
                        exp_final = exp_hist[si][-1] if exp_hist[si] else None
                        row = df.loc[row_indices[si]]
                        length_final = int(row.get("length", 0) or 0)
                        finished_flag = bool(row.get("finished", False))
                        correct_flag = bool(row.get("correct", False))
                        val_print = (f"{float(exp_final):.6f}" if exp_final is not None else "nan")
                        correct_symbol = "✓" if correct_flag else "✗"
                        print(f"[FINISH] sample {si} len={length_final} finished={finished_flag} E[value]_final={val_print} correct={correct_flag} [{correct_symbol}]", flush=True)
                        if args.viz_pruning:
                            # Show the final predicted joint grid at completion
                            try:
                                log_joint_distribution_grid(
                                    current[si],
                                    length_bins, L, V, reward_values,
                                    title_prefix=f"FINAL (sample {si})"
                                )
                            except Exception:
                                pass
                        finished_reported[si] = True

            # ---------------------------
            # 1) Threshold-based pruning
            # ---------------------------
            threshold_pruned_now: set[int] = set()
            if (args.eval_mode == "policy") and (float(args.ev_threshold) >= 0):
                rv_np = np.asarray(reward_values, dtype=np.float64)
                for si in range(len(current)):
                    if pruned[si] or (not unfinished_mask_now[si]):
                        continue  # already pruned or already finished -> skip
                    rm_now = current[si].sum(axis=1).astype(np.float64)
                    e_val_now = float(np.dot(rm_now, rv_np))
                    if e_val_now < float(args.ev_threshold) - TOL:
                        pruned[si] = True
                        # record prune length as the tokens observed for this step
                        tkns = (
                            steps_per_sample[si][step_idx]
                            if step_idx < len(steps_per_sample[si]) else
                            (steps_per_sample[si][-1] if steps_per_sample[si] else 0)
                        )
                        pruned_at_tokens[si] = int(tkns)
                        threshold_pruned_now.add(si)
                # Optional logging for threshold prunes
                if _do_zip_logs and threshold_pruned_now:
                    print(f"[ACTION] Threshold-pruning {len(threshold_pruned_now)} sample(s) at step {step_idx} "
                          f"(threshold={float(args.ev_threshold):.6f}): {sorted(threshold_pruned_now)}", flush=True)
                    if args.debug:
                        for si in sorted(threshold_pruned_now):
                            if args.zip_logs_sample_idx >= 0 and si != int(args.zip_logs_sample_idx):
                                continue
                            rm_now = current[si].sum(axis=1).astype(np.float64)
                            mids_np_dbg = np.asarray([(length_bins[i] + length_bins[i+1]) / 2.0 for i in range(L)], dtype=np.float64)
                            tm_now = current[si].sum(axis=0).astype(np.float64)
                            e_val_now = float(np.dot(rm_now, rv_np))
                            e_tok_now = float(np.dot(tm_now, mids_np_dbg))
                            length_now = (
                                int(steps_per_sample[si][step_idx]) if step_idx < len(steps_per_sample[si])
                                else int(steps_per_sample[si][-1] if steps_per_sample[si] else 0)
                            )
                            print(f"  • [THRESHOLD] sample {si}: len={length_now} "
                                  f"E[value]={e_val_now:.6f} < {float(args.ev_threshold):.6f}  "
                                  f"E[tokens]={e_tok_now:.2f}", flush=True)

            if args.eval_mode == "policy":
                # Choose action (same combinatorics as online)
                _prev = None if prev_S is None else (np.isin(np.arange(len(current)), list(prev_S)) & unfinished_mask_now).tobytes()
                globals().update({"PREV_ACTIVE_MASK_BYTES": _prev, "SWITCH_PENALTY": float(args.switch_penalty)})
                S, Bfuture, _ = _choose_action(
                    current, pruned, reward_values, length_bins, args.c_pt, args.c_seq,
                    unfinished_mask_now, float(args.selection_geom_alpha),
                    norm_tokens_per_sample=norm_for_obj
                )
                if args.max_active is not None:
                    act = [i for i in sorted(S) if unfinished_mask_now[i]]
                    S -= set(act[int(args.max_active):])
                # Early-terminate: if no unfinished item is kept active, there's nothing left to decode.
                active_now = [i for i in sorted(S) if unfinished_mask_now[i]]
                if len(active_now) == 0:
                    break
                changed = (prev_S is None) or (S != prev_S) or (Bfuture != prev_B)  # detect decision change

                # --- Event-gated logging: first step OR right before we actually prune now ---
                unpruned = [i for i, pflag in enumerate(pruned) if not pflag]
                # Only unfinished items are eligible to be pruned.
                pruned_now = {i for i in unpruned if unfinished_mask_now[i] and (i not in S)}
                is_initial_step = (step_idx == 0)
                should_log = _do_zip_logs and changed
                if should_log:
                    # Build modified marginals for chosen action (to compute the printed components)
                    reward_values_np = np.array(reward_values, dtype=np.float64)
                    mids = [(length_bins[i]+length_bins[i+1])/2.0 for i in range(L)]
                    mids_np = np.array(mids, dtype=np.float64)

                    def _modified_marginals_for(sample_idx: int, prune_bin: int | None) -> Tuple[np.ndarray, np.ndarray]:
                        if prune_bin is None:
                            rm = current[sample_idx].sum(axis=1).astype(np.float64)
                            tm = current[sample_idx].sum(axis=0).astype(np.float64)
                            return rm, tm
                        # Collapse mass as in online: bins >= prune_bin moved to reward≈0 at that bin
                        zero_idx = int(np.argmin(np.abs(reward_values_np - 0.0)))
                        j = current[sample_idx]
                        rm_prefix = j[:, :prune_bin].sum(axis=1).astype(np.float64)
                        collapsed = j[:, prune_bin:].sum().astype(np.float64)
                        rm = rm_prefix.copy()
                        if collapsed > 0:
                            rm[zero_idx] += float(collapsed)
                        tm = np.zeros_like(j.sum(axis=0), dtype=np.float64)
                        if prune_bin > 0:
                            tm[:prune_bin] = j.sum(axis=0)[:prune_bin]
                        tm[prune_bin] = float(collapsed)
                        return rm, tm

                    # Create modified distributions for chosen action (2D for viz) + marginals
                    chosen_modified_2d: List[np.ndarray] = []
                    marg_vals_list: List[np.ndarray] = []
                    marg_tok_list: List[np.ndarray] = []
                    for i_mod in range(len(current)):
                        if pruned[i_mod] or (i_mod not in S):
                            after = _modify_for_prune(current[i_mod], 0, reward_values)  # prune now
                        else:
                            # keep with planned future prune (if any)
                            after = current[i_mod] if (Bfuture is None) else _modify_for_prune(current[i_mod], int(Bfuture), reward_values)
                        chosen_modified_2d.append(after)
                        marg_vals_list.append(after.sum(axis=1).astype(np.float64))
                        marg_tok_list.append(after.sum(axis=0).astype(np.float64))

                    # Compute components to print (same semantics as the objective):
                    #   • value uses ALL kept (S & ~pruned)
                    #   • token-cost uses only ACTIVE kept (unfinished ∧ kept)
                    def _group_obj(mvals, mtok, keep_mask_bool, active_mask_bool):
                        kept_idxs_loc = np.nonzero(keep_mask_bool)[0].tolist()
                        if kept_idxs_loc:
                            emax_val_loc = _expected_max([mvals[i] for i in kept_idxs_loc], bin_values=reward_values)
                            e_vals_kept_loc = [float(np.dot(mvals[i], reward_values_np)) for i in kept_idxs_loc]
                            eavg_val_loc = float(sum(e_vals_kept_loc) / len(e_vals_kept_loc)) if e_vals_kept_loc else 0.0
                        else:
                            emax_val_loc = 0.0
                            eavg_val_loc = 0.0
                        e_value_term_loc = _geom_interpolate(eavg_val_loc, emax_val_loc, float(args.selection_geom_alpha))
                        active_idxs_loc = np.nonzero(active_mask_bool)[0].tolist()
                        if active_idxs_loc:
                            active_tok_margs_loc = [mtok[i] for i in active_idxs_loc]
                            e_max_tok_loc = _expected_max(marginals=active_tok_margs_loc, bin_values=mids)
                            e_sum_rem_loc = float(sum(float(np.dot(active_tok_margs_loc[j], mids_np)) for j in range(len(active_tok_margs_loc))))
                        else:
                            e_max_tok_loc = 0.0
                            e_sum_rem_loc = 0.0
                        denom_loc = max(float(norm_for_obj), 1e-12)
                        g_loc = e_value_term_loc - args.c_pt*(e_sum_rem_loc/denom_loc) - args.c_seq*(e_max_tok_loc/denom_loc)
                        return g_loc, e_value_term_loc, e_sum_rem_loc, e_max_tok_loc, emax_val_loc, eavg_val_loc

                    # Build masks consistent with the objective:
                    keep_mask_eval   = np.zeros(len(current), dtype=bool)
                    if len(current):
                        keep_mask_eval[list(S)] = True
                    keep_mask_eval &= (~np.array(pruned, dtype=bool))
                    active_mask_eval = keep_mask_eval & unfinished_mask_now
                    best_group_obj, e_value_term, e_sum_rem, e_max_tok, emax_val, eavg_val = _group_obj(marg_vals_list, marg_tok_list, keep_mask_eval, active_mask_eval)
                    print(f"[OBJECTIVE] step={step_idx}  G={best_group_obj:.6f}  kept={sorted(S)}  future_bin={Bfuture}", flush=True)

                    # Emit one line per sample (match online print), but only at first or pre-prune steps
                    for si in range(len(current)):
                        if args.zip_logs_sample_idx >= 0 and si != int(args.zip_logs_sample_idx):
                            continue
                        length_now = int(steps_per_sample[si][adv_idx[si]] if steps_per_sample[si] else 0)
                        rm_now = current[si].sum(axis=1).astype(np.float64)
                        tm_now = current[si].sum(axis=0).astype(np.float64)
                        e_val_now = float(np.dot(rm_now, reward_values_np))
                        e_tok_now = float(np.dot(tm_now, mids_np))
                        # Get correctness information for this sample
                        row_si = df.loc[row_indices[si]]
                        correct_flag_si = bool(row_si.get("correct", False))
                        correct_symbol_si = "✓" if correct_flag_si else "✗"
                        print(
                            f"Sample {si} length={length_now} E[value]={e_val_now} E[tokens]={e_tok_now} "
                            f"correct={correct_flag_si} [{correct_symbol_si}] "
                            f"Best G(S): {best_group_obj} (E[value term]={e_value_term:.6f}, "
                            f"E[max]={emax_val:.6f}, E[avg]={eavg_val:.6f}, "
                            f"E[Σ rem]={e_sum_rem:.2f} (norm={e_sum_rem/max(norm_for_obj,1e-12):.4f}), "
                            f"E[max rem]={e_max_tok:.2f} (norm={e_max_tok/max(norm_for_obj,1e-12):.4f}), "
                            f"norm_denom={norm_for_obj:.4f}) "
                            f"with kept set {set(np.nonzero(keep_mask_eval)[0])} and future bin {Bfuture}",
                            flush=True
                        )

                    # Optional: ACTION lines (like online debug)
                    if args.debug and unpruned:
                        if pruned_now:
                            print(f"[ACTION] Pruning {len(pruned_now)} samples now: {pruned_now}", flush=True)
                        if Bfuture is not None:
                            print(f"[ACTION] Planning to prune all remaining at bin {Bfuture} (~{length_bins[int(Bfuture)]} tokens)", flush=True)

                    # Initial visual context: show BEFORE/AFTER at the very first interval
                    # Use the chosen action (S, Bfuture) to construct the AFTER view.
                    if is_initial_step and args.viz_pruning:
                        for si in range(len(current)):
                            if args.zip_logs_sample_idx >= 0 and si != int(args.zip_logs_sample_idx):
                                continue
                            # Decide the "planned" action for this sample at t=0
                            if pruned[si]:
                                prune_plan_bin = 0  # already pruned (shouldn't usually happen in policy mode at t=0)
                            elif si in S:
                                prune_plan_bin = (None if Bfuture is None else int(Bfuture))
                            else:
                                prune_plan_bin = 0  # prune now
                            before = current[si]
                            after = before if (prune_plan_bin is None) else _modify_for_prune(before, prune_plan_bin, reward_values)
                            try:
                                log_before_after_pruning(
                                    before,
                                    after,
                                    length_bins, L, V, reward_values,
                                    sample_idx=si,
                                    prune_at_bin=(None if prune_plan_bin is None else int(prune_plan_bin))
                                )
                            except Exception:
                                pass
                    elif is_initial_step:
                        # If --viz-pruning wasn't requested, keep a lightweight single grid for context
                        for si in range(len(current)):
                            if args.zip_logs_sample_idx >= 0 and si != int(args.zip_logs_sample_idx):
                                continue
                            try:
                                log_joint_distribution_grid(
                                    current[si],
                                    length_bins, L, V, reward_values,
                                    title_prefix=f"INITIAL (sample {si})"
                                )
                            except Exception:
                                pass

                    # Right-before-prune rationale & grids: only for samples being pruned now
                    if pruned_now:
                        print(f"[PRE-PRUNE] group={pid} step_idx={step_idx} future_bin={Bfuture}", flush=True)
                        for si in sorted(pruned_now):
                            if args.zip_logs_sample_idx >= 0 and si != int(args.zip_logs_sample_idx):
                                continue
                            length_now = int(steps_per_sample[si][adv_idx[si]] if steps_per_sample[si] else 0)
                            rm_now = current[si].sum(axis=1).astype(np.float64)
                            tm_now = current[si].sum(axis=0).astype(np.float64)
                            e_val_now = float(np.dot(rm_now, reward_values_np))
                            e_tok_now = float(np.dot(tm_now, mids_np))
                            # Counterfactual: keep si active (with same Bfuture) instead of pruning now
                            keep_rm, keep_tm = _modified_marginals_for(si, (None if Bfuture is None else int(Bfuture)))
                            mvals_alt = list(marg_vals_list)
                            mtok_alt = list(marg_tok_list)
                            mvals_alt[si] = keep_rm
                            mtok_alt[si] = keep_tm
                            active_mask_alt = active_mask_eval.copy()
                            # Only set si to True if it's also unfinished (for cost terms)
                            if unfinished_mask_now[si]:
                                active_mask_alt[si] = True
                            keep_mask_alt = keep_mask_eval.copy(); keep_mask_alt[si] = True
                            # _group_obj now returns 6 values; only the objective is needed here.
                            g_alt, *_ = _group_obj(mvals_alt, mtok_alt, keep_mask_alt, active_mask_alt)
                            delta_g = g_alt - best_group_obj
                            print(f"  • sample {si}: len={length_now} E[value]={e_val_now:.6f} E[tokens]={e_tok_now:.2f}  ΔG_if_kept={delta_g:.6f}", flush=True)
                            # On step 0 we've already shown BEFORE/AFTER for all samples above,
                            # so avoid duplicating grids here (still print ΔG rationale).
                            if args.viz_pruning and not is_initial_step:
                                try:
                                    log_before_after_pruning(
                                        current[si],
                                        _modify_for_prune(current[si], 0, reward_values),
                                        length_bins, L, V, reward_values,
                                        sample_idx=si, prune_at_bin=0
                                    )
                                except Exception:
                                    pass

                        # Also show context for the ACTIVE set (unpruned) at this event:
                        # their "planned" joint distributions after applying future_bin (if any).
                        if args.viz_pruning:
                            for si in sorted(S):
                                if args.zip_logs_sample_idx >= 0 and si != int(args.zip_logs_sample_idx):
                                    continue
                                try:
                                    after_active = chosen_modified_2d[si]
                                    suffix = f"(future_bin={Bfuture})" if Bfuture is not None else "(no future prune)"
                                    log_joint_distribution_grid(
                                        after_active,
                                        length_bins, L, V, reward_values,
                                        title_prefix=f"ACTIVE planned {suffix} (sample {si})"
                                    )
                                except Exception:
                                    pass

                    # Optional: viz of E[max] distributions + inputs (only at event times)
                    if args.viz_max:
                        # Visualize with the same semantics as the objective:
                        #   • Value-side: KEPT + FINISHED (exclude pruned)
                        #   • Token-side: ACTIVE ONLY (kept ∧ unfinished)
                        mids_list = [(length_bins[i] + length_bins[i+1]) / 2 for i in range(L)]

                        kept_idxs   = np.nonzero(keep_mask_eval)[0].tolist()
                        active_idxs = np.nonzero(active_mask_eval)[0].tolist()

                        # ---- Value-side (kept + finished; exclude pruned) ----
                        mvals_kept = [marg_vals_list[i] for i in kept_idxs]
                        prob_bins_val, vals_sorted, order_idx = _max_bin_probs_np(mvals_kept, reward_values)
                        re_marg_vals = _reorder_marginals(mvals_kept, order_idx)
                        log_max_distribution(prob_bins_val, vals_sorted, title="Distribution of max value (kept+finished)")
                        log_max_input_distributions(re_marg_vals, vals_sorted, ids=kept_idxs, title_prefix="Inputs for max (value)")

                        # ---- Token-side (ACTIVE ONLY) ----
                        mtok_active = [marg_tok_list[i] for i in active_idxs]
                        prob_bins_tok, mids_sorted, order_idx_t = _max_bin_probs_np(mtok_active, mids_list)
                        re_marg_tok = _reorder_marginals(mtok_active, order_idx_t)
                        log_max_distribution(prob_bins_tok, mids_sorted, length_bins=length_bins, title="Distribution of max remaining tokens (active only)")
                        log_max_input_distributions(re_marg_tok, mids_sorted, ids=active_idxs, length_bins=length_bins, title_prefix="Inputs for max (remaining tokens) [ACTIVE]")

                    _did_any_logging_for_group = True

                # --- No-harm pruning guard (always active, even when not logging) ---
                # If including a candidate does not STRICTLY reduce the group objective,
                # keep it (prevents arbitrary prunes on ties / numerical jitter).
                if pruned_now:
                    # Prepare per-sample marginals under the CHOSEN future plan
                    reward_values_np = np.array(reward_values, dtype=np.float64)
                    mids = [(length_bins[i]+length_bins[i+1])/2.0 for i in range(L)]
                    mids_np = np.array(mids, dtype=np.float64)

                    def _after_keep_rm_tm(j2d):
                        if Bfuture is None:
                            a = j2d
                        else:
                            a = _modify_for_prune(j2d, int(Bfuture), reward_values)
                        return a.sum(axis=1).astype(np.float64), a.sum(axis=0).astype(np.float64)

                    def _after_prune_rm_tm(j2d):
                        a = _modify_for_prune(j2d, 0, reward_values)
                        return a.sum(axis=1).astype(np.float64), a.sum(axis=0).astype(np.float64)

                    rm_keep = []
                    tm_keep = []
                    rm_prune = []
                    tm_prune = []
                    for jj in range(len(current)):
                        rmk, tmk = _after_keep_rm_tm(current[jj])
                        rmp, tmp = _after_prune_rm_tm(current[jj])
                        rm_keep.append(rmk); tm_keep.append(tmk)
                        rm_prune.append(rmp); tm_prune.append(tmp)

                    def _g_from_masks(keep_mask_bool: np.ndarray, active_mask_bool: np.ndarray) -> float:
                        kept_idxs_loc = np.nonzero(keep_mask_bool)[0].tolist()
                        if kept_idxs_loc:
                            emax_val_loc = _expected_max([rm_keep[i] for i in kept_idxs_loc], bin_values=reward_values)
                            e_vals_kept_loc = [float(np.dot(rm_keep[i], reward_values_np)) for i in kept_idxs_loc]
                            eavg_val_loc = float(sum(e_vals_kept_loc) / len(e_vals_kept_loc)) if e_vals_kept_loc else 0.0
                        else:
                            emax_val_loc = 0.0
                            eavg_val_loc = 0.0
                        e_value_term_loc = _geom_interpolate(eavg_val_loc, emax_val_loc, float(args.selection_geom_alpha))
                        active_idxs_loc = np.nonzero(active_mask_bool)[0].tolist()
                        if active_idxs_loc:
                            e_max_tok_loc = _expected_max([tm_keep[i] for i in active_idxs_loc], bin_values=mids)
                            e_sum_rem_loc = float(sum(float(np.dot(tm_keep[i], mids_np)) for i in active_idxs_loc))
                        else:
                            e_max_tok_loc = 0.0
                            e_sum_rem_loc = 0.0
                        denom_loc = max(float(norm_for_obj), 1e-12)
                        return e_value_term_loc - args.c_pt * (e_sum_rem_loc/denom_loc) - args.c_seq * (e_max_tok_loc/denom_loc)

                    # Compute objective for the selected masks
                    keep_mask_eval = np.zeros(len(current), dtype=bool)
                    if len(current):
                        keep_mask_eval[list(S)] = True
                    keep_mask_eval &= (~np.array(pruned, dtype=bool))
                    active_mask_eval = keep_mask_eval & unfinished_mask_now
                    g_best = _g_from_masks(keep_mask_eval, active_mask_eval)

                    # Try adding back each to-be-pruned candidate; keep if no strict loss
                    added_any = False
                    for si in sorted(pruned_now):
                        km_try = keep_mask_eval.copy(); km_try[si] = True
                        am_try = km_try & unfinished_mask_now
                        g_try = _g_from_masks(km_try, am_try)
                        if g_try >= g_best - TOL:
                            # keep it; update best baseline for subsequent comparisons
                            keep_mask_eval[si] = True
                            active_mask_eval = keep_mask_eval & unfinished_mask_now
                            g_best = _g_from_masks(keep_mask_eval, active_mask_eval)
                            added_any = True
                    if added_any:
                        # Update S to match the guarded mask
                        S = set(np.nonzero(keep_mask_eval)[0].tolist())
                        # Recompute pruned_now under guarded S
                        pruned_now = {i for i in unpruned if unfinished_mask_now[i] and (i not in S)}

                # Advance exactly the kept (active) set S by one interval; others are paused for this tick.
                if args.max_active is not None:
                    act = [i for i in sorted(S) if unfinished_mask_now[i]]
                    S -= set(act[int(args.max_active):])
                # Advance only the unfinished active items; count a tick if any progressed.
                advanced_this_tick = False
                # Reuse the filtered list from above
                for si in active_now:
                    if steps_per_sample[si] and adv_idx[si] < (len(steps_per_sample[si]) - 1):
                        adv_idx[si] += 1
                        advanced_this_tick = True
                if advanced_this_tick:
                    ticks_used += 1
                prev_S, prev_B = S, Bfuture  # remember decision for next tick
            else:
                # In baselines we do not change pruning decisions over time
                pass
        group_obj_dt = time.perf_counter() - group_obj_t0 if row_indices else 0.0
        # Count this group only if we actually logged anything for it
        if _do_zip_logs and _did_any_logging_for_group:
            logged_groups += 1

        # Build final rows based on progressed length; no permanent pruning
        ticks_used_tokens = int(ticks_used) * int(args.step_size)
        for i, ridx in enumerate(row_indices):
            row = df.loc[ridx]
            out_ids_full = _parse_token_ids(row["output_token_ids"])
            prompt_ids = _parse_token_ids(row["prompt_token_ids"]) 
            # Initialize phases with per-row shares of earlier group phases
            phases: dict[str, float] = {
                "joint": (group_joint_dt / max(1, len(row_indices))),
                "objective": (group_obj_dt / max(1, len(row_indices))),
            }

            decode_t0 = time.perf_counter()
            # Keep up to the progressed length (tokens at adv_idx)
            keep = (steps_per_sample[i][adv_idx[i]] if steps_per_sample[i] else 0)
            out_ids = out_ids_full[:keep]
            text = tok.decode(out_ids, skip_special_tokens=True).strip()
            length = len(out_ids)
            finished = (length >= len(out_ids_full))
            phases["decode"] = time.perf_counter() - decode_t0

            # expected reward final = last in history (or None)
            exp_last[i] = exp_hist[i][-1] if exp_hist[i] else None

            write_t0 = time.perf_counter()
            rec = {
                "prompt_idx": int(row["prompt_idx"]),
                "prompt": row["prompt"],
                "answer": row.get("answer", None),
                "response": text,
                "length": length,
                "original_length": int(len(out_ids_full)),  # NEW: baseline full length (no prune)
                "finished": bool(finished),
                "pruned": False,
                "expected_reward": float(exp_last[i]) if exp_last[i] is not None else None,
                "reward_values": row["reward_values"],   # consistent with inference.py expectations
                "expected_tokens": None,                  # optional; could compute from last joint if desired
                "expected_reward_history": exp_hist[i],
                "distribution_logits_history": [],        # placeholder for schema parity
                "prompt_token_ids": prompt_ids,
                "output_token_ids": out_ids,
                # NEW: carry cached labels if available; pruned/unfinished => not used
                "correct": bool(row.get("correct", False)) if finished else False,
                "extracted_answer": (row.get("extracted_answer", None) if finished else None),
                "mode": (row.get("__mode__") if "__mode__" in df.columns else None),
                "ticks_used": ticks_used_tokens,  # real serial latency in tokens for this try
            }
            # Optional grouping columns to support per-try evaluation downstream
            if has_try_col:
                rec["try_idx"] = int(row["try_idx"])
            if has_group_col:
                rec["group_id"] = row["group_id"]
            final_rows.append(rec)
            phases["write"] = time.perf_counter() - write_t0

            # Log progress for this row
            orig_len = len(out_ids_full)
            pruned_len = len(out_ids)
            # Get correctness from the rec we just built
            is_correct = rec["correct"]
            stats.log_row(orig_len=orig_len, pruned_len=pruned_len, phases=phases, correct=is_correct, extra=None)

    table = pa.Table.from_pandas(pd.DataFrame(final_rows))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    print(f"✓ wrote {len(final_rows)} rows to {args.out}")
    # Close stats writer if active
    stats.close()

    # ---------------- METRICS (no model calls; relies on prelabels) ----------------
    if args.compute_metrics:
        df_final = pd.DataFrame(final_rows)
        metrics: Dict[str, object] = {
            "labeled_parquet": [os.path.abspath(p) for p in labeled_paths],
            "joint_npz": [os.path.abspath(p) for p in npz_paths],
            "results_parquet": os.path.abspath(args.out),
            "c_pt": float(args.c_pt),
            "c_seq": float(args.c_seq),
            "step_size": int(args.step_size),
            "eval_mode": str(args.eval_mode),
            "samples_per_try": (int(args.samples_per_try) if args.samples_per_try is not None else None),
            "use_consistency": bool(args.use_consistency),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "selection_geom_alpha": float(args.selection_geom_alpha),
            # NEW: surface calibration flags and EV threshold in metrics
            "ev_threshold": float(args.ev_threshold),
            "joint_temp": float(args.joint_temp),
            "joint_top_p": float(args.joint_top_p),
            # NEW: aggregation mode
            "aggregation_mode": "macro_equal_weight_prompts",
        }

        # How many rows/tries we dropped because they were incomplete
        metrics["dropped_incomplete_tries"] = {
            "num_partial_tries_dropped": int(dropped_partial_tries),
            "num_rows_dropped": int(dropped_rows_incomplete),
        }

        # Prefer stable prompt id
        prompt_col = "prompt_idx" if "prompt_idx" in df_final.columns else "prompt"
        # Each try is either group_id (if present) or a single-try prompt
        try_col = "group_id" if "group_id" in df_final.columns else prompt_col

        # Helpers -------------------------------------------------------------------
        def has_thinking(token_ids) -> bool:
            if isinstance(token_ids, str):
                try:
                    token_ids = eval(token_ids)
                except Exception:
                    return False
            try:
                return args.thinking_token_id in token_ids
            except Exception:
                return False

        # Ensure prompt column is 1-D scalar values (avoid ValueError on set_index)
        import numpy as _np
        def _ensure_scalar_prompt(x):
            if isinstance(x, (list, tuple, _np.ndarray)):
                return (x[0] if len(x) > 0 else "")
            return x
        df_final[prompt_col] = df_final[prompt_col].apply(_ensure_scalar_prompt)

        # Map each try to its prompt (index-aligned for groupby-on-Series)
        try_to_prompt = (
            df_final[[try_col, prompt_col]]
            .drop_duplicates()
            .set_index(try_col)[prompt_col]
        )

        by_try = df_final.groupby(try_col)

        def macro_mean_from_try_series(s: pd.Series) -> float:
            """
            Equal-weighted mean across prompts: average s within each prompt (mean over its tries),
            then average those prompt means over prompts. NaNs are skipped within each stage.
            """
            if s.empty:
                return 0.0
            per_prompt = s.groupby(try_to_prompt).mean()
            return float(per_prompt.mean(skipna=True))

        def macro_median_from_try_series(s: pd.Series) -> float:
            if s.empty:
                return 0.0
            per_prompt = s.groupby(try_to_prompt).median()
            return float(per_prompt.median(skipna=True))

        # High-level counts ---------------------------------------------------------
        num_prompts = int(df_final[prompt_col].nunique())
        num_tries   = int(df_final[try_col].nunique())
        tries_per_prompt = df_final.groupby(prompt_col)[try_col].nunique()
        metrics.update({
            "num_prompts": num_prompts,
            "num_tries": num_tries,
            "tries_per_prompt_summary": {
                "mean": float(tries_per_prompt.mean()),
                "median": float(tries_per_prompt.median()),
                "min": int(tries_per_prompt.min()) if not tries_per_prompt.empty else 0,
                "max": int(tries_per_prompt.max()) if not tries_per_prompt.empty else 0,
            }
        })
        print(f"Prompts: {num_prompts} | Tries: {num_tries} | "
              f"tries/prompt (mean/median/min/max): "
              f"{metrics['tries_per_prompt_summary']['mean']:.2f}/"
              f"{metrics['tries_per_prompt_summary']['median']:.2f}/"
              f"{metrics['tries_per_prompt_summary']['min']}/"
              f"{metrics['tries_per_prompt_summary']['max']}")

        # Finished / pruned rates (macro vs micro) ---------------------------------
        finished_frac_try = by_try["finished"].mean()            # per-try fraction finished
        pruned_frac_try   = by_try["pruned"].mean()              # per-try fraction pruned
        finished_pct_macro = macro_mean_from_try_series(finished_frac_try) * 100.0
        pruned_pct_macro   = macro_mean_from_try_series(pruned_frac_try) * 100.0

        # Micro reference (over rows)
        finished_pct_micro = float(df_final["finished"].mean() * 100.0 if len(df_final) else 0.0)
        pruned_pct_micro   = float(df_final["pruned"].mean() * 100.0 if len(df_final) else 0.0)
        print(f"Finished responses (macro, per-prompt): {finished_pct_macro:.2f}%   "
              f"[micro rows: {finished_pct_micro:.2f}%]")
        print(f"Pruned responses   (macro, per-prompt): {pruned_pct_macro:.2f}%   "
              f"[micro rows: {pruned_pct_micro:.2f}%]")

        metrics.update({
            "finished_percentage": finished_pct_macro,
            "finished_percentage_micro": finished_pct_micro,
            "pruned_percentage": pruned_pct_macro,
            "pruned_percentage_micro": pruned_pct_micro,
        })

        # Token/latency aggregates --------------------------------------------------
        try_total_tokens = by_try["length"].sum()
        # n-latency = serial tokens spent; we recorded 'ticks_used' per row (already in tokens)
        if "ticks_used" not in df_final.columns:
            df_final["ticks_used"] = 0
        try_n_latency = by_try["ticks_used"].max().fillna(0).astype(int)
        # Fallback for fast baselines / legacy runs without ticks_used
        try_n_latency = try_n_latency.where(try_n_latency > 0, by_try["length"].max())
        try_finished_samples = by_try["finished"].sum()

        avg_total_tokens_macro   = macro_mean_from_try_series(try_total_tokens)
        avg_n_latency_macro      = macro_mean_from_try_series(try_n_latency)
        avg_finished_per_prompt  = macro_mean_from_try_series(try_finished_samples)

        print(f"Average total tokens (macro, per-prompt):      {int(avg_total_tokens_macro)}")
        print(f"Average n-latency    (macro, per-prompt):      {avg_n_latency_macro:.2f}")
        print(f"Average finished samples per prompt (macro):   {avg_finished_per_prompt:.2f}")

        metrics.update({
            "average_total_tokens": int(avg_total_tokens_macro),
            "average_n_latency": float(avg_n_latency_macro),
            "average_finished_samples_per_prompt": float(avg_finished_per_prompt),
        })

        # ---------------- Normalized cost & latency (thinking-baseline per prompt from INPUTS) ----------------
        # We normalize each try by the average token length of THINKING samples for that prompt
        # computed from the ORIGINAL merged inputs (df_inputs_all), independent of what was run.
        if ("original_length" in df_final.columns) and ("mode" in df_final.columns):
            pkey_col_inputs = base_group_key
            pkey_col_final  = prompt_col
            df_inputs_all["_pkey"] = df_inputs_all[pkey_col_inputs].astype(str)
            df_final["_pkey"]      = df_final[pkey_col_final].astype(str)

            th_base = (df_inputs_all.loc[df_inputs_all["__mode__"] == "thinking"].groupby("_pkey")["length"].mean())
            nt_base = (df_inputs_all.loc[df_inputs_all["__mode__"] == "nonthinking"].groupby("_pkey")["length"].mean())
            p_mean  = df_inputs_all.groupby("_pkey")["length"].mean()

            thinking_baseline_by_pkey = th_base.combine_first(nt_base).combine_first(p_mean)

            try_to_prompt_pkey = try_to_prompt.astype(str)
            denom_by_try = try_to_prompt_pkey.map(thinking_baseline_by_pkey).astype(float)

            sum_len_after_per_try = by_try["length"].sum()
            n_per_try = by_try.size()

            safe_denom = denom_by_try.replace(0, np.nan)
            samples_used_norm_try = (sum_len_after_per_try / safe_denom)
            # Use the real n-latency (ticks×step_size), not max sample length:
            n_latency_norm_try    = (try_n_latency / safe_denom)
            frac_samples_used_try = (samples_used_norm_try / n_per_try).clip(0, 1)

            avg_samples_used_norm_macro    = macro_mean_from_try_series(samples_used_norm_try)
            median_samples_used_norm_macro = macro_median_from_try_series(samples_used_norm_try)
            avg_n_latency_norm_macro       = macro_mean_from_try_series(n_latency_norm_try)
            median_n_latency_norm_macro    = macro_median_from_try_series(n_latency_norm_try)
            avg_frac_samples_used_macro    = macro_mean_from_try_series(frac_samples_used_try)

            print("[Normalization] Using THINKING baseline per prompt from INPUTS; falling back to non-thinking only if a prompt has no thinking rows in the data.")
            print(f"Samples used (normalized, mean over prompts):   {avg_samples_used_norm_macro:.2f}")
            print(f"Samples used (normalized, median over prompts): {median_samples_used_norm_macro:.2f}")
            print(f"n-latency (normalized, mean over prompts):      {avg_n_latency_norm_macro:.2f}")
            print(f"n-latency (normalized, median over prompts):    {median_n_latency_norm_macro:.2f}")
            print(f"Fraction of samples used (mean over prompts):   {avg_frac_samples_used_macro*100:.2f}%")

            # Rows/tries by mode in RESULTS for transparency
            rows_by_mode = df_final["mode"].value_counts(dropna=False).to_dict()
            try_mode = df_final.groupby(try_col)["mode"].first()
            tries_by_mode = try_mode.value_counts(dropna=False).to_dict()
            print(f"Results rows by mode: {rows_by_mode}")
            print(f"Results tries by mode: {tries_by_mode}")

            # Persist to JSON
            metrics.update({
                "normalized_denominator_policy": "thinking_baseline_from_inputs",
                "avg_samples_used_normalized":    float(avg_samples_used_norm_macro),
                "median_samples_used_normalized": float(median_samples_used_norm_macro),
                "avg_n_latency_normalized":       float(avg_n_latency_norm_macro),
                "median_n_latency_normalized":    float(median_n_latency_norm_macro),
                "avg_fraction_of_samples_used":   float(avg_frac_samples_used_macro),
                "rows_by_mode": rows_by_mode,
                "tries_by_mode": tries_by_mode,
                "normalization_baseline_counts": {
                    "prompts_with_thinking_in_inputs": int((df_inputs_all.groupby("_pkey")["__mode__"].apply(lambda s: (s == "thinking").any())).sum()),
                    "prompts_without_thinking_in_inputs": int((df_inputs_all.groupby("_pkey")["__mode__"].apply(lambda s: ~(s == "thinking").any())).sum()),
                },
            })
            if 'dropped_leftover_rows_total' in locals():
                metrics["dropped_leftover_rows"] = int(dropped_leftover_rows_total)
        else:
            print("Skipping normalized cost/latency metrics: missing 'original_length' or 'mode'.")

        # Unpruned accuracy & Pass@N (macro) ---------------------------------------
        if "correct" in df_final.columns:
            # Accuracy among unpruned rows, averaged per try then per prompt
            unpruned = df_final.loc[~df_final["pruned"]]
            if not unpruned.empty:
                unpruned_acc_try = unpruned.groupby(try_col)["correct"].mean()
                unpruned_acc_macro = macro_mean_from_try_series(unpruned_acc_try) * 100.0
                print(f"Correctness (unpruned; macro over prompts): {unpruned_acc_macro:.2f}%")
                metrics["correctness_unpruned"] = float(unpruned_acc_macro)

            # Pass@N per try (any correct), then macro over prompts
            pass_at_n_try = by_try["correct"].any().astype(float)
            pass_at_n_macro = macro_mean_from_try_series(pass_at_n_try) * 100.0
            print(f"Pass@n Accuracy (macro over prompts): {pass_at_n_macro:.2f}%")
            metrics["pass_at_n_accuracy"] = float(pass_at_n_macro)

        # ---- selector helpers (same logic, but aggregate macro over prompts) ------
        def select_best_with_reasoning_preference(group):
            finished = group[group["finished"]]
            if finished.empty:
                return None
            finished = finished.copy()
            finished["used_reasoning"] = finished["output_token_ids"].apply(has_thinking)
            er = finished["expected_reward"].astype(float).fillna(float("-inf"))
            finished["__er__"] = er
            reasoning = finished[finished["used_reasoning"]]
            candidate_subset = reasoning if not reasoning.empty else finished
            return candidate_subset["__er__"].idxmax()

        def select_shortest_with_reasoning_preference(group):
            finished = group[group["finished"]]
            if finished.empty:
                return None
            finished = finished.copy()
            finished["used_reasoning"] = finished["output_token_ids"].apply(has_thinking)
            reasoning = finished[finished["used_reasoning"]]
            return (reasoning["length"].idxmin() if not reasoning.empty else finished["length"].idxmin())

        def _find_most_common_answer(answers: list[str]) -> str:
            if not answers: return ""
            from collections import Counter
            c = Counter(answers)
            m = max(c.values())
            top = {a for a, v in c.items() if v == m}
            for a in reversed(answers):
                if a in top: return a
            return answers[-1]

        def select_best_weighted_with_reasoning_preference(group):
            finished = group[group["finished"]]
            if finished.empty:
                return None
            finished = finished.copy()
            finished["used_reasoning"] = finished["output_token_ids"].apply(has_thinking)
            er = finished["expected_reward"].astype(float).fillna(float("-inf"))
            finished["__er__"] = er
            if "extracted_answer" not in finished.columns:
                return None
            scores = finished.groupby("extracted_answer")["__er__"].sum()
            if scores.empty:
                return None
            best_answers = scores[scores == scores.max()].index.tolist()
            if len(best_answers) > 1:
                r_scores = finished[finished["used_reasoning"]].groupby("extracted_answer")["__er__"].sum()
                r_scores = r_scores[r_scores.index.isin(best_answers)]
                if not r_scores.empty:
                    best_answers = [r_scores.idxmax()]
                else:
                    best_answers = [best_answers[0]]
            best_answer = best_answers[0]
            subset = finished[finished["extracted_answer"] == best_answer]
            if subset.empty:
                return None
            reasoning_subset = subset[subset["used_reasoning"]]
            candidate_subset = reasoning_subset if not reasoning_subset.empty else subset
            return candidate_subset["__er__"].idxmax()

        # Generic: per-try chosen-correct flags → macro mean over prompts
        def _selector_macro_mean_correct(selector_fn):
            try:
                idx_by_try = df_final.groupby(try_col, group_keys=False).apply(selector_fn, include_groups=False)
            except TypeError:
                idx_by_try = df_final.groupby(try_col, group_keys=False).apply(selector_fn)
            corr_by_try = idx_by_try.apply(lambda idx: bool(df_final.loc[idx, "correct"]) if pd.notna(idx) else False).astype(float)
            macro = macro_mean_from_try_series(corr_by_try) * 100.0
            return macro, idx_by_try

        # Best-of-n (expected_reward, reasoning-preferred)
        if "expected_reward" in df_final.columns and "correct" in df_final.columns:
            best_of_n_macro, chosen_idx_by_try = _selector_macro_mean_correct(select_best_with_reasoning_preference)
            print(f"Best-of-n Accuracy (expected_reward, reasoning-preferred; macro): {best_of_n_macro:.2f}%")
            metrics["best_of_n_accuracy_expected_reward"] = float(best_of_n_macro)

            # Precision among answered tries, macro over prompts
            answered_by_try = by_try.apply(lambda g: not g[(g["finished"]) & (~g["pruned"])].empty)
            answered_idx = answered_by_try[answered_by_try].index
            if len(answered_idx) > 0:
                corr_only_answered = chosen_idx_by_try.loc[answered_idx].apply(
                    lambda idx: bool(df_final.loc[idx, "correct"]) if pd.notna(idx) else False
                ).astype(float)
                per_prompt = corr_only_answered.groupby(try_to_prompt.loc[answered_idx]).mean()
                precision_macro = float(per_prompt.mean() * 100.0)
            else:
                precision_macro = 0.0
            print(f"Precision (best_of_n expected_reward; macro): {precision_macro:.2f}%")
            metrics["precision_best_of_n_expected_reward"] = float(precision_macro)

        # Consistency & weighted-consistency
        if args.use_consistency and "extracted_answer" in df_final.columns and "correct" in df_final.columns:
            def _select_consistency_idx(group):
                finished = group[group["finished"]]
                if finished.empty:
                    return None
                answers = finished["extracted_answer"].tolist()
                most_common = _find_most_common_answer(answers)
                for idx in reversed(finished.index.tolist()):
                    if finished.at[idx, "extracted_answer"] == most_common:
                        return idx
                return finished.index[-1]

            # Consistency
            cons_macro, cons_idx_by_try = _selector_macro_mean_correct(_select_consistency_idx)
            print(f"Best-of-n Accuracy (consistency / majority-vote; macro): {cons_macro:.2f}%")
            metrics["best_of_n_accuracy_consistency"] = float(cons_macro)

            # Weighted (expected_reward + consistency)
            if "expected_reward" in df_final.columns:
                weighted_macro, weighted_idx_by_try = _selector_macro_mean_correct(select_best_weighted_with_reasoning_preference)
                print(f"Best-of-n Accuracy (expected_reward + consistency; macro): {weighted_macro:.2f}%")
                metrics["best_of_n_accuracy_expected_reward_consistency"] = float(weighted_macro)

                # (Optional) list selected answers per try as before, but keep JSON compact:
                chosen_idx_weighted = weighted_idx_by_try.dropna()
                try_selected = {}
                for gid, idx in chosen_idx_weighted.items():
                    idx = int(idx)
                    if idx not in df_final.index:
                        continue
                    ans = df_final.at[idx, "extracted_answer"]
                    corr = bool(df_final.at[idx, "correct"]) if "correct" in df_final.columns else False
                    try_selected[str(gid)] = {"answer": ("" if ans is None else str(ans)), "correct": corr}
                metrics["selected_answers_expected_reward_consistency_by_try"] = try_selected

        # Shortest-response, reasoning-preferred baseline (macro)
        shortest_macro, shortest_idx_by_try = _selector_macro_mean_correct(select_shortest_with_reasoning_preference)
        print(f"Best-of-n Accuracy (shortest-response, reasoning-preferred; macro): {shortest_macro:.2f}%")
        metrics["best_of_n_accuracy_shortest_response"] = float(shortest_macro)

        # Breakdown for best-of-n (expected_reward): tokens vs correctness (macro) ----
        if "expected_reward" in df_final.columns and "correct" in df_final.columns:
            bo_idx_by_try = df_final.groupby(try_col, group_keys=False).apply(select_best_with_reasoning_preference)
            chosen_mask = bo_idx_by_try.notna()
            if chosen_mask.any():
                chosen_correct_flag = bo_idx_by_try[chosen_mask].apply(lambda idx: bool(df_final.loc[idx, "correct"]))
                # Align per-try totals/n-latency with the chosen correctness flags
                tt = try_total_tokens.reindex(chosen_correct_flag.index)
                nl = try_n_latency.reindex(chosen_correct_flag.index)

                # Per prompt means among correct tries and among incorrect tries
                correct_tt_per_prompt   = tt[chosen_correct_flag].groupby(try_to_prompt.loc[chosen_correct_flag.index]).mean()
                incorrect_tt_per_prompt = tt[~chosen_correct_flag].groupby(try_to_prompt.loc[chosen_correct_flag.index]).mean()
                correct_nl_per_prompt   = nl[chosen_correct_flag].groupby(try_to_prompt.loc[chosen_correct_flag.index]).mean()
                incorrect_nl_per_prompt = nl[~chosen_correct_flag].groupby(try_to_prompt.loc[chosen_correct_flag.index]).mean()

                correct_total_tokens_macro   = float(correct_tt_per_prompt.mean(skipna=True) if not correct_tt_per_prompt.empty else 0.0)
                incorrect_total_tokens_macro = float(incorrect_tt_per_prompt.mean(skipna=True) if not incorrect_tt_per_prompt.empty else 0.0)
                correct_n_latency_macro      = float(correct_nl_per_prompt.mean(skipna=True) if not correct_nl_per_prompt.empty else 0.0)
                incorrect_n_latency_macro    = float(incorrect_nl_per_prompt.mean(skipna=True) if not incorrect_nl_per_prompt.empty else 0.0)

                print(f"Average total tokens for correct best-of-n selections (macro):   {int(correct_total_tokens_macro)}")
                print(f"Average n-latency for correct best-of-n selections (macro):      {correct_n_latency_macro:.2f}")
                print(f"Average total tokens for incorrect best-of-n selections (macro): {int(incorrect_total_tokens_macro)}")
                print(f"Average n-latency for incorrect best-of-n selections (macro):    {incorrect_n_latency_macro:.2f}")

                metrics.update({
                    "correct_best_of_n_avg_total_tokens": int(correct_total_tokens_macro),
                    "correct_best_of_n_avg_n_latency": float(correct_n_latency_macro),
                    "incorrect_best_of_n_avg_total_tokens": int(incorrect_total_tokens_macro),
                    "incorrect_best_of_n_avg_n_latency": float(incorrect_n_latency_macro),
                })

                # Reasoning usage among selections (macro rate over prompts)
                used_reasoning_by_try = bo_idx_by_try[chosen_mask].apply(
                    lambda idx: has_thinking(df_final.loc[idx, "output_token_ids"])
                ).astype(float)
                reasoning_rate_macro = macro_mean_from_try_series(used_reasoning_by_try) * 100.0
                print(f"Best-of-n selections using reasoning (macro rate over prompts): {reasoning_rate_macro:.2f}%")
                metrics.update({
                    "best_of_n_selections_reasoning_percentage": float(reasoning_rate_macro),
                    # micro counts for reference:
                    "best_of_n_selections_total": int(chosen_mask.sum()),
                    "best_of_n_selections_using_reasoning": int(used_reasoning_by_try.sum()),
                })

        # ----- Write metrics JSON -----
        metrics_path = args.metrics_json_out
        if metrics_path is None:
            out_dir = os.path.dirname(args.out) or "results"
            base = os.path.splitext(os.path.basename(args.out))[0]
            cpt_tag = _fmt_float_tag(args.c_pt)
            cseq_tag = _fmt_float_tag(args.c_seq)
            spt_tag = _fmt_int_tag(args.samples_per_try)
            eval_mode_tag = str(args.eval_mode)
            if (args.eval_mode == "no_prune") and (args.no_prune_try_size is not None):
                try:
                    npt = int(args.no_prune_try_size)
                    if npt > 0:
                        eval_mode_tag = f"no_prune_{npt}"
                except Exception:
                    pass
            ev_threshold_part = f"_evt{_fmt_float_tag(args.ev_threshold)}" if float(args.ev_threshold) != 0.0 else ""
            alpha_part = f"_alpha{_fmt_float_tag(args.selection_geom_alpha)}" if abs(float(args.selection_geom_alpha) - 1.0) > 1e-12 else ""
            topp_part = f"_topp{_fmt_float_tag(args.joint_top_p)}" if float(args.joint_top_p) < 1.0 - 1e-12 else ""
            temp_part = f"_temp{_fmt_float_tag(args.joint_temp)}" if abs(float(args.joint_temp) - 1.0) > 1e-12 else ""
            metrics_path = os.path.join(out_dir, f"{base}_metrics_cpt{cpt_tag}_cseq{cseq_tag}_spt{spt_tag}_{eval_mode_tag}{ev_threshold_part}{alpha_part}{topp_part}{temp_part}.json")

        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved metrics to {metrics_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure faulthandler timer is cancelled to avoid shutdown races
        try:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass