#!/usr/bin/env python3
"""
Visualization utilities for ZIP (Zero-overhead Inference-time Prediction).

This module contains functions for creating ASCII visualizations of joint distributions
during both training and inference, helping to understand model predictions and progress.
"""

import torch
import torch.nn.functional as F
import numpy as np


def create_ascii_bar_chart(values, labels, title, max_width=50, display_as_percent=False):
    """Create an ASCII bar chart for visualization.

    If display_as_percent is True, prints values as percentages with one decimal place.
    """
    if values is None or len(values) == 0:
        return f"\n{title}\n(No data)\n"
    
    # Use probability scale (0-1) instead of normalizing to max
    lines = [f"\n{title}"]
    lines.append("─" * (max_width + 20))
    
    for i, (val, label) in enumerate(zip(values, labels)):
        # Clamp values to [0, 1] and scale to bar width
        clamped_val = max(0, min(1, val))
        bar_length = int(clamped_val * max_width)
        bar = "█" * bar_length + "░" * (max_width - bar_length)
        if display_as_percent:
            lines.append(f"{label:<12} │{bar}│ {clamped_val*100:.1f}%")
        else:
            lines.append(f"{label:<12} │{bar}│ {val:.4f}")
    
    lines.append("─" * (max_width + 20))
    return "\n".join(lines) + "\n"


def log_inference_distribution(joint_probs, length_bins, num_length_bins, num_reward_states=2, reward_values=None, sample_idx=-1):
    """Grid visualization of the joint distribution during inference.
    Replaces the legacy bar‑chart view."""
    if joint_probs is None:
        return
    # Convert to 2D [V, L] if needed
    if hasattr(joint_probs, "detach"):
        arr = joint_probs.detach().cpu().float().numpy()
    else:
        arr = np.array(joint_probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(int(num_reward_states), int(num_length_bins))
    log_joint_distribution_grid(
        arr,
        length_bins,
        num_length_bins,
        num_reward_states=num_reward_states,
        reward_values=reward_values,
        title_prefix=f"Predicted (sample {sample_idx})",
    )


def _make_length_labels(length_bins, num_length_bins):
    labels = []
    for i in range(num_length_bins):
        start, end = length_bins[i], length_bins[i + 1] - 1
        if end >= 32767:
            labels.append(f"{start}+")
        else:
            labels.append(f"{start}-{end}")
    return labels


def log_before_after_pruning(before_dist, after_dist, length_bins, num_length_bins, num_reward_states=2, reward_values=None, sample_idx=-1, prune_at_bin=None):
    """Grid visualize a joint distribution BEFORE and AFTER applying pruning."""
    if before_dist is None or after_dist is None:
        return
    # Convert to numpy arrays if needed
    if hasattr(before_dist, "detach"):
        before = before_dist.detach().cpu().float().numpy()
    else:
        before = np.array(before_dist, dtype=float)
    if hasattr(after_dist, "detach"):
        after = after_dist.detach().cpu().float().numpy()
    else:
        after = np.array(after_dist, dtype=float)

    suffix = f"  (prune_at_bin={prune_at_bin})" if prune_at_bin is not None else ""
    log_joint_distribution_grid(
        before, length_bins, num_length_bins, num_reward_states, reward_values,
        title_prefix=f"BEFORE (sample {sample_idx}){suffix}"
    )
    log_joint_distribution_grid(
        after, length_bins, num_length_bins, num_reward_states, reward_values,
        title_prefix=f"AFTER  (sample {sample_idx}){suffix}"
    )


def log_max_distribution(prob_in_bin, bin_values, *, length_bins=None, title="Distribution of maximum"):
    """ASCII visualize the distribution of the maximum across samples.

    Args:
        prob_in_bin: 1D iterable of probabilities for each bin where max falls
        bin_values: 1D iterable of sorted bin representative values (same order as prob_in_bin)
        length_bins: when visualizing length maxima, pass boundaries to label intervals
        title: section title
    """
    if prob_in_bin is None:
        return

    # Convert torch tensors if needed
    if hasattr(prob_in_bin, 'detach'):
        probs = prob_in_bin.detach().cpu().float().numpy()
    else:
        probs = prob_in_bin

    values = list(bin_values)

    # Build labels
    if length_bins is not None:
        # Length case: labels are intervals; assume values correspond to midpoints derived from length_bins
        # We will derive labels from boundaries in ascending order
        num_length_bins = len(length_bins) - 1
        labels = _make_length_labels(length_bins, num_length_bins)
        # In general values may have been sorted; ensure we permute labels to match sorted order
        # We reconstruct midpoints to map value->index
        midpoints = [ (length_bins[i] + length_bins[i+1]) / 2 for i in range(num_length_bins) ]
        # Build mapping from value to original index (robust to float issues via nearest)
        def nearest_index(val):
            diffs = [abs(val - m) for m in midpoints]
            return int(np.argmin(diffs))
        ordered_labels = [ labels[nearest_index(v)] for v in values ]
        chart_title = title
        print(create_ascii_bar_chart(probs, ordered_labels, chart_title, display_as_percent=True))
    else:
        # Value case
        value_labels = [f"{float(v):.1f}" for v in values]
        chart_title = title
        print(create_ascii_bar_chart(probs, value_labels, chart_title, display_as_percent=True))

def log_max_input_distributions(marginal_distributions, bin_values, ids=None, *, length_bins=None, title_prefix="Inputs for max"):
    """ASCII visualize the marginal distributions that feed into the max.

    Args:
        marginal_distributions: list of 1D arrays/tensors (probabilities across bins), should be ordered to match bin_values order
        bin_values: list of values corresponding to each bin (already sorted to match the marginals)
        ids: optional list of identifiers (e.g., sample indices) corresponding to each marginal
        length_bins: if provided, treat as length visualization and derive interval labels; otherwise treat as reward
        title_prefix: header prefix for each chart
    """
    if not marginal_distributions:
        return

    # Build labels for bins
    if length_bins is not None:
        num_length_bins = len(length_bins) - 1
        bin_labels = _make_length_labels(length_bins, num_length_bins)
    else:
        bin_labels = [f"{float(v):.1f}" for v in bin_values]

    for idx, dist in enumerate(marginal_distributions):
        if hasattr(dist, 'detach'):
            values = dist.detach().cpu().float().numpy()
        else:
            values = dist
        name = f"{title_prefix} - s={ids[idx]}" if ids is not None else f"{title_prefix} - #{idx}"
        print(create_ascii_bar_chart(values, bin_labels, name, display_as_percent=True))

def visualize_predictions(model, batch, distribution_token_id, num_bins, length_bins, device):
    """Extract joint distribution predictions vs ground truth at first and last labeled positions.

    Returns a dict with optional entries for 'first' and 'last':
    {
        'first': (pred_probs, gt_probs),
        'last': (pred_probs, gt_probs)
    }
    Any entry may be missing if indices are invalid.
    """
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        
        # Use the original model if it's wrapped in DDP
        original_model = model.module if hasattr(model, 'module') else model
        
        # Compute only the final hidden states to avoid materializing [B,S,V]
        outputs = original_model._orig_mod(input_ids=input_ids, output_hidden_states=True) if hasattr(original_model, '_orig_mod') else original_model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [B, S, E]
        # Resolve lm_head for projecting to needed vocab slice
        lm_module = original_model._orig_mod if hasattr(original_model, '_orig_mod') else original_model
        lm_head = lm_module.lm_head if hasattr(lm_module, 'lm_head') else lm_module.get_output_embeddings()
        
        # Collect predictions and ground truth for first sample in batch
        sample_idx = 0
        if sample_idx >= len(batch["label_positions"]) or not batch["label_positions"][sample_idx]:
            return {}
        
        pos_list = batch["label_positions"][sample_idx]
        label_list = batch["bin_labels"][sample_idx]
        
        if not pos_list or not label_list:
            return {}
        results = {}

        # Helper to extract probs at a given position index into pos_list/label_list
        def extract_at(idx_in_list):
            pos = pos_list[idx_in_list]
            if pos >= hidden_states.size(1):
                return None
            # Project only distribution slice logits for this position
            h = hidden_states[sample_idx, pos, :]
            weight_bins = lm_head.weight[distribution_token_id:distribution_token_id + num_bins]
            bias_bins = lm_head.bias[distribution_token_id:distribution_token_id + num_bins] if hasattr(lm_head, 'bias') and lm_head.bias is not None else None
            dist_logits = F.linear(h, weight_bins, bias_bins)
            pred = F.softmax(dist_logits, dim=0).cpu().float().numpy()
            gt_label = label_list[idx_in_list]
            gt = np.zeros(num_bins)
            if 0 <= gt_label < num_bins:
                gt[gt_label] = 1.0
            return (pred, gt)

        first_pair = extract_at(0)
        if first_pair is not None:
            results['first'] = first_pair
        last_pair = extract_at(len(pos_list) - 1)
        if last_pair is not None:
            results['last'] = last_pair

        return results


def log_prediction_distributions(pred_probs, gt_probs, length_bins, num_length_bins, num_reward_states=2, reward_values=None):
    """Grid visualizations for joint distribution predictions vs ground truth."""
    if pred_probs is None or gt_probs is None:
        return
    def _to_2d(x):
        if hasattr(x, "detach"):
            a = x.detach().cpu().float().numpy()
        else:
            a = np.array(x, dtype=float)
        if a.ndim == 1:
            a = a.reshape(int(num_reward_states), int(num_length_bins))
        return a
    pred2d = _to_2d(pred_probs)
    gt2d = _to_2d(gt_probs)
    log_joint_distribution_grid(
        pred2d, length_bins, num_length_bins, num_reward_states, reward_values,
        title_prefix="Model Predictions"
    )
    log_joint_distribution_grid(
        gt2d, length_bins, num_length_bins, num_reward_states, reward_values,
        title_prefix="Ground Truth"
    )


def _make_value_labels(reward_values, num_reward_states):
    if reward_values is None:
        return [str(i) for i in range(num_reward_states)]
    return [str(v) for v in reward_values]


def create_ascii_heatmap(
    matrix2d,
    row_labels,
    col_labels,
    title,
    shades=None,
    normalize="global",  # unused; kept for compatibility
    gamma=1.0,            # unused; kept for compatibility
    pmin=0.0,             # unused; kept for compatibility
    pmax=100.0,           # unused; kept for compatibility
    cell_width=1,
):
    """Render a 2D probability grid as an ASCII/Unicode heatmap with robust scaling.

    - Robust contrast via percentile clipping (pmin/pmax) and gamma correction.
    - Normalization can be global (entire grid) or per-row to highlight within-row structure.
    - Uses compact Unicode blocks by default for clearer, denser grids.
    """
    import numpy as np
    if hasattr(matrix2d, 'detach'):
        grid = matrix2d.detach().cpu().float().numpy()
    else:
        grid = np.array(matrix2d, dtype=float)

    # Default shades: Unicode blocks (light→dark). ASCII fallback provided.
    if shades is None:
        shades = " ▁▂▃▄▅▆▇█"  # includes leading space for zero
    n_levels = max(1, len(shades) - 1)

    print(f"\n{'='*80}")
    print(title)
    print("(raw probabilities)")
    print(f"{'='*80}")

    # Use raw probabilities (clamped to [0,1])
    scaled = np.clip(grid, 0.0, 1.0)

    # Render as fixed-width (4 chars) zero-padded integer percentages with no spacing
    def fmt_percent(p):
        n = int(round(float(p) * 100))
        if n < 0:
            n = 0
        elif n > 100:
            n = 100
        return f"{n:03d}%"  # e.g., 000%, 005%, 042%, 100%

    for r, row in enumerate(scaled):
        cells = [fmt_percent(val) for val in row]
        print("".join(cells))
    print(f"{'='*80}\n", flush=True)


def log_joint_distribution_grid(prob_2d, length_bins, num_length_bins, num_reward_states=2, reward_values=None, title_prefix="Predicted"):
    """Log a 2D ASCII heatmap with value on X (low→high) and remaining tokens on Y (high→low)."""
    # Prepare labels
    length_labels_asc = _make_length_labels(length_bins, num_length_bins)
    value_labels_unsorted = _make_value_labels(reward_values, num_reward_states)

    # Determine ordering: values ascending (left→right), lengths descending (top→bottom)
    if reward_values is None:
        value_values = list(range(num_reward_states))
    else:
        value_values = list(reward_values)
    value_perm = list(np.argsort(np.array(value_values, dtype=float)))
    length_perm_desc = list(range(num_length_bins - 1, -1, -1))

    # Convert to numpy and orient to [length x value] first
    if hasattr(prob_2d, 'detach'):
        grid = prob_2d.detach().cpu().float().numpy()
    else:
        grid = np.array(prob_2d, dtype=float)
    # grid is [value x length]; transpose to [length x value]
    grid_lv = grid.T

    # Apply permutations (rows: length high→low, cols: value low→high)
    grid_lv = grid_lv[np.array(length_perm_desc), :]
    grid_lv = grid_lv[:, np.array(value_perm)]

    # Permute labels to match
    row_labels = [length_labels_asc[i] for i in length_perm_desc]
    col_labels = [value_labels_unsorted[i] for i in value_perm]

    create_ascii_heatmap(grid_lv, row_labels, col_labels, f"{title_prefix} joint distribution (length high→low x value low→high)")


def log_gaussian_params_list(items, title="Gaussian head parameters"):
    """Print a compact summary of Gaussian parameters.

    Args:
        items: list of dicts, each with keys: label, mu_t, mu_y, sigma_t, sigma_y, rho
        title: header title
    """
    if not items:
        return
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    for it in items:
        try:
            # Clamp printing ranges for readability
            mu_t = max(-12.0, min(12.0, float(it['mu_t'])))
            mu_y = max(-8.0, min(8.0, float(it['mu_y'])))
            sigma_t = max(0.0, min(5.0, float(it['sigma_t'])))
            sigma_y = max(0.0, min(5.0, float(it['sigma_y'])))
            rho = max(-0.9, min(0.9, float(it['rho'])))
            line = (f"[{it.get('label', '')}] "
                    f"mu_t={mu_t: .4f}  mu_y={mu_y: .4f}  "
                    f"sigma_t={sigma_t: .4f}  sigma_y={sigma_y: .4f}  "
                    f"rho={rho: .4f}")
            print(line)
        except Exception:
            print(str(it))
    print(f"{'='*80}\n", flush=True)