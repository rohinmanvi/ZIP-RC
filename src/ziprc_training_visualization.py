#!/usr/bin/env python3
"""
Visualization utilities for ZIP (Zero-overhead Introspective Prediction).

This module contains functions for creating ASCII visualizations of joint distributions
during both training and inference, helping to understand model predictions and progress.
"""

import torch
import torch.nn.functional as F
import numpy as np


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


def _make_length_labels(length_bins, num_length_bins):
    labels = []
    for i in range(num_length_bins):
        start, end = length_bins[i], length_bins[i + 1] - 1
        if end >= 32767:
            labels.append(f"{start}+")
        else:
            labels.append(f"{start}-{end}")
    return labels


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
