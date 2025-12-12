"""Shared dataset utilities for ZIP-RC training scripts."""

from __future__ import annotations

import ast
from typing import List

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ZIPDataset(Dataset):
    """Dataset that supplies joint reward/length bin labels for ZIP-RC training."""

    def __init__(
        self,
        table,
        max_length: int = 32_768,
        thinking_only: bool = False,
        thinking_token_id: int = 151667,
        reward_values: List[float] | None = None,
        label_column: str = "correct",
    ):
        self.table = pq.read_table(table).to_pandas() if isinstance(table, str) else table
        cols = set(self.table.columns)

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

        if self.reward_column == "correct":
            uniq = set(self.table["correct"].dropna().unique().tolist())
            uniq_numeric = {float(x) for x in uniq}
            if not uniq_numeric.issubset({0.0, 1.0}):
                raise ValueError(
                    f"Column 'correct' must be binary in {0,1}/{False,True}; found values {sorted(uniq, key=str)}."
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

            thinking_mask = self.table["input_ids"].apply(lambda ids: thinking_token_id in _as_list_safe(ids))
            self.table = self.table[thinking_mask].reset_index(drop=True)
            print(
                f"Filtered to thinking samples only: {len(self.table)}/{orig_len} "
                f"({len(self.table) / orig_len:.1%}) samples retained."
            )

        self.max_length = max_length

        self.length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        self.num_length_bins = len(self.length_bins) - 1  # 8 bins

        if reward_values is not None:
            self.reward_values = reward_values
        else:
            if self.reward_column == "correct":
                self.reward_values = [0.0, 1.0]
            else:
                self.reward_values = [0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0]
        self.num_reward_states = len(self.reward_values)
        self.num_bins = self.num_length_bins * self.num_reward_states

        if self.num_reward_states >= 2:
            edges = [0.0] * (self.num_reward_states + 1)
            for i in range(1, self.num_reward_states):
                edges[i] = 0.5 * (self.reward_values[i - 1] + self.reward_values[i])
            first_step = self.reward_values[1] - self.reward_values[0]
            last_step = self.reward_values[-1] - self.reward_values[-2]
            edges[0] = self.reward_values[0] - 0.5 * first_step
            edges[-1] = self.reward_values[-1] + 0.5 * last_step
        else:
            edges = [self.reward_values[0] - 0.5, self.reward_values[0] + 0.5]
        edges[0] = max(0.0, edges[0])
        edges[-1] = min(1.0, edges[-1])
        self.value_bin_edges = edges

    def __len__(self):
        return len(self.table)

    def _get_bin_idx(self, tokens_to_completion, reward):
        length_bin = 0
        for i in range(len(self.length_bins) - 1):
            if tokens_to_completion >= self.length_bins[i] and tokens_to_completion < self.length_bins[i + 1]:
                length_bin = i
                break

        if tokens_to_completion >= self.length_bins[-1]:
            length_bin = self.num_length_bins - 1

        reward_state = 0
        for i in range(len(self.value_bin_edges) - 1):
            if reward >= self.value_bin_edges[i] and reward < self.value_bin_edges[i + 1]:
                reward_state = i
                break
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

        bin_labels = []
        total_length = len(ids)
        for pos in label_positions:
            tokens_to_completion = total_length - pos - 1
            reward_value = float(row[self.reward_column])
            if reward_value < 0.0:
                reward_value = 0.0
            elif reward_value > 1.0:
                reward_value = 1.0
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
