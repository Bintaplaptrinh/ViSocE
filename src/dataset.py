"""Dataset and split utilities for multi-label emotion classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass
class SplitBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class TikTokEmotionDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, normalize_fn, max_len: int, label_map: dict[str, int]):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.normalize_fn = normalize_fn
        self.max_len = max_len
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data.iloc[idx]
        context = self.normalize_fn(item.get("context", ""))
        comment = self.normalize_fn(item.get("comment", ""))
        text = f"Context: {context} Comment: {comment}"

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = np.zeros(len(self.label_map), dtype=np.float32)
        raw_labels = item.get("labels", []) if isinstance(item.get("labels", []), list) else []

        for label in raw_labels:
            if label in self.label_map:
                labels[self.label_map[label]] = 1.0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def split_dataframe(df: pd.DataFrame, seed: int) -> SplitBundle:
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, shuffle=True)
    return SplitBundle(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def compute_pos_weight(train_df: pd.DataFrame, label_map: dict[str, int], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(label_map), dtype=np.float32)

    for labels in train_df["labels"]:
        if not isinstance(labels, list):
            continue
        for label in labels:
            if label in label_map:
                counts[label_map[label]] += 1.0

    total = len(train_df)
    weights = [((total - c) / (c + 1e-6)) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)
