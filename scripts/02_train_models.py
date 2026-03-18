"""Step 2: Train selected model architectures with early stopping."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

CURRENT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT.parents[1]
PARENT_PIPELINE = CURRENT.parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PARENT_PIPELINE))

from src.config_loader import load_config, resolve_path
from src.constants import LABEL_MAP_6, LABEL_MAP_7
from src.dataset import TikTokEmotionDataset, compute_pos_weight
from src.modeling import build_model
from src.text_preprocess import load_slang_dict, normalize_text
from src.train import early_stopping_train_loop
from src.utils import set_seed


def _load_split(path: Path) -> pd.DataFrame:
    return pd.read_json(path)


def main():
    cfg = load_config(PROJECT_ROOT / "config" / "project.yaml")
    set_seed(int(cfg["project"]["seed"]))

    outputs_dir = resolve_path(PROJECT_ROOT, cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split(outputs_dir / "train.json")
    val_df = _load_split(outputs_dir / "val.json")

    include_other = bool(cfg["dataset"]["include_other"])
    label_map = LABEL_MAP_7 if include_other else LABEL_MAP_6
    num_labels = len(label_map)

    slang_path = resolve_path(PROJECT_ROOT, cfg["paths"]["slang_dict"])
    slang_dict = load_slang_dict(slang_path)
    normalize_fn = partial(normalize_text, slang_dict=slang_dict)

    max_length = int(cfg["dataset"]["max_length"])
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    train_ds = TikTokEmotionDataset(train_df, tokenizer, normalize_fn, max_length, label_map)
    val_ds = TikTokEmotionDataset(val_df, tokenizer, normalize_fn, max_length, label_map)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size_train"]),
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size_eval"]),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()
    pos_weight = compute_pos_weight(train_df, label_map, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = int(cfg["training"]["epochs"])
    lr = float(cfg["training"]["learning_rate"])
    warmup_ratio = float(cfg["training"]["warmup_ratio"])
    weight_decay = float(cfg["training"]["weight_decay"])
    patience = int(cfg["training"]["patience"])
    min_delta = float(cfg["training"]["min_delta"])
    dropout = float(cfg["training"]["dropout"])
    num_heads = int(cfg["training"]["num_heads"])
    grid_size = int(cfg["training"]["grid_size"])

    model_names = cfg["experiments"]["models"]

    for model_name in model_names:
        print("=" * 70)
        print(f"Training model: {model_name}")

        model = build_model(model_name, num_labels, dropout, num_heads, grid_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = GradScaler("cuda", enabled=amp_enabled)

        thresholds = torch.full((num_labels,), 0.5, dtype=torch.float32).numpy()
        ckpt_name = f"best_{model_name}.pt"
        ckpt_path = outputs_dir / ckpt_name

        best_val_f1 = early_stopping_train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epochs=epochs,
            thresholds=thresholds,
            checkpoint_path=ckpt_path,
            amp_enabled=amp_enabled,
            min_delta=min_delta,
            patience=patience,
        )

        print(f"Best val F1-macro ({model_name}): {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
