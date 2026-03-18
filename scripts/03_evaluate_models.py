"""Step 3: Evaluate trained models and tune thresholds on validation split."""

from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

CURRENT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT.parents[1]
PARENT_PIPELINE = CURRENT.parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PARENT_PIPELINE))

from src.config_loader import load_config, resolve_path
from src.constants import LABEL_MAP_6, LABEL_MAP_7, LABELS_6, LABELS_7
from src.dataset import TikTokEmotionDataset
from src.evaluate import evaluate_with_thresholds, tune_thresholds_constrained
from src.modeling import build_model
from src.text_preprocess import load_slang_dict, normalize_text
from src.train import collect_probs_labels
from src.utils import write_json


def _load_split(path: Path) -> pd.DataFrame:
    return pd.read_json(path)


def main():
    cfg = load_config(PROJECT_ROOT / "config" / "project.yaml")

    outputs_dir = resolve_path(PROJECT_ROOT, cfg["paths"]["outputs_dir"])
    include_other = bool(cfg["dataset"]["include_other"])
    label_map = LABEL_MAP_7 if include_other else LABEL_MAP_6
    label_names = LABELS_7 if include_other else LABELS_6

    val_df = _load_split(outputs_dir / "val.json")
    test_df = _load_split(outputs_dir / "test.json")

    slang_path = resolve_path(PROJECT_ROOT, cfg["paths"]["slang_dict"])
    slang_dict = load_slang_dict(slang_path)
    normalize_fn = partial(normalize_text, slang_dict=slang_dict)

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    max_length = int(cfg["dataset"]["max_length"])

    val_ds = TikTokEmotionDataset(val_df, tokenizer, normalize_fn, max_length, label_map)
    test_ds = TikTokEmotionDataset(test_df, tokenizer, normalize_fn, max_length, label_map)

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size_eval"]),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["training"]["batch_size_eval"]),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = torch.cuda.is_available()

    min_precision_default = float(cfg["threshold_tuning"]["min_precision_default"])
    per_label_min_precision = cfg["threshold_tuning"]["per_label_min_precision"]

    results = []

    for model_name in cfg["experiments"]["models"]:
        ckpt_path = outputs_dir / f"best_{model_name}.pt"
        if not ckpt_path.exists():
            print(f"Skip {model_name}: checkpoint not found at {ckpt_path}")
            continue

        model = build_model(
            name=model_name,
            num_labels=len(label_map),
            dropout=float(cfg["training"]["dropout"]),
            num_heads=int(cfg["training"]["num_heads"]),
            grid_size=int(cfg["training"]["grid_size"]),
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        val_prob, val_true = collect_probs_labels(model, val_loader, device, len(label_map), amp_enabled)
        thresholds, tuning_df = tune_thresholds_constrained(
            y_true=val_true,
            y_prob=val_prob,
            label_names=label_names,
            min_precision=min_precision_default,
            per_label_min_precision=per_label_min_precision,
        )

        test_prob, test_true = collect_probs_labels(model, test_loader, device, len(label_map), amp_enabled)
        test_metrics, report_dict, conf_mtx = evaluate_with_thresholds(test_true, test_prob, thresholds, label_names)

        model_result = {
            "model": model_name,
            "checkpoint": str(ckpt_path),
            "thresholds": thresholds.tolist(),
            "metrics": test_metrics,
            "classification_report": report_dict,
            "multilabel_confusion_matrix": conf_mtx,
            "threshold_tuning": tuning_df.to_dict(orient="records"),
        }
        results.append(model_result)

        write_json(model_result, outputs_dir / f"metrics_{model_name}.json")
        print(f"Saved test metrics for {model_name}")

    summary_rows = []
    for item in results:
        summary_rows.append(
            {
                "model": item["model"],
                "f1_micro": item["metrics"]["F1-micro"],
                "f1_macro": item["metrics"]["F1-macro"],
                "f1_weighted": item["metrics"]["F1-weighted"],
                "accuracy": item["metrics"]["Accuracy"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="f1_micro", ascending=False)
    summary_df.to_csv(outputs_dir / "model_ranking.csv", index=False)
    write_json(results, outputs_dir / "all_model_results.json")
    print("Saved ranking and all model metrics.")


if __name__ == "__main__":
    main()
