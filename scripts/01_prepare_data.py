"""Step 1: Load and split data for reproducible experiments."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

CURRENT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT.parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_path
from src.constants import LABELS_6, LABELS_7
from src.dataset import split_dataframe
from src.utils import write_json


def main():
    cfg = load_config(PROJECT_ROOT / "config" / "project.yaml")
    data_path = resolve_path(PROJECT_ROOT, cfg["paths"]["data_json"])
    include_other = bool(cfg["dataset"]["include_other"])

    df = pd.read_json(data_path)
    if not include_other:
        df = df[~df["labels"].apply(lambda x: isinstance(x, list) and ("Other" in x))].reset_index(drop=True)

    splits = split_dataframe(df, seed=int(cfg["project"]["seed"]))
    outputs_dir = resolve_path(PROJECT_ROOT, cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    splits.train.to_json(outputs_dir / "train.json", force_ascii=False, orient="records", indent=2)
    splits.val.to_json(outputs_dir / "val.json", force_ascii=False, orient="records", indent=2)
    splits.test.to_json(outputs_dir / "test.json", force_ascii=False, orient="records", indent=2)

    label_names = LABELS_7 if include_other else LABELS_6
    metadata = {
        "include_other": include_other,
        "labels": label_names,
        "num_samples": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
    }
    write_json(metadata, outputs_dir / "split_metadata.json")

    print("Saved split files:")
    print(outputs_dir / "train.json")
    print(outputs_dir / "val.json")
    print(outputs_dir / "test.json")


if __name__ == "__main__":
    main()
