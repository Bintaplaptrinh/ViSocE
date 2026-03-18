"""Step 4: Export IEEE-ready tables from evaluation outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

CURRENT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT.parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_path


def _build_ieee_table(df: pd.DataFrame) -> str:
    required_columns = ["model", "f1_micro", "f1_macro", "f1_weighted", "accuracy"]
    if "model" not in df.columns:
        if "architecture" in df.columns:
            df = df.rename(columns={"architecture": "model"})
        elif "model_file" in df.columns:
            df = df.rename(columns={"model_file": "model"})

    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    def fmt(value):
        if pd.isna(value):
            return "-"
        return f"{float(value):.4f}"

    lines = []
    lb = r"\\"
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-model evaluation on ViSocE split}")
    lines.append("\\label{tab:final_results}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & F1-micro & F1-macro & F1-weighted & Accuracy " + lb)
    lines.append("\\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{row['model']} & {fmt(row['f1_micro'])} & {fmt(row['f1_macro'])} & {fmt(row['f1_weighted'])} & {fmt(row['accuracy'])} " + lb)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    cfg = load_config(PROJECT_ROOT / "config" / "project.yaml")
    outputs_dir = resolve_path(PROJECT_ROOT, cfg["paths"]["outputs_dir"])
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = outputs_dir / "model_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError("model_ranking.csv not found. Run Step 3 first.")

    ranking_df = pd.read_csv(ranking_path)
    latex_table = _build_ieee_table(ranking_df)

    out_file = reports_dir / "results_table.tex"
    out_file.write_text(latex_table, encoding="utf-8")

    print("IEEE table exported:")
    print(out_file)


if __name__ == "__main__":
    main()
