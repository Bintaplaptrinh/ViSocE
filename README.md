# ViSocE Emotion Classification
## 1. Project Overview
This project implements a context-aware multi-label emotion classification pipeline for Vietnamese Gen Z social media comments.

The project follows the ViSocE direction described in the manuscript in [../main (1).tex](../main%20(1).tex), with these core ideas:
- Emotion interpretation must use both Context and Comment.
- Multi-label output is preferred for mixed emotions such as sarcasm and schadenfreude.
- Compact label space avoids severe sparsity while preserving expressive power.

## 2. Research Objective
The objective is to evaluate how architecture choice and threshold tuning affect multi-label emotion quality under context-aware input:
- Baseline: PhoBERT + MLP
- Intermediate: PhoBERT + MHA + MLP
- Advanced: PhoBERT + Dual-stream
- Proposed advanced variant: PhoBERT + Multi-Vector Attention + KAN 1D

## 3. Label Space
Depending on configuration, the pipeline supports:
- 7 labels: Enjoyment, Sadness, Anger, Surprise, Fear, Disgust, Other
- 6 labels: Enjoyment, Sadness, Anger, Surprise, Fear, Disgust (Other excluded)

Default configuration keeps Other enabled.

## 4. Project Structure
final/
- config/
  - project.yaml: central experiment configuration
- scripts/
  - 01_prepare_data.py: data loading and split
  - 02_train_models.py: optional model training with early stopping
  - 03_test_existing_models_split.py: evaluate existing checkpoints on local split data
  - 04_export_ieee_tables.py: LaTeX table export for IEEE manuscript
- src/
  - constants.py: labels and mappings
  - config_loader.py: YAML configuration utilities
  - text_preprocess.py: slang normalization and text cleanup
  - dataset.py: dataset class, split, pos_weight
  - modeling.py: all model architectures
  - train.py: training loops and early stopping
  - evaluate.py: constrained threshold tuning and metrics
  - utils.py: IO and reproducibility helpers
- outputs/
  - generated artifacts: split files, checkpoints, metrics, rankings
- data/
  - ViSocE.json: local copied dataset file
  - dictionary.json: local copied slang dictionary
- reports/
  - generated table files
- requirements.txt
- run_pipeline.py
- README.md

## 5. Methodology Mapping to IEEE Sections
### 5.1 Data and Preprocessing
- Input fields: context, comment, labels
- Slang replacement from dictionary_merged.json
- Text normalization preserves Vietnamese diacritics
- Training text format: Context: ... Comment: ...

### 5.2 Data Split
- 70 percent train
- 15 percent validation
- 15 percent test
- Random seed controlled by config

### 5.3 Training Strategy
- Loss: BCEWithLogitsLoss with pos_weight from training distribution
- Optimizer: AdamW
- LR scheduler: linear warmup and linear decay
- Mixed precision enabled on CUDA
- Early stopping with patience and minimum delta

### 5.4 Evaluation Strategy
- Validation threshold tuning per label under minimum precision constraints
- Test metrics with tuned thresholds
- Exported outputs:
  - per-model JSON metrics
  - ranking CSV
  - aggregated result JSON
  - LaTeX table

## 6. Configuration
Edit [config/project.yaml](config/project.yaml) for:
- paths
- whether to include Other
- model list
- batch sizes, epochs, lr, warmup, dropout
- threshold tuning precision constraints

## 7. Reproducibility Steps
Run from the [final](.) directory.

### Step 1
python scripts/01_prepare_data.py

### Step 2
python scripts/03_test_existing_models_split.py

### Step 3
python scripts/04_export_ieee_tables.py

Or run full pipeline:
python run_pipeline.py

## 8. Input Data Requirements
Expected JSON schema:

[
  {
    "context": "video or post context",
    "comment": "user comment",
    "labels": ["Enjoyment", "Surprise"]
  }
]

Notes:
- labels must be a list
- with 7-label mode, Other must be included exactly as text Other
- default config reads from local files in data/

## 9. Artifact Outputs
After execution, [outputs](outputs) contains:
- train.json, val.json, test.json
- split_metadata.json
- best_<MODEL_NAME>.pt
- metrics_<MODEL_NAME>.json
- model_ranking.csv
- all_model_results.json

And [reports](reports) contains:
- results_table.tex

## 10. How to Use in Paper Writing
Use [reports/results_table.tex](reports/ieee_results_table.tex) directly in LaTeX manuscript.
Recommended manuscript alignment:
- Introduction and theory: from the ViSocE framework
- Experimental settings: from config/project.yaml
- Result table: from reports/results_table.tex

## 11. Limitations and Notes
- KAN model requires [../kan_Arch_1D.py](../kan_Arch_1D.py) to be available.
- GPU training is strongly recommended.
- If include_other is true, class imbalance can still reduce macro F1 for rare labels.

## 12. Citation Style Recommendation
Use bibliography style and report:
- Macro and micro F1
- Threshold policy
- Split ratio
- Whether Other is included

## 13. Quick Start
1. Install dependencies:
   pip install -r requirements.txt
2. Check paths in config/project.yaml
3. Run python scripts/01_prepare_data.py
4. Run python scripts/03_test_existing_models_split.py
5. Run python scripts/04_export_ieee_tables.py
6. Attach reports/ieee_results_table.tex to the manuscript
