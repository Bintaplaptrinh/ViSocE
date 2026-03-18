from __future__ import annotations

import subprocess
from pathlib import Path
import sys

CURRENT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT.parents[1]
PYTHON = sys.executable


def main() -> None:
    step_1 = PROJECT_ROOT / "scripts" / "01_prepare_data.py"
    step_3 = PROJECT_ROOT / "scripts" / "03_test_existing_models_split.py"
    step_4 = PROJECT_ROOT / "scripts" / "04_export_tables.py"

    for step in (step_1, step_3, step_4):
        print(f"Running {step.name}")
        subprocess.run([PYTHON, str(step)], check=True)

    print("No-rerun evaluation completed using existing checkpoints on split data.")


if __name__ == "__main__":
    main()
