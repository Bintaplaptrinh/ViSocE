from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(step_script: str) -> None:
    print("=" * 80)
    print(f"Running: {step_script}")
    command = [sys.executable, step_script]
    subprocess.run(command, check=True)


def main():
    root = Path(__file__).resolve().parent
    scripts = [
        root / "scripts" / "01_prepare_data.py",
        root / "scripts" / "03_test_existing_models_split.py",
        root / "scripts" / "04_export_tables.py",
    ]

    for script in scripts:
        run_step(str(script))

    print("All steps completed")


if __name__ == "__main__":
    main()
