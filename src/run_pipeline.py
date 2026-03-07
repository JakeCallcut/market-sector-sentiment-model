import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent
SCRIPTS_DIR = SRC_DIR / "scripts"
MODELS_DIR = SRC_DIR / "models"


def run(script: Path):
    print(f"\n>>> Running {script.name}")
    subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)


if __name__ == "__main__":
    # Data collection & processing
    run(SCRIPTS_DIR / "yfinance_utils.py")
    run(SCRIPTS_DIR / "cleaning.py")
    run(SCRIPTS_DIR / "preprocessing.py")

    # Model training
    run(MODELS_DIR / "mn_log_reg.py")
    run(MODELS_DIR / "random_forest.py")

    print("\n✓ Pipeline complete")
