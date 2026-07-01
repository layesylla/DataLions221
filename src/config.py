from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA = DATA_DIR / "raw"

SUBMISSION_DIR = DATA_DIR / "submissions"
STACK_DIR = DATA_DIR / "stack"

STACK_DIR.mkdir(exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

