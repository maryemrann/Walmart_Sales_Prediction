from pathlib import Path

BASE_DIR = Path("E:/maryam/Elevvo/Sales_Forecasting")
RAW_DIR = BASE_DIR
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"

# Create folders if they don't exist
PROCESSED_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
