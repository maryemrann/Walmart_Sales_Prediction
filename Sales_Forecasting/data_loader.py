# data_loader.py
import pandas as pd
from pathlib import Path
from config import RAW_DIR

def load_raw():
    """
    Loads walmart.csv from the RAW_DIR folder.
    Expects walmart.csv to be in the same folder as this script.
    """
    file_path = RAW_DIR / "walmart.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found. Please place walmart.csv in the folder {RAW_DIR}.")

    df = pd.read_csv(file_path)
    return df
