# preprocessing.py
import pandas as pd
from pathlib import Path
from config import PROCESSED_DIR, RAW_DIR

def preprocess_data():
    # Load dataset
    df = pd.read_csv(RAW_DIR / "walmart.csv")

    # Parse Date (your file has DD-MM-YYYY)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Fill missing values if any (example)
    if 'Weekly_Sales' in df.columns:
        df['Weekly_Sales'] = df['Weekly_Sales'].fillna(0)

    # Save processed merged data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "train_merged.parquet", index=False)
    print("Processed data saved to", PROCESSED_DIR / "train_merged.parquet")
    return df

if __name__ == "__main__":
    preprocess_data()
