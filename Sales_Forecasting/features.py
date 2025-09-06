# features.py
import pandas as pd
from config import PROCESSED_DIR

def add_time_features(df):
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    return df

def build_features():
    df = pd.read_parquet(PROCESSED_DIR / "train_merged.parquet")
    df = add_time_features(df)
    PROCESSED_DIR.mkdir(exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "train_fe.parquet", index=False)
    print("Processed dataset saved to", PROCESSED_DIR / "train_fe.parquet")
    return df

if __name__ == "__main__":
    build_features()
