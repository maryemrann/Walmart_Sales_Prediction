from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("processed")
df = pd.read_parquet(PROCESSED_DIR / "train_fe.parquet")
print(df.columns)
