# evaluate.py
import pandas as pd
import matplotlib.pyplot as plt
from config import FIGURES_DIR, PROCESSED_DIR, MODELS_DIR
import joblib
from sklearn.metrics import mean_absolute_error
import numpy as np

def plot_actual_vs_pred():
    model_path = MODELS_DIR / "sales_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")

    model = joblib.load(model_path)
    df = pd.read_parquet(PROCESSED_DIR / "train_fe.parquet")

    features = ['Store','Year','Month','Week','DayOfWeek','IsWeekend']
    X = df[features].copy()
    X['Store'] = X['Store'].astype('category')

    preds = model.predict(X)
    df['pred'] = preds

    plt.figure(figsize=(12,5))
    plt.plot(df['Date'], df['Weekly_Sales'], label='actual')
    plt.plot(df['Date'], df['pred'], label='pred')
    plt.legend()
    plt.title("Actual vs Predicted")
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "actual_vs_pred.png")
    print("Saved figure to", FIGURES_DIR / "actual_vs_pred.png")

if __name__ == "__main__":
    plot_actual_vs_pred()
