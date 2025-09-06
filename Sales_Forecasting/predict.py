# predict.py
import pandas as pd
from pathlib import Path
import joblib

# Directories
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Features used in training
FEATURES = ['Store','Year','Month','Week','DayOfWeek','IsWeekend']

def load_model():
    model_path = MODELS_DIR / "sales_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    model = joblib.load(model_path)
    return model

def load_test():
    test_path = Path("Walmart.csv")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}.")
    test = pd.read_csv(test_path)
    return test

def add_time_features(df):
    # Make sure 'Date' column is datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # <-- set dayfirst=True
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1=Monday
    df['IsWeekend'] = df['DayOfWeek'].isin([6,7]).astype(int)
    return df


def prepare_X(df):
    df = add_time_features(df)
    X = df[FEATURES].copy()
    for c in ['Store']:
        X[c] = X[c].astype('category')
    return X

def main():
    model = load_model()
    test = load_test()
    X_test = prepare_X(test)
    test['Weekly_Sales_Pred'] = model.predict(X_test)
    output_file = OUTPUT_DIR / "predictions.csv"
    test.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
