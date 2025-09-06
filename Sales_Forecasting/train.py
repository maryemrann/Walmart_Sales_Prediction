# train.py
import pandas as pd
import numpy as np
from pathlib import Path
from config import PROCESSED_DIR, MODELS_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
import joblib

SEED = 42

def load_feature_data():
    """Load processed feature dataset"""
    df = pd.read_parquet(PROCESSED_DIR / "train_fe.parquet")
    return df

def get_features_and_target(df):
    """Select features and target for training"""
    # Only columns present in your dataset
    features = ['Store', 'Year', 'Month', 'Week', 'DayOfWeek', 'IsWeekend']
    X = df[features].copy()
    y = df['Weekly_Sales'].copy()
    
    # Convert categorical
    X['Store'] = X['Store'].astype('category')
    return X, y, features

def train_lgb(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    model = LGBMRegressor(
        objective='regression',
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=1000,
        random_state=SEED
    )
    
    # Fit model (no early_stopping_rounds to avoid errors)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds) ** 0.5  # take sqrt manually
    mape = np.mean(np.abs((y_val - preds) / (y_val + 1e-6))) * 100
    print(f"VAL MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    return preds, (mae, rmse, mape)


def main():
    df = load_feature_data()

    # Split train/validation (20% validation)
    X, y, features = get_features_and_target(df)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=False
    )

    # Train model
    model = train_lgb(X_train, y_train, X_val, y_val)

    # Evaluate
    preds, metrics = evaluate_model(model, X_val, y_val)

    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    model_file = MODELS_DIR / "sales_model.pkl"
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    main()
