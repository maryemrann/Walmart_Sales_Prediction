# plot_predictions.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
OUTPUT_DIR = Path("output")
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"

# Load predictions
if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(f"{PREDICTIONS_FILE} not found. Run predict.py first.")

df = pd.read_csv(PREDICTIONS_FILE)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

# Sort by Date
df = df.sort_values('Date')

# Plot
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['Weekly_Sales'], label='Actual Sales', color='blue')
plt.plot(df['Date'], df['Weekly_Sales_Pred'], label='Predicted Sales', color='red', alpha=0.7)

# Optional: add 4-week rolling average for smoother visualization
df['Pred_Rolling'] = df['Weekly_Sales_Pred'].rolling(window=4).mean()
plt.plot(df['Date'], df['Pred_Rolling'], label='Predicted (4-week Rolling Avg)', color='green', linestyle='--')

plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.title("Actual vs Predicted Weekly Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
PLOT_FILE = OUTPUT_DIR / "actual_vs_predicted.png"
plt.savefig(PLOT_FILE)
plt.show()

print(f"Plot saved to {PLOT_FILE}")
