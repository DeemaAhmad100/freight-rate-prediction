# First model: uses ALL features, including market_index and quote_signal.
# The final model (05_final_predictions.py) drops them after the comparison in 06_robust_validation.py.

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# ---------- 1) Load ----------
train = pd.read_csv("data/train_test.csv", parse_dates=["date"])

# ---------- 2) Clean ----------
# Negative weights are impossible -> use the absolute value
train["weight"] = train["weight"].abs()

# Remove strange prices (price per mile far from normal) from TRAINING only
train["rate_per_mile"] = train["posted_rate"] / train["distance"]
train = train[(train["rate_per_mile"] >= 0.8) & (train["rate_per_mile"] <= 6)].copy()

# ---------- 3) Features ----------
train["month"] = train["date"].dt.month
train["day_of_week"] = train["date"].dt.dayofweek
train["equipment"] = train["equipment"].astype("category")

features = ["distance", "equipment", "weight", "month", "day_of_week",
            "market_index", "quote_signal",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

# ---------- 4) Time-based split ----------
cutoff = pd.Timestamp("2025-09-01")
fit_part = train[train["date"] < cutoff]
holdout = train[train["date"] >= cutoff]

# ---------- 5) Simple baseline: distance * average rate per mile ----------
avg_rpm = fit_part["rate_per_mile"].mean()
baseline_pred = holdout["distance"] * avg_rpm
print("Baseline MAE:", round(mean_absolute_error(holdout["posted_rate"], baseline_pred), 2))

# ---------- 6) Train LightGBM ----------
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1)
model.fit(fit_part[features], fit_part["posted_rate"])

# ---------- 7) Evaluate on the holdout ----------
pred = model.predict(holdout[features])
mae = mean_absolute_error(holdout["posted_rate"], pred)
mape = (abs(holdout["posted_rate"] - pred) / holdout["posted_rate"]).mean() * 100
print("Model MAE :", round(mae, 2), "dollars")
print("Model MAPE:", round(mape, 2), "%")

# ---------- 8) Which features matter most ----------
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)