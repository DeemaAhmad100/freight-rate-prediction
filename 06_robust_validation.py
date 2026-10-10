"""
Extra validation: how much can we trust the model?

1) Forward-chaining: train on the past, test on the next two months (3 times).
2) Honest holdout: evaluate with and without the unusual-price rows.
3) Market columns: does the model do better WITH or WITHOUT market_index / quote_signal?
"""
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# ---------- Load and clean (same rules as the final script) ----------
train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
train["weight"] = train["weight"].abs()
train["rate_per_mile"] = train["posted_rate"] / train["distance"]
train["month"] = train["date"].dt.month
train["day_of_week"] = train["date"].dt.dayofweek
train["equipment"] = pd.Categorical(train["equipment"], categories=["Dry Van", "Flatbed", "Reefer"])

clean = train[(train["rate_per_mile"] >= 0.8) & (train["rate_per_mile"] <= 6)]

WITH_MARKET = ["distance", "equipment", "weight", "month", "day_of_week",
               "market_index", "quote_signal",
               "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
NO_MARKET = [f for f in WITH_MARKET if f not in ("market_index", "quote_signal")]


def fit(df, features):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1)
    model.fit(df[features], df["posted_rate"])
    return model


def mae(df, model, features):
    return mean_absolute_error(df["posted_rate"], model.predict(df[features]))


# ---------- 1) + 3) Forward-chaining, with vs without market columns ----------
print("Forward-chaining validation (MAE in dollars, lower is better)")
print(f"{'train until':<12} {'test rows':>9} {'with market':>12} {'no market':>10}")
folds = [("2025-07-01", "2025-09-01"), ("2025-08-01", "2025-10-01"), ("2025-09-01", "2025-11-01")]
for start, end in folds:
    fit_part = clean[clean["date"] < start]
    test_part = clean[(clean["date"] >= start) & (clean["date"] < end)]
    a = mae(test_part, fit(fit_part, WITH_MARKET), WITH_MARKET)
    b = mae(test_part, fit(fit_part, NO_MARKET), NO_MARKET)
    print(f"{start:<12} {len(test_part):>9} {a:>12.1f} {b:>10.1f}")

# ---------- 2) Honest holdout: include the unusual-price rows in the test ----------
fit_part = clean[clean["date"] < "2025-09-01"]
holdout_clean = clean[clean["date"] >= "2025-09-01"]
holdout_all = train[train["date"] >= "2025-09-01"]
model = fit(fit_part, WITH_MARKET)
print("\nHoldout Sep-Oct (with market columns)")
print("MAE on cleaned rows      :", round(mae(holdout_clean, model, WITH_MARKET), 1))
print("MAE incl. unusual prices :", round(mae(holdout_all, model, WITH_MARKET), 1))

# ---------- Error by equipment type ----------
errors = (holdout_clean["posted_rate"] - model.predict(holdout_clean[WITH_MARKET])).abs()
print("\nMean absolute error by equipment:")
print(errors.groupby(holdout_clean["equipment"], observed=True).mean().round(1))