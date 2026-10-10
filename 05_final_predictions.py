import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------- 1) Load ----------
train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
valid = pd.read_csv("data/validation.csv", parse_dates=["date"])
template = pd.read_csv("data/validation_predictions_template.csv")
december = pd.read_csv("data/december_chart_inputs.csv", parse_dates=["date"])

# ---------- 2) Clean ----------
# Negative weights are impossible -> absolute value (train AND validation)
train["weight"] = train["weight"].abs()
valid["weight"] = valid["weight"].abs()

# Remove strange prices from TRAINING only (never drop validation rows)
train["rate_per_mile"] = train["posted_rate"] / train["distance"]
train = train[(train["rate_per_mile"] >= 0.8) & (train["rate_per_mile"] <= 6)].copy()

# ---------- 3) Features ----------
# market_index and quote_signal are NOT used: in forward-chaining validation the model
# was more accurate without them, and the December chart file does not contain them.
EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]

def add_features(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT_TYPES)
    return df

features = ["distance", "equipment", "weight", "month", "day_of_week",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

train = add_features(train)
valid = add_features(valid)

# ---------- 4) Train the FINAL model on ALL cleaned training data ----------
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1)
model.fit(train[features], train["posted_rate"])

# ---------- 5) Predict the 12,000 validation loads ----------
valid["predicted_rate"] = model.predict(valid[features]).clip(min=1.0)

# Fill the template in its original order
out = template[["load_id"]].merge(valid[["load_id", "predicted_rate"]], on="load_id", how="left")
assert out["predicted_rate"].notna().all(), "Some load_id values have no prediction"
out.to_csv("validation_predictions.csv", index=False)
print("Saved validation_predictions.csv:", out.shape)

# ---------- 6) December chart ----------
# City coordinates are fixed per city, so look them up from the data
all_loads = pd.concat([train, valid])
pickup_xy = all_loads[all_loads["pickup"] == "Lexington"][["pickup_lat", "pickup_lon"]].iloc[0]
delivery_xy = all_loads[all_loads["delivery"] == "Fort Wayne"][["delivery_lat", "delivery_lon"]].iloc[0]

dec = december.copy()
dec["pickup_lat"], dec["pickup_lon"] = pickup_xy["pickup_lat"], pickup_xy["pickup_lon"]
dec["delivery_lat"], dec["delivery_lon"] = delivery_xy["delivery_lat"], delivery_xy["delivery_lon"]
dec = add_features(dec)

preds = model.predict(dec[features]).clip(min=1.0)

# Keep the original 7 columns, in the original order, with the date as text
result = december.copy()
result["date"] = result["date"].dt.strftime("%Y-%m-%d")
result["predicted_rate"] = preds
result.to_csv("december_predictions.csv", index=False)
print("Saved december_predictions.csv:", result.shape)