import pandas as pd

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])

# Step 1: list of columns
print(train.columns.tolist())

# Step 2: does distance affect price? (closer to 1 = strong link)
print("Correlation price vs distance:", train["posted_rate"].corr(train["distance"]))

# Step 3: does the truck type (equipment) change the price?
train["rate_per_mile"] = train["posted_rate"] / train["distance"]
print(train.groupby("equipment")["rate_per_mile"].mean())

# Step 4: does the month change the price?
train["month"] = train["date"].dt.month
print(train.groupby("month")["rate_per_mile"].mean())

# Step 5: do the market columns relate to the price?
print(train["market_index"].corr(train["rate_per_mile"]))
print(train["quote_signal"].corr(train["rate_per_mile"]))

# Step 5b: correlation of ALL numeric features
train["weight"] = train["weight"].abs()  # fix negative weights first
numeric_cols = ["distance", "weight", "market_index", "quote_signal",
                "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
print(train[numeric_cols + ["posted_rate", "rate_per_mile"]].corr()[["posted_rate", "rate_per_mile"]].round(3))

# Step 6: time-based split (train on the past, test on the "future")
cutoff = pd.Timestamp("2025-09-01")
fit_part = train[train["date"] < cutoff]
holdout_part = train[train["date"] >= cutoff]

print("Fit part:", len(fit_part), "rows,", fit_part["date"].min().date(), "->", fit_part["date"].max().date())
print("Holdout part:", len(holdout_part), "rows,", holdout_part["date"].min().date(), "->", holdout_part["date"].max().date())





