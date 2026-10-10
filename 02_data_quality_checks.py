import pandas as pd

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
valid = pd.read_csv("data/validation.csv", parse_dates=["date"])

# --- Check 1: negative weights (a truck cannot carry negative weight) ---
print("Negative weights  -> train:", (train["weight"] < 0).sum(), "| validation:", (valid["weight"] < 0).sum())

# --- Check 2: rate per mile (price / distance) to spot weird prices ---
train["rate_per_mile"] = train["posted_rate"] / train["distance"]
print("\nRate per mile summary:")
print(train["rate_per_mile"].describe().round(2))
print("Rows with rate/mile > 6  :", (train["rate_per_mile"] > 6).sum())
print("Rows with rate/mile < 0.8:", (train["rate_per_mile"] < 0.8).sum())

# --- Check 3: some of the strangest rows ---
cols = ["pickup", "delivery", "distance", "equipment", "posted_rate", "rate_per_mile"]
print("\nHighest rate per mile:")
print(train.sort_values("rate_per_mile").tail(5)[cols])
print("\nLowest rate per mile:")
print(train.sort_values("rate_per_mile").head(5)[cols])

# --- Check 4: duplicates ---
print("\nDuplicate load_id:", train["load_id"].duplicated().sum())

# --- Check 5: validation missing values ---
missing_v = valid.isna().sum()
print("\nMissing values in validation:")
print(missing_v[missing_v > 0])