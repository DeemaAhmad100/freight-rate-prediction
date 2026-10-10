import pandas as pd

# 1) Load the data files (parse the "date" column as real dates)
train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
valid = pd.read_csv("data/validation.csv", parse_dates=["date"])

# 2) Check the size of each table (rows, columns)
print("Train shape:", train.shape)
print("Validation shape:", valid.shape)

# 3) Look at the first 5 rows to understand the columns
print("\nFirst 5 rows of train:")
print(train.head())

# 4) Check the time range of each file
print("\nTrain dates:", train["date"].min().date(), "->", train["date"].max().date())
print("Validation dates:", valid["date"].min().date(), "->", valid["date"].max().date())

# 5) Count missing values per column (only show columns that have any)
missing = train.isna().sum()
print("\nMissing values in train:")
print(missing[missing > 0])

# 6) Summary statistics of the target (what we want to predict)
print("\nposted_rate summary:")
print(train["posted_rate"].describe())