# Freight Rate Prediction

## Overview

This project predicts the posted rate (USD) of truck loads from their lane, distance,
equipment, weight and date. It is a supervised regression problem on tabular data.
The model is a LightGBM regressor, validated with time-based splits (train on the past,
test on the future). The final model predicts the 12,000 validation loads (Nov-Dec 2025)
and a fixed December 2025 chart for one lane (Lexington to Fort Wayne).

## Setup

1. Install Python 3.11+ and the dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Load the data: create a `data/` folder in the project root and put the provided CSV files in it with these names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

The `data/` folder is not included in this repository (the data is provided by the company).

## How to run

Run the scripts in this order from the project root:

```bash
python 01_explore_data.py          # data overview: shape, date ranges, missing values
python 02_data_quality_checks.py   # data-quality issues: negative weights, unusual prices
python 03_feature_analysis.py      # feature exploration and the time-based split
python 04_validate_model.py        # first model (all features), Sep-Oct 2025 holdout
python 06_robust_validation.py     # forward-chaining validation, with vs without market columns
python 07_model_comparison.py      # LightGBM vs Linear Regression, Random Forest, HistGradientBoosting
python 05_final_predictions.py     # final model -> validation_predictions.csv + december_predictions.csv
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
python -m pytest -q                # checks the output files
```

## Output

| File | Contents |
|---|---|
| `validation_predictions.csv` | Two columns, `load_id,predicted_rate`: one predicted rate (USD) for each of the 12,000 validation loads, in the template's order |
| `december_predictions.csv` | The 7 columns of `december_chart_inputs.csv` with `predicted_rate` filled for each day of December 2025 (31 rows) |
| `scorer_results/candidate_december.png` | December chart produced by `score.py` |

## Quality checks

**Data-quality issues found and how they were handled**

- Negative weights (292 train, 145 validation): replaced with the absolute value.
- Missing `weight` (300 train, 165 validation) and `market_index` (374 train, 249 validation): left as missing; LightGBM handles missing values natively.
- Unrealistic prices (536 training rows, 1.1%): rate per mile above $6 or below $0.80. Removed from training only; no validation row is dropped.
- Duplicate `load_id`: none found.

**Validation checks performed on the outputs**

- `score.py` checks that `validation_predictions.csv` has exactly the columns `load_id,predicted_rate`, 12,000 rows, unique IDs matching the validation set, and positive numeric rates. It also checks that the December file has the original seven columns, 31 daily rows (2025-12-01 to 2025-12-31), fixed inputs, and positive rates.
- `tests/test_outputs.py` (pytest) re-checks the columns, row count, unique IDs, ID order against the template, positive rates, and the December dates.

## Approach and results

- **Split:** time-based. Train on Jan-Aug 2025, validate on Sep-Oct 2025, because the validation set (Nov-Dec 2025) comes after the training period and a random split would leak future information. Forward-chaining with three folds confirms the result is stable over time.
- **Features:** distance, equipment, weight, month, day of week, pickup/delivery latitude and longitude.
- **Market columns:** `market_index` and `quote_signal` were tested with and without. The model was more accurate without them in the two most recent folds, and the December chart file does not contain them, so the final model excludes them.
- **Model:** LightGBM (500 trees, learning rate 0.05). Handles missing values natively and needs no feature scaling. The tree-based alternatives tested performed similarly.
- **Final model:** retrained on all cleaned training data before predicting.

| Sep-Oct 2025 holdout (final feature set) | MAE |
|---|---|
| Baseline: distance x average rate per mile | $232.7 |
| LightGBM, cleaned test rows | $69.0 (about 3.0%) |
| LightGBM, including unusual-price rows | $113.9 |

The final validation metrics are calculated by the company after submission.