import pandas as pd
import pytest

PRED_FILE = "validation_predictions.csv"
DEC_FILE = "december_predictions.csv"


@pytest.fixture(scope="module")
def predictions():
    return pd.read_csv(PRED_FILE)


@pytest.fixture(scope="module")
def december():
    return pd.read_csv(DEC_FILE)


def test_validation_has_exact_columns(predictions):
    assert list(predictions.columns) == ["load_id", "predicted_rate"]


def test_validation_has_12000_unique_rows(predictions):
    assert len(predictions) == 12_000
    assert predictions["load_id"].is_unique


def test_validation_ids_match_the_template(predictions):
    template = pd.read_csv("data/validation_predictions_template.csv")
    assert list(predictions["load_id"]) == list(template["load_id"])


def test_validation_rates_are_positive_numbers(predictions):
    assert predictions["predicted_rate"].notna().all()
    assert (predictions["predicted_rate"] > 0).all()


def test_december_has_31_days_and_positive_rates(december):
    assert len(december) == 31
    assert december["date"].min() == "2025-12-01"
    assert december["date"].max() == "2025-12-31"
    assert (december["predicted_rate"] > 0).all()