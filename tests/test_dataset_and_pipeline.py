# pytest checks for dataset rules, preprocessing, and evaluation helpers
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate
import preprocess

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FULL_CSV = RESULTS / "nhanes_2011_2018_dataset.csv"
TRAIN_CSV = RESULTS / "train.csv"
TEST_CSV = RESULTS / "test.csv"
HAS_CSVS = FULL_CSV.exists() and TRAIN_CSV.exists() and TEST_CSV.exists()

needs_data = pytest.mark.skipif(not HAS_CSVS, reason="generated dataset csv files are not present")


@needs_data
def test_train_test_no_overlap():
    full = pd.read_csv(FULL_CSV, usecols=["cycle", "seqn"])
    train = pd.read_csv(TRAIN_CSV, usecols=["cycle", "seqn"])
    test = pd.read_csv(TEST_CSV, usecols=["cycle", "seqn"])

    train_keys = set(zip(train["cycle"], train["seqn"]))
    test_keys = set(zip(test["cycle"], test["seqn"]))
    assert len(train_keys & test_keys) == 0
    assert len(train) + len(test) == len(full)


@needs_data
def test_feature_lists_match():
    expected = (ROOT / "features.txt").read_text(encoding="utf-8").splitlines()
    assert len(expected) == 63
    assert len(set(expected)) == 63
    assert set(preprocess.ALL_FEATURES) == set(expected)

    for path in (FULL_CSV, TRAIN_CSV, TEST_CSV):
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        assert cols == ["cycle", "seqn", "target", *expected]


@needs_data
def test_bmi_waist_and_bp_rules():
    df = pd.read_csv(FULL_CSV)

    # check BMI matches weight / height^2 (NHANES rounds BMI to 0.1)
    bmi_ok = df[["weight_kg", "height_cm", "bmi"]].dropna()
    calc_bmi = bmi_ok["weight_kg"] / (bmi_ok["height_cm"] / 100.0) ** 2
    assert (calc_bmi - bmi_ok["bmi"]).abs().max() < 0.25

    # check waist_height_ratio matches waist_cm / height_cm
    whr_ok = df[["waist_cm", "height_cm", "waist_height_ratio"]].dropna()
    calc_whr = whr_ok["waist_cm"] / whr_ok["height_cm"]
    assert np.allclose(calc_whr, whr_ok["waist_height_ratio"], atol=1e-6)

    # check age bounds (20 to 80 top-coded) and diastolic >= 30
    assert df["age"].between(20, 80).all()
    assert df["diastolic_mean"].dropna().ge(30.0).all()


@needs_data
def test_age_gated_columns_are_empty_below_limit():
    df = pd.read_csv(FULL_CSV)

    for col in ["self_suspected_gum_disease", "mouth_aching_frequency_1y", "floss_days_7d", "loose_tooth_without_injury"]:
        assert df.loc[df["age"] < 30, col].isna().all(), col

    assert df.loc[df["age"] < 36, "reported_weight_change_10y_kg"].isna().all()

    for col in ["taste_difficulty_1y", "persistent_mouth_sensation_1y"]:
        assert df.loc[df["age"] < 40, col].isna().all(), col


@needs_data
def test_target_values_and_split_balance():
    train = pd.read_csv(TRAIN_CSV, usecols=["target"])
    test = pd.read_csv(TEST_CSV, usecols=["target"])

    assert train["target"].isin([0, 1]).all()
    assert test["target"].isin([0, 1]).all()
    assert abs(train["target"].mean() - test["target"].mean()) < 0.005


def test_evaluate_metrics_and_threshold_synthetic():
    # small fake dataset to test metrics and threshold selection without raw files
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.12, 0.18, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55, 0.40, 0.60, 0.70, 0.85, 0.95])

    thr, m = evaluate.find_best_threshold(y_true, y_prob, min_specificity=0.70)
    assert 0.01 <= thr <= 0.99
    assert m["specificity"] >= 0.70
    assert m["roc_auc"] > 0.90
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y_true)


def test_make_cv_splits_synthetic():
    df = pd.DataFrame({
        "cycle": ["2011-2012"] * 20 + ["2013-2014"] * 20,
        "target": [0, 0, 0, 1] * 10,
    })
    splits = evaluate.make_cv_splits(df, n_splits=5, seed=42)
    assert len(splits) == 5
    seen = np.concatenate([va for _, va in splits])
    assert len(seen) == len(df) and len(set(seen)) == len(df)


def test_preprocessor_with_engineered_features_synthetic():
    df = pd.DataFrame({
        "age": [30.0, 50.0, 65.0, 40.0],
        "weight_kg": [80.0, 90.0, 70.0, 85.0],
        "highest_ever_weight_kg": [75.0, 100.0, 70.0, np.nan],
        "sex": [1.0, 2.0, 1.0, 2.0],
    })
    cols = ["age", "weight_kg", "highest_ever_weight_kg", "sex"]
    for m in ("logreg", "rf", "catboost"):
        prep = preprocess.build_preprocessor(m, cols, use_engineered=True)
        out = prep.fit_transform(df)
        if m == "catboost":
            assert "weight_to_highest_ratio" in out.columns
            assert "highest_weight_inconsistent" in out.columns
            assert out["highest_weight_inconsistent"].tolist() == ["1", "0", "0", "missing"]
        else:
            assert not np.isnan(out).any()
