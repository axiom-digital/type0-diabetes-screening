# shared preprocessing pipelines for logreg, rf, and catboost
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "results" / "train.csv"
KEY_COLS = ["cycle", "seqn", "target"]

# unordered categories
NOMINAL_COLS = [
    "sex",
    "marital_status",
    "smoking_status",
    "employment_status_last_week",
]

# yes/no columns (some have missing values because of age or cycle rules)
BINARY_COLS = [
    "family_diabetes",
    "reported_unintentional_loss_10lb",
    "self_suspected_gum_disease",
    "urge_urinary_leakage",
    "loose_tooth_without_injury",
    "attempted_weight_loss_1y",
    "walking_difficulty_without_equipment",
    "serious_vision_difficulty",
    "taste_difficulty_1y",
    "persistent_mouth_sensation_1y",
    "smoking_inside_home",
]

# ordered survey scales (keep number order)
ORDINAL_COLS = [
    "education",
    "alcohol_frequency",
    "nightly_urination_count_5plus",
    "fatigue_frequency_2w",
    "daytime_sleepiness_frequency",
    "self_rated_teeth_and_gums",
    "poor_appetite_or_overeating_frequency",
    "concentration_difficulty_frequency",
    "sleep_breathing_difficulty_frequency",
    "snoring_frequency",
    "self_rated_diet_quality",
    "mouth_aching_frequency_1y",
    "sleep_problems_or_oversleep_frequency",
    "adult_food_security",
]

# continuous numeric columns
NUMERIC_COLS = [
    "age",
    "income_poverty_ratio",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "waist_height_ratio",
    "systolic_mean",
    "diastolic_mean",
    "pulse",
    "work_activity_min_week",
    "leisure_activity_min_week",
    "transport_min_week",
    "sedentary_min_day",
    "cigarettes_per_smoking_day",
    "drinks_per_drinking_day",
    "sleep_workday_hours",
    "weight_1year_ago_kg",
    "energy_kcal",
    "carbohydrate_g",
    "protein_g",
    "total_fat_g",
    "total_sugar_g",
    "fiber_g",
    "plain_water_day1_g",
    "weight_change_1y_kg",
    "floss_days_7d",
    "reported_weight_change_10y_kg",
    "highest_ever_weight_kg",
    "away_from_home_meals_7d",
    "fast_food_pizza_meals_7d",
    "ready_to_eat_foods_30d",
    "frozen_meals_pizza_30d",
    "sweetened_beverages_day1_g",
]

ALL_FEATURES = NOMINAL_COLS + BINARY_COLS + ORDINAL_COLS + NUMERIC_COLS

# engineered features that improved CV in 03_feature_selection.ipynb
ENGINEERED_NUMERIC_COLS = ["weight_to_highest_ratio"]
ENGINEERED_BINARY_COLS = ["highest_weight_inconsistent"]
ENGINEERED_FEATURES = ENGINEERED_NUMERIC_COLS + ENGINEERED_BINARY_COLS


# load train.csv and return dataframe plus feature list
def load_train_data(path=TRAIN_PATH):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in KEY_COLS]
    return df, feature_cols


# turn category codes into clean strings and mark NaN as "missing"
def to_cat_strings(df):
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        vals = []
        for v in s:
            if pd.isna(v):
                vals.append("missing")
            elif isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
                vals.append(str(int(v)))
            else:
                vals.append(str(v))
        out[col] = vals
    return out


# fill structural zeros for non-smokers and non-drinkers before imputation
class StructuralZeroFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        df = X.copy()
        if "cigarettes_per_smoking_day" in df.columns and "smoking_status" in df.columns:
            non_smoker = df["smoking_status"].isin([0, 1])
            df.loc[non_smoker & df["cigarettes_per_smoking_day"].isna(), "cigarettes_per_smoking_day"] = 0.0
        if "drinks_per_drinking_day" in df.columns and "alcohol_frequency" in df.columns:
            non_drinker = df["alcohol_frequency"].eq(0)
            df.loc[non_drinker & df["drinks_per_drinking_day"].isna(), "drinks_per_drinking_day"] = 0.0
        return df


# keep NaN in numeric columns for catboost and turn category columns into strings
class CatBoostPrep(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        cols = self.num_cols + self.cat_cols
        df = X[cols].copy()
        for col in self.num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        if self.cat_cols:
            df[self.cat_cols] = to_cat_strings(df[self.cat_cols])
        return df


# add the 2 engineered weight features inside the pipeline
class WeightFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        df = X.copy()
        if "weight_kg" in df.columns and "highest_ever_weight_kg" in df.columns:
            high = df["highest_ever_weight_kg"]
            cur = df["weight_kg"]
            df["weight_to_highest_ratio"] = cur / high.where(high > 0)
            both = high.notna() & cur.notna()
            df["highest_weight_inconsistent"] = np.where(
                both,
                (high < cur - 2.0).astype(float),
                np.nan,
            )
        return df


# build a leak-free preprocessor for the chosen model and feature list
def build_preprocessor(model_type, feature_cols=None, use_engineered=False):
    model_type = model_type.lower()
    if model_type not in {"logreg", "rf", "catboost"}:
        raise ValueError("model_type must be 'logreg', 'rf', or 'catboost'")

    cols = list(ALL_FEATURES) if feature_cols is None else list(feature_cols)
    add_fe = use_engineered and "weight_kg" in cols and "highest_ever_weight_kg" in cols
    if add_fe:
        for c in ENGINEERED_FEATURES:
            if c not in cols:
                cols.append(c)

    num_set = set(NUMERIC_COLS) | set(ORDINAL_COLS) | set(ENGINEERED_NUMERIC_COLS)
    cat_set = set(NOMINAL_COLS) | set(BINARY_COLS) | set(ENGINEERED_BINARY_COLS)

    num_cols = [c for c in cols if c in num_set]
    cat_cols = [c for c in cols if c in cat_set]
    unknown = [c for c in cols if c not in num_set and c not in cat_set]
    if unknown:
        raise ValueError(f"unknown columns: {unknown}")

    steps = []
    if add_fe:
        steps.append(("fe", WeightFeatureAdder()))
    steps.append(("zeros", StructuralZeroFiller()))

    if model_type == "catboost":
        steps.append(("prep", CatBoostPrep(cat_cols=cat_cols, num_cols=num_cols)))
        return Pipeline(steps)

    # logreg needs scaling; rf does not
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if model_type == "logreg":
        num_steps.append(("scaler", RobustScaler()))

    cat_pipe = Pipeline([
        ("to_str", FunctionTransformer(to_cat_strings, validate=False)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    steps.append(("cols", ColumnTransformer(transformers=transformers, remainder="drop")))
    return Pipeline(steps)
