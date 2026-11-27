# shared feature sets used to compare logreg, rf, and catboost fairly
from src.preprocess import ALL_FEATURES

FEATURE_SETS = {
    # classic 7-question screening baseline
    "ada_like_7": [
        "age", "sex", "bmi", "waist_cm", "systolic_mean",
        "family_diabetes", "leisure_activity_min_week",
    ],
    # 10-question phone survey with no medical devices
    "custom_10": [
        "age", "sex", "bmi", "family_diabetes", "alcohol_frequency",
        "smoking_status", "nightly_urination_count_5plus",
        "self_rated_teeth_and_gums", "urge_urinary_leakage",
        "highest_ever_weight_kg",
    ],
    # 15-question home survey with no blood pressure or waist tape
    "no_device_15": [
        "age", "sex", "height_cm", "weight_kg", "bmi",
        "weight_1year_ago_kg", "highest_ever_weight_kg",
        "weight_to_highest_ratio", "reported_weight_change_10y_kg",
        "family_diabetes", "education", "smoking_status",
        "alcohol_frequency", "sleep_workday_hours", "self_rated_diet_quality",
    ],
    # top 20 practical features (no 24h food diary, no >50% cycle-only cols)
    "practical_20": [
        "age", "alcohol_frequency", "weight_to_highest_ratio", "systolic_mean",
        "waist_height_ratio", "bmi", "sex", "education",
        "reported_weight_change_10y_kg", "cigarettes_per_smoking_day",
        "frozen_meals_pizza_30d", "highest_ever_weight_kg",
        "sleep_workday_hours", "self_suspected_gum_disease", "height_cm",
        "drinks_per_drinking_day", "family_diabetes",
        "sleep_problems_or_oversleep_frequency", "self_rated_diet_quality",
        "weight_kg",
    ],
    # top 30 from catboost permutation ranking (includes weight_kg for weight_to_highest_ratio)
    "top_30": [
        "age", "alcohol_frequency", "weight_to_highest_ratio", "systolic_mean",
        "waist_height_ratio", "bmi", "sex", "education",
        "reported_weight_change_10y_kg", "cigarettes_per_smoking_day",
        "frozen_meals_pizza_30d", "highest_ever_weight_kg",
        "sleep_workday_hours", "self_suspected_gum_disease", "height_cm",
        "drinks_per_drinking_day", "family_diabetes",
        "sleep_problems_or_oversleep_frequency", "fiber_g",
        "taste_difficulty_1y", "self_rated_diet_quality", "waist_cm",
        "mouth_aching_frequency_1y", "floss_days_7d",
        "sweetened_beverages_day1_g", "weight_1year_ago_kg",
        "marital_status", "snoring_frequency", "away_from_home_meals_7d",
        "weight_kg",
    ],
    # all 63 raw features plus weight_to_highest_ratio
    "all_64": list(ALL_FEATURES) + ["weight_to_highest_ratio"],
}


# return only columns that exist in raw train.csv (engineered ones are built in pipeline)
def raw_input_cols(cols):
    raw_set = set(ALL_FEATURES)
    return [c for c in cols if c in raw_set]
