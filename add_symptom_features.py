# add symptom and dental questions to the 30-feature table
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_availability import CYCLES, codebook

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
KEYS = ["cycle", "seqn"]


# read xpt columns, fix 2**-260 zero bug, and set missing codes to NaN
def read_fields(folder, stem, fields, optional=()):
    data = pd.read_sas(folder / f"{stem}.xpt", format="xport")
    data.columns = data.columns.str.upper()
    assert data.SEQN.notna().all() and data.SEQN.is_unique, stem
    data = data.set_index(data.SEQN.astype(int))
    metadata = codebook(folder / f"{stem}.htm")
    cleaned = {}
    for field in fields:
        if field in optional and field not in data and field not in metadata:
            cleaned[field] = pd.Series(np.nan, index=data.index)
            continue
        cleaned[field] = data[field].replace(
            [np.ldexp(1.0, -260), *metadata[field]["missing_codes"]],
            [0, *([np.nan] * len(metadata[field]["missing_codes"]))],
        )
    return pd.DataFrame(cleaned, index=data.index)


def main():
    cohort = pd.read_csv(RESULTS / "cohort_30_features.csv")
    combined = []
    dental_parts = []
    for year, suffix in CYCLES:
        cycle = f"{year}-{year+1}"
        folder = ROOT / "raw" / cycle
        base = cohort.loc[cohort.cycle.eq(cycle)].set_index("seqn", drop=False)
        kidney = read_fields(folder, f"KIQ_U_{suffix}", ["KIQ480"]).reindex(base.index)
        fatigue = read_fields(folder, f"DPQ_{suffix}", ["DPQ040"]).reindex(base.index)
        first = read_fields(folder, f"DR1TOT_{suffix}", ["DR1DRSTZ", "DR1_320Z"]).reindex(base.index)
        sleep = read_fields(folder, f"SLQ_{suffix}", ["SLQ120"], optional=["SLQ120"]).reindex(base.index)
        weight = read_fields(folder, f"WHQ_{suffix}", ["WHD020", "WHD050", "WHQ060"]).reindex(base.index)
        # weight loss in the last year (lb) and if losing 10+ lb was not on purpose
        loss_lb = weight.WHD050.where(weight.WHD050.gt(0)) - weight.WHD020.where(weight.WHD020.gt(0))
        intentional = weight.WHQ060.where(weight.WHQ060.isin([1, 2]) & loss_lb.ge(10))
        loss_flag = pd.Series(np.nan, index=base.index)
        loss_flag.loc[loss_lb.lt(10) | intentional.eq(1)] = 0
        loss_flag.loc[intentional.eq(2)] = 1
        base["nightly_urination_count_5plus"] = kidney.KIQ480.where(kidney.KIQ480.between(0, 5))
        base["plain_water_day1_g"] = first.DR1_320Z.where(first.DR1DRSTZ.eq(1))
        # lb to kg
        base["self_reported_weight_loss_1y_kg"] = loss_lb * 0.45359237
        base["reported_unintentional_loss_10lb"] = loss_flag
        base["fatigue_frequency_2w"] = fatigue.DPQ040.where(fatigue.DPQ040.between(0, 3))
        base["daytime_sleepiness_frequency"] = sleep.SLQ120.where(sleep.SLQ120.between(0, 4))
        oral = read_fields(folder, f"OHQ_{suffix}", ["OHQ835", "OHQ845"]).reindex(base.index)
        dental = base[KEYS].copy()
        dental["self_rated_teeth_and_gums"] = oral.OHQ845.where(oral.OHQ845.isin([1, 2, 3, 4, 5]))
        # gum disease question (OHQ835) is only for age 30+
        dental["self_suspected_gum_disease"] = oral.OHQ835.where(oral.OHQ835.isin([1, 2]) & base.age.ge(30))
        assert not (oral.OHQ835.isin([1, 2]) & base.age.lt(30)).any()
        dental_parts.append(dental.reset_index(drop=True))
        assert base.loc[loss_flag.eq(1), "self_reported_weight_loss_1y_kg"].ge(4.535923699).all()
        # SLQ120 (daytime sleepiness) was not in the survey before 2015
        if year < 2015:
            assert base.daytime_sleepiness_frequency.isna().all()
        combined.append(base.reset_index(drop=True))

    extended = pd.concat(combined, ignore_index=True)
    pd.testing.assert_frame_equal(extended[cohort.columns], cohort)
    assert not extended.duplicated(KEYS).any()
    new_cols = extended.columns.difference(cohort.columns, sort=False)
    assert len(new_cols) == 6
    extended.to_csv(RESULTS / "cohort_with_symptom_candidates.csv", index=False)
    saved = pd.read_csv(RESULTS / "cohort_with_symptom_candidates.csv")
    pd.testing.assert_frame_equal(saved, extended)
    print(saved.groupby("cycle")[list(new_cols)].count().T.to_string())
    print(saved[list(new_cols)].count().to_string())
    print("Reported unintentional losses:", int(saved.reported_unintentional_loss_10lb.eq(1).sum()))
    dental = pd.concat(dental_parts, ignore_index=True)
    pd.testing.assert_frame_equal(dental[KEYS], extended[KEYS])
    final = extended.merge(dental, on=KEYS, how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(final[extended.columns], extended)
    destination = RESULTS / "cohort_with_symptom_and_oral_health_candidates.csv"
    final.to_csv(destination, index=False)
    pd.testing.assert_frame_equal(pd.read_csv(destination), final)
    for field in dental.columns.difference(KEYS, sort=False):
        print(field, int(final[field].notna().sum()))
    print("Age 30+:", int(final.age.ge(30).sum()))


if __name__ == "__main__":
    main()
