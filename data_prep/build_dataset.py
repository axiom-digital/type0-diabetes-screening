# main script to build the final 63-feature dataset

import pandas as pd

import add_symptom_features
import analyze_availability
from expand_dataset import cycle_features

ROOT = analyze_availability.ROOT
RESULTS = analyze_availability.OUT
CYCLES = analyze_availability.CYCLES
KEYS = ["cycle", "seqn"]

FEATURES = [
    ("urge_urinary_leakage", "KIQ_U", "KIQ044", (1, 2), (2011, 2013, 2015, 2017)),
    ("poor_appetite_or_overeating_frequency", "DPQ", "DPQ050", (0, 1, 2, 3), (2011, 2013, 2015, 2017)),
    ("concentration_difficulty_frequency", "DPQ", "DPQ070", (0, 1, 2, 3), (2011, 2013, 2015, 2017)),
    ("loose_tooth_without_injury", "OHQ", "OHQ855", (1, 2), (2011, 2013)),
    ("sleep_breathing_difficulty_frequency", "SLQ", "SLQ040", (0, 1, 2, 3), (2015, 2017)),
    ("snoring_frequency", "SLQ", "SLQ030", (0, 1, 2, 3), (2015, 2017)),
]


# get simple 1-column questions by SEQN and keep only valid codes
def add_direct_questions(cohort):
    enriched = cohort.copy()
    for name, module, code, valid_codes, years in FEATURES:
        enriched[name] = pd.Series(pd.NA, index=cohort.index, dtype="Int64")
        for year, suffix in CYCLES:
            if year not in years:
                continue
            cycle = f"{year}-{year + 1}"
            folder = ROOT / "raw" / cycle
            source = add_symptom_features.read_fields(folder, f"{module}_{suffix}", [code])
            eligible = cohort["cycle"].eq(cycle)
            responses = cohort.loc[eligible, "seqn"].map(source[code]).round()
            enriched.loc[eligible, name] = responses.where(responses.isin(valid_codes)).astype("Int64")
    return enriched


def main():
    intermediate = RESULTS / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    # save middle files in results/intermediate and run the first two steps
    analyze_availability.OUT = intermediate
    add_symptom_features.RESULTS = intermediate
    analyze_availability.main()
    add_symptom_features.main()
    base = add_direct_questions(pd.read_csv(intermediate / "cohort_with_symptom_and_oral_health_candidates.csv"))

    # build extra features for each cycle
    parts = []
    for year, suffix in CYCLES:
        cycle = base.loc[base.cycle.eq(f"{year}-{year + 1}")].reset_index(drop=True)
        derived, _ = cycle_features(cycle, year, suffix)
        parts.append(pd.concat([cycle[KEYS], derived], axis=1))
    additions = pd.concat(parts, ignore_index=True)
    dataset = base.merge(additions, on=KEYS, how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(dataset[base.columns], base)

    # take only the 63 columns from features.txt
    selected = (ROOT / "features.txt").read_text(encoding="utf-8").splitlines()
    if len(selected) != 63 or len(set(selected)) != 63:
        raise ValueError("features.txt, tekil 63 aday sütun adı içermeli.")
    dataset = dataset[[*KEYS, "target", *selected]]

    output = RESULTS / "nhanes_2011_2018_dataset.csv"
    dataset.to_csv(output, index=False)
    print(f"{len(dataset):,} katılımcı, {len(selected)} özellik, {len(dataset.columns)} toplam sütun")
    print(output)


if __name__ == "__main__":
    main()
