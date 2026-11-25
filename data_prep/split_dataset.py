# split the final dataset into 80% train and 20% test
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
KEYS = ["cycle", "seqn"]
# keep same seed so the split does not change
SEED = 42


def main():
    data = pd.read_csv(RESULTS / "nhanes_2011_2018_dataset.csv")
    if data[KEYS].isna().any().any() or data.duplicated(KEYS).any():
        raise ValueError("Katılımcı anahtarları eksiksiz ve tekil olmalı.")
    if not data["target"].isin([0, 1]).all():
        raise ValueError("Hedef sütununda yalnızca 0 ve 1 bulunmalı.")

    data = data.sort_values(KEYS).reset_index(drop=True)
    # keep similar cycle and target balance in train and test
    strata = data["cycle"].astype(str) + "_" + data["target"].astype(str)
    train, test = train_test_split(data, test_size=0.20, random_state=SEED, stratify=strata)
    train.sort_index().to_csv(RESULTS / "train.csv", index=False)
    test.sort_index().to_csv(RESULTS / "test.csv", index=False)
    print(f"Eğitim: {len(train)} kişi | Test: {len(test)} kişi")
    print(f"Eğitim pozitif oranı: {train['target'].mean():.2%} | Test: {test['target'].mean():.2%}")


if __name__ == "__main__":
    main()
