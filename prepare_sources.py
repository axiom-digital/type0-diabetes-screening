# download nhanes files from cdc and food category excel files from usda
from zipfile import is_zipfile

from download_sources import CYCLES, MODULES, ROOT, main as download_nhanes, retrieve

EXTRA_MODULES = (
    "KIQ_U", "DPQ", "OHQ", "DBQ", "SMQFAM", "FSQ", "OCQ",
    "PFQ", "DR1IFF", "DLQ", "CSQ",
)
# dlq and csq are not in all survey years
RESTRICTED_YEARS = {"DLQ": {2013, 2015, 2017}, "CSQ": {2011, 2013}}
USDA_BASE = "https://www.ars.usda.gov/ARSUserFiles/80400530/apps"


def main():
    download_nhanes([*MODULES, *EXTRA_MODULES], RESTRICTED_YEARS)
    for year, _ in CYCLES:
        cycle = f"{year}-{year + 1}"
        target = ROOT / "raw" / cycle / "wweia_food_categories.xlsx"
        label = f"{year % 100:02}{(year + 1) % 100:02}"
        # xlsx files are zip files, so we check if it is already downloaded and ok
        if not is_zipfile(target):
            url = f"{USDA_BASE}/WWEIA{label}_foodcat_FNDDS.xlsx"
            contents, _ = retrieve(url)
            # write to a temp file first so we don't keep a broken file
            temp = target.with_suffix(".part")
            temp.write_bytes(contents)
            if not is_zipfile(temp):
                temp.unlink()
                raise ValueError(f"Unvalid USDA folder: {url}")
            temp.replace(target)
        print(f"Ready: {cycle} USDA WWEIA", flush=True)


if __name__ == "__main__":
    main()
