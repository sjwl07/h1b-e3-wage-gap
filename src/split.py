"""Step 2: hold out a confirmation set before any EDA.

80/20 StratifiedGroupKFold: grouped by employer_key so no employer is in both
sets (an employer's filings are near-copies of each other), stratified by
fiscal_year x case_status. The first of the 5 folds is the confirmation set.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import ALL_PATH, CONFIRM_FRACTION, CONFIRM_PATH, EXPLORE_PATH, REPORT, SEED


def share_table(df, is_confirm, col):
    t = pd.crosstab(df[col], is_confirm, normalize="columns").mul(100).round(2)
    t.columns = ["explore_%", "confirm_%"]
    return {str(k): v for k, v in t.to_dict("index").items()}


def main():
    df = pd.read_parquet(ALL_PATH)
    n = len(df)

    # 92 rows have no employer name; they share one group so they stay together.
    groups = df["employer_key"].astype("str").fillna("<MISSING>").to_numpy()
    strata = (df["fiscal_year"].astype(str) + "_" + df["case_status"].astype(str)).to_numpy()

    n_splits = round(1 / CONFIRM_FRACTION)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    _, confirm_idx = next(sgkf.split(np.zeros(n), strata, groups))
    is_confirm = np.zeros(n, dtype=bool)
    is_confirm[confirm_idx] = True

    overlap = set(groups[is_confirm]) & set(groups[~is_confirm])
    assert not overlap, f"{len(overlap)} employers in both sets"

    explore, confirm = df[~is_confirm], df[is_confirm]
    assert len(explore) + len(confirm) == n
    explore.to_parquet(EXPLORE_PATH, index=False)
    confirm.to_parquet(CONFIRM_PATH, index=False)

    top = df["employer_key"].value_counts().head(10).index
    log = {
        "seed": SEED,
        "method": f"StratifiedGroupKFold(n_splits={n_splits}, shuffle=True), first fold held out",
        "group": "employer_key", "stratify": "fiscal_year x case_status",
        "rows_total": n, "rows_explore": len(explore), "rows_confirm": len(confirm),
        "confirm_fraction_actual": round(len(confirm) / n, 4),
        "employers_explore": int(explore["employer_key"].nunique()),
        "employers_confirm": int(confirm["employer_key"].nunique()),
        "employers_in_both": 0,
        "top10_employers_in_confirm": [k for k in top if k in set(confirm["employer_key"].astype(str))],
        "share_by_fiscal_year": share_table(df, is_confirm, "fiscal_year"),
        "share_by_case_status": share_table(df, is_confirm, "case_status"),
        "share_by_visa_class": share_table(df, is_confirm, "visa_class"),
        "share_by_full_time": share_table(df, is_confirm, "full_time"),
        "median_prevailing_wage_year_unit": {
            "explore": float(explore.loc[explore["unit_of_pay"] == "Year", "prevailing_wage"].median()),
            "confirm": float(confirm.loc[confirm["unit_of_pay"] == "Year", "prevailing_wage"].median()),
        },
    }
    (REPORT / "split_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
