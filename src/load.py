"""Step 1: read the six yearly CSVs and harmonize them into one parquet file.

Only parsing and renaming happen here; nothing is dropped or imputed (that is
cleaning, decided on the exploration set after the split).
"""
import json
import re

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from config import (ALL_PATH, CASE_STATUS_MAP, DATA_RAW, LEGAL_SUFFIXES,
                    RAW_CSV, RENAME, REPORT, YEARS)

TEXT_COLS = ["visa_class", "employer_name", "soc_title", "job_title", "full_time",
             "unit_of_pay", "employer_country", "case_status", "quarter",
             "worksite_city", "worksite_state", "employer_city", "employer_state"]
WAGE_COLS = ["prevailing_wage", "wage_offered_fy2019"]


def employer_key(name):
    """Upper-case, drop punctuation and trailing legal suffixes.

    'Infosys Limited', 'INFOSYS LTD.' -> 'INFOSYS'. Used to group filings by
    employer for the split; the raw name is kept.
    """
    if not isinstance(name, str):
        return np.nan
    s = name.upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    tokens = s.split()
    if tokens and tokens[0] == "THE":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens) or np.nan


def split_location(s):
    """'City, State' -> (city, state). Splits on the last comma."""
    parts = s.str.rsplit(",", n=1, expand=True)
    if parts.shape[1] == 1:
        parts[1] = np.nan
    no_comma = parts[1].isna() & s.notna()
    city = parts[0].str.strip().where(~no_comma)
    state = parts[1].str.strip()
    return city, state, int(no_comma.sum())


def load_year(year, log):
    df = pd.read_csv(RAW_CSV[year], dtype=str, keep_default_na=False)
    n = len(df)
    df = df.rename(columns=RENAME)
    for c in df.columns:
        df[c] = df[c].str.strip().replace("", np.nan)

    for c in WAGE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    df["quarter"] = df["quarter"] if "quarter" in df else np.nan

    raw_status = df["case_status"]
    df["case_status"] = raw_status.map(CASE_STATUS_MAP)
    unmapped = int((df["case_status"].isna() & raw_status.notna()).sum())

    df["worksite_city"], df["worksite_state"], ws_bad = split_location(df.pop("worksite"))
    df["employer_city"], df["employer_state"], el_bad = split_location(df.pop("employer_location"))

    df["fiscal_year"] = np.int16(year)
    for c in TEXT_COLS:
        df[c] = df[c].astype("category")
    # Series.map on a categorical maps each category once, not each row.
    df["employer_key"] = df["employer_name"].map(employer_key).astype("category")

    log[year] = {"rows": n, "case_status_unmapped": unmapped,
                 "worksite_without_comma": ws_bad, "employer_location_without_comma": el_bad}
    return df


def fy2019_wage_check():
    """Evidence that FY2019's Wage_Rate_Of_Pay is the offered wage, not a renamed
    prevailing wage. Offered wages are set by employers (often round numbers);
    prevailing wages come from DOL wage surveys (almost never round)."""
    rows, med = [], {}
    for y in YEARS:
        df = pd.read_csv(RAW_CSV[y], usecols=lambda c: c in
                         {"SOC_Title", "Prevailing_Wage", "Wage_Rate_Of_Pay", "Unit_Of_Pay"})
        col = "Wage_Rate_Of_Pay" if y == 2019 else "Prevailing_Wage"
        d = df[(df["Unit_Of_Pay"] == "Year") & (df[col] > 0)]
        med[y] = d.groupby("SOC_Title")[col].median()
        rows.append({"fiscal_year": y, "column": col, "median_annual": d[col].median(),
                     "pct_multiple_of_1000": 100 * ((d[col] % 1000) == 0).mean()})
    steps = []
    for a, b in zip(YEARS, YEARS[1:]):
        j = pd.concat([med[a], med[b]], axis=1, join="inner").dropna()
        steps.append({"from": a, "to": b, "common_soc_titles": len(j),
                      "median_pct_change": 100 * np.expm1(np.log(j.iloc[:, 1] / j.iloc[:, 0]).median())})
    t1, t2 = pd.DataFrame(rows), pd.DataFrame(steps)
    md = ["# FY2019 wage column: offered or prevailing?", "",
          "FY2019 has `Wage_Rate_Of_Pay`; every other year has `Prevailing_Wage`. Annual-unit rows only.", "",
          t1.to_markdown(index=False, floatfmt=".1f"), "",
          "Median year-on-year change of the per-SOC-title median wage:", "",
          t2.to_markdown(index=False, floatfmt=".1f"), "",
          "**Decision:** `Wage_Rate_Of_Pay` is the **offered wage**. About 40% of its values are whole "
          "thousands, against under 1% in every prevailing-wage year, and the per-SOC median jumps up into "
          "FY2019 and falls back out of it, while the other years move smoothly. It is kept as "
          "`wage_offered_fy2019`, and `prevailing_wage` is missing for FY2019 (missing by design). "
          "Prevailing-wage analyses therefore use FY2017-18 and FY2020-22.", ""]
    (REPORT / "fy2019_wage_check.md").write_text("\n".join(md))
    return t1, t2


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    print(fy2019_wage_check()[0].to_string(index=False))

    log, frames = {}, []
    for y in YEARS:
        frames.append(load_year(y, log))
        print(y, log[y])

    # Concatenate with unified categories (plain concat would fall back to strings).
    cols = ["fiscal_year", "quarter", "visa_class", "case_status", "employer_name",
            "employer_key", "employer_city", "employer_state", "employer_country",
            "soc_title", "job_title", "full_time", "worksite_city", "worksite_state",
            "prevailing_wage", "wage_offered_fy2019", "unit_of_pay"]
    out = {}
    for c in cols:
        parts = [f[c] for f in frames]
        if c in TEXT_COLS + ["employer_key"]:
            # all-NaN columns (quarter in FY2017-19) get float categories, so cast them to str first
            out[c] = union_categoricals([x.cat.set_categories(x.cat.categories.astype("str")) for x in parts],
                                        ignore_order=True)
        else:
            out[c] = pd.concat(parts, ignore_index=True).to_numpy()
    del frames
    df = pd.DataFrame(out)

    total = sum(v["rows"] for v in log.values())
    assert len(df) == total, (len(df), total)
    df.to_parquet(ALL_PATH, index=False)

    summary = {"per_year": {str(k): v for k, v in log.items()}, "total_rows": total,
               "columns": cols, "unique_employer_name": int(df["employer_name"].nunique()),
               "unique_employer_key": int(df["employer_key"].nunique())}
    (REPORT / "load_log.json").write_text(json.dumps(summary, indent=2))
    print(f"{total:,} rows x {len(cols)} columns -> {ALL_PATH}")
    print(f"employer names {summary['unique_employer_name']:,} -> keys {summary['unique_employer_key']:,}")


if __name__ == "__main__":
    main()
