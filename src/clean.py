"""Step 4: cleaning.

Every rule is fit on the exploration set only. The fitted parameters (SOC title
mapping, IQR bounds) are saved so the *same* rules can later be applied to the
confirmation set without re-fitting on it.

No rows are dropped: case_status is known for every row and the prevailing
wage is only one of the outcomes, so a bad wage is set to NaN and flagged
instead of throwing away a valid case decision with it.
"""
import difflib
import json
import re

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, EXPLORE_PATH, REPORT

PAY_MULT = {"Year": 1, "Month": 12, "Bi-Weekly": 26, "Week": 52, "Hour": 2080}  # DOL: 2,080 h/year
MAX_ANNUAL = 1_000_000  # no prevailing wage is near $1M/year; above it the unit must be wrong
MIN_ANNUAL_FIGURE = 15_000  # a "per hour/week/month" value this large can only be an annual figure
SOC_MIN_ROWS = 30  # SOC spellings rarer than this are fuzzy-matched to a common one
SOC_FUZZY_CUTOFF = 0.92
SOC_TOP_K = 50  # top 50 titles cover 87.7% of exploration rows; the rest -> "Other" for encoding
IQR_K = 1.5
IQR_MIN_GROUP = 30  # SOC titles with fewer wages use pooled bounds; quartiles of <30 points are unstable

SOURCE_COLS = ["fiscal_year", "quarter", "visa_class", "case_status", "employer_name", "employer_city",
               "employer_state", "employer_country", "soc_title", "job_title", "full_time",
               "worksite_city", "worksite_state", "prevailing_wage", "wage_offered_fy2019", "unit_of_pay"]

ONE_HOT_COLS = ["fiscal_year", "quarter", "visa_class", "case_status", "full_time", "unit_of_pay",
                "worksite_state", "worksite_region", "employer_state", "soc_group"]
NUMERIC_COLS = ["log_pw", "log_offered_fy2019", "n_identical", "log_employer_volume", "same_state",
                "unit_relabelled", "pw_outlier", "offered_outlier", "worksite_missing",
                "is_senior", "is_lead", "is_manager", "is_junior"]

US_STATES = {
    "Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
                  "New Jersey", "New York", "Pennsylvania"],
    "Midwest": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota",
                "Missouri", "Nebraska", "North Dakota", "South Dakota"],
    "South": ["Delaware", "District Of Columbia", "Florida", "Georgia", "Maryland", "North Carolina",
              "South Carolina", "Virginia", "West Virginia", "Alabama", "Kentucky", "Mississippi",
              "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "West": ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming",
             "Alaska", "California", "Hawaii", "Oregon", "Washington"],
    "Territory": ["Puerto Rico", "Guam", "U.S. Virgin Islands", "Northern Mariana Islands", "American Samoa"],
}
REGION = {s.lower(): r for r, states in US_STATES.items() for s in states}
USA = "United States Of America"

SENIORITY = {
    "is_senior": r"\b(?:SENIOR|SR)\b",
    "is_lead": r"\b(?:LEAD|PRINCIPAL|STAFF|ARCHITECT)\b",
    "is_manager": r"\b(?:MANAGER|MGR|DIRECTOR|HEAD|VP|VICE PRESIDENT)\b",
    "is_junior": r"\b(?:JUNIOR|JR|ENTRY|INTERN|TRAINEE|GRADUATE)\b",
}


def per_category(s, f):
    """Apply f to the categories of a categorical Series once, then broadcast to rows."""
    s = s.astype("category")
    vals = pd.Series(f(s.cat.categories.astype("str")), index=range(len(s.cat.categories)))
    out = vals.reindex(s.cat.codes).to_numpy()
    return pd.Series(out, index=s.index)


def by_design_missing(df):
    """Cells missing because that year's file doesn't have the column at all."""
    fy = df["fiscal_year"]
    n = (df["quarter"].isna() & (fy < 2020)).sum() if "quarter" in df else 0
    for c in ["prevailing_wage", "annual_pw", "log_pw"]:
        if c in df:
            n += (df[c].isna() & (fy == 2019)).sum()
    for c in ["wage_offered_fy2019", "annual_offered_fy2019", "log_offered_fy2019"]:
        if c in df:
            n += (df[c].isna() & (fy != 2019)).sum()
    return n


class Log:
    def __init__(self):
        self.rows = []

    def add(self, step, df, reason, prev_rows=None, cells_changed=0):
        na = df.isna().sum().sum()
        self.rows.append({
            "step": step,
            "rows": len(df),
            "rows_removed": 0 if prev_rows is None else prev_rows - len(df),
            "features": df.shape[1],
            "cells_changed": int(cells_changed),
            "% missing cells": round(100 * na / df.size, 2),
            "% missing (excl. by design)": round(100 * (na - by_design_missing(df)) / df.size, 2),
            "reason": reason,
        })

    def frame(self):
        return pd.DataFrame(self.rows)


# ---------- SOC title harmonization ----------

def soc_key(t):
    """Lower-case letters only, leading SOC code and plural 's' removed: '15-1121.00Computersystemsanalysts',
    'Computer Systems Analyst' and 'COMPUTER SYSTEMS ANALYSTS' share one key."""
    t = re.sub(r"^[\d\-\.\s]+", "", t.lower().replace("&amp;", "&"))
    return "".join(re.sub(r"s$", "", w) for w in re.findall(r"[a-z]+", t))


def fit_soc_map(soc):
    counts = soc.value_counts()
    counts = counts[counts > 0]
    keys = pd.Series([soc_key(t) for t in counts.index], index=counts.index)
    key_n = counts.groupby(keys).sum()
    # canonical label for a key: its most frequent spelling
    canon = {k: grp.idxmax() for k, grp in counts.groupby(keys)}
    common = key_n[key_n >= SOC_MIN_ROWS].index.tolist()
    key_to_label = {}
    for k in key_n.index:
        target = k
        if key_n[k] < SOC_MIN_ROWS:
            m = difflib.get_close_matches(k, common, n=1, cutoff=SOC_FUZZY_CUTOFF)
            target = m[0] if m else k
        key_to_label[k] = canon[target]
    return {"key_to_label": key_to_label, "common_keys": common}


def apply_soc_map(soc, p):
    k2l, common = p["key_to_label"], p["common_keys"]

    def f(cats):
        out = []
        for t in cats:
            k = soc_key(t)
            if k in k2l:
                out.append(k2l[k])
            else:  # spelling never seen in exploration
                m = difflib.get_close_matches(k, common, n=1, cutoff=SOC_FUZZY_CUTOFF)
                out.append(k2l[m[0]] if m else t)
        return out
    return per_category(soc, f).astype("category")


# ---------- IQR on log annual wage within SOC title ----------

def _iqr_bounds(lv):
    q1, q3 = lv.quantile([0.25, 0.75])
    return float(q1 - IQR_K * (q3 - q1)), float(q3 + IQR_K * (q3 - q1))


def fit_iqr(log_wage, group):
    lv = log_wage.dropna()
    g = group.loc[lv.index].astype("str")
    sizes = g.value_counts()
    big = sizes[sizes >= IQR_MIN_GROUP].index
    per = lv[g.isin(big)].groupby(g[g.isin(big)]).quantile([0.25, 0.75]).unstack()
    iqr = per[0.75] - per[0.25]
    lo, hi = per[0.25] - IQR_K * iqr, per[0.75] + IQR_K * iqr
    return {"pooled": _iqr_bounds(lv),
            "per_soc": {k: (float(lo[k]), float(hi[k])) for k in per.index}}


def iqr_outlier_mask(log_wage, group, p):
    g = group.astype("str")
    lo = g.map({k: v[0] for k, v in p["per_soc"].items()}).astype(float).fillna(p["pooled"][0])
    hi = g.map({k: v[1] for k, v in p["per_soc"].items()}).astype(float).fillna(p["pooled"][1])
    return log_wage.notna() & ((log_wage < lo) | (log_wage > hi))


# ---------- main cleaning ----------

def clean(df, params=None):
    """Clean `df`. If `params` is None they are fit on `df` (exploration set only)."""
    fitting = params is None
    params = params or {}
    df = df.copy()
    log = Log()
    log.add("0. raw (rules fitted here)" if fitting else "0. raw (exploration-fitted rules applied)", df,
            "as loaded and harmonized by load.py; blanks already NaN")

    # 1. exact duplicates: kept, counted
    n_identical = df.groupby(SOURCE_COLS, observed=True, dropna=False)["fiscal_year"].transform("size")
    df["n_identical"] = n_identical.astype("int32")
    n_dup = int((n_identical > 1).sum() - df.loc[n_identical > 1].drop_duplicates(SOURCE_COLS).shape[0])
    dup_rate = (n_identical > 1).groupby(df["case_status"], observed=True).mean().mul(100).round(1).to_dict()
    log.add("1. exact duplicates: KEEP, add n_identical", df,
            f"{n_dup:,} rows are exact copies of another row. Kept: the Kaggle file is a column subset of DOL "
            "data where every row is a separate LCA with its own case number, so an identical row is most "
            "likely a separate filing for another worker (employers file one LCA per position). Evidence: "
            "the share of duplicated rows rises with employer filing volume, and it differs by status "
            f"({dup_rate} %), which an export glitch wouldn't produce. n_identical keeps the information; "
            "correlated filings are handled later with employer-clustered SEs")

    # 2. text normalization of job titles
    before = df["job_title"].nunique()
    old = df["job_title"]
    df["job_title"] = per_category(old, lambda c: (
        c.str.upper().str.replace(r"\[[\d\-\.]+\]", "", regex=True)   # '[15-1199.02]' SOC codes
         .str.replace(r"\s+", " ", regex=True).str.strip())).astype("category")
    n_changed = int((df["job_title"].astype("str") != old.astype("str")).sum())
    log.add("2. normalize job_title text", df,
            f"upper-case, remove bracketed SOC codes, collapse spaces: 'Software Engineer' and "
            f"'SOFTWARE ENGINEER' are one title. Distinct titles {before:,} -> {df['job_title'].nunique():,}",
            cells_changed=n_changed)

    # 3. employer_country
    st = df["employer_state"].astype("str").str.lower()
    us_state = st.isin(REGION)
    ctry = df["employer_country"].astype("str")
    fix = us_state & (df["employer_country"].isna() | ctry.str.lower().isin(REGION))
    df["employer_country"] = ctry.where(~fix, USA).astype("category")
    log.add("3. fill / fix employer_country", df,
            "blank country (15.6% of FY2017 rows) or a US state given as the country ('Georgia') with a US "
            "employer state -> United States Of America", cells_changed=fix.sum())

    # 4. sentinel / unusable wages
    w = df["prevailing_wage"].fillna(df["wage_offered_fy2019"])
    bad = w.notna() & ((w <= 0) | df["unit_of_pay"].isna())
    for c in ["prevailing_wage", "wage_offered_fy2019"]:
        df[c] = df[c].mask(bad)
    log.add("4. wage = 0 or no unit -> NaN", df,
            "a $0 prevailing wage is impossible, and a wage without a unit can't be annualized; these are "
            "mostly DENIED filings with incomplete forms, so the row (and its decision) is kept",
            cells_changed=bad.sum())

    # 5. mislabelled pay units
    w = df["prevailing_wage"].fillna(df["wage_offered_fy2019"])
    mult = df["unit_of_pay"].astype("str").map(PAY_MULT).astype(float)
    relabel = ((df["unit_of_pay"] != "Year") & (w * mult > MAX_ANNUAL)
               & w.between(MIN_ANNUAL_FIGURE, MAX_ANNUAL)).fillna(False)
    by_unit = df.loc[relabel, "unit_of_pay"].value_counts()
    df["unit_relabelled"] = relabel.astype("int8")
    df["unit_of_pay"] = df["unit_of_pay"].astype("str").where(~relabel, "Year").astype("category")
    log.add("5. fix mislabelled pay units -> Year", df,
            f"value x unit > ${MAX_ANNUAL:,}/year but the value itself is a normal annual salary "
            f"(${MIN_ANNUAL_FIGURE:,}-${MAX_ANNUAL:,}), e.g. 'Hour' = 46,301 for a programmer: the salary "
            f"was typed with the wrong unit. Relabelled: {by_unit[by_unit > 0].to_dict()}",
            cells_changed=relabel.sum())

    # 6. annualize
    mult = df["unit_of_pay"].astype("str").map(PAY_MULT).astype(float)
    df["annual_pw"] = df["prevailing_wage"] * mult
    df["annual_offered_fy2019"] = df["wage_offered_fy2019"] * mult
    log.add("6. annualize wages", df,
            "wages are quoted per hour/week/bi-week/month/year; x2080 / 52 / 26 / 12 puts them on one "
            "scale (2,080 h = DOL's full-time year)", cells_changed=df["annual_pw"].notna().sum()
            + df["annual_offered_fy2019"].notna().sum())

    # 7. harmonize SOC titles
    if fitting:
        params["soc"] = fit_soc_map(df["soc_title"])
    df["soc_harmonized"] = apply_soc_map(df["soc_title"], params["soc"])
    changed = (df["soc_harmonized"].astype("str") != df["soc_title"].astype("str")).sum()
    log.add("7. harmonize SOC titles", df,
            f"{df['soc_title'].nunique():,} spellings -> {df['soc_harmonized'].nunique():,} titles: same title "
            "with different case/punctuation or a leading SOC code ('15-1121.00Computersystemsanalysts'), "
            f"and rare (<{SOC_MIN_ROWS} rows) misspellings fuzzy-matched to a common title (similarity "
            f">= {SOC_FUZZY_CUTOFF}). Titles only in the 2018 SOC (e.g. 'Data Scientists', FY2022, ~0.1% of "
            "rows) are kept as their own level; raw soc_title kept", cells_changed=changed)

    # 8. IQR outliers on log annual wage within SOC title -> NaN + flag
    if fitting:
        params["iqr"] = {}
    counts = {}
    for col, flag, lcol in [("annual_pw", "pw_outlier", "log_pw"),
                            ("annual_offered_fy2019", "offered_outlier", "log_offered_fy2019")]:
        lw = np.log(df[col])
        if fitting:
            params["iqr"][col] = fit_iqr(lw, df["soc_harmonized"])
        out = iqr_outlier_mask(lw, df["soc_harmonized"], params["iqr"][col])
        df[flag] = out.astype("int8")
        df[col] = df[col].mask(out)
        df[lcol] = np.log(df[col])
        counts[col] = int(out.sum())
    log.add("8. IQR outliers in log annual wage (within SOC) -> NaN + flag", df,
            "log scale: wages are right-skewed; within SOC title: $200k is normal for a surgeon, not for a "
            "cook. Outlying wage set to NaN (not the row) because the case decision is still valid. "
            f"Flagged: {counts}", cells_changed=sum(counts.values()))

    # 9. worksite gap (FY2019): flag, don't impute
    df["worksite_missing"] = df["worksite_state"].isna().astype("int8")
    log.add("9. flag missing worksite (not imputed)", df,
            "88% of FY2019 rows have no worksite in the source file. Not imputed from employer location: "
            "the two states agree in only ~52% of the rows that have both, and the gap isn't random "
            "(blank rows are 93% CERTIFIED vs 61% for the others). Worksite analyses exclude FY2019")

    # 10. derived features
    ws = df["worksite_state"].astype("str").str.lower()
    df["worksite_region"] = ws.map(REGION).where(df["worksite_state"].notna()).astype("category")
    df["same_state"] = (ws == df["employer_state"].astype("str").str.lower()).where(
        df["worksite_state"].notna()).astype("Int8")
    # filings per employer per year, counted within this set: the split is grouped by employer,
    # so every filing of an employer is in the same set and the count is exact in both sets
    vol = df.groupby(["employer_key", "fiscal_year"], observed=True)["fiscal_year"].transform("size")
    df["employer_volume"] = vol.astype("Int32")
    df["log_employer_volume"] = np.log(vol)
    for flag, pat in SENIORITY.items():
        df[flag] = per_category(df["job_title"], lambda c, p=pat: c.str.contains(p, regex=True)).fillna(False).astype("int8")
    if fitting:
        params["soc_top"] = df["soc_harmonized"].value_counts().head(SOC_TOP_K).index.astype("str").tolist()
    soc = df["soc_harmonized"].astype("str")
    df["soc_group"] = soc.where(soc.isin(params["soc_top"]), "Other").astype("category")
    for s in ["CERTIFIED", "DENIED", "WITHDRAWN", "CERTIFIED-WITHDRAWN"]:
        df[s.lower().replace("-", "_")] = (df["case_status"] == s).astype("int8")
    log.add("10. derived features", df,
            "annual & log wages, worksite_region (Census), same_state (worksite = employer state), employer "
            "filing volume per year, job-title seniority flags (senior/lead/manager/junior), status dummies, "
            f"soc_group = the {SOC_TOP_K} most common SOC titles (fitted on exploration) + 'Other', used for "
            "encoding; soc_harmonized (all titles) is kept for within-occupation comparisons")
    return df, log.frame(), params


def encoded_shape(df):
    n_onehot = int(sum(df[c].nunique(dropna=False) for c in ONE_HOT_COLS))
    return len(df), n_onehot + len(NUMERIC_COLS), n_onehot, len(NUMERIC_COLS)


def main():
    raw = pd.read_parquet(EXPLORE_PATH)
    df, log, params = clean(raw)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "explore_clean.parquet", index=False)
    (DATA_PROCESSED / "clean_params.json").write_text(json.dumps(params, indent=1, default=float))

    rows, feats, n_oh, n_num = encoded_shape(df)
    pw_before = raw["prevailing_wage"].notna().sum()
    pw_after = df["annual_pw"].notna().sum()
    by_year = pd.DataFrame({
        "rows": df.groupby("fiscal_year").size(),
        "usable prevailing wage": df.groupby("fiscal_year")["annual_pw"].count(),
        "usable offered wage": df.groupby("fiscal_year")["annual_offered_fy2019"].count(),
        "worksite known": df.groupby("fiscal_year")["worksite_state"].count(),
    })
    summary = (
        f"\n## Size after encoding\n{rows:,} rows × {feats:,} features "
        f"({n_oh:,} one-hot levels of {', '.join(ONE_HOT_COLS)} + {n_num} numeric/flags) = "
        f"**{rows * feats:,}** cells (floor 3,000,000: {'PASS' if rows * feats >= 3_000_000 else 'FAIL'}). "
        f"SOC titles are encoded as soc_group: the {SOC_TOP_K} most common "
        f"({100 * (df['soc_group'] != 'Other').mean():.1f}% of rows) + 'Other'; the full 1,123-title version would "
        "give ~1,280 features, almost all for titles with a handful of rows. "
        "employer_key and job_title are not one-hot encoded (132k / 400k+ levels, almost all tiny); they "
        "enter through employer_volume, n_identical and the seniority flags.\n"
        f"\n## What was lost\nNo rows are removed. Prevailing wages usable: {pw_before:,} raw -> {pw_after:,} "
        f"after cleaning ({pw_before - pw_after:,} set to NaN, {100 * (pw_before - pw_after) / pw_before:.2f}%).\n\n"
        + by_year.to_markdown() + "\n"
    )

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "cleaning_log.md").write_text(
        "# Cleaning log (exploration set)\n\n" + log.to_markdown(index=False) + "\n" + summary)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(log.drop(columns="reason").to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
