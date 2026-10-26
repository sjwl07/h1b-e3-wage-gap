"""Step 3: data dictionary, computed on the exploration set only.

Type / unit / description are hand-written from the DOL LCA disclosure file
layout; range, cardinality and missingness are measured. Missingness is shown
overall and per fiscal year, because several gaps are structural (a column
that a year's file simply doesn't have) rather than random.
"""
import pandas as pd

from config import EXPLORE_PATH, REPORT, YEARS

WAGE_COLS = ["prevailing_wage", "wage_offered_fy2019"]

# column -> (type, role, unit, description)
META = {
    "fiscal_year": ("discrete (ordinal)", "feature", "year", "US federal fiscal year of the decision (FY2020 = Oct 2019 - Sep 2020); added from the file name"),
    "quarter": ("ordinal", "feature", "Q1-Q4", "Fiscal quarter; only in the FY2020-22 files"),
    "visa_class": ("nominal", "feature", "-", "H-1B, E-3 Australian, H-1B1 Chile, H-1B1 Singapore"),
    "case_status": ("nominal", "target", "-", "DOL decision: CERTIFIED, CERTIFIED-WITHDRAWN (withdrawn after approval), WITHDRAWN (before decision), DENIED"),
    "employer_name": ("nominal", "identifier", "-", "Employer's legal name as typed on the form (free text)"),
    "employer_key": ("nominal", "identifier", "-", "Derived: normalized employer name; used for grouping (split, clustered SEs), not one-hot encoded"),
    "employer_city": ("nominal", "feature", "-", "Derived from Employer_Location ('City, State')"),
    "employer_state": ("nominal", "feature", "-", "Derived from Employer_Location"),
    "employer_country": ("nominal", "feature", "-", "Employer's country; almost always USA"),
    "soc_title": ("nominal", "feature", "-", "Standard Occupational Classification title used to set the prevailing wage"),
    "job_title": ("nominal", "feature", "-", "Employer's own job title (free text)"),
    "full_time": ("nominal (binary)", "feature", "Y/N", "Full-time position"),
    "worksite_city": ("nominal", "feature", "-", "Derived from Worksite ('City, State'): where the worker will work"),
    "worksite_state": ("nominal", "feature", "-", "Derived from Worksite; 50 states + DC + territories"),
    "prevailing_wage": ("continuous", "target", "USD per unit_of_pay", "Minimum wage DOL requires for this occupation and area; FY2019 file doesn't have it"),
    "wage_offered_fy2019": ("continuous", "feature", "USD per unit_of_pay", "Wage the employer offers; only in the FY2019 file (see fy2019_wage_check.md)"),
    "unit_of_pay": ("nominal", "feature", "-", "Year, Hour, Month, Week, Bi-Weekly: the unit the wage is quoted in"),
}


def describe_range(s, kind):
    s = s.dropna()
    if s.empty:
        return "all missing"
    if kind.startswith("continuous") or kind.startswith("discrete"):
        return f"{s.min():g} – {s.max():g}"
    vc = s.value_counts()
    vc = vc[vc > 0]  # categorical columns list categories that only occur in the confirmation set
    return f"{len(vc)} levels; top: {', '.join(map(str, vc.index[:3]))}"


def main():
    df = pd.read_parquet(EXPLORE_PATH)
    n = len(df)
    assert set(META) == set(df.columns), set(df.columns) ^ set(META)
    miss_by_year = df.drop(columns="fiscal_year").isna().groupby(df["fiscal_year"]).mean().mul(100)

    rows = []
    for col, (kind, role, unit, desc) in META.items():
        s = df[col]
        n_na = int(s.isna().sum())
        n_zero = int((s == 0).sum()) if col in WAGE_COLS else 0
        by_year = "" if col == "fiscal_year" else " · ".join(
            f"{str(y)[2:]}:{miss_by_year.loc[y, col]:.0f}" for y in YEARS)
        rows.append({
            "attribute": col,
            "type": kind,
            "role": role,
            "unit": unit,
            "range / levels": describe_range(s, kind),
            "n_unique": int(s.nunique()),
            "% blank": round(100 * n_na / n, 2),
            "% sentinel zero": round(100 * n_zero / n, 3),
            "% blank by FY (17…22)": by_year,
            "description": desc,
        })
    dd = pd.DataFrame(rows)

    REPORT.mkdir(parents=True, exist_ok=True)
    md = (
        f"# Data dictionary\n\nComputed on the exploration set ({n:,} rows). "
        "'Sentinel zero' = a wage of 0, which can't be a real prevailing wage. "
        "The per-year column shows structural gaps: `quarter` is absent before FY2020, FY2019 has no "
        "prevailing wage (it has the offered wage instead), and FY2019's worksite is blank in most rows.\n\n"
        + dd.to_markdown(index=False) + "\n"
    )
    (REPORT / "data_dictionary.md").write_text(md)
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(dd.drop(columns="description").to_string(index=False))


if __name__ == "__main__":
    main()
