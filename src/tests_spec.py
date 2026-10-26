"""The four pre-registered tests, defined once and shared by power.py and inference.py,
so the power check and the confirmatory run use exactly the same populations and models.
The comparison cell and employer-size bins are also used by eda.py."""
import numpy as np
import pandas as pd

MID_WAGE = np.log(1.05)  # 5% in prevailing wage
MID_DENIAL = 0.005  # 0.5 percentage points
ALPHA_FAMILY = 0.05
WAGE_YEARS = [2017, 2018, 2020, 2021, 2022]  # FY2019 has no prevailing wage
VOLUME_BINS = [0, 1, 10, 100, 1000, np.inf]
VOLUME_LABELS = ["1", "2-10", "11-100", "101-1000", ">1000"]
REL_CELL = ["soc_harmonized", "worksite_state", "fiscal_year"]
MIN_CELL = 5  # relative wage only where the comparison cell has >= 5 wages

COLS = ["fiscal_year", "visa_class", "employer_key", "soc_harmonized", "worksite_state", "log_pw",
        "employer_volume", "denied"]


def relative_pw(df):
    """log PW minus the median log PW of the same SOC title x state x year (the EDA wage-level proxy)."""
    g = df.groupby(REL_CELL, observed=True)["log_pw"]
    return (df["log_pw"] - g.transform("median")).where(g.transform("count") >= MIN_CELL)


def volume_bin(df):
    return pd.cut(df["employer_volume"].astype(float), VOLUME_BINS, labels=VOLUME_LABELS)


def prepare(df):
    """E-3 vs H-1B only; H-1B1 (0.4% of filings) is too small and a different program."""
    d = df[df["visa_class"].isin(["H-1B", "E-3 Australian"])].copy()
    d["e3"] = (d["visa_class"] == "E-3 Australian").astype("int8")
    d["employer_key"] = d["employer_key"].astype("str").fillna("<MISSING>")
    d["volume_bin"] = volume_bin(d)
    d["denied"] = d["denied"].astype(float)
    return d


def wage_pop(d):
    return d[d["fiscal_year"].isin(WAGE_YEARS) & d["log_pw"].notna() & d["worksite_state"].notna()]


TESTS = {
    "T1": {"label": "E-3 vs H-1B prevailing wage, same SOC × state × year",
           "pop": wage_pop, "y": "log_pw", "cells": REL_CELL,
           "mid": MID_WAGE, "unit": "log"},
    "T2": {"label": "E-3 vs H-1B prevailing wage, same employer × SOC × state × year",
           "pop": wage_pop, "y": "log_pw",
           "cells": ["employer_key", "soc_harmonized", "worksite_state", "fiscal_year"],
           "mid": MID_WAGE, "unit": "log"},
    "T3": {"label": "E-3 vs H-1B denial rate, same employer-size bin × year",
           "pop": lambda d: d, "y": "denied", "cells": ["volume_bin", "fiscal_year"],
           "mid": MID_DENIAL, "unit": "pp"},
    "T4": {"label": "E-3 vs H-1B denial rate, same employer × year",
           "pop": lambda d: d, "y": "denied", "cells": ["employer_key", "fiscal_year"],
           "mid": MID_DENIAL, "unit": "pp"},
}
