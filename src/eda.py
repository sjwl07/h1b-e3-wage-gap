"""Step 5: exploratory data analysis on the cleaned exploration set.

Writes figures to figures/ and the captioned write-up to report/eda.md.
The confirmation set is never loaded here.

Wage comparisons use the *relative* prevailing wage: log PW minus the median
log PW of the same SOC title x worksite state x fiscal year. DOL sets the
prevailing wage by occupation, area and wage level, so once occupation, state
and year are fixed, what's left mostly reflects the wage level (I-IV) the
employer chose, which this Kaggle extract doesn't contain directly.
"""
import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

from config import DATA_PROCESSED, EXPLORE_PATH, FIGURES, REPORT

CELL = ["soc_harmonized", "worksite_state", "fiscal_year"]
MIN_CELL = 5  # relative wage only where the comparison cell has >= 5 wages
VOLUME_BINS = [0, 1, 10, 100, 1000, np.inf]
VOLUME_LABELS = ["1", "2-10", "11-100", "101-1000", ">1000"]
STATUSES = ["CERTIFIED", "CERTIFIED-WITHDRAWN", "WITHDRAWN", "DENIED"]
VISAS = ["H-1B", "E-3 Australian", "H-1B1 Chile", "H-1B1 Singapore"]
SAMPLE_N = 300_000  # Cramér's V on a random sample; chi² on 3M rows is slow and V is already stable
COLS = ["fiscal_year", "quarter", "visa_class", "case_status", "employer_key", "employer_state", "full_time",
        "worksite_state", "worksite_region", "soc_harmonized", "soc_group", "unit_of_pay", "annual_pw",
        "log_pw", "annual_offered_fy2019", "n_identical", "employer_volume", "same_state", "is_senior",
        "is_lead", "is_manager", "is_junior", "certified", "denied", "withdrawn", "certified_withdrawn"]

sns.set_theme(style="whitegrid", context="notebook")
STATS = {}


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=130)
    plt.close(fig)


def money(x):
    return f"${x / 1e3:,.0f}k"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def load():
    df = pd.read_parquet(DATA_PROCESSED / "explore_clean.parquet", columns=COLS)
    g = df.groupby(CELL, observed=True)["log_pw"]
    n_cell = g.transform("count")
    df["rel_pw"] = (df["log_pw"] - g.transform("median")).where(n_cell >= MIN_CELL)
    df["volume_bin"] = pd.cut(df["employer_volume"].astype(float), VOLUME_BINS, labels=VOLUME_LABELS)
    df["visa_class"] = pd.Categorical(df["visa_class"].astype(str), VISAS)
    df["case_status"] = pd.Categorical(df["case_status"].astype(str), STATUSES)
    return df


def rates(df, by):
    t = df.groupby(by, observed=True)[["denied", "withdrawn", "certified_withdrawn"]].mean()
    t["n"] = df.groupby(by, observed=True).size()
    return t


def fig_wage_distribution(df):
    w = df["annual_pw"].dropna()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(w / 1e3, bins=120, color="tab:blue")
    ax[0].set(xlabel="Annual prevailing wage ($k)", ylabel="Filings", title="Raw")
    ax[1].hist(np.log(w), bins=100, color="tab:green")
    ax[1].set(xlabel="log(annual prevailing wage)", title="Log")
    save(fig, "fig01_wage_distribution")
    STATS["wage"] = {"n": int(len(w)), "median": float(w.median()), "mean": float(w.mean()),
                     "skew_raw": round(float(w.skew()), 2), "skew_log": round(float(np.log(w).skew()), 2),
                     "p01": float(w.quantile(0.01)), "p99": float(w.quantile(0.99)),
                     "min": float(w.min()), "max": float(w.max())}


def fig_year(df):
    by = df.groupby("fiscal_year")
    n = by.size()
    med = by["annual_pw"].median()
    off = by["annual_offered_fy2019"].median()
    top5 = df["soc_harmonized"].value_counts().head(3).index
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(n.index, n.values / 1e3, color="tab:blue")
    ax[0].set(xlabel="Fiscal year", ylabel="Filings (thousands)", title="Filings per year")
    ax[1].plot(med.index, med.values / 1e3, "o-", label="all occupations (prevailing)")
    for s in top5:
        m = df[df["soc_harmonized"] == s].groupby("fiscal_year")["annual_pw"].median()
        ax[1].plot(m.index, m.values / 1e3, "o--", alpha=0.7, label=s[:32])
    ax[1].plot([2019], [off[2019] / 1e3], "s", mfc="none", ms=10, color="k", label="FY2019 offered wage")
    ax[1].set(xlabel="Fiscal year", ylabel="Median annual wage ($k)",
              title="Median prevailing wage (FY2019 has none)")
    ax[1].legend(fontsize=7)
    save(fig, "fig02_year")
    STATS["year"] = {"n": n.to_dict(), "median_pw": med.dropna().to_dict(), "median_offered_2019": float(off[2019]),
                     "growth_2017_2022_pct": round(100 * (med[2022] / med[2017] - 1), 1)}


def fig_state(df):
    d = df.dropna(subset=["worksite_state"])
    top = d["worksite_state"].value_counts().head(15)
    med = d[d["worksite_state"].isin(top.index)].groupby("worksite_state", observed=True)["annual_pw"].median()
    med = med.reindex(top.index)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].barh(top.index[::-1].astype(str), med.values[::-1] / 1e3, color="tab:blue")
    for i, (s, v) in enumerate(zip(top.index[::-1], top.values[::-1])):
        ax[0].text(1, i, f"{v / 1e3:.0f}k filings", va="center", fontsize=7, color="white")
    ax[0].set(xlabel="Median annual prevailing wage ($k)", title="15 largest worksite states")
    regions = ["West", "Northeast", "South", "Midwest", "Territory"]
    sns.boxplot(data=d, x="worksite_region", y="log_pw", order=regions, showfliers=False, ax=ax[1])
    ax[1].set(xlabel="", ylabel="log(annual prevailing wage)", title="By Census region")
    save(fig, "fig03_state")
    reg = d.groupby("worksite_region", observed=True)["annual_pw"].median()
    STATS["state"] = {"top15_share": round(float(top.sum() / len(d)), 3),
                      "top3": {str(k): int(v) for k, v in top.head(3).items()},
                      "median_pw_top15": {str(k): float(v) for k, v in med.items()},
                      "region_median": {str(k): float(v) for k, v in reg.items()}}


def fig_soc(df):
    vc = df["soc_harmonized"].value_counts()
    vc = vc[vc > 0]
    top = vc.head(12).index
    d = df[df["soc_harmonized"].isin(top)].copy()
    d["soc"] = pd.Categorical(d["soc_harmonized"].astype(str), [str(t) for t in top])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    ax[0].barh([t[:38] for t in top[::-1].astype(str)], 100 * vc.head(12).values[::-1] / vc.sum(), color="tab:blue")
    ax[0].set(xlabel="% of filings", title="12 most common SOC titles")
    sns.boxplot(data=d, y="soc", x="annual_pw", showfliers=False, ax=ax[1], orient="h")
    ax[1].set(ylabel="", xlabel="Annual prevailing wage ($)", title="Wage spread within each title")
    ax[1].set_yticks(range(12), [t[:38] for t in top.astype(str)])
    save(fig, "fig04_soc")
    med = d.groupby("soc", observed=True)["annual_pw"].median()
    iqr = d.groupby("soc", observed=True)["annual_pw"].quantile([0.25, 0.75]).unstack()
    computer = vc[vc.index.astype(str).str.contains("Computer|Software|Database|Network|Information Security")]
    STATS["soc"] = {"n_titles": int(len(vc)), "top1": str(top[0]), "top1_share": round(float(vc.iloc[0] / vc.sum()), 3),
                    "top5_share": round(float(vc.head(5).sum() / vc.sum()), 3),
                    "computer_share": round(float(computer.sum() / vc.sum()), 3),
                    "median_pw": {str(k): float(v) for k, v in med.items()},
                    "iqr_ratio_median": round(float((iqr[0.75] / iqr[0.25]).median()), 2)}


def fig_visa(df):
    share = df["visa_class"].value_counts(normalize=True).reindex(VISAS)
    d = df.dropna(subset=["rel_pw"])
    by_bin = d.groupby(["volume_bin", "visa_class"], observed=True)["rel_pw"].mean().unstack()[["H-1B", "E-3 Australian"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    m = d.groupby("visa_class", observed=True)["rel_pw"].agg(["mean", "sem", "count"]).reindex(VISAS)
    ax[0].bar(VISAS, 100 * m["mean"], yerr=196 * m["sem"], color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    for i, v in enumerate(VISAS):
        ax[0].text(i, 0.3, f"{share[v]:.1%}\nof filings", ha="center", fontsize=8)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(ylabel="Mean relative PW (%, ≈ log points ×100)",
              title="Prevailing wage vs same SOC × state × year (95% CI)")
    (100 * by_bin).plot(kind="bar", ax=ax[1], color=["tab:blue", "tab:orange"], rot=0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="Employer filings per year", ylabel="Mean relative PW (%)",
              title="E-3 premium holds within employer size")
    save(fig, "fig05_visa")
    STATS["visa"] = {"share": share.round(4).to_dict(), "rel_mean": m["mean"].round(4).to_dict(),
                     "rel_median": d.groupby("visa_class", observed=True)["rel_pw"].median().round(4).to_dict(),
                     "n": m["count"].astype(int).to_dict(),
                     "rel_by_volume_bin": {str(k): v for k, v in by_bin.round(4).to_dict("index").items()}}


def fig_status(df):
    yr = pd.crosstab(df["fiscal_year"], df["case_status"], normalize="index")[STATUSES]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    (100 * yr[STATUSES[1:]]).plot(kind="bar", stacked=True, ax=ax[0], rot=0,
                                  color=["tab:orange", "tab:gray", "tab:red"])
    ax[0].set(xlabel="Fiscal year", ylabel="% of filings (rest = CERTIFIED)", title="Non-certified outcomes by year")
    groups = {"H-1B": df["visa_class"] == "H-1B", "E-3": df["visa_class"] == "E-3 Australian",
              "H-1B1": df["visa_class"].astype(str).str.startswith("H-1B1"),
              "full-time": df["full_time"] == "Y", "part-time": df["full_time"] == "N",
              "worksite = employer state": df["same_state"] == 1,
              "worksite ≠ employer state": df["same_state"] == 0}
    den = pd.Series({k: df.loc[m, "denied"].mean() for k, m in groups.items()})
    ax[1].barh(den.index[::-1], 100 * den.values[::-1], color="tab:red")
    ax[1].set(xlabel="% denied", title="Denial rate by group")
    save(fig, "fig06_status")
    STATS["status"] = {"overall": df["case_status"].value_counts(normalize=True).round(4).to_dict(),
                       "by_year": {int(k): v for k, v in yr.round(4).to_dict("index").items()},
                       "denied_by_group": den.round(4).to_dict()}


def fig_employer_volume(df):
    r = rates(df, "volume_bin")
    ey = df.groupby(["employer_key", "fiscal_year"], observed=True)["denied"].agg(["sum", "size"])
    small = ey[ey["size"].between(2, 10)]
    with_denial = small[small["sum"] > 0]
    visa_bin = df.groupby(["volume_bin", "visa_class"], observed=True)["denied"].mean().unstack()[["H-1B", "E-3 Australian"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    (100 * r[["denied", "withdrawn", "certified_withdrawn"]]).plot(kind="bar", ax=ax[0], rot=0,
                                                                    color=["tab:red", "tab:gray", "tab:orange"])
    ax[0].set(xlabel="Employer filings per year", ylabel="% of filings", title="Outcome rates by employer size")
    (100 * visa_bin).plot(kind="bar", ax=ax[1], rot=0, color=["tab:blue", "tab:orange"])
    ax[1].set(xlabel="Employer filings per year", ylabel="% denied", title="Denial rate, H-1B vs E-3, within size")
    save(fig, "fig07_employer_volume")
    STATS["volume"] = {
        "rates": {str(k): v for k, v in r.round(4).to_dict("index").items()},
        "denied_by_bin_visa": {str(k): v for k, v in visa_bin.round(4).to_dict("index").items()},
        "share_small_employers_by_visa": (df["employer_volume"] <= 10).groupby(df["visa_class"], observed=True)
        .mean().round(3).to_dict(),
        "small_employer_years_with_denial": round(float(len(with_denial) / len(small)), 3),
        "denied_share_in_those": round(float((with_denial["sum"] / with_denial["size"]).mean()), 3),
    }


def fig_top_filers(df):
    d = df.dropna(subset=["rel_pw"])
    top = df["employer_key"].value_counts().head(15).index
    m = d[d["employer_key"].isin(top)].groupby("employer_key", observed=True)["rel_pw"].mean().sort_values()
    share = df["employer_key"].isin(top).mean()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(m.index.astype(str), 100 * m.values, color=np.where(m.values < 0, "tab:red", "tab:blue"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Mean relative prevailing wage (%)", title="15 largest filers vs same SOC × state × year")
    save(fig, "fig08_top_filers")
    STATS["top_filers"] = {"share_rows": round(float(share), 3), "rel_mean": m.round(4).to_dict()}


def fig_seniority(df):
    d = df.dropna(subset=["rel_pw"])
    flags = {"senior / sr": "is_senior", "lead / principal / staff / architect": "is_lead",
             "manager / director / head / VP": "is_manager", "junior / entry / intern / trainee": "is_junior"}
    rows = []
    for label, f in flags.items():
        g = d.groupby(f)["rel_pw"].agg(["mean", "sem", "count"])
        rows.append({"label": label, "diff": g.loc[1, "mean"] - g.loc[0, "mean"],
                     "se": float(np.hypot(g.loc[1, "sem"], g.loc[0, "sem"])), "n": int(g.loc[1, "count"]),
                     "share": float(df[f].mean())})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(t["label"], 100 * t["diff"], xerr=196 * t["se"], color=np.where(t["diff"] < 0, "tab:red", "tab:blue"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Relative PW, titles with keyword minus without (%)",
           title="Job-title keywords vs prevailing wage (95% CI)")
    save(fig, "fig09_seniority")
    STATS["seniority"] = t.set_index("label").round(4).to_dict("index")


def fig_missingness():
    raw = pd.read_parquet(EXPLORE_PATH)
    cols = ["quarter", "employer_country", "worksite_state", "prevailing_wage", "wage_offered_fy2019", "unit_of_pay"]
    t = raw[cols].isna().groupby(raw["fiscal_year"]).mean().mul(100).T
    del raw
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.heatmap(t, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "% blank"})
    ax.set(xlabel="Fiscal year", ylabel="", title="Blank cells by year in the source files (exploration set)")
    save(fig, "fig10_missingness")
    STATS["missingness"] = {c: {int(y): round(v, 2) for y, v in t.loc[c].items()} for c in cols}


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))  # Bergsma bias correction
    r_c, k_c = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(r_c - 1, k_c - 1))))


def fig_associations(df):
    s = df.dropna(subset=["worksite_state"]).sample(SAMPLE_N, random_state=0)
    cats = ["visa_class", "case_status", "full_time", "unit_of_pay", "fiscal_year", "worksite_region",
            "worksite_state", "employer_state", "soc_group"]
    cv = pd.DataFrame(1.0, index=cats, columns=cats)
    for a, b in combinations(cats, 2):
        cv.loc[a, b] = cv.loc[b, a] = cramers_v(s[a].astype(str), s[b].astype(str))
    num = ["log_pw", "rel_pw", "employer_volume", "n_identical", "same_state", "is_senior", "is_junior"]
    s["employer_volume"] = np.log(s["employer_volume"].astype(float))
    s["n_identical"] = np.log(s["n_identical"])
    corr = s[num].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    sns.heatmap(cv, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax[0])
    ax[0].set_title(f"Cramér's V, categorical (random {SAMPLE_N:,} rows)")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=ax[1])
    ax[1].set_title("Spearman ρ, numeric / flags")
    save(fig, "fig11_associations")
    STATS["associations"] = {"cramers_v": {f"{a} ~ {b}": round(cv.loc[a, b], 3) for a, b in combinations(cats, 2)},
                             "spearman": {f"{a} ~ {b}": round(float(corr.loc[a, b]), 3)
                                          for a, b in combinations(num, 2)}}


def fig_duplicates(df):
    dup = df["n_identical"] > 1
    by_bin = dup.groupby(df["volume_bin"], observed=True).mean()
    by_status = dup.groupby(df["case_status"], observed=True).mean().reindex(STATUSES)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(by_bin.index.astype(str), 100 * by_bin.values, color="tab:purple")
    ax[0].set(xlabel="Employer filings per year", ylabel="% rows with an identical twin",
              title="Identical rows concentrate in big filers")
    ax[1].bar([s.replace("CERTIFIED-", "CERT-") for s in STATUSES], 100 * by_status.values, color="tab:purple")
    ax[1].set(ylabel="% rows with an identical twin", title="…and differ by outcome")
    save(fig, "fig12_duplicates")
    STATS["duplicates"] = {"share": round(float(dup.mean()), 3), "by_volume_bin": by_bin.round(3).to_dict(),
                           "by_status": by_status.round(3).to_dict(), "max_group": int(df["n_identical"].max())}


def write_report():
    s = STATS
    w, yr, st, soc, v, stat, vol, tf, sen = (s["wage"], s["year"], s["state"], s["soc"], s["visa"], s["status"],
                                             s["volume"], s["top_filers"], s["seniority"])
    cv, sp = s["associations"]["cramers_v"], s["associations"]["spearman"]
    reg = st["region_median"]
    by = stat["by_year"]
    dg = stat["denied_by_group"]
    vb = vol["denied_by_bin_visa"]
    rb = v["rel_by_volume_bin"]
    neg = [k for k, x in tf["rel_mean"].items() if x < 0]
    pos = sorted(tf["rel_mean"].items(), key=lambda kv: -kv[1])[:2]
    miss = s["missingness"]
    dup = s["duplicates"]
    e3_rel_big = [rb[b]["E-3 Australian"] for b in ["11-100", "101-1000", ">1000"]]
    h1_rel_big = [rb[b]["H-1B"] for b in ["11-100", "101-1000", ">1000"]]

    figs = [
        ("fig01_wage_distribution",
         f"**The prevailing wage is moderately right-skewed; log wage is almost exactly symmetric.** Annual PW has "
         f"skew {w['skew_raw']} (median {money(w['median'])}, mean {money(w['mean'])}); log PW has skew "
         f"{w['skew_log']}. After cleaning, the range is {money(w['min'])} (exactly $7.25 × 2,080 h, the federal "
         f"minimum wage) to {money(w['max'])} (physicians). Means, t-tests and linear models on log PW are reasonable."),
        ("fig02_year",
         f"**Filings and wages both rose over the period, and FY2019 can't be compared on prevailing wage.** "
         f"The median PW grew {yr['growth_2017_2022_pct']}% from FY2017 ({money(yr['median_pw'][2017])}) to FY2022 "
         f"({money(yr['median_pw'][2022])}); FY2021 had the most filings ({yr['n'][2021]:,} in this set). FY2019 "
         f"has only the offered wage (median {money(yr['median_offered_2019'])}), which isn't the same quantity, "
         f"so it's shown separately. Any wage comparison has to hold the year fixed."),
        ("fig03_state",
         f"**Location shifts the wage a lot, so comparisons must be within state.** The 15 largest worksite states "
         f"hold {pct(st['top15_share'], 0)} of filings, led by {', '.join(st['top3'])}. Median PW is highest in "
         f"the West ({money(reg['West'])}) and lowest in the Midwest ({money(reg['Midwest'])}) and the South "
         f"({money(reg['South'])}). West vs Midwest is a {100 * (reg['West'] / reg['Midwest'] - 1):.0f}% gap, about as "
         f"large as the whole FY2017–22 wage growth. (Territories are lower, {money(reg['Territory'])}, but small.)"),
        ("fig04_soc",
         f"**This is mostly a computer-jobs dataset.** {soc['n_titles']:,} SOC titles appear, but "
         f"'{soc['top1']}' alone is {pct(soc['top1_share'])} of filings, the top 5 are {pct(soc['top5_share'])}, and "
         f"computer and software titles together are {pct(soc['computer_share'])}. Within a title, the 75th "
         f"percentile wage is typically {soc['iqr_ratio_median']}× the 25th (wage level and metro area), so "
         f"occupation has to be fixed before comparing anything else."),
        ("fig05_visa",
         f"**E-3 (Australian) filings carry a higher prevailing wage than H-1B filings for the same job, place and "
         f"year.** Relative to the median for their SOC × state × year, E-3 filings are +{100 * v['rel_mean']['E-3 Australian']:.1f}% "
         f"on average (median +{100 * v['rel_median']['E-3 Australian']:.1f}%), H-1B +{100 * v['rel_mean']['H-1B']:.1f}%. "
         f"This is not explained by employer size: in every size group above 10 filings a year, E-3 is about "
         f"+{100 * np.mean(e3_rel_big):.0f}% against +{100 * min(h1_rel_big):.0f} to "
         f"+{100 * max(h1_rel_big):.0f}% for H-1B. Since PW is set by DOL from occupation, area and wage level, the "
         f"likely source is that employers file Australians at higher wage levels. **Candidate problem statement.**"),
        ("fig06_status",
         f"**Denials are rare and falling, and they are concentrated in a few groups.** Overall "
         f"{pct(stat['overall']['DENIED'], 2)} of filings are denied; the rate fell from {pct(by[2017]['DENIED'], 2)} "
         f"in FY2017 to {pct(by[2022]['DENIED'], 2)} in FY2022. CERTIFIED-WITHDRAWN dropped sharply in FY2020 "
         f"({pct(by[2019]['CERTIFIED-WITHDRAWN'])} → {pct(by[2020]['CERTIFIED-WITHDRAWN'])}). E-3 filings are denied "
         f"{dg['E-3'] / dg['H-1B']:.1f}× as often as H-1B ({pct(dg['E-3'], 2)} vs {pct(dg['H-1B'], 2)}), part-time "
         f"{dg['part-time'] / dg['full-time']:.1f}× as often as full-time, and filings for the employer's own state "
         f"{dg['worksite = employer state'] / dg['worksite ≠ employer state']:.1f}× as often as out-of-state ones."),
        ("fig07_employer_volume",
         f"**Denial rates peak at small repeat filers, and part of the E-3 denial gap is employer mix.** Employers "
         f"with 2–10 filings a year have a {pct(vol['rates']['2-10']['denied'])} denial rate, against "
         f"{pct(vol['rates']['1']['denied'])} for one-time filers and {pct(vol['rates']['>1000']['denied'], 2)} for "
         f"the largest. Among 2–10-filing employer-years with any denial, {pct(vol['denied_share_in_those'], 0)} of "
         f"their filings are denied, which looks like the same application refiled after a denial. "
         f"{pct(vol['share_small_employers_by_visa']['E-3 Australian'], 0)} of E-3 filings come from employers with "
         f"≤10 filings, against {pct(vol['share_small_employers_by_visa']['H-1B'], 0)} of H-1B; but even within "
         f"size groups E-3 is denied more (2–10 filings: {pct(vb['2-10']['E-3 Australian'])} vs "
         f"{pct(vb['2-10']['H-1B'])}). **Candidate problem statement.**"),
        ("fig08_top_filers",
         f"**Big filers don't all file low.** The 15 largest employers ({pct(tf['share_rows'], 0)} of filings) range "
         f"from {100 * min(tf['rel_mean'].values()):+.1f}% to {100 * max(tf['rel_mean'].values()):+.1f}% against "
         f"comparable jobs: {', '.join(neg[:4])} file below the cell median, while {pos[0][0]} "
         f"({100 * pos[0][1]:+.1f}%) and {pos[1][0]} ({100 * pos[1][1]:+.1f}%) file above it. The common belief that "
         f"outsourcing firms file at the lowest levels isn't a clean split here."),
        ("fig09_seniority",
         f"**Job-title keywords behave as expected, which supports relative PW as a wage-level proxy.** 'Senior' "
         f"titles ({pct(sen['senior / sr']['share'], 0)} of filings) sit {100 * sen['senior / sr']['diff']:+.1f}% "
         f"above other titles in the same SOC × state × year, manager/director titles "
         f"{100 * sen['manager / director / head / VP']['diff']:+.1f}%, and junior/entry/intern titles "
         f"{100 * sen['junior / entry / intern / trainee']['diff']:+.1f}% (only {sen['junior / entry / intern / trainee']['n']:,} filings)."),
        ("fig10_missingness",
         f"**Missingness is structural, by year, not random.** Quarter is blank before FY2020 (column absent); "
         f"FY2019 has no prevailing wage ({miss['prevailing_wage'][2019]:.0f}% blank) and no worksite in "
         f"{miss['worksite_state'][2019]:.0f}% of rows; employer country is blank in {miss['employer_country'][2017]:.0f}% "
         f"of FY2017 rows (filled during cleaning). Every analysis has to state which years it can use."),
        ("fig11_associations",
         f"**Most categorical features are only weakly related; the strong links are geography and pay unit.** "
         f"Cramér's V: worksite state ~ region {cv['worksite_region ~ worksite_state']:.2f} (nested), worksite ~ "
         f"employer state {cv['worksite_state ~ employer_state']:.2f}, full-time ~ unit of pay "
         f"{cv['full_time ~ unit_of_pay']:.2f} (part-time jobs are paid by the hour); everything else is ≤ "
         f"{sorted(cv.values())[-5]:.2f}. Visa class is nearly unrelated to everything else (V ≤ "
         f"{max(x for k, x in cv.items() if 'visa_class' in k):.2f}), so a visa comparison isn't confounded by one "
         f"dominant variable. Numerically, log employer volume and log n_identical are ρ = "
         f"{sp['employer_volume ~ n_identical']:.2f}, so they shouldn't both enter one model unregularized, and "
         f"bigger filers send workers out of their own state more often (volume ~ same_state ρ = "
         f"{sp['employer_volume ~ same_state']:.2f}): consulting placements."),
        ("fig12_duplicates",
         f"**Identical rows look like real separate filings, which supports keeping them (cleaning step 1).** "
         f"{pct(dup['share'], 0)} of rows have an identical twin; the share rises from "
         f"{pct(dup['by_volume_bin']['2-10'], 0)} at employers with 2–10 filings to {pct(dup['by_volume_bin']['>1000'], 0)} "
         f"at the largest, and it is {pct(dup['by_status']['CERTIFIED'], 0)} for CERTIFIED against "
         f"{pct(dup['by_status']['WITHDRAWN'], 0)} for WITHDRAWN. A copying error wouldn't depend on employer size or "
         f"outcome. Because one employer's filings are so similar, tests must cluster by employer."),
    ]

    lines = [
        "# Exploratory data analysis (exploration set)",
        "",
        f"{s['n_rows']:,} cleaned filings (FY2017–2022). The confirmation set was not used. **Relative PW** = log "
        f"prevailing wage minus the median of the same SOC title × worksite state × fiscal year (cells with ≥ "
        f"{MIN_CELL} wages); a value of +0.05 is about 5% above comparable jobs.",
        "",
    ]
    for name, cap in figs:
        lines += [f"![{name}](../figures/{name}.png)", "", cap, ""]
    lines += [
        "## What this means for the next steps",
        "",
        "- Work on **log prevailing wage**, always **within SOC title × state × year**; raw comparisons mix "
        "occupation, place and time.",
        "- Use FY2017–18 and FY2020–22 for prevailing-wage questions; FY2019 only for status questions.",
        "- The two non-obvious patterns worth testing involve **visa class**: the E-3 wage premium (Fig. 5) and "
        "the E-3 denial gap that remains after employer size (Fig. 7).",
        "- Filings from one employer are near-copies (Fig. 12): **inference must use employer-clustered SEs**.",
        "",
    ]
    (REPORT / "eda.md").write_text("\n".join(lines))


def main():
    fig_missingness()
    df = load()
    STATS["n_rows"] = len(df)
    for f in [fig_wage_distribution, fig_year, fig_state, fig_soc, fig_visa, fig_status, fig_employer_volume,
              fig_top_filers, fig_seniority, fig_associations, fig_duplicates]:
        f(df)
        print("done", f.__name__)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "eda_stats.json").write_text(json.dumps(STATS, indent=1, default=str))
    write_report()
    print(f"Wrote {len(list(FIGURES.glob('fig*.png')))} figures to {FIGURES} and {REPORT / 'eda.md'}")


if __name__ == "__main__":
    main()
