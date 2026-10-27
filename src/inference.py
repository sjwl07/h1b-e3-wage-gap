"""Step 7: run the planned tests (report/preregistration.md) on the confirmation set.

The confirmation set is cleaned with the exploration-fitted parameters (nothing
re-fitted), then each test runs once. Populations and models come from
src/tests_spec.py, the same definitions power.py used.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

from clean import clean
from config import CONFIRM_PATH, DATA_PROCESSED, FIGURES, REPORT
from fe import fe_ols, informative
from tests_spec import ALPHA_FAMILY, TESTS, prepare, relative_pw

# ---------- inference helpers ----------

def ci(r, level=0.95):
    q = stats.t.ppf(0.5 + level / 2, r["df"])
    return r["beta"] - q * r["se"], r["beta"] + q * r["se"]


def classify(reject, lo, hi, mid):
    """Verdict from the 95% CI against the smallest meaningful difference (MID)."""
    if lo > mid or hi < -mid:
        return "meaningful"
    if -mid < lo and hi < mid:
        return "negligible"
    return "possibly too small" if reject else "inconclusive"


def to_pct(x, unit):
    """log points -> % difference; proportions -> percentage points."""
    return 100 * np.expm1(x) if unit == "log" else 100 * x


def fmt_effect(x, unit):
    return f"{to_pct(x, unit):+.2f}%" if unit == "log" else f"{to_pct(x, unit):+.3f} pp"


# ---------- robustness checks ----------

def naive_checks(pop, t):
    """T1: Welch two-sample t-test on the median-based relative PW used in EDA (treats filings as independent)."""
    rel = relative_pw(pop)
    a, b = rel[pop["e3"] == 1].dropna(), rel[pop["e3"] == 0].dropna()
    w = stats.ttest_ind(a, b, equal_var=False)
    return {
        "naive Welch t on relative PW (ignores clustering)": {
            "diff": float(a.mean() - b.mean()), "t": float(w.statistic), "p": float(w.pvalue),
            "naive SE": float(np.sqrt(a.var() / len(a) + b.var() / len(b)))},
    }


def by_year(pop, t):
    rows = {}
    for y, sub in pop.groupby("fiscal_year"):
        if informative(sub, "e3", t["cells"]).sum() == 0:
            continue
        r = fe_ols(sub, t["y"], "e3", t["cells"], "employer_key")
        lo, hi = ci(r)
        rows[int(y)] = {"beta": r["beta"], "lo": lo, "hi": hi, "n_e3": r["n_treated"]}
    return rows


def cmh(pop, cells):
    tables = []
    for _, sub in pop.groupby(cells, observed=True):
        tab = pd.crosstab(sub["e3"], sub["denied"]).reindex(index=[1, 0], columns=[1.0, 0.0], fill_value=0)
        if (tab.sum(axis=1) > 0).all() and (tab.sum(axis=0) > 0).all():  # strata without both margins add nothing
            tables.append(tab.to_numpy().astype(float))
    st = StratifiedTable(tables)
    res = st.test_null_odds(correction=True)
    return {"pooled MH odds ratio": float(st.oddsratio_pooled), "chi2": float(res.statistic),
            "p": float(res.pvalue), "informative strata": len(tables)}


def size_checks(pop, t):
    """T3: a logit with the same size x year fixed effects."""
    out = {}
    # logistic regression with the same cell fixed effects, employer-clustered;
    # fit on rows aggregated to employer x cell x visa (binomial counts) to keep it small
    agg = (pop.groupby(["employer_key"] + t["cells"] + ["e3"], observed=True)["denied"]
           .agg(["sum", "size"]).reset_index())
    cell = agg[t["cells"]].astype(str).agg("|".join, axis=1)
    X = pd.get_dummies(cell, drop_first=True, dtype=float)
    X.insert(0, "e3", agg["e3"].astype(float))
    X = sm.add_constant(X)
    endog = np.column_stack([agg["sum"], agg["size"] - agg["sum"]])
    fit = sm.GLM(endog, X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": agg["employer_key"].astype("category").cat.codes})
    out["logit with size x year FE, employer-clustered"] = {
        "OR": float(np.exp(fit.params["e3"])), "z": float(fit.tvalues["e3"]), "p": float(fit.pvalues["e3"])}
    return out


def employer_paired(pop, t):
    """T2, T4 (added after the planned run): one gap per employer, the average of its within-cell
    E-3 minus H-1B mean differences, so employers are the independent units. Then a one-sample t-test
    on those gaps: no regression, no clustering."""
    d = pop[informative(pop, "e3", t["cells"])]
    m = d.groupby(t["cells"] + ["e3"], observed=True)[t["y"]].mean().unstack("e3")
    gap = (m[1] - m[0]).groupby(level="employer_key").mean()
    tt = stats.ttest_1samp(gap, 0)
    return {
        "employer-level paired t (one gap per employer)": {
            "mean gap": float(gap.mean()), "t": float(tt.statistic), "df": len(gap) - 1, "p": float(tt.pvalue),
            "employers": len(gap), "share of employers with E-3 higher": float((gap > 0).mean())},
    }


def employer_checks(pop, t):
    """T4: CMH test stratified by employer x year, plus the employer-level paired tests."""
    d = pop[informative(pop, "e3", t["cells"])]
    return {"Cochran-Mantel-Haenszel by employer x year (ignores clustering)": cmh(d, t["cells"]),
            **employer_paired(pop, t)}


# T2 shares T1's population, so the Welch check would only repeat T1's
EXTRA_CHECKS = {"T1": naive_checks, "T2": employer_paired, "T3": size_checks, "T4": employer_checks}


# ---------- figures ----------

def figures(results, years):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for axis, names, unit, xl in [(ax[0], ["T1", "T2"], "log", "E-3 minus H-1B prevailing wage (%)"),
                                  (ax[1], ["T3", "T4"], "pp", "E-3 minus H-1B denial rate (pp)")]:
        mid = results[names[0]]["mid"]
        f = lambda v, u=unit: to_pct(v, u)
        axis.axvspan(f(-mid), f(mid), color="0.9", label="within ±MID")
        axis.axvline(0, color="k", lw=0.8)
        for i, n in enumerate(names):
            r = results[n]
            axis.errorbar(f(r["beta"]), i, xerr=[[f(r["beta"]) - f(r["ci95"][0])], [f(r["ci95"][1]) - f(r["beta"])]],
                          fmt="o", capsize=5, color="tab:red" if r["holm_reject"] else "tab:gray")
            axis.text(f(r["beta"]), i - 0.22, r["verdict"], ha="center", fontsize=8)
        axis.set_yticks(range(len(names)), [f"{n}: {TESTS[n]['label'].split(', ', 1)[1]}" for n in names], fontsize=8)
        axis.set_ylim(-0.6, len(names) - 0.4)
        axis.invert_yaxis()
        axis.set(xlabel=xl)
        axis.legend(fontsize=8, loc="lower right")
    ax[0].set_title("P1: prevailing wage (95% CI)")
    ax[1].set_title("P2: denial rate (95% CI)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_inference_estimates.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for axis, n, unit in [(ax[0], "T1", "log"), (ax[1], "T3", "pp")]:
        f = lambda v, u=unit: to_pct(v, u)
        yy = sorted(years[n])
        b = [f(years[n][y]["beta"]) for y in yy]
        lo = [f(years[n][y]["lo"]) for y in yy]
        hi = [f(years[n][y]["hi"]) for y in yy]
        axis.errorbar(yy, b, yerr=[np.subtract(b, lo), np.subtract(hi, b)], fmt="o-", capsize=4)
        mid = f(results[n]["mid"])
        axis.axhspan(-mid, mid, color="0.9")
        axis.axhline(0, color="k", lw=0.8)
        axis.set(xlabel="Fiscal year", ylabel="% (wage)" if unit == "log" else "pp (denial)",
                 title=f"{n} estimated separately for each year (95% CI)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig14_inference_by_year.png", dpi=130)
    plt.close(fig)


# ---------- main ----------

def main():
    params = json.loads((DATA_PROCESSED / "clean_params.json").read_text())
    raw = pd.read_parquet(CONFIRM_PATH)
    cdf, clog, _ = clean(raw, params)
    del raw
    cdf.to_parquet(DATA_PROCESSED / "confirm_clean.parquet", index=False)
    (REPORT / "cleaning_log_confirm.md").write_text(
        "# Cleaning log (confirmation set, exploration-fitted rules)\n\n"
        + clog.drop(columns="reason").to_markdown(index=False) + "\n")
    d = prepare(cdf)
    del cdf

    results, checks, years = {}, {}, {}
    for name, t in TESTS.items():
        pop = t["pop"](d)
        r = fe_ols(pop, t["y"], "e3", t["cells"], "employer_key")
        raw_gap = pop.loc[pop["e3"] == 1, t["y"]].mean() - pop.loc[pop["e3"] == 0, t["y"]].mean()
        used = pop[informative(pop, "e3", t["cells"])]
        results[name] = {**r, "label": t["label"], "mid": t["mid"], "unit": t["unit"],
                         "raw_gap_no_fe": float(raw_gap),
                         "mean_y_h1b": float(pop.loc[pop["e3"] == 0, t["y"]].mean()),
                         "mean_y_e3": float(pop.loc[pop["e3"] == 1, t["y"]].mean()),
                         "mean_y_h1b_used": float(used.loc[used["e3"] == 0, t["y"]].mean())}
        extra = EXTRA_CHECKS.get(name)
        if extra:
            checks[name] = extra(pop, t)
        if name in ("T1", "T3"):
            years[name] = by_year(pop, t)
        print(name, {k: v for k, v in results[name].items() if k in ("beta", "se", "t", "df", "p")})

    names = list(TESTS)
    reject, p_holm, _, _ = multipletests([results[n]["p"] for n in names], alpha=ALPHA_FAMILY, method="holm")
    for n, rej, ph in zip(names, reject, p_holm):
        r = results[n]
        lo, hi = ci(r)
        r.update({"p_holm": float(ph), "holm_reject": bool(rej), "ci95": [float(lo), float(hi)]})
        r["verdict"] = classify(rej, lo, hi, r["mid"])

    figures(results, years)
    out = {"results": results, "robustness": checks, "by_year": years,
           "n_confirm_rows": int(len(d))}
    (REPORT / "inference_results.json").write_text(json.dumps(out, indent=1, default=float))
    write_md(results, checks, years, clog)


def interpretation(results, checks, years):
    r1, r2, r3, r4 = (results[n] for n in ["T1", "T2", "T3", "T4"])
    pc = lambda v: to_pct(v, "log")
    naive = checks["T1"]["naive Welch t on relative PW (ignores clustering)"]
    y1 = [pc(v["beta"]) for v in years["T1"].values()]
    y3 = years["T3"]
    cmh4 = checks["T4"]["Cochran-Mantel-Haenszel by employer x year (ignores clustering)"]
    logit3 = checks["T3"]["logit with size x year FE, employer-clustered"]
    e3_rate4 = r4["mean_y_h1b_used"] + r4["beta"]
    return [
        "## Interpretation",
        "",
        f"**P1: E-3 filings carry a meaningfully higher prevailing wage, and most of it is within employer.**",
        f"- T1: for the same SOC title, state and year, E-3 filings are {pc(r1['beta']):+.1f}% "
        f"[{pc(r1['ci95'][0]):+.1f}, {pc(r1['ci95'][1]):+.1f}] above H-1B; the whole CI is above the 5% MID. "
        f"The raw gap is {pc(r1['raw_gap_no_fe']):+.1f}%, so occupation, state and year mix explain almost none "
        "of it.",
        f"- T2: within the same employer, job, state and year the gap is {pc(r2['beta']):+.1f}% "
        f"[{pc(r2['ci95'][0]):+.1f}, {pc(r2['ci95'][1]):+.1f}], still entirely above the MID. "
        f"About {100 * r2['beta'] / r1['beta']:.0f}% of the T1 gap (in log points) remains when the employer is "
        "held fixed (T2 uses only employers filing both visas for the same job, so the two numbers come from "
        "different sets of employers), so it isn't mainly that higher-paying employers hire Australians: the same employer files "
        "its E-3 workers at a higher wage level.",
        f"- Stable across years ({min(y1):+.1f}% to {max(y1):+.1f}%, Fig. 14).",
        f"- Clustering matters: treating filings as independent gives an SE of {naive['naive SE']:.4f}, against "
        f"{r1['se']:.4f} clustered by employer ({r1['se'] / naive['naive SE']:.1f}× larger).",
        "",
        f"**P2: E-3 filings are denied more often, but within employer the gap is small in absolute terms.**",
        f"- T3: holding employer size and year fixed, E-3 filings are denied {100 * r3['beta']:+.2f} pp "
        f"[{100 * r3['ci95'][0]:+.2f}, {100 * r3['ci95'][1]:+.2f}] more often (H-1B rate "
        f"{100 * r3['mean_y_h1b']:.2f}%). The raw gap is {100 * r3['raw_gap_no_fe']:+.2f} pp, so employer size "
        f"explains only {100 * (1 - r3['beta'] / r3['raw_gap_no_fe']):.0f}% of it. The power check expected only "
        "0.16 power at the 0.5 pp MID, but the effect is far larger than the MID, so that doesn't limit this "
        f"result. Logit odds ratio {logit3['OR']:.1f}.",
        f"- The T3 gap shrinks over time, from {100 * y3[2018]['beta']:+.1f} pp in FY2018 to "
        f"{100 * y3[2022]['beta']:+.1f} pp in FY2022 (Fig. 14), in line with the overall fall in denials.",
        f"- T4: within the same employer and year the gap is {100 * r4['beta']:+.2f} pp "
        f"[{100 * r4['ci95'][0]:+.2f}, {100 * r4['ci95'][1]:+.2f}], which rejects H₀ but the CI straddles the "
        f"0.5 pp MID, so it may be too small to matter. In relative terms it isn't small: these employers' H-1B "
        f"rate is {100 * r4['mean_y_h1b_used']:.2f}%, so E-3 is about {e3_rate4 / r4['mean_y_h1b_used']:.1f}× that "
        f"(CMH odds ratio {cmh4['pooled MH odds ratio']:.2f}).",
        f"- So most of the E-3 denial gap is **between** employers: {100 * (1 - r4['beta'] / r3['beta']):.0f}% of "
        "the T3 gap disappears when the employer is fixed (again from different sets of employers: T4 covers only "
        "employers filing both visas in a year). E-3 filings come disproportionately from employers "
        "that get denied often, beyond what their size predicts.",
        "",
        "**Caveats.** These are differences between filings, not effects of the visa (see the causation note in "
        "`report/problem_statements.md`). T2 and T4 describe employers that file both visa types, which are larger "
        "than average. The wage outcome is the prevailing wage, not the wage paid.",
        "",
    ]


def write_md(results, checks, years, clog):
    rows = []
    for n, r in results.items():
        u = r["unit"]
        rows.append({
            "test": n, "comparison": r["label"],
            "rows / E-3 rows": f"{r['n']:,} / {r['n_treated']:,}",
            "employers (G)": f"{r['clusters']:,}",
            "estimate": fmt_effect(r["beta"], u),
            "95% CI": f"[{fmt_effect(r['ci95'][0], u)}, {fmt_effect(r['ci95'][1], u)}]",
            "t (df)": f"{r['t']:.2f} ({r['df']:,})",
            "p": f"{r['p']:.2e}", "p Holm": f"{r['p_holm']:.2e}",
            "Holm decision": "reject H₀" if r["holm_reject"] else "fail to reject H₀",
            "MID": fmt_effect(r["mid"], u).lstrip("+"),
            "verdict": r["verdict"],
        })
    tab = pd.DataFrame(rows)
    raw_rows = pd.DataFrame([{"test": n, "E-3 mean": (f"{np.exp(r['mean_y_e3']):,.0f} (geo. mean $)" if r["unit"] == "log"
                                                      else f"{100 * r['mean_y_e3']:.2f}%"),
                              "H-1B mean": (f"{np.exp(r['mean_y_h1b']):,.0f}" if r["unit"] == "log"
                                            else f"{100 * r['mean_y_h1b']:.2f}%"),
                              "raw gap (no fixed effects)": fmt_effect(r["raw_gap_no_fe"], r["unit"]),
                              "adjusted gap (test)": fmt_effect(r["beta"], r["unit"])}
                             for n, r in results.items()])

    def chk(n):
        lines = []
        for k, v in checks[n].items():
            if isinstance(v, dict):
                lines.append(f"- {k}: " + ", ".join(
                    f"{a} = {b:.3g}" if isinstance(b, float) else f"{a} = {b}" for a, b in v.items()))
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def yrs(n):
        u = results[n]["unit"]
        return ", ".join(f"FY{y}: {fmt_effect(v['beta'], u)} [{fmt_effect(v['lo'], u)}, {fmt_effect(v['hi'], u)}]"
                         for y, v in years[n].items())

    md = [
        "# Hypothesis test results (confirmation set)",
        "",
        f"Planned in `report/preregistration.md`; each test run once. Confirmation set cleaned with the "
        f"exploration-fitted rules (`report/cleaning_log_confirm.md`): {clog['rows'].iloc[-1]:,} filings, no rows "
        f"removed. Family-wise α = {ALPHA_FAMILY} (Holm, 4 tests), two-sided, SEs clustered by employer, "
        f"t-test with employers − 1 degrees of freedom.",
        "",
        tab.to_markdown(index=False),
        "",
        "![estimates](../figures/fig13_inference_estimates.png)",
        "",
        "## Raw vs adjusted gaps",
        "",
        raw_rows.to_markdown(index=False),
        "",
        *interpretation(results, checks, years),
        "## Robustness checks",
        "",
        "**T1**", chk("T1"), "", f"By year: {yrs('T1')}", "",
        "**T2**", chk("T2"), "",
        "**T3**", chk("T3"), "", f"By year: {yrs('T3')}", "",
        "**T4**", chk("T4"), "",
        "![by year](../figures/fig14_inference_by_year.png)",
        "",
    ]
    (REPORT / "inference_results.md").write_text("\n".join(md))
    print(tab.to_string(index=False))
    print(raw_rows.to_string(index=False))
    for n in results:
        print(n, json.dumps(checks[n], default=float))
    for n in years:
        print(n, yrs(n))


if __name__ == "__main__":
    main()
