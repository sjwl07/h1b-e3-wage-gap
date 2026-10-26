"""Step 6: power check for the planned tests: the chance each test detects a gap
exactly the size of the smallest meaningful difference (MID).

Each model is fit on the exploration set to estimate how noisy it is (its SE);
the estimate itself is never read. The confirmation set has ~1/4 as many
employers, so its SE will be about sqrt(4) = 2x larger, and the SE is scaled up
to match. Power uses alpha = 0.05/4, the strictest threshold Holm can apply.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_PROCESSED, REPORT
from fe import fe_ols
from tests_spec import ALPHA_FAMILY, COLS, TESTS, prepare


def power(se, delta, alpha):
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - delta / se) + stats.norm.cdf(-z - delta / se))


def main():
    split = json.loads((REPORT / "split_log.json").read_text())
    scale = np.sqrt(split["employers_explore"] / split["employers_confirm"])
    alpha = ALPHA_FAMILY / len(TESTS)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)

    d = prepare(pd.read_parquet(DATA_PROCESSED / "explore_clean.parquet", columns=COLS))
    rows = []
    for name, t in TESTS.items():
        r = fe_ols(t["pop"](d), t["y"], "e3", t["cells"], "employer_key")
        se = r["se"] * scale
        mde = z * se
        rows.append({
            "test": name, "description": t["label"],
            "rows used (explore)": r["n"], "E-3 rows": r["n_treated"], "employers": r["clusters"],
            "employers with E-3": r["clusters_treated"],
            "SE explore": round(r["se"], 5), "SE confirm (expected)": round(se, 5),
            "MID": f"{100 * (np.exp(t['mid']) - 1):.0f}%" if t["unit"] == "log" else f"{100 * t['mid']:.1f} pp",
            "power at MID": round(power(se, t["mid"], alpha), 3),
            "MDE (80%)": f"{100 * (np.exp(mde) - 1):.1f}%" if t["unit"] == "log" else f"{100 * mde:.2f} pp",
        })
        print(rows[-1])
    tab = pd.DataFrame(rows)
    note = (f"alpha = {alpha:.4f} (Holm worst case, {ALPHA_FAMILY}/{len(TESTS)}). SEs are employer-clustered; "
            f"confirm SE = explore SE × sqrt({split['employers_explore']:,} / {split['employers_confirm']:,} "
            f"employers) = × {scale:.2f}. MDE = smallest effect detectable with 80% power. Only SEs were read "
            "from the exploration fits, not the estimates.")
    (REPORT / "power.md").write_text("# Power check (exploration-set SEs)\n\n" + tab.to_markdown(index=False)
                                     + "\n\n" + note + "\n")
    print(note)


if __name__ == "__main__":
    main()
