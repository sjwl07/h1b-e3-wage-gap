"""One-regressor fixed-effects OLS with employer-clustered SEs, for millions of rows.

y = beta * x + cell fixed effect + error. The fixed effects are removed by
demeaning y and x within each cell (Frisch-Waugh-Lovell), so no dummy matrix is
built. Cluster-robust variance (CR1):

    V = G/(G-1) * sum_g (sum_{i in g} x~_i e_i)^2 / (sum_i x~_i^2)^2

with a t reference on G-1 df. The fixed-effect cells here are nested within
or much smaller than employers, so the usual G/(G-1) correction is used
without an extra (N-1)/(N-K) term (as in Stata's reghdfe for nested FE).
"""
import numpy as np
import pandas as pd
from scipy import stats


def demean(df, cols, cells):
    g = df.groupby(cells, observed=True, sort=False)
    return pd.DataFrame({c: df[c] - g[c].transform("mean") for c in cols}, index=df.index)


def informative(df, x, cells):
    """Rows whose cell has variation in x (others drop out of the within estimate)."""
    return df.groupby(cells, observed=True, sort=False)[x].transform("nunique") > 1


def fe_ols(df, y, x, cells, cluster):
    d = df[informative(df, x, cells)]
    dm = demean(d, [y, x], cells)
    xt, yt = dm[x].to_numpy(float), dm[y].to_numpy(float)
    sxx = float(xt @ xt)
    beta = float(xt @ yt) / sxx
    e = yt - beta * xt
    score = pd.Series(xt * e).groupby(d[cluster].to_numpy()).sum()
    G = len(score)
    se = float(np.sqrt(G / (G - 1) * (score.to_numpy() ** 2).sum()) / sxx)
    t = beta / se
    p = float(2 * stats.t.sf(abs(t), G - 1))
    return {"beta": beta, "se": se, "t": t, "df": G - 1, "p": p, "n": len(d), "clusters": G,
            "n_cells": int(d.groupby(cells, observed=True).ngroups),
            "n_treated": int((d[x] == 1).sum()), "clusters_treated": int(d.loc[d[x] == 1, cluster].nunique()),
            "_xt": xt, "_e": e, "_yt": yt, "_g": d[cluster].to_numpy(), "_sxx": sxx}


def public(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}
