"""Section 2: data integrity and lineage."""
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from . import Check
from ..evidence import feature, read_quarterly


def run(ev, out_dir):
    th = ev.cfg["thresholds"]
    checks = []

    # D1 lineage: the developer's prepared target vs the validator's rebuild from raw data + events
    p = os.path.join(ev.run_dir, "fitted", "data", "target.csv")
    if os.path.exists(p):
        dev = read_quarterly(p)["organic_change"]
        gap = (dev - ev.growth.reindex(dev.index)).abs()
        mx = float(gap.max())
        ok = mx < 1e-9
        checks.append(Check(
            "D1", "data", "Target lineage: raw balances and events rebuilt independently",
            "pass" if ok else "fail",
            ("The validator's organic growth, rebuilt from the raw balance file and the events file, matches the "
             f"developer's prepared target in every quarter (largest gap {mx:.1e}).") if ok else
            (f"The developer's prepared target differs from the independent rebuild by up to {mx:.2e} "
             "log points: the event adjustment or the raw data was changed between files."),
            {"max_abs_gap": mx, "quarters_compared": int(gap.notna().sum())}, "gap < 1e-9",
            None if ok else "high", "" if ok else "Reconcile the prepared target to the raw data and document every adjustment."))

    # D2 completeness: gaps or missing values in the history the model uses
    y, X = ev.estimation_frame()
    full = ev.design().loc[ev.est_start:ev.hist_end].assign(_y=ev.growth)
    miss = int(full.isna().any(axis=1).sum())
    idx = ev.level.index
    gaps = int(len(pd.period_range(idx.min(), idx.max(), freq="Q")) - len(idx))
    ok = miss == 0 and gaps == 0
    checks.append(Check(
        "D2", "data", "Completeness of the estimation sample", "pass" if ok else "warn",
        f"{len(y)} quarters ({y.index[0]} to {y.index[-1]}) with no missing quarters and no missing values."
        if ok else f"{miss} quarters have missing values and {gaps} quarters are absent from the target file.",
        {"n_obs": len(y), "start": str(y.index[0]), "end": str(y.index[-1]), "rows_with_missing": miss,
         "absent_quarters": gaps}, "no gaps", None if ok else "medium"))

    # D3 outliers: robust z-scores of organic growth, and whether a dummy or event covers them
    g = ev.growth.loc[ev.est_start:ev.hist_end].dropna()
    med = g.median()
    mad = 1.4826 * (g - med).abs().median()
    z = (g - med) / mad
    out = z[z.abs() > 3.5]
    covered = {str(q) for q in out.index if f"D_{q}" in ev.dummies or ev.event_adj.get(q, 0) != 0}
    uncovered = [str(q) for q in out.index if str(q) not in covered]
    tab = pd.DataFrame({"quarter": [str(q) for q in out.index], "growth_pct": (g[out.index] * 100).values,
                        "robust_z": out.values.round(2),
                        "treated_by": ["dummy" if f"D_{q}" in ev.dummies else ("" if str(q) not in covered else "event")
                                       for q in out.index]})
    status = "pass" if not uncovered else "warn"
    checks.append(Check(
        "D3", "data", "Outliers in target growth", status,
        (("No quarter of target growth has |robust z| > 3.5." if len(out) == 0 else
          f"{len(out)} quarters have |robust z| > 3.5; all are treated by a dummy or an event adjustment.")
         if not uncovered else
         f"{len(uncovered)} of {len(out)} outlying quarters ({', '.join(uncovered)}) are not treated and pull the "
         "coefficients toward them."),
        {"n_outliers": len(out), "untreated": len(uncovered)}, "|robust z| > 3.5 treated",
        None if not uncovered else "low", "" if not uncovered else "Explain or treat the untreated outliers.",
        table=tab))

    # D4 stationarity of target growth and every driver (independent ADF and KPSS)
    rows = []
    series = {"target growth": g}
    for d in ev.drivers:
        series[d] = feature(ev.macro_hist, d).loc[ev.est_start:ev.hist_end].dropna()
    for name, s in series.items():
        adf_p = adfuller(s, autolag="AIC")[1]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_p = kpss(s, regression="c", nlags="auto")[1]
        rows.append({"series": name, "adf_p": adf_p, "kpss_p": kpss_p,
                     "verdict": "stationary" if adf_p < 0.05 and kpss_p > 0.05 else
                     ("conflicting" if (adf_p < 0.05) == (kpss_p < 0.05) else "non-stationary")})
    tab = pd.DataFrame(rows)
    bad = tab[tab.verdict != "stationary"]
    status = "pass" if bad.empty else "warn"
    checks.append(Check(
        "D4", "data", "Stationarity of target growth and drivers (ADF and KPSS)", status,
        "Target growth and every driver are stationary on both tests." if bad.empty else
        f"{len(bad)} series are not clearly stationary on both tests: {', '.join(bad.series)}.",
        {"series_tested": len(tab), "not_clear": len(bad)}, "ADF p < 0.05 and KPSS p > 0.05",
        None if bad.empty else "low", "" if bad.empty else "Confirm the transform, or test for a cointegrating relation.",
        table=tab))

    # D5 scenario extrapolation: forecast-period driver values outside their historical range
    rows = []
    for sc, mac in ev.scenarios.items():
        for d in ev.drivers:
            f = feature(mac, d)
            hist = f.loc[ev.est_start:ev.hist_end].dropna()
            fut = f.loc[ev.hist_end + 1:ev.hist_end + 12].dropna()
            outside = ((fut < hist.min()) | (fut > hist.max())).mean()
            rows.append({"scenario": sc, "driver": d, "hist_min": hist.min(), "hist_max": hist.max(),
                         "scen_min": fut.min(), "scen_max": fut.max(), "share_outside": outside})
    tab = pd.DataFrame(rows)
    worst = tab.sort_values("share_outside", ascending=False).iloc[0]
    lim = th["max_extrapolation_share"]
    status = "pass" if worst.share_outside <= lim else "warn"
    checks.append(Check(
        "D5", "data", "Scenario inputs inside the historical range of each driver", status,
        ("In every scenario, every driver stays inside its historical range over the 12 forecast quarters."
         if worst.share_outside == 0 else
         f"In every scenario at most {worst.share_outside:.0%} of forecast quarters fall outside a driver's history.")
        if status == "pass" else
        f"{worst.driver} in {worst.scenario} is outside its historical range in {worst.share_outside:.0%} of forecast "
        "quarters: the model is extrapolating there.",
        {"worst_share_outside": float(worst.share_outside), "worst_driver": worst.driver,
         "worst_scenario": worst.scenario}, f"share outside <= {lim:.0%}", None if status == "pass" else "medium",
        "" if status == "pass" else "Add an expert overlay or a bounded functional form for extrapolated quarters.",
        table=tab))
    return checks
