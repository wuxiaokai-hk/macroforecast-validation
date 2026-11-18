"""Section 5: forecast performance against simple benchmarks.

Two kinds of evidence:
- the developer's own out-of-sample test (drivers re-selected at each origin): read and judged for depth;
- the validator's independent rolling-origin backtest of the fixed equation, re-estimated at every quarter,
  conditional on the actual path of the drivers (the way a scenario model is used). Because the drivers were
  chosen on the full sample, this backtest is optimistic for the model; it is a ceiling, not a guarantee.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import Check
from .. import charts
from .replication import forecast_path

COVID = pd.period_range("2020Q1", "2021Q2", freq="Q")


def dm_test(e1, e2, h):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction (squared-error loss).

    Returns (statistic, two-sided p). Negative statistic = model 1 more accurate.
    """
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    dbar = d.mean()
    gam = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return hln, 2 * stats.t.sf(abs(hln), df=n - 1)


def _ols(y, X):
    keep = [c for c in X.columns if X[c].abs().sum() > 0]
    return sm.OLS(y, X[keep]).fit().params


def backtest(ev, horizons):
    """Rolling-origin errors (cumulative log-growth error x 100 ~ % balance error) for model, AR(1), drift."""
    y_all, X_all = ev.estimation_frame()
    Xfull = ev.design()
    first = pd.Period(ev.cfg["independent_backtest"]["first_origin"], freq="Q")
    H = max(horizons)
    ar_terms = ["const", "y_lag1"] + ev.dummies
    rows = []
    for o in pd.period_range(first, ev.hist_end - 1, freq="Q"):
        y, X = y_all.loc[:o], X_all.loc[:o]
        b_m = _ols(y, X)
        b_ar = _ols(y, X[[c for c in ar_terms if c in X]] if "y_lag1" in X else
                    X[["const"]].assign(y_lag1=ev.growth.shift(1).loc[X.index]))
        drift = y.iloc[-8:].mean()
        pm = forecast_path(ev, b_m, ev.macro_hist, horizon=H, origin=o, X=Xfull.reindex(
            pd.period_range(Xfull.index[0], ev.hist_end + H, freq="Q")))
        pa = forecast_path(ev, b_ar, ev.macro_hist, horizon=H, origin=o, X=Xfull.reindex(
            pd.period_range(Xfull.index[0], ev.hist_end + H, freq="Q")))
        for h in horizons:
            if o + h > ev.hist_end:
                continue
            actual = ev.growth.loc[o + 1:o + h].sum()
            rows.append({"origin": str(o), "h": h,
                         "covid": bool(any(q in COVID for q in pd.period_range(o + 1, o + h, freq="Q"))),
                         "model": (pm.change[:h].sum() - actual) * 100,
                         "ar1": (pa.change[:h].sum() - actual) * 100,
                         "drift": (drift * h - actual) * 100})
    return pd.DataFrame(rows)


def run(ev, out_dir):
    th = ev.cfg["thresholds"]
    horizons = ev.cfg["independent_backtest"]["horizons"]
    checks = []

    # P1 depth of the developer's out-of-sample evidence
    oos = ev.run_file("oos", "6_all_forecast_errors.csv")
    n_orig = int(oos[oos.model == ev.model].origin.nunique()) if oos is not None else 0
    orig = sorted(oos[oos.model == ev.model].origin.unique()) if oos is not None else []
    ok = n_orig >= th["min_oos_origins"]
    checks.append(Check(
        "P1", "performance", "Depth of the developer's out-of-sample test (drivers re-selected at each origin)",
        "pass" if ok else "fail",
        f"The developer's true out-of-sample test uses {n_orig} forecast origins." if ok else
        (f"The developer's out-of-sample test, the only one that re-selects drivers, rests on {n_orig} forecast "
         f"origin(s) ({', '.join(orig)}), so its ranking of models (and the choice of '{ev.model}') is not "
         "statistically supported; the developer's DM tests have no power at this depth."),
        {"origins": n_orig, "origin_list": orig, "required": th["min_oos_origins"]},
        f">= {th['min_oos_origins']} origins", None if ok else "high",
        "" if ok else "Re-run the out-of-sample test with a quarterly or semi-annual step (oos.step) and an earlier "
                      "first origin, and choose the model on that evidence."))

    # P2 independent backtest vs benchmarks
    bt = backtest(ev, horizons)
    clean = bt[~bt.covid]
    rows = []
    for h in horizons:
        g = clean[clean.h == h]
        rm = np.sqrt((g.model ** 2).mean())
        ra = np.sqrt((g.ar1 ** 2).mean())
        rd = np.sqrt((g.drift ** 2).mean())
        dm, p = dm_test(g.model, g.ar1, h)
        rows.append({"h": h, "n": len(g), "rmse_model": rm, "rmse_ar1": ra, "rmse_drift": rd,
                     "skill_vs_ar1_pct": (1 - rm / ra) * 100, "bias_model": g.model.mean(), "dm_vs_ar1": dm,
                     "dm_p": p})
    perf = pd.DataFrame(rows)
    key = perf[perf.h.isin([4, 8])]
    weak = key[(key.skill_vs_ar1_pct < th["min_skill_vs_ar1_pct"]) | (key.dm_p > th["dm_alpha"])]
    fig, ax = charts.plt.subplots(figsize=(6.2, 3.0))
    for i, (col, lab) in enumerate([("rmse_model", ev.model), ("rmse_ar1", "AR(1)"), ("rmse_drift", "Drift")]):
        ax.plot(perf.h, perf[col], marker="o", ms=5, color=charts.SERIES[i], label=lab)
    ax.set_xlabel("quarters ahead")
    ax.set_ylabel("RMSE, % of balance")
    ax.set_xticks(horizons)
    ax.set_title("Independent backtest: balance error by horizon (COVID quarters excluded)")
    ax.legend(loc="upper left")
    chart = charts.save(fig, out_dir, "p2_backtest_rmse.png")
    h8 = perf[perf.h == 8].iloc[0]
    checks.append(Check(
        "P2", "performance", "Independent rolling-origin backtest against AR(1) and drift",
        "pass" if weak.empty else "warn",
        (f"Re-estimated at {int(bt.origin.nunique())} quarterly origins, the equation beats AR(1) at every tested "
         f"horizon; at 8 quarters RMSE is {h8.rmse_model:.2f}% of the balance vs {h8.rmse_ar1:.2f}% for AR(1) "
         f"({h8.skill_vs_ar1_pct:.0f}% better, DM p = {h8.dm_p:.3f}). This is conditional on actual driver paths "
         "and full-sample driver choice, so it is an upper bound on real accuracy.") if weak.empty else
        f"At h = {', '.join(str(int(x)) for x in weak.h)} the model does not beat AR(1) by "
        f"{th['min_skill_vs_ar1_pct']}% with DM p <= {th['dm_alpha']}.",
        {"origins": int(bt.origin.nunique()), "rmse_h8_model": h8.rmse_model, "rmse_h8_ar1": h8.rmse_ar1,
         "skill_h8_pct": h8.skill_vs_ar1_pct, "dm_p_h8": h8.dm_p},
        f"skill >= {th['min_skill_vs_ar1_pct']}% and DM p <= {th['dm_alpha']} at h=4 and h=8",
        None if weak.empty else "medium", table=perf, chart=chart))

    # P3 behaviour through stress: origins whose forecast window covers the 2008-09 recession
    gfc = bt[(bt.origin >= "2008Q1") & (bt.origin <= "2009Q2") & (bt.h == 4)]
    worse = int((gfc.model.abs() > gfc.ar1.abs()).sum())
    under = gfc.model.mean()
    ok = worse <= len(gfc) / 2
    checks.append(Check(
        "P3", "performance", "Accuracy through the 2008-09 stress period (4 quarters ahead)",
        "pass" if ok else "warn",
        f"For the {len(gfc)} origins from 2008Q1 to 2009Q2 the model's 4-quarter error averages {under:+.2f}% "
        f"(AR(1): {gfc.ar1.mean():+.2f}%); it is less accurate than AR(1) at {worse} of them.",
        {"origins": len(gfc), "mean_error_model": under, "mean_error_ar1": gfc.ar1.mean(), "worse_than_ar1": worse},
        "no worse than AR(1) at most stress origins", None if ok else "medium",
        table=gfc[["origin", "model", "ar1", "drift"]].reset_index(drop=True)))

    # P4 bias: are errors centred on zero? (HAC t-test, h = 4 and 8)
    rows = []
    for h in (4, 8):
        e = clean[clean.h == h].model.values
        r = sm.OLS(e, np.ones(len(e))).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        rows.append({"h": h, "mean_error": e.mean(), "t": r.tvalues[0], "p": r.pvalues[0]})
    b = pd.DataFrame(rows)
    biased = b[b.p < 0.05]
    checks.append(Check(
        "P4", "performance", "Forecast bias", "pass" if biased.empty else "warn",
        "Mean errors at 4 and 8 quarters are not significantly different from zero "
        f"({b.mean_error.iloc[0]:+.2f}% and {b.mean_error.iloc[1]:+.2f}%)." if biased.empty else
        "Errors are significantly biased: " + ", ".join(
            f"{r.mean_error:+.2f}% at {int(r.h)} quarters (p = {r.p:.3f})" for _, r in biased.iterrows()) +
        (". The model systematically under-forecasts the balance." if (biased.mean_error < 0).all() else
         ". The model systematically over-forecasts the balance." if (biased.mean_error > 0).all() else "."),
        {"mean_error_h4": b.mean_error.iloc[0], "mean_error_h8": b.mean_error.iloc[1]}, "HAC p >= 0.05",
        None if biased.empty else "medium", table=b))
    return checks, bt
