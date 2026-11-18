"""Section 4: independent replication of the estimates and the scenario forecasts."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import Check


def forecast_path(ev, coefs, macro, horizon=12, start_growth=None, start_level=None, origin=None, X=None):
    """Recursive forecast of the fixed equation from `origin` (default: end of history); dummies are zero."""
    if X is None:
        X = ev.design(macro=macro, growth=ev.growth.reindex(macro.index), terms=list(coefs.index))
    q0 = ev.hist_end if origin is None else origin
    prev = ev.growth[q0] if start_growth is None else start_growth
    level = ev.level[q0] if start_level is None else start_level
    rows = []
    for q in pd.period_range(q0 + 1, q0 + horizon, freq="Q"):
        g = 0.0
        for t, b in coefs.items():
            if t == "const":
                g += b
            elif t == "y_lag1":
                g += b * prev
            elif t.startswith("D_"):
                continue
            else:
                g += b * X.loc[q, t]
        level *= np.exp(g)
        rows.append({"date": str(q), "change": g, "level": level})
        prev = g
    return pd.DataFrame(rows)


def run(ev, out_dir):
    th = ev.cfg["thresholds"]
    checks = []
    y, X = ev.estimation_frame()
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    tab = pd.DataFrame({"term": ev.terms, "developer": ev.coefs.values, "validator": res.params[ev.terms].values})
    tab["rel_diff"] = (tab.validator - tab.developer).abs() / tab.developer.abs().clip(lower=1e-12)
    mx = float(tab.rel_diff.max())
    ok = mx <= th["replication_tol"]
    checks.append(Check(
        "R1", "replication", "Coefficients re-estimated from raw data", "pass" if ok else "fail",
        f"All {len(tab)} coefficients reproduce from the raw files with an independent OLS (largest relative "
        f"difference {mx:.1e})." if ok else
        f"Coefficients do not reproduce: largest relative difference {mx:.2%} on "
        f"{tab.sort_values('rel_diff').term.iloc[-1]}.",
        {"terms": len(tab), "max_rel_diff": mx, "n_obs": int(res.nobs)}, f"relative difference <= {th['replication_tol']}",
        None if ok else "high", "" if ok else "Trace the difference to data, sample or transform before relying on results.",
        table=tab))

    dev = ev.fc.predict(ev.model)
    rows = []
    for sc, mac in ev.scenarios.items():
        mine = forecast_path(ev, ev.coefs, mac)
        d = dev[dev.scenario == sc].set_index("date")
        if d.empty:
            continue
        m = mine.set_index("date").join(d[["change_forecast", "level_forecast"]], how="inner")
        rows.append({"scenario": sc, "quarters": len(m),
                     "max_change_gap": float((m.change - m.change_forecast).abs().max()),
                     "max_level_gap_pct": float(((m.level / m.level_forecast) - 1).abs().max() * 100)})
    tab = pd.DataFrame(rows)
    mx = float(tab.max_change_gap.max())
    ok = mx <= th["forecast_tol"]
    checks.append(Check(
        "R2", "replication", "Scenario forecasts reproduced from the equation and scenario files",
        "pass" if ok else "fail",
        f"The validator's own recursion reproduces the developer's forecasts in all {len(tab)} scenarios "
        f"(largest gap {mx:.1e} log points per quarter)." if ok else
        f"The developer's scenario paths cannot be reproduced from the published equation (gap up to {mx:.2e}).",
        {"scenarios": len(tab), "max_change_gap": mx, "max_level_gap_pct": float(tab.max_level_gap_pct.max())},
        f"gap <= {th['forecast_tol']}", None if ok else "high", table=tab))
    return checks
