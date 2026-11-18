"""Section 3: conceptual soundness of the chosen equation."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import Check
from .. import charts
from ..evidence import parse_feature

CIRCULAR_THEMES = {"Money & credit"}


def _fit(y, X):
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})


def run(ev, out_dir):
    th = ev.cfg["thresholds"]
    y, X = ev.estimation_frame()
    res = _fit(y, X)
    checks = []

    # C1 signs against the economic prior in the metadata
    rows = []
    for d in ev.drivers:
        code, tr, lag = parse_feature(d)
        exp = int(ev.meta.loc[code, "expected_sign"]) if code in ev.meta.index else 0
        c = res.params[d]
        rows.append({"driver": d, "description": ev.meta.loc[code, "description"] if code in ev.meta.index else "",
                     "theme": ev.meta.loc[code, "theme"] if code in ev.meta.index else "",
                     "coef": c, "expected_sign": exp, "sign_ok": exp == 0 or np.sign(c) == exp})
    signs = pd.DataFrame(rows)
    wrong = signs[~signs.sign_ok]
    checks.append(Check(
        "C1", "soundness", "Driver signs agree with economic theory", "pass" if wrong.empty else "fail",
        f"All {len(signs)} drivers have the expected sign." if wrong.empty else
        f"{len(wrong)} drivers have the wrong sign: {', '.join(wrong.driver)}.",
        {"drivers": len(signs), "wrong_sign": len(wrong)}, "sign = expected_sign in metadata",
        None if wrong.empty else "high", "" if wrong.empty else "Remove or justify wrong-signed drivers.", table=signs))

    # C2 significance and multicollinearity
    core = [t for t in X.columns if t not in ev.dummies and t != "const"]
    vif = {t: variance_inflation_factor(X[core + ["const"]].values, i) for i, t in enumerate(core)} if len(core) > 1 \
        else {t: 1.0 for t in core}
    tab = pd.DataFrame({"term": core, "coef": res.params[core].values, "se_hac": res.bse[core].values,
                        "p_value": res.pvalues[core].values, "vif": [vif[t] for t in core]})
    weak = tab[(tab.p_value > th["max_driver_p"]) | (tab.vif > th["max_vif"])]
    checks.append(Check(
        "C2", "soundness", "Significance (HAC) and multicollinearity of drivers", "pass" if weak.empty else "warn",
        f"Every driver is significant at {th['max_driver_p']:.0%} with HAC errors and the largest VIF is "
        f"{tab.vif.max():.2f}." if weak.empty else
        f"{len(weak)} terms are weak or collinear: {', '.join(weak.term)}.",
        {"max_p": float(tab.p_value.max()), "max_vif": float(tab.vif.max())},
        f"p <= {th['max_driver_p']}, VIF <= {th['max_vif']}", None if weak.empty else "medium", table=tab))

    # C3 dummy variables: how many, and whether the data support them
    dum = pd.DataFrame({"dummy": ev.dummies, "coef": res.params[ev.dummies].values,
                        "p_value": res.pvalues[ev.dummies].values}) if ev.dummies else pd.DataFrame()
    n_weak = int((dum.p_value > th["dummy_p"]).sum()) if len(dum) else 0
    r2_no = sm.OLS(y, X.drop(columns=ev.dummies)).fit().rsquared_adj if ev.dummies else res.rsquared_adj
    status = "pass" if n_weak == 0 else "warn"
    checks.append(Check(
        "C3", "soundness", "Dummy variables are few and supported by the data", status,
        (f"The equation uses {len(ev.dummies)} quarterly dummies "
         f"({n_weak} not significant at {th['dummy_p']:.0%}); adjusted R2 is {res.rsquared_adj:.3f} with them and "
         f"{r2_no:.3f} without. The dummied quarters get zero residuals, so the COVID-period shock is excluded "
         "from what the model learns about stress."),
        {"n_dummies": len(ev.dummies), "not_significant": n_weak, "adj_r2": res.rsquared_adj,
         "adj_r2_without_dummies": r2_no}, f"dummy p <= {th['dummy_p']}",
        None if status == "pass" else "medium",
        "" if status == "pass" else "Keep only dummies with a documented event and a significant effect.",
        table=dum if len(dum) else None))

    # C4 theory coverage: channels in the developer's own literature core that the equation leaves out
    lit = ev.run_file("comparison", "3_drivers.csv")
    lit_themes, missing = set(), set()
    if lit is not None:
        core_rows = lit[lit.model.str.startswith("Literature: four-channel")]
        for t in core_rows.term:
            pf = parse_feature(str(t))
            if pf and pf[0] in ev.meta.index:
                lit_themes.add(ev.meta.loc[pf[0], "theme"])
        model_themes = {ev.meta.loc[parse_feature(d)[0], "theme"] for d in ev.drivers}
        missing = lit_themes - model_themes
    status = "info" if not lit_themes else ("pass" if not missing else "warn")
    checks.append(Check(
        "C4", "soundness", "Coverage of the channels in the developer's literature core", status,
        "No literature core in the run." if status == "info" else
        ("The equation covers every channel of the literature core." if not missing else
         f"The equation leaves out the {' and '.join(sorted(missing))} channel{'s' if len(missing) > 1 else ''} that the developer's own literature "
         "core includes, so scenarios that move them (rate shocks, job losses) are transmitted only through "
         "the remaining drivers."),
        {"literature_themes": sorted(lit_themes), "missing_themes": sorted(missing)},
        "every literature channel represented or its omission justified",
        None if status in ("pass", "info") else "medium",
        "" if status in ("pass", "info") else "Justify the omission, or add the channel with a constrained coefficient."))

    # C5 circularity: a credit aggregate used to forecast a credit balance
    circ = [d for d in ev.drivers if ev.meta.loc[parse_feature(d)[0], "theme"] in CIRCULAR_THEMES]
    checks.append(Check(
        "C5", "soundness", "No circular drivers (credit aggregates forecasting credit)",
        "pass" if not circ else "warn",
        "No driver is itself a money or credit aggregate." if not circ else
        f"{', '.join(circ)} is a credit/money aggregate: it largely restates the target and passes the scenario "
        "provider's credit assumption straight into the forecast.",
        {"circular_drivers": circ}, "no money & credit drivers", None if not circ else "medium"))

    # C6 parameter stability: expanding-window re-estimation of the fixed equation
    first = pd.Period(ev.cfg["independent_backtest"]["first_origin"], freq="Q")
    rows = []
    for end in pd.period_range(first, ev.hist_end, freq="Q"):
        yy, XX = y.loc[:end], X.loc[:end]
        keep = [c for c in XX.columns if XX[c].abs().sum() > 0]
        r = _fit(yy, XX[keep])
        for d in ev.drivers + (["y_lag1"] if ev.has_ar else []):
            rows.append({"end": str(end), "term": d, "coef": r.params[d], "se": r.bse[d]})
    paths = pd.DataFrame(rows)
    summ = []
    for d, g in paths.groupby("term", sort=False):
        full = res.params[d]
        summ.append({"term": d, "full_sample": full, "min": g.coef.min(), "max": g.coef.max(),
                     "range_vs_full": (g.coef.max() - g.coef.min()) / abs(full),
                     "sign_changes": int((np.sign(g.coef) != np.sign(full)).sum()),
                     "outside_full_2se": float(((g.coef - full).abs() > 2 * res.bse[d]).mean())})
    summ = pd.DataFrame(summ)
    unstable = summ[(summ.sign_changes > 0) | (summ.outside_full_2se > 0.25)]
    fig, axes = charts.plt.subplots(1, len(summ), figsize=(3.2 * len(summ), 2.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (d, g) in zip(axes, paths.groupby("term", sort=False)):
        x = pd.PeriodIndex(g.end, freq="Q").to_timestamp()
        ax.fill_between(x, g.coef - 2 * g.se, g.coef + 2 * g.se, color=charts.SERIES[0], alpha=0.15, lw=0)
        ax.plot(x, g.coef, color=charts.SERIES[0])
        ax.axhline(0, color=charts.INK2, lw=0.8)
        ax.set_title(d, fontsize=9)
        charts.year_axis(ax, 6)
    chart = charts.save(fig, out_dir, "c6_coefficient_paths.png")
    checks.append(Check(
        "C6", "soundness", "Coefficient stability (expanding-window re-estimation)",
        "pass" if unstable.empty else "warn",
        f"Re-estimated every quarter from {first} on, no coefficient changes sign and none spends more than a "
        "quarter of the time outside two standard errors of its full-sample value." if unstable.empty else
        "; ".join(f"{r.term} ranges {r['min']:.3f} to {r['max']:.3f} across windows (full sample {r.full_sample:.3f}) and is "
                  f"outside +/- 2 SE of the full-sample value in {r.outside_full_2se:.0%} of them"
                  for _, r in unstable.iterrows()) + ". The effect is less precisely pinned down than the full-sample standard error suggests.",
        {"windows": int(paths.end.nunique()), "unstable_terms": list(unstable.term)},
        "no sign change; <= 25% of windows outside +/- 2 SE",
        None if unstable.empty else ("medium" if (unstable.sign_changes > 0).any() else "low"),
        table=summ, chart=chart))
    return checks
