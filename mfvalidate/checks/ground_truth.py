"""Section 8: ground-truth test. Only possible with simulated data, where the true equation is known.

This is the strongest test a validator can run on a selection procedure: does it find the drivers that really
generated the data, how often, and what does missing one cost in a stress scenario?
"""
import numpy as np
import pandas as pd

from . import Check
from .. import charts
from ..evidence import parse_feature
from .replication import forecast_path


def _truth(ev):
    gt = ev.cfg["ground_truth"]
    return gt, dict(gt["drivers"])


def run(ev, out_dir):
    gt, true = _truth(ev)
    if not gt.get("enabled"):
        return []
    checks = []
    found = set(ev.drivers)
    true_codes = {parse_feature(t)[0]: t for t in true}
    rows = []
    for t, b in true.items():
        code = parse_feature(t)[0]
        near = [d for d in found if parse_feature(d)[0] == code and d != t]
        rows.append({"true_driver": t, "true_coef": b, "selected": t in found,
                     "selected_coef": ev.coefs.get(t, np.nan), "other_form_selected": ", ".join(near)})
    tab = pd.DataFrame(rows)
    false_pos = [d for d in found if parse_feature(d)[0] not in true_codes]
    n_hit = int(tab.selected.sum())
    status = "pass" if n_hit == len(tab) and not false_pos else "warn"
    checks.append(Check(
        "G1", "ground_truth", "True drivers recovered on this dataset", status,
        f"The model contains {n_hit} of the {len(tab)} true drivers"
        + (f" ({', '.join(tab[tab.selected].true_driver)})" if n_hit else "")
        + (f" and misses {', '.join(tab[~tab.selected].true_driver)}" if n_hit < len(tab) else "")
        + (f"; it also includes {len(false_pos)} driver(s) not in the true equation." if false_pos else
           "; it includes no false drivers."),
        {"true_drivers": len(tab), "recovered": n_hit, "false_drivers": len(false_pos)},
        "all true drivers, no false ones", None if status == "pass" else "medium", table=tab))

    # G2 Monte Carlo: how often does the selection procedure recover the truth on fresh simulated datasets?
    path = gt.get("monte_carlo_results")
    try:
        mc = pd.read_csv(path)
    except (FileNotFoundError, ValueError, TypeError):
        mc = None
    if mc is not None:
        reps = mc.seed.nunique()
        rows = []
        for m, g in mc.groupby("model"):
            for t in true:
                code = parse_feature(t)[0]
                exact = g[g.term == t].seed.nunique() / reps
                anyform = g[g.term.map(lambda s: (parse_feature(s) or ("",))[0] == code)].seed.nunique() / reps
                c = g[g.term == t].coef
                rows.append({"approach": m, "true_driver": t, "exact_rate": exact, "variable_rate": anyform,
                             "mean_coef_when_found": c.mean() if len(c) else np.nan, "true_coef": true[t]})
            drv = g[g.term.map(lambda s: parse_feature(s) is not None)]
            fp = drv[~drv.term.map(lambda s: parse_feature(s)[0] in true_codes)].groupby("seed").size()
            rows.append({"approach": m, "true_driver": "(false drivers per run)",
                         "exact_rate": fp.reindex(g.seed.unique(), fill_value=0).mean(), "variable_rate": np.nan,
                         "mean_coef_when_found": np.nan, "true_coef": np.nan})
            ar = g[g.term == "y_lag1"].coef
            rows.append({"approach": m, "true_driver": "y_lag1 (AR term)", "exact_rate": len(ar) / reps,
                         "variable_rate": np.nan, "mean_coef_when_found": ar.mean(), "true_coef": gt["ar1"]})
        mct = pd.DataFrame(rows)
        rec = mct[mct.true_driver.isin(true)]
        fig, ax = charts.plt.subplots(figsize=(6.2, 2.9))
        apps = list(rec.approach.unique())
        w = 0.8 / len(apps)
        for i, a in enumerate(apps):
            r = rec[rec.approach == a]
            x = np.arange(len(r)) + i * w
            ax.bar(x, r.exact_rate * 100, width=w * 0.92, color=charts.SERIES[i], label=a)
        ax.set_xticks(np.arange(len(true)) + w * (len(apps) - 1) / 2)
        ax.set_xticklabels([t.replace("|", " ") for t in true], fontsize=8)
        ax.set_ylabel("% of simulated datasets")
        ax.set_ylim(0, 100)
        ax.set_title(f"How often each true driver is selected ({reps} simulated datasets)")
        ax.legend(loc="upper right")
        chart = charts.save(fig, out_dir, "g2_monte_carlo.png")
        worst = rec.sort_values("exact_rate").iloc[0]
        status = "pass" if rec.exact_rate.min() >= 0.8 else "warn"
        checks.append(Check(
            "G2", "ground_truth", "Selection procedure recovers the truth on fresh simulated datasets", status,
            f"Across {reps} datasets simulated from the true equation, " +
            "; ".join(f"{a} picks a given true driver in {rec[rec.approach == a].exact_rate.mean():.0%} of datasets on "
                      f"average and adds {mct[(mct.approach == a) & (mct.true_driver == '(false drivers per run)')].exact_rate.iloc[0]:.1f} "
                      "false drivers per run" for a in apps) +
            ". Found in under 20% of datasets even by the better approach: " +
            (", ".join(t for t, r in rec.groupby("true_driver").exact_rate.max().items() if r < 0.2) or "none") +
            ". With about 95 quarters of data, statistical selection cannot be relied on to find these channels." +
            "".join(f" {t} is in the model under review, but {ev.model} finds it in only "
                    f"{rec[(rec.approach == ev.model) & (rec.true_driver == t)].exact_rate.iloc[0]:.0%} of fresh datasets: "
                    "its inclusion here is partly luck of the sample."
                    for t in ev.drivers if t in true and ev.model in apps and
                    rec[(rec.approach == ev.model) & (rec.true_driver == t)].exact_rate.iloc[0] < 0.5),
            {"replications": reps, "min_exact_rate": float(worst.exact_rate),
             "mean_exact_rate": {a: float(rec[rec.approach == a].exact_rate.mean()) for a in apps}},
            "each true driver found in >= 80% of datasets", None if status == "pass" else "medium",
            "" if status == "pass" else "Do not rely on statistical selection alone for low-signal channels; "
                                        "impose theory-based channels with constrained coefficients.",
            table=mct, chart=chart))

    # G3 cost of the omitted drivers in the severe scenario (true equation vs model)
    tc = pd.Series({"const": gt["const"], "y_lag1": gt["ar1"], **true})
    order = ev.cfg["inputs"].get("scenario_order") or list(ev.scenarios)
    base, sev = ev.base_name, order[-1]
    p_true = {s: forecast_path(ev, tc, ev.scenarios[s], horizon=8) for s in (base, sev)}
    p_mod = {s: forecast_path(ev, ev.coefs, ev.scenarios[s], horizon=8) for s in (base, sev)}
    gap_true = (np.log(p_true[base].level.iloc[-1]) - np.log(p_true[sev].level.iloc[-1])) * 100
    gap_mod = (np.log(p_mod[base].level.iloc[-1]) - np.log(p_mod[sev].level.iloc[-1])) * 100
    under = (1 - gap_mod / gap_true) * 100
    # contribution of the omitted drivers to history: explains bias in the backtest (P4)
    omitted = [t for t in true if t not in found]
    X = ev.design(terms=omitted).loc[pd.Period(ev.cfg["independent_backtest"]["first_origin"], freq="Q"):ev.hist_end]
    contrib = float((X * pd.Series({t: true[t] for t in omitted})).sum(axis=1).mean() * 100) if omitted else 0.0
    status = "pass" if abs(under) < 20 else "warn"
    checks.append(Check(
        "G3", "ground_truth", "Cost of the omitted drivers in the severe scenario", status,
        f"Under the true equation {sev} ends 8 quarters {gap_true:.1f} pp below {base}; the model gives "
        f"{gap_mod:.1f} pp, so it {'understates' if under > 0 else 'overstates'} the stress impact by "
        f"{abs(under):.0f}%. Over the backtest period the omitted drivers contributed {contrib:+.3f} pp to quarterly "
        "growth on average, " + ("so leaving them out pushes forecasts down, consistent with the under-forecasting in P4."
                                 if contrib > 0.01 else "too little to explain the forecast bias in P4."),
        {"true_gap_pp": gap_true, "model_gap_pp": gap_mod, "understatement_pct": under,
         "omitted_contribution_pp_per_q": contrib, "omitted": omitted},
        "stress gap within 20% of the truth", None if status == "pass" else "high"))
    return checks
