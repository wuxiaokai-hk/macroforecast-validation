"""Section 7: sensitivity to drivers and the behaviour of the scenario forecasts."""
import numpy as np
import pandas as pd

from . import Check
from .. import charts
from ..evidence import parse_feature, feature
from .replication import forecast_path

H = 8


def shock_macro(macro, code, tr, size, start):
    """Raise the transformed series by `size` in every quarter from `start` on, by editing the raw level."""
    m = macro.copy()
    x = m[code].astype(float)
    k = pd.Series(np.where(x.index >= start, 1.0, 0.0), index=x.index).cumsum() * size
    if tr == "dlog":
        m[code] = x * np.exp(k)
    elif tr == "diff":
        m[code] = x + k
    else:
        raise NotImplementedError(tr)
    return m


def literature_probes(ev):
    lit = ev.run_file("comparison", "3_drivers.csv")
    if lit is None:
        return []
    probes = []
    used = {parse_feature(d)[0] for d in ev.drivers}
    for t in lit[lit.model.str.startswith("Literature: four-channel")].term:
        pf = parse_feature(str(t))
        if pf and pf[0] not in used and pf[1] in ("dlog", "diff"):
            probes.append(t)
    return probes


def run(ev, out_dir):
    th = ev.cfg["thresholds"]
    base = ev.scenarios[ev.base_name]
    start = ev.hist_end + 1
    b0 = forecast_path(ev, ev.coefs, base, horizon=H)
    checks = []

    # S1 one-SD shocks to each driver, and to the literature channels the model leaves out
    rows = []
    for d in ev.drivers + literature_probes(ev):
        code, tr, lag = parse_feature(d)
        sd = feature(ev.macro_hist, f"{code}|{tr}|L0").loc[ev.est_start:ev.hist_end].std()
        p = forecast_path(ev, ev.coefs, shock_macro(base, code, tr, sd, start), horizon=H)
        impact = (p.level.iloc[-1] / b0.level.iloc[-1] - 1) * 100
        exp = int(ev.meta.loc[code, "expected_sign"])
        rows.append({"shock": f"+1 SD {code} ({tr}) each quarter", "in_model": d in ev.drivers,
                     "description": ev.meta.loc[code, "description"], "sd": sd,
                     "impact_8q_pct": impact, "expected_sign": exp,
                     "direction_ok": (np.sign(impact) == exp) if d in ev.drivers else None})
    tab = pd.DataFrame(rows)
    bad = tab[tab.in_model & (tab.direction_ok == False)]  # noqa: E712
    deaf = tab[~tab.in_model]
    status = "fail" if len(bad) else ("warn" if len(deaf) else "pass")
    finding = ("Each driver moves the 8-quarter balance in the expected direction: " +
               "; ".join(f"{r.shock} {r.impact_8q_pct:+.2f}%" for _, r in tab[tab.in_model].iterrows()) + ".")
    if len(bad):
        finding = f"{len(bad)} driver shocks move the balance the wrong way."
    if len(deaf):
        finding += (" Shocks to " + " and ".join(deaf.description) +
                    " have no effect at all, although the developer's literature core treats them as channels.")
    fig, ax = charts.plt.subplots(figsize=(6.2, 0.45 * len(tab) + 0.9))
    labels = [s.replace("+1 SD ", "").replace(" each quarter", "") for s in tab.shock]
    ax.barh(labels, tab.impact_8q_pct, color=[charts.SERIES[0] if m else charts.GRID for m in tab.in_model],
            height=0.5)
    ax.axvline(0, color=charts.INK2, lw=0.8)
    for i, v in enumerate(tab.impact_8q_pct):
        ax.annotate(f"{v:+.2f}%" if abs(v) > 1e-9 else "0 (not in model)", (v, i), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center", ha="left" if v >= 0 else "right", fontsize=8,
                    color=charts.INK2)
    ax.invert_yaxis()
    ax.set_xlabel("balance after 8 quarters vs baseline, %")
    ax.set_title("Response to a one-SD shock in every forecast quarter")
    chart = charts.save(fig, out_dir, "s1_shocks.png")
    checks.append(Check("S1", "sensitivity", "Response to one-SD shocks in each driver", status, finding,
                        {"shocks": len(tab), "wrong_direction": len(bad), "no_response_channels": len(deaf)},
                        "expected direction; literature channels respond",
                        None if status == "pass" else ("high" if len(bad) else "medium"), table=tab, chart=chart))

    # S2 scenario ranking
    order = ev.cfg["inputs"].get("scenario_order") or list(ev.scenarios)
    paths = {sc: forecast_path(ev, ev.coefs, ev.scenarios[sc], horizon=12) for sc in order}
    end = {sc: paths[sc].level.iloc[H - 1] for sc in order}
    ranked = all(end[a] >= end[b] - 1e-9 for a, b in zip(order, order[1:]))
    spread = (end[order[0]] / end[order[-1]] - 1) * 100
    fig, ax = charts.plt.subplots(figsize=(6.2, 3.0))
    hist = ev.level.loc[ev.hist_end - 11:ev.hist_end]
    ax.plot(hist.index.to_timestamp(), hist / 1e3, color=charts.INK2, lw=1.5)
    for i, sc in enumerate(order):
        p = paths[sc]
        x = pd.PeriodIndex(p.date, freq="Q").to_timestamp()
        ax.plot(x, p.level / 1e3, color=charts.SERIES[i])
        charts.label_end(ax, x[-1], p.level.iloc[-1] / 1e3, sc, charts.SERIES[i])
    ax.set_ylabel("balance, $ billions")
    ax.set_title("Scenario forecasts of the model under review")
    ax.margins(x=0.18)
    chart = charts.save(fig, out_dir, "s2_scenarios.png")
    checks.append(Check(
        "S2", "sensitivity", "Scenario forecasts rank in the expected order", "pass" if ranked else "fail",
        ("After 8 quarters the balances rank " + " > ".join(order) + f"; the spread between {order[0]} and "
         f"{order[-1]} is {spread:.1f}%.") if ranked else
        "Scenario balances after 8 quarters are not in the expected order: " +
        ", ".join(f"{sc} {end[sc]:,.0f}" for sc in order) + ".",
        {"balance_8q": {k: round(v, 1) for k, v in end.items()}, "spread_pct": spread},
        " >= ".join(order), None if ranked else "high", chart=chart))

    # S3 severity of the stress response relative to history
    g = ev.growth.loc[ev.est_start:ev.hist_end].dropna()
    roll = g.rolling(H).sum()
    hist_drop = (roll.mean() - roll.min()) * 100
    sev = order[-1]
    scen_drop = (np.log(end[order[1]] if len(order) > 2 else end[order[0]]) - np.log(end[sev])) * 100
    ratio = scen_drop / hist_drop
    ok = ratio >= th["stress_drawdown_ratio"]
    checks.append(Check(
        "S3", "sensitivity", "Severity of the stress response compared with history", "pass" if ok else "warn",
        f"{sev} lowers 8-quarter growth by {scen_drop:.1f} pp against Baseline, {ratio:.0%} of the gap between the "
        f"average and the worst 8-quarter growth in history ({hist_drop:.1f} pp)." +
        ("" if ok else " The stress response looks muted for a severe scenario; the omitted rate and labour "
                       "channels (C4, S1) are a likely cause."),
        {"scenario_drop_pp": scen_drop, "historical_drop_pp": hist_drop, "ratio": ratio},
        f"ratio >= {th['stress_drawdown_ratio']:.0%}", None if ok else "medium"))
    return checks
