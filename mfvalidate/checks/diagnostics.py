"""Section 6: residual diagnostics, recomputed by the validator."""
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (acorr_ljungbox, het_breuschpagan, het_arch, linear_reset,
                                          breaks_cusumolsresid)
from statsmodels.stats.stattools import jarque_bera

from . import Check
from .. import charts


def _chow_sup(y, X, trim=0.15):
    """Largest Chow F over candidate break dates in the middle of the sample (Andrews sup-F statistic)."""
    n, k = X.shape
    rss = sm.OLS(y, X).fit().ssr
    best = (0.0, None)
    for i in range(int(n * trim), int(n * (1 - trim))):
        r1 = sm.OLS(y.iloc[:i], X.iloc[:i]).fit().ssr
        r2 = sm.OLS(y.iloc[i:], X.iloc[i:]).fit().ssr
        f = ((rss - r1 - r2) / k) / ((r1 + r2) / (n - 2 * k))
        if f > best[0]:
            best = (f, str(y.index[i]))
    return best


# Andrews (1993) 5% critical values for sup-F (Wald/q) with 15% trimming, by number of parameters q
ANDREWS_5 = {1: 8.85, 2: 5.91, 3: 4.71, 4: 4.09, 5: 3.66, 6: 3.37, 7: 3.15, 8: 2.98, 9: 2.84, 10: 2.71}


def run(ev, out_dir):
    y, X = ev.estimation_frame()
    res = sm.OLS(y, X).fit()
    e = res.resid
    checks = []

    lb = acorr_ljungbox(e, lags=[4, 8], return_df=True)
    p4, p8 = lb.lb_pvalue.iloc[0], lb.lb_pvalue.iloc[1]
    ok = min(p4, p8) > 0.05
    checks.append(Check("E1", "diagnostics", "No residual autocorrelation (Ljung-Box)", "pass" if ok else "warn",
                        f"Ljung-Box p = {p4:.3f} at 4 lags and {p8:.3f} at 8 lags." +
                        ("" if ok else " Residual autocorrelation means momentum or a driver lag is missing."),
                        {"lb_p4": p4, "lb_p8": p8}, "p > 0.05", None if ok else "medium"))

    bp = het_breuschpagan(e, X)[1]
    arch = het_arch(e, nlags=4)[1]
    ok = min(bp, arch) > 0.05
    checks.append(Check("E2", "diagnostics", "Constant error variance (Breusch-Pagan, ARCH)", "pass" if ok else "warn",
                        f"Breusch-Pagan p = {bp:.3f}; ARCH(4) p = {arch:.3f}." +
                        ("" if ok else " Standard errors and forecast bands need HAC or a variance model."),
                        {"bp_p": bp, "arch_p": arch}, "p > 0.05", None if ok else "low"))

    jb = jarque_bera(e)
    ok = jb[1] > 0.05
    checks.append(Check("E3", "diagnostics", "Normal residuals (Jarque-Bera)", "pass" if ok else "warn",
                        f"Jarque-Bera p = {jb[1]:.3f} (skew {jb[2]:.2f}, kurtosis {jb[3]:.2f})." +
                        ("" if ok else " Fat tails: Gaussian forecast intervals understate tail risk."),
                        {"jb_p": jb[1], "skew": jb[2], "kurtosis": jb[3]}, "p > 0.05", None if ok else "low"))

    reset = linear_reset(res, power=2, use_f=True).pvalue
    ok = reset > 0.05
    checks.append(Check("E4", "diagnostics", "Functional form (Ramsey RESET)", "pass" if ok else "warn",
                        f"RESET p = {reset:.3f}." + ("" if ok else " A non-linearity or interaction is missing."),
                        {"reset_p": float(reset)}, "p > 0.05", None if ok else "medium"))

    cus_p = breaks_cusumolsresid(e.values, ddof=X.shape[1])[1]
    core = X.drop(columns=ev.dummies)
    supf, when = _chow_sup(y, core)
    cv = ANDREWS_5.get(core.shape[1], 2.7)
    ok = cus_p > 0.05 and supf < cv
    checks.append(Check("E5", "diagnostics", "Structural stability (CUSUM, Andrews sup-F)", "pass" if ok else "warn",
                        f"CUSUM p = {cus_p:.3f}; largest Chow F = {supf:.2f} at {when} against a 5% critical value of "
                        f"{cv:.2f}." + ("" if ok else " There is evidence of a break; consider a shorter or split sample."),
                        {"cusum_p": cus_p, "sup_f": supf, "sup_f_date": when, "crit_5pct": cv},
                        "CUSUM p > 0.05 and sup-F < 5% critical value", None if ok else "medium"))

    fig, ax = charts.plt.subplots(figsize=(6.2, 2.6))
    x = e.index.to_timestamp()
    ax.bar(x, e * 100, width=60, color=charts.SERIES[0])
    sd = e.std() * 100
    for s in (-2, 2):
        ax.axhline(s * sd, color=charts.INK2, lw=0.8, ls="--")
    ax.set_ylabel("residual, pp of q/q growth")
    ax.set_title("Residuals of the equation (dashed: +/- 2 SD)")
    checks[0].chart = charts.save(fig, out_dir, "e_residuals.png")
    return checks
