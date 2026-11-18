# macroforecast-validation: an independent validation of a forecasting model, with an agent that drafts and a human who signs

This repo is a **second-line model validation** of a real piece of first-line work: the loan-balance forecasting
model built with [`macroforecast`](https://github.com/wuxiaokai-hk/Forecasting-test). It treats that project the
way a model risk team would. It re-reads the raw data, re-implements the transforms, re-estimates the equation and
re-runs the scenarios independently. It then challenges the model on data, theory, performance, stability and
stress, and writes a rated report.

Two things make it different from a notebook of tests:

1. **A ground-truth test.** The example data are simulated from a known equation. So besides asking "is the
   model good?", the validation asks "does the developer's *selection procedure* find the true drivers?". It
   answers by re-simulating 50 fresh datasets and re-running the selection on each one.
2. **An agent with hard limits.** A drafting agent (a local LLM through Ollama, with no API key and nothing
   leaving the machine) writes the narrative of each section. A grounding guard rejects any draft that invents a
   number, skips a warning or uses approval language. Only a named human reviewer can approve sections, set the
   rating and finalize, and every step goes to an audit log.

**[Open the sample report](sample_report/validation_report.html)** (download it and open it in a browser; it is
one self-contained HTML file). It is a *draft* on purpose: no reviewer has signed it.

## What the validation found

The model under review is the developer's recommended model, *Elastic Net first*:
loan growth = AR(1) + house prices + disposable income + 6 COVID dummies.

| | Finding | Severity |
|---|---|---|
| **P1** | The developer's out-of-sample test, the only one that re-selects drivers, rests on **3 forecast origins**. The ranking of about 20 models, and the choice of this one, is not statistically supported. | High |
| **G2** | On 50 datasets simulated from the *same* true equation, the selection procedure finds disposable income in **2%** of them and never finds the mortgage-rate or unemployment channel. Including income here was largely the luck of this sample. | Medium |
| **C4 / S1** | The equation leaves out the interest-rate and labour channels that the developer's own literature review includes. A rate or unemployment shock has **zero** effect on the forecast. | Medium |
| **P4** | Errors in the independent backtest are significantly biased: the model under-forecasts by 0.33% at 4 quarters and 0.74% at 8. | Medium |
| **C3** | 6 COVID dummies, one not significant. They remove the pandemic quarters from what the model learns about stress. | Medium |
| **C6** | The house-price coefficient moves between 0.12 and 0.22 as data are added. | Low |

What holds up: the data lineage reconciles exactly (D1), and the coefficients and all 4 scenario paths reproduce
from raw files to 1e-12 (R1, R2). Residuals are clean (E1–E5). The scenarios rank correctly, with a severe-case
response larger than history (S2, S3). In 71 quarterly re-estimations the equation beats AR(1) by 55% at 8
quarters (P2), though that is an upper bound because the drivers were chosen on the full sample. The ground-truth
test also shows that leaving out the rate and labour channels costs little *in this scenario set*: the severe
impact is understated by about 5% (G3).

Proposed rating (rule-based): **Needs improvement**. Only a reviewer sets the final rating.

## How it works

```
 developer's run ─┐                     ┌─> evidence.json (29 tests: status, severity, numbers)
 raw data files ──┼─> independent checks┤
 true equation ───┘   (code, no LLM)    └─> charts
                                              │
                         drafting agent  <────┘   reads one section's results, writes one paragraph
                               │
                         grounding guard          every number must come from the evidence; every
                               │                  warning must be mentioned; no rating or approval words;
                               │                  1 retry with the objections, then the template writer
                               ▼
                      draft report (DRAFT banner)
                               │
                 human reviewer: show / edit / approve / rate / finalize
                               │   approvals are tied to an evidence fingerprint: if any number changes
                               ▼   on a re-run, the approval is voided and the section returns to draft
                      final report + audit_log.jsonl
```

| Section | Tests |
|---|---|
| 2 Data integrity | D1 lineage: raw balances and events rebuilt independently · D2 completeness · D3 outliers · D4 ADF and KPSS · D5 scenario extrapolation |
| 3 Conceptual soundness | C1 signs vs theory · C2 HAC significance and VIF · C3 dummies · C4 channels missing from the literature core · C5 circular credit drivers · C6 expanding-window coefficient stability |
| 4 Replication | R1 coefficients re-estimated from raw data · R2 scenario paths re-computed from the equation |
| 5 Performance | P1 depth of the developer's OOS test · P2 independent rolling-origin backtest vs AR(1) and drift with Diebold-Mariano (HLN) tests · P3 2008-09 stress period · P4 bias |
| 6 Diagnostics | E1 Ljung-Box · E2 Breusch-Pagan and ARCH · E3 Jarque-Bera · E4 RESET · E5 CUSUM and Andrews sup-F |
| 7 Sensitivity | S1 one-SD driver shocks, including channels the model omits · S2 scenario ordering · S3 stress severity vs history |
| 8 Ground truth | G1 true drivers recovered · G2 Monte Carlo recovery rates · G3 cost of the omitted drivers under stress |

Thresholds, the model profile and the risk tier are in [`validation.yaml`](validation.yaml). The method for each
test is explained in [`docs/methodology.md`](docs/methodology.md).

## Run it

```bash
git clone https://github.com/wuxiaokai-hk/Forecasting-test ../Forecasting-test   # the model under review
pip install -e ../Forecasting-test
pip install -r requirements.txt

python examples/make_run.py            # the developer's run, about 2 minutes -> runs/loan_balance
python tools/monte_carlo.py --reps 50  # optional, about 15 min; the result is already in evidence/
python -m mfvalidate run --out report  # 29 tests, charts, agent drafts -> report/validation_report.html
```

For LLM-drafted sections, start [Ollama](https://ollama.com) (`ollama pull llama3.1:8b`). Without it, the
deterministic template writer drafts every section, so the report never depends on a model being available.

The review loop:

```bash
python -m mfvalidate status   --out report
python -m mfvalidate show     --out report --section performance
python -m mfvalidate edit     --out report --section performance --file my_text.md --reviewer "Your Name"
python -m mfvalidate approve  --out report --section all --reviewer "Your Name" --comment "Reviewed"
python -m mfvalidate rate     --out report --rating "Needs improvement" --reviewer "Your Name" \
                              --rationale "P1 must be remediated before the model is used for capital planning"
python -m mfvalidate finalize --out report --reviewer "Your Name"
```

`finalize` refuses to run until every section is approved, a rating is set, and no number has changed since
the approvals.

## Tests

```bash
python -m pytest -q        # 27 tests
```

They cover: the validator's transforms against the developer's (all 9), the Diebold-Mariano test, event removal,
the rating rules, and the guard (it accepts grounded text and rejects invented numbers, missing warnings and
approval language). The LLM path is tested against a fake local model: a rejected draft is retried, and a model
that keeps hallucinating falls back to the template. The sign-off rules are tested too: no approval on changed
evidence, no finalize without approvals and a rating, and new evidence voids old approvals. One end-to-end test
runs the full flow.

## Layout

```
mfvalidate/
  evidence.py        loads the run; rebuilds target, transforms and regressors from raw files
  checks/            data, soundness, replication, performance, diagnostics, sensitivity, ground_truth
  findings.py        findings, rule-based rating proposal, evidence fingerprint
  agent.py           drafting agent (Ollama or template) and the grounding guard
  signoff.py         human approvals, rating, finalize, audit log
  report.py          self-contained HTML report
  cli.py             mfvalidate run | status | show | edit | approve | rate | finalize
tools/monte_carlo.py re-simulates the true process and re-runs driver selection
evidence/            Monte Carlo results used by G2
sample_report/       the draft report produced by `mfvalidate run`
```
