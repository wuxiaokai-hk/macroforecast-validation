# Methodology

The structure follows the usual model-risk validation scope (conceptual soundness, data, implementation,
outcomes analysis, ongoing use) as described in supervisory guidance such as OSFI E-23 and the US SR 11-7.
Every number in the report comes from code in `mfvalidate/checks/`. The agent writes only words.

## Principles

- **Independence.** The validator does not reuse the developer's prepared data or helper functions. It reads
  the raw balance, events, metadata and scenario files and re-implements each transform. A unit test confirms
  that the re-implementation agrees with the developer's code. The developer's package is used only to load
  the finished run (coefficients, forecasts and the developer's own OOS files).
- **Challenge, not repetition.** The developer's tool already reports diagnostics and out-of-sample tests. The
  validation recomputes the diagnostics, judges whether the developer's evidence is deep enough (P1), and adds
  tests the developer did not run (P2–P4, C4–C6, S1, section 8).
- **Severity reflects use.** The model feeds capital planning under stress scenarios (risk tier High). So
  findings about stress behaviour and model choice weigh more than in-sample fit.

## Section notes

**D1 Lineage.** Organic growth is ln((L_t − event_t)/L_{t−1}), with the events file applied to the target only.
It is compared quarter by quarter with the developer's `fitted/data/target.csv`.

**D5 Extrapolation.** For each scenario and driver, this is the share of the 12 forecast quarters where the
transformed driver lies outside its historical min–max range.

**C4 Literature channels.** These are the themes (from the metadata) of the variables in the developer's
"Literature: four-channel core" model, compared with the themes of the drivers in the model under review.

**C6 Stability.** The fixed equation is re-estimated on expanding samples ending each quarter from 2008Q1. A
term is flagged if it changes sign, or if it lies outside ±2 HAC standard errors of its full-sample value in
more than 25% of the windows.

**P1 Depth of OOS evidence.** The developer's OOS test is the only one that re-runs driver selection at each
origin, which makes it the honest test of a selection procedure. Its default step of 8 quarters leaves very few
origins. The threshold is 8.

**P2 Independent backtest.** At every quarterly origin from 2008Q1, the equation (with its drivers fixed) and an
AR(1) are re-estimated on data up to the origin. Each is forecast recursively, conditional on the *actual* driver
path; a random walk with drift (the mean of the last 8 quarters) is added. The error is the cumulative log-growth
error × 100, which is about the % balance error. Windows that touch 2020Q1–2021Q2 are excluded. The
Diebold-Mariano test uses squared-error loss, an (h−1)-lag Newey-West variance and the Harvey-Leybourne-Newbold
correction. Because the drivers were chosen on the full sample, this result is an *upper bound*.

**P4 Bias.** This is a HAC t-test that the mean error equals zero, at h = 4 and h = 8.

**E5 Breaks.** The test is the CUSUM of OLS residuals, plus the largest Chow F over break dates in the middle
70% of the sample (excluding the dummies). That F is compared with Andrews' (1993) 5% critical value.

**S1 Shocks.** Each driver is raised by one historical standard deviation of its transformed value in *every*
forecast quarter. The raw level is edited so that the shock passes through the same transforms. The result is
the 8-quarter balance impact against the baseline path. Variables in the developer's literature core that the
model omits are shocked the same way, to show the model does not respond to them.

**S3 Severity.** This compares the severe scenario's 8-quarter growth shortfall against Baseline with the gap
between the average and the worst rolling 8-quarter growth in history.

**G2 Monte Carlo.** The developer's simulator is run unchanged except for the random seed (seeds 1001–1050).
The developer's driver selection (both approaches, `fast` preset, no literature workbook) is re-run on each
dataset. An exact match on variable, transform and lag counts as recovery. "False drivers" are drivers whose
variable is not in the true equation (close proxies such as Toronto house prices count as false).

**G3 Cost of omission.** The true equation (no dummies) and the model are each run through Baseline and the
severe scenario. Their 8-quarter Baseline-minus-severe gaps are compared.

## Rating rule (a proposal only)

| Proposed rating | When |
|---|---|
| Unsatisfactory | a high-severity failure in data or replication (the results cannot be relied on), or 3 or more high findings |
| Needs improvement | at least one high finding, or 3 or more medium findings |
| Satisfactory | otherwise |

The reviewer sets the final rating with `mfvalidate rate` and gives a rationale. If it differs from the
proposal, it is recorded as an override.

## Agent controls

| Control | Where |
|---|---|
| The agent sees one section's results only, and has no tools | `agent.py: _prompt` |
| Numbers in a draft must match the evidence (±0.6% or ±0.05) | `agent.py: guard` |
| Every warning and failure in the section must be mentioned | `agent.py: guard` |
| No rating or approval language | `agent.py: FORBIDDEN` |
| 1 retry with the objections, then the deterministic template | `agent.py: draft_section` |
| Every draft, objection, fallback and human action is logged with a timestamp | `signoff.py: Ledger.log` |
| Approvals are tied to an evidence fingerprint and voided if the evidence changes | `signoff.py` |
| `finalize` requires all approvals and a human rating | `signoff.py: finalize` |
