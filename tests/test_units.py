import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from mfvalidate import agent
from mfvalidate.checks import Check
from mfvalidate.checks.performance import dm_test
from mfvalidate.evidence import organic_growth, transform, parse_feature
from mfvalidate.findings import proposed_rating, fingerprint
from mfvalidate.signoff import Ledger


# ------------------------------------------------------------------ statistics and data
def test_dm_test_detects_better_model():
    rng = np.random.default_rng(0)
    e_good = rng.normal(0, 0.5, 80)
    e_bad = rng.normal(0, 1.5, 80)
    stat, p = dm_test(e_good, e_bad, h=1)
    assert stat < 0 and p < 0.01


def test_dm_test_equal_models_not_significant():
    rng = np.random.default_rng(1)
    stat, p = dm_test(rng.normal(0, 1, 80), rng.normal(0, 1, 80), h=4)
    assert p > 0.05


def test_dm_test_too_few_points():
    assert np.isnan(dm_test([1, 2], [2, 3], h=1)[1])


@pytest.mark.parametrize("tr", ["dlog", "diff", "diff4", "yoy", "dlog2", "acc", "gap", "rgap", "lvl"])
def test_transforms_match_developer(tr):
    """The validator re-implements every transform; it must agree with the developer's code."""
    dev = pytest.importorskip("src.features.transforms")
    idx = pd.period_range("2000Q1", periods=60, freq="Q")
    x = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.01, 0.02, 60))), index=idx)
    mine = transform(x, tr)
    theirs = dev.transform(pd.DataFrame({"X": x}), "X", tr)
    pd.testing.assert_series_equal(mine, theirs, check_names=False)


def test_parse_feature():
    assert parse_feature("FHPI.ION|dlog|L1") == ("FHPI.ION", "dlog", 1)
    assert parse_feature("FGDP$.ION|dlog2|L0") == ("FGDP$.ION", "dlog2", 0)
    assert parse_feature("y_lag1") is None


def test_organic_growth_removes_event():
    idx = pd.period_range("2020Q1", periods=4, freq="Q")
    level = pd.Series([100.0, 101.0, 111.0, 112.0], index=idx)
    events = pd.DataFrame({"date": ["2020Q3"], "series": ["loans"], "amount": [9.0]})
    g, _ = organic_growth(level, events, "loans")
    assert g["2020Q3"] == pytest.approx(np.log(102 / 101))
    assert g["2020Q4"] == pytest.approx(np.log(112 / 111))


# ------------------------------------------------------------------ findings and rating
def _c(i, status, sev=None, section="performance"):
    return Check(i, section, f"test {i}", status, f"finding {i}", {"x": 1.0}, severity=sev)


def test_check_requires_severity_for_warning():
    with pytest.raises(ValueError):
        Check("X", "data", "t", "warn", "f")


def test_proposed_rating_rules():
    assert proposed_rating([_c("A", "pass")])[0] == "Satisfactory"
    assert proposed_rating([_c("A", "warn", "high")])[0] == "Needs improvement"
    assert proposed_rating([_c("A", "warn", "medium")] * 3)[0] == "Needs improvement"
    assert proposed_rating([_c("R1", "fail", "high", "replication")])[0] == "Unsatisfactory"


def test_fingerprint_changes_with_numbers():
    a = [_c("A", "pass")]
    b = [Check("A", "performance", "test A", "pass", "finding A", {"x": 2.0})]
    assert fingerprint(a) != fingerprint(b)


# ------------------------------------------------------------------ grounding guard
CHECKS = [
    {"id": "P1", "title": "OOS depth", "status": "fail", "severity": "high",
     "finding": "The test rests on 3 forecast origins.", "metrics": {"origins": 3}, "threshold": ">= 8 origins"},
    {"id": "P2", "title": "Backtest", "status": "pass", "severity": None,
     "finding": "RMSE is 1.10% vs 2.45% for AR(1).", "metrics": {"rmse": 1.1019, "skill": 54.98}, "threshold": ""},
]


def test_guard_accepts_grounded_text():
    text = "P1 is the main weakness: only 3 origins. The backtest RMSE of 1.10% compares with 2.45% for AR(1), 55% better."
    assert agent.guard(text, CHECKS) == []


def test_guard_rejects_invented_number():
    text = "P1: only 3 origins. RMSE is 0.85%."
    assert any("0.85" in p for p in agent.guard(text, CHECKS))


def test_guard_rejects_missing_failure():
    assert any("P1" in p for p in agent.guard("The backtest RMSE is 1.10%.", CHECKS))


def test_guard_rejects_approval_language():
    assert agent.guard("P1 aside, the model is approved and fit for purpose.", CHECKS)


def test_template_is_always_grounded():
    assert agent.guard(agent.template_draft("performance", CHECKS), CHECKS) == []


# ------------------------------------------------------------------ the LLM path, with a fake local model
class _Fake(BaseHTTPRequestHandler):
    replies = []

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"models": []}')

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        text = _Fake.replies.pop(0) if _Fake.replies else "nothing"
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"response": text}).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_ollama():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _cfg(url):
    return {"agent": {"backend": "ollama", "ollama_url": url, "ollama_model": "fake", "max_retries": 1}}


def test_llm_retry_after_guard_objection(fake_ollama):
    _Fake.replies = ["P1: RMSE fell to 0.42%.", "P1 is the weak point: 3 origins only; RMSE 1.10% vs 2.45%."]
    log = []
    text, by = agent.draft_section("performance", CHECKS, _cfg(fake_ollama), lambda *a: log.append(a))
    assert by == "agent (ollama:fake)" and "3 origins" in text
    assert log[0][2]["guard_objections"] and log[1][2]["guard_objections"] == []


def test_llm_falls_back_to_template(fake_ollama):
    _Fake.replies = ["The model is approved.", "Everything is fine, approved."]
    log = []
    text, by = agent.draft_section("performance", CHECKS, _cfg(fake_ollama), lambda *a: log.append(a))
    assert by == "agent (template writer)"
    assert any(a[0] == "agent_fallback" for a in log)


# ------------------------------------------------------------------ human sign-off
def test_signoff_flow(tmp_path):
    led = Ledger(str(tmp_path))
    led.record_draft("performance", "draft text", "agent", "fp1")
    led.record_draft("data", "draft text", "agent", "fp1")
    with pytest.raises(SystemExit):
        led.finalize("Reviewer", "fp1")                      # nothing approved yet
    with pytest.raises(SystemExit):
        led.approve("performance", "Reviewer", None, "fp2")  # evidence changed since the draft
    led.approve("performance", "Reviewer", "ok", "fp1")
    led.approve("data", "Reviewer", None, "fp1")
    with pytest.raises(SystemExit):
        led.finalize("Reviewer", "fp1")                      # no rating yet
    with pytest.raises(SystemExit):
        led.rate("Great", "Reviewer", "x", "Satisfactory")   # not a valid rating
    led.rate("Needs improvement", "Reviewer", "P1 must be fixed", "Satisfactory")
    led.finalize("Reviewer", "fp1")
    assert led.is_final("fp1") and not led.is_final("fp2")
    actions = [e["action"] for e in led.entries()]
    assert actions == ["approve", "approve", "rate", "finalize"]


def test_new_evidence_voids_approval(tmp_path):
    led = Ledger(str(tmp_path))
    led.record_draft("data", "v1", "agent", "fp1")
    led.approve("data", "Reviewer", None, "fp1")
    assert led.record_draft("data", "v2", "agent", "fp1") is False   # same evidence: approval kept
    assert led.record_draft("data", "v2", "agent", "fp2") is True    # new evidence: back to draft
    assert led.state["sections"]["data"]["status"] == "draft"
    assert led.entries()[-1]["action"] == "approval_voided"
