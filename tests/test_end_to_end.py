"""End-to-end on the example run (skipped when the run folder has not been built)."""
import json
import os

import pytest

from mfvalidate.cli import main

ROOT = os.path.dirname(os.path.dirname(__file__))
CFG = os.path.join(ROOT, "validation.yaml")


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "runs", "loan_balance")),
                    reason="run python examples/make_run.py first")
def test_full_validation(tmp_path):
    out = str(tmp_path / "report")
    main(["run", "--config", CFG, "--out", out])
    ev = json.load(open(os.path.join(out, "evidence.json")))
    status = {c["id"]: c["status"] for c in ev["checks"]}
    assert status["R1"] == "pass" and status["R2"] == "pass"   # independent replication
    assert status["D1"] == "pass"                               # lineage
    assert os.path.exists(os.path.join(out, "validation_report.html"))
    main(["approve", "--config", CFG, "--out", out, "--section", "all", "--reviewer", "Test"])
    main(["rate", "--config", CFG, "--out", out, "--rating", "Needs improvement", "--reviewer", "Test",
          "--rationale", "test"])
    main(["finalize", "--config", CFG, "--out", out, "--reviewer", "Test"])
    assert "Final" in open(os.path.join(out, "validation_report.html")).read()
