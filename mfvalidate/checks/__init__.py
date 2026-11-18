"""Common result type for every validation test.

A check never edits the model or the developer's outputs. It reads evidence, recomputes what it can
independently, and returns a Check with a status (pass / warn / fail / info), a severity for anything
that is not a pass, the numbers it rests on, and a one-line finding in plain words.
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

STATUSES = ("pass", "warn", "fail", "info")
SEVERITIES = ("high", "medium", "low")

SECTIONS = {
    "scope": "1. Scope, use and risk tier",
    "data": "2. Data integrity and lineage",
    "soundness": "3. Conceptual soundness",
    "replication": "4. Independent replication",
    "performance": "5. Forecast performance",
    "diagnostics": "6. Residual diagnostics",
    "sensitivity": "7. Sensitivity and scenarios",
    "ground_truth": "8. Ground-truth test (simulated data only)",
}


@dataclass
class Check:
    id: str                      # e.g. "P2"
    section: str                 # key of SECTIONS
    title: str
    status: str                  # pass | warn | fail | info
    finding: str                 # one sentence, plain words
    metrics: dict = field(default_factory=dict)
    threshold: str = ""
    severity: Optional[str] = None
    recommendation: str = ""
    table: Optional[pd.DataFrame] = None
    chart: Optional[str] = None  # file name of a chart in the report folder

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"{self.id}: bad status {self.status}")
        if self.status in ("warn", "fail") and self.severity not in SEVERITIES:
            raise ValueError(f"{self.id}: a {self.status} needs a severity")
        if self.status in ("pass", "info"):
            self.severity = None

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "table"}
        d["metrics"] = {k: _plain(v) for k, v in self.metrics.items()}
        if self.table is not None:
            d["table"] = self.table.round(6).astype(object).where(self.table.notna(), None).to_dict("records")
        return d


def _plain(v):
    """Make numpy scalars JSON friendly."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float):
        return round(v, 6)
    return v
