"""Human review: the only way a section is approved, a rating is set or a report becomes final.

State lives in <report>/signoff.json; every action is appended to <report>/audit_log.jsonl.
An approval records the evidence fingerprint it was given on; if the checks are re-run and any status or
number changes, the approval no longer applies and the section goes back to draft.
"""
import datetime as dt
import json
import os

from .checks import SECTIONS
from .findings import RATINGS


def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


class Ledger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.state_path = os.path.join(out_dir, "signoff.json")
        self.log_path = os.path.join(out_dir, "audit_log.jsonl")
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as f:
                self.state = json.load(f)
        else:
            self.state = {"sections": {}, "rating": {}, "final": None}

    # ---------------------------------------------------------------- persistence
    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=1)

    def log(self, actor, action, section=None, details=None):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": now(), "actor": actor, "action": action, "section": section,
                                "details": details or {}}) + "\n")

    def entries(self):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---------------------------------------------------------------- agent side
    def record_draft(self, section, text, drafted_by, fp):
        s = self.state["sections"].get(section)
        if s and s["status"] == "approved" and s["evidence"] == fp:
            return False  # approved on the same evidence: keep the human's version
        if s and s["status"] == "approved":
            self.log("system", "approval_voided", section, {"reason": "evidence changed", "old": s["evidence"],
                                                           "new": fp})
        if s and s.get("edited_by") and s["evidence"] == fp:
            return False  # a human edit on the same evidence is kept
        self.state["sections"][section] = {"status": "draft", "text": text, "drafted_by": drafted_by,
                                           "evidence": fp, "approved_by": None, "approved_at": None,
                                           "comment": None, "edited_by": None}
        return True

    # ---------------------------------------------------------------- human side
    def _need(self, section):
        if section not in self.state["sections"]:
            raise SystemExit(f"Unknown section '{section}'. Sections: {', '.join(self.state['sections'])}")
        return self.state["sections"][section]

    def edit(self, section, text, reviewer, objections):
        s = self._need(section)
        s.update(text=text, edited_by=reviewer, status="draft", approved_by=None, approved_at=None)
        self.log(f"human:{reviewer}", "edit", section, {"guard_objections": objections})

    def approve(self, section, reviewer, comment, fp):
        s = self._need(section)
        if s["evidence"] != fp:
            raise SystemExit("The evidence changed since this draft was written; re-run 'mfvalidate run' first.")
        s.update(status="approved", approved_by=reviewer, approved_at=now(), comment=comment)
        self.log(f"human:{reviewer}", "approve", section, {"comment": comment, "evidence": fp})

    def rate(self, rating, reviewer, rationale, proposed):
        if rating not in RATINGS:
            raise SystemExit(f"Rating must be one of: {', '.join(RATINGS)}")
        self.state["rating"] = {"final": rating, "by": reviewer, "at": now(), "rationale": rationale,
                                "proposed": proposed, "override": rating != proposed}
        self.log(f"human:{reviewer}", "rate", None, {"rating": rating, "proposed": proposed,
                                                     "override": rating != proposed, "rationale": rationale})

    def finalize(self, reviewer, fp):
        pending = [k for k, s in self.state["sections"].items() if s["status"] != "approved"]
        if pending:
            raise SystemExit("Cannot finalize; sections not approved: " + ", ".join(SECTIONS[k] for k in pending))
        if not self.state["rating"].get("final"):
            raise SystemExit("Cannot finalize; no rating set ('mfvalidate rate').")
        stale = [k for k, s in self.state["sections"].items() if s["evidence"] != fp]
        if stale:
            raise SystemExit("Cannot finalize; evidence changed after approval of: " + ", ".join(stale))
        self.state["final"] = {"by": reviewer, "at": now(), "evidence": fp}
        self.log(f"human:{reviewer}", "finalize", None, {"evidence": fp})

    def is_final(self, fp):
        f = self.state.get("final")
        return bool(f) and f["evidence"] == fp
