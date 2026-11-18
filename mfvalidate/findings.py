"""Turn check results into findings, a proposed rating and a fingerprint of the evidence."""
import hashlib
import json

from .checks import Check

ORDER = {"high": 0, "medium": 1, "low": 2}
RATINGS = ("Satisfactory", "Needs improvement", "Unsatisfactory")


def scope_check(cfg):
    p = cfg.get("model_profile", {})
    return Check("A1", "scope", "Model profile, intended use and risk tier", "info",
                 f"{p.get('name', 'The model')} is used to {p.get('purpose', 'n/a').lower()}; risk tier "
                 f"{p.get('risk_tier', 'n/a')}. The validation tests the model against that use.",
                 {k: v for k, v in p.items()})


def findings(checks):
    f = [c for c in checks if c.status in ("warn", "fail")]
    return sorted(f, key=lambda c: (ORDER[c.severity], c.id))


def proposed_rating(checks):
    """Rule-based proposal. The final rating is always set by a human reviewer (see signoff.py)."""
    f = findings(checks)
    high = [c for c in f if c.severity == "high"]
    medium = [c for c in f if c.severity == "medium"]
    blocking = [c for c in high if c.section in ("replication", "data") and c.status == "fail"]
    if blocking or len(high) >= 3:
        rating = "Unsatisfactory"
    elif high or len(medium) >= 3:
        rating = "Needs improvement"
    else:
        rating = "Satisfactory"
    why = (f"{len(high)} high and {len(medium)} medium findings"
           + (f"; {', '.join(c.id for c in blocking)} block reliance on the results" if blocking else ""))
    return rating, why


def fingerprint(checks):
    """Hash of every status and number: if the evidence changes, earlier approvals no longer apply."""
    payload = [{"id": c.id, "status": c.status, "severity": c.severity, "metrics": c.to_dict()["metrics"]}
               for c in checks]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
