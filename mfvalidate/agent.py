"""The drafting agent: writes the narrative of each report section from the check results.

What the agent may do, and what it may not:
- it READS the check results of one section (ids, statuses, numbers, one-line findings);
- it WRITES a draft paragraph for that section, and nothing else;
- it cannot run tests, change a status, a severity or the rating, or approve anything;
- every draft passes a grounding guard before it is accepted (numbers must come from the evidence, every
  warning and failure must be mentioned, no approval or rating language). A draft that fails is retried once
  with the guard's objections, then replaced by the deterministic template writer;
- every call is written to the audit log (backend, model, prompt hash, guard result).

Backends: a local Ollama model (no API key, nothing leaves the machine) or the template writer.
"""
import hashlib
import json
import re
import urllib.request

from .checks import SECTIONS

FORBIDDEN = [r"\bapprov", r"\bfit for (purpose|use)\b", r"\bsatisfactory\b", r"\bunsatisfactory\b",
             r"\bneeds improvement\b", r"\bsigned off\b", r"\bguarantee"]
NUM = re.compile(r"(?<![A-Za-z_|.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
SKIP = re.compile(r"\b(19|20)\d{2}Q[1-4]\b|\b[A-Z]\d\b|\b(19|20)\d{2}\b|[A-Z0-9$]+\.[A-Z]+\|[a-z0-9]+\|L\d")


# ----------------------------------------------------------------------------------------- guard
def _numbers(text):
    text = SKIP.sub(" ", text)
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "").rstrip("%")))
        except ValueError:
            pass
    return out


def _allowed(section_checks):
    vals = set()
    for c in section_checks:
        vals.update(_numbers(c["finding"]))
        vals.update(_numbers(c.get("threshold", "")))
        stack = list(c["metrics"].values())
        while stack:
            v = stack.pop()
            if isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, (list, tuple)):
                stack.extend(v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.update({v, v * 100})
    return vals


def guard(draft, section_checks):
    """Return a list of objections (empty = the draft is grounded)."""
    problems = []
    allowed = _allowed(section_checks)
    for n in _numbers(draft):
        if abs(n) <= 12 and float(n).is_integer():
            continue  # counts, horizons and lags
        if not any(abs(n - a) <= max(0.051, abs(a) * 0.006) for a in allowed):
            problems.append(f"number {n:g} is not in the evidence")
    for c in section_checks:
        if c["status"] in ("warn", "fail") and c["id"] not in draft:
            problems.append(f"{c['id']} ({c['status']}) is not mentioned")
    for pat in FORBIDDEN:
        if re.search(pat, draft, re.I):
            problems.append(f"uses reserved language /{pat}/ (ratings and approvals are for the human reviewer)")
    return problems


# ----------------------------------------------------------------------------------------- backends
def template_draft(section, section_checks):
    """Deterministic writer: always grounded, used when no LLM is available or a draft fails the guard."""
    n = len(section_checks)
    bad = [c for c in section_checks if c["status"] in ("warn", "fail")]
    ok = [c for c in section_checks if c["status"] == "pass"]
    info = [c for c in section_checks if c["status"] == "info"]
    parts = []
    if info:
        parts += [c["finding"] for c in info]
    if n - len(info):
        parts.append(f"{len(ok)} of {n - len(info)} tests in this section pass.")
    for c in sorted(bad, key=lambda c: {"high": 0, "medium": 1, "low": 2}[c["severity"]]):
        parts.append(f"{c['id']} ({c['severity']} severity, {c['status']}): {c['finding']}")
    if ok and bad:
        parts.append("Passed: " + "; ".join(f"{c['id']} {c['title'][0].lower() + c['title'][1:]}" for c in ok) + ".")
    elif ok:
        parts.append(" ".join(f"{c['id']}: {c['finding']}" for c in ok))
    return " ".join(parts)


def ollama_available(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def _prompt(section, section_checks, objections=None):
    facts = [{k: c[k] for k in ("id", "title", "status", "severity", "finding")} for c in section_checks]
    p = (f"You are drafting section '{SECTIONS[section]}' of an independent model validation report.\n"
         "Write one paragraph of at most 150 words for a model risk committee, in plain English.\n"
         "Rules: use ONLY the facts below; copy numbers exactly as written; mention every test id whose status is "
         "warn or fail, most severe first; do not give an overall rating, do not say anything is approved, fit for "
         "purpose or satisfactory; no recommendations beyond what the facts say.\n"
         f"FACTS:\n{json.dumps(facts, indent=1)}\n")
    if objections:
        p += "Your previous draft was rejected for these reasons; fix them:\n- " + "\n- ".join(objections) + "\n"
    return p + "PARAGRAPH:"


def _ollama(url, model, prompt):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.1}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/generate", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"].strip()


def draft_section(section, section_checks, cfg, log):
    """Draft one section. `log(action, section, details)` writes to the audit trail."""
    a = cfg.get("agent", {})
    backend = a.get("backend", "auto")
    use_llm = backend == "ollama" or (backend == "auto" and ollama_available(a.get("ollama_url", "")))
    if use_llm:
        objections = None
        for attempt in range(1 + int(a.get("max_retries", 1))):
            prompt = _prompt(section, section_checks, objections)
            try:
                text = _ollama(a["ollama_url"], a["ollama_model"], prompt)
            except OSError as e:
                log("agent_error", section, {"error": str(e)})
                break
            objections = guard(text, section_checks)
            log("agent_draft", section, {"backend": f"ollama:{a['ollama_model']}", "attempt": attempt + 1,
                                         "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12],
                                         "guard_objections": objections})
            if not objections:
                return text, f"agent (ollama:{a['ollama_model']})"
        log("agent_fallback", section, {"reason": "draft failed the grounding guard or the model was unreachable"})
    text = template_draft(section, section_checks)
    objections = guard(text, section_checks)
    log("agent_draft", section, {"backend": "template", "guard_objections": objections})
    return text, "agent (template writer)"
