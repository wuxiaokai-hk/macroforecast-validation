"""Command line: run the validation, review drafts, approve, rate and finalize.

    mfvalidate run      --config validation.yaml --out report
    mfvalidate status   --out report
    mfvalidate show     --out report --section performance
    mfvalidate edit     --out report --section performance --file my_text.md --reviewer "Name"
    mfvalidate approve  --out report --section performance --reviewer "Name" [--comment "..."]
    mfvalidate rate     --out report --rating "Needs improvement" --reviewer "Name" --rationale "..."
    mfvalidate finalize --out report --reviewer "Name"
"""
import argparse
import json
import os
import warnings

from .checks import SECTIONS


def _evaluate(cfg_path, out_dir):
    from .evidence import Evidence, load_config
    from .checks import data, soundness, replication, performance, diagnostics, sensitivity, ground_truth
    from .findings import scope_check, proposed_rating, fingerprint

    warnings.filterwarnings("ignore")
    cfg = load_config(cfg_path)
    ev = Evidence(cfg)
    os.makedirs(out_dir, exist_ok=True)
    checks = [scope_check(cfg)]
    checks += data.run(ev, out_dir)
    checks += soundness.run(ev, out_dir)
    checks += replication.run(ev, out_dir)
    perf, bt = performance.run(ev, out_dir)
    checks += perf
    checks += diagnostics.run(ev, out_dir)
    checks += sensitivity.run(ev, out_dir)
    checks += ground_truth.run(ev, out_dir)
    bt.to_csv(os.path.join(out_dir, "backtest_errors.csv"), index=False)
    fp = fingerprint(checks)
    prop = proposed_rating(checks)
    with open(os.path.join(out_dir, "evidence.json"), "w", encoding="utf-8") as f:
        json.dump({"model": ev.model, "run_dir": os.path.relpath(ev.run_dir, cfg["_base"]), "fingerprint": fp,
                   "proposed_rating": {"rating": prop[0], "why": prop[1]},
                   "checks": [c.to_dict() for c in checks]}, f, indent=1, default=str)
    return cfg, ev, checks, fp, prop


def cmd_run(a):
    from .agent import draft_section
    from .report import render
    from .signoff import Ledger

    cfg, ev, checks, fp, prop = _evaluate(a.config, a.out)
    led = Ledger(a.out)
    led.log("system", "run", None, {"model": ev.model, "evidence": fp, "tests": len(checks),
                                    "proposed_rating": prop[0]})
    for key in SECTIONS:
        sc = [c.to_dict() for c in checks if c.section == key]
        if not sc:
            continue
        s = led.state["sections"].get(key)
        if s and s["evidence"] == fp and (s["status"] == "approved" or s.get("edited_by")):
            continue
        text, by = draft_section(key, sc, cfg, lambda action, section, d: led.log("agent", action, section, d))
        led.record_draft(key, text, by, fp)
    led.save()
    path = render(a.out, ev, checks, led, fp, prop)
    print(f"{len(checks)} tests · proposed rating: {prop[0]} ({prop[1]}) · evidence {fp}")
    print(f"Report: {path}")


def _reload(a):
    """Rebuild the checks (no drafting) so approvals are tied to the current evidence."""
    from .report import render
    from .signoff import Ledger

    cfg, ev, checks, fp, prop = _evaluate(a.config, a.out)
    return Ledger(a.out), ev, checks, fp, prop, render


def cmd_status(a):
    from .signoff import Ledger
    led = Ledger(a.out)
    for k, s in led.state["sections"].items():
        tag = f"approved by {s['approved_by']}" if s["status"] == "approved" else "draft"
        print(f"{SECTIONS[k]:<45} {tag}")
    r = led.state.get("rating") or {}
    print(f"\nRating: {r.get('final', 'not set')}   Final: {'yes' if led.state.get('final') else 'no'}")


def cmd_show(a):
    from .signoff import Ledger
    s = Ledger(a.out).state["sections"][a.section]
    print(f"{SECTIONS[a.section]} [{s['status']}] drafted by {s['drafted_by']}\n\n{s['text']}")


def cmd_edit(a):
    from .agent import guard
    led, ev, checks, fp, prop, render = _reload(a)
    text = open(a.file, encoding="utf-8").read().strip()
    objections = guard(text, [c.to_dict() for c in checks if c.section == a.section])
    led.edit(a.section, text, a.reviewer, objections)
    led.save()
    render(a.out, ev, checks, led, fp, prop)
    print("Saved." + ("" if not objections else " Guard notes (kept for the record): " + "; ".join(objections)))


def cmd_approve(a):
    led, ev, checks, fp, prop, render = _reload(a)
    for sec in (list(led.state["sections"]) if a.section == "all" else [a.section]):
        led.approve(sec, a.reviewer, a.comment, fp)
    led.save()
    render(a.out, ev, checks, led, fp, prop)
    print("Approved.")


def cmd_rate(a):
    led, ev, checks, fp, prop, render = _reload(a)
    led.rate(a.rating, a.reviewer, a.rationale, prop[0])
    led.save()
    render(a.out, ev, checks, led, fp, prop)
    print(f"Rating set: {a.rating}" + (f" (proposal was {prop[0]})" if a.rating != prop[0] else ""))


def cmd_finalize(a):
    led, ev, checks, fp, prop, render = _reload(a)
    led.finalize(a.reviewer, fp)
    led.save()
    print("Final report:", render(a.out, ev, checks, led, fp, prop))


def main(argv=None):
    p = argparse.ArgumentParser(prog="mfvalidate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *extra):
        s = sub.add_parser(name)
        s.add_argument("--out", default="report")
        s.add_argument("--config", default="validation.yaml")
        for e in extra:
            s.add_argument(e[0], **e[1])
        s.set_defaults(fn=fn)
        return s

    sec = ("--section", {"required": True, "choices": list(SECTIONS) + ["all"]})
    rev = ("--reviewer", {"required": True})
    add("run", cmd_run)
    add("status", cmd_status)
    add("show", cmd_show, ("--section", {"required": True, "choices": list(SECTIONS)}))
    add("edit", cmd_edit, ("--section", {"required": True, "choices": list(SECTIONS)}), ("--file", {"required": True}),
        rev)
    add("approve", cmd_approve, sec, rev, ("--comment", {"default": None}))
    add("rate", cmd_rate, ("--rating", {"required": True}), rev, ("--rationale", {"required": True}))
    add("finalize", cmd_finalize, rev)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
