"""Render the validation report as one self-contained HTML file (charts embedded)."""
import base64
import html
import os

import pandas as pd

from .checks import SECTIONS
from .findings import findings

STATUS = {"pass": ("✓", "Pass", "#0ca30c"), "warn": ("!", "Warning", "#fab219"),
          "fail": ("✕", "Fail", "#d03b3b"), "info": ("i", "Info", "#52514e")}
SEV = {"high": ("#d03b3b", "High"), "medium": ("#ec835a", "Medium"), "low": ("#fab219", "Low")}

CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6;--card:#ffffff}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--card:#232321}
 img.chart{filter:invert(.92) hue-rotate(180deg)}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:28px;margin:8px 0 4px;letter-spacing:-.01em}h2{font-size:20px;margin:44px 0 10px;
 padding-top:14px;border-top:1px solid var(--line)}h3{font-size:15px;margin:18px 0 6px}
.sub{color:var(--ink2);margin:0 0 18px}.banner{display:inline-block;padding:4px 10px;border-radius:4px;
 font-weight:600;font-size:12px;letter-spacing:.06em;text-transform:uppercase;border:1.5px solid}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .k{font-size:12px;color:var(--ink2)}.tile .v{font-size:22px;font-weight:650;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 14px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:12px}td.num{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:600;white-space:nowrap}
.dot{display:inline-grid;place-items:center;width:17px;height:17px;border-radius:50%;color:#fff;font-size:11px}
.narr{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:6px;
 padding:12px 16px;margin:10px 0}.meta{font-size:12px;color:var(--ink2);margin-top:8px}
img.chart{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;margin:6px 0 12px;background:#fcfcfb}
details{margin:4px 0 12px}summary{cursor:pointer;color:var(--ink2);font-size:13px}
.small{font-size:12px;color:var(--ink2)}code{font-size:12px}
"""


def _badge(status):
    icon, label, col = STATUS[status]
    return f'<span class="badge"><span class="dot" style="background:{col}">{icon}</span>{label}</span>'


def _sev(sev):
    if not sev:
        return ""
    col, label = SEV[sev]
    return f'<span class="badge"><span class="dot" style="background:{col}">●</span>{label}</span>'


def _img(out_dir, name):
    if not name or not os.path.exists(os.path.join(out_dir, name)):
        return ""
    p = os.path.join(out_dir, name)
    raw = open(p, "rb").read()
    width = int.from_bytes(raw[16:20], "big") * 96 // 150   # charts are saved at 150 dpi; show at CSS size
    b = base64.b64encode(raw).decode()
    return f'<img class="chart" width="{width}" alt="{html.escape(name)}" src="data:image/png;base64,{b}">'


def _table(df, max_rows=40):
    if df is None or len(df) == 0:
        return ""
    df = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append(f'<td class="num">{"" if pd.isna(v) else f"{v:,.4g}"}</td>')
            else:
                cells.append(f"<td>{html.escape('' if v is None else str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render(out_dir, ev, checks, ledger, fp, rating_prop):
    st = ledger.state
    final = ledger.is_final(fp)
    prof = ev.cfg.get("model_profile", {})
    rating = st.get("rating", {}).get("final")
    approved = sum(1 for s in st["sections"].values() if s["status"] == "approved")
    banner_col = "#0ca30c" if final else "#d03b3b"
    banner = "Final" if final else "Draft — not approved"
    f = findings(checks)
    counts = {s: sum(1 for c in f if c.severity == s) for s in ("high", "medium", "low")}

    h = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Validation report — {html.escape(ev.model)}</title>"
         f"<style>{CSS}</style></head><body><main>"]
    h.append(f'<span class="banner" style="color:{banner_col};border-color:{banner_col}">{banner}</span>')
    h.append(f"<h1>Independent validation: {html.escape(prof.get('name', ev.model))}</h1>")
    h.append(f'<p class="sub">Model under review: <b>{html.escape(ev.model)}</b> · run folder '
             f'<code>{html.escape(os.path.basename(ev.run_dir))}</code> · evidence fingerprint <code>{fp}</code></p>')
    h.append('<div class="grid">')
    tiles = [("Proposed rating (rules)", rating_prop[0]), ("Reviewer's rating", rating or "not set"),
             ("High findings", counts["high"]), ("Medium / low", f"{counts['medium']} / {counts['low']}"),
             ("Tests run", len(checks)), ("Sections approved", f"{approved} / {len(st['sections'])}")]
    h += [f'<div class="tile"><div class="k">{k}</div><div class="v">{html.escape(str(v))}</div></div>'
          for k, v in tiles]
    h.append("</div>")
    if rating:
        r = st["rating"]
        h.append(f'<p class="small">Rating set by {html.escape(r["by"])} on {r["at"]}'
                 f'{" (overrides the rule-based proposal)" if r.get("override") else ""}: '
                 f'{html.escape(r.get("rationale") or "")}</p>')
    else:
        h.append(f'<p class="small">Rule-based proposal: {rating_prop[0]} ({rating_prop[1]}). '
                 "The agent cannot set the rating; a reviewer does with <code>mfvalidate rate</code>.</p>")

    h.append("<h2>Findings</h2>")
    if f:
        rows = "".join(f"<tr><td>{c.id}</td><td>{_sev(c.severity)}</td><td>{_badge(c.status)}</td>"
                       f"<td>{html.escape(c.finding)}"
                       f"{'<br><span class=small>Recommendation: ' + html.escape(c.recommendation) + '</span>' if c.recommendation else ''}"
                       f"</td></tr>" for c in f)
        h.append(f"<table><thead><tr><th>ID</th><th>Severity</th><th>Status</th><th>Finding</th></tr></thead>"
                 f"<tbody>{rows}</tbody></table>")
    else:
        h.append("<p>No findings.</p>")

    h.append("<h2>Profile and use</h2>")
    h.append(_table(pd.DataFrame({"item": list(prof), "value": [str(v) for v in prof.values()]})))
    h.append("<h3>Equation under review</h3>")
    h.append(_table(ev.coef_table[[c for c in ("term", "coef", "std_err", "t", "p_value") if c in ev.coef_table]]))

    for key, title in SECTIONS.items():
        cs = [c for c in checks if c.section == key]
        if not cs:
            continue
        s = st["sections"].get(key, {})
        h.append(f"<h2>{html.escape(title)}</h2>")
        if s:
            who = f"Drafted by {html.escape(s['drafted_by'])}"
            if s.get("edited_by"):
                who += f" · edited by {html.escape(s['edited_by'])}"
            who += (f" · <b>approved by {html.escape(s['approved_by'])}</b> on {s['approved_at']}"
                    if s["status"] == "approved" else " · <b>awaiting review</b>")
            if s.get("comment"):
                who += f" · reviewer comment: {html.escape(s['comment'])}"
            h.append(f'<div class="narr">{html.escape(s["text"])}<div class="meta">{who}</div></div>')
        rows = "".join(f"<tr><td>{c.id}</td><td>{_badge(c.status)}</td><td>{_sev(c.severity)}</td>"
                       f"<td><b>{html.escape(c.title)}</b><br>{html.escape(c.finding)}"
                       f"{'<br><span class=small>Threshold: ' + html.escape(c.threshold) + '</span>' if c.threshold else ''}"
                       f"</td></tr>" for c in cs)
        h.append(f"<table><thead><tr><th>ID</th><th>Status</th><th>Severity</th><th>Test and result</th></tr></thead>"
                 f"<tbody>{rows}</tbody></table>")
        for c in cs:
            h.append(_img(out_dir, c.chart))
            if c.table is not None and len(c.table):
                h.append(f"<details><summary>{c.id} data ({len(c.table)} rows)</summary>{_table(c.table)}</details>")

    h.append("<h2>Audit trail</h2>")
    log = ledger.entries()
    if log:
        h.append(_table(pd.DataFrame([{"time": e["time"], "actor": e["actor"], "action": e["action"],
                                       "section": e.get("section") or "",
                                       "details": "; ".join(f"{k}: {v}" for k, v in e["details"].items())}
                                      for e in log[-60:]]), max_rows=60))
    h.append('<p class="small">The agent drafts narrative only. Tests, statuses, severities and numbers come from '
             "code; ratings, approvals and the final sign-off are recorded by a named reviewer.</p>")
    h.append("</main></body></html>")
    path = os.path.join(out_dir, "validation_report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(h))
    return path
