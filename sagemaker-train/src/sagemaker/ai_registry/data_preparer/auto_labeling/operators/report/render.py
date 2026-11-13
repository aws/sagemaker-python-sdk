# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""The end-of-loop run report, rendered to ``report.html`` and ``report.md``.

Both documents come from the same sections. The HTML is self-contained (inline CSS,
CSS-only ``<details>``); the Markdown pastes into a CR description, Quip or Slack.

Two cases:

    * The corpus run (the usual case). Its labels stay behind the API, so the report is
      built from what is on the client: ``labeled_dataset.json`` (the labeled DataSet ARN
      and the labels' S3 URI), the job's ``labeling_summary.json`` beside it (or
      ``calibration``, or the corpus run's run-log entry), ``held_out_report.json``, the
      fold report and the run log. It shows the job status, the certified bound, the
      cutoff and spot-check, the held-out per-class F1 and confusions, the review set's
      counts, the deliverables and the jurors' responses. No row of the labeled corpus is
      read.
    * A local ``labels.jsonl``: the whole-corpus cold start, where people labeled every
      row. The report states the rows, who labeled them, the class balance, a few
      samples and the columns.

With neither file the report refuses: the run did not complete.
"""

from __future__ import absolute_import

import html
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _run_log
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (
    LEGACY_REVIEW_REASON_FIELDS,
    REVIEW_REASON_FIELD,
    job_summary_sha256,
    labeled_by_people,
    stated_confidence_missing,
    verbalized_confidence_missing,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import (
    no_promise_reason,
    parse_qa_match,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import run_log
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

__all__ = [
    "write_report",
    "build_html",
    "build_markdown",
]

_log = logging.getLogger(__name__)

DASH = "n/a"  # placeholder for an unknown value (never an em dash)

# The formats write_report writes, and the tasks it accepts (the job graded the task).
FORMATS = ("html", "md", "both")
TASKS = ("classification", "ner", "qa", "ordinal", "llm_judge", "summarization")


# ---------------------------------------------------------------------------
# io + number formatting
# ---------------------------------------------------------------------------


def _j(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _jl(p, required=False):
    """Read a JSONL file to a list of rows; a missing or unreadable file is ``[]``.

    A malformed line raises naming the file and line when ``required`` (a corrupt
    labels.jsonl is not an incomplete run), else is skipped.
    """
    try:
        f = open(p)
    except OSError:
        return []
    rows = []
    with f:
        for n, l in enumerate(f, 1):
            if not l.strip():
                continue
            try:
                rows.append(json.loads(l))
            except ValueError as e:
                if required:
                    raise DataPreparerOperatorError(
                        f"{p}: malformed JSON on line {n} ({e}); corrupt file, "
                        "not an incomplete run"
                    )
                continue
    return rows


def _job_manifests(project):
    """The newest labeling run per labeled output (``run_log.latest_labeling_runs``).

    The apply run (output at the project root) first, then the runs under work/.
    """
    try:
        runs = run_log.latest_labeling_runs(
            _run_log.read_entries(project), os.path.abspath(project)
        )
    except Exception:  # noqa: BLE001 - an unreadable run log reads as no runs
        runs = []
    return sorted(runs, key=lambda r: os.sep in str(r.get("output") or ""))


def _confidence_shortfall(project):
    """The ``confidence_shortfall`` of the first labeling run that records one, or None.

    The apply run leads (:func:`_job_manifests`).
    """
    for m in _job_manifests(project):
        if m.get("confidence_shortfall"):
            return m["confidence_shortfall"]
    return None


# The corpus run's --out (its deliverable record), and a local labels file.
CORPUS_OUTPUT = "labeled_dataset.json"
LOCAL_LABELS = "labels.jsonl"


def _corpus_deliverable(project):
    """labeled_dataset.json when it names a DataSet ARN or the labels' S3 URI, else None."""
    doc = _j(os.path.join(project, CORPUS_OUTPUT))
    return (
        doc
        if isinstance(doc, dict) and (doc.get("dataset_arn") or doc.get("labels_s3_uri"))
        else None
    )


def _job_summary(project, path=None):
    """The labeling job's labeling_summary.json.

    ``path`` when given, else the file the corpus run wrote beside labeled_dataset.json,
    else the ``job_summary`` of the corpus run's run-log entry. None when there is none
    (a run not labeled by a job).
    """
    if path:
        doc = _j(path)
        return doc if isinstance(doc, dict) else None
    doc = _j(os.path.join(project, "labeling_summary.json"))
    if isinstance(doc, dict) and doc.get("Status"):
        return doc
    runs = sorted(
        _job_manifests(project),
        key=lambda r: os.path.basename(str(r.get("output") or ""))
        not in (CORPUS_OUTPUT, LOCAL_LABELS),
    )
    for m in runs:
        js = m.get("job_summary") if isinstance(m, dict) else None
        if isinstance(js, dict) and js.get("Status"):
            return js
    return None


def _esc(x):
    return html.escape(str(x))


def _pct(x):
    return f"{100 * x:.1f}%"


def _num(x):
    """The value when it is a real number, else None (a bool is not a number here)."""
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _pct_or_dash(x):
    return _pct(x) if isinstance(x, (int, float)) else DASH


# Prediction fields other than `label` (the job's prediction fields), in the order
# _resolve_pred_field tries them.
_TASK_PRED_FIELD_ORDER = ("entities", "answer", "judge_score", "score", "summary")


def _resolve_pred_field(labels, pred_field):
    """The prediction field of a local labels file.

    When the caller left the default ``label`` and the rows carry no ``label`` key at all
    but a task's output field, use that field. A present-but-null ``label`` keeps
    ``label``, even if an input column happens to be named ``score`` or ``answer``.
    """
    if pred_field == "label" and not any(r.get("label") is not None for r in labels[:25]):
        if not any("label" in r for r in labels[:25] if isinstance(r, dict)):
            for cand in _TASK_PRED_FIELD_ORDER:
                if any(r.get(cand) is not None for r in labels[:25]):
                    return cand
    return pred_field


# ---------------------------------------------------------------------------
# format adapter: one section-building path, two renderers (html and md)
# ---------------------------------------------------------------------------


def _slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _inline_html(s):
    """Escape data, then turn the author markup **bold** and `code` into tags."""
    s = _esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def _md_escape(s):
    """A value as Markdown table-cell text.

    Newlines are flattened and the specials that would break a row or read as markup are
    escaped (the backslash first).
    """
    s = str(s).replace("\r", " ").replace("\n", " ")
    # `[` is escaped so data cannot inject a [text](url) link; `_` is not, since
    # snake_case labels are common and it injects nothing.
    return (
        s.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("*", "\\*")
        .replace("`", "\\`")
        .replace("[", "\\[")
    )


# The HTML report's stylesheet.
_CSS = (
    ":root{--ink:#1f2933;--accent:#5b7fa4;--good:#6aad8e;--amber:#d4a853;--line:#e3e7ec;"
    "--bg:#f8f9fa}\n"
    "*{box-sizing:border-box}\n"
    "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe "
    "UI',Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);margin:0;"
    "padding:32px;line-height:1.6}\n"
    ".wrap{max-width:900px;margin:0 auto}\n"
    "h1{font-size:24px;margin:0 0 4px} .subtitle{color:#6b7280;font-size:13px;"
    "margin:0 0 16px}\n"
    "h2{font-size:17px;margin:8px 0 10px;border-bottom:2px solid var(--line);"
    "padding-bottom:6px}\n"
    "h3{font-size:14px;margin:18px 0 8px} h4{font-size:12.5px;margin:12px 0 6px;"
    "color:#42566b}\n"
    "p{font-size:13px;margin:6px 0} .note{color:#6b7280;font-style:italic;font-size:12px}\n"
    "ul{font-size:13px;margin:6px 0;padding-left:20px} li{margin:3px 0}\n"
    "nav.topnav{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);"
    "border-bottom:1px solid var(--line);\n"
    "  margin:0 -32px 18px;padding:8px 32px;display:flex;flex-wrap:wrap;gap:2px 4px;"
    "font-size:12px;backdrop-filter:saturate(180%) blur(6px)}\n"
    "nav.topnav a{color:#5c6773;text-decoration:none;padding:4px 10px;border-radius:6px;"
    "white-space:nowrap;border-bottom:2px solid transparent}\n"
    "nav.topnav a + a{border-left:1px solid var(--line)}\n"
    "nav.topnav a:hover{color:var(--ink);border-bottom-color:var(--line)}\n"
    "nav.topnav a.active{color:var(--accent);border-bottom-color:var(--accent);"
    "font-weight:600}\n"
    "section{margin:14px 0;"
    "scroll-margin-top:52px} .totop{margin-top:8px} .totop a{font-size:11px;color:#9ca3af;"
    "text-decoration:none}\n"
    ".verdict{background:#fff;border:1px solid var(--line);"
    "border-left:4px solid var(--accent);border-radius:12px;padding:14px 16px;margin:8px 0}\n"
    ".verdict-line{font-size:14px;font-weight:600} .verdict-rule{margin-top:6px;"
    "color:#6b7280;font-style:italic;font-size:11px}\n"
    ".badge-verdict{display:inline-block;margin-right:8px;padding:2px 10px;"
    "border-radius:20px;font-size:11px;font-weight:700;text-transform:uppercase;"
    "letter-spacing:.03em}\n"
    ".badge-verdict.ready{background:#dcefe6;"
    "color:#20614a} .badge-verdict.needs-review{background:#fdeede;"
    "color:#a5641f} .badge-verdict.not-ready{background:#fbe1e1;color:#a13636}\n"
    ".spec-rendered{background:#f6f8fa;border:1px solid var(--line);border-radius:8px;"
    "padding:8px 14px;margin:6px 0}\n"
    ".spec-rendered h4,.spec-rendered h5,.spec-rendered h6{margin:10px 0 4px;"
    "color:var(--ink)} .spec-rendered h4{font-size:13.5px} .spec-rendered "
    "h5{font-size:12.5px} .spec-rendered h6{font-size:11.5px}\n"
    ".spec-rendered p,.spec-rendered li{font-size:12.5px} details.rawtoggle{border:none;"
    "background:none;padding:0;margin:6px 0} details.rawtoggle>summary{font-size:11.5px;"
    "color:#6b7280}\n"
    ".metrics-strip{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;background:#fff;"
    "border:1px solid var(--line);border-radius:12px;padding:18px 20px}\n"
    ".metric-item{flex:1 1 120px;min-width:110px}\n"
    ".metric-value{font-size:24px;font-weight:700;color:var(--accent)}\n"
    ".metric-value.positive{color:var(--good)} .metric-value.negative{color:#d4a853}\n"
    ".metric-label{font-size:11px;margin-top:2px;text-transform:uppercase;"
    "letter-spacing:.04em;color:#4b5563}\n"
    ".metric-detail{margin-top:3px;color:#6b7280;font-size:11px;line-height:1.3}\n"
    ".summary-panel,.handoff{background:#fff;border-radius:12px;border:1px solid var(--line);"
    "padding:18px 20px;margin:10px 0}\n"
    ".handoff{background:#eef4fb;border-color:#cfe0f2;border-left:4px solid var(--accent)}\n"
    ".handoff-title{font-size:15px;font-weight:700;color:#31527a}\n"
    ".diagram{max-width:100%;border:1px solid var(--line);border-radius:8px;margin:8px 0;"
    "background:#fff}\n"
    "table.grid{border-collapse:collapse;width:100%;background:#fff;"
    "border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:12.5px;"
    "margin:8px 0}\n"
    ".grid th{text-align:left;background:#f2f4f7;padding:7px 10px} .grid td{padding:6px 10px;"
    "border-top:1px solid var(--line);vertical-align:top}\n"
    ".grid tr.best-run td{background:#edf7f2;color:#315f4f;border-top-color:#cfe7dc}\n"
    ".grid tr.best-run td:first-child{border-left:4px solid var(--good);font-weight:600}\n"
    ".grid tr.hi td{background:#eef7ec;"
    "font-weight:600} .grid tr.warnrow td{background:#fdeede}\n"
    ".best-run-badge{display:inline-block;margin-left:6px;padding:1px 6px;border-radius:4px;"
    "background:#dcefe6;color:#315f4f;font-size:10px;font-weight:700;"
    "text-transform:uppercase}\n"
    ".samples td.reason,.reason{color:#555}\n"
    ".badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;"
    "font-weight:600} .badge.auto{background:#eef7ec;"
    "color:#3d7a34} .badge.route{background:#fdeede;color:#a5641f}\n"
    ".sagemaker-job{background:#fff;border:1px solid var(--line);border-radius:10px;"
    "padding:14px 16px;margin:8px 0}\n"
    ".sagemaker-job-type{font-size:11px;font-weight:700;text-transform:uppercase;"
    "letter-spacing:.04em;color:var(--accent)}\n"
    ".sagemaker-job-desc{font-size:13px;margin:3px 0}\n"
    ".sagemaker-job-arn{font-family:'SF Mono','Fira Code',monospace;font-size:11px;"
    "color:#6b7280;word-break:break-all}\n"
    ".sagemaker-job-cost{font-size:11.5px;color:#4b5563;margin-top:4px}\n"
    ".sagemaker-job-paths{margin-top:10px;padding-top:8px;border-top:1px solid var(--line)}\n"
    ".sagemaker-job-path{margin-top:6px}\n"
    ".sagemaker-job-path-label{font-size:10.5px;font-weight:600;color:var(--accent)}\n"
    ".sagemaker-job-path-value{display:block;margin-top:2px;"
    "font-family:'SF Mono','Fira Code',monospace;font-size:11px;color:#4a5568;"
    "word-break:break-all}\n"
    ".loop-steps-note{color:#6b7280;font-size:11.5px;font-style:italic;margin:4px 0 10px}\n"
    ".step-card{background:#fff;border:1px solid var(--line);"
    "border-left:4px solid var(--good);border-radius:12px;padding:14px 16px;margin:10px 0}\n"
    ".step-card .step-head{display:flex;justify-content:space-between;align-items:baseline;"
    "gap:8px}\n"
    ".step-title{font-size:14px;font-weight:600}\n"
    ".step-tag{font-size:11px;color:#6b7280;background:#f2f4f7;padding:2px 8px;"
    "border-radius:6px;white-space:nowrap}\n"
    ".step-card ul{margin:6px 0}\n"
    ".iter-card{background:#fff;border:1px solid var(--line);"
    "border-left:4px solid var(--accent);border-radius:12px;padding:12px 16px;margin:8px 0}\n"
    ".iter-card.best{border-left-color:var(--good);background:#f4faf7}\n"
    ".iter-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}\n"
    ".iter-name{font-size:13.5px;font-weight:600}\n"
    ".iter-metrics{font-size:12px;color:#42566b;margin-top:4px}\n"
    ".iter-spec{font-family:'SF Mono','Fira Code',monospace;font-size:10.5px;color:#6b7280}\n"
    "details{margin:8px 0;background:#fff;border:1px solid var(--line);border-radius:8px;"
    "padding:6px 12px} summary{cursor:pointer;font-size:13px;color:var(--accent);"
    "font-weight:600}\n"
    "code{font-family:'SF Mono','Fira Code',monospace;font-size:12px;background:#eef1f5;"
    "padding:1px 4px;border-radius:3px}\n"
    ".mono{font-family:'SF Mono','Fira Code',monospace;font-size:11px;color:#4a5568;"
    "word-break:break-all}\n"
    "pre.spec{background:#f6f8fa;border:1px solid var(--line);border-radius:8px;padding:12px;"
    "font-size:11.5px;white-space:pre-wrap;word-break:break-word;max-height:420px;"
    "overflow:auto}\n"
    ".stack{display:flex;height:22px;width:100%;border-radius:6px;overflow:hidden;"
    "border:1px solid var(--line);margin:8px 0} .stack .seg{height:100%}\n"
    ".hbar{display:inline-block;height:9px;background:var(--accent);border-radius:2px;"
    "margin-right:6px;vertical-align:middle}\n"
    ".chart{margin:10px 0;background:#fff;border:1px solid var(--line);border-radius:8px;"
    "padding:10px 12px}\n"
    ".chart svg{display:block}\n"
    ".warn{background:#fdeede;border-left:3px solid #a5641f;padding:6px 10px;"
    "border-radius:4px;font-size:12.5px}\n"
    "footer{margin-top:34px;color:#9ca3af;font-size:11px;border-top:1px solid var(--line);"
    "padding-top:12px}\n"
    "@keyframes fadein{from{opacity:0;transform:translateY(6px)}to{opacity:1;"
    "transform:none}}\n"
    "section{animation:fadein .4s ease both}\n"
    "@media (prefers-reduced-motion:reduce){section{animation:none}}\n"
)


class Fmt:
    """Renders section blocks to one of two targets, ``html`` or ``md``.

    Section builders call these methods and stay format-agnostic, so HTML and Markdown
    share one code path.
    """

    def __init__(self, kind):
        self.kind = kind

    # --- inline pieces ---
    def _cell(self, c):
        # ("raw", html, md) is report-authored markup, passed through per format.
        # Anything else is data: escaped, never read as **bold** or `code`.
        if isinstance(c, tuple) and c and c[0] == "raw":
            return c[1] if self.kind == "html" else c[2]
        s = "" if c is None else str(c)
        if self.kind == "html":
            return _esc(s)
        return _md_escape(s)

    # --- block builders ---
    def h(self, level, text):
        """A heading of ``level``."""
        if self.kind == "html":
            return f"<h{level}>{_inline_html(text)}</h{level}>"
        return f"\n{'#' * level} {text}\n"

    def h3(self, text):
        """A level-3 heading."""
        return self.h(3, text)

    def h4(self, text):
        """A level-4 heading."""
        return self.h(4, text)

    def p(self, text):
        """A paragraph of author markup."""
        if self.kind == "html":
            return f"<p>{_inline_html(text)}</p>"
        return f"\n{text}\n"

    def note(self, text):
        """An italic aside."""
        if self.kind == "html":
            return f'<p class="note">{_inline_html(text)}</p>'
        return f"\n_{text}_\n"

    def warn(self, text):
        """A warning (a callout in HTML, a quote in Markdown)."""
        if self.kind == "html":
            return f'<p class="warn">{_inline_html(text)}</p>'
        return f"\n> {text}\n"

    def table(self, headers, rows):
        """A table; each cell is data (escaped) or a ``("raw", html, md)`` tuple."""
        if self.kind == "html":
            ths = "".join(f"<th>{_inline_html(h)}</th>" for h in headers)
            trs = ""
            for row in rows:
                tds = "".join(f"<td>{self._cell(c)}</td>" for c in row)
                trs += f"<tr>{tds}</tr>"
            return (
                f'<table class="grid"><thead><tr>{ths}</tr></thead>' f"<tbody>{trs}</tbody></table>"
            )
        out = [
            "",
            "| " + " | ".join(str(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            out.append("| " + " | ".join(self._cell(c) for c in row) + " |")
        return "\n".join(out) + "\n"

    def strip(self, tiles):
        """A strip of metric tiles (``label``, ``value``, optional ``detail`` and ``cls``)."""
        if self.kind == "html":
            out = ""
            for t in tiles:
                cls = f' {t["cls"]}' if t.get("cls") else ""
                detail = (
                    f'<div class="metric-detail">{_esc(t["detail"])}</div>'
                    if t.get("detail")
                    else ""
                )
                out += (
                    f'<div class="metric-item"><div class="metric-value{cls}">'
                    f'{_esc(t["value"])}</div>'
                    f'<div class="metric-label">{_esc(t["label"])}</div>{detail}</div>'
                )
            return f"<div class=metrics-strip>{out}</div>"
        rows = [[t["label"], t["value"], t.get("detail", "")] for t in tiles]
        return self.table(["Metric", "Value", "Notes"], rows)

    # --- document assembly ---
    def document(self, title, subtitle, sections):
        """The whole document; sections with an empty body are dropped."""
        sections = [(t, b) for t, b in sections if b and b.strip()]
        if self.kind == "html":
            return self._html_doc(title, subtitle, sections)
        return self._md_doc(title, subtitle, sections)

    def _html_doc(self, title, subtitle, sections):
        nav = "".join(f'<a href="#{_slug(t)}">{_esc(t)}</a>' for t, _ in sections)
        body = ""
        for t, b in sections:
            body += (
                f'<section id="{_slug(t)}"><h2>{_esc(t)}</h2>{b}'
                f'<div class="totop"><a href="#top">Back to top</a></div></section>'
            )
        return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>{_esc(title)}</title>
<style>
{_CSS}</style></head><body><a id="top"></a>
<nav class="topnav">{nav}</nav>
<div class=wrap>
<h1>{_esc(title)}</h1>
<p class=subtitle>{_esc(subtitle)}</p>
{body}
<footer>Generated from this run's own outputs (labels.jsonl or labeled_dataset.json,
labeling_summary.json, held_out_report.json, run-log.jsonl). Every number is a pure function
of those artifacts; "{DASH}" means the artifact did not carry that field.</footer>
</div>
<script>
/* highlight the current section link in the top nav (progressive enhancement) */
if(window.IntersectionObserver){{
  var links={{}};
  document.querySelectorAll('nav.topnav a').forEach(function(a){{
    links[a.getAttribute('href').slice(1)]=a;}});
  var obs=new IntersectionObserver(function(es){{
    es.forEach(function(e){{
      if(e.isIntersecting){{
        for(var k in links){{links[k].classList.remove('active');}}
        var a=links[e.target.id]; if(a){{a.classList.add('active');}}
      }}
    }});
  }},{{rootMargin:'-45% 0px -50% 0px'}});
  document.querySelectorAll('section[id]').forEach(function(s){{obs.observe(s);}});
}}
</script>
</body></html>"""

    def _md_doc(self, title, subtitle, sections):
        toc = "\n".join(f"- [{t}](#{_slug(t)})" for t, _ in sections)
        parts = [f"# {title}", "", subtitle, "", "## Contents", "", toc]
        for t, b in sections:
            parts.append(f'\n<a id="{_slug(t)}"></a>')
            parts.append(f"## {t}")
            parts.append(b)
        parts.append(
            "\n---\n_Generated from this run's own outputs "
            "(labels.jsonl or labeled_dataset.json, labeling_summary.json, "
            f'held_out_report.json, run-log.jsonl). "{DASH}" means the artifact did not carry '
            "that field._"
        )
        return "\n".join(parts) + "\n"


_INPUT_FIELDS = (
    "text",
    "input",
    "sentence",
    "question",
    "context",
    "source",
    "candidate",
    "premise",
    "hypothesis",
    "document",
    "passage",
    "review_text",
    "body",
)
_RESERVED_INPUT_KEYS = {
    "id",
    "label",
    "gt_label",
    "entities",
    "gt_entities",
    "answer",
    "gt_answers",
    "summary",
    "gt_summary",
    "score",
    "gt_score",
    "judge_score",
    "confidence_score",
    "confidence",
    "reasoning",
    "jury_votes",
    "min_span_confidence",
    "_failed",
    "_model_output",
    "_input_index",
    "_pos",
    "_text_missing",
    "_extra",
}
# Every routing-tag alias the review-file reader accepts (_fields.review_reason_of),
# so a legacy routing tag on a row never shows up as the row's input text.
_RESERVED_INPUT_KEYS |= {REVIEW_REASON_FIELD, *LEGACY_REVIEW_REASON_FIELDS}


def _input_text(row):
    """Display text for a row's input.

    The _INPUT_FIELDS it has, joined with ' | ', else its first non-reserved string
    field, else "".
    """
    parts = [str(row[f]) for f in _INPUT_FIELDS if isinstance(row.get(f), str) and row.get(f)]
    if parts:
        return " | ".join(parts)
    for k, v in row.items():
        if k not in _RESERVED_INPUT_KEYS and isinstance(v, str) and v:
            return v
    return ""


# The fold's report (_fields.fold_report_path) rides on the summary under this key, so
# _calibration_block shows the spot-check beside the bound.
FOLD_KEY = "_fold_report"
# The corpus run's side-file fold writes review_seed.fold.json (full-loop Step 7b); a fold
# of a local review file (cold start, relabel) writes hil_seed.fold.json.
FOLD_REPORT_CANDIDATES = ("review_seed.fold.json", "hil_seed.fold.json")


def _attach_fold_report(cal, project):
    """``cal`` with the project's fold report under FOLD_KEY, else ``cal`` unchanged.

    The fold report is attached when its ``job_summary_sha256`` is this summary's: a fold
    of another job, or one run without a job summary, is never shown beside this job's
    bound.
    """
    if not isinstance(cal, dict) or not project:
        return cal
    want = job_summary_sha256({k: v for k, v in cal.items() if k != FOLD_KEY})
    for rel in FOLD_REPORT_CANDIDATES:
        fold = _j(os.path.join(project, rel))
        if not (isinstance(fold, dict) and isinstance(fold.get("spot_check"), dict)):
            continue
        if fold.get("job_summary_sha256") != want:
            _log.warning(
                "%s was folded against %s; its spot-check is not shown. Fold this job's "
                "review export (full-loop Step 7b) to show it.",
                rel,
                "another labeling job" if fold.get("job_summary_sha256") else "no job summary",
            )
            continue
        return dict(cal, **{FOLD_KEY: fold})
    return cal


def _load_calibration(project, path=None):
    """The job's labeling_summary.json (the certificate), or None.

    ``path`` when given (warned about when it does not load), else :func:`_job_summary`.
    """
    if path:
        cal = _j(path)
        if not isinstance(cal, dict):
            _log.warning(
                "calibration %s is missing or not a JSON object; the certified accuracy "
                "bound is not shown.",
                path,
            )
            return None
        return cal
    return _job_summary(project)


def _no_promise_reason(cal):
    """Why the job's summary carries no certificate, or None when one may be shown.

    A task without an accuracy promise (``_rowrule.no_promise_reason``, from the
    summary's ``Type``), else the job's ``StatusReason`` when it did not certify.
    """
    task = str(cal.get("Type") or "").replace("labeling.", "") or cal.get("task")
    reason = no_promise_reason(task)
    if reason:
        return reason
    if cal.get("Status") not in ("Certified",) and cal.get("StatusReason") not in (
        None,
        "",
        "spot_check_pending",
    ):
        return {
            "insufficient_eval_rows": "fewer than 50 eval rows: the job certified nothing",
            "target_not_met": "the accepted rows' lower bound did not clear the target",
            "no_target": "no accuracy target was sent, so the job routed on the 0.85 bar",
            "no_eval_data": "no evaluation data, so the job routed on the 0.85 bar",
            "task_not_certifiable": "this task ships without an accuracy promise",
        }.get(cal["StatusReason"], f"the job reports {cal['StatusReason']}")
    return None


# The job summary's ConfidenceMethod, as the certificate table names it.
CONFIDENCE_METHOD_TEXT = {
    "agreement": "jury vote share",
    "stated": ("jurors' stated confidence (the 0.75 floor and the 0.85 bar read the vote share)"),
    "verbalized": (
        "one model's verbalized confidence, 0 to 1 (no jury; the 0.75 floor unless "
        "Confidence.FloorBypass)"
    ),
}
ESTIMATOR_TEXT = {
    "judge": "a judge's score, 0 to 1, of one model's label",
    "panel": "the mean of 2 to 3 judges' scores, 0 to 1, of one model's label",
}


def _calibration_block(f, cal):
    """Render the labeling job's certificate from its labeling_summary.json.

    The Status, the accuracy target, the certified (Clopper-Pearson, one-sided) lower
    bound on the auto-accepted rows, the coverage, the cutoff and the spot-check, all
    measured on the reserved test rows. Empty when there is no summary.
    """
    if not isinstance(cal, dict) or not cal.get("Status"):
        return ""
    no_promise = _no_promise_reason(cal)
    status, reason = cal.get("Status"), cal.get("StatusReason")
    target, lb = _num(cal.get("Target")), _num(cal.get("AcceptedLowerBound95"))
    acc_acc, n_acc = _num(cal.get("AcceptedHeldOutAccuracy")), cal.get("AcceptedHeldOutRows")
    cutoff = _num(cal.get("AcceptThreshold"))
    n_eval = cal.get("HeldOutRows")
    n_lab, n_accepted, n_routed = (
        cal.get("RowsLabeled"),
        cal.get("RowsAccepted"),
        cal.get("RowsRouted"),
    )
    spot, withdraw = cal.get("SpotCheckRows"), cal.get("SpotCheckWithdrawAt")
    cov = (n_accepted / n_lab) if _num(n_accepted) is not None and _num(n_lab) else None
    fold = cal.get(FOLD_KEY) if isinstance(cal.get(FOLD_KEY), dict) else None
    sc = (fold or {}).get("spot_check") if fold else None
    # A Certified job's spot-check checks its promise; any other run's (routed on the
    # 0.85 bar: no_target, or no_eval_data with a target) checks the bar, with no promise.
    promise = status == "Certified" and not no_promise
    spot_rule = (
        f"withdraws the promise at {withdraw} or more errors"
        if promise
        else f"the accepted rows miss the 0.85 bar at {withdraw} or more errors"
    )
    rows = [
        ["Job status", f"{status}" + (f" ({reason})" if reason else "")],
        ["Accuracy target (auto-accepted rows)", _pct_or_dash(target)],
        [
            "Certified lower bound (accepted rows)",
            (
                f"{_pct(lb)} (95% one-sided, Clopper-Pearson"
                + (f", {n_acc} test rows)" if n_acc else ")")
                if lb is not None
                else DASH
            ),
        ],
        ["Accepted-row test accuracy", _pct_or_dash(acc_acc)],
        [
            "Coverage (rows auto-accepted)",
            (_pct_or_dash(cov) + (f" ({n_accepted} of {n_lab})" if cov is not None else "")),
        ],
        ["Confidence cutoff", f"{cutoff:g}" if cutoff is not None else "none (nothing accepted)"],
        ["Held-out rows scored", f"{n_eval}" if n_eval is not None else DASH],
        [
            "Spot-check",
            (
                f"{spot} accepted rows, {spot_rule}"
                if spot and withdraw
                else (f"{spot} rows" if spot else "none")
            ),
        ],
        ["Rows routed to review", f"{n_routed}" if n_routed is not None else DASH],
    ]
    # Rows for the opt-in options, shown only when the job's summary carries them.
    extra = []
    if cal.get("ConfidenceMethod") is not None:
        extra.append(
            [
                "Confidence signal",
                ESTIMATOR_TEXT.get(cal.get("ConfidenceEstimator"))
                or CONFIDENCE_METHOD_TEXT.get(
                    cal["ConfidenceMethod"], str(cal["ConfidenceMethod"])
                ),
            ]
        )
    use = cal.get("ConfidenceEstimatorUsage")
    if isinstance(use, dict):
        lower = (
            " (a lower bound: rows restored from a checkpoint are not counted)"
            if isinstance(cal.get("RowsLabeled"), int)
            and (use.get("RowsJudged") or 0) < cal["RowsLabeled"]
            else ""
        )
        extra.append(
            [
                "Confidence judge",
                f"{cal.get('ConfidenceEstimator')}: {use.get('Calls', 0)} "
                f"calls, {use.get('InputTokens', 0)} input and {use.get('OutputTokens', 0)} "
                f"output tokens, {use.get('CallFailures', 0)} failed, "
                f"{use.get('ParseFailures', 0)} unparsed{lower}",
            ]
        )
    rec = cal.get("ConfidenceRecalibration")
    if isinstance(rec, dict):
        extra.append(
            [
                "Confidence recalibration",
                f"{rec.get('Map')} {rec.get('Status')}, a = "
                f"{rec.get('A')}, b = {rec.get('B')}, on {rec.get('FitRows')} fit rows"
                + (
                    f", effective raw floor {rec['EffectiveRawFloor']}"
                    if rec.get("EffectiveRawFloor") is not None
                    else ""
                ),
            ]
        )
    vmissing = verbalized_confidence_missing(cal)
    if vmissing:
        extra.append(
            [
                "Verbalized confidence missing",
                f"{vmissing['missing']} of {vmissing['of']} rows "
                f"({100 * vmissing['rate']:.1f}%)"
                + (
                    "; each counts 0, so with the 0.75 floor on (the default) those rows "
                    "never auto-accept, unless --confidence-recalibration (the floor reads "
                    "the mapped value)"
                    if vmissing["high"]
                    else ""
                ),
            ]
        )
    missing = stated_confidence_missing(cal)
    if missing:
        extra.append(
            [
                "Stated confidence missing",
                f"{missing['missing']} of {missing['of']} {missing['unit']} "
                f"({100 * missing['rate']:.1f}%)"
                + (
                    "; each counts 0 in the row score, so fewer rows auto-accept"
                    if missing["high"]
                    else ""
                ),
            ]
        )
    rows[6:6] = extra
    if isinstance(sc, dict):
        rate = _num(sc.get("agreement_rate"))
        rows.insert(
            3,
            [
                "Spot-check agreement (people vs. model)",
                (
                    (
                        f"{_pct(rate)} ({sc.get('confirmed')} confirmed, {sc.get('errors')} "
                        f"errors of {sc.get('rows')}"
                        + ("" if sc.get("complete", True) else "; incomplete")
                        + ")"
                    )
                    if rate is not None
                    else f"not compared ({sc.get('no_model_answer')} rows with no model answer)"
                ),
            ],
        )
    bar_failed = (
        f.warn(
            f"Spot-check failed the bar: {sc.get('errors')} of {sc.get('rows')} "
            f"spot-check rows were wrong (it fails at {withdraw}), so the "
            "auto-accepted rows do not meet the 0.85 bar. Review more rows, or "
            "fold the review into the seed, refine and run the corpus job again."
        )
        if fold and fold.get("spot_check_failed") and not promise and isinstance(sc, dict)
        else ""
    )
    if no_promise:
        return (
            f.h3("Accuracy guarantee (the labeling job's certificate)")
            + f.table(["Field", "Value"], rows)
            + f.warn(
                f"No accuracy guarantee: {no_promise}. Report the held-out score "
                "instead; rows below the cutoff went to review."
            )
            + bar_failed
        )
    if fold and fold.get("certificate_withdrawn"):
        return (
            f.h3("Accuracy guarantee (the labeling job's certificate)")
            + f.table(["Field", "Value"], rows)
            + f.warn(
                f"Certificate withdrawn: the spot-check found {sc.get('errors')} wrong "
                f"rows of {sc.get('rows')} (the promise is withdrawn at {withdraw}). "
                "Fold the review into the seed and run the corpus job again."
            )
        )
    if status == "Certified":
        msg = (
            f"Certified by the job: with 95% confidence, at least {_pct_or_dash(lb)} of the "
            f"auto-accepted rows are correct, which clears the {_pct_or_dash(target)} "
            f"target, at {_pct_or_dash(cov)} coverage. Measured on the reserved test rows, "
            "so it holds for rows drawn like them."
        )
        if isinstance(sc, dict) and not fold.get("spot_check_failed") and sc.get("complete", True):
            msg += (
                f" The spot-check passed: {sc.get('errors')} wrong rows of {sc.get('rows')}, "
                f"under the {withdraw} that would withdraw it."
            )
        elif reason == "spot_check_pending":
            msg += (
                " The certificate is pending the spot-check: it is withdrawn at "
                f"{withdraw} or more wrong rows out of {spot}."
            )
        return (
            f.h3("Accuracy guarantee (the labeling job's certificate)")
            + f.table(["Field", "Value"], rows)
            + f.p(msg)
        )
    if status == "Partial":
        msg = (
            "The job stopped short: it scored only the rows it labeled and the rest were "
            "routed to review (RowsMissing in the summary). Treat the bound as provisional."
        )
    else:
        msg = (
            f"Not certified ({reason}): the job accepted nothing on a guarantee, so every "
            "model-labeled row is in the review set. Add gold rows, lower the target "
            "(0.85 at the least), or ship labels with no promise."
        )
    return (
        f.h3("Accuracy guarantee (the labeling job's certificate)")
        + f.table(["Field", "Value"], rows)
        + f.warn(msg)
        + bar_failed
    )


def _juror_response_block(f, project):
    """Per-juror missing ballots from the run log's ``confidence_shortfall``.

    With a warning naming each degraded juror. Empty when every juror answered or the job
    resampled one model.
    """
    sf = _confidence_shortfall(project) if project else None
    if (sf or {}).get("ballot_unit") == "sample":
        return ""  # the job resamples one model: no juror to name
    missing = (sf or {}).get("missing_by_juror") or {}
    n = (sf or {}).get("rows")
    if not missing or _num(n) is None or n <= 0:
        return ""
    degraded = sf.get("degraded_jurors") or {}
    rows = [
        [m, f"{c}/{n}", _pct(c / n), "degraded" if m in degraded else "ok"]
        for m, c in sorted(missing.items(), key=lambda kv: -kv[1])
    ]
    body = f.p(
        "Rows where a juror cast no usable ballot, from the labeling run's "
        "record. Those rows were voted by the other jurors only."
    )
    body += f.table(["Juror model", "Missing", "Rate", "Status"], rows)
    if degraded:
        names = ", ".join(
            f"{m} ({d.get('missing')}/{d.get('rows')} rows)" for m, d in degraded.items()
        )
        body += f.warn(
            f"**Degraded jury:** {names}. The SDG output does not say why a "
            "reply was dropped; a model refusal is the usual cause."
        )
    return body


def _review_split_counts(project, summary=None):
    """``({split: rows}, total)`` of the review set.

    RowsRouted as ``low_confidence`` and SpotCheckRows as ``spot_check`` from the summary
    (they do not overlap), else a local review.jsonl's ``split`` tags.
    """
    summary = summary or {}
    routed, spot = summary.get("RowsRouted"), summary.get("SpotCheckRows")
    if isinstance(routed, int):
        counts = {
            k: v
            for k, v in (
                ("low_confidence", routed),
                ("spot_check", spot if isinstance(spot, int) else 0),
            )
            if v
        }
        return counts, sum(counts.values())
    counts = {}
    rows = _jl(os.path.join(project, "review.jsonl"))
    for r in rows:
        key = r.get("split") or r.get("_review_reason") or "routed"
        counts[key] = counts.get(key, 0) + 1
    return counts, len(rows)


def _build_job_sections(f, project, calibration=None):
    """The corpus run's sections.

    From the deliverable record, the job's summary (with the fold's spot-check), the
    held-out report and the run log. No labeled row is read. Returns (sections, rows
    labeled, held-out rows).
    """
    dl = _corpus_deliverable(project)
    cal = _attach_fold_report(_load_calibration(project, calibration), project) or {}
    score = _j(os.path.join(project, "held_out_report.json")) or {}
    n = cal.get("RowsLabeled")
    n_missing = cal.get("RowsMissing") or 0
    tiles = [
        {
            "label": "Rows labeled",
            "value": f"{n}" if n is not None else DASH,
            "detail": f"{n_missing} with no output" if n_missing else "",
        },
        {
            "label": "Auto-accepted",
            "value": (
                f"{cal.get('RowsAccepted')}" if cal.get("RowsAccepted") is not None else DASH
            ),
        },
        {
            "label": "Routed to review",
            "value": f"{cal.get('RowsRouted')}" if cal.get("RowsRouted") is not None else DASH,
        },
        {
            "label": "Held-out accuracy",
            "value": _pct_or_dash(_num(cal.get("HeldOutAccuracy"))),
            "detail": (f"{cal.get('HeldOutRows')} test rows" if cal.get("HeldOutRows") else ""),
        },
        {
            "label": "Job status",
            "value": str(cal.get("Status") or DASH),
            "detail": str(cal.get("StatusReason") or ""),
        },
    ]
    perf = f.strip(tiles) + _calibration_block(f, cal)
    breakdown = score.get("breakdown") or {}
    if breakdown:
        perf += f.h4(
            "Per-class F1 (held-out)" if score.get("task") != "ner" else "Per-type F1 (held-out)"
        )
        perf += f.table(
            ["Class", "F1"], [[k, _pct_or_dash(_num(v))] for k, v in sorted(breakdown.items())]
        )
    pairs = score.get("confusion_pairs") or []
    if pairs:
        # The job writes each pair as {"gt", "pred", "n"} (ner spans stay objects).
        def _cell(v):
            return v if isinstance(v, (str, int, float)) or v is None else json.dumps(v)

        rows = [
            (
                [_cell(p.get("gt")), _cell(p.get("pred")), p.get("n")]
                if isinstance(p, dict)
                else [_cell(v) for v in list(p)[:3]]
            )
            for p in pairs[:10]
        ]
        perf += f.h4("Most confused (held-out)")
        perf += f.table(["Gold", "Predicted", "Rows"], rows)

    counts, n_review = _review_split_counts(project, cal)
    if n_review:
        review = f.p(
            f"The job routed **{n_review}** rows to people (the review DataSet, "
            "blind: no model answer shown; counts from the job's summary). Their "
            "labels and the spot-check verdict come back through the hand-off "
            "(references/full-loop.md, Step 7b)."
        )
        review += f.table(
            ["Why it is in the review set", "Rows"], [[k, v] for k, v in sorted(counts.items())]
        )
    else:
        review = f.p("The job routed no rows to review.")

    deliver = [] if dl.get("dataset_arn") else [["Labeled DataSet", "not registered"]]
    for label, key in (
        ("Labeled DataSet", "dataset_arn"),
        ("Review DataSet", "review_dataset_arn"),
        ("Labels (S3)", "labels_s3_uri"),
        ("Review rows (S3)", "review_rows_s3_uri"),
        ("Job report (S3)", "report_s3_uri"),
        ("Labeling job", "job_arn"),
    ):
        if dl.get(key):
            deliver.append(
                [
                    label,
                    (
                        "raw",
                        f'<span class="mono">{_esc(dl[key])}</span>',
                        f"`{_md_escape(dl[key])}`",
                    ),
                ]
            )
    handoff = f.table(["Deliverable", "Where"], deliver)
    handoff += f.note(
        "The labeled rows stay behind the API: this report reads the job's "
        "summary and its held-out report, never the labeled corpus or the "
        "review rows. Hand the labeled DataSet ARN to fine-tuning / DI / model "
        "customization. People's corrections are applied by a "
        "dataset-transformation overlay by id, whose output becomes that "
        "DataSet's next version (references/full-loop.md, Step 7b)."
    )
    if os.path.exists(os.path.join(project, "job_report.html")):
        handoff += f.p("The job's own report: `job_report.html`.")
    sections = [("Performance", perf), ("Review set", review), ("Handoff", handoff)]
    jury = _juror_response_block(f, project)
    if jury:
        sections.append(("Jury", jury))
    return sections, n or 0, cal.get("HeldOutRows") or 0


def _build_sections(f, project, pred_field, calibration=None):
    """The report's sections.

    From the corpus run's job summary when the run wrote its deliverable record (the
    labeled corpus stays behind the API), else from a local labels.jsonl
    (:func:`_build_local_sections`).

    Raises:
        DataPreparerOperatorError: Neither file is in the project, or labels.jsonl is
            corrupt.
    """
    labels = _jl(os.path.join(project, LOCAL_LABELS), required=True)
    if not labels and _corpus_deliverable(project) is not None:
        return _build_job_sections(f, project, calibration)
    if not labels:
        raise DataPreparerOperatorError(
            "neither labels.jsonl nor the corpus run's labeled_dataset.json is "
            f"in {project}. The labeling run did not complete, so there is no dataset to "
            "report. Rerun the apply step before generating the report."
        )
    return _build_local_sections(f, project, labels, pred_field)


def _build_local_sections(f, project, labels, pred_field):
    """The sections for a local labels.jsonl (the whole-corpus cold start).

    Rows, who labeled them, class balance, samples, columns.
    """
    pred_field = _resolve_pred_field(labels, pred_field)
    n = len(labels)
    people = sum(1 for r in labels if labeled_by_people(r))
    path = os.path.join(project, LOCAL_LABELS)
    if people == n:
        lead = (
            f"Labeled **{n}** rows. All {n} carry labels people gave (the cold-start "
            "sample): no model label ships, and there is no held-out score, "
            "certificate or refined spec."
        )
    else:
        lead = (
            f"Labeled **{n}** rows: {people} labeled by people, {n - people} by the "
            "model. A local labels file carries no held-out score or certificate."
        )
    counts = Counter(r.get(pred_field) for r in labels if isinstance(r.get(pred_field), str))
    balance = ", ".join(f"{k} {c / n:.1%}" for k, c in counts.most_common()) or DASH
    summary = f.p(lead) + f.table(
        ["Field", "Value"],
        [
            [
                "Labeled dataset",
                ("raw", f'<span class="mono">{_esc(path)}</span>', f"`{_md_escape(path)}`"),
            ],
            ["Rows", n],
            ["Labeled by people", people],
            ["Class balance", balance],
        ],
    )
    summary += f.note(
        "Hand this file to fine-tuning / DI / model customization as the "
        "training or eval dataset."
    )
    sections = [("Summary", summary)]
    if counts:
        sections.append(
            (
                "Class distribution",
                f.table(
                    ["Class", "Rows", "%"],
                    [[k, c, f"{c / n:.1%}"] for k, c in counts.most_common()],
                ),
            )
        )

    def _shown(v):
        return v if isinstance(v, str) or v is None else json.dumps(v, ensure_ascii=False)

    sections.append(
        (
            "Samples",
            f.table(
                ["Input", "Label"],
                [[_input_text(r)[:150], _shown(r.get(pred_field))] for r in labels[:5]],
            ),
        )
    )
    cols = [k for k in labels[0] if not k.startswith("_")]
    sections.append(
        (
            "Columns",
            f.p(
                "One JSON object per row, in input order. Columns: "
                + ", ".join(f"`{c}`" for c in cols)
                + "."
            ),
        )
    )
    return sections, n, 0


def _title(dataset):
    return f"Auto-Labeling Report - {dataset}" if dataset else "Auto-Labeling Report"


_SUBTITLE = (
    "A labeled dataset with measured accuracy and per-row confidence, produced "
    "end to end by the auto-labeling skill, the data-producing pre-step into "
    "fine-tuning / DI / model customization."
)
# A run with no held-out score (a standalone apply run, or a cold-start opt-out, whose
# labels carry no accuracy promise) is not described as measured.
_SUBTITLE_UNSCORED = (
    "A labeled dataset with per-row confidence and no measured accuracy, "
    "produced end to end by the auto-labeling skill, the data-producing "
    "pre-step into fine-tuning / DI / model customization."
)
# A local labels.jsonl (the whole-corpus cold start): people's labels, no model confidence.
_SUBTITLE_LOCAL = (
    "A labeled dataset from a local labels file, the data-producing pre-step "
    "into fine-tuning / DI / model customization."
)


def _subtitle(project, gt_path=None, n_gold=None):
    """The report's subtitle.

    _SUBTITLE_LOCAL for a local labels.jsonl, else measured accuracy only when
    :func:`_is_scored`.
    """
    if _jl(os.path.join(project, LOCAL_LABELS), required=True):
        return _SUBTITLE_LOCAL
    return _SUBTITLE if _is_scored(project, gt_path, n_gold) else _SUBTITLE_UNSCORED


def _is_scored(project, gt_path=None, n_gold=None):
    """True when the project has held_out_report.json, or ``gt_path`` is given.

    A given ``gt_path`` counts when ``n_gold`` is None or positive.
    """
    gt_scored = bool(gt_path) and (n_gold is None or n_gold > 0)
    return gt_scored or os.path.exists(os.path.join(project, "held_out_report.json"))


def build_html(  # pylint: disable=too-many-positional-arguments
    project_dir: str,
    dataset: str = "",
    threshold: Optional[float] = None,
    diagram: Optional[str] = None,
    gt: Optional[str] = None,
    gt_field: Optional[str] = None,
    pred_field: str = "label",
    task: Optional[str] = None,
    calibration: Optional[str] = None,
    qa_match: Optional[str] = None,
) -> str:
    """Render the project's run report as a self-contained HTML document.

    The corpus run's report is built from ``labeled_dataset.json``, the job's
    ``labeling_summary.json`` (with the fold's spot-check when the fold report names this
    job), ``held_out_report.json`` and the run log; a local ``labels.jsonl`` (the
    whole-corpus cold start) wins over them. No row of the labeled corpus is read.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import build_html

        page = build_html("tickets", dataset="tickets-labeled")

    Args:
        project_dir: The project folder holding the run outputs.
        dataset: The dataset name shown in the title.
        threshold: Accepted for earlier callers; the report shows the job's own cutoff
            (``AcceptThreshold``).
        diagram: Accepted for earlier callers; not rendered.
        gt: A gold file. It is never read; when given, the run is described as measured.
        gt_field: Accepted for earlier callers.
        pred_field: The prediction field of a local ``labels.jsonl`` (a task's output
            field is used when no row has ``label``).
        task: Accepted for earlier callers; the job graded the task.
        calibration: The labeling job's ``labeling_summary.json``. Default:
            ``<project_dir>/labeling_summary.json``, else the corpus run's run-log entry.
        qa_match: Accepted for earlier callers; the certificate is the job's.

    Returns:
        The HTML document.

    Raises:
        DataPreparerOperatorError: Neither ``labels.jsonl`` nor the corpus run's
            ``labeled_dataset.json`` is in the project (the run did not complete), or
            ``labels.jsonl`` has a malformed line.
    """
    f = Fmt("html")
    sections, _n, n_gold = _build_sections(f, project_dir, pred_field, calibration)
    return f.document(_title(dataset), _subtitle(project_dir, gt, n_gold), sections)


def build_markdown(  # pylint: disable=too-many-positional-arguments
    project_dir: str,
    dataset: str = "",
    threshold: Optional[float] = None,
    diagram: Optional[str] = None,
    gt: Optional[str] = None,
    gt_field: Optional[str] = None,
    pred_field: str = "label",
    task: Optional[str] = None,
    calibration: Optional[str] = None,
    qa_match: Optional[str] = None,
) -> str:
    """Render the project's run report as Markdown, from the same sections as the HTML.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import build_markdown

        text = build_markdown("tickets", dataset="tickets-labeled")

    Args:
        project_dir: The project folder holding the run outputs.
        dataset: The dataset name shown in the title.
        threshold: Accepted for earlier callers; the report shows the job's own cutoff.
        diagram: Accepted for earlier callers; not rendered.
        gt: A gold file. It is never read; when given, the run is described as measured.
        gt_field: Accepted for earlier callers.
        pred_field: The prediction field of a local ``labels.jsonl``.
        task: Accepted for earlier callers; the job graded the task.
        calibration: The labeling job's ``labeling_summary.json`` (default: the
            project's, else the corpus run's run-log entry).
        qa_match: Accepted for earlier callers; the certificate is the job's.

    Returns:
        The Markdown document.

    Raises:
        DataPreparerOperatorError: The run did not complete (see :func:`build_html`), or
            ``labels.jsonl`` has a malformed line.
    """
    f = Fmt("md")
    sections, _n, n_gold = _build_sections(f, project_dir, pred_field, calibration)
    return f.document(_title(dataset), _subtitle(project_dir, gt, n_gold), sections)


def write_report(
    project_dir: str,
    *,
    dataset: str = "",
    fmt: str = "both",
    out: Optional[str] = None,
    calibration: Optional[str] = None,
    pred_field: str = "label",
    gt_field: Optional[str] = None,
    task: Optional[str] = None,
    qa_match: Optional[str] = None,
    threshold: Optional[float] = None,
    diagram: Optional[str] = None,
    gt: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the run report to ``<project_dir>/report.html`` and/or ``report.md``.

    Both documents are built before anything is written, so a run that did not complete
    writes no file. With ``fmt="both"``, ``out`` names the directory both files go into
    (``report.html`` and ``report.md``); it is created when missing, a path that is or
    looks like a file is refused, and when the second write fails the first file is removed
    again (and the directory, when this call created it and it is empty), so no stray
    report is left for a reader to mistake for a finished one. NER predictions live under
    ``entities``: with ``gt_field`` ``gt_entities`` and the default ``pred_field``,
    ``entities`` is read. ``gt`` is
    passed to the builders as is; checking it against the frozen split is the caller's
    (``guard_read`` with the ``dev``, ``val`` and ``test`` splits, then
    ``log_test_read`` once this returns).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import write_report

        written = write_report("tickets", dataset="tickets-labeled")
        written["html"], written["md"]

    Args:
        project_dir: The project folder holding the run outputs.
        dataset: The dataset name shown in the title.
        fmt: ``html``, ``md`` or ``both``.
        out: The output path for a single format, or the output directory for ``both``;
            ``None`` writes into ``project_dir``.
        calibration: The labeling job's ``labeling_summary.json`` (default: the
            project's, else the corpus run's run-log entry).
        pred_field: The prediction field of a local ``labels.jsonl``.
        gt_field: Accepted for earlier callers (``gt_entities`` selects ``entities``).
        task: Accepted for earlier callers; one of the labeling tasks.
        qa_match: Accepted for earlier callers; a value that is not a QA rule is refused.
        threshold: Accepted for earlier callers; the report shows the job's own cutoff.
        diagram: Accepted for earlier callers; not rendered.
        gt: A gold file, passed to the builders (see :func:`build_html`).

    Returns:
        ``{"wrote": [paths], "html": path or None, "md": path or None}``.

    Raises:
        DataPreparerOperatorError: ``fmt``, ``task`` or ``qa_match`` is not a value this
            accepts, ``out`` is a file with ``fmt="both"``, or the run did not complete
            (nothing is written).
    """
    if fmt not in FORMATS:
        raise DataPreparerOperatorError(f"fmt must be one of {', '.join(FORMATS)}, not {fmt!r}")
    if task is not None and task not in TASKS:
        raise DataPreparerOperatorError(f"task must be one of {', '.join(TASKS)}, not {task!r}")
    # NER predictions live under `entities`, not the default `label`.
    if gt_field == "gt_entities" and pred_field == "label":
        pred_field = "entities"
    if qa_match is not None:  # a bad qa_match rule is still refused
        parse_qa_match(qa_match)

    out_dir = project_dir
    created_dir = False
    if fmt == "both" and out:
        if os.path.isfile(out) or out.lower().endswith((".html", ".md")):
            raise DataPreparerOperatorError(
                f"out {out!r} is a file; with fmt 'both' it is the directory both reports go into "
                "(pass fmt 'html' or 'md' to name one report file)"
            )
        out_dir = out
    html_out = md_out = None
    if fmt in ("html", "both"):
        html_out = out if (out and fmt == "html") else os.path.join(out_dir, "report.html")
    if fmt in ("md", "both"):
        md_out = out if (out and fmt == "md") else os.path.join(out_dir, "report.md")

    # Build first: the completion gate raises before anything is written.
    args = (project_dir, dataset, threshold, diagram, gt, gt_field, pred_field, task)
    html_doc = build_html(*args, calibration, qa_match) if html_out else None
    md_doc = build_markdown(*args, calibration, qa_match) if md_out else None

    if fmt == "both" and out and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        created_dir = True
    wrote: List[str] = []
    try:
        if html_out:
            with open(html_out, "w") as fp:
                fp.write(html_doc)
            wrote.append(html_out)
        if md_out:
            with open(md_out, "w") as fp:
                fp.write(md_doc)
            wrote.append(md_out)
    except OSError:
        # A second write that fails must not leave the first report behind.
        for path in wrote:
            if os.path.exists(path):
                os.remove(path)
        if created_dir and os.path.isdir(out_dir) and not os.listdir(out_dir):
            os.rmdir(out_dir)
        raise
    return {"wrote": wrote, "html": html_out, "md": md_out}
