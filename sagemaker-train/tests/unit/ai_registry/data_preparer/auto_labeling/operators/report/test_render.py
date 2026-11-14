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
"""Tests for the end-of-loop run report (operators.report.render).

The report has two cases:
  - the corpus run: built from labeled_dataset.json, the job's labeling_summary.json,
    held_out_report.json and the run log, never from a labeled row;
  - a local labels.jsonl (the whole-corpus cold start, people labeled every row).
With neither file it refuses. HTML and Markdown come from the same sections, and data in
a cell is escaped, never read as markup.
"""

from __future__ import absolute_import

import json
import os

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.report import render as report
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

SUMMARY = {
    "Status": "Certified",
    "StatusReason": "",
    "RowsLabeled": 1000,
    "RowsMissing": 3,
    "RowsAccepted": 820,
    "RowsRouted": 150,
    "SpotCheckRows": 29,
    "SpotCheckWithdrawAt": 7,
    "HeldOutAccuracy": 0.93,
    "HeldOutRows": 120,
    "AcceptThreshold": 0.86,
    "Target": 0.9,
    "AcceptedHeldOutAccuracy": 0.95,
    "AcceptedLowerBound95": 0.92,
    "Type": "labeling.classification",
    "ConfidenceMethod": "agreement",
}
DELIVERABLE = {
    "dataset_name": "ds",
    "dataset_arn": "arn:aws:sagemaker:us-west-2:111122223333:hub-content/h/DataSet/ds/1.0.0",
    "labels_s3_uri": "s3://b/p/job/output/labels.jsonl",
    "job_arn": "arn:aws:sagemaker:us-west-2:111122223333:job/sdg/x",
}


def _write(path, obj, jsonl=False):
    with open(path, "w") as f:
        if jsonl:
            for r in obj:
                f.write(json.dumps(r) + "\n")
        else:
            json.dump(obj, f)


def _read(path):
    with open(path) as f:
        return f.read()


@pytest.fixture
def proj(tmp_path):
    return str(tmp_path)


def _html(proj, **kw):
    return report.build_html(
        proj, kw.get("dataset", "t"), 0.85, None, None, None, kw.get("pred_field", "label")
    )


def _md(proj, **kw):
    return report.build_markdown(
        proj, kw.get("dataset", "t"), 0.85, None, None, None, kw.get("pred_field", "label")
    )


# --- the completion gate ---------------------------------------------------------------


def test_fails_loud_with_neither_file(proj):
    with pytest.raises(DataPreparerOperatorError) as e:
        _html(proj)
    assert "labeled_dataset.json" in str(e.value)


def test_fails_loud_on_empty_labels(proj):
    _write(os.path.join(proj, "labels.jsonl"), [], jsonl=True)
    with pytest.raises(DataPreparerOperatorError):
        _md(proj)


def test_corrupt_labels_line_fails_loud_naming_the_line(proj):
    with open(os.path.join(proj, "labels.jsonl"), "w") as f:
        f.write('{"id": "0", "label": "a"}\n{not json\n')
    with pytest.raises(DataPreparerOperatorError) as e:
        _html(proj)
    assert "line 2" in str(e.value)


def test_write_report_writes_nothing_when_the_gate_fails(proj):
    with pytest.raises(DataPreparerOperatorError):
        report.write_report(proj)
    assert not os.path.exists(os.path.join(proj, "report.html"))
    assert not os.path.exists(os.path.join(proj, "report.md"))


def test_jl_reads_a_directory_as_no_rows(proj):
    assert report._jl(proj) == []


# --- the corpus run: the job's summary -------------------------------------------------


@pytest.fixture
def job_proj(proj):
    _write(os.path.join(proj, "labeled_dataset.json"), DELIVERABLE)
    _write(os.path.join(proj, "labeling_summary.json"), SUMMARY)
    _write(
        os.path.join(proj, "held_out_report.json"),
        {
            "task": "classification",
            "breakdown": {"a": 0.9, "b": 0.8},
            "confusion_pairs": [{"gt": "a", "pred": "b", "n": 4}],
        },
    )
    return proj


def test_sections_and_certificate(job_proj):
    md = _md(job_proj)
    for title in ("## Performance", "## Review set", "## Handoff"):
        assert title in md
    assert "Certified by the job" in md
    assert "92.0%" in md  # the accepted rows' lower bound
    assert "Per-class F1 (held-out)" in md
    assert "| a | b | 4 |" in md  # the confusion pair
    assert DELIVERABLE["dataset_arn"] in md


def test_review_counts_come_from_the_summary(job_proj):
    md = _md(job_proj)
    assert "**179** rows" in md  # 150 routed + 29 spot-check
    assert "| low_confidence | 150 |" in md
    assert "| spot_check | 29 |" in md


def test_review_counts_fall_back_to_a_local_review_file(job_proj):
    _write(
        os.path.join(job_proj, "labeling_summary.json"),
        {k: v for k, v in SUMMARY.items() if k != "RowsRouted"},
    )
    _write(
        os.path.join(job_proj, "review.jsonl"),
        [{"id": "1", "split": "low_confidence"}, {"id": "2"}],
        jsonl=True,
    )
    md = _md(job_proj)
    assert "**2** rows" in md
    assert "| routed | 1 |" in md


def test_the_cutoff_is_the_jobs_whatever_threshold_is_passed(job_proj):
    assert "| Confidence cutoff | 0.86 |" in _md(job_proj)
    report.write_report(job_proj, threshold=0.5)
    md = _read(os.path.join(job_proj, "report.md"))
    assert "| Confidence cutoff | 0.86 |" in md
    assert "0.5" not in md


def test_null_cutoff_says_nothing_was_accepted(job_proj):
    _write(os.path.join(job_proj, "labeling_summary.json"), dict(SUMMARY, AcceptThreshold=None))
    assert "none (nothing accepted)" in _md(job_proj)


def test_not_certified_job_warns(job_proj):
    _write(
        os.path.join(job_proj, "labeling_summary.json"),
        dict(SUMMARY, Status="NotCertified", StatusReason="target_not_met"),
    )
    md = _md(job_proj)
    assert "did not clear the target" in md
    assert "Certified by the job" not in md


def test_subtitle_says_measured_when_scored(job_proj):
    assert report._SUBTITLE in _md(job_proj)
    os.remove(os.path.join(job_proj, "held_out_report.json"))
    assert report._SUBTITLE_UNSCORED in _md(job_proj)


def test_write_report_writes_both_formats(job_proj):
    res = report.write_report(job_proj)
    md = _read(os.path.join(job_proj, "report.md"))
    assert "# Auto-Labeling Report" in md
    assert "## Contents" in md
    assert "<!doctype html>" in _read(os.path.join(job_proj, "report.html"))
    html_path = os.path.join(job_proj, "report.html")
    md_path = os.path.join(job_proj, "report.md")
    assert res == {"wrote": [html_path, md_path], "html": html_path, "md": md_path}


def test_write_report_single_format_to_out(job_proj):
    out = os.path.join(job_proj, "out.html")
    res = report.write_report(job_proj, fmt="html", out=out)
    assert "<!doctype html>" in _read(out)
    assert not os.path.exists(os.path.join(job_proj, "report.md"))
    assert res == {"wrote": [out], "html": out, "md": None}


def test_earlier_arguments_are_still_accepted(job_proj):
    report.write_report(
        job_proj,
        task="ordinal",
        pred_field="score",
        gt_field="gt_score",
        threshold=0.9,
        qa_match="exact",
        diagram=os.path.join(job_proj, "missing.png"),
    )
    assert os.path.exists(os.path.join(job_proj, "report.md"))


def test_bad_qa_match_is_refused(job_proj):
    with pytest.raises(DataPreparerOperatorError):
        report.write_report(job_proj, qa_match="fuzzy")
    assert not os.path.exists(os.path.join(job_proj, "report.md"))


def test_no_em_dash_or_vendor_names(job_proj):
    for doc in (_html(job_proj), _md(job_proj)):
        assert "\u2014" not in doc
        for bad in ("claude", "anthropic", "opus", "sonnet", "haiku"):
            assert bad not in doc.lower()


# --- a local labels.jsonl (the whole-corpus cold start) --------------------------------


@pytest.fixture
def local_proj(proj):
    rows = [
        {
            "id": str(i),
            "text": f"row {i}",
            "label": "a" if i < 3 else "b",
            "provenance": "human_verified",
            "_labeled_by": "people",
        }
        for i in range(4)
    ]
    _write(os.path.join(proj, "labels.jsonl"), rows, jsonl=True)
    return proj


def test_local_sections(local_proj):
    md = _md(local_proj)
    for title in ("## Summary", "## Class distribution", "## Samples", "## Columns"):
        assert title in md
    assert "All 4 carry labels people gave" in md
    assert "a 75.0%, b 25.0%" in md
    assert report._SUBTITLE_LOCAL in md
    assert "_labeled_by" not in md.split("## Columns")[1]


def test_counts_model_rows_apart(local_proj):
    _write(
        os.path.join(local_proj, "labels.jsonl"),
        [
            {
                "id": "0",
                "text": "x",
                "label": "a",
                "provenance": "human_verified",
                "_labeled_by": "people",
            },
            {"id": "1", "text": "y", "label": "b"},
        ],
        jsonl=True,
    )
    assert "1 labeled by people, 1 by the model" in _md(local_proj)


def test_local_file_wins_over_a_deliverable_record(local_proj):
    _write(os.path.join(local_proj, "labeled_dataset.json"), DELIVERABLE)
    md = _md(local_proj)
    assert "## Summary" in md
    assert "## Handoff" not in md


def test_task_prediction_field_is_found(local_proj):
    _write(os.path.join(local_proj, "labels.jsonl"), [{"id": "0", "text": "x", "score": 4}], True)
    assert "| x | 4 |" in _md(local_proj)


def test_resolve_pred_field():
    assert report._resolve_pred_field([{"answer": "a"}], "label") == "answer"
    assert report._resolve_pred_field([{"score": 4}], "label") == "score"
    assert report._resolve_pred_field([{"label": "a"}], "label") == "label"
    # a present-but-null label keeps `label`
    assert report._resolve_pred_field([{"label": None, "score": 4}], "label") == "label"


def test_input_text():
    assert report._input_text({"text": "hi", "label": "a"}) == "hi"
    assert report._input_text({"question": "Q?", "context": "C"}) == "Q? | C"
    assert report._input_text({"id": "1", "foo": "bar", "label": "a"}) == "bar"
    assert report._input_text({"id": "1", "label": "a", "confidence_score": 0.9}) == ""


# --- escaping: data in a cell is escaped only; ("raw", html, md) cells are authored -----


def test_html_data_cell_not_read_as_markup():
    tbl = report.Fmt("html").table(["A"], [["buy 2 ** get 1 free and `code` <b>"]])
    assert "<b>" not in tbl
    assert "<code>" not in tbl
    assert "**" in tbl
    assert "&lt;b&gt;" in tbl


def test_md_cell_is_one_row():
    tbl = report.Fmt("md").table(["A", "B"], [["x ** y | z\r\nw `c`", "ok"]])
    rows = [ln for ln in tbl.splitlines() if ln.startswith("| x")]
    assert len(rows) == 1
    assert "\\*\\*" in rows[0]
    assert "\\|" in rows[0]
    assert "\\`c\\`" in rows[0]


def test_raw_cell_passes_through_per_format():
    cell = ("raw", "<i>h</i>", "*m*")
    assert "<td><i>h</i></td>" in report.Fmt("html").table(["A"], [[cell]])
    assert "| *m* |" in report.Fmt("md").table(["A"], [[cell]])


def test_md_escape():
    assert report._md_escape("a|b\nc*d`e") == "a\\|b c\\*d\\`e"
    assert report._md_escape("back\\slash") == "back\\\\slash"
    assert "\\[click]" in report._md_escape("see [click](http://evil.example)")
    assert "\\_" not in report._md_escape("snake_case_label")


def test_inline_html_authored_markup():
    assert report._inline_html("**b** `c` <x>") == "<b>b</b> <code>c</code> &lt;x&gt;"


# --- fmt "both" with an out directory -------------------------------------------------


def _listing(d):
    return {
        n: _read(os.path.join(d, n))
        for n in sorted(os.listdir(d))
        if os.path.isfile(os.path.join(d, n))
    }


def test_both_into_an_out_dir_leaves_the_project_untouched(job_proj, tmp_path):
    before = _listing(job_proj)
    out = os.path.join(str(tmp_path), "reports", "nested")  # not created yet
    res = report.write_report(job_proj, out=out)
    assert sorted(os.listdir(out)) == ["report.html", "report.md"]
    assert "# Auto-Labeling Report" in _read(os.path.join(out, "report.md"))
    assert res == {
        "wrote": [os.path.join(out, "report.html"), os.path.join(out, "report.md")],
        "html": os.path.join(out, "report.html"),
        "md": os.path.join(out, "report.md"),
    }
    assert _listing(job_proj) == before


def test_both_into_an_existing_out_dir(job_proj, tmp_path):
    out = str(tmp_path / "existing")
    os.makedirs(out)
    report.write_report(job_proj, out=out)
    assert sorted(os.listdir(out)) == ["report.html", "report.md"]


@pytest.mark.parametrize("name", ["report.html", "r.md", "existing-file"])
def test_both_refuses_an_out_that_is_a_file(job_proj, tmp_path, name):
    out = str(tmp_path / name)
    if name == "existing-file":
        with open(out, "w") as fp:
            fp.write("x")
    before = _listing(job_proj)
    with pytest.raises(DataPreparerOperatorError, match="is a file"):
        report.write_report(job_proj, out=out)
    assert _listing(job_proj) == before
    assert not os.path.exists(out) or name == "existing-file"  # nothing created, nothing clobbered


def test_both_incomplete_run_creates_no_out_dir(proj, tmp_path):
    # the completion gate raises before anything is written: no directory appears either
    out = str(tmp_path / "out")
    with pytest.raises(DataPreparerOperatorError):
        report.write_report(proj, out=out)
    assert not os.path.exists(out)


def test_both_partial_write_removes_the_first_report(job_proj, tmp_path):
    # OUT exists already; a directory where report.md goes makes the second write fail.
    out = str(tmp_path / "out")
    os.makedirs(os.path.join(out, "report.md"))
    with pytest.raises(OSError):
        report.write_report(job_proj, out=out)
    assert os.path.isdir(out)  # pre-existing: never removed
    assert not os.path.exists(os.path.join(out, "report.html"))


def test_both_partial_write_removes_a_dir_this_call_created(job_proj, tmp_path, monkeypatch):
    out = str(tmp_path / "fresh")
    real_open = open

    def failing_md(path, *a, **k):
        if str(path).endswith("report.md"):
            raise OSError("disk full")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", failing_md)
    with pytest.raises(OSError, match="disk full"):
        report.write_report(job_proj, out=out)
    assert not os.path.exists(out)


def test_single_format_out_is_still_the_file(job_proj, tmp_path):
    out = str(tmp_path / "r.md")
    res = report.write_report(job_proj, fmt="md", out=out)
    assert os.path.exists(out) and res["md"] == out and res["html"] is None
    assert not os.path.exists(os.path.join(job_proj, "report.md"))
