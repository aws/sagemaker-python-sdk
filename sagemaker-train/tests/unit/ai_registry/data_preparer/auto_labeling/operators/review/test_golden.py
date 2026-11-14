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
"""Unit tests for operators.auto_labeling.golden_review: route, apply, check_reviewed on a project."""

from __future__ import absolute_import

import hashlib
import json
import os
from pathlib import Path

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import golden as G
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split as split_state

TEST = [{"id": f"t{i}", "text": f"ticket {i}", "gt_label": "a" if i % 2 else "b"} for i in range(4)]


def _jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


class _Proj:
    """A project with a frozen split (dev/val/test), a routed file and an output folder."""

    def __init__(self, root, freeze=True, test=TEST):
        self.root = str(root)
        self.splits = os.path.join(self.root, "splits")
        _jsonl(
            os.path.join(self.splits, "dev.jsonl"), [{"id": "d0", "text": "dev 0", "gt_label": "a"}]
        )
        _jsonl(
            os.path.join(self.splits, "val.jsonl"), [{"id": "v0", "text": "val 0", "gt_label": "b"}]
        )
        self.test = _jsonl(os.path.join(self.splits, "test.jsonl"), test)
        if freeze:
            res = split_state.freeze(self.splits, ["text"])
            Path(res.state_path).write_text(json.dumps(res.state), encoding="utf-8")
        self.routed = _jsonl(
            os.path.join(self.root, "audit", "golden", "routed.jsonl"),
            [
                {"id": "t1", "checks": ["keyword_rules", "guideline_judge"]},
                {"id": "t2", "checks": ["keyword_rules"]},
            ],
        )
        self.out_dir = os.path.join(self.root, "golden_review")
        self.manifest = os.path.join(self.out_dir, G.MANIFEST_FILE)
        self.reviewed = os.path.join(self.out_dir, G.REVIEWED_FILE)

    def route(self, **kw):
        kw.setdefault("project_dir", self.root)
        res = G.route(self.test, self.routed, self.out_dir, ["text"], **kw)
        os.makedirs(self.out_dir, exist_ok=True)
        Path(self.out_dir, G.REVIEW_FILE).write_text(res.review_text, encoding="utf-8")
        Path(self.manifest).write_text(G.manifest_text(res.manifest), encoding="utf-8")
        return res

    def apply(self, rows, **kw):
        export = _jsonl(os.path.join(self.root, "export.jsonl"), rows)
        kw.setdefault("export_id_field", "record_id")
        kw.setdefault("export_label_field", "corrected_completion")
        kw.setdefault("project_dir", self.root)
        res = G.apply(self.manifest, export, **kw)
        Path(res.reviewed_path).write_text(res.reviewed_text, encoding="utf-8")
        Path(self.manifest).write_text(G.manifest_text(res.manifest), encoding="utf-8")
        return res

    def check(self, step="certificate"):
        return G.check_reviewed(self.manifest, self.reviewed, step, self.root)

    def rewrite_manifest(self, **changes):
        m = json.loads(Path(self.manifest).read_text())
        m.update(changes)
        Path(self.manifest).write_text(G.manifest_text(m), encoding="utf-8")
        return m

    def rewrite_copy(self, rows):
        text = "".join(json.dumps(r) + "\n" for r in rows)
        Path(self.reviewed).write_text(text, encoding="utf-8")
        self.rewrite_manifest(reviewed_sha256=hashlib.sha256(text.encode()).hexdigest())


@pytest.fixture
def proj(tmp_path):
    return _Proj(tmp_path)


def _person(rid, label, who="alice", **extra):
    return {"record_id": rid, "corrected_completion": label, "annotator": who, **extra}


# ---- route ------------------------------------------------------------------------------


def test_route_writes_a_blind_review_and_the_manifest(proj):
    res = proj.route()
    assert res.review_rows == [
        {"id": "t1", "text": "ticket 1", "_review_reason": "golden flag"},
        {"id": "t2", "text": "ticket 2", "_review_reason": "golden flag"},
    ]
    assert res.review_text == "".join(json.dumps(r) + "\n" for r in res.review_rows)
    m = res.manifest
    assert m["version"] == 1 and m["eval"] == proj.test and m["routed_file"] == proj.routed
    assert m["eval_sha256"] == _digest.sha256_file(proj.test)
    assert m["golden_rows"] == 4 and m["id_field"] == "id" and m["label_field"] == "gt_label"
    assert m["routed"] == {
        "t1": {"original": "a", "checks": ["guideline_judge", "keyword_rules"]},
        "t2": {"original": "b", "checks": ["keyword_rules"]},
    }
    assert m["routed_per_check"] == {"guideline_judge": 1, "keyword_rules": 2}
    assert m["review"] == os.path.join(proj.out_dir, "review.jsonl")
    assert m["label_set"] == ["a", "b"] and m["label_set_given"] is False
    # the test read is returned, not written
    (ev,) = res.events
    assert (
        ev["kind"] == "analysis" and ev["status"] == "test-read" and ev["step"] == "golden-review"
    )
    assert ev["rows"] == 4 and ev["file"] == proj.test
    assert G.manifest_text(m).endswith("\n") and json.loads(G.manifest_text(m)) == m
    assert G.route_summary(m, proj.manifest, "e1") == {
        "golden_rows": 4,
        "routed": 2,
        "routed_per_check": m["routed_per_check"],
        "review": m["review"],
        "manifest": proj.manifest,
        "run_log_entry": "e1",
    }


def test_route_with_a_label_set_and_without_a_frozen_split(tmp_path):
    p = _Proj(tmp_path / "free", freeze=False)
    res = p.route(label_set=["a", "b", "c"])
    assert res.manifest["label_set"] == ["a", "b", "c"] and res.manifest["label_set_given"] is True
    assert res.events == []  # no split, no guarded read


def test_route_refusals(tmp_path):
    p = _Proj(tmp_path)
    for fields, match in (
        (["text", "gt_label"], "names 'gt_label', which is gold"),
        (["gt_other"], "names 'gt_other'"),
        (["provenance"], "names 'provenance', which is gold, metadata or a model answer"),
        (["reasoning"], "names 'reasoning'"),
    ):
        with pytest.raises(G.GoldenReviewError, match=match):
            G.route(p.test, p.routed, p.out_dir, fields, project_dir=p.root)
    with pytest.raises(G.GoldenReviewError, match="is not the frozen test split"):
        G.route(
            os.path.join(p.splits, "dev.jsonl"), p.routed, p.out_dir, ["text"], project_dir=p.root
        )
    for routed, match in (
        ([{"id": "nope", "checks": ["x"]}], r"row 1 \('nope'\) is not a new --eval row"),
        ([{"id": "t1", "checks": []}], r"row 1 \('t1'\)"),
        ([{"id": "t1", "checks": ["x"]}, {"id": "t1", "checks": ["y"]}], r"row 2 \('t1'\)"),
        ([{"id": "t1", "checks": "x"}], "row 1"),
    ):
        _jsonl(p.routed, routed)
        with pytest.raises(G.GoldenReviewError, match=match):
            G.route(p.test, p.routed, p.out_dir, ["text"], project_dir=p.root)
    ints = _Proj(tmp_path / "ints", test=[dict(r, gt_label=i) for i, r in enumerate(TEST)])
    with pytest.raises(G.GoldenReviewError, match="golden labels are not strings"):
        ints.route()
    assert ints.route(label_set=["0", "1"]).manifest["label_set"] == ["0", "1"]
    bad = _jsonl(tmp_path / "bad.jsonl", [{"id": "t1"}])
    Path(bad).write_text('{"id": "t1"}\n[1, 2]\n')
    with pytest.raises(G.GoldenReviewError, match="bad.jsonl:2 is not a JSON object"):
        G.read_lines(bad)
    assert G.read_lines(_jsonl(tmp_path / "ok.jsonl", [{"a": 1}]))[0] == ['{"a": 1}\n']
    with pytest.raises(G.GoldenReviewError, match="row 2 has no 'id'"):
        G._ids([{"id": "a"}, {"x": 1}], "id", "f")
    with pytest.raises(G.GoldenReviewError, match="duplicate ids"):
        G._ids([{"id": "a"}, {"id": "a"}], "id", "f")
    assert G._ids([{"id": 1}, {"id": "b"}], "id", "f") == ["1", "b"]


# ---- apply ------------------------------------------------------------------------------


def test_apply_folds_the_persons_export(proj):
    proj.route()
    res = proj.apply(
        [_person("t1", "b"), _person("t2", "b", who="bob")]
    )  # t1 relabeled, t2 confirmed
    m = res.manifest
    assert m["resolutions"] == {
        "t1": {"label": "b", "changed": True, "annotators": ["alice"]},
        "t2": {"label": "b", "changed": False, "annotators": ["bob"]},
    }
    assert (m["unresolved"], m["conflicting"], m["invalid_labels"], m["excluded"]) == (
        [],
        [],
        [],
        ["t1"],
    )
    assert m["export_id_field"] == "record_id" and m["export_label_field"] == "corrected_completion"
    assert m["export"].endswith("export.jsonl") and m["export_sha256"] == _digest.sha256_file(
        m["export"]
    )
    assert m["reviewed"] == res.reviewed_path == proj.reviewed
    assert m["reviewed_sha256"] == hashlib.sha256(res.reviewed_text.encode()).hexdigest()
    rows = [json.loads(x) for x in res.reviewed_text.splitlines()]
    assert [r["id"] for r in rows] == ["t0", "t2", "t3"]  # t1 left the golden rows
    assert rows[0] == {**TEST[0], "_golden_review": {"resolved_by": None}}
    assert rows[1] == {
        **TEST[2],
        "provenance": "human_verified",
        "_golden_review": {"original": "b", "resolved_by": "person", "annotators": ["bob"]},
    }
    assert len(res.events) == 1 and res.events[0]["status"] == "test-read"
    assert G.apply_summary(m, "e2") == {
        "golden_rows": 4,
        "routed": 2,
        "changed": 1,
        "confirmed": 1,
        "unresolved": 0,
        "conflicting": 0,
        "invalid_labels": 0,
        "excluded_from_golden_rows": 1,
        "reviewed": proj.reviewed,
        "run_log_entry": "e2",
    }
    check = proj.check()
    assert check.to_json() == {"golden_rows": 4, "changed": 1, "confirmed": 1, "unresolved": 0}
    assert check.events[0]["step"] == "certificate"
    assert G.has_mark(rows) and not G.has_mark(TEST)


def test_apply_unresolved_invalid_conflicting_and_label_studio(proj):
    proj.route()
    rows = [
        {
            "record_id": "t1",
            "corrected_completion": "a",
            "status": "completed",
        },  # confirmed, no reviewer named
        _person("t2", "zzz"),  # not in the label set
        {
            "data": {"record_id": "t2", "text": "x"},
            "annotations": [  # a Label Studio task, two annotations
                {
                    "annotator": "c",
                    "result": [
                        {
                            "from_name": "corrected_completion",
                            "type": "choices",
                            "value": {"choices": ["a"]},
                        }
                    ],
                },
                {
                    "annotator": "d",
                    "result": [
                        {
                            "from_name": "corrected_completion",
                            "type": "choices",
                            "value": {"choices": ["b"]},
                        }
                    ],
                },
            ],
        },
    ]
    res = proj.apply(rows)
    m = res.manifest
    assert m["resolutions"] == {"t1": {"label": "a", "changed": False, "annotators": []}}
    assert m["unresolved"] == ["t2"] and m["conflicting"] == ["t2"] and m["invalid_labels"] == []
    assert [json.loads(x)["id"] for x in res.reviewed_text.splitlines()] == ["t0", "t1", "t2", "t3"]
    assert proj.check().to_json() == {
        "golden_rows": 4,
        "changed": 0,
        "confirmed": 1,
        "unresolved": 1,
    }
    # an unsure return, an invalid label alone, a blank label
    res = proj.apply([_person("t1", "b", status="unsure"), _person("t2", "zzz"), _person("t2", "")])
    assert res.manifest["invalid_labels"] == ["t2"] and res.manifest["unresolved"] == ["t1", "t2"]


def test_apply_refusals(proj):
    proj.route()
    for rows, match in (
        (
            [_person("t1", "b", provenance="auto_approved")],
            "has a row marked auto_approved: a model's label",
        ),
        (
            [{"record_id": "t1", "corrected_completion": "a"}],
            r"1 returned row\(s\) name no reviewer and no completed",
        ),
        ([_person("t0", "a")], r"1 export id\(s\) were not routed \(e.g. \['t0'\]\)"),
        (
            [{"record_id": "t1", "corrected_completion": "b", "status": "completed"}],
            "golden row 't1' is relabeled with no reviewer named",
        ),
    ):
        with pytest.raises(G.GoldenReviewError, match=match):
            proj.apply(rows)
    Path(proj.test).write_text(Path(proj.test).read_text() + "\n")
    with pytest.raises(G.GoldenReviewError, match="changed since route; run route again"):
        proj.apply([_person("t1", "a")])
    Path(proj.manifest).write_text('{"version": 2}')
    with pytest.raises(
        G.GoldenReviewError, match=r"is not a golden_review.py manifest \(version 1\)"
    ):
        proj.apply([_person("t1", "a")])
    assert isinstance(G.GoldenReviewError("x"), DataPreparerOperatorError)


# ---- check_reviewed -----------------------------------------------------------------------


def test_check_refuses_every_edit(proj):
    proj.route()
    proj.apply([_person("t1", "b"), _person("t2", "b")])
    good = Path(proj.manifest).read_text()
    copy = Path(proj.reviewed).read_text()
    rows = [json.loads(x) for x in copy.splitlines()]

    def reset():
        Path(proj.manifest).write_text(good)
        Path(proj.reviewed).write_text(copy)

    with pytest.raises(G.GoldenReviewError, match="has no person's resolutions yet; run apply"):
        m = json.loads(good)
        m.pop("reviewed_sha256")
        Path(proj.manifest).write_text(G.manifest_text(m))
        proj.check()
    reset()
    Path(proj.reviewed).write_text(copy + "\n")
    with pytest.raises(
        G.GoldenReviewError, match=r"is not the file golden_review.py apply wrote \(hash differs\)"
    ):
        proj.check()
    reset()
    proj.rewrite_manifest(export="/nowhere")
    with pytest.raises(
        G.GoldenReviewError, match="export '/nowhere' is missing or changed since apply"
    ):
        proj.check()
    reset()
    _jsonl(
        proj.routed, [{"id": "t1", "checks": ["other"]}, {"id": "t2", "checks": ["keyword_rules"]}]
    )
    with pytest.raises(G.GoldenReviewError, match="routed.jsonl is missing or changed since route"):
        proj.check()
    _jsonl(
        proj.routed,
        [
            {"id": "t1", "checks": ["keyword_rules", "guideline_judge"]},
            {"id": "t2", "checks": ["keyword_rules"]},
        ],
    )
    reset()
    m = json.loads(good)
    m["routed"]["t0"] = {"original": "b", "checks": ["x"]}
    Path(proj.manifest).write_text(G.manifest_text(m))
    with pytest.raises(
        G.GoldenReviewError, match="routed rows or label set are not the ones route wrote"
    ):
        proj.check()
    reset()
    proj.rewrite_manifest(label_set=["a"])
    with pytest.raises(G.GoldenReviewError, match="routed rows or label set"):
        proj.check()
    reset()
    proj.rewrite_manifest(unresolved=["t9"])
    with pytest.raises(
        G.GoldenReviewError, match="resolutions are not the ones the person's export gives"
    ):
        proj.check()
    reset()
    proj.rewrite_copy(rows[1:])
    with pytest.raises(
        G.GoldenReviewError, match=r"\(1 missing, e.g. \['t0'\]\): a golden row is never dropped"
    ):
        proj.check()
    reset()
    proj.rewrite_copy([dict(rows[0], provenance="auto_approved")] + rows[1:])
    with pytest.raises(G.GoldenReviewError, match="'t0' has provenance 'auto_approved', not None"):
        proj.check()
    reset()
    proj.rewrite_copy([dict(rows[0], gt_label="zz")] + rows[1:])
    with pytest.raises(G.GoldenReviewError, match="'t0' has a label no person's return set"):
        proj.check()
    reset()
    proj.rewrite_copy([rows[0], dict(rows[1], provenance=None)] + rows[2:])
    with pytest.raises(G.GoldenReviewError, match="'t2' has provenance None, not 'human_verified'"):
        proj.check()
    reset()
    Path(proj.test).write_text(Path(proj.test).read_text() + "\n")
    with pytest.raises(G.GoldenReviewError, match="changed after the review; run route again"):
        proj.check()


def test_helpers():
    assert G.observed_label_set([{"gt_label": "b"}, {"gt_label": "a"}, {"x": 1}], "gt_label") == [
        "a",
        "b",
    ]
    assert (
        G.observed_label_set([{"gt_label": 1}], "gt_label") is None
        and G.observed_label_set([], "g") is None
    )
    assert G.valid_label("a", ["a"]) and G.valid_label("a", None) and not G.valid_label(" ", None)
    assert not G.valid_label(3, None) and not G.valid_label("c", ["a"])
    assert G.review_row(
        {"id": "t", "text": "x", "note": "n"}, "id", "gt_label", ["text", "missing"]
    ) == {"id": "t", "text": "x", "_review_reason": "golden flag"}
    assert G.read_routed([{"id": 1, "checks": ["b", "a"]}], ["1"]) == {"1": ["a", "b"]}
