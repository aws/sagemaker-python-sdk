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
"""Unit tests for operators.auto_labeling.seed: append, merge, review drop and overlay."""

from __future__ import absolute_import

import logging

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import labels as S

HIL = {"_bootstrap": "hil"}


# ---- the verdict rule ----------------------------------------------------------------------


def test_review_verdict(caplog):
    assert S.review_verdict({"pred_field": "label", "model_output": {"label": "a"}}, "a") is True
    assert S.review_verdict({"pred_field": "label", "model_output": {"label": "1"}}, 1) is True
    assert S.review_verdict({"pred_field": "label", "model_output": {"label": "a"}}, "b") is False
    assert (
        S.review_verdict({"pred_field": "label", "model_output": {"label": None}}, "a")
        == "no_model_answer"
    )
    assert S.review_verdict({"pred_field": "label", "model_output": {}}, "a") is None
    assert S.review_verdict({"model_output": {"label": "a"}}, "a") is None
    assert (
        S.review_verdict({"pred_field": "judge_score", "model_output": {"judge_score": 2.6}}, 3)
        is True
    )
    assert (
        S.review_verdict(
            {"pred_field": "judge_score", "model_output": {"judge_score": 2.6}}, 3, tolerance=0.1
        )
        is False
    )
    assert (
        S.review_verdict({"pred_field": "answer", "model_output": {"answer": "Paris"}}, "paris")
        is True
    )
    S._WARNED_UNSCORED_FIELDS.discard("summary")
    with caplog.at_level(logging.WARNING, logger=S.__name__):
        assert (
            S.review_verdict({"pred_field": "summary", "model_output": {"summary": "s"}}, "s")
            is None
        )
        assert (
            S.review_verdict({"pred_field": "summary", "model_output": {"summary": "s"}}, "s")
            is None
        )
    assert caplog.text.count("no task rule scores") == 1  # warned once per field


def test_spot_check_confirms():
    assert S.spot_check_confirms({"label": "a"}, None, "a") is True
    assert (
        S.spot_check_confirms({"score": 2, "label": "a"}, None, "b") is False
    )  # label is the first PRED_FIELD
    assert S.spot_check_confirms({"score": 2}, None, 2) is True
    assert S.spot_check_confirms({"label": None}, "label", "a") is None
    assert S.spot_check_confirms({}, None, "a") is None
    assert S.spot_check_confirms({"summary": "s"}, "summary", "s") is None


# ---- append ----------------------------------------------------------------------------------


def test_check_seed_append():
    existing = [
        {"id": "r1", "gt_label": "a", **HIL},
        {"id": 2, "gt_label": "b", **HIL},
        {"id": "s1", "gt_label": "x"},
        "junk",
    ]
    assert S.check_seed_append(existing, [{"id": "n1", "gt_label": "a"}], "seed.jsonl") == (1, [])
    assert S.check_seed_append(existing, [], "seed.jsonl") == (0, [])
    # the same fold again: nothing to write
    assert S.check_seed_append(
        existing, [{"id": "r1", "gt_label": "a"}, {"id": "2", "gt_label": "b"}], "s"
    ) == (0, ["2", "r1"])
    with pytest.raises(S.SeedAppendRefused) as info:
        S.check_seed_append(
            existing, [{"id": "r1", "gt_label": "a"}, {"id": "n1", "gt_label": "a"}], "seed.jsonl"
        )
    assert str(info.value) == (
        "1 of the 2 folded id(s) are already in seed.jsonl (r1); a different review of the same rows or an id "
        "clash with the seed. Nothing was appended: fold only the new rows, or remove the old ones from the "
        "seed first"
    )
    with pytest.raises(S.SeedAppendRefused, match=r"\(r1; 1 with another label there: r1\)"):
        S.check_seed_append(existing, [{"id": "r1", "gt_label": "z"}], "seed.jsonl")
    with pytest.raises(
        S.SeedAppendRefused, match="1 of the 1 folded id"
    ):  # a non-hil seed row with that id
        S.check_seed_append(existing, [{"id": "s1", "gt_label": "x"}], "seed.jsonl")
    assert isinstance(info.value, DataPreparerOperatorError)
    many = [{"id": f"r{i}", "gt_label": "a", **HIL} for i in range(12)]
    with pytest.raises(S.SeedAppendRefused, match=r"r8, \.\.\.\)"):
        S.check_seed_append(many, many[:11] + [{"id": "new", "gt_label": "a"}], "s")


# ---- merge -------------------------------------------------------------------------------------


def test_seed_input_fields_and_review_tag():
    row = {
        "id": "r1",
        "text": "t",
        "title": "u",
        "gt_label": "a",
        "label": "b",
        "confidence_score": 0.5,
        "_labeled_by": "people",
        "provenance": "human_verified",
        "_model_output": {},
        "reasoning": "x",
    }
    assert S.seed_input_fields(row, "id", "gt_label", "label") == ["text", "title"]
    assert S.seed_review_tag({"_review_reason": "calibration"}) == "calibration"
    assert S.seed_review_tag({"routed_reason": "low_confidence"}) == "low_confidence"
    assert S.seed_review_tag({"_model_output": {"label": "a"}}) == "routed"
    assert S.seed_review_tag({"reasoning": "x"}) == "routed"
    assert S.seed_review_tag({"id": "r1", "text": "t"}) is None


def test_merge_seed_labels():
    seed = [
        {
            "id": "r1",
            "text": "t1",
            "gt_label": "a",
            "provenance": "human_verified",
            "_review_reason": "low_confidence",
        },
        {
            "id": 2,
            "text": "t2",
            "gt_label": "b",
            **HIL,
            "reasoning": "model said",
        },  # legacy marker -> human_verified
        {"id": "r3", "text": "t3", "gt_label": "c", "provenance": "spot_checked"},
        {
            "id": "r4",
            "text": "t4",
            "gt_label": "d",
            "provenance": "auto_approved",
        },  # a model's label: not merged
        {"id": "r5", "text": "t5", "provenance": "human_verified"},  # no gold
        {"id": None, "gt_label": "x", "provenance": "human_verified"},
        {"id": "r9", "text": "gone", "gt_label": "z", "provenance": "human_verified"},
        "junk",
    ]
    labeled = [
        {
            "id": "r1",
            "text": "t1",
            "label": "b",
            "confidence_score": 0.4,
            "reasoning": "why",
            "_failed": True,
            "_failed_reason": "parse",
            "routed_reason": "old",
        },
        {"id": "2", "text": "other", "label": "b"},
        {
            "id": "r3",
            "text": "t3",
            "label": "c",
            "provenance": "spot_checked",
            "_model_output": {"label": "q"},
        },
        {"id": "r4", "text": "t4", "label": "m"},
        {"id": "r5", "text": "t5", "label": "m"},
        {"id": "r6", "text": "t6", "label": "m"},
        "junk",
    ]
    rows, report = S.merge_seed_labels(labeled, seed)
    assert rows[0] == {
        "id": "r1",
        "text": "t1",
        "label": "a",
        "provenance": "human_verified",
        "_labeled_by": "people",
        "_review_reason": "low_confidence",
        "_model_output": {
            "label": "b",
            "confidence_score": 0.4,
            "reasoning": "why",
            "_failed": True,
            "_failed_reason": "parse",
        },
    }
    assert rows[1] == {
        "id": "2",
        "text": "other",
        "label": "b",
        "_model_output": {"label": "b"},
        "provenance": "human_verified",
        "_labeled_by": "people",
        "_review_reason": "routed",
    }
    # merged before: the stored model answer stays
    assert rows[2] == {
        "id": "r3",
        "text": "t3",
        "label": "c",
        "provenance": "spot_checked",
        "_model_output": {"label": "q"},
        "_labeled_by": "people",
    }
    assert rows[3:] == labeled[3:]
    assert report == {
        "mode": "merge",
        "labels_rows": 7,
        "seed_human_rows": 4,
        "merged": 3,
        "label_changed": 2,
        "already_merged": 1,
        "missing_from_labels": ["r9"],
        "content_mismatch": ["2"],
    }
    # a row with no model answer gets no model-output field
    rows, _ = S.merge_seed_labels([{"id": "r1", "text": "t1"}], seed[:1])
    assert "_model_output" not in rows[0] and rows[0]["label"] == "a"


def test_drop_people_rows_and_known_spot_check():
    review = [{"id": "r1"}, {"id": 2}, {"id": "r3", "split": "spot_check"}, "junk", {"id": None}]
    side = [
        {"id": "r1", "x": 1},
        {"id": "2"},
        {"id": "r3", "pred_field": "label", "model_output": {"label": "a"}},
        "junk",
    ]
    kept, side_out, dropped = S.drop_people_rows_from_review(review, side, ["r1", 2, "r3", "r7"])
    assert kept == ["junk", {"id": None}] and dropped == ["2", "r1", "r3"]
    assert side_out == ["junk"]
    assert S.known_spot_check_path("audit/review.jsonl") == "audit/review.known_spot_check.json"
    assert S.known_spot_check_path("review") == "review.known_spot_check.json"
    labeled = [
        {"id": "r3", "label": "a", "_labeled_by": "people", "provenance": "human_verified"},
        {"id": "r4", "label": "b", "_labeled_by": "people", "provenance": "spot_checked"},
        {"id": "r5", "label": "b"},  # not a person's label: no known gold
    ]
    dropped_rows = [
        {"id": "r3", "split": "spot_check"},
        {"id": "r4", "split": "spot_check", "_model_output": {"label": "a"}},
        {"id": "r5", "split": "spot_check"},
        {"id": "r1"},
    ]
    assert S.score_known_spot_check(dropped_rows, side, labeled) == {
        "rows": 3,
        "corrected": 1,
        "no_model_answer": 1,
        "ids": ["r3", "r4", "r5"],
    }
    assert S.score_known_spot_check([{"id": "r1"}], side, labeled) is None


# ---- overlay -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,is_model",
    [
        ("jury", True),
        ("model", True),
        ("resample:x", True),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("us.anthropic.claude-sonnet-4-6", True),
        ("amazon.nova-pro-v1:0", True),
        ("openai.gpt-oss-120b", True),
        ("Reuters", False),
        ("en.wikipedia.org", False),
        ("google.gemma", False),
        ("", False),
        (7, False),
        (None, False),
    ],
)
def test_model_source(source, is_model):
    assert S.model_source(source) is is_model


@pytest.mark.parametrize(
    "row,want",
    [
        ({"provenance": "human_verified"}, "human_verified"),
        ({"_provenance": "spot_checked"}, "spot_checked"),
        ({"provenance": "auto_approved"}, "auto_approved"),
        ({"provenance": "human_verified", "_provenance": "spot_checked"}, None),
        ({"provenance": "weird"}, None),
        ({"_bootstrap": " HIL "}, "human_verified"),
        ({"_bootstrap": "self"}, "auto_approved"),
        ({"_bootstrap": "odd"}, None),
        ({"_bootstrap": 3}, None),
        ({"provenance": "human_verified", "_bootstrap": "self"}, None),
        ({"provenance": "spot_checked", "_bootstrap": "hil"}, "spot_checked"),
        ({"provenance": "human_verified", "source": "jury"}, "human_verified"),
        ({"source": "jury"}, None),
        ({"source": "Reuters"}, "human_verified"),
        ({}, "human_verified"),
    ],
)
def test_overlay_provenance(row, want):
    assert S.overlay_provenance(row) == want
    assert S.untagged(row) is all(
        row.get(k) is None for k in ("provenance", "_provenance", "_bootstrap")
    )


def test_build_overlay():
    seed = [
        {
            "id": "r1",
            "text": "t",
            "gt_label": "a",
            "provenance": "human_verified",
            "_review_reason": "calibration",
        },
        {
            "id": "r1",
            "text": "t",
            "gt_label": "b",
            "provenance": "human_verified",
        },  # a later fold wins
        {"id": 2, "gt_label": "c"},  # untagged: a person's label
        {"id": "r3", "gt_label": "d", "provenance": "auto_approved"},
        {"id": "r4", "gt_label": "e", "source": "jury"},
        {"id": "r5", "provenance": "human_verified"},
        {"id": "r6", "gt_label": "f", "_bootstrap": "hil", "reasoning": "m"},
        "junk",
    ]
    rows, report = S.build_overlay(seed)
    assert rows == [
        {"id": "r1", "label": "b", "provenance": "human_verified", "_labeled_by": "people"},
        {"id": 2, "label": "c", "provenance": "human_verified", "_labeled_by": "people"},
        {
            "id": "r6",
            "label": "f",
            "provenance": "human_verified",
            "_labeled_by": "people",
            "_review_reason": "routed",
        },
    ]
    assert report == {
        "mode": "overlay",
        "overlay_rows": 3,
        "by_source": {"sample": 2, "routed": 1},
        "untagged_as_human": 1,
    }
    assert S.build_overlay([], gt_field="category", pred_field="score") == (
        [],
        {"mode": "overlay", "overlay_rows": 0, "by_source": {}, "untagged_as_human": 0},
    )


def test_overlay_seed_rows_and_seed_rows_for_overlay():
    files = [
        ("a.jsonl", [{"id": "r1", "category": "x"}, "junk"]),
        ("b.jsonl", [{"id": "r2", "gt_label": "y", "category": "z"}]),
        ("c.jsonl", []),
    ]
    rows, fields = S.overlay_seed_rows(files, {"label_field": "category"})
    assert fields == {"a.jsonl": "category", "b.jsonl": "gt_label", "c.jsonl": None}
    assert rows == [
        {"id": "r1", "category": "x", "__overlay_gold__": "x"},
        {"id": "r2", "gt_label": "y", "category": "z", "__overlay_gold__": "y"},
    ]
    assert S.overlay_seed_rows(files[:1], {}, gt_field="category")[1] == {"a.jsonl": "category"}
    with pytest.raises(DataPreparerOperatorError) as info:
        S.overlay_seed_rows(files[:1], "junk")
    assert str(info.value) == (
        "--seed a.jsonl: no row has a gold value under gt_label (its columns: category, id), so none of its "
        "people's labels would reach the overlay. Pass --gt-field <its label column>"
    )
    # the task's default gold field for judge_score is gt_score too: named once
    with pytest.raises(DataPreparerOperatorError, match="under gt_score / other"):
        S.overlay_seed_rows(
            files[:1], {"label_field": "other"}, gt_field="gt_score", pred_field="judge_score"
        )
    seed = [{"id": "r1", "source_export": "e"}, {"id": "r2"}, "junk"]
    assert S.seed_rows_for_overlay(seed, {"eval_rows_labeled_for_scoring": 2}) == ([seed[0]], 2)
    for d in (
        {"eval_rows_labeled_for_scoring": 0},
        {"eval_rows_labeled_for_scoring": True},
        {},
        "junk",
        None,
    ):
        assert S.seed_rows_for_overlay(seed, d) == (seed, 0)


def test_overlay_spec_and_handoff():
    spec = S.overlay_spec("label", "id")
    assert spec.startswith(
        "Two inputs. input_0 is the labeled dataset (JSONL, every row has 'id'). "
    )
    assert "'label', 'provenance', '_labeled_by', '_review_reason'" in spec
    assert "'label', 'reasoning', 'prediction_error', 'confidence', 'confidence_score'" in spec
    assert spec.endswith("Write JSONL.")
    h = S.overlay_handoff(
        {"dataset_arn": "arn:ds", "dataset_name": "tickets"}, "overlay.jsonl", overlay_rows=12
    )
    assert h["Operation"] == "overlay_by_id" and h["DtJobs"] == 1
    assert h["Inputs"]["LabeledDatasetArn"] == "arn:ds" and h["Inputs"]["OverlayRowCount"] == 12
    assert h["ReplaceFields"] == ["label", "provenance", "_labeled_by", "_review_reason"]
    assert h["ModelAnswerTo"] == "_model_output" and h["KeyField"] == "id"
    assert (
        h["Register"]
        == "python scripts/hub_io.py register --labels overlay.jsonl --name tickets-corrections --s3-prefix <s3://...>"
    )
    assert h["DtView"] == "python scripts/hub_io.py dt-view --arn arn:ds --project <proj>"
    assert "overlay_matched equals 12." in h["Check"] and h["Specification"] == spec
    assert h["Promote"].endswith("--to-arn arn:ds --project <proj>")
    assert h["Output"] == {
        "DatasetName": "tickets",
        "As": "the labeled DataSet's next version, the dataset_arn hub_io.py promote prints",
    }
    assert (
        "overlay_matched equals the overlay's row count."
        in S.overlay_handoff({"dataset_arn": "a"}, "o")["Check"]
    )
    assert S.overlay_handoff({"dataset_arn": "a"}, "o")["Output"]["DatasetName"] == "labeled"
    with pytest.raises(DataPreparerOperatorError, match="names no dataset_arn"):
        S.overlay_handoff({}, "o")


def test_fill_missing_ids():
    rows = [{"text": "a"}, {"id": None}, {"id": "x"}, "junk", {"id": 0}]
    assert S.fill_missing_ids(rows) == 2
    assert rows == [{"text": "a", "id": "0"}, {"id": "1"}, {"id": "x"}, "junk", {"id": 0}]
    assert S.fill_missing_ids([{"text": "a"}], "rid") == 0
