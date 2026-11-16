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
"""Unit tests for operators.auto_labeling.fields: the row and field contract."""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _fields as f


class TestBlindRows:
    def test_blind_then_unblind_roundtrip(self):
        row = {
            "id": "1",
            "text": "t",
            "label": "a",
            "reasoning": "r",
            "confidence_score": 0.9,
            "split": "dev",
        }
        blind = f.blind_row(row, "label")
        assert blind == {
            "id": "1",
            "text": "t",
            "split": "dev",
            f.MODEL_OUTPUT_FIELD: {"label": "a", "reasoning": "r", "confidence_score": 0.9},
        }
        assert row["label"] == "a"  # input unchanged
        assert f.unblind_row(blind) == row

    def test_unblind_keeps_a_non_null_value_on_the_row(self):
        row = {"label": "human", f.MODEL_OUTPUT_FIELD: {"label": "model", "reasoning": "r"}}
        assert f.unblind_row(row) == {"label": "human", "reasoning": "r"}

    def test_blind_merges_an_existing_hidden_dict(self):
        row = {"label": "a", f.MODEL_OUTPUT_FIELD: {"reasoning": "old"}}
        assert f.blind_row(row, "label")[f.MODEL_OUTPUT_FIELD] == {"reasoning": "old", "label": "a"}


class TestReservedFields:
    def test_conflicts_and_reasons(self):
        assert f.reserved_text_field_conflicts(
            ["text", "label", "id", "split", "_image_missing", "body"], "label"
        ) == ["_image_missing", "id", "label", "split"]
        assert f.reserved_field_reason("id") == "the job keys rows on it"
        assert f.reserved_field_reason("split") == "the run's split tag"
        assert "writes it on each labeled row" in f.reserved_field_reason("score", "score")
        assert "hidden from reviewers" in f.reserved_field_reason("jury_votes")
        assert f.reserved_field_reason("body") is None

    def test_hidden_fields_sorted_with_pred_field(self):
        hidden = f.review_hidden_fields("answer")
        assert hidden == sorted(hidden) and "answer" in hidden and f.REVIEW_REASON_FIELD in hidden
        assert "answer" not in f.review_hidden_fields()


class TestReviewTags:
    def test_review_reason_accepts_legacy_names(self):
        assert f.review_reason_of({"routed_reason": "low_conf"}) == "low_conf"
        assert f.review_reason_of({f.REVIEW_REASON_FIELD: "x", "route_reason": "y"}) == "x"
        assert f.review_reason_of({}) is None

    def test_labeled_by_people(self):
        assert f.labeled_by_people({f.LABELED_BY_FIELD: "people", "provenance": "human_verified"})
        assert not f.labeled_by_people({"provenance": "human_verified"})
        assert not f.labeled_by_people({f.LABELED_BY_FIELD: "people", "provenance": "model"})
        assert not f.labeled_by_people("row")


class TestSideFiles:
    def test_paths(self):
        assert f.model_output_path("a/review.jsonl") == "a/review.model_output.jsonl"
        assert f.model_output_candidates("a/review_rows.jsonl") == [
            "a/review_rows.model_output.jsonl",
            "a/review_model_output.jsonl",
        ]
        assert f.model_output_candidates("seed") == ["seed.model_output.jsonl"]
        assert f.fold_report_path("review_seed.jsonl") == "review_seed.fold.json"

    def test_join_model_output(self):
        assert f.join_model_output({"label": None, "x": 1}, {"label": "a", "x": 2}) == {
            "label": "a",
            "x": 1,
        }
        assert f.join_model_output({"a": 1}, None) == {"a": 1}

    def test_job_summary_sha256_ignores_key_order(self):
        assert f.job_summary_sha256({"a": 1, "b": 2}) == f.job_summary_sha256({"b": 2, "a": 1})
        assert f.job_summary_sha256(None) is None


class TestConfidenceMissing:
    def test_stated_uses_ballots_for_a_jury(self):
        s = {"StatedConfidenceMissing": 6, "RowsLabeled": 20, "JurorModelIds": ["a", "b", "c"]}
        assert f.stated_confidence_missing(s) == {
            "missing": 6,
            "of": 60,
            "unit": "ballots",
            "rate": 0.1,
            "high": True,
        }
        s["JurorModelIds"] = ["a"]
        assert f.stated_confidence_missing(s)["unit"] == "rows"
        assert f.stated_confidence_missing({"RowsLabeled": 0, "StatedConfidenceMissing": 1}) is None

    def test_verbalized_over_rows(self):
        s = {"VerbalizedConfidenceMissing": 1, "RowsLabeled": 100}
        assert f.verbalized_confidence_missing(s) == {
            "missing": 1,
            "of": 100,
            "unit": "rows",
            "rate": 0.01,
            "high": False,
        }
        assert f.verbalized_confidence_missing({"RowsLabeled": 100}) is None
