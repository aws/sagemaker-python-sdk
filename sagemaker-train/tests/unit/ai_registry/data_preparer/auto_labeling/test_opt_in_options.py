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
"""Unit tests for the opt-in labeling options: Confidence, Optimizer and Review.

The service's CreateJob validator (``ValidationConstants``) is the source of truth for
the wire keys asserted here: ``Confidence`` {Method, Estimator, Models, Recalibration,
FloorBypass}, ``Optimizer`` {Mode, TeacherModel, Seed, Parser, Elicitation}, ``Review``
{LabelField, Scope, PredictionsS3Uris, RulesS3Uri, JudgeModels, JudgeScope, MaxJudgeRows,
EstimateOnly}; each is accepted on ``labeling.classification`` only, and ``Review`` is
refused alongside Confidence, Optimizer, Routing, Demos, EvaluationData and Scoring.
"""

from __future__ import absolute_import

import pytest
from pydantic import ValidationError

from sagemaker.ai_registry.data_preparer.auto_labeling.shapes import (
    ClassificationLabelingOptions,
    Confidence,
    Demos,
    EvaluationData,
    JudgeModel,
    LlmJudgeLabelingOptions,
    NerLabelingOptions,
    Optimizer,
    OrdinalLabelingOptions,
    QaLabelingOptions,
    Review,
    Routing,
    Scoring,
    SummarizationLabelingOptions,
)

NON_CLASSIFICATION = [
    (NerLabelingOptions, {}),
    (QaLabelingOptions, {}),
    (OrdinalLabelingOptions, {}),
    (LlmJudgeLabelingOptions, {}),
    (SummarizationLabelingOptions, {"graders": [{"model_id": "g"}]}),
]


def _config(options):
    return options.to_dataset_options()["Config"]


class TestConfidenceWire:
    def test_every_field_serializes_under_the_service_keys(self):
        opts = ClassificationLabelingOptions(
            text_fields=["t"],
            confidence=Confidence(
                method="verbalized",
                estimator="judge",
                models=[JudgeModel(model_id="m", reasoning_effort="low")],
                recalibration="platt",
                floor_bypass=True,
            ),
        )
        assert _config(opts)["Confidence"] == {
            "Method": "verbalized",
            "Estimator": "judge",
            "Models": [{"ModelId": "m", "ReasoningEffort": "low"}],
            "Recalibration": "platt",
            "FloorBypass": True,
        }

    def test_method_alone_sends_only_method(self):
        opts = ClassificationLabelingOptions(
            text_fields=["t"], confidence=Confidence(method="stated")
        )
        assert _config(opts)["Confidence"] == {"Method": "stated"}

    def test_unset_confidence_is_omitted(self):
        assert "Confidence" not in _config(ClassificationLabelingOptions(text_fields=["t"]))

    def test_models_without_estimator_rejected(self):
        with pytest.raises(ValidationError, match="estimator"):
            Confidence(method="agreement", models=[JudgeModel(model_id="m")])

    def test_method_is_required(self):
        with pytest.raises(ValidationError):
            Confidence(estimator="judge")

    def test_misspelled_field_rejected(self):
        with pytest.raises(ValidationError):
            Confidence(method="agreement", estimater="judge")

    def test_pascal_case_input_accepted(self):
        c = Confidence(Method="verbalized", FloorBypass=True)
        assert c.method == "verbalized" and c.floor_bypass is True

    @pytest.mark.parametrize("cls, extra", NON_CLASSIFICATION)
    def test_only_classification_accepts_confidence(self, cls, extra):
        with pytest.raises(ValidationError):
            cls(text_fields=["t"], confidence=Confidence(method="agreement"), **extra)


class TestOptimizerWire:
    def test_every_field_serializes_under_the_service_keys(self):
        opts = ClassificationLabelingOptions(
            text_fields=["t"],
            optimizer=Optimizer(
                mode="protegi",
                teacher_model=JudgeModel(model_id="teacher", reasoning_effort="high"),
                seed=7,
                parser="salvage",
                elicitation="confidence",
            ),
        )
        assert _config(opts)["Optimizer"] == {
            "Mode": "protegi",
            "TeacherModel": {"ModelId": "teacher", "ReasoningEffort": "high"},
            "Seed": 7,
            "Parser": "salvage",
            "Elicitation": "confidence",
        }

    def test_mode_alone(self):
        opts = ClassificationLabelingOptions(text_fields=["t"], optimizer=Optimizer(mode="protegi"))
        assert _config(opts)["Optimizer"] == {"Mode": "protegi"}

    def test_mode_is_required(self):
        with pytest.raises(ValidationError):
            Optimizer(seed=1)

    def test_misspelled_field_rejected(self):
        with pytest.raises(ValidationError):
            Optimizer(mode="protegi", teacher="x")

    @pytest.mark.parametrize("cls, extra", NON_CLASSIFICATION)
    def test_only_classification_accepts_optimizer(self, cls, extra):
        with pytest.raises(ValidationError):
            cls(text_fields=["t"], optimizer=Optimizer(mode="protegi"), **extra)

    def test_optimizer_with_confidence_serializes_both(self):
        opts = ClassificationLabelingOptions(
            text_fields=["t"],
            confidence=Confidence(method="agreement"),
            optimizer=Optimizer(mode="protegi"),
        )
        cfg = _config(opts)
        assert cfg["Confidence"] == {"Method": "agreement"}
        assert cfg["Optimizer"] == {"Mode": "protegi"}


class TestReviewWire:
    def test_every_field_serializes_under_the_service_keys(self):
        opts = ClassificationLabelingOptions(
            text_fields=["t"],
            review=Review(
                label_field="gt_label",
                scope="golden",
                predictions_s3_uris=["s3://b/dev.jsonl", "s3://b/val.jsonl"],
                rules_s3_uri="s3://b/rules.json",
                judge_models=[JudgeModel(model_id="j")],
                judge_scope="candidates",
                max_judge_rows=250,
                estimate_only=True,
            ),
        )
        assert _config(opts) == {
            "TextFields": ["t"],
            "Review": {
                "LabelField": "gt_label",
                "Scope": "golden",
                "PredictionsS3Uris": ["s3://b/dev.jsonl", "s3://b/val.jsonl"],
                "RulesS3Uri": "s3://b/rules.json",
                "JudgeModels": [{"ModelId": "j"}],
                "JudgeScope": "candidates",
                "MaxJudgeRows": 250,
                "EstimateOnly": True,
            },
        }

    def test_empty_review_is_a_seed_review(self):
        """Every Review key is optional: ``Review()`` is the service's default seed review."""
        opts = ClassificationLabelingOptions(text_fields=["t"], review=Review())
        assert _config(opts)["Review"] == {}

    def test_false_estimate_only_is_sent(self):
        """``exclude_none`` keeps an explicit False; only unset fields are omitted."""
        opts = ClassificationLabelingOptions(text_fields=["t"], review=Review(estimate_only=False))
        assert _config(opts)["Review"] == {"EstimateOnly": False}

    def test_misspelled_field_rejected(self):
        with pytest.raises(ValidationError):
            Review(predictions_s3_uri=["s3://b/x"])

    @pytest.mark.parametrize("cls, extra", NON_CLASSIFICATION)
    def test_only_classification_accepts_review(self, cls, extra):
        with pytest.raises(ValidationError):
            cls(text_fields=["t"], review=Review(), **extra)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("demos", Demos(s3_uri="s3://b/demos.jsonl")),
            ("routing", Routing(review_dataset_name="r")),
            ("evaluation_data", EvaluationData(label_field="g", s3_uri="s3://b/gold.jsonl")),
            ("scoring", Scoring(target=0.9)),
            ("confidence", Confidence(method="agreement")),
            ("optimizer", Optimizer(mode="protegi")),
        ],
    )
    def test_review_refuses_each_labeling_only_field(self, field, value):
        """The service refuses these keys with Review (REVIEW_REFUSED_CONFIG_KEYS); so does the SDK."""
        with pytest.raises(ValidationError, match=field):
            ClassificationLabelingOptions(text_fields=["t"], review=Review(), **{field: value})

    def test_review_error_names_every_clashing_field(self):
        with pytest.raises(ValidationError, match="demos, routing"):
            ClassificationLabelingOptions(
                text_fields=["t"],
                review=Review(),
                demos=Demos(s3_uri="s3://b/d"),
                routing=Routing(review_dataset_name="r"),
            )


class TestJudgeModel:
    def test_model_id_required(self):
        with pytest.raises(ValidationError):
            JudgeModel(reasoning_effort="low")

    def test_pascal_case_input(self):
        j = JudgeModel(ModelId="m", ReasoningEffort="max")
        assert j.model_id == "m" and j.reasoning_effort == "max"

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError):
            JudgeModel(model_id="m", target_tokens_per_minute=10)


class TestUnchangedWithoutOptions:
    """Adding the fields changes nothing for a request that does not set them."""

    def test_full_labeling_request_byte_identical(self):
        opts = ClassificationLabelingOptions(
            text_fields=["subject", "body"],
            demos=Demos(s3_uri="s3://b/demos.jsonl"),
            evaluation_data=EvaluationData(label_field="label", s3_uri="s3://b/gold.jsonl"),
            scoring=Scoring(target=0.9),
            routing=Routing(review_dataset_name="r"),
        )
        assert opts.to_dataset_options() == {
            "Type": "labeling.classification",
            "Config": {
                "TextFields": ["subject", "body"],
                "Demos": {"S3Uri": "s3://b/demos.jsonl"},
                "Routing": {"ReviewDatasetName": "r"},
                "EvaluationData": {"LabelField": "label", "S3Uri": "s3://b/gold.jsonl"},
                "Scoring": {"Target": 0.9},
            },
        }
