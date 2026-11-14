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
"""The typed options reproduce the hand-built ``DatasetOptions`` documents byte for byte.

``labeling_request_fixtures.json`` holds request documents captured from a client that
builds ``GenerationConfig.DatasetOptions`` by hand, one per task and option
combination (refine round, demos + gold, routed corpus run, summarization graders,
and each classification opt-in). A caller that switches from the raw dict to these
classes must send the same request; this test pins that.
"""

from __future__ import absolute_import

import json
from pathlib import Path

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling import shapes as S

_FIXTURES = json.loads(
    (Path(__file__).parent / "labeling_request_fixtures.json").read_text(encoding="utf-8")
)

_CLASSES = {
    "classification": S.ClassificationLabelingOptions,
    "ner": S.NerLabelingOptions,
    "qa": S.QaLabelingOptions,
    "ordinal": S.OrdinalLabelingOptions,
    "llm_judge": S.LlmJudgeLabelingOptions,
    "summarization": S.SummarizationLabelingOptions,
}


def _judge(value):
    if isinstance(value, dict):
        return S.JudgeModel(
            model_id=value["ModelId"], reasoning_effort=value.get("ReasoningEffort")
        )
    return S.JudgeModel(model_id=value)


def _options_from_inputs(inputs):
    """Build the typed options from the same keyword inputs the hand-built client took."""
    task = inputs["task"]
    kwargs = {"text_fields": inputs["text_fields"]}
    if inputs.get("demos_s3_uri"):
        kwargs["demos"] = S.Demos(s3_uri=inputs["demos_s3_uri"])
    if inputs.get("review_dataset_name"):
        kwargs["routing"] = S.Routing(review_dataset_name=inputs["review_dataset_name"])
    if task == "summarization":
        kwargs["graders"] = [
            (
                S.Grader(model_id=g)
                if isinstance(g, str)
                else S.Grader(model_id=g["ModelId"], reasoning_effort=g.get("ReasoningEffort"))
            )
            for g in inputs["graders"]
        ]
    else:
        if inputs.get("eval_s3_uri") or inputs.get("eval_dataset_arn"):
            kwargs["evaluation_data"] = S.EvaluationData(
                label_field=inputs["label_field"],
                s3_uri=inputs.get("eval_s3_uri"),
                dataset_arn=inputs.get("eval_dataset_arn"),
            )
        if inputs.get("target") is not None:
            kwargs["scoring"] = S.Scoring(target=float(inputs["target"]))
    if inputs.get("confidence_method"):
        kwargs["confidence"] = S.Confidence(
            method=inputs["confidence_method"],
            estimator=inputs.get("confidence_estimator"),
            models=[_judge(m) for m in inputs.get("judge_models") or []] or None,
            recalibration=inputs.get("confidence_recalibration"),
            floor_bypass=True if inputs.get("confidence_floor_bypass") else None,
        )
    if inputs.get("optimizer"):
        op = inputs["optimizer"]
        kwargs["optimizer"] = S.Optimizer(
            mode=op["Mode"],
            teacher_model=_judge(op["TeacherModel"]) if "TeacherModel" in op else None,
            seed=op.get("Seed"),
            parser=op.get("Parser"),
            elicitation=op.get("Elicitation"),
        )
    return _CLASSES[task](**kwargs)


def _fixture_id(fixture):
    opts = sorted(k for k in fixture["inputs"] if k not in ("task", "text_fields", "model_ids"))
    return f"{fixture['inputs']['task']}-{'+'.join(opts) or 'bare'}"


@pytest.mark.parametrize("fixture", _FIXTURES, ids=[_fixture_id(f) for f in _FIXTURES])
def test_typed_options_match_the_hand_built_document(fixture):
    produced = _options_from_inputs(fixture["inputs"]).to_dataset_options()
    expected = fixture["expected"]
    assert json.dumps(produced, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_fixtures_cover_every_task_and_every_opt_in():
    tasks = {f["inputs"]["task"] for f in _FIXTURES}
    assert tasks == set(_CLASSES)
    keys = set().union(*(f["expected"]["Config"].keys() for f in _FIXTURES))
    assert {"Confidence", "Optimizer", "Routing", "Demos", "EvaluationData", "Scoring"} <= keys
