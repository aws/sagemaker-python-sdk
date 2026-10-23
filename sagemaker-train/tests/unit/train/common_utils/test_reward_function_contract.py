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
"""Tests for the typed reward-function contract of ``verify_reward_function``.

Before this change the verifier had no typed input contract: it accepted any dict as a
sample, sent every sample in one invocation, and selected the response format only with
``is_nova``. ``verify_reward_function`` and ``RewardFunctionOutput`` were importable only
from ``sagemaker.train.common_utils.rlvr_reward_verifier``.
"""

from __future__ import absolute_import

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

import sagemaker.train as train_module
from sagemaker.train.common_utils.rlvr_reward_verifier import (
    EvalRewardSample,
    RewardFunctionOutput,
    RewardSample,
    RewardVerificationOptions,
    verify_reward_function,
)

TRAIN_SAMPLES = [
    {
        "id": "sample_1",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "reference_answer": "4",
    },
    {
        "id": "sample_2",
        "messages": [
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ],
        "reference_answer": {"text": "Paris"},
    },
]

EVAL_SAMPLES = [
    {
        "id": "eval_1",
        "model_response": "4",
        "query": "What is 2+2?",
        "response": "4",
        "reference_answer": {"text": "4"},
    },
    {"id": "eval_2", "model_response": "Paris", "query": "Capital of France?"},
]

RECORDING_HANDLER = """
import json

CALLS = []

def lambda_handler(event, context):
    CALLS.append(event)
    results = [
        {"id": sample["id"], "aggregate_reward_score": 1.0} for sample in event
    ]
    with open(RECORD_PATH, "a") as record:
        record.write(json.dumps(event) + "\\n")
    return ENVELOPE(results)
"""

OSS_RETURN = """
def ENVELOPE(results):
    import json
    return {"statusCode": 200, "headers": {}, "body": json.dumps(results)}
"""

NOVA_RETURN = """
def ENVELOPE(results):
    return results
"""


@pytest.fixture
def recorder(tmp_path):
    """Write a handler that records each event it gets and returns the selected envelope."""
    record_path = tmp_path / "events.jsonl"

    def write(return_code):
        code = f'RECORD_PATH = "{record_path}"\n' + return_code + RECORDING_HANDLER
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
            handle.write(code)
            return handle.name

    def events():
        if not record_path.exists():
            return []
        lines = record_path.read_text().splitlines()
        return [json.loads(line) for line in lines]

    return write, events


def _run(reward_file, sample_data, **options):
    try:
        return verify_reward_function(
            reward_function=reward_file, sample_data=sample_data, **options
        )
    finally:
        Path(reward_file).unlink()


def test_oss_envelope_sends_all_samples_in_one_invocation(recorder):
    write, events = recorder

    result = _run(write(OSS_RETURN), TRAIN_SAMPLES, envelope="oss")

    assert result["successful_samples"] == 2
    assert events() == [TRAIN_SAMPLES]


def test_nova_envelope_accepts_a_bare_list_response(recorder):
    write, events = recorder

    result = _run(write(NOVA_RETURN), TRAIN_SAMPLES, envelope="nova")

    assert result["successful_samples"] == 2
    assert events() == [TRAIN_SAMPLES]


def test_oss_envelope_rejects_a_bare_list_response(recorder):
    write, _ = recorder

    with pytest.raises(ValueError) as excinfo:
        _run(write(NOVA_RETURN), TRAIN_SAMPLES, envelope="oss")

    message = str(excinfo.value)
    assert "2/2 sample(s) failed validation" in message
    assert "OSS reward function must return a dict with 'statusCode' and 'body'" in message


def test_envelope_wins_over_is_nova(recorder):
    write, _ = recorder

    result = _run(write(OSS_RETURN), TRAIN_SAMPLES, envelope="oss", is_nova=True)

    assert result["successful_samples"] == 2


def test_eval_oss_envelope_sends_one_invocation_per_sample(recorder):
    write, events = recorder

    result = _run(write(OSS_RETURN), EVAL_SAMPLES, envelope="eval_oss")

    assert result["successful_samples"] == 2
    assert events() == [[EVAL_SAMPLES[0]], [EVAL_SAMPLES[1]]]
    assert [item["output"]["id"] for item in result["results"]] == ["eval_1", "eval_2"]


def test_eval_nova_envelope_expects_the_status_code_body_shape(recorder):
    write, events = recorder

    result = _run(write(OSS_RETURN), EVAL_SAMPLES, envelope="eval_nova")

    assert result["successful_samples"] == 2
    assert events() == [[EVAL_SAMPLES[0]], [EVAL_SAMPLES[1]]]


def test_no_envelope_keeps_is_nova_behavior(recorder):
    write, events = recorder
    untyped_samples = [{"messages": [{"role": "user", "content": "hi"}]}]
    code = 'RECORD_PATH = "unused"\n' + NOVA_RETURN + """
def lambda_handler(event, context):
    return ENVELOPE([{"id": "x", "aggregate_reward_score": 0.5} for _ in event])
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as handle:
        handle.write(code)
        reward_file = handle.name

    result = _run(reward_file, untyped_samples)

    assert result["successful_samples"] == 1


def test_typed_samples_reject_a_missing_id(recorder):
    write, events = recorder
    sample_without_id = {"messages": [{"role": "user", "content": "hi"}]}

    with pytest.raises(ValueError) as excinfo:
        _run(write(NOVA_RETURN), [sample_without_id], envelope="nova")

    message = str(excinfo.value)
    assert "RewardSample" in message
    assert "Sample 0: id - Field required" in message
    assert events() == []


def test_eval_envelope_rejects_a_training_sample(recorder):
    write, _ = recorder

    with pytest.raises(ValueError) as excinfo:
        _run(write(OSS_RETURN), TRAIN_SAMPLES, envelope="eval_oss")

    message = str(excinfo.value)
    assert "EvalRewardSample" in message
    assert "model_response - Field required" in message


def test_options_object_selects_the_envelope(recorder):
    write, events = recorder
    options = RewardVerificationOptions(envelope="eval_oss")

    result = _run(write(OSS_RETURN), EVAL_SAMPLES, options=options)

    assert result["successful_samples"] == 2
    assert len(events()) == 2


def test_options_object_and_keywords_together_raise(recorder):
    write, _ = recorder

    with pytest.raises(ValueError, match="not the two together"):
        _run(write(OSS_RETURN), EVAL_SAMPLES, options=RewardVerificationOptions(), envelope="oss")


def test_reward_sample_model_requires_id_and_messages():
    with pytest.raises(ValidationError) as excinfo:
        RewardSample.model_validate({"messages": []})
    assert excinfo.value.errors()[0]["loc"] == ("id",)

    sample = RewardSample.model_validate(TRAIN_SAMPLES[1])
    assert sample.reference_answer == {"text": "Paris"}


def test_eval_reward_sample_model_requires_id_and_model_response():
    with pytest.raises(ValidationError) as excinfo:
        EvalRewardSample.model_validate({"query": "q"})
    assert {error["loc"] for error in excinfo.value.errors()} == {("id",), ("model_response",)}

    sample = EvalRewardSample.model_validate(EVAL_SAMPLES[1])
    assert sample.response is None


def test_exports_resolve_from_sagemaker_train():
    assert train_module.verify_reward_function is verify_reward_function
    assert train_module.RewardFunctionOutput is RewardFunctionOutput
    assert train_module.RewardSample is RewardSample
    assert train_module.EvalRewardSample is EvalRewardSample
    assert train_module.RewardVerificationOptions is RewardVerificationOptions
