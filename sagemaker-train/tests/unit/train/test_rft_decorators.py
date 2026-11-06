"""Unit tests for RFT handler argument forwarding."""

import asyncio
import json
from functools import wraps

import pytest

from sagemaker.train.rft import RolloutFeedbackClient, sagemaker_rft_handler
from sagemaker.train.rft.context import get_inference_params, get_rollout_context


@pytest.fixture(params=["sync", "async"])
def decorate(request):
    def decorator(handler):
        if request.param == "async":

            @wraps(handler)
            async def async_handler(*args, **kwargs):
                return handler(*args, **kwargs)

            return sagemaker_rft_handler(async_handler)
        return sagemaker_rft_handler(handler)

    return decorator


@pytest.fixture
def feedback(monkeypatch):
    reports = {}

    def record(self, path, body):
        reports[path] = json.loads(body)

    monkeypatch.setattr(RolloutFeedbackClient, "_bearer_post", record)
    return reports


@pytest.mark.parametrize("with_context", [False, True])
def test_handler_invocation(decorate, feedback, with_context):
    def handler(payload):
        return {"reward": payload["score"]}

    def handler_with_context(payload, context):
        return {**handler(payload), "context": context}

    decorated = decorate(handler_with_context if with_context else handler)
    payload = {"score": 0.75, "metadata": {"jobArn": "test-job", "rolloutId": "test-rollout"}}
    context = object()

    async def invoke():
        result = decorated(payload, context) if with_context else decorated(payload)
        return await result if asyncio.iscoroutine(result) else result

    result = asyncio.run(invoke())
    assert result["reward"] == 0.75
    if with_context:
        assert result["context"] is context
    assert feedback["/complete-rollout"]["Status"] == "ready"
    assert feedback["/update-reward"]["Rewards"] == [0.75]


@pytest.mark.parametrize("fail", [False, True])
def test_keyword_context_and_rollout_lifecycle(decorate, feedback, fail):
    context = object()
    metadata = {"jobArn": "test-job", "rolloutId": "test-rollout"}
    inference_params = {"temperature": 0.5}

    def handler(payload, context=None):
        assert get_rollout_context() == metadata
        assert get_inference_params() == inference_params
        if fail:
            raise ValueError("agent failed")
        return {"reward": payload["score"], "context": context}

    decorated = decorate(handler)
    payload = {"score": 0.75, "metadata": metadata, "inferenceParams": inference_params}

    async def invoke():
        try:
            result = decorated(payload, context=context)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            assert get_rollout_context() is None
            assert get_inference_params() is None

    if fail:
        with pytest.raises(ValueError, match="agent failed"):
            asyncio.run(invoke())
    else:
        result = asyncio.run(invoke())
        assert result["context"] is context
        assert result["reward"] == 0.75

    assert feedback["/complete-rollout"]["Status"] == ("failed" if fail else "ready")
    assert feedback["/update-reward"]["Rewards"] == [0.0 if fail else 0.75]
