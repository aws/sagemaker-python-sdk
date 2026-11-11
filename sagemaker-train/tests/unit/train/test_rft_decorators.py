"""Tests for automatic RFT reward reporting."""

import asyncio
import json
from unittest.mock import patch

import pytest

from sagemaker.train.rft import RolloutFeedbackClient, sagemaker_rft_handler


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "result, status, rewards",
    [
        pytest.param({"reward": 0}, "ready", [0], id="integer"),
        pytest.param({"reward": 0.5}, "ready", [0.5], id="float"),
        pytest.param({"reward": [-1, 0.5]}, "ready", [-1, 0.5], id="per-turn"),
        pytest.param({"reward": (-1, 0.5)}, "ready", [-1, 0.5], id="tuple"),
        pytest.param({}, "failed", [0.0], id="missing"),
        pytest.param({"reward": None}, "failed", [0.0], id="none"),
        pytest.param({"reward": "0.5"}, "failed", [0.0], id="string"),
        pytest.param({"reward": {}}, "failed", [0.0], id="dict"),
        pytest.param({"reward": float("nan")}, "failed", [0.0], id="nan"),
        pytest.param({"reward": float("inf")}, "failed", [0.0], id="infinity"),
        pytest.param({"reward": float("-inf")}, "failed", [0.0], id="negative-infinity"),
        pytest.param({"reward": []}, "failed", [0.0], id="empty-list"),
        pytest.param({"reward": [0.5, "bad"]}, "failed", [0.0], id="string-in-list"),
        pytest.param({"reward": [float("nan")]}, "failed", [0.0], id="nan-in-list"),
        pytest.param({"reward": [[0.5]]}, "failed", [0.0], id="nested-list"),
        pytest.param(
            {"status": "error", "error": "agent failed", "reward": 0.5},
            "failed",
            [0.0],
            id="explicit-error",
        ),
        pytest.param({"status": "skipped"}, None, None, id="skipped"),
    ],
)
def test_reward_feedback(result, status, rewards, is_async):
    """Report only validated rewards as ready, preserving the handler's return value."""

    def handler(payload):
        return result

    async def async_handler(payload):
        return result

    wrapped = sagemaker_rft_handler(async_handler if is_async else handler)
    payload = {"metadata": {"jobArn": "test-job", "rolloutId": "test-rollout"}}
    with patch.object(RolloutFeedbackClient, "_bearer_post") as post:
        returned = asyncio.run(wrapped(payload)) if is_async else wrapped(payload)

    assert returned is result
    calls = [(call.args[0], json.loads(call.args[1])) for call in post.call_args_list]
    identity = {"JobArn": "test-job", "TrajectoryId": "test-rollout"}
    expected = (
        [
            ("/complete-rollout", {**identity, "Status": status}),
            ("/update-reward", {**identity, "Rewards": rewards}),
        ]
        if status is not None
        else []
    )
    assert calls == expected
