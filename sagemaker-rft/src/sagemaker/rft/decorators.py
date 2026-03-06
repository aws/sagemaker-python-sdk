"""Decorators for RFT integration.

Provides rft_handler decorator for AgentCore Runtime entrypoints.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from sagemaker.rft.context import set_rollout_context, clear_rollout_context
from sagemaker.rft.feedback import RolloutFeedbackClient


def rft_handler(func: Callable) -> Callable:
    """Decorator for AgentCore Runtime entrypoints to handle RFT rollout lifecycle.

    Automatically:
    1. Sets rollout context (metadata + inference_params) for header injection
    2. Reports rollout completion status to the training platform
    3. Reports errors if the rollout fails

    Args:
        func: An async function that handles the rollout.

    Returns:
        Wrapped async function with automatic context and reporting.

    Example::

        @app.entrypoint
        @rft_handler
        async def invoke_agent(payload):
            result = await agent.invoke_async(payload["instance"])
            return result
    """

    @wraps(func)
    async def wrapper(payload: dict) -> Any:
        metadata = payload.get("metadata")
        inference_params = payload.get("inference_params")

        set_rollout_context(metadata, inference_params)

        try:
            result = await func(payload)
            RolloutFeedbackClient(metadata).report_complete()
            return result
        except Exception:
            RolloutFeedbackClient(metadata).report_error()
            raise
        finally:
            clear_rollout_context()

    return wrapper
