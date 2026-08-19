"""Decorators for RFT integration.

Provides rft_handler decorator for AgentCore Runtime entrypoints.
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any, Callable

from sagemaker.rft.context import set_rollout_context, clear_rollout_context

logger = logging.getLogger(__name__)


def rft_handler(func: Callable) -> Callable:
    """Decorator for AgentCore Runtime entrypoints to handle RFT rollout lifecycle.

    Automatically:
    1. Sets rollout context (metadata + inference_params) for header injection
    2. Clears context when done
    3. Logs errors if the rollout fails

    Works with both sync and async functions.

    Args:
        func: A sync or async function that handles the rollout.

    Returns:
        Wrapped function with automatic context management.

    Example::

        @app.entrypoint
        @rft_handler
        def invoke_agent(payload):
            result = agent(payload["instance"])
            return result
    """
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(payload: dict) -> Any:
            metadata = payload.get("metadata")
            inference_params = payload.get("inference_params")
            set_rollout_context(metadata, inference_params)
            try:
                return await func(payload)
            except Exception:
                logger.exception("RFT rollout failed")
                raise
            finally:
                clear_rollout_context()

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(payload: dict) -> Any:
            metadata = payload.get("metadata")
            inference_params = payload.get("inference_params")
            set_rollout_context(metadata, inference_params)
            try:
                return func(payload)
            except Exception:
                logger.exception("RFT rollout failed")
                raise
            finally:
                clear_rollout_context()

        return sync_wrapper
