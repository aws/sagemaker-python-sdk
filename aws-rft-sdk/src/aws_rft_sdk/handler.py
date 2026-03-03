"""@rft_handler decorator — wraps an entrypoint to manage RFT rollout context."""

import asyncio
import functools
import inspect
import logging

from aws_rft_sdk.client import RolloutFeedbackClient
from aws_rft_sdk.context import _set_metadata, _clear_metadata

logger = logging.getLogger(__name__)


def rft_handler(func):
    """Decorator that sets up RFT rollout context around an entrypoint.

    Extracts ``metadata`` from the payload, makes it available via
    ``RFTContext.get_headers()`` (used by ``wrap_model``), and auto-reports
    errors if the function raises.

    Works with both sync and async functions.

    Example::

        @app.entrypoint
        @rft_handler
        async def invoke_agent(payload):
            user_input = payload.get("instance")
            response = await agent.invoke_async(user_input)
            return response.message["content"][0]["text"]
    """

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(payload, *args, **kwargs):
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            _set_metadata(metadata)
            try:
                return await func(payload, *args, **kwargs)
            except Exception as e:
                logger.error("RFT rollout failed: %s", e)
                try:
                    RolloutFeedbackClient(metadata).report_error(str(e))
                except Exception:
                    logger.exception("Failed to report rollout error")
                raise
            finally:
                _clear_metadata()

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(payload, *args, **kwargs):
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            _set_metadata(metadata)
            try:
                return func(payload, *args, **kwargs)
            except Exception as e:
                logger.error("RFT rollout failed: %s", e)
                try:
                    RolloutFeedbackClient(metadata).report_error(str(e))
                except Exception:
                    logger.exception("Failed to report rollout error")
                raise
            finally:
                _clear_metadata()

        return sync_wrapper
