"""Decorators for RFT integration.

Provides sagemaker_rft_handler decorator for AgentCore Runtime entrypoints.
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any, Callable

from sagemaker.train.rft.context import set_rollout_context, clear_rollout_context
from sagemaker.train.rft.feedback import RolloutFeedbackClient, _is_trajectory_already_processed

logger = logging.getLogger(__name__)


def sagemaker_rft_handler(func: Callable) -> Callable:
    """Decorator for AgentCore Runtime entrypoints to handle RFT rollout lifecycle.

    Automatically:
    1. Sets rollout context (metadata + inference_params) for header injection
    2. On success, calls CompleteTrajectory + UpdateReward if "reward" in result
    3. On error, calls report_error
    4. Clears context when done

    Works with both sync and async functions.
    """
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(payload: dict) -> Any:
            metadata = payload.get("metadata") or {}
            inference_params = payload.get("inferenceParams") or payload.get("inference_params")
            set_rollout_context(metadata, inference_params)
            feedback = RolloutFeedbackClient(metadata)
            try:
                result = await func(payload)
            except Exception as e:
                error_str = str(e)
                if _is_trajectory_already_processed(error_str):
                    logger.warning("Trajectory already processed, skipping: %s", error_str)
                    return {"status": "skipped", "error": error_str}
                logger.error("RFT rollout failed: %s", e)
                try:
                    feedback.report_error(error_str)
                except Exception:
                    logger.exception("Failed to report rollout error")
                raise
            else:
                try:
                    _handle_result(feedback, result)
                except Exception:
                    logger.exception("Failed to report rollout result (non-fatal)")
                return result
            finally:
                clear_rollout_context()

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(payload: dict) -> Any:
            metadata = payload.get("metadata") or {}
            inference_params = payload.get("inferenceParams") or payload.get("inference_params")
            set_rollout_context(metadata, inference_params)
            feedback = RolloutFeedbackClient(metadata)
            try:
                result = func(payload)
            except Exception as e:
                error_str = str(e)
                if _is_trajectory_already_processed(error_str):
                    logger.warning("Trajectory already processed, skipping: %s", error_str)
                    return {"status": "skipped", "error": error_str}
                logger.error("RFT rollout failed: %s", e)
                try:
                    feedback.report_error(error_str)
                except Exception:
                    logger.exception("Failed to report rollout error")
                raise
            else:
                try:
                    _handle_result(feedback, result)
                except Exception:
                    logger.exception("Failed to report rollout result (non-fatal)")
                return result
            finally:
                clear_rollout_context()

        return sync_wrapper


def _handle_result(feedback: RolloutFeedbackClient, result: Any) -> None:
    """Handle rollout result: report success or error based on result status."""
    if not isinstance(result, dict):
        return

    # If the agent reported an error, mark trajectory as failed.
    # This catches cases where the agent returns {"status": "error", "reward": 0.0}
    # instead of raising — e.g. streaming errors caught by the agent.
    status = result.get("status", "")
    if status == "skipped":
        logger.info("Trajectory already processed, skipping feedback reporting")
        return
    if status == "error":
        error_msg = result.get("error", "unknown error")
        logger.warning("Agent returned error status: %s", error_msg)
        feedback.report_error(error_msg)
        return

    reward = result.get("reward")
    if reward is not None:
        if isinstance(reward, list):
            feedback.complete_rollout()
            feedback.update_reward(reward)
        else:
            feedback.report_complete(reward)
    else:
        feedback.complete_rollout()
