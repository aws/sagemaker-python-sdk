"""Header utilities for inference calls.

Provides functions to create the HTTP headers required by the
AgenticRFTRuntimeService for each inference request.

The runtime service expects three separate headers:
  - ``X-Rft-Job-Arn``: job ARN that identifies the training session
  - ``X-Trajectory-Id``: groups turns into a single trajectory
  - ``X-Span-Id``: unique ID for each turn within the trajectory
"""

from __future__ import annotations

import uuid
import warnings
from typing import Any

from sagemaker.rft.context import get_rollout_context
from sagemaker.rft.models import RolloutMetadata


def make_inference_headers(metadata: dict[str, Any] | RolloutMetadata) -> dict[str, str]:
    """Create headers dict for inference calls.

    Add these headers to your HTTP client when making inference calls
    to the RFT Runtime Service.

    Args:
        metadata: The metadata from the rollout request (dict or RolloutMetadata).

    Returns:
        Headers dict to add to inference calls.
    """
    if isinstance(metadata, RolloutMetadata):
        metadata = metadata.model_dump()
    headers: dict[str, str] = {}
    if metadata.get("job_arn"):
        headers["X-Rft-Job-Arn"] = metadata["job_arn"]
    if metadata.get("trajectory_id"):
        headers["X-Trajectory-Id"] = metadata["trajectory_id"]
        headers["X-Span-Id"] = str(uuid.uuid4())
    return headers


def get_inference_headers() -> dict[str, str]:
    """Get headers from current rollout context.

    For use with set_rollout_context() when you need to retrieve headers
    deep in the call stack without passing them explicitly.

    A new ``X-Span-Id`` is generated on every call so each inference
    turn gets a unique span within the trajectory.

    Returns:
        Headers dict, or empty dict if no context set (with warning).
    """
    metadata = get_rollout_context()
    if metadata is None:
        warnings.warn(
            "get_inference_headers() called but no rollout context set. "
            "Did you forget to call set_rollout_context()? "
            "Returning empty headers.",
            stacklevel=2,
        )
        return {}
    return make_inference_headers(metadata)
