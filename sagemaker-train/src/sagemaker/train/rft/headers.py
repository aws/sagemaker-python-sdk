"""Header utilities for inference calls.

Provides functions to create the HTTP headers required by the
AgenticRFTRuntimeService for each inference request.

The runtime service expects individual headers:
  - ``X-Amzn-SageMaker-Job-Arn``: job ARN that identifies the training session
  - ``X-Amzn-SageMaker-Trajectory-Id``: unique trajectory identifier
"""

from __future__ import annotations

import warnings
from typing import Any

from sagemaker.train.rft.context import get_rollout_context
from sagemaker.train.rft.models import RolloutMetadata


def make_inference_headers(metadata: dict[str, Any] | RolloutMetadata) -> dict[str, str]:
    """Create headers dict for inference calls.

    Produces the individual headers required by the RFT Runtime Service:
      - ``X-Amzn-SageMaker-Job-Arn``: job ARN that identifies the training session
      - ``X-Amzn-SageMaker-Trajectory-Id``: unique trajectory identifier

    Args:
        metadata: The metadata from the rollout request (dict or RolloutMetadata).

    Returns:
        Headers dict to add to inference calls.
    """
    if isinstance(metadata, RolloutMetadata):
        metadata = metadata.model_dump()

    # Accept both camelCase (from TLM) and snake_case field names
    job_arn = metadata.get("job_arn") or metadata.get("jobArn")
    trajectory_id = (
        metadata.get("trajectory_id")
        or metadata.get("trajectoryId")
        or metadata.get("rolloutId")
    )

    headers: dict[str, str] = {}
    if job_arn:
        headers["X-Amzn-SageMaker-Job-Arn"] = job_arn
    if trajectory_id:
        headers["X-Amzn-SageMaker-Trajectory-Id"] = trajectory_id

    return headers


def get_inference_headers() -> dict[str, str]:
    """Get headers from current rollout context.

    For use with set_rollout_context() when you need to retrieve headers
    deep in the call stack without passing them explicitly.

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
