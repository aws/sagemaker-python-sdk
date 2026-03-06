"""Header utilities for inference calls.

Provides functions to create the X-Metadata header required for inference
requests to the training platform.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

from sagemaker.rft.context import get_rollout_context
from sagemaker.rft.models import RolloutMetadata


def make_inference_headers(metadata: dict[str, Any] | RolloutMetadata) -> dict[str, str]:
    """Create headers dict for inference calls.

    Add these headers to your HTTP client when making inference calls
    to the training platform.

    Args:
        metadata: The metadata from the rollout request (dict or RolloutMetadata).

    Returns:
        Headers dict to add to inference calls.
    """
    if isinstance(metadata, RolloutMetadata):
        metadata = metadata.model_dump()
    return {"X-Metadata": json.dumps(metadata)}


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
    return {"X-Metadata": json.dumps(metadata)}
