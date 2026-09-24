"""Rollout context management using contextvars.

Provides thread-safe storage for rollout metadata and inference parameters
that can be set at the top level and accessed deep in the call stack.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_rollout_metadata: ContextVar[dict[str, Any] | None] = ContextVar(
    "rollout_metadata", default=None
)
_inference_params: ContextVar[dict[str, Any] | None] = ContextVar(
    "inference_params", default=None
)


def set_rollout_context(
    metadata: dict[str, Any],
    inference_params: dict[str, Any] | None = None,
) -> None:
    """Store rollout metadata and inference params in context.

    Call this at the start of a rollout handler. Values are available
    via get_rollout_context() and get_inference_params() anywhere in the
    same thread/async context.

    Args:
        metadata: Rollout metadata dict from the rollout request.
        inference_params: Optional dict with sampling parameters
            (temperature, max_tokens, top_p).
    """
    _rollout_metadata.set(metadata)
    _inference_params.set(inference_params)


def get_rollout_context() -> dict[str, Any] | None:
    """Retrieve rollout metadata from context.

    Returns:
        The metadata dict if set, None otherwise.
    """
    return _rollout_metadata.get()


def get_inference_params() -> dict[str, Any] | None:
    """Retrieve inference parameters from context.

    Returns:
        The inference_params dict if set, None otherwise.
    """
    return _inference_params.get()


def clear_rollout_context() -> None:
    """Clear rollout metadata and inference params from context."""
    _rollout_metadata.set(None)
    _inference_params.set(None)
