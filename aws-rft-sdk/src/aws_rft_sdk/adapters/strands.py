"""Strands model adapter — wraps a Strands model to inject RFT headers.

Usage::

    from aws_rft_sdk.adapters.strands import wrap_model
    from strands.models.openai import OpenAIModel

    model = OpenAIModel(
        client_args={"api_key": key, "base_url": endpoint},
        model_id="my-model",
    )
    model = wrap_model(model)  # Now injects X-RFT-* headers on every call

Requires the Strands OpenAIModel to pass through ``extra_headers`` kwarg
to the underlying OpenAI client (supported since strands-agents >= X.Y.Z).
"""

import logging
from typing import Any

from aws_rft_sdk.context import RFTContext

logger = logging.getLogger(__name__)


def wrap_model(model: Any) -> Any:
    """Wrap a Strands model to automatically inject RFT training headers.

    The wrapper reads the current rollout context (populated by ``@rft_handler``)
    and adds ``X-RFT-*`` headers to every inference request so the training
    inference endpoint can correlate requests with rollouts.

    Args:
        model: A Strands model instance (e.g., ``OpenAIModel``).

    Returns:
        A wrapped model that transparently injects RFT headers.
    """
    return _RFTModelWrapper(model)


class _RFTModelWrapper:
    """Transparent proxy that injects RFT headers into Strands model calls.

    Delegates all attribute access to the inner model so it quacks like
    the original. Intercepts ``stream()`` to inject ``extra_headers``.
    """

    def __init__(self, inner_model: Any):
        object.__setattr__(self, "_inner", inner_model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any):
        if name == "_inner":
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept stream() to inject RFT headers via extra_headers kwarg."""
        rft_headers = RFTContext.get_headers()
        if rft_headers:
            existing = kwargs.get("extra_headers") or {}
            existing.update(rft_headers)
            kwargs["extra_headers"] = existing
            logger.debug("Injected RFT headers: %s", list(rft_headers.keys()))
        return self._inner.stream(*args, **kwargs)

    def update_config(self, **model_config: Any) -> None:
        return self._inner.update_config(**model_config)

    def get_config(self) -> Any:
        return self._inner.get_config()

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.structured_output(*args, **kwargs)
