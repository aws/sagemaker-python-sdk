"""Strands framework adapter for automatic header and inference param injection.

Provides wrap_model() which wraps a Strands model to automatically inject
RFT headers and inference parameters into requests using the rollout context.

The wrapper intercepts ``stream()`` and injects headers via
``client_args["default_headers"]`` because Strands ``OpenAIModel`` creates
a new OpenAI client per request from ``client_args``.
"""

from __future__ import annotations

import logging
from typing import Any

from sagemaker.rft.headers import get_inference_headers
from sagemaker.rft.context import get_inference_params

logger = logging.getLogger(__name__)


def wrap_model(model: Any) -> Any:
    """Wrap a Strands model to auto-inject headers and inference params from context.

    Creates a transparent proxy that:
    1. Injects RFT headers (X-Rft-Job-Arn, X-Trajectory-Id, X-Span-Id) via
       client_args["default_headers"] on every stream() call
    2. Injects inference parameters (temperature, max_tokens, top_p)

    Args:
        model: A Strands model instance (e.g., ``OpenAIModel``).

    Returns:
        A wrapped model that transparently injects RFT headers.
    """
    return _RFTModelWrapper(model)


class _RFTModelWrapper:
    """Transparent proxy that injects RFT headers into Strands model calls.

    Delegates all attribute access to the inner model so it quacks like
    the original. Intercepts ``stream()`` to inject headers.
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
        """Intercept stream() to inject RFT headers via client_args default_headers.

        The OpenAI client supports ``default_headers`` in its constructor,
        which are sent with every request. We inject the RFT headers there since
        Strands OpenAIModel creates a new client per request from ``client_args``.
        """
        rft_headers = get_inference_headers()
        if rft_headers:
            client_args = getattr(self._inner, "client_args", None)
            if client_args is not None:
                existing = client_args.get("default_headers") or {}
                existing.update(rft_headers)
                client_args["default_headers"] = existing
            logger.debug("Injected RFT headers: %s", list(rft_headers.keys()))

        # Inject inference params if available
        inference_params = get_inference_params()
        if inference_params:
            params = getattr(self._inner, "params", None)
            if params is not None and isinstance(params, dict):
                for key in ["temperature", "max_tokens", "top_p"]:
                    if inference_params.get(key) is not None:
                        params[key] = inference_params[key]

        return self._inner.stream(*args, **kwargs)

    def update_config(self, **model_config: Any) -> None:
        return self._inner.update_config(**model_config)

    def get_config(self) -> Any:
        return self._inner.get_config()

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.structured_output(*args, **kwargs)
