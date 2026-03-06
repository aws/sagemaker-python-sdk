"""Strands framework adapter for automatic header and inference param injection.

Provides wrap_model() which wraps Strands models to automatically inject
X-Metadata headers and inference parameters into requests using the rollout context.
"""

from __future__ import annotations

from typing import Any

from sagemaker.rft.headers import get_inference_headers
from sagemaker.rft.context import get_inference_params


def wrap_model(model: Any) -> Any:
    """Wrap a Strands model to auto-inject headers and inference params from context.

    Wraps the model's format_request() method to:
    1. Add the X-Metadata header with rollout metadata
    2. Inject inference parameters (temperature, max_tokens, top_p)

    Works with Strands OpenAI-compatible models:
    - OpenAIModel
    - LiteLLMModel / LiteLLMModelNonStreaming

    Args:
        model: A Strands model instance with a format_request() method.

    Returns:
        The same model instance with format_request() wrapped.
    """
    original_format_request = model.format_request

    def wrapped_format_request(*args: Any, **kwargs: Any) -> dict:
        request = original_format_request(*args, **kwargs)

        headers = get_inference_headers()
        if headers:
            extra_headers = request.setdefault("extra_headers", {})
            extra_headers.update(headers)

        inference_params = get_inference_params()
        if inference_params:
            for key in ["temperature", "max_tokens", "top_p"]:
                if inference_params.get(key) is not None:
                    request[key] = inference_params[key]

        return request

    model.format_request = wrapped_format_request
    return model
