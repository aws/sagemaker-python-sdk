"""LangChain adapter for automatic header injection.

Provides helper functions to create httpx clients that automatically inject
X-Metadata headers into inference requests using the rollout context.
"""

from __future__ import annotations

import httpx

from sagemaker.rft.headers import get_inference_headers


def _inject_headers(request: httpx.Request) -> None:
    """Event hook that injects headers from rollout context."""
    headers = get_inference_headers()
    if headers:
        request.headers.update(headers)


def create_http_client(**kwargs) -> httpx.Client:
    """Create an httpx Client that auto-injects X-Metadata headers.

    Use with LangChain's ChatOpenAI http_client parameter.

    Args:
        **kwargs: Additional arguments passed to httpx.Client.

    Returns:
        httpx.Client configured with header injection.
    """
    event_hooks = kwargs.pop("event_hooks", {})
    request_hooks = event_hooks.get("request", [])
    request_hooks.append(_inject_headers)
    event_hooks["request"] = request_hooks
    return httpx.Client(event_hooks=event_hooks, **kwargs)


def create_async_http_client(**kwargs) -> httpx.AsyncClient:
    """Create an httpx AsyncClient that auto-injects X-Metadata headers.

    Use with LangChain's ChatOpenAI http_async_client parameter.

    Args:
        **kwargs: Additional arguments passed to httpx.AsyncClient.

    Returns:
        httpx.AsyncClient configured with header injection.
    """
    event_hooks = kwargs.pop("event_hooks", {})
    request_hooks = event_hooks.get("request", [])
    request_hooks.append(_inject_headers)
    event_hooks["request"] = request_hooks
    return httpx.AsyncClient(event_hooks=event_hooks, **kwargs)


def create_http_clients(**kwargs) -> tuple[httpx.Client, httpx.AsyncClient]:
    """Create both sync and async httpx clients with header injection.

    Args:
        **kwargs: Additional arguments passed to both clients.

    Returns:
        Tuple of (httpx.Client, httpx.AsyncClient).
    """
    return create_http_client(**kwargs), create_async_http_client(**kwargs)
