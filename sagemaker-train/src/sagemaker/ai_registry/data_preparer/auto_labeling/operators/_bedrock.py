# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A Bedrock Converse client for the one single-shot model call a labeling loop makes.

Data generation, labeling, the confidence jury and the graders all run inside a
labeling job. Spec induction is the one remaining single-shot call: it is not a
generation job, so it goes to Bedrock directly through ``bedrock-runtime`` Converse.
Callers keep the small ``make_client(region)`` + ``invoke(client, model_id, prompt)``
contract; retries and backoff for transient errors live here, in one place.
"""

from __future__ import absolute_import

import time
from typing import Optional

_TEMPERATURE = 0.0
_MAX_TOKENS = 512
_MAX_ATTEMPTS = 6
# Bedrock throttling / transient error codes worth retrying.
_RETRY_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
    }
)


def _set_telemetry_attribution() -> None:
    """Tag the process's SageMaker telemetry as the SageMaker agent plugin's (usage attribution).

    Best-effort and process-global, as the command line had it: the operators are the
    agent plugin's code paths, so :func:`make_client` and the labeling job client
    (:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client`) set it
    before their first model or service call.
    """
    try:
        from sagemaker.core import Attribution, set_attribution

        set_attribution(Attribution.SAGEMAKER_AGENT_PLUGIN)
    except Exception:  # noqa: BLE001 - attribution is best-effort
        pass


def make_client(region: Optional[str], read_timeout: int = 120, connect_timeout: int = 10):
    """A boto3 ``bedrock-runtime`` client for ``region``.

    It tags the process as the SageMaker agent plugin (telemetry attribution) before any
    model call. Adaptive rate limiting is on and boto3's own retries are off
    (``max_attempts=1``), so retries live in exactly one place: the backoff loop in
    :func:`invoke`.

    Args:
        region: The Bedrock region.
        read_timeout: Seconds to wait for a response. 120 suits a short call; a long
            generation (a spec of up to 16,000 tokens) needs more, 600 say, since a
            botocore ``ReadTimeoutError`` is not a ``ClientError`` and is not retried.
        connect_timeout: Seconds to wait for the connection.

    Returns:
        The client.
    """
    import boto3
    from botocore.config import Config

    _set_telemetry_attribution()
    cfg = Config(
        retries={"mode": "adaptive", "max_attempts": 1},
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
    )
    return boto3.client("bedrock-runtime", region_name=region, config=cfg)


def invoke(  # pylint: disable=too-many-positional-arguments
    client,
    model_id: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    omit_temperature: bool = False,
) -> str:
    """One Converse call; the model's text.

    Transient Bedrock errors (throttling, service unavailable, model timeout) and
    connection failures are retried up to ``_MAX_ATTEMPTS`` times with a deterministic
    exponential backoff (1, 2, 4, 8, 16 s; no jitter, so a resumed run is reproducible).
    When the model refuses the temperature field (:func:`rejects_temperature`) the same
    request goes once more without it.

    Args:
        client: A :func:`make_client` client.
        model_id: The Bedrock model id.
        prompt: The user turn.
        system_prompt: An optional system turn.
        temperature: Overrides the deterministic 0.0 default.
        max_tokens: Overrides the short default (512) for a long generation.
        omit_temperature: Send no temperature field.

    Returns:
        The text of every text block in the reply, joined.

    Raises:
        botocore.exceptions.ClientError: A non-retryable error, or the last retry.
    """
    from botocore.exceptions import (
        ClientError,
        ConnectionError as BotoConnectionError,
        ConnectionClosedError,
    )

    inference = {"maxTokens": max_tokens or _MAX_TOKENS}
    if not omit_temperature:
        inference["temperature"] = _TEMPERATURE if temperature is None else temperature
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    kwargs = {"modelId": model_id, "messages": messages, "inferenceConfig": inference}
    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = client.converse(**kwargs)
            # Join every text block: a long generation spans several, and a reasoning
            # model may lead with a non-text block ([0]["text"] would raise KeyError).
            blocks = resp["output"]["message"]["content"]
            return "".join(b.get("text", "") for b in blocks)
        except ClientError as e:
            if not omit_temperature and rejects_temperature(e):
                # The model takes no temperature: the same request once more without
                # it (one shot: that call sends none, so it cannot be refused again).
                return invoke(
                    client,
                    model_id,
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    omit_temperature=True,
                )
            code = e.response.get("Error", {}).get("Code", "")
            if code not in _RETRY_CODES or attempt == _MAX_ATTEMPTS:
                raise
            # Deterministic exponential backoff: 1, 2, 4, 8, 16s (no jitter so a
            # resumed run is reproducible).
            time.sleep(2 ** (attempt - 1))
        except (BotoConnectionError, ConnectionClosedError):
            # boto's own retries are off, so retry connection failures here, including
            # ConnectionClosedError (an HTTPClientError, not a ConnectionError), common
            # on long generations. ReadTimeoutError stays non-retryable.
            if attempt == _MAX_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))


def rejects_temperature(error) -> bool:
    """True when a Converse ``ClientError`` is the model refusing the temperature field.

    A ``ValidationException`` whose message names ``temperature`` and says it is
    deprecated or not supported, as models that take an effort setting instead do.
    """
    resp = getattr(error, "response", None) or {}
    err = resp.get("Error") or {}
    msg = str(err.get("Message", "")).lower()
    return (
        err.get("Code") == "ValidationException"
        and "temperature" in msg
        and any(w in msg for w in ("deprecated", "not supported", "unsupported"))
    )
