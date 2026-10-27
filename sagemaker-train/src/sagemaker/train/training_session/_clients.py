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
"""boto3 clients for the training-session control plane and runtime.

Both clients are ordinary boto3 clients: botocore serializes the request,
signs it, applies its own retry policy, and parses the response. Neither
endpoint is reached by hand-signed HTTP.

The control-plane operations (``CreateTrainingSession`` and friends) are part of
the ``sagemaker`` service; the runtime is a separate service. Both models come
from the installed botocore, so a botocore that predates either surface fails
with an actionable message rather than an opaque lookup error.

Neither client is given an endpoint to call. Both resolve it from their Region
through botocore's endpoint rules, so a configuration that changes which host a
Region answers on is honoured for free. A URL in the environment (see
:mod:`sagemaker.train.training_session._endpoints`) overrides that resolution
for the one endpoint it names.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import boto3
import botocore
from botocore.config import Config
from botocore.exceptions import UnknownServiceError

from sagemaker.train.training_session._endpoints import (
    _endpoint_overrides,
    _resolve_region,
)

logger = logging.getLogger(__name__)

#: botocore's name for the runtime service.
_RUNTIME_SERVICE_NAME = "sagemakertrainingsessionruntime"

#: An operation that only a botocore carrying the training-session API has, used
#: to check the installed version without trusting its version string.
_CONTROL_PLANE_PROBE_OPERATION = "CreateTrainingSession"

#: Control-plane retries per operation, on top of the first try, in botocore's
#: ``standard`` retry mode: throttling, 5xx and connection errors are retried
#: with exponential backoff, and everything else, ``ValidationException`` and
#: ``ResourceNotFound`` above all, is raised on the first try. Three retries
#: means four attempts in total.
_CONTROL_PLANE_RETRIES = 3

#: Runtime calls are sent once. Submitting an operation and polling for its
#: result are both already wrapped in retry loops that understand which failures
#: an in-flight operation can survive, so a second retry policy underneath them
#: would multiply the wait for no gain.
_RUNTIME_RETRIES = 0

#: Connections kept in the runtime pool. Sampling fans out far wider than the
#: botocore default of 10 (128 concurrent episodes * 8 samples = 1024 in-flight
#: polls), and a pool smaller than the fan-out serializes the polls behind it.
_DEFAULT_MAX_POOL_CONNECTIONS = 512

#: Seconds to wait for a connection to be established. Reaching the endpoint is
#: either quick or blocked, so a connection that has not opened by now will not,
#: and waiting out botocore's default of 60 only delays the retry that recovers.
_CONNECT_TIMEOUT = 10

#: Seconds to wait for a response once the request is sent. Every operation here
#: returns promptly (work is submitted and polled for, never awaited on the
#: connection), so this bounds a stalled socket rather than a slow operation.
#: Left at botocore's default, stated so it is a decision rather than inherited.
_READ_TIMEOUT = 60


def _control_plane_client(
    region: Optional[str] = None,
    *,
    boto_session: Optional[boto3.Session] = None,
) -> Any:
    """Return a ``sagemaker`` client for the training-session control plane.

    Raises:
        RuntimeError: The installed botocore has no training-session operations.
    """
    resolved_region = _resolve_region(region, boto_session=boto_session)
    endpoint, _ = _endpoint_overrides()
    session = boto_session or boto3.Session()
    client = session.client(
        "sagemaker",
        region_name=resolved_region,
        endpoint_url=endpoint,  # None leaves the endpoint to botocore's rules.
        config=Config(
            retries={"mode": "standard", "max_attempts": _CONTROL_PLANE_RETRIES},
            connect_timeout=_CONNECT_TIMEOUT,
            read_timeout=_READ_TIMEOUT,
        ),
    )
    _require_control_plane_operations(client)
    return client


def _require_control_plane_operations(client: Any) -> None:
    """Fail with an actionable message when botocore predates the API.

    Asks the loaded model what it supports, not what its version string claims,
    which answers the question directly.
    """
    if _CONTROL_PLANE_PROBE_OPERATION in client.meta.service_model.operation_names:
        return
    raise RuntimeError(
        "The installed botocore does not describe the SageMaker training-session "
        f"operations (botocore {botocore.__version__}), so there is nothing for "
        "this client to call. Install a botocore that carries them, then create "
        "the client again."
    )


def _runtime_client(
    region: Optional[str] = None,
    *,
    boto_session: Optional[boto3.Session] = None,
    max_pool_connections: int = _DEFAULT_MAX_POOL_CONNECTIONS,
) -> Any:
    """Return a client for the training-session runtime.

    Args:
        region: AWS region; resolved from the environment when not supplied.
        boto_session: Session whose credentials sign the calls. Defaults to a
            fresh session using the standard credential chain.
        max_pool_connections: Connections kept in the HTTP pool. Raise it for
            sampling workloads that fan out past the default.

    Raises:
        RuntimeError: The installed botocore does not model the runtime.
    """
    resolved_region = _resolve_region(region, boto_session=boto_session)
    _, endpoint = _endpoint_overrides()
    session = boto_session or boto3.Session()
    try:
        return session.client(
            _RUNTIME_SERVICE_NAME,
            region_name=resolved_region,
            endpoint_url=endpoint,  # None leaves the endpoint to botocore's rules.
            config=Config(
                retries={"mode": "standard", "max_attempts": _RUNTIME_RETRIES},
                max_pool_connections=max_pool_connections,
                connect_timeout=_CONNECT_TIMEOUT,
                read_timeout=_READ_TIMEOUT,
            ),
        )
    except UnknownServiceError as error:
        raise RuntimeError(
            "The installed botocore does not describe the SageMaker training-session "
            f"runtime (botocore {botocore.__version__}), so there is nothing for "
            "this client to call. Install a botocore that carries it, then create "
            "the client again."
        ) from error
