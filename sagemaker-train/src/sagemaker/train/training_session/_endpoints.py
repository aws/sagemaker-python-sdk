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
"""Region and endpoint-override resolution for the training-session clients.

Neither endpoint is computed here. Both clients built in
:mod:`sagemaker.train.training_session._clients` resolve their endpoint the way
every other boto3 client does, from the region through botocore's endpoint
rules, so a configuration that changes which host a Region answers on is
honoured without this package having to know about it.

To send calls somewhere else, an interface VPC endpoint for example, set the full
URL through an environment variable, the env-driven form of boto3's own
``endpoint_url``:

    SAGEMAKER_TRAINING_SESSION_ENDPOINT          full endpoint URL
    SAGEMAKER_TRAINING_SESSION_RUNTIME_ENDPOINT  full runtime endpoint URL

The two resolve independently, so a call is sent to a given URL only when the
variable naming that endpoint is set.
"""

from __future__ import annotations

import logging
import os

import boto3

logger = logging.getLogger(__name__)

# Endpoint overrides, read from the environment and not taken as client
# parameters.
_SAGEMAKER_ENDPOINT_ENV = "SAGEMAKER_TRAINING_SESSION_ENDPOINT"
_RUNTIME_ENDPOINT_ENV = "SAGEMAKER_TRAINING_SESSION_RUNTIME_ENDPOINT"


def _resolve_region(
    region: str | None = None,
    *,
    boto_session: boto3.Session | None = None,
) -> str:
    """Resolve the AWS Region to run in.

    In order: the Region named outright, then the Region of a session the
    caller supplied, then AWS_REGION, then the standard chain.

    ``Session.region_name`` covers AWS_DEFAULT_REGION and the shared config
    file, the same chain sagemaker-core resolves through, but does not read
    AWS_REGION, so that one is checked explicitly.

    A supplied session answers on its own, so its clients and these ones agree
    on where they are pointed even when the environment names somewhere else.

    Args:
        region: Region to use as given, resolved no further.
        boto_session: Session whose Region to prefer. ``None`` means the
            caller named no session, and the standard chain answers instead.

    Raises:
        ValueError: No region could be determined. Raised rather than defaulted,
            so an unconfigured environment cannot quietly create sessions,
            and bill for their GPUs, in a Region the caller never named.
    """
    if boto_session is not None:
        # The session answers on its own. Falling through to the standard chain
        # would take the Region from a different profile's configuration while
        # the credentials came from this one.
        resolved = region or boto_session.region_name
    else:
        resolved = region or os.environ.get("AWS_REGION") or boto3.Session().region_name
    if not resolved:
        raise ValueError(
            "No AWS Region configured. Pass region= to the client, or set up a "
            "local AWS configuration with a Region that offers training "
            "sessions (AWS_REGION, AWS_DEFAULT_REGION, or the region of your "
            "AWS profile)."
        )
    logger.info("Resolved AWS region: %s", resolved)
    return resolved


def _endpoint_overrides() -> tuple[str | None, str | None]:
    """Return ``(sagemaker_endpoint, runtime_endpoint)`` from the environment.

    ``None`` for an endpoint no variable names, which leaves that client to
    resolve the endpoint from its Region.
    """
    return (
        _checked_override(_SAGEMAKER_ENDPOINT_ENV),
        _checked_override(_RUNTIME_ENDPOINT_ENV),
    )


def _checked_override(variable: str) -> str | None:
    """Read an endpoint override, warning if it would send requests in the clear.

    A plain-HTTP endpoint is honoured rather than rejected, since a local test
    server is a reasonable thing to point at, but it is worth saying out loud:
    the request carries signed credentials and the training data itself.
    """
    url = os.environ.get(variable) or None
    if url is not None and url.startswith("http://"):
        logger.warning(
            "%s is an http:// URL, so requests to it, including their "
            "credentials and training data, are sent unencrypted. Use https:// "
            "unless this is a local endpoint you control.",
            variable,
        )
    return url
