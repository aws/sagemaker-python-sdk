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
"""Shared pytest configuration for sagemaker-serve integration tests.

These tests run under ``pytest -n auto`` (dozens of xdist workers). Several of
them resolve/validate an IAM execution role during ``ModelBuilder.build()`` /
``.deploy()``, which internally calls the low-TPS ``iam:SimulatePrincipalPolicy``
API. With many workers hitting it at once IAM throttles the request, surfacing as
``ClientError: (Throttling) ... Rate exceeded`` and failing the build.

This is purely a test-harness concurrency problem, so the mitigation lives here in
the test layer rather than in SDK source. It is intentionally identical to the
block in ``sagemaker-train`` / ``sagemaker-mlops`` integ conftests:

* ``_configure_boto_adaptive_retries`` (autouse) — set adaptive retries via the
  ``AWS_RETRY_MODE`` / ``AWS_MAX_ATTEMPTS`` environment variables. Env vars apply
  to *every* boto3 client created in the worker, so the internal IAM calls ride
  out transient throttling whether the resolver falls back to the default session
  or builds its client from an explicitly-passed ``Session`` (several serve tests
  pass their own session, whose IAM client would otherwise carry botocore's
  default 4-attempt retry policy). ``adaptive`` mode also adds client-side rate
  limiting to smooth bursts.

Throttling that still exhausts the adaptive retry budget is deliberately left to
fail the test loudly (rather than being converted to a skip), so a persistent
rate-limit regression stays visible instead of silently disappearing from the
results.
"""
from __future__ import absolute_import

import os

import pytest

# botocore adaptive retry settings for throttling-prone IAM validation calls.
# Applied via env vars so every client in the worker inherits them, regardless of
# which boto session the SDK ends up using to build its IAM client.
_RETRY_MODE = "adaptive"
_MAX_ATTEMPTS = "10"


@pytest.fixture(autouse=True, scope="session")
def _configure_boto_adaptive_retries():
    """Give every boto3 client in this xdist worker adaptive retries so the IAM
    clients built by the role resolver absorb transient SimulatePrincipalPolicy
    throttling. Restores any pre-existing values on teardown."""
    previous = {
        "AWS_RETRY_MODE": os.environ.get("AWS_RETRY_MODE"),
        "AWS_MAX_ATTEMPTS": os.environ.get("AWS_MAX_ATTEMPTS"),
    }
    os.environ["AWS_RETRY_MODE"] = _RETRY_MODE
    os.environ["AWS_MAX_ATTEMPTS"] = _MAX_ATTEMPTS
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
