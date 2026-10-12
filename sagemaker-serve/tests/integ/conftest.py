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

GPU capacity is a separate environmental failure: in us-west-2 SageMaker
regularly cannot provision the requested GPU instances and fails the endpoint /
job with an InsufficientInstanceCapacity reason after tens of minutes. Tests
marked ``xfail_on_insufficient_capacity`` report XFAIL instead of FAIL when they
fail for exactly that reason. The failure reason set by SageMaker is the signal:
the test still ends when SageMaker gives up (no client-side timeout that could
misclassify a slow but healthy deployment), and any other failure still fails
the test.
"""

from __future__ import absolute_import

import os
import re

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


# Failure reasons SageMaker reports when it cannot provision the requested
# instance capacity, as seen in this account:
#   Endpoint:          "Unable to provision requested ML compute capacity due to
#                       InsufficientInstanceCapacity error ..."
#   AIRecommendationJob: "... Could not deploy an endpoint for instance type
#                       'ml.g5.2xlarge': all reservation and on-demand capacity
#                       attempts were exhausted."
#   OptimizationJob:   "EC2InsufficientCapacityException: ... not available due
#                       to insufficient capacity."
_INSUFFICIENT_CAPACITY_REASON = re.compile(
    r"insufficient\s*(?:instance\s*)?capacity|capacity attempts were exhausted",
    re.IGNORECASE,
)
_XFAIL_REASON_MAX_CHARS = 500


def pytest_configure(config):
    """Register the ``xfail_on_insufficient_capacity`` marker."""
    config.addinivalue_line(
        "markers",
        "xfail_on_insufficient_capacity: report the test as XFAIL instead of FAIL when it "
        "fails because SageMaker could not provision the requested instance capacity.",
    )


def _is_insufficient_capacity(exc):
    """True if ``exc``, or an exception it was raised from, reports a capacity shortage."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if _INSUFFICIENT_CAPACITY_REASON.search(str(exc)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """XFAIL a marked test whose failure is an insufficient-capacity error."""
    try:
        return (yield)
    except Exception as exc:
        marked = item.get_closest_marker("xfail_on_insufficient_capacity") is not None
        if marked and _is_insufficient_capacity(exc):
            pytest.xfail(
                "SageMaker could not provision the requested instance capacity: "
                f"{str(exc)[:_XFAIL_REASON_MAX_CHARS]}"
            )
        raise
