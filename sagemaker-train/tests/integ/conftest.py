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
"""Shared pytest configuration for sagemaker-train integration tests.

These tests run under ``pytest -n auto`` (dozens of xdist workers). Many of them
resolve/validate an IAM execution role via ``TrainDefaults.get_role`` ->
``resolve_and_validate_role``, which internally calls the low-TPS
``iam:SimulatePrincipalPolicy`` API. With many workers hitting it at once IAM
throttles the request, surfacing as ``ClientError: (Throttling) ... Rate
exceeded`` and failing the test during setup.

This is purely a test-harness concurrency problem, so the mitigation lives here
in the test layer rather than in SDK source. It is intentionally identical to the
block in ``sagemaker-serve`` / ``sagemaker-mlops`` integ conftests:

* ``_configure_boto_adaptive_retries`` (autouse) — set adaptive retries via the
  ``AWS_RETRY_MODE`` / ``AWS_MAX_ATTEMPTS`` environment variables. Unlike setting
  a retry ``Config`` on a single boto session, env vars apply to *every* boto3
  client created in the worker — including the IAM clients the role resolver
  builds from an explicitly-passed ``Session`` (which carries no retry config of
  its own). ``adaptive`` mode adds client-side rate limiting so bursts of
  ``SimulatePrincipalPolicy`` calls ride out transient throttling.

* ``pytest_runtest_makereport`` — a belt-and-suspenders fallback. If a residual
  ``SimulatePrincipalPolicy`` throttling error still escapes after the adaptive
  retries, convert the failure into a skip so a transient rate limit never reds
  the build. Genuine failures (and throttling on any other operation) are
  untouched.
"""
from __future__ import absolute_import

import os

import pytest

# botocore adaptive retry settings for throttling-prone IAM validation calls.
# Applied via env vars so every client in the worker inherits them, regardless of
# which boto session the SDK ends up using to build its IAM client.
_RETRY_MODE = "adaptive"
_MAX_ATTEMPTS = "10"

# Only tolerate throttling on the IAM permission simulation used during role
# validation, so unrelated throttling still fails loudly.
_THROTTLE_ERROR_CODES = ("Throttling", "ThrottlingException", "RequestLimitExceeded")
_SIMULATE_OP = "SimulatePrincipalPolicy"


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


def _is_simulate_policy_throttle(exc):
    """Return True if ``exc`` is a SimulatePrincipalPolicy throttling ClientError."""
    from botocore.exceptions import ClientError

    if not isinstance(exc, ClientError):
        return False
    error_code = exc.response.get("Error", {}).get("Code", "")
    operation = getattr(exc, "operation_name", "") or ""
    return error_code in _THROTTLE_ERROR_CODES and operation == _SIMULATE_OP


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Skip (rather than fail) on residual SimulatePrincipalPolicy throttling that
    survives the adaptive retries under heavy concurrency. Applies to setup and
    call phases, since role resolution frequently happens in fixture setup."""
    outcome = yield
    report = outcome.get_result()

    if report.when not in ("setup", "call") or not report.failed:
        return

    exc = call.excinfo.value if call.excinfo else None
    # Walk the exception chain so a wrapped ClientError is still detected.
    while exc is not None:
        if _is_simulate_policy_throttle(exc):
            report.outcome = "skipped"
            report.wasxfail = (
                "iam:SimulatePrincipalPolicy throttled under concurrent test load"
            )
            break
        exc = exc.__cause__ or exc.__context__
