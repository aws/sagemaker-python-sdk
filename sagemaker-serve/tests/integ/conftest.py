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
the test layer rather than in SDK source:

* ``_configure_default_boto_retries`` (autouse) — approach "A": set an adaptive
  retry policy on boto3's *default* session. The role resolver, when no explicit
  session is passed, falls back to ``sagemaker.core...Session()`` whose boto
  session is ``boto3.DEFAULT_SESSION`` (see session_helper._initialize). Clients
  created from it therefore inherit these retries, letting the internal IAM calls
  ride out transient throttling without any source change.

* ``pytest_runtest_makereport`` — approach "C": a belt-and-suspenders fallback. If
  a residual ``SimulatePrincipalPolicy`` throttling error still escapes after the
  adaptive retries, convert the failure into an xfail so a transient rate limit
  never reds the build. Genuine failures are untouched.
"""
from __future__ import absolute_import

import boto3
import pytest
from botocore.config import Config

# Higher, adaptive retry budget for the throttling-prone IAM validation calls.
_ADAPTIVE_RETRY_CONFIG = Config(retries={"max_attempts": 10, "mode": "adaptive"})

# Signature of the throttled call we tolerate: the IAM permission simulation used
# during role validation. Kept specific so unrelated throttling still fails loudly.
_THROTTLE_ERROR_CODES = ("Throttling", "ThrottlingException", "RequestLimitExceeded")
_SIMULATE_OP = "SimulatePrincipalPolicy"


@pytest.fixture(autouse=True, scope="session")
def _configure_default_boto_retries():
    """Approach A: give boto3's default session adaptive retries for IAM calls.

    The role resolver builds its IAM client from the default boto session when no
    explicit session is supplied, so clients created after this runs inherit the
    adaptive retry policy and absorb transient ``SimulatePrincipalPolicy``
    throttling instead of erroring out.
    """
    boto3.setup_default_session()
    # botocore honors this default config for every client created from the
    # session afterwards (including ones built deep inside the SDK).
    boto3.DEFAULT_SESSION._session.set_default_client_config(_ADAPTIVE_RETRY_CONFIG)
    yield


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
    """Approach C: xfail (don't fail) on residual SimulatePrincipalPolicy throttling.

    If the adaptive retries above are still not enough under extreme concurrency,
    treat the throttled role-permission simulation as an expected-environmental
    condition rather than a test failure. Any other error is reported normally.
    """
    outcome = yield
    report = outcome.get_result()

    if report.when != "call" or not report.failed:
        return

    exc = call.excinfo.value if call.excinfo else None
    # Walk the exception chain so a wrapped/chained ClientError is still detected.
    while exc is not None:
        if _is_simulate_policy_throttle(exc):
            report.outcome = "skipped"
            report.wasxfail = (
                "iam:SimulatePrincipalPolicy throttled under concurrent test load"
            )
            break
        exc = exc.__cause__ or exc.__context__
