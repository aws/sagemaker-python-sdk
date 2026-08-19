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

* ``_memoize_role_validation`` (autouse) — retries alone were not enough. A PR-gate
  run failed four tests with ``(Throttling) ... SimulatePrincipalPolicy (reached
  max retries: 9)``: the adaptive budget was exhausted, not merely stressed. The
  cause is volume, not burstiness — ~190 tests each construct a trainer, every
  construction calls ``get_role``, and each of those runs a *paginated*
  ``SimulatePrincipalPolicy`` over ~20 action names. Under ``-n auto`` on a large
  CodeBuild container that is thousands of calls against a low, account-wide TPS
  limit, so raising the retry budget only trades failures for a slower build.

  Since the arguments repeat, the result does too: this memoizes
  ``resolve_and_validate_role`` per worker, collapsing those calls to one per
  distinct ``(provided_role, role_type, region)``. Validation still happens — once,
  and its outcome (including a raised ``RoleValidationError``) is what gets reused,
  so a genuinely bad role still fails every test that uses it.

Throttling that still exhausts the retry budget after memoization is deliberately
left to fail the test loudly (rather than being converted to a skip), so a
persistent rate-limit regression stays visible instead of silently disappearing
from the results.
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


# Modules that did `from ...iam_role_resolver import resolve_and_validate_role`
# hold their own reference to the original function, so patching only the defining
# module would leave those bindings calling IAM directly. Each importer is patched
# too. Kept as a list of (module path, attribute) so adding a caller is one line.
_ROLE_RESOLVER_CALLERS = (
    ("sagemaker.core.helper.iam_role_resolver", "resolve_and_validate_role"),
    ("sagemaker.train.defaults", "resolve_and_validate_role"),
    ("sagemaker.train.evaluate.base_evaluator", "resolve_and_validate_role"),
)


@pytest.fixture(autouse=True, scope="session")
def _memoize_role_validation():
    """Validate each distinct role once per xdist worker instead of once per test.

    See this module's docstring for why retries alone were insufficient. Caches on
    ``(provided_role, role_type, region)`` -- region is part of the key because the
    Nova tests validate the same role against us-east-1, and a role's resolution is
    region-scoped. Exceptions are cached alongside successes so a bad role keeps
    failing rather than being silently retried per test.
    """
    import importlib

    patched = []
    cache = {}

    try:
        source = importlib.import_module(_ROLE_RESOLVER_CALLERS[0][0])
    except ImportError:  # pragma: no cover - SDK layout changed
        yield
        return

    original = source.resolve_and_validate_role

    def memoized(provided_role=None, role_type=None, sagemaker_session=None, **kwargs):
        region = None
        if sagemaker_session is not None:
            region = getattr(sagemaker_session, "boto_region_name", None)
        key = (provided_role, role_type, region)

        if key not in cache:
            try:
                cache[key] = (
                    original(
                        provided_role=provided_role,
                        role_type=role_type,
                        sagemaker_session=sagemaker_session,
                        **kwargs,
                    ),
                    None,
                )
            except Exception as exc:  # cache the verdict, not just the happy path
                cache[key] = (None, exc)

        result, error = cache[key]
        if error is not None:
            raise error
        return result

    for module_path, attribute in _ROLE_RESOLVER_CALLERS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue  # optional/renamed caller; the others still get patched
        if getattr(module, attribute, None) is original:
            setattr(module, attribute, memoized)
        patched.append((module, attribute))

    yield

    # Restore by checking for `memoized` rather than only undoing what was patched
    # above: a caller imported *after* the source module was patched binds the
    # memoized function at its own import time, so it needs restoring too even
    # though this fixture never set it.
    for module, attribute in patched:
        if getattr(module, attribute, None) is memoized:
            setattr(module, attribute, original)
