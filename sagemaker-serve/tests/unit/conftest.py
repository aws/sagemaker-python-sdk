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
"""Shared fixtures for the sagemaker-serve unit tests.

Keeps role resolution off the network. ``ModelBuilder.__post_init__`` resolves a
default execution role when none is supplied, and ``resolve_and_validate_role``
does that by calling ``sts:GetCallerIdentity`` followed by a paginated
``iam:SimulatePrincipalPolicy``. Any unit test that builds a ModelBuilder without
passing ``role_arn`` therefore makes live IAM calls, which only succeed because
CI credentials happen to be present.

Those calls are the dominant source of CI flakiness for this package: IAM
throttles on a shared, account-wide quota, so builds fail with
``Throttling ... SimulatePrincipalPolicy ... reached max retries`` on whichever
test happened to be running. The failures are not reproducible -- each build runs
the suite three times (py310/py311/py312) and these failures almost always hit
only one of the three.
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest

# Stands in for "the caller's own identity role", which the real resolver obtains
# from sts:GetCallerIdentity.
DEFAULT_UNIT_TEST_ROLE = "arn:aws:iam::123456789012:role/UnitTestExecutionRole"

# resolve_and_validate_role has to be patched once per importing module, not once
# at its definition site: both modules below bind it with a module-level
# ``from ... import resolve_and_validate_role``, so each holds its own reference
# and patching sagemaker.core.helper.iam_role_resolver would not rebind either.
_RESOLVER_LOOKUP_SITES = (
    "sagemaker.serve.model_builder.resolve_and_validate_role",
    "sagemaker.serve.bedrock_model_builder.resolve_and_validate_role",
)


def _resolve_role_offline(provided_role=None, *, role_arn=None, **kwargs):
    """Mirror resolve_and_validate_role's contract without touching IAM.

    The real function returns the role it was given and only falls back to the
    caller's identity when none was supplied. Preserving that passthrough matters:
    four of its five call sites in this package pass a real role, and tests assert
    on the value that comes back.
    """
    return provided_role or role_arn or DEFAULT_UNIT_TEST_ROLE


@pytest.fixture(autouse=True)
def stub_iam_role_resolution():
    """Resolve execution roles locally for every unit test in this package.

    Tests that patch ``resolve_and_validate_role`` themselves still win, because
    their patch is applied inside this one and so rebinds the same attribute last.
    """
    with ExitStack() as stack:
        for target in _RESOLVER_LOOKUP_SITES:
            stack.enter_context(patch(target, side_effect=_resolve_role_offline))
        yield
