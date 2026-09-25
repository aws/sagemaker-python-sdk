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
"""Tests for lazy initialization of DEFAULT_JUMPSTART_SAGEMAKER_SESSION (GH #4468)."""

from __future__ import absolute_import

import copy
from types import SimpleNamespace

import pytest
from mock import MagicMock, patch

from sagemaker.jumpstart import constants
from sagemaker.jumpstart.constants import _LazyJumpStartSagemakerSession


@pytest.fixture(autouse=True)
def reset_lazy_session_cache():
    """Ensure each test starts and ends with an unresolved proxy cache."""
    _LazyJumpStartSagemakerSession._resolved = False
    _LazyJumpStartSagemakerSession._session = None
    yield
    _LazyJumpStartSagemakerSession._resolved = False
    _LazyJumpStartSagemakerSession._session = None


def test_default_session_is_a_lazy_proxy():
    assert isinstance(constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION, _LazyJumpStartSagemakerSession)


def test_truthiness_does_not_build_a_session():
    """``session or DEFAULT_...`` / ``if session:`` must stay lazy (no boto clients)."""
    with patch.object(constants, "Session") as session_cls:
        assert bool(constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION) is True
        assert _LazyJumpStartSagemakerSession._resolved is False
        session_cls.assert_not_called()


def test_first_attribute_access_builds_session_once():
    fake = MagicMock()
    fake.boto_region_name = "us-west-2"
    with patch.object(constants, "Session", return_value=fake) as session_cls:
        assert constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.boto_region_name == "us-west-2"
        assert _LazyJumpStartSagemakerSession._resolved is True
        _ = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.boto_region_name
        session_cls.assert_called_once()


def test_setattr_is_forwarded_to_real_session():
    fake = MagicMock()
    with patch.object(constants, "Session", return_value=fake):
        constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.sagemaker_client = "client"
        assert fake.sagemaker_client == "client"


def test_copy_returns_the_real_session():
    """utils.get_default_jumpstart_session_with_user_agent_suffix copies then mutates,
    so copy.copy(proxy) must yield a real (copyable, mutable) session, not the proxy."""
    fake = SimpleNamespace(boto_session="orig", sagemaker_client="orig")
    with patch.object(constants, "Session", return_value=fake):
        result = copy.copy(constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION)
        assert not isinstance(result, _LazyJumpStartSagemakerSession)
        assert isinstance(result, SimpleNamespace)
        assert result is not fake
        result.boto_session = "new"
        assert fake.boto_session == "orig"


def test_failed_build_degrades_to_none_contract():
    """If Session construction raises, resolution yields None and logs a warning;
    attribute access then behaves exactly as it would on ``None``."""
    with patch.object(constants, "Session", side_effect=RuntimeError("boom")):
        assert bool(constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION) is True
        assert _LazyJumpStartSagemakerSession._resolve() is None
        with pytest.raises(AttributeError):
            _ = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.boto_region_name
