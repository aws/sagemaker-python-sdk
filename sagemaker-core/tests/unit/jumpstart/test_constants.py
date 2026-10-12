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
from unittest.mock import MagicMock, patch

from sagemaker.core.jumpstart import constants
from sagemaker.core.jumpstart.constants import _LazyJumpStartSagemakerSession


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
        # First access materializes the real Session.
        assert constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.boto_region_name == "us-west-2"
        assert _LazyJumpStartSagemakerSession._resolved is True
        # Second access reuses the cached Session (not rebuilt).
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
        # A shallow copy of the resolved session -- distinct object, same contents,
        # and crucially NOT the lazy proxy.
        assert not isinstance(result, _LazyJumpStartSagemakerSession)
        assert isinstance(result, SimpleNamespace)
        assert result is not fake
        # Mutating the copy (as the real caller does) must not raise.
        result.boto_session = "new"
        assert fake.boto_session == "orig"


def test_failed_build_degrades_to_none_contract(caplog):
    """If Session construction raises, resolution yields None and logs a warning;
    attribute access then behaves exactly as it would on ``None``."""
    with patch.object(constants, "Session", side_effect=RuntimeError("boom")):
        # Truthiness is still cheap and does not raise.
        assert bool(constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION) is True
        assert _LazyJumpStartSagemakerSession._resolve() is None
        with pytest.raises(AttributeError):
            _ = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION.boto_region_name


def test_mock_patch_of_class_level_attribute_tears_down_cleanly():
    """``mock.patch`` on a class-level session method must restore the original and
    leave nothing behind on the process-wide session.

    mock records ``is_local=False`` for a class-level attribute (it is absent from
    the instance ``__dict__``) and restores it by calling ``delattr``, so the proxy
    must forward ``__delattr__``. Without that forwarding the teardown raises
    ``AttributeError`` and the mock leaks into every later test in the same worker.
    """

    class FakeSession:
        """Stands in for ``Session``: ``read_s3_file`` is a class-level attribute."""

        def read_s3_file(self):
            return "real"

    fake = FakeSession()
    with patch.object(constants, "Session", return_value=fake):
        proxy = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION
        assert proxy.read_s3_file() == "real"

        with patch.object(proxy, "read_s3_file", return_value="mocked"):
            assert proxy.read_s3_file() == "mocked"

        # Teardown must restore the class method and leave no shadowing instance
        # attribute behind on the shared session.
        assert proxy.read_s3_file() == "real"
        assert "read_s3_file" not in fake.__dict__


def test_mock_patch_of_instance_level_attribute_tears_down_cleanly():
    """The instance-attribute path (``is_local=True``, restored via ``setattr``)
    must keep working -- guards against a regression in ``__setattr__`` forwarding."""

    class FakeSession:
        def __init__(self):
            self.sagemaker_client = "real-client"

    fake = FakeSession()
    with patch.object(constants, "Session", return_value=fake):
        proxy = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION
        with patch.object(proxy, "sagemaker_client", "mock-client"):
            assert proxy.sagemaker_client == "mock-client"
        assert proxy.sagemaker_client == "real-client"
