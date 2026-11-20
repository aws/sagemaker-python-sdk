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
"""Unit tests for sagemaker.core.user_agent SDK version resolution."""
from __future__ import absolute_import

from unittest.mock import patch

import importlib_metadata

from sagemaker.core import user_agent


def test_resolve_prefers_meta_sagemaker():
    """The meta ``sagemaker`` distribution version is preferred and tried first."""
    with patch.object(
        user_agent.importlib_metadata, "version", return_value="3.27.0"
    ) as mock_version:
        assert user_agent._resolve_sdk_version() == "3.27.0"
        mock_version.assert_called_with("sagemaker")


def test_resolve_falls_back_to_core_when_meta_absent():
    """When the meta package is not installed, fall back to ``sagemaker-core``."""

    def fake_version(distribution):
        if distribution == "sagemaker":
            raise importlib_metadata.PackageNotFoundError(distribution)
        return "2.27.0"

    with patch.object(user_agent.importlib_metadata, "version", side_effect=fake_version):
        assert user_agent._resolve_sdk_version() == "2.27.0"


def test_resolve_static_fallback_when_none_installed():
    """When neither distribution resolves, return the static fallback (never raise)."""

    def raise_not_found(distribution):
        raise importlib_metadata.PackageNotFoundError(distribution)

    with patch.object(user_agent.importlib_metadata, "version", side_effect=raise_not_found):
        assert user_agent._resolve_sdk_version() == "3.0"


def test_version_is_dynamic_not_hardcoded():
    """Regression guard for the hardcoded "3.0": the version must echo installed metadata."""
    with patch.object(user_agent.importlib_metadata, "version", return_value="9.9.9"):
        assert user_agent._resolve_sdk_version() == "9.9.9"


def test_user_agent_suffix_uses_resolved_version():
    """The user-agent suffix carries the resolved SDK version."""
    with patch.object(user_agent, "SDK_VERSION", "9.9.9"):
        assert user_agent.get_user_agent_extra_suffix() == f"lib/{user_agent.SDK_PREFIX}#9.9.9"
