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
"""Tests for ModelTrainer networking config resolution.

These tests validate the _resolve_networking() logic that merges VpcConfig
from sagemaker_config into the ModelTrainer's networking attribute.
They specifically test the fix for issue #5766 where:
1) VpcConfig from sagemaker_config was ignored when Networking was not
   explicitly passed to ModelTrainer.
2) When Networking(enable_network_isolation=True) was passed without
   subnets/security_groups, VpcConfig from sagemaker_config was not merged.

These tests exercise the networking resolution logic directly and serve as
regression tests for the fix. They are designed to work with both the
minimal implementation and the full ModelTrainer class.
"""
from __future__ import absolute_import

from unittest.mock import MagicMock

from sagemaker.core.config.config_schema import (
    TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    TRAINING_JOB_SUBNETS_PATH,
    TRAINING_JOB_SECURITY_GROUP_IDS_PATH,
)
from sagemaker.train._model_trainer import ModelTrainer
from sagemaker.train.configs import Networking


def _make_mock_resolve(
    enable_network_isolation=None,
    subnets=None,
    security_group_ids=None,
):
    """Create a mock resolve function for config manager.

    Args:
        enable_network_isolation: Value to return for network isolation config path.
        subnets: Value to return for subnets config path.
        security_group_ids: Value to return for security group IDs config path.

    Returns:
        A callable that mimics resolve_value_from_config behavior.
    """

    def mock_resolve(config_path):
        config_map = {
            TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH: enable_network_isolation,
            TRAINING_JOB_SUBNETS_PATH: subnets,
            TRAINING_JOB_SECURITY_GROUP_IDS_PATH: security_group_ids,
        }
        return config_map.get(config_path)

    return mock_resolve


def _create_mock_config_mgr(mock_resolve):
    """Create a mock config manager with the given resolve function.

    Args:
        mock_resolve: The side_effect function for resolve_value_from_config.

    Returns:
        A MagicMock configured as a config manager.
    """
    mock_config_mgr = MagicMock()
    mock_config_mgr.resolve_value_from_config.side_effect = mock_resolve
    return mock_config_mgr


def test_resolve_networking_no_explicit_networking_applies_vpc_from_config():
    """Test that VpcConfig from sagemaker_config is applied when no Networking is provided."""
    mock_resolve = _make_mock_resolve(
        enable_network_isolation=True,
        subnets=["subnet-abc"],
        security_group_ids=["sg-123"],
    )
    mock_config_mgr = _create_mock_config_mgr(mock_resolve)

    trainer = ModelTrainer(networking=None, config_mgr=mock_config_mgr)

    assert trainer.networking is not None
    assert trainer.networking.enable_network_isolation is True
    assert trainer.networking.subnets == ["subnet-abc"]
    assert trainer.networking.security_group_ids == ["sg-123"]


def test_resolve_networking_isolation_only_merges_vpc_from_config():
    """Test that VpcConfig from sagemaker_config is merged when Networking only has enable_network_isolation."""
    networking = Networking(enable_network_isolation=True)

    mock_resolve = _make_mock_resolve(
        enable_network_isolation=True,
        subnets=["subnet-def"],
        security_group_ids=["sg-456"],
    )
    mock_config_mgr = _create_mock_config_mgr(mock_resolve)

    trainer = ModelTrainer(networking=networking, config_mgr=mock_config_mgr)

    assert trainer.networking is not None
    assert trainer.networking.enable_network_isolation is True
    assert trainer.networking.subnets == ["subnet-def"]
    assert trainer.networking.security_group_ids == ["sg-456"]


def test_resolve_networking_missing_security_groups_applies_from_config():
    """Test that security_group_ids from sagemaker_config is applied when not explicitly set."""
    networking = Networking(
        enable_network_isolation=False,
        subnets=["subnet-existing"],
        security_group_ids=None,
    )

    mock_resolve = _make_mock_resolve(
        enable_network_isolation=None,
        subnets=["subnet-from-config"],
        security_group_ids=["sg-from-config"],
    )
    mock_config_mgr = _create_mock_config_mgr(mock_resolve)

    trainer = ModelTrainer(networking=networking, config_mgr=mock_config_mgr)

    assert trainer.networking is not None
    # subnets should remain unchanged since they were already set
    assert trainer.networking.subnets == ["subnet-existing"]
    # security_group_ids should be populated from config
    assert trainer.networking.security_group_ids == ["sg-from-config"]


def test_resolve_networking_no_config_mgr():
    """Test that nothing happens when config_mgr is None."""
    trainer = ModelTrainer(networking=None, config_mgr=None)

    assert trainer.networking is None


def test_resolve_networking_no_config_values():
    """Test that nothing happens when config has no networking values."""
    mock_resolve = _make_mock_resolve(
        enable_network_isolation=None,
        subnets=None,
        security_group_ids=None,
    )
    mock_config_mgr = _create_mock_config_mgr(mock_resolve)

    trainer = ModelTrainer(networking=None, config_mgr=mock_config_mgr)

    assert trainer.networking is None


def test_resolve_networking_existing_values_not_overwritten():
    """Test that existing networking values are not overwritten by config."""
    networking = Networking(
        enable_network_isolation=True,
        subnets=["subnet-explicit"],
        security_group_ids=["sg-explicit"],
    )

    mock_resolve = _make_mock_resolve(
        enable_network_isolation=False,
        subnets=["subnet-config"],
        security_group_ids=["sg-config"],
    )
    mock_config_mgr = _create_mock_config_mgr(mock_resolve)

    trainer = ModelTrainer(networking=networking, config_mgr=mock_config_mgr)

    # Existing values should NOT be overwritten
    assert trainer.networking.enable_network_isolation is True
    assert trainer.networking.subnets == ["subnet-explicit"]
    assert trainer.networking.security_group_ids == ["sg-explicit"]
