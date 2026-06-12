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
from __future__ import absolute_import
import pytest
from unittest.mock import patch, Mock
from sagemaker.jumpstart import utils
from sagemaker.jumpstart.enums import JumpStartScriptScope, JumpStartConfigRankingName
from sagemaker.jumpstart.factory.estimator import _add_config_name_to_kwargs
from sagemaker.jumpstart.factory.model import (
    _add_config_name_to_init_kwargs,
    _add_config_name_to_deploy_kwargs,
)
from sagemaker.jumpstart.types import JumpStartEstimatorInitKwargs, JumpStartModelInitKwargs


class TestAutoConfigResolution:
    """Test auto resolution of config names based on instance type."""

    def create_mock_configs(self, scope):
        """Create mock configs for testing with different supported instance types."""
        # Mock the config object structure
        config1 = Mock()
        config1.config_name = "config1"
        config1.resolved_config = {
            "supported_inference_instance_types": ["ml.g5.xlarge", "ml.g5.2xlarge"]
            if scope == JumpStartScriptScope.INFERENCE
            else [],
            "supported_training_instance_types": ["ml.g5.xlarge", "ml.g5.2xlarge"]
            if scope == JumpStartScriptScope.TRAINING
            else [],
        }
        
        config2 = Mock()
        config2.config_name = "config2"
        config2.resolved_config = {
            "supported_inference_instance_types": ["ml.p4d.24xlarge", "ml.p5.48xlarge"]
            if scope == JumpStartScriptScope.INFERENCE
            else [],
            "supported_training_instance_types": ["ml.p4d.24xlarge", "ml.p5.48xlarge"]
            if scope == JumpStartScriptScope.TRAINING
            else [],
        }
        
        # Config with no instance type restrictions
        config3 = Mock()
        config3.config_name = "config3"
        config3.resolved_config = {
            "supported_inference_instance_types": []
            if scope == JumpStartScriptScope.INFERENCE
            else [],
            "supported_training_instance_types": []
            if scope == JumpStartScriptScope.TRAINING
            else [],
        }

        # Mock config rankings
        ranking = Mock()
        ranking.rankings = ["config1", "config2", "config3"]

        # Mock the metadata configs container
        configs = Mock()
        configs.scope = scope
        configs.configs = {
            "config1": config1,
            "config2": config2,
            "config3": config3,
        }
        configs.config_rankings = {JumpStartConfigRankingName.DEFAULT: ranking}

        # Import the actual get_top_config_from_ranking method so we can test it
        from sagemaker.jumpstart.types import JumpStartMetadataConfigs
        configs.get_top_config_from_ranking = JumpStartMetadataConfigs.get_top_config_from_ranking.__get__(configs)

        return configs

    def test_get_top_config_from_ranking_with_matching_instance_type(self):
        """Test that get_top_config_from_ranking returns config that supports the instance type."""
        configs = self.create_mock_configs(JumpStartScriptScope.INFERENCE)
        
        # Test with instance type that matches config1
        result = configs.get_top_config_from_ranking(instance_type="ml.g5.xlarge")
        assert result is not None
        assert result.config_name == "config1"
        
        # Test with instance type that matches config2
        result = configs.get_top_config_from_ranking(instance_type="ml.p4d.24xlarge")
        assert result is not None
        assert result.config_name == "config2"

    def test_get_top_config_from_ranking_with_no_matching_instance_type(self):
        """Test behavior when no config supports the requested instance type."""
        configs = self.create_mock_configs(JumpStartScriptScope.INFERENCE)
        
        # Test with instance type that doesn't match any config
        result = configs.get_top_config_from_ranking(instance_type="ml.m5.xlarge")
        assert result is not None
        assert result.config_name == "config3"  # Should fall back to config with no restrictions

    def test_get_top_config_from_ranking_without_instance_type(self):
        """Test that get_top_config_from_ranking returns first ranked config when no instance type specified."""
        configs = self.create_mock_configs(JumpStartScriptScope.INFERENCE)
        
        result = configs.get_top_config_from_ranking()
        assert result is not None
        assert result.config_name == "config1"  # First in ranking

    def test_get_top_config_from_ranking_training_scope(self):
        """Test get_top_config_from_ranking with training scope."""
        configs = self.create_mock_configs(JumpStartScriptScope.TRAINING)
        
        # Test with training instance type
        result = configs.get_top_config_from_ranking(instance_type="ml.g5.xlarge")
        assert result is not None
        assert result.config_name == "config1"

    def test_get_top_config_from_ranking_with_object_resolved_config(self):
        """Test get_top_config_from_ranking when resolved_config is an object (not dict)."""
        # Create a mock object with getattr support
        mock_resolved_config = Mock()
        mock_resolved_config.supported_inference_instance_types = ["ml.g5.xlarge"]
        
        config = Mock()
        config.config_name = "test_config"
        config.resolved_config = mock_resolved_config
        
        ranking = Mock()
        ranking.rankings = ["test_config"]
        
        configs = Mock()
        configs.scope = JumpStartScriptScope.INFERENCE
        configs.configs = {"test_config": config}
        configs.config_rankings = {JumpStartConfigRankingName.DEFAULT: ranking}
        
        # Import the actual method
        from sagemaker.jumpstart.types import JumpStartMetadataConfigs
        configs.get_top_config_from_ranking = JumpStartMetadataConfigs.get_top_config_from_ranking.__get__(configs)
        
        result = configs.get_top_config_from_ranking(instance_type="ml.g5.xlarge")
        assert result is not None
        assert result.config_name == "test_config"

    def test_get_top_config_from_ranking_empty_supported_instance_types(self):
        """Test behavior when config has empty supported_instance_types list."""
        config = Mock()
        config.config_name = "empty_config"
        config.resolved_config = {
            "supported_inference_instance_types": [],
        }
        
        ranking = Mock()
        ranking.rankings = ["empty_config"]
        
        configs = Mock()
        configs.scope = JumpStartScriptScope.INFERENCE
        configs.configs = {"empty_config": config}
        configs.config_rankings = {JumpStartConfigRankingName.DEFAULT: ranking}
        
        # Import the actual method
        from sagemaker.jumpstart.types import JumpStartMetadataConfigs
        configs.get_top_config_from_ranking = JumpStartMetadataConfigs.get_top_config_from_ranking.__get__(configs)
        
        # Should return config even with empty list (no restrictions)
        result = configs.get_top_config_from_ranking(instance_type="ml.g5.xlarge")
        assert result is not None
        assert result.config_name == "empty_config"

    def test_instance_type_parameter_signature(self):
        """Test that get_top_ranked_config_name function accepts instance_type parameter."""
        # Import and inspect the function signature
        import inspect
        from typing import Optional
        sig = inspect.signature(utils.get_top_ranked_config_name)
        
        # Verify that instance_type parameter exists in the signature
        assert "instance_type" in sig.parameters
        
        # Verify it's an optional parameter with None default
        instance_type_param = sig.parameters["instance_type"]
        assert instance_type_param.default is None
        assert instance_type_param.annotation == Optional[str]

    def test_get_top_config_from_ranking_preserves_existing_config_name(self):
        """Test that existing config_name is preserved when already specified."""
        mock_get_config = Mock(return_value="auto_selected")
        
        with patch("sagemaker.jumpstart.utils.get_top_ranked_config_name", mock_get_config):
            kwargs = JumpStartEstimatorInitKwargs(
                model_id="test-model",
                instance_type="ml.g5.xlarge",
                config_name="user_specified_config",
            )
            
            result = _add_config_name_to_kwargs(kwargs)
            
            # Should not call get_top_ranked_config_name when config_name already exists
            mock_get_config.assert_not_called()
            assert result.config_name == "user_specified_config"

    def test_config_ranking_respects_priority_with_instance_type_filter(self):
        """Test that config ranking priority is respected when filtering by instance type."""
        # Create configs where config2 is ranked higher but config1 matches instance type
        config1 = Mock()
        config1.config_name = "config1"
        config1.resolved_config = {"supported_inference_instance_types": ["ml.g5.xlarge"]}
        
        config2 = Mock()
        config2.config_name = "config2"
        config2.resolved_config = {"supported_inference_instance_types": ["ml.p4d.24xlarge"]}
        
        # Rank config2 higher than config1
        ranking = Mock()
        ranking.rankings = ["config2", "config1"]
        
        configs = Mock()
        configs.scope = JumpStartScriptScope.INFERENCE
        configs.configs = {"config1": config1, "config2": config2}
        configs.config_rankings = {JumpStartConfigRankingName.DEFAULT: ranking}
        
        # Import the actual method
        from sagemaker.jumpstart.types import JumpStartMetadataConfigs
        configs.get_top_config_from_ranking = JumpStartMetadataConfigs.get_top_config_from_ranking.__get__(configs)
        
        # Even though config2 is ranked higher, config1 should be returned because it matches instance type
        result = configs.get_top_config_from_ranking(instance_type="ml.g5.xlarge")
        assert result is not None
        assert result.config_name == "config1"