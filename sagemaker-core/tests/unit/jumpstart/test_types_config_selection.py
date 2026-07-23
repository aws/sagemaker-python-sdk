"""
Unit tests for JumpStartMetadataConfigs.get_top_config_from_ranking()

These tests verify that config selection correctly filters by instance_type,
handling the case where resolved_config is a dict (from deep_override_dict).

This addresses a bug where getattr() was incorrectly used on dict objects
instead of dict key access, causing instance_type filtering to fail.
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


class TestGetTopConfigFromRanking:
    """Tests for get_top_config_from_ranking method."""

    @pytest.fixture
    def mock_gpu_config(self):
        """Create a mock GPU config with dict resolved_config."""
        config = MagicMock()
        config.config_name = "gpu-lmi-tgi"
        # resolved_config is a dict (as returned by deep_override_dict)
        config.resolved_config = {
            "supported_inference_instance_types": [
                "ml.g5.xlarge",
                "ml.g5.2xlarge",
                "ml.g5.4xlarge",
                "ml.g5.12xlarge",
                "ml.p4d.24xlarge",
            ],
            "model_id": "meta-llama/Llama-2-7b",
        }
        return config

    @pytest.fixture
    def mock_neuron_config(self):
        """Create a mock Neuron config with dict resolved_config."""
        config = MagicMock()
        config.config_name = "neuron-inference"
        # resolved_config is a dict (as returned by deep_override_dict)
        config.resolved_config = {
            "supported_inference_instance_types": [
                "ml.inf2.xlarge",
                "ml.inf2.8xlarge",
                "ml.inf2.24xlarge",
                "ml.inf2.48xlarge",
            ],
            "model_id": "meta-llama/Llama-2-7b",
        }
        return config

    @pytest.fixture
    def mock_ranking(self):
        """Create a mock ranking with GPU first, then Neuron."""
        ranking = MagicMock()
        ranking.rankings = ["gpu-lmi-tgi", "neuron-inference"]
        return ranking

    def test_no_instance_type_returns_highest_ranked(
        self, mock_gpu_config, mock_neuron_config, mock_ranking
    ):
        """When no instance_type specified, return highest ranked config."""
        from sagemaker.core.jumpstart.types import JumpStartMetadataConfigs
        from sagemaker.core.jumpstart.enums import JumpStartScriptScope

        configs = JumpStartMetadataConfigs(
            configs={
                "gpu-lmi-tgi": mock_gpu_config,
                "neuron-inference": mock_neuron_config,
            },
            config_rankings={"default": mock_ranking},
            scope=JumpStartScriptScope.INFERENCE,
        )

        result = configs.get_top_config_from_ranking(instance_type=None)
        assert result is not None
        assert result.config_name == "gpu-lmi-tgi"

    def test_gpu_instance_returns_gpu_config(
        self, mock_gpu_config, mock_neuron_config, mock_ranking
    ):
        """When GPU instance specified, return GPU config."""
        from sagemaker.core.jumpstart.types import JumpStartMetadataConfigs
        from sagemaker.core.jumpstart.enums import JumpStartScriptScope

        configs = JumpStartMetadataConfigs(
            configs={
                "gpu-lmi-tgi": mock_gpu_config,
                "neuron-inference": mock_neuron_config,
            },
            config_rankings={"default": mock_ranking},
            scope=JumpStartScriptScope.INFERENCE,
        )

        result = configs.get_top_config_from_ranking(instance_type="ml.g5.xlarge")
        assert result is not None
        assert result.config_name == "gpu-lmi-tgi"

    def test_inferentia_instance_returns_neuron_config(
        self, mock_gpu_config, mock_neuron_config, mock_ranking
    ):
        """
        When Inferentia instance specified, return Neuron config.

        This is the critical test case that was failing before the fix.
        The bug caused GPU config to be returned even for Inferentia instances
        because getattr() was used on a dict instead of dict key access.
        """
        from sagemaker.core.jumpstart.types import JumpStartMetadataConfigs
        from sagemaker.core.jumpstart.enums import JumpStartScriptScope

        configs = JumpStartMetadataConfigs(
            configs={
                "gpu-lmi-tgi": mock_gpu_config,
                "neuron-inference": mock_neuron_config,
            },
            config_rankings={"default": mock_ranking},
            scope=JumpStartScriptScope.INFERENCE,
        )

        result = configs.get_top_config_from_ranking(instance_type="ml.inf2.24xlarge")
        assert result is not None
        assert result.config_name == "neuron-inference"

    def test_unsupported_instance_returns_none(
        self, mock_gpu_config, mock_neuron_config, mock_ranking
    ):
        """When unsupported instance specified, return None."""
        from sagemaker.core.jumpstart.types import JumpStartMetadataConfigs
        from sagemaker.core.jumpstart.enums import JumpStartScriptScope

        configs = JumpStartMetadataConfigs(
            configs={
                "gpu-lmi-tgi": mock_gpu_config,
                "neuron-inference": mock_neuron_config,
            },
            config_rankings={"default": mock_ranking},
            scope=JumpStartScriptScope.INFERENCE,
        )

        result = configs.get_top_config_from_ranking(instance_type="ml.trn1.32xlarge")
        assert result is None

    def test_training_scope_uses_training_instance_types(self):
        """Verify training scope uses supported_training_instance_types."""
        from sagemaker.core.jumpstart.types import JumpStartMetadataConfigs
        from sagemaker.core.jumpstart.enums import JumpStartScriptScope

        gpu_config = MagicMock()
        gpu_config.config_name = "gpu-training"
        gpu_config.resolved_config = {
            "supported_training_instance_types": [
                "ml.p4d.24xlarge",
                "ml.p5.48xlarge",
            ],
        }

        trn_config = MagicMock()
        trn_config.config_name = "trainium-training"
        trn_config.resolved_config = {
            "supported_training_instance_types": [
                "ml.trn1.32xlarge",
                "ml.trn1n.32xlarge",
            ],
        }

        ranking = MagicMock()
        ranking.rankings = ["gpu-training", "trainium-training"]

        configs = JumpStartMetadataConfigs(
            configs={
                "gpu-training": gpu_config,
                "trainium-training": trn_config,
            },
            config_rankings={"default": ranking},
            scope=JumpStartScriptScope.TRAINING,
        )

        # Trainium instance should select trainium config
        result = configs.get_top_config_from_ranking(instance_type="ml.trn1.32xlarge")
        assert result is not None
        assert result.config_name == "trainium-training"

    def test_resolved_config_as_object_still_works(self):
        """
        Verify that if resolved_config is an object (not dict), getattr still works.

        This ensures backwards compatibility with any code paths where
        resolved_config might be an object with attributes.
        """
        from sagemaker.core.jumpstart.types import JumpStartMetadataConfigs
        from sagemaker.core.jumpstart.enums import JumpStartScriptScope

        # Create a config where resolved_config is an object, not a dict
        class ResolvedConfigObject:
            supported_inference_instance_types = ["ml.g5.xlarge", "ml.g5.2xlarge"]

        config = MagicMock()
        config.config_name = "object-config"
        config.resolved_config = ResolvedConfigObject()

        ranking = MagicMock()
        ranking.rankings = ["object-config"]

        configs = JumpStartMetadataConfigs(
            configs={"object-config": config},
            config_rankings={"default": ranking},
            scope=JumpStartScriptScope.INFERENCE,
        )

        result = configs.get_top_config_from_ranking(instance_type="ml.g5.xlarge")
        assert result is not None
        assert result.config_name == "object-config"


class TestResolvedConfigType:
    """Tests verifying that resolved_config is correctly identified as dict."""

    def test_deep_override_dict_returns_dict(self):
        """Verify deep_override_dict returns a plain dict."""
        from sagemaker.core.common_utils import deep_override_dict

        base = {"field1": "value1", "nested": {"a": 1}}
        override = {"field2": "value2", "nested": {"b": 2}}

        result = deep_override_dict(base, override)

        assert isinstance(result, dict)
        assert "field1" in result
        assert "field2" in result

    def test_getattr_fails_on_dict(self):
        """Verify that getattr fails on dict for non-existent attributes."""
        d = {"supported_inference_instance_types": ["ml.g5.xlarge"]}

        with pytest.raises(AttributeError):
            getattr(d, "supported_inference_instance_types")

    def test_dict_get_works(self):
        """Verify that dict.get() works correctly."""
        d = {"supported_inference_instance_types": ["ml.g5.xlarge"]}

        result = d.get("supported_inference_instance_types", [])
        assert result == ["ml.g5.xlarge"]

        # Non-existent key returns default
        result = d.get("nonexistent", [])
        assert result == []
