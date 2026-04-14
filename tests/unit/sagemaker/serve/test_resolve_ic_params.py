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
"""Unit tests for IC parameter resolvers and wiring logic."""
from __future__ import absolute_import

import pytest
from unittest.mock import MagicMock, patch, ANY

from sagemaker.core.shapes import (
    InferenceComponentDataCacheConfig,
    InferenceComponentContainerSpecification,
)
from sagemaker.serve.model_builder_utils import _ModelBuilderUtils


class ConcreteUtils(_ModelBuilderUtils):
    """Concrete class to test mixin methods."""
    pass


@pytest.fixture
def utils():
    return ConcreteUtils()


# ============================================================
# Tests for _resolve_data_cache_config
# ============================================================

class TestResolveDataCacheConfig:
    def test_none_returns_none(self, utils):
        assert utils._resolve_data_cache_config(None) is None

    def test_already_typed_passthrough(self, utils):
        config = InferenceComponentDataCacheConfig(enable_caching=True)
        result = utils._resolve_data_cache_config(config)
        assert result is config
        assert result.enable_caching is True

    def test_dict_with_enable_caching_true(self, utils):
        result = utils._resolve_data_cache_config({"enable_caching": True})
        assert isinstance(result, InferenceComponentDataCacheConfig)
        assert result.enable_caching is True

    def test_dict_with_enable_caching_false(self, utils):
        result = utils._resolve_data_cache_config({"enable_caching": False})
        assert isinstance(result, InferenceComponentDataCacheConfig)
        assert result.enable_caching is False

    def test_dict_missing_enable_caching_raises(self, utils):
        with pytest.raises(ValueError, match="must contain the required 'enable_caching' key"):
            utils._resolve_data_cache_config({})

    def test_dict_with_extra_keys_still_works(self, utils):
        """Extra keys in the input dict are ignored (not forwarded to the Pydantic constructor).

        The resolver only extracts 'enable_caching' from the dict, so extra keys
        do not cause Pydantic validation errors even if the model forbids extras.
        """
        result = utils._resolve_data_cache_config(
            {"enable_caching": True, "extra_key": "ignored"}
        )
        assert isinstance(result, InferenceComponentDataCacheConfig)
        assert result.enable_caching is True

    def test_invalid_type_raises(self, utils):
        with pytest.raises(ValueError, match="data_cache_config must be a dict"):
            utils._resolve_data_cache_config("invalid")

    def test_invalid_type_int_raises(self, utils):
        with pytest.raises(ValueError, match="data_cache_config must be a dict"):
            utils._resolve_data_cache_config(42)

    def test_invalid_type_list_raises(self, utils):
        with pytest.raises(ValueError, match="data_cache_config must be a dict"):
            utils._resolve_data_cache_config([True])


# ============================================================
# Tests for _resolve_container_spec
# ============================================================

class TestResolveContainerSpec:
    def test_none_returns_none(self, utils):
        assert utils._resolve_container_spec(None) is None

    def test_already_typed_passthrough(self, utils):
        spec = InferenceComponentContainerSpecification(
            image="my-image:latest",
            artifact_url="s3://bucket/artifact",
            environment={"KEY": "VALUE"},
        )
        result = utils._resolve_container_spec(spec)
        assert result is spec

    def test_dict_full(self, utils):
        result = utils._resolve_container_spec({
            "image": "my-image:latest",
            "artifact_url": "s3://bucket/artifact",
            "environment": {"KEY": "VALUE"},
        })
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.image == "my-image:latest"
        assert result.artifact_url == "s3://bucket/artifact"
        assert result.environment == {"KEY": "VALUE"}

    def test_dict_image_only(self, utils):
        result = utils._resolve_container_spec({"image": "my-image:latest"})
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.image == "my-image:latest"

    def test_dict_artifact_url_only(self, utils):
        result = utils._resolve_container_spec({"artifact_url": "s3://bucket/model.tar.gz"})
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.artifact_url == "s3://bucket/model.tar.gz"

    def test_dict_environment_only(self, utils):
        result = utils._resolve_container_spec({"environment": {"A": "B"}})
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.environment == {"A": "B"}

    def test_dict_empty(self, utils):
        """Empty dict creates a spec with no fields set."""
        result = utils._resolve_container_spec({})
        assert isinstance(result, InferenceComponentContainerSpecification)

    def test_dict_with_extra_keys(self, utils):
        """Extra keys are filtered out before passing to the Pydantic constructor.

        This ensures compatibility even if InferenceComponentContainerSpecification
        has extra='forbid' in its Pydantic model config.
        """
        result = utils._resolve_container_spec({
            "image": "img",
            "unknown_key": "ignored",
        })
        assert isinstance(result, InferenceComponentContainerSpecification)
        assert result.image == "img"

    def test_invalid_type_raises(self, utils):
        with pytest.raises(ValueError, match="container must be a dict"):
            utils._resolve_container_spec("invalid")

    def test_invalid_type_int_raises(self, utils):
        with pytest.raises(ValueError, match="container must be a dict"):
            utils._resolve_container_spec(123)

    def test_invalid_type_list_raises(self, utils):
        with pytest.raises(ValueError, match="container must be a dict"):
            utils._resolve_container_spec([{"image": "img"}])


# ============================================================
# Tests for core wiring logic in _deploy_core_endpoint
# ============================================================

class TestDeployCoreEndpointWiring:
    """Tests that new IC parameters are correctly wired through _deploy_core_endpoint."""

    def _make_model_builder(self):
        """Create a minimally-configured ModelBuilder for testing _deploy_core_endpoint."""
        from sagemaker.serve.model_builder import ModelBuilder

        mb = object.__new__(ModelBuilder)
        # Set minimum required attributes
        mb.model_name = "test-model"
        mb.endpoint_name = None
        mb.inference_component_name = None
        mb.instance_type = "ml.g5.2xlarge"
        mb.instance_count = 1
        mb.accelerator_type = None
        mb._tags = None
        mb.kms_key = None
        mb.async_inference_config = None
        mb.serverless_inference_config = None
        mb.model_data_download_timeout = None
        mb.resource_requirements = None
        mb.container_startup_health_check_timeout = None
        mb.inference_ami_version = None
        mb._is_sharded_model = False
        mb._enable_network_isolation = False
        mb.role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"
        mb.vpc_config = None
        mb.inference_recommender_job_results = None
        mb.model_server = None
        mb.mode = None
        mb.region = "us-east-1"

        # Mock built_model
        mb.built_model = MagicMock()
        mb.built_model.model_name = "test-model"

        # Mock sagemaker_session
        mb.sagemaker_session = MagicMock()
        mb.sagemaker_session.endpoint_in_service_or_not.return_value = True
        mb.sagemaker_session.boto_session = MagicMock()
        mb.sagemaker_session.boto_region_name = "us-east-1"

        return mb

    @patch("sagemaker.serve.model_builder.Endpoint")
    def test_variant_name_defaults_to_all_traffic(self, mock_endpoint_cls):
        """When variant_name is not provided, it defaults to 'AllTraffic'."""
        mb = self._make_model_builder()
        mock_endpoint_cls.get.return_value = MagicMock()

        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._deploy_core_endpoint(
            endpoint_type="INFERENCE_COMPONENT_BASED",
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            wait=False,
        )

        # Verify create_inference_component was called with variant_name="AllTraffic"
        mb.sagemaker_session.create_inference_component.assert_called_once()
        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        assert call_kwargs[1]["variant_name"] == "AllTraffic" or \
            (len(call_kwargs[0]) > 2 and call_kwargs[0][2] == "AllTraffic")

    @patch("sagemaker.serve.model_builder.Endpoint")
    def test_variant_name_custom(self, mock_endpoint_cls):
        """When variant_name is provided, it is used instead of 'AllTraffic'."""
        mb = self._make_model_builder()
        mock_endpoint_cls.get.return_value = MagicMock()

        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._deploy_core_endpoint(
            endpoint_type="INFERENCE_COMPONENT_BASED",
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            variant_name="MyVariant",
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        assert call_kwargs[1]["variant_name"] == "MyVariant" or \
            (len(call_kwargs[0]) > 2 and call_kwargs[0][2] == "MyVariant")

    @patch("sagemaker.serve.model_builder.Endpoint")
    def test_data_cache_config_wired_into_spec(self, mock_endpoint_cls):
        """data_cache_config dict is resolved and added to inference_component_spec."""
        mb = self._make_model_builder()
        mock_endpoint_cls.get.return_value = MagicMock()

        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._deploy_core_endpoint(
            endpoint_type="INFERENCE_COMPONENT_BASED",
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            data_cache_config={"enable_caching": True},
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs[1]["specification"]
        assert "DataCacheConfig" in spec
        assert spec["DataCacheConfig"]["EnableCaching"] is True

    @patch("sagemaker.serve.model_builder.Endpoint")
    def test_base_inference_component_name_wired_into_spec(self, mock_endpoint_cls):
        """base_inference_component_name is added to inference_component_spec."""
        mb = self._make_model_builder()
        mock_endpoint_cls.get.return_value = MagicMock()

        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._deploy_core_endpoint(
            endpoint_type="INFERENCE_COMPONENT_BASED",
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            base_inference_component_name="base-ic-name",
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs[1]["specification"]
        assert spec["BaseInferenceComponentName"] == "base-ic-name"

    @patch("sagemaker.serve.model_builder.Endpoint")
    def test_container_wired_into_spec(self, mock_endpoint_cls):
        """container dict is resolved and added to inference_component_spec."""
        mb = self._make_model_builder()
        mock_endpoint_cls.get.return_value = MagicMock()

        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._deploy_core_endpoint(
            endpoint_type="INFERENCE_COMPONENT_BASED",
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            container={
                "image": "my-image:latest",
                "artifact_url": "s3://bucket/artifact",
                "environment": {"KEY": "VALUE"},
            },
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs[1]["specification"]
        assert "Container" in spec
        assert spec["Container"]["Image"] == "my-image:latest"
        assert spec["Container"]["ArtifactUrl"] == "s3://bucket/artifact"
        assert spec["Container"]["Environment"] == {"KEY": "VALUE"}

    @patch("sagemaker.serve.model_builder.Endpoint")
    def test_no_optional_params_no_extra_keys_in_spec(self, mock_endpoint_cls):
        """When no optional IC params are provided, spec has no extra keys."""
        mb = self._make_model_builder()
        mock_endpoint_cls.get.return_value = MagicMock()

        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._deploy_core_endpoint(
            endpoint_type="INFERENCE_COMPONENT_BASED",
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs[1]["specification"]
        assert "DataCacheConfig" not in spec
        assert "BaseInferenceComponentName" not in spec
        assert "Container" not in spec

    @patch("sagemaker.serve.model_builder.Endpoint")
    def test_data_cache_config_typed_object_wired(self, mock_endpoint_cls):
        """InferenceComponentDataCacheConfig object is correctly wired."""
        mb = self._make_model_builder()
        mock_endpoint_cls.get.return_value = MagicMock()

        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        config = InferenceComponentDataCacheConfig(enable_caching=True)
        mb._deploy_core_endpoint(
            endpoint_type="INFERENCE_COMPONENT_BASED",
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            data_cache_config=config,
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs[1]["specification"]
        assert spec["DataCacheConfig"]["EnableCaching"] is True
