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
from sagemaker.core.enums import EndpointType
from sagemaker.serve.model_builder_utils import _ModelBuilderUtils


class ConcreteUtils(_ModelBuilderUtils):
    """Concrete class to test mixin methods.

    _ModelBuilderUtils is a mixin that does not define __init__,
    so this can be instantiated without arguments.
    """
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
        """Extra keys in the input dict are ignored.

        The resolver only extracts 'enable_caching' from the dict, so extra keys
        do not cause Pydantic validation errors even if the model forbids extras.
        We verify the result has enable_caching=True and does not expose extra_key.
        """
        result = utils._resolve_data_cache_config(
            {"enable_caching": True, "extra_key": "ignored"}
        )
        assert isinstance(result, InferenceComponentDataCacheConfig)
        assert result.enable_caching is True
        # Verify extra_key is not present on the result object
        assert not hasattr(result, "extra_key") or getattr(result, "extra_key", None) is None

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
# Tests for _apply_optional_ic_params helper
# ============================================================

class TestApplyOptionalIcParams:
    """Tests for the static helper that wires optional IC params into a spec dict."""

    def test_no_params_no_mutation(self):
        from sagemaker.serve.model_builder import ModelBuilder
        spec = {"ModelName": "m"}
        ModelBuilder._apply_optional_ic_params(spec)
        assert "DataCacheConfig" not in spec
        assert "BaseInferenceComponentName" not in spec
        assert "Container" not in spec

    def test_data_cache_config_dict(self):
        from sagemaker.serve.model_builder import ModelBuilder
        spec = {"ModelName": "m"}
        ModelBuilder._apply_optional_ic_params(
            spec, data_cache_config={"enable_caching": True}
        )
        assert spec["DataCacheConfig"] == {"EnableCaching": True}

    def test_data_cache_config_typed(self):
        from sagemaker.serve.model_builder import ModelBuilder
        spec = {"ModelName": "m"}
        cfg = InferenceComponentDataCacheConfig(enable_caching=False)
        ModelBuilder._apply_optional_ic_params(spec, data_cache_config=cfg)
        assert spec["DataCacheConfig"] == {"EnableCaching": False}

    def test_base_inference_component_name(self):
        from sagemaker.serve.model_builder import ModelBuilder
        spec = {"ModelName": "m"}
        ModelBuilder._apply_optional_ic_params(
            spec, base_inference_component_name="base-ic"
        )
        assert spec["BaseInferenceComponentName"] == "base-ic"

    def test_container_dict(self):
        from sagemaker.serve.model_builder import ModelBuilder
        spec = {"ModelName": "m"}
        ModelBuilder._apply_optional_ic_params(
            spec,
            container={
                "image": "img:latest",
                "artifact_url": "s3://b/a",
                "environment": {"K": "V"},
            },
        )
        assert spec["Container"] == {
            "Image": "img:latest",
            "ArtifactUrl": "s3://b/a",
            "Environment": {"K": "V"},
        }

    def test_container_typed(self):
        from sagemaker.serve.model_builder import ModelBuilder
        spec = {"ModelName": "m"}
        c = InferenceComponentContainerSpecification(image="img")
        ModelBuilder._apply_optional_ic_params(spec, container=c)
        assert spec["Container"] == {"Image": "img"}

    def test_all_params_together(self):
        from sagemaker.serve.model_builder import ModelBuilder
        spec = {"ModelName": "m"}
        ModelBuilder._apply_optional_ic_params(
            spec,
            data_cache_config={"enable_caching": True},
            base_inference_component_name="base",
            container={"image": "img"},
        )
        assert spec["DataCacheConfig"] == {"EnableCaching": True}
        assert spec["BaseInferenceComponentName"] == "base"
        assert spec["Container"] == {"Image": "img"}


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
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            wait=False,
        )

        # Verify create_inference_component was called with variant_name="AllTraffic"
        mb.sagemaker_session.create_inference_component.assert_called_once()
        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        assert call_kwargs.kwargs["variant_name"] == "AllTraffic"

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
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            variant_name="MyVariant",
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        assert call_kwargs.kwargs["variant_name"] == "MyVariant"

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
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            data_cache_config={"enable_caching": True},
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs.kwargs["specification"]
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
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            base_inference_component_name="base-ic-name",
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs.kwargs["specification"]
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
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
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
        spec = call_kwargs.kwargs["specification"]
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
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs.kwargs["specification"]
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
            endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
            resources=resources,
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            data_cache_config=config,
            wait=False,
        )

        call_kwargs = mb.sagemaker_session.create_inference_component.call_args
        spec = call_kwargs.kwargs["specification"]
        assert spec["DataCacheConfig"]["EnableCaching"] is True

    @patch("sagemaker.serve.model_builder.Endpoint")
    def test_variant_name_passed_to_production_variant_on_new_endpoint(self, mock_endpoint_cls):
        """When creating a new endpoint, variant_name is passed to production_variant."""
        mb = self._make_model_builder()
        mock_endpoint_cls.get.return_value = MagicMock()
        # Simulate endpoint does NOT exist yet
        mb.sagemaker_session.endpoint_in_service_or_not.return_value = False

        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        with patch("sagemaker.serve.model_builder.session_helper.production_variant") as mock_pv:
            mock_pv.return_value = {"VariantName": "CustomVariant"}
            mb._deploy_core_endpoint(
                endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
                resources=resources,
                instance_type="ml.g5.2xlarge",
                initial_instance_count=1,
                variant_name="CustomVariant",
                wait=False,
            )

            # Verify production_variant was called with variant_name="CustomVariant"
            mock_pv.assert_called_once()
            pv_call = mock_pv.call_args
            # variant_name may be in kwargs or positional args
            variant = pv_call.kwargs.get("variant_name") or (
                pv_call.args[3] if len(pv_call.args) > 3 else None
            )
            assert variant == "CustomVariant"


# ============================================================
# Tests for _update_inference_component wiring
# ============================================================

class TestUpdateInferenceComponentWiring:
    """Tests that _update_inference_component correctly wires optional IC params."""

    def _make_model_builder(self):
        from sagemaker.serve.model_builder import ModelBuilder

        mb = object.__new__(ModelBuilder)
        mb.model_name = "test-model"
        mb.sagemaker_session = MagicMock()
        return mb

    def test_update_ic_with_data_cache_config(self):
        mb = self._make_model_builder()
        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._update_inference_component(
            "my-ic", resources, data_cache_config={"enable_caching": True}
        )

        call_kwargs = mb.sagemaker_session.update_inference_component.call_args
        spec = call_kwargs.kwargs["specification"]
        assert spec["DataCacheConfig"] == {"EnableCaching": True}

    def test_update_ic_with_container(self):
        mb = self._make_model_builder()
        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._update_inference_component(
            "my-ic", resources, container={"image": "img:v1"}
        )

        call_kwargs = mb.sagemaker_session.update_inference_component.call_args
        spec = call_kwargs.kwargs["specification"]
        assert spec["Container"] == {"Image": "img:v1"}

    def test_update_ic_with_base_inference_component_name(self):
        mb = self._make_model_builder()
        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._update_inference_component(
            "my-ic", resources, base_inference_component_name="base-ic"
        )

        call_kwargs = mb.sagemaker_session.update_inference_component.call_args
        spec = call_kwargs.kwargs["specification"]
        assert spec["BaseInferenceComponentName"] == "base-ic"

    def test_update_ic_no_optional_params(self):
        mb = self._make_model_builder()
        from sagemaker.core.inference_config import ResourceRequirements
        resources = ResourceRequirements(
            requests={"memory": 8192, "num_accelerators": 1, "num_cpus": 2, "copies": 1}
        )

        mb._update_inference_component("my-ic", resources)

        call_kwargs = mb.sagemaker_session.update_inference_component.call_args
        spec = call_kwargs.kwargs["specification"]
        assert "DataCacheConfig" not in spec
        assert "BaseInferenceComponentName" not in spec
        assert "Container" not in spec


# ============================================================
# Tests for deploy() parameter forwarding
# ============================================================

class TestDeployParameterForwarding:
    """Tests that deploy() correctly forwards new IC params into kwargs."""

    def test_deploy_forwards_variant_name_to_kwargs(self):
        """deploy() should set kwargs['variant_name'] to the provided value."""
        from sagemaker.serve.model_builder import ModelBuilder

        mb = object.__new__(ModelBuilder)
        mb.built_model = MagicMock()
        mb._deployed = False
        mb._is_sharded_model = False
        mb.model_name = "test"
        mb.instance_type = "ml.m5.large"
        mb.endpoint_name = None
        mb.mode = None
        mb.model_server = None

        # Mock _is_model_customization to return False
        mb._is_model_customization = MagicMock(return_value=False)
        # Mock _deploy to capture kwargs
        captured = {}

        def fake_deploy(**kw):
            captured.update(kw)
            return MagicMock()

        mb._deploy = fake_deploy

        mb.deploy(
            endpoint_name="ep",
            instance_type="ml.m5.large",
            initial_instance_count=1,
            variant_name="MyVariant",
            data_cache_config={"enable_caching": True},
            base_inference_component_name="base-ic",
            container={"image": "img"},
        )

        assert captured["variant_name"] == "MyVariant"
        assert captured["data_cache_config"] == {"enable_caching": True}
        assert captured["base_inference_component_name"] == "base-ic"
        assert captured["container"] == {"image": "img"}

    def test_deploy_does_not_set_variant_name_when_not_provided(self):
        """deploy() should NOT set variant_name in kwargs when not provided.

        This allows downstream methods to use their own defaults:
        - _deploy_core_endpoint defaults to 'AllTraffic'
        - _deploy_model_customization defaults to endpoint_name
        """
        from sagemaker.serve.model_builder import ModelBuilder

        mb = object.__new__(ModelBuilder)
        mb.built_model = MagicMock()
        mb._deployed = False
        mb._is_sharded_model = False
        mb.model_name = "test"
        mb.instance_type = "ml.m5.large"
        mb.endpoint_name = None
        mb.mode = None
        mb.model_server = None
        mb._is_model_customization = MagicMock(return_value=False)

        captured = {}

        def fake_deploy(**kw):
            captured.update(kw)
            return MagicMock()

        mb._deploy = fake_deploy

        mb.deploy(
            endpoint_name="ep",
            instance_type="ml.m5.large",
            initial_instance_count=1,
        )

        # variant_name should NOT be in kwargs when not explicitly provided
        assert "variant_name" not in captured
        # Optional params should not be in kwargs when not provided
        assert "data_cache_config" not in captured
        assert "base_inference_component_name" not in captured
        assert "container" not in captured

    def test_deploy_forwards_variant_name_none_is_not_forwarded(self):
        """deploy(variant_name=None) should NOT forward variant_name.

        None is the default, so it should behave the same as not providing it.
        """
        from sagemaker.serve.model_builder import ModelBuilder

        mb = object.__new__(ModelBuilder)
        mb.built_model = MagicMock()
        mb._deployed = False
        mb._is_sharded_model = False
        mb.model_name = "test"
        mb.instance_type = "ml.m5.large"
        mb.endpoint_name = None
        mb.mode = None
        mb.model_server = None
        mb._is_model_customization = MagicMock(return_value=False)

        captured = {}

        def fake_deploy(**kw):
            captured.update(kw)
            return MagicMock()

        mb._deploy = fake_deploy

        mb.deploy(
            endpoint_name="ep",
            instance_type="ml.m5.large",
            initial_instance_count=1,
            variant_name=None,
        )

        # variant_name=None should not be forwarded
        assert "variant_name" not in captured
