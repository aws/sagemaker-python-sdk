"""
Unit tests for ModelBuilder checkpoint-related changes.

Tests the is_checkpoint logic for:
- _resolve_model_artifact_uri returning hf_merged path
- build() setting s3_upload_path to hf_merged path for non-LORA
- _fetch_peft returning None when is_checkpoint is False
- Inference component using model_name instead of container spec
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import uuid

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer


class TestResolveModelArtifactUriCheckpoint(unittest.TestCase):
    """Test _resolve_model_artifact_uri with is_checkpoint logic."""

    def setUp(self):
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}

    def _create_builder(self):
        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )
        return builder

    def _make_model_package(self, s3_uri, is_checkpoint=None, base_model=None):
        container = Mock()
        container.model_data_source = Mock()
        container.model_data_source.s3_data_source = Mock()
        container.model_data_source.s3_data_source.s3_uri = s3_uri
        container.is_checkpoint = is_checkpoint
        container.base_model = base_model

        model_package = Mock()
        model_package.inference_specification = Mock()
        model_package.inference_specification.containers = [container]
        return model_package

    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    def test_returns_hf_merged_uri_when_is_checkpoint_false(
        self, mock_fetch_mp, mock_is_mc, mock_fetch_peft
    ):
        """When is_checkpoint is False, should return s3_uri + /checkpoints/hf_merged/."""
        mock_fetch_peft.return_value = None
        mock_is_mc.return_value = True
        s3_uri = "s3://bucket/training-output"
        mock_fetch_mp.return_value = self._make_model_package(s3_uri, is_checkpoint=False)

        builder = self._create_builder()
        result = builder._resolve_model_artifact_uri()

        self.assertEqual(result, "s3://bucket/training-output/checkpoints/hf_merged/")

    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    def test_returns_hf_merged_uri_strips_trailing_slash(
        self, mock_fetch_mp, mock_is_mc, mock_fetch_peft
    ):
        """When is_checkpoint is False with trailing slash in s3_uri, strips before appending."""
        mock_fetch_peft.return_value = None
        mock_is_mc.return_value = True
        s3_uri = "s3://bucket/training-output/"
        mock_fetch_mp.return_value = self._make_model_package(s3_uri, is_checkpoint=False)

        builder = self._create_builder()
        result = builder._resolve_model_artifact_uri()

        self.assertEqual(result, "s3://bucket/training-output/checkpoints/hf_merged/")

    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    def test_returns_s3_uri_when_is_checkpoint_true(
        self, mock_fetch_mp, mock_is_mc, mock_fetch_peft
    ):
        """When is_checkpoint is True, should return s3_uri directly."""
        mock_fetch_peft.return_value = None
        mock_is_mc.return_value = True
        s3_uri = "s3://bucket/training-output"
        mock_fetch_mp.return_value = self._make_model_package(s3_uri, is_checkpoint=True)

        builder = self._create_builder()
        result = builder._resolve_model_artifact_uri()

        self.assertEqual(result, s3_uri)

    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    def test_returns_s3_uri_when_is_checkpoint_none(
        self, mock_fetch_mp, mock_is_mc, mock_fetch_peft
    ):
        """When is_checkpoint is None (not set), should return s3_uri directly."""
        mock_fetch_peft.return_value = None
        mock_is_mc.return_value = True
        s3_uri = "s3://bucket/training-output"
        mock_fetch_mp.return_value = self._make_model_package(s3_uri, is_checkpoint=None)

        builder = self._create_builder()
        result = builder._resolve_model_artifact_uri()

        self.assertEqual(result, s3_uri)


class TestFetchPeftCheckpoint(unittest.TestCase):
    """Test _fetch_peft with is_checkpoint logic."""

    def setUp(self):
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}

    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    def test_returns_none_when_is_checkpoint_false(self, mock_fetch_mp):
        """When is_checkpoint is False, _fetch_peft should return None (not LORA)."""
        from sagemaker.core.resources import ModelPackage

        container = Mock()
        container.is_checkpoint = False
        container.base_model = Mock()
        container.base_model.recipe_name = "meta-llama/Llama-3-8b-lora"

        model_package = Mock()
        model_package.inference_specification = Mock()
        model_package.inference_specification.containers = [container]
        mock_fetch_mp.return_value = model_package

        mock_model = Mock(spec=ModelPackage)
        builder = ModelBuilder(
            model=mock_model,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        result = builder._fetch_peft()

        self.assertIsNone(result)

    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    def test_returns_lora_when_is_checkpoint_true_and_recipe_has_lora(self, mock_fetch_mp):
        """When is_checkpoint is True and recipe has 'lora', should return LORA."""
        from sagemaker.core.resources import ModelPackage

        container = Mock()
        container.is_checkpoint = True
        container.base_model = Mock()
        container.base_model.recipe_name = "meta-llama/Llama-3-8b-lora"

        model_package = Mock()
        model_package.inference_specification = Mock()
        model_package.inference_specification.containers = [container]
        mock_fetch_mp.return_value = model_package

        mock_model = Mock(spec=ModelPackage)
        builder = ModelBuilder(
            model=mock_model,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        result = builder._fetch_peft()

        self.assertEqual(result, "LORA")

    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    def test_returns_none_when_is_checkpoint_none_and_no_lora(self, mock_fetch_mp):
        """When is_checkpoint is None and recipe has no lora, should return None."""
        from sagemaker.core.resources import ModelPackage

        container = Mock()
        container.is_checkpoint = None
        container.base_model = Mock()
        container.base_model.recipe_name = "meta-llama/Llama-3-8b"

        model_package = Mock()
        model_package.inference_specification = Mock()
        model_package.inference_specification.containers = [container]
        mock_fetch_mp.return_value = model_package

        mock_model = Mock(spec=ModelPackage)
        builder = ModelBuilder(
            model=mock_model,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
        )

        result = builder._fetch_peft()

        self.assertIsNone(result)


class TestBuildNonLoraCheckpoint(unittest.TestCase):
    """Test build() sets s3_upload_path correctly for non-LORA with is_checkpoint."""

    def setUp(self):
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client

    def _make_model_package(self, s3_uri, is_checkpoint=None, recipe_name=""):
        container = Mock()
        container.model_data_source = Mock()
        container.model_data_source.s3_data_source = Mock()
        container.model_data_source.s3_data_source.s3_uri = s3_uri
        container.is_checkpoint = is_checkpoint
        container.base_model = None
        container.image = "test-image:latest"

        model_package = Mock()
        model_package.inference_specification = Mock()
        model_package.inference_specification.containers = [container]
        model_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        return model_package

    @patch("sagemaker.core.resources.Model.create")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization")
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_serve_setting")
    def test_build_sets_hf_merged_path_when_is_checkpoint_false(
        self, mock_get_serve, mock_is_mc, mock_fetch_peft, mock_fetch_mp,
        mock_is_nova, mock_model_create
    ):
        """build() should set s3_upload_path to hf_merged when is_checkpoint is False."""
        from sagemaker.core.resources import ModelPackage

        mock_is_mc.return_value = True
        mock_fetch_peft.return_value = None
        mock_is_nova.return_value = False
        s3_uri = "s3://bucket/training-output"
        mock_fetch_mp.return_value = self._make_model_package(s3_uri, is_checkpoint=False)

        mock_built_model = Mock()
        mock_built_model.model_name = "test-model"
        mock_model_create.return_value = mock_built_model

        mock_model = Mock(spec=ModelPackage)
        builder = ModelBuilder(
            model=mock_model,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
        )
        builder.env_vars = {"KEY": "value"}
        builder.model_name = None

        builder.build()

        self.assertEqual(
            builder.s3_upload_path, "s3://bucket/training-output/checkpoints/hf_merged/"
        )

    @patch("sagemaker.core.resources.Model.create")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization")
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_serve_setting")
    def test_build_sets_raw_s3_path_when_is_checkpoint_true(
        self, mock_get_serve, mock_is_mc, mock_fetch_peft, mock_fetch_mp,
        mock_is_nova, mock_model_create
    ):
        """build() should set s3_upload_path to raw s3_uri when is_checkpoint is True."""
        from sagemaker.core.resources import ModelPackage

        mock_is_mc.return_value = True
        mock_fetch_peft.return_value = None
        mock_is_nova.return_value = False
        s3_uri = "s3://bucket/training-output"
        mock_fetch_mp.return_value = self._make_model_package(s3_uri, is_checkpoint=True)

        mock_built_model = Mock()
        mock_built_model.model_name = "test-model"
        mock_model_create.return_value = mock_built_model

        mock_model = Mock(spec=ModelPackage)
        builder = ModelBuilder(
            model=mock_model,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
        )
        builder.env_vars = {"KEY": "value"}
        builder.model_name = None

        builder.build()

        self.assertEqual(builder.s3_upload_path, s3_uri)


class TestInferenceComponentUsesModelName(unittest.TestCase):
    """Test that non-LORA IC creation uses model_name instead of container spec."""

    def setUp(self):
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch("sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization")
    def test_ic_spec_uses_model_name(
        self,
        mock_is_mc,
        mock_fetch_peft,
        mock_fetch_mp_arn,
        mock_fetch_mp,
        mock_is_nova,
        mock_endpoint_exists,
        mock_epc_create,
        mock_endpoint_create,
        mock_ic_create,
        mock_ic_get,
    ):
        """InferenceComponentSpecification should use model_name from built_model."""
        mock_is_mc.return_value = True
        mock_fetch_peft.return_value = None
        mock_is_nova.return_value = False
        mock_endpoint_exists.return_value = False
        mock_fetch_mp_arn.return_value = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"

        model_package = Mock()
        model_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        model_package.inference_specification = Mock()
        container = Mock()
        container.is_checkpoint = False
        container.base_model = Mock()
        container.base_model.recipe_name = "meta-llama/Llama-3-8b"
        model_package.inference_specification.containers = [container]
        mock_fetch_mp.return_value = model_package

        mock_endpoint = Mock()
        mock_endpoint.endpoint_name = "test-endpoint"
        mock_endpoint_create.return_value = mock_endpoint

        mock_ic_get.return_value = Mock(
            inference_component_arn="arn:aws:sagemaker:us-west-2:123456789012:inference-component/test"
        )

        builder = ModelBuilder(
            model=Mock(),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
        )
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements

        builder.env_vars = {"KEY": "value"}
        builder.instance_type = "ml.g5.xlarge"
        builder.built_model = Mock()
        builder.built_model.model_name = "my-built-model"
        builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=1024,
            number_of_accelerator_devices_required=1,
        )

        with patch("sagemaker.core.resources.Action.create"):
            with patch("sagemaker.core.resources.Artifact.get_all", return_value=[]):
                builder._deploy_model_customization(
                    endpoint_name="test-endpoint",
                    instance_type="ml.g5.xlarge",
                    initial_instance_count=1,
                )

        mock_ic_create.assert_called_once()
        call_kwargs = mock_ic_create.call_args[1]
        ic_spec = call_kwargs["specification"]
        self.assertEqual(ic_spec.model_name, "my-built-model")


if __name__ == "__main__":
    unittest.main()
