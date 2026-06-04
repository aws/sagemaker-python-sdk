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
"""Unit tests for Restricted Model Package support in ModelBuilder."""

import unittest
from unittest.mock import Mock, patch

from sagemaker.serve.utils.model_package_utils import is_restricted_model_package, get_s3_uri_from_inference_spec
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode


def _make_rmp_model_package():
    """Create a mock RMP ModelPackage (managed_storage_type=Restricted)."""
    pkg = Mock()
    pkg.managed_storage_type = "Restricted"
    container = Mock()
    container.base_model = Mock()
    container.base_model.recipe_name = "nova-lite"
    container.base_model.hub_content_name = "nova-textgeneration-lite-v2"
    container.base_model.hub_content_version = "3.48.0"
    container.model_data_source = Mock()
    container.model_data_source.s3_data_source = Mock()
    container.model_data_source.s3_data_source.s3_uri = None
    container.model_data_source.s3_data_source.s3_data_type = "S3Prefix"
    container.model_data_source.s3_data_source.compression_type = "None"
    pkg.inference_specification.containers = [container]
    pkg.model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp-nova/1"
    return pkg


def _make_normal_model_package(s3_uri="s3://bucket/model/output/"):
    """Create a mock normal ModelPackage (not restricted)."""
    pkg = Mock()
    pkg.managed_storage_type = None
    container = Mock()
    container.base_model = Mock()
    container.base_model.recipe_name = "llama-3"
    container.base_model.hub_content_name = "llama-3"
    container.base_model.hub_content_version = "1.0"
    container.model_data_source = Mock()
    container.model_data_source.s3_data_source = Mock()
    container.model_data_source.s3_data_source.s3_uri = s3_uri
    container.model_data_source.s3_data_source.s3_data_type = "S3Prefix"
    container.model_data_source.s3_data_source.compression_type = "None"
    pkg.inference_specification.containers = [container]
    pkg.model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/llama/1"
    return pkg


class TestIsRestrictedModelPackage(unittest.TestCase):
    """Tests for is_restricted_model_package detection."""

    def test_restricted_returns_true(self):
        pkg = _make_rmp_model_package()
        self.assertTrue(is_restricted_model_package(pkg))

    def test_normal_package_returns_false(self):
        pkg = _make_normal_model_package()
        self.assertFalse(is_restricted_model_package(pkg))

    def test_none_returns_false(self):
        self.assertFalse(is_restricted_model_package(None))

    def test_none_managed_storage_type_returns_false(self):
        pkg = Mock()
        pkg.managed_storage_type = None
        self.assertFalse(is_restricted_model_package(pkg))

    def test_unassigned_managed_storage_type_returns_false(self):
        from sagemaker.core.utils.utils import Unassigned
        pkg = Mock()
        pkg.managed_storage_type = Unassigned()
        self.assertFalse(is_restricted_model_package(pkg))

    def test_other_storage_type_returns_false(self):
        pkg = Mock()
        pkg.managed_storage_type = "Standard"
        self.assertFalse(is_restricted_model_package(pkg))

    def test_no_managed_storage_type_attr_returns_false(self):
        pkg = Mock(spec=[])
        self.assertFalse(is_restricted_model_package(pkg))


class TestGetS3UriFromInferenceSpec(unittest.TestCase):
    """Tests for get_s3_uri_from_inference_spec utility."""

    def test_returns_none_for_rmp(self):
        pkg = _make_rmp_model_package()
        self.assertIsNone(get_s3_uri_from_inference_spec(pkg.inference_specification))

    def test_returns_uri_for_normal(self):
        pkg = _make_normal_model_package("s3://bucket/path/")
        self.assertEqual(get_s3_uri_from_inference_spec(pkg.inference_specification), "s3://bucket/path/")

    def test_returns_none_when_spec_is_none(self):
        self.assertIsNone(get_s3_uri_from_inference_spec(None))

    def test_returns_none_when_containers_empty(self):
        spec = Mock()
        spec.containers = []
        self.assertIsNone(get_s3_uri_from_inference_spec(spec))

    def test_returns_none_when_no_data_source(self):
        spec = Mock()
        container = Mock()
        container.model_data_source = None
        spec.containers = [container]
        self.assertIsNone(get_s3_uri_from_inference_spec(spec))

    def test_returns_none_when_no_s3_data_source(self):
        spec = Mock()
        container = Mock()
        container.model_data_source = Mock()
        container.model_data_source.s3_data_source = None
        spec.containers = [container]
        self.assertIsNone(get_s3_uri_from_inference_spec(spec))


class TestModelBuilderRMPBuild(unittest.TestCase):
    """Tests for ModelBuilder build path with restricted model packages."""

    def setUp(self):
        self.rmp_package = _make_rmp_model_package()
        self.normal_package = _make_normal_model_package()

    @patch("sagemaker.core.resources.Model.create")
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_nova_hosting_config")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model", return_value=True)
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization", return_value=True)
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_serve_setting")
    def test_build_rmp_nova_includes_env_vars(
        self, mock_serve, mock_is_mc, mock_fetch_mp, mock_arn,
        mock_is_nova, mock_nova_config, mock_create
    ):
        """Nova RMP build includes environment variables from hosting config."""
        mock_fetch_mp.return_value = self.rmp_package
        mock_arn.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp-nova/1"
        mock_nova_config.return_value = {
            "image_uri": "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-inference-repo:latest",
            "env_vars": {"CONTEXT_LENGTH": "8000", "MAX_CONCURRENCY": "8"},
            "instance_type": "ml.g6.48xlarge",
        }
        mock_create.return_value = Mock(model_arn="arn:aws:sagemaker:us-east-1:123:model/test")

        builder = ModelBuilder(model=self.rmp_package, role_arn="arn:aws:iam::123:role/Role")
        builder.mode = Mode.SAGEMAKER_ENDPOINT

        builder._build_single_modelbuilder()

        call_kwargs = mock_create.call_args[1]
        container = call_kwargs["containers"][0]
        self.assertEqual(container.model_package_name, "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp-nova/1")
        self.assertEqual(container.environment, {"CONTEXT_LENGTH": "8000", "MAX_CONCURRENCY": "8"})
        self.assertEqual(container.image, "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-inference-repo:latest")

    @patch("sagemaker.core.resources.Model.create")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model", return_value=False)
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization", return_value=True)
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_serve_setting")
    def test_build_rmp_non_nova_with_user_image(
        self, mock_serve, mock_is_mc, mock_fetch_mp, mock_arn,
        mock_is_nova, mock_create
    ):
        """Non-Nova RMP with user-provided image_uri uses it."""
        mock_fetch_mp.return_value = self.rmp_package
        mock_arn.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp-nova/1"
        mock_create.return_value = Mock(model_arn="arn:aws:sagemaker:us-east-1:123:model/test")

        builder = ModelBuilder(model=self.rmp_package, role_arn="arn:aws:iam::123:role/Role")
        builder.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0"
        builder.mode = Mode.SAGEMAKER_ENDPOINT

        builder._build_single_modelbuilder()

        call_kwargs = mock_create.call_args[1]
        container = call_kwargs["containers"][0]
        self.assertEqual(container.model_package_name, "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp-nova/1")
        self.assertEqual(container.image, "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0")

    @patch("sagemaker.core.resources.Model.create")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model", return_value=False)
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization", return_value=True)
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_serve_setting")
    def test_build_rmp_non_nova_no_image(
        self, mock_serve, mock_is_mc, mock_fetch_mp, mock_arn,
        mock_is_nova, mock_create
    ):
        """Non-Nova RMP without image_uri passes only model_package_name."""
        mock_fetch_mp.return_value = self.rmp_package
        mock_arn.return_value = "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp-nova/1"
        mock_create.return_value = Mock(model_arn="arn:aws:sagemaker:us-east-1:123:model/test")

        builder = ModelBuilder(model=self.rmp_package, role_arn="arn:aws:iam::123:role/Role")
        builder.mode = Mode.SAGEMAKER_ENDPOINT

        builder._build_single_modelbuilder()

        call_kwargs = mock_create.call_args[1]
        container = call_kwargs["containers"][0]
        self.assertEqual(container.model_package_name, "arn:aws:sagemaker:us-east-1:123456789012:model-package/rmp-nova/1")

    @patch("sagemaker.core.resources.Model.create")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft", return_value="FULL")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_and_cache_recipe_config")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model", return_value=False)
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization", return_value=True)
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_serve_setting")
    def test_build_non_lora_normal_uses_s3_uri(
        self, mock_serve, mock_is_mc, mock_fetch_mp, mock_is_nova,
        mock_recipe, mock_peft, mock_create
    ):
        """Regression: normal non-LORA build still uses s3_data_source with s3_uri."""
        mock_fetch_mp.return_value = self.normal_package
        mock_create.return_value = Mock(model_arn="arn:aws:sagemaker:us-east-1:123:model/test")

        builder = ModelBuilder(model=self.normal_package, role_arn="arn:aws:iam::123:role/Role")
        builder.image_uri = "test-image:latest"
        builder.mode = Mode.SAGEMAKER_ENDPOINT

        builder._build_single_modelbuilder()

        call_kwargs = mock_create.call_args[1]
        container = call_kwargs["containers"][0]
        self.assertEqual(container.model_data_source.s3_data_source.s3_uri, "s3://bucket/model/output/")


class TestModelBuilderRMPRecipeConfig(unittest.TestCase):
    """Tests for _fetch_and_cache_recipe_config with restricted model packages."""

    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model", return_value=False)
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization", return_value=True)
    def test_no_crash_when_s3_uri_is_none(self, mock_is_mc, mock_fetch_mp, mock_fetch_hub, mock_is_nova):
        rmp = _make_rmp_model_package()
        mock_fetch_mp.return_value = rmp
        mock_fetch_hub.return_value = {
            "RecipeCollection": [{"Name": "nova-lite", "HostingConfigs": [{"Profile": "Default", "EcrAddress": "img", "InstanceType": "ml.g6.48xlarge"}]}]
        }

        builder = ModelBuilder(model=rmp, role_arn="arn:aws:iam::123:role/Role")
        builder.s3_upload_path = None
        builder._fetch_and_cache_recipe_config()
        self.assertIsNone(builder.s3_upload_path)

    @patch("sagemaker.serve.model_builder.ModelBuilder._is_nova_model", return_value=False)
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_model_customization", return_value=True)
    def test_sets_path_for_normal(self, mock_is_mc, mock_fetch_mp, mock_fetch_hub, mock_is_nova):
        normal = _make_normal_model_package()
        mock_fetch_mp.return_value = normal
        mock_fetch_hub.return_value = {
            "RecipeCollection": [{"Name": "llama-3", "HostingConfigs": [{"Profile": "Default", "EcrAddress": "img", "InstanceType": "ml.g5.2xlarge"}]}]
        }

        builder = ModelBuilder(model=normal, role_arn="arn:aws:iam::123:role/Role")
        builder.s3_upload_path = None
        builder._fetch_and_cache_recipe_config()
        self.assertEqual(builder.s3_upload_path, "s3://bucket/model/output/")


class TestModelBuilderRMPConvertLocal(unittest.TestCase):
    """Tests for _convert_model_data_source_to_local with restricted model packages."""

    def test_returns_none_for_rmp(self):
        builder = ModelBuilder(model=_make_rmp_model_package(), role_arn="arn:aws:iam::123:role/Role")
        data_source = _make_rmp_model_package().inference_specification.containers[0].model_data_source
        self.assertIsNone(builder._convert_model_data_source_to_local(data_source))

    def test_works_for_normal(self):
        builder = ModelBuilder(model=_make_normal_model_package(), role_arn="arn:aws:iam::123:role/Role")
        data_source = _make_normal_model_package().inference_specification.containers[0].model_data_source
        result = builder._convert_model_data_source_to_local(data_source)
        self.assertEqual(result["S3DataSource"]["S3Uri"], "s3://bucket/model/output/")

    def test_returns_none_when_data_source_is_none(self):
        builder = ModelBuilder(model=_make_normal_model_package(), role_arn="arn:aws:iam::123:role/Role")
        self.assertIsNone(builder._convert_model_data_source_to_local(None))


if __name__ == "__main__":
    unittest.main()
