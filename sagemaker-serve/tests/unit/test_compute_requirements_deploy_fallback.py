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
"""Compute requirements fallback when build() never cached them.

``build(reuse_resources=True)`` returns as soon as it finds a reusable Model, so the
recipe hosting config — and with it ``_cached_compute_requirements`` — is never applied.
``_deploy_model_customization`` must resolve the requirements on demand instead of
raising ``AttributeError``.
"""
from __future__ import absolute_import

from unittest.mock import Mock, patch

from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements
from sagemaker.serve.model_builder import ModelBuilder


RESOLVED = InferenceComponentComputeResourceRequirements(
    min_memory_required_in_mb=24576, number_of_accelerator_devices_required=4
)


def _model_package():
    package = Mock()
    package.model_package_arn = (
        "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
    )
    container = Mock()
    container.base_model = Mock()
    container.base_model.recipe_name = "test-base-model"
    container.model_data_source = Mock()
    container.model_data_source.s3_data_source = Mock()
    container.model_data_source.s3_data_source.s3_uri = "s3://bucket/fine-tuned-model/"
    package.inference_specification = Mock()
    package.inference_specification.containers = [container]
    return package


class TestComputeRequirementsDeployFallback:
    """Deploy resolves compute requirements when the build-time cache is unset."""

    def test_cached_compute_requirements_defaults_to_none(self):
        # Declared as a dataclass field, so reading it never raises AttributeError.
        assert ModelBuilder()._cached_compute_requirements is None

    def test_get_compute_requirements_prefers_cache(self):
        builder = ModelBuilder(instance_type="ml.g5.12xlarge")
        builder._cached_compute_requirements = RESOLVED

        with patch.object(ModelBuilder, "_resolve_compute_requirements") as mock_resolve:
            assert builder._get_compute_requirements_for_deploy() is RESOLVED

        mock_resolve.assert_not_called()

    @patch.object(ModelBuilder, "_resolve_compute_requirements", return_value=RESOLVED)
    def test_get_compute_requirements_resolves_and_caches(self, mock_resolve):
        builder = ModelBuilder(instance_type="ml.g5.12xlarge")

        assert builder._get_compute_requirements_for_deploy() is RESOLVED
        # Second call is served from the cache — resolution happens at most once.
        assert builder._get_compute_requirements_for_deploy() is RESOLVED

        mock_resolve.assert_called_once_with(instance_type="ml.g5.12xlarge")
        assert builder._cached_compute_requirements is RESOLVED

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch.object(ModelBuilder, "_resolve_compute_requirements", return_value=RESOLVED)
    @patch.object(ModelBuilder, "_fetch_model_package_arn")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft", return_value=None)
    @patch.object(ModelBuilder, "_does_endpoint_exist", return_value=False)
    @patch.object(ModelBuilder, "_is_nova_model", return_value=False)
    def test_deploy_without_cache_resolves_requirements(
        self,
        mock_is_nova,
        mock_endpoint_exists,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_resolve,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_ic_create,
        mock_ic_get,
    ):
        package = _model_package()
        mock_fetch_package.return_value = package
        mock_fetch_package_arn.return_value = package.model_package_arn
        mock_endpoint_create.return_value = Mock(wait_for_status=Mock())
        mock_ic_get.return_value = Mock(
            inference_component_arn=(
                "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test-ic"
            )
        )

        builder = ModelBuilder(
            model=package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        builder.built_model = Mock(model_name="test-model")
        # No build() call, so nothing was cached — mirrors the reuse_resources path.
        assert builder._cached_compute_requirements is None

        with patch("sagemaker.core.resources.Action"), patch(
            "sagemaker.core.resources.Association"
        ), patch("sagemaker.core.resources.Artifact"):
            builder._deploy_model_customization(endpoint_name="test-endpoint")

        mock_resolve.assert_called_once_with(instance_type="ml.g5.12xlarge")
        spec = mock_ic_create.call_args[1]["specification"]
        assert spec.compute_resource_requirements is RESOLVED

    @patch("sagemaker.core.resources.InferenceComponent.get_all", return_value=[])
    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch.object(ModelBuilder, "_resolve_compute_requirements", return_value=RESOLVED)
    @patch.object(ModelBuilder, "_fetch_model_package_arn")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft", return_value="LORA")
    @patch.object(ModelBuilder, "_does_endpoint_exist", return_value=False)
    @patch.object(ModelBuilder, "_is_nova_model", return_value=False)
    def test_lora_base_component_without_cache_resolves_requirements(
        self,
        mock_is_nova,
        mock_endpoint_exists,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_resolve,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_ic_create,
        mock_ic_get,
        mock_ic_get_all,
    ):
        package = _model_package()
        mock_fetch_package.return_value = package
        mock_fetch_package_arn.return_value = package.model_package_arn
        mock_endpoint_create.return_value = Mock(wait_for_status=Mock())
        mock_ic_get.return_value = Mock(
            wait_for_status=Mock(),
            inference_component_arn=(
                "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test-ic"
            ),
        )

        builder = ModelBuilder(
            model=package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        builder.built_model = Mock(model_name="test-model")
        builder._adapter_s3_uri = "s3://bucket/adapter/"

        with patch("sagemaker.core.resources.Action"), patch(
            "sagemaker.core.resources.Association"
        ), patch("sagemaker.core.resources.Artifact"):
            builder._deploy_model_customization(endpoint_name="test-endpoint")

        mock_resolve.assert_called_once_with(instance_type="ml.g5.12xlarge")
        base_spec = mock_ic_create.call_args_list[0][1]["specification"]
        assert base_spec.compute_resource_requirements is RESOLVED
