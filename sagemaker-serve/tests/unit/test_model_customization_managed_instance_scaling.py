"""Unit tests for managed_instance_scaling on the model customization deploy path."""

import unittest
from unittest.mock import Mock, patch

import pytest

from sagemaker.core.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.core.utils.utils import Unassigned
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder import ModelBuilder


def _patches(func):
    for target in (
        "sagemaker.serve.model_builder.ModelBuilder._is_nova_model",
        "sagemaker.core.resources.InferenceComponent.get_all",
        "sagemaker.core.resources.InferenceComponent.create",
        "sagemaker.core.resources.Endpoint.get",
        "sagemaker.core.resources.Endpoint.create",
        "sagemaker.core.resources.EndpointConfig.create",
        "sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist",
        "sagemaker.serve.model_builder.ModelBuilder._fetch_model_package",
        "sagemaker.serve.model_builder.ModelBuilder._fetch_peft",
        "sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn",
        "sagemaker.core.resources.Association.add",
        "sagemaker.core.resources.Artifact.get_all",
        "sagemaker.core.resources.Action.create",
        "sagemaker.core.resources.InferenceComponent.get",
    ):
        func = patch(target)(func)
    return func


ENDPOINT_CREATE, CONFIG_CREATE = 4, 5


class TestModelCustomizationManagedInstanceScaling(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        self.session.boto_region_name = "us-west-2"
        self.session.default_bucket.return_value = "test-bucket"
        self.session.default_bucket_prefix = "test-prefix"
        self.session.config = {}
        self.session.sagemaker_config = {}
        self.session.settings = Mock()
        self.session.settings.include_jumpstart_tags = False
        credentials = Mock(access_key="test-key", secret_key="test-secret", token=None)
        self.session.boto_session = Mock(region_name="us-west-2")
        self.session.boto_session.get_credentials.return_value = credentials

    def _deploy(self, mocks, endpoint_exists=False, **deploy_kwargs):
        # patch() decorators applied first come first: see _patches.
        (
            is_nova,
            _ic_get_all,
            _ic_create,
            _endpoint_get,
            endpoint_create,
            _config_create,
            does_endpoint_exist,
            fetch_package,
            fetch_peft,
            fetch_package_arn,
            _association_add,
            artifact_get_all,
            _action_create,
            ic_get,
        ) = mocks
        is_nova.return_value = False
        package = Mock()
        package.model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/p"
        package.inference_specification.containers = [Mock()]
        package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri = (
            "s3://test-bucket/model"
        )
        fetch_package.return_value = package
        fetch_peft.return_value = "FULL"
        does_endpoint_exist.return_value = endpoint_exists
        endpoint_create.return_value = Mock(wait_for_status=Mock())
        ic_get.return_value = Mock(
            inference_component_arn="arn:aws:sagemaker:us-west-2:123456789012:inference-component/c"
        )
        fetch_package_arn.return_value = package.model_package_arn
        artifact_get_all.return_value = [
            Mock(artifact_arn="arn:aws:sagemaker:us-west-2:123456789012:artifact/a")
        ]
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0",
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )
        builder.built_model = Mock(model_name="test-model")
        builder._deploy_model_customization(
            endpoint_name="test-endpoint",
            inference_config=ResourceRequirements(requests={"num_cpus": 4, "memory": 8192}),
            **deploy_kwargs,
        )

    @staticmethod
    def _variant(config_create):
        (variant,) = config_create.call_args[1]["production_variants"]
        return variant

    @_patches
    def test_sets_managed_instance_scaling_on_a_new_endpoint_configuration(self, *mocks):
        self._deploy(mocks, managed_instance_scaling={"MinInstanceCount": 1, "MaxInstanceCount": 4})
        scaling = self._variant(mocks[CONFIG_CREATE]).managed_instance_scaling
        assert (scaling.status, scaling.min_instance_count, scaling.max_instance_count) == (
            "ENABLED",
            1,
            4,
        )

    @_patches
    def test_min_instance_count_defaults_to_the_initial_instance_count(self, *mocks):
        self._deploy(
            mocks, initial_instance_count=2, managed_instance_scaling={"MaxInstanceCount": 4}
        )
        assert self._variant(mocks[CONFIG_CREATE]).managed_instance_scaling.min_instance_count == 2

    @_patches
    def test_no_managed_instance_scaling_unless_requested(self, *mocks):
        self._deploy(mocks)
        assert isinstance(self._variant(mocks[CONFIG_CREATE]).managed_instance_scaling, Unassigned)

    @_patches
    def test_an_existing_endpoint_keeps_its_configuration(self, *mocks):
        with patch("sagemaker.serve.model_builder.logger") as logger:
            self._deploy(
                mocks,
                endpoint_exists=True,
                managed_instance_scaling={"MinInstanceCount": 1, "MaxInstanceCount": 4},
            )
        assert not mocks[CONFIG_CREATE].called
        assert any("already exists" in c.args[0] for c in logger.warning.call_args_list)

    @_patches
    def test_a_minimum_of_zero_is_refused_before_anything_is_created(self, *mocks):
        with pytest.raises(ValueError, match="MinInstanceCount 0"):
            self._deploy(
                mocks, managed_instance_scaling={"MinInstanceCount": 0, "MaxInstanceCount": 2}
            )
        assert not mocks[CONFIG_CREATE].called
        assert not mocks[ENDPOINT_CREATE].called


if __name__ == "__main__":
    unittest.main()
