from __future__ import absolute_import
import unittest

from mock.mock import patch

from sagemaker.jumpstart.curated_hub.content_copy import ContentCopier, CopyContentConfig
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectLocation,
)
import uuid


class ContentCopierTest(unittest.TestCase):

    custom_hub_name = "test-curated-hub-chrstfu"
    test_region = "test_region"

    @patch("botocore.client.BaseClient")
    @patch(
        "sagemaker.jumpstart.curated_hub.accessors.model_dependency_s3_accessor.ModelDependencyS3Accessor"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.accessors.model_dependency_s3_accessor.ModelDependencyS3Accessor"
    )
    def setUp(self, mock_src, mock_dst, mock_client):
        self.mock_src = mock_src
        self.mock_dst = mock_dst
        self.mock_client = mock_client
        self.test_content_copier = ContentCopier(self.test_region, mock_client, mock_src, mock_dst)

    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_get_copy_configs_for_inference_dependencies_no_prepack(self, mock_model_specs):
        test_artifact_copy_config = self._generate_random_copy_config("inference artifact")
        self.mock_src.get_inference_artifact_s3_reference.return_value = (
            test_artifact_copy_config.src
        )
        self.mock_dst.get_inference_artifact_s3_reference.return_value = (
            test_artifact_copy_config.dst
        )

        test_script_copy_config = self._generate_random_copy_config("inference script")
        self.mock_src.get_inference_script_s3_reference.return_value = test_script_copy_config.src
        self.mock_dst.get_inference_script_s3_reference.return_value = test_script_copy_config.dst

        mock_model_specs.supports_prepacked_inference.return_value = False

        copy_configs: List[
            CopyContentConfig
        ] = self.test_content_copier._get_copy_configs_for_inference_dependencies(mock_model_specs)

        self.assertIn(test_artifact_copy_config, copy_configs)
        self.assertIn(test_script_copy_config, copy_configs)

    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_get_copy_configs_for_inference_dependencies_prepack(self, mock_model_specs):
        test_artifact_copy_config = self._generate_random_copy_config("inference artifact")
        self.mock_src.get_inference_artifact_s3_reference.return_value = (
            test_artifact_copy_config.src
        )
        self.mock_dst.get_inference_artifact_s3_reference.return_value = (
            test_artifact_copy_config.dst
        )

        test_script_copy_config = self._generate_random_copy_config("inference script")
        self.mock_src.get_inference_script_s3_reference.return_value = test_script_copy_config.src
        self.mock_dst.get_inference_script_s3_reference.return_value = test_script_copy_config.dst

        mock_model_specs.supports_prepacked_inference.return_value = True

        copy_configs: List[
            CopyContentConfig
        ] = self.test_content_copier._get_copy_configs_for_inference_dependencies(mock_model_specs)

        self.assertIn(test_artifact_copy_config, copy_configs)
        self.assertNotIn(test_script_copy_config, copy_configs)

    # TODO: unittest ContentCopier._get_copy_configs_for_training_dataset

    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_training_dataset"
    )
    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_get_copy_configs_for_training_dependencies(
        self, mock_model_specs, mock_get_copy_configs_for_training_dataset
    ):
        test_artifact_copy_config = self._generate_random_copy_config("training artifact")
        self.mock_src.get_training_artifact_s3_reference.return_value = (
            test_artifact_copy_config.src
        )
        self.mock_dst.get_training_artifact_s3_reference.return_value = (
            test_artifact_copy_config.dst
        )

        test_script_copy_config = self._generate_random_copy_config("training script")
        self.mock_src.get_training_script_s3_reference.return_value = test_script_copy_config.src
        self.mock_dst.get_training_script_s3_reference.return_value = test_script_copy_config.dst

        test_dataset_copy_config = self._generate_random_copy_config("training dataset")
        mock_get_copy_configs_for_training_dataset.return_value = [test_dataset_copy_config]

        copy_configs: List[
            CopyContentConfig
        ] = self.test_content_copier._get_copy_configs_for_training_dependencies(mock_model_specs)

        self.assertIn(test_artifact_copy_config, copy_configs)
        self.assertIn(test_script_copy_config, copy_configs)
        self.assertIn(test_dataset_copy_config, copy_configs)

    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_get_copy_configs_for_demo_notebook_dependencies(self, mock_model_specs):
        test_demo_notebook_copy_config = self._generate_random_copy_config("demo notebook")
        self.mock_src.get_demo_notebook_s3_reference.return_value = (
            test_demo_notebook_copy_config.src
        )
        self.mock_dst.get_demo_notebook_s3_reference.return_value = (
            test_demo_notebook_copy_config.dst
        )

        copy_configs: List[
            CopyContentConfig
        ] = self.test_content_copier._get_copy_configs_for_demo_notebook_dependencies(
            mock_model_specs
        )

        self.assertIn(test_demo_notebook_copy_config, copy_configs)

    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_get_copy_configs_for_markdown_dependencies(self, mock_model_specs):
        test_markdown_copy_config = self._generate_random_copy_config("markdown")
        self.mock_src.get_markdown_s3_reference.return_value = test_markdown_copy_config.src
        self.mock_dst.get_markdown_s3_reference.return_value = test_markdown_copy_config.dst

        copy_configs: List[
            CopyContentConfig
        ] = self.test_content_copier._get_copy_configs_for_markdown_dependencies(mock_model_specs)

        self.assertIn(test_markdown_copy_config, copy_configs)

    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._parallel_execute_copy_configs"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_training_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_markdown_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_demo_notebook_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_inference_dependencies"
    )
    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_copy_hub_content_dependencies_to_hub_bucket_no_training(
        self,
        mock_model_specs,
        mock_inference_dep,
        mock_demo_notebook_deps,
        mock_markdown_deps,
        mock_training_deps,
        mock_copy,
    ):
        mock_model_specs.training_supported = False

        self.test_content_copier.copy_hub_content_dependencies_to_hub_bucket(mock_model_specs)

        mock_inference_dep.assert_called_once()
        mock_demo_notebook_deps.assert_called_once()
        mock_markdown_deps.assert_called_once()
        mock_copy.assert_called_once()

        mock_training_deps.assert_not_called()

    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._parallel_execute_copy_configs"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_training_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_markdown_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_demo_notebook_dependencies"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.content_copy.ContentCopier._get_copy_configs_for_inference_dependencies"
    )
    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_copy_hub_content_dependencies_to_hub_bucket_training(
        self,
        mock_model_specs,
        mock_inference_dep,
        mock_demo_notebook_deps,
        mock_markdown_deps,
        mock_training_deps,
        mock_copy,
    ):
        mock_model_specs.training_supported = True

        self.test_content_copier.copy_hub_content_dependencies_to_hub_bucket(mock_model_specs)

        mock_inference_dep.assert_called_once()
        mock_demo_notebook_deps.assert_called_once()
        mock_markdown_deps.assert_called_once()
        mock_copy.assert_called_once()
        mock_training_deps.assert_called_once()

    def _generate_random_copy_config(self, display_name: str) -> CopyContentConfig:
        test_src = S3ObjectLocation(bucket=str(uuid.uuid4()), key=str(uuid.uuid4()))
        test_dst = S3ObjectLocation(bucket=str(uuid.uuid4()), key=str(uuid.uuid4()))
        return CopyContentConfig(src=test_src, dst=test_dst, logging_name=display_name)
