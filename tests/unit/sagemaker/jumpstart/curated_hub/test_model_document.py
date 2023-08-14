from __future__ import absolute_import
import unittest

from mock.mock import patch

from sagemaker.jumpstart.curated_hub.model_document import ModelDocumentCreator
from sagemaker.jumpstart.curated_hub.content_copy import CopyContentConfig
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectLocation,
)
import uuid
from sagemaker.jumpstart.curated_hub.hub_model_specs.hub_model_specs import ModelCapabilities


class ModelDocumentCreatorTest(unittest.TestCase):

    custom_hub_name = "test-curated-hub-chrstfu"
    test_region = "test_region"
    test_model_id = "test_model_id"

    @patch(
        "sagemaker.jumpstart.curated_hub.model_document.ModelDocumentCreator._make_hub_dependency_list"
    )
    @patch("sagemaker.jumpstart.curated_hub.accessors.model_dependency_s3_accessor.ModoelDependencyS3Accessor")
    @patch("sagemaker.jumpstart.curated_hub.accessors.model_dependency_s3_accessor.ModoelDependencyS3Accessor")
    def setUp(self, mock_src, mock_dst, mock_hub_dependency_list):
        self.mock_src = mock_src
        self.mock_dst = mock_dst
        mock_hub_dependency_list.return_value = (
            None  # mocking dependencies as they're all from the content copy
        )

        self.mock_studio_metadata = {
            self.test_model_id: {"dataType": "test_dataType", "problemType": "test_problemType"}
        }

        self.test_document_creator = ModelDocumentCreator(
            self.test_region, mock_src, mock_dst, self.mock_studio_metadata
        )

    # Testing overall workflow

    @patch("sagemaker.jumpstart.curated_hub.model_document.ModelDocumentCreator._dataset_config")
    @patch(
        "sagemaker.jumpstart.curated_hub.model_document."
        + "ModelDocumentCreator._make_hub_content_default_training_config"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.model_document."
        + "ModelDocumentCreator._make_hub_content_default_deployment_config"
    )
    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_no_training_removes_fields(
        self, mock_model_specs, mock_deployment_config, mock_training_config, mock_dataset_config
    ):
        mock_model_specs.training_supported = False
        mock_model_specs.incremental_training_supported = False
        mock_model_specs.supports_prepacked_inference.return_value = False

        mock_model_specs.model_id = self.test_model_id
        mock_model_specs.hosting_ecr_specs.framework = "test_hosting_framework"
        mock_deployment_config.return_value = {
            "ModelArtifactConfig": "test_should_keep",
            "ScriptConfig": "test_should_remove",
        }

        # ensure that all the relevant variables are present / not present
        document = self.test_document_creator._make_hub_content_document_json(mock_model_specs)

        self.assertEqual(document["Capabilities"], [])
        self.assertEqual(document["Framework"], "test_hosting_framework")
        self.assertEqual(document["Origin"], None)
        self.assertEqual(document["DataType"], "test_dataType")
        self.assertEqual(document["MlTask"], "test_problemType")
        self.assertNotIn("DefaultTrainingConfig", document.keys())
        self.assertNotIn("DatasetConfig", document.keys())

        self.assertIn("ScriptConfig", document["DefaultDeploymentConfig"].keys())
        self.assertIn("ModelArtifactConfig", document["DefaultDeploymentConfig"].keys())

    @patch("sagemaker.jumpstart.curated_hub.model_document.ModelDocumentCreator._dataset_config")
    @patch(
        "sagemaker.jumpstart.curated_hub.model_document."
        + "ModelDocumentCreator._make_hub_content_default_training_config"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.model_document."
        + "ModelDocumentCreator._make_hub_content_default_deployment_config"
    )
    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_training_supported_adds_capability_and_training_configs(
        self, mock_model_specs, mock_deployment_config, mock_training_config, mock_dataset_config
    ):
        mock_model_specs.training_supported = True
        mock_model_specs.incremental_training_supported = False
        mock_model_specs.supports_prepacked_inference.return_value = False

        mock_model_specs.model_id = self.test_model_id
        mock_model_specs.hosting_ecr_specs.framework = "test_hosting_framework"

        document = self.test_document_creator._make_hub_content_document_json(mock_model_specs)

        self.assertEqual(document["Capabilities"], [ModelCapabilities.TRAINING])
        self.assertIn("DefaultTrainingConfig", document.keys())
        self.assertIn("DatasetConfig", document.keys())

    @patch("sagemaker.jumpstart.curated_hub.model_document.ModelDocumentCreator._dataset_config")
    @patch(
        "sagemaker.jumpstart.curated_hub.model_document."
        + "ModelDocumentCreator._make_hub_content_default_training_config"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.model_document."
        + "ModelDocumentCreator._make_hub_content_default_deployment_config"
    )
    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_incremental_training_supported_adds_capability(
        self, mock_model_specs, mock_deployment_config, mock_training_config, mock_dataset_config
    ):
        mock_model_specs.training_supported = False
        mock_model_specs.incremental_training_supported = True
        mock_model_specs.supports_prepacked_inference.return_value = False

        mock_model_specs.model_id = self.test_model_id
        mock_model_specs.hosting_ecr_specs.framework = "test_hosting_framework"

        document = self.test_document_creator._make_hub_content_document_json(mock_model_specs)

        self.assertEqual(document["Capabilities"], [ModelCapabilities.INCREMENTAL_TRAINING])

    @patch("sagemaker.jumpstart.curated_hub.model_document.ModelDocumentCreator._dataset_config")
    @patch(
        "sagemaker.jumpstart.curated_hub.model_document."
        + "ModelDocumentCreator._make_hub_content_default_training_config"
    )
    @patch(
        "sagemaker.jumpstart.curated_hub.model_document."
        + "ModelDocumentCreator._make_hub_content_default_deployment_config"
    )
    @patch("sagemaker.jumpstart.types.JumpStartModelSpecs")
    def test_prepack_removes_script_uri(
        self, mock_model_specs, mock_deployment_config, mock_training_config, mock_dataset_config
    ):
        mock_model_specs.training_supported = False
        mock_model_specs.incremental_training_supported = False
        mock_model_specs.supports_prepacked_inference.return_value = True

        mock_model_specs.model_id = self.test_model_id
        mock_model_specs.hosting_ecr_specs.framework = "test_hosting_framework"
        mock_deployment_config.return_value = {
            "ModelArtifactConfig": "test_should_keep",
            "ScriptConfig": "test_should_remove",
        }

        document = self.test_document_creator._make_hub_content_document_json(mock_model_specs)

        self.assertNotIn("ScriptConfig", document["DefaultDeploymentConfig"].keys())
        self.assertIn("ModelArtifactConfig", document["DefaultDeploymentConfig"].keys())

    def _generate_random_copy_config(self, display_name: str) -> CopyContentConfig:
        test_src = S3ObjectLocation(bucket=str(uuid.uuid4()), key=str(uuid.uuid4()))
        test_dst = S3ObjectLocation(bucket=str(uuid.uuid4()), key=str(uuid.uuid4()))
        return CopyContentConfig(src=test_src, dst=test_dst, logging_name=display_name)
