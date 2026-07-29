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
import pytest
from unittest.mock import Mock, patch, MagicMock

from sagemaker.train.common_utils.finetune_utils import (
    get_hyperpod_training_image,
    extract_image_from_hyperpod_template,
)


SAMPLE_TEMPLATE_WITH_IMAGE = """\
---
# Source: my-chart/templates/training-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config
data:
  config.yaml: |-
    run:
      name: {{name}}
      data_s3_path: {{data_s3_path}}
    trainer:
      max_steps: 100
---
# Source: my-chart/templates/pytorchjob.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
spec:
  pytorchReplicaSpecs:
    Worker:
      template:
        spec:
          containers:
            - name: pytorch
              image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/repo:tag
              resources:
                limits:
                  nvidia.com/gpu: 8
"""

SAMPLE_TEMPLATE_NO_IMAGE = """\
---
# Source: my-chart/templates/training-config.yaml
apiVersion: v1
kind: ConfigMap
data:
  config.yaml: |-
    run:
      name: {{name}}
"""

SAMPLE_TEMPLATE_NO_TRAINING_CONFIG = """\
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: worker
      image: some-image:latest
"""


class TestExtractImageFromHyperpodTemplate:
    """Tests for the shared extract_image_from_hyperpod_template helper."""

    def test_extracts_image_from_valid_template(self):
        result = extract_image_from_hyperpod_template(SAMPLE_TEMPLATE_WITH_IMAGE)
        assert result == "123456789012.dkr.ecr.us-east-1.amazonaws.com/repo:tag"

    def test_returns_none_when_no_training_config_section(self):
        result = extract_image_from_hyperpod_template(SAMPLE_TEMPLATE_NO_TRAINING_CONFIG)
        assert result is None

    def test_returns_none_when_no_image_match(self):
        result = extract_image_from_hyperpod_template(SAMPLE_TEMPLATE_NO_IMAGE)
        assert result is None

    def test_returns_none_for_empty_string(self):
        result = extract_image_from_hyperpod_template("")
        assert result is None


_PATCH_GET_RECIPE = "sagemaker.train.common_utils.finetune_utils._get_recipe_entry_and_override_spec"


class TestGetHyperpodTrainingImage:
    """Tests for get_hyperpod_training_image end-to-end."""

    @patch(_PATCH_GET_RECIPE)
    def test_returns_image_from_template(self, mock_get_recipe):
        mock_get_recipe.return_value = (
            {"HpEksPayloadTemplateS3Uri": "s3://bucket/template.yaml"},
            {},
        )

        mock_session = MagicMock()
        mock_body = Mock()
        mock_body.read.return_value = SAMPLE_TEMPLATE_WITH_IMAGE.encode("utf-8")
        mock_session.boto_session.client.return_value.get_object.return_value = {
            "Body": mock_body
        }

        result = get_hyperpod_training_image(
            model_name="nova-textgeneration-lite-v2",
            customization_technique="CPT",
            training_type="FULL",
            sagemaker_session=mock_session,
        )

        assert result == "123456789012.dkr.ecr.us-east-1.amazonaws.com/repo:tag"

    @patch(_PATCH_GET_RECIPE)
    def test_returns_none_when_no_recipe_found(self, mock_get_recipe):
        mock_get_recipe.side_effect = ValueError("No recipes found")

        mock_session = MagicMock()
        result = get_hyperpod_training_image(
            model_name="unknown-model",
            customization_technique="CPT",
            training_type="FULL",
            sagemaker_session=mock_session,
        )

        assert result is None

    @patch(_PATCH_GET_RECIPE)
    def test_returns_none_when_no_template_uri(self, mock_get_recipe):
        mock_get_recipe.return_value = ({"Name": "some-recipe"}, {})

        mock_session = MagicMock()
        result = get_hyperpod_training_image(
            model_name="nova-textgeneration-lite-v2",
            customization_technique="CPT",
            training_type="FULL",
            sagemaker_session=mock_session,
        )

        assert result is None

    @patch(_PATCH_GET_RECIPE)
    def test_returns_none_when_s3_download_fails(self, mock_get_recipe):
        mock_get_recipe.return_value = (
            {"HpEksPayloadTemplateS3Uri": "s3://bucket/template.yaml"},
            {},
        )

        mock_session = MagicMock()
        mock_session.boto_session.client.return_value.get_object.side_effect = Exception("Access Denied")

        result = get_hyperpod_training_image(
            model_name="nova-textgeneration-lite-v2",
            customization_technique="CPT",
            training_type="FULL",
            sagemaker_session=mock_session,
        )

        assert result is None

    @patch(_PATCH_GET_RECIPE)
    def test_returns_none_when_template_has_no_image(self, mock_get_recipe):
        mock_get_recipe.return_value = (
            {"HpEksPayloadTemplateS3Uri": "s3://bucket/template.yaml"},
            {},
        )

        mock_session = MagicMock()
        mock_body = Mock()
        mock_body.read.return_value = SAMPLE_TEMPLATE_NO_IMAGE.encode("utf-8")
        mock_session.boto_session.client.return_value.get_object.return_value = {
            "Body": mock_body
        }

        result = get_hyperpod_training_image(
            model_name="nova-textgeneration-lite-v2",
            customization_technique="CPT",
            training_type="FULL",
            sagemaker_session=mock_session,
        )

        assert result is None


class TestTrainHyperpodRaisesWhenNoImage:
    """Tests that _train_hyperpod raises ValueError when no image can be resolved."""

    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.base_trainer.get_hyperpod_training_image", return_value=None)
    @patch("sagemaker.train.base_trainer.get_training_image", return_value=None)
    @patch("sagemaker.train.base_trainer.subprocess")
    def test_raises_valueerror_when_image_is_none(
        self, mock_subprocess, mock_get_smtj_image, mock_get_hp_image, mock_get_session, mock_validate
    ):
        """_train_hyperpod raises ValueError if training_image is None and cannot be resolved."""
        from sagemaker.train.sft_trainer import SFTTrainer

        mock_get_session.return_value = MagicMock()
        mock_subprocess.run.return_value = MagicMock(stdout="NAME: job\n", stderr="")

        trainer = SFTTrainer.__new__(SFTTrainer)
        trainer.sagemaker_session = MagicMock()
        trainer.compute = MagicMock()
        trainer.compute.cluster_name = "my-cluster"
        trainer.compute.namespace = "kubeflow"
        trainer.compute.instance_type = "ml.p5.48xlarge"
        trainer.compute.node_count = 1
        trainer._model_name = "nova-textgeneration-lite-v2"
        trainer._customization_technique = "CPT"
        trainer.training_type = "FULL"
        trainer.training_image = None  # No user-provided image
        trainer.base_job_name = None
        trainer.training_dataset = None
        trainer.validation_dataset = None
        trainer.s3_output_path = None
        trainer.hyperparameters = None
        trainer._recipe_path = "some-recipe"
        trainer._overrides = None
        trainer.networking = None
        trainer.mlflow_resource_arn = None
        trainer.mlflow_experiment_name = None
        trainer.mlflow_run_name = None

        with pytest.raises(ValueError, match="training_image is required for HyperPod"):
            trainer._train_hyperpod(wait=False)

        # Verify subprocess (job submission) was never called for start-job
        start_job_calls = [
            c for c in mock_subprocess.run.call_args_list
            if c[0][0][0:2] == ["hyperpod", "start-job"]
        ]
        assert len(start_job_calls) == 0
