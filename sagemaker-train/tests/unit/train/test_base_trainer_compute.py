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
"""Unit tests for how BaseTrainer threads the ``compute`` field into its two
training backends.

``compute`` is not a ``BaseTrainer.__init__`` parameter; concrete trainers
(SFT/DPO/RLVR/...) assign ``self.compute`` and the shared backends consume it:

* ``_train_serverful_smtj`` maps it onto the ``TrainingJobCompute`` handed to
  ``ModelTrainer.from_recipe``.
* ``_train_hyperpod`` reads ``cluster_name`` / ``instance_type`` / ``node_count``
  into the HyperPod start-job override parameters.

These tests pin that wiring with all external boundaries mocked.
"""
from __future__ import absolute_import

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.train.base_trainer import BaseTrainer


class _ConcreteTrainer(BaseTrainer):
    """Minimal concrete BaseTrainer for exercising the shared backends."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name = "nova-lite"
        self._customization_technique = "sft"
        self.training_type = "lora"
        self.training_dataset = None
        self.validation_dataset = None
        self.compute = None
        self.networking = None
        self.stopping_condition = None
        self.training_image = "123.dkr.ecr.us-east-1.amazonaws.com/nova:latest"
        self.base_job_name = "nova-lite-sft"
        self.s3_output_path = None

    def train(self, *args, **kwargs):  # pragma: no cover - abstract impl
        return self._train_serverful_smtj(*args, **kwargs)


class TestServerfulComputeMapping:
    """``_train_serverful_smtj`` maps ``self.compute`` onto ``TrainingJobCompute``."""

    def _run(self, trainer):
        mock_model_trainer = MagicMock()
        mock_model_trainer._latest_training_job = MagicMock()
        mock_session = MagicMock()
        mock_session.boto_session.client.return_value.download_file.return_value = None

        with patch(
            "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session",
            return_value=mock_session,
        ), patch(
            "sagemaker.train.defaults.TrainDefaults.get_role",
            return_value="arn:aws:iam::1:role/x",
        ), patch(
            "sagemaker.train.common_utils.finetune_utils.get_recipe_s3_uri",
            return_value="s3://bucket/recipe.yaml",
        ), patch(
            "sagemaker.train.common_utils.finetune_utils.get_training_image",
            return_value="image:latest",
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._validate_hyperparameter_values"
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._get_smtj_override_spec",
            return_value={},
        ), patch(
            "sagemaker.train.common_utils.finetune_utils._render_recipe_placeholders",
            side_effect=lambda content, spec: content,
        ), patch(
            "sagemaker.train.model_trainer.ModelTrainer.from_recipe",
            return_value=mock_model_trainer,
        ) as mock_from_recipe:
            trainer.hyperparameters = MagicMock()
            trainer.hyperparameters.to_dict.return_value = {}
            trainer.train(wait=False)

        return mock_from_recipe.call_args.kwargs

    def test_compute_attributes_forwarded(self):
        trainer = _ConcreteTrainer()
        trainer.compute = MagicMock(
            instance_type="ml.p4d.24xlarge",
            instance_count=4,
            volume_size_in_gb=300,
            keep_alive_period_in_seconds=1200,
        )
        trainer.training_dataset = "s3://my-bucket/data/train/"

        kwargs = self._run(trainer)

        forwarded = kwargs["compute"]
        assert forwarded.instance_type == "ml.p4d.24xlarge"
        assert forwarded.instance_count == 4
        assert forwarded.volume_size_in_gb == 300
        assert forwarded.keep_alive_period_in_seconds == 1200


def _make_hyperpod_trainer(cluster_name="my-cluster", node_count=2):
    """Build an SFTTrainer instance for _train_hyperpod without heavy __init__."""
    from sagemaker.train.sft_trainer import SFTTrainer

    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.sagemaker_session = MagicMock()
    trainer.compute = SimpleNamespace(
        cluster_name=cluster_name,
        namespace="kubeflow",
        instance_type="ml.p5.48xlarge",
        node_count=node_count,
    )
    trainer._model_name = "amazon.nova-lite-v2"
    trainer._customization_technique = "SFT"
    trainer.training_type = "LORA"
    trainer.training_image = "123.dkr.ecr.us-west-2.amazonaws.com/img:latest"
    trainer.base_job_name = None
    trainer.training_dataset = None
    trainer.validation_dataset = None
    trainer.s3_output_path = "s3://bucket/out/"
    trainer.hyperparameters = None
    trainer._recipe_path = "recipe-name"
    trainer._overrides = None
    return trainer


class TestHyperPodComputeMapping:
    """``_train_hyperpod`` reads cluster config from ``self.compute``."""

    @patch("sagemaker.train.base_trainer.subprocess")
    @patch("sagemaker.train.base_trainer.TrainDefaults.verify_hyperpod_caller_permissions")
    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    def test_compute_fields_land_in_override_parameters(
        self, mock_get_session, mock_validate, mock_verify, mock_subprocess
    ):
        mock_get_session.return_value = MagicMock()
        mock_subprocess.run.return_value = SimpleNamespace(
            stdout="NAME: my-job-123\n", stderr=""
        )

        trainer = _make_hyperpod_trainer(node_count=3)
        with patch(
            "sagemaker.train.common_utils.finetune_utils.get_training_image",
            return_value=None,
        ):
            job_name = trainer._train_hyperpod(wait=False)

        assert job_name == "my-job-123"

        # The start-job command carries the override parameters as a JSON blob.
        start_cmd = mock_subprocess.run.call_args_list[-1].args[0]
        overrides = json.loads(start_cmd[start_cmd.index("--override-parameters") + 1])
        assert overrides["instance_type"] == "ml.p5.48xlarge"
        assert overrides["recipes.run.replicas"] == 3

    @patch("sagemaker.train.base_trainer.subprocess")
    @patch("sagemaker.train.base_trainer.TrainDefaults.verify_hyperpod_caller_permissions")
    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    def test_missing_cluster_name_raises(
        self, mock_get_session, mock_validate, mock_verify, mock_subprocess
    ):
        mock_get_session.return_value = MagicMock()
        trainer = _make_hyperpod_trainer(cluster_name="")

        with pytest.raises(ValueError, match="cluster_name is required"):
            trainer._train_hyperpod(wait=False)

        mock_verify.assert_not_called()
