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
"""Unit tests for the BaseEvaluator ``compute`` constructor field.

These cover the newly added ``compute`` field (``Compute`` for serverful SMTJ
evaluation, ``HyperPodCompute`` for cluster evaluation) and the two methods that
consume it: ``_write_and_submit_smtj_recipe`` (serverful) and
``_submit_hyperpod_eval_job`` (HyperPod).
"""
from __future__ import absolute_import

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core.training.configs import Compute, HyperPodCompute
from sagemaker.train.evaluate.base_evaluator import BaseEvaluator


DEFAULT_MODEL = "llama3-2-1b-instruct"
DEFAULT_S3_OUTPUT = "s3://my-bucket/outputs"
DEFAULT_MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
DEFAULT_MODEL_PACKAGE_GROUP_ARN = (
    "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-package"
)
DEFAULT_HUB_CONTENT_ARN = "arn:aws:sagemaker:us-west-2:aws:hub-content/HubName/Model/llama3/1"


@pytest.fixture
def mock_session():
    """Mock SageMaker session that skips MLflow/region resolution work."""
    session = MagicMock()
    session.boto_region_name = "us-west-2"
    session.boto_session = MagicMock()
    session.get_caller_identity_arn.return_value = (
        "arn:aws:iam::123456789012:role/test-role"
    )
    return session


@pytest.fixture
def mock_model_info():
    info = MagicMock()
    info.base_model_name = DEFAULT_MODEL
    info.base_model_arn = DEFAULT_HUB_CONTENT_ARN
    info.source_model_package_arn = None
    return info


def _build_evaluator(mock_session, compute):
    """Construct a BaseEvaluator with model resolution mocked out."""
    return BaseEvaluator(
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
        compute=compute,
    )


class TestComputeField:
    """The ``compute`` field accepts Compute / HyperPodCompute and defaults None."""

    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_compute_defaults_to_none(self, mock_resolve, mock_session, mock_model_info):
        mock_resolve.return_value = mock_model_info

        evaluator = BaseEvaluator(
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )

        assert evaluator.compute is None

    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_accepts_serverful_compute(self, mock_resolve, mock_session, mock_model_info):
        mock_resolve.return_value = mock_model_info
        compute = Compute(instance_type="ml.g5.12xlarge", instance_count=2)

        evaluator = _build_evaluator(mock_session, compute)

        assert isinstance(evaluator.compute, Compute)
        assert evaluator.compute.instance_type == "ml.g5.12xlarge"
        assert evaluator.compute.instance_count == 2

    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_accepts_hyperpod_compute(self, mock_resolve, mock_session, mock_model_info):
        mock_resolve.return_value = mock_model_info
        compute = HyperPodCompute(cluster_name="my-cluster", instance_type="ml.p5.48xlarge")

        evaluator = _build_evaluator(mock_session, compute)

        assert isinstance(evaluator.compute, HyperPodCompute)
        assert evaluator.compute.cluster_name == "my-cluster"


def _bare_evaluator(compute):
    """Build a BaseEvaluator without running __init__ (no Hub resolution)."""
    evaluator = BaseEvaluator.__new__(BaseEvaluator)
    object.__setattr__(evaluator, "compute", compute)
    object.__setattr__(evaluator, "sagemaker_session", MagicMock())
    object.__setattr__(evaluator, "training_image", "123.dkr.ecr.us-west-2.amazonaws.com/img:latest")
    object.__setattr__(evaluator, "s3_output_path", DEFAULT_S3_OUTPUT)
    object.__setattr__(evaluator, "base_eval_name", "eval")
    object.__setattr__(evaluator, "recipe", "recipe-name")
    object.__setattr__(evaluator, "model", DEFAULT_MODEL)
    object.__setattr__(evaluator, "base_model_name", DEFAULT_MODEL)
    object.__setattr__(evaluator, "mlflow_resource_arn", None)
    object.__setattr__(evaluator, "mlflow_experiment_name", None)
    object.__setattr__(evaluator, "mlflow_run_name", None)
    return evaluator


class TestWriteAndSubmitSmtjRecipe:
    """``_write_and_submit_smtj_recipe`` builds compute from the ``compute`` field."""

    @patch("sagemaker.train.model_trainer.ModelTrainer.from_recipe")
    def test_compute_forwarded_to_model_trainer(self, mock_from_recipe, tmp_path):
        compute = Compute(
            instance_type="ml.g5.12xlarge",
            instance_count=3,
            volume_size_in_gb=200,
            keep_alive_period_in_seconds=600,
        )
        evaluator = _bare_evaluator(compute)
        evaluator._build_output_data_config = MagicMock(return_value=None)

        mock_trainer = MagicMock()
        mock_trainer._latest_training_job = MagicMock()
        mock_from_recipe.return_value = mock_trainer

        recipe_path = str(tmp_path / "recipe.yaml")
        result = evaluator._write_and_submit_smtj_recipe(
            recipe_dict={"training_config": {"foo": "bar"}},
            recipe_tmp_path=recipe_path,
            training_image="image:latest",
            sagemaker_session=MagicMock(),
            role="arn:aws:iam::1:role/x",
            base_job_name="eval-job",
        )

        forwarded = mock_from_recipe.call_args.kwargs["compute"]
        assert forwarded.instance_type == "ml.g5.12xlarge"
        assert forwarded.instance_count == 3
        assert forwarded.volume_size_in_gb == 200
        assert forwarded.keep_alive_period_in_seconds == 600
        assert result is mock_trainer._latest_training_job

    @patch("sagemaker.train.model_trainer.ModelTrainer.from_recipe")
    def test_unresolved_placeholder_raises_before_submit(self, mock_from_recipe, tmp_path):
        evaluator = _bare_evaluator(Compute(instance_type="ml.g5.12xlarge", instance_count=1))
        evaluator._build_output_data_config = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="unresolved placeholders"):
            evaluator._write_and_submit_smtj_recipe(
                recipe_dict={"run": {"name": "{{job_name}}"}},
                recipe_tmp_path=str(tmp_path / "recipe.yaml"),
                training_image="image:latest",
                sagemaker_session=MagicMock(),
                role="arn:aws:iam::1:role/x",
                base_job_name="eval-job",
            )

        mock_from_recipe.assert_not_called()


class TestSubmitHyperpodEvalJob:
    """``_submit_hyperpod_eval_job`` reads cluster config from the ``compute`` field."""

    @patch("sagemaker.train.evaluate.base_evaluator.validate_hyperpod_compute")
    @patch("sagemaker.train.evaluate.base_evaluator.TrainDefaults.verify_hyperpod_caller_permissions")
    @patch("subprocess.run")
    def test_uses_compute_cluster_and_parses_job_name(
        self, mock_run, mock_verify, mock_validate
    ):
        compute = HyperPodCompute(
            cluster_name="my-cluster",
            instance_type="ml.p5.48xlarge",
            node_count=2,
            namespace="kubeflow",
        )
        evaluator = _bare_evaluator(compute)

        mock_run.side_effect = [
            SimpleNamespace(stdout="", stderr=""),  # connect-cluster
            SimpleNamespace(stdout="NAME: eval-job-123\n", stderr=""),  # start-job
        ]

        job_name = evaluator._submit_hyperpod_eval_job(base_job_name="eval")

        assert job_name == "eval-job-123"
        mock_verify.assert_called_once()
        assert mock_verify.call_args.kwargs["cluster_name"] == "my-cluster"
        # node_count and instance_type from compute land in the start-job overrides.
        start_cmd = mock_run.call_args_list[1].args[0]
        overrides = start_cmd[start_cmd.index("--override-parameters") + 1]
        assert '"instance_type": "ml.p5.48xlarge"' in overrides
        assert '"recipes.run.replicas": 2' in overrides

    @patch("sagemaker.train.evaluate.base_evaluator.validate_hyperpod_compute")
    @patch("sagemaker.train.evaluate.base_evaluator.TrainDefaults.verify_hyperpod_caller_permissions")
    @patch("subprocess.run")
    def test_missing_cluster_name_raises(self, mock_run, mock_verify, mock_validate):
        compute = HyperPodCompute(cluster_name="", instance_type="ml.p5.48xlarge")
        evaluator = _bare_evaluator(compute)

        with pytest.raises(ValueError, match="cluster_name is required"):
            evaluator._submit_hyperpod_eval_job()

        mock_verify.assert_not_called()
        mock_run.assert_not_called()
