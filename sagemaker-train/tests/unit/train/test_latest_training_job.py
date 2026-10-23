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
"""``latest_training_job`` is the public handle for the job a trainer created.

Before this property the only handle was the private ``_latest_training_job``, so callers
that needed the job name or ARN after ``train(wait=False)`` had to read the private
attribute or page ``ListTrainingJobs`` by name prefix. ``ModelTrainer.train()`` also
returned ``None``, so the job could not be chained from the call that created it.
"""

from __future__ import absolute_import

from unittest.mock import MagicMock, Mock, patch

import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import TrainingJob
from sagemaker.core.shapes import OutputDataConfig
from sagemaker.core.training.configs import Compute
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType

ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"
IMAGE = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-algo:latest"
BUCKET = "my-bucket"


@pytest.fixture(scope="module", autouse=True)
def modules_session():
    with (
        patch("sagemaker.train.Session", spec=Session) as session_mock,
        patch("sagemaker.train.defaults.resolve_and_validate_role", return_value=ROLE),
    ):
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = BUCKET
        session_instance.get_caller_identity_arn.return_value = ROLE
        session_instance.default_bucket_prefix = None
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        session_instance.boto_region_name = "us-west-2"
        yield session_instance


@pytest.fixture
def model_trainer(modules_session):
    return ModelTrainer(
        training_image=IMAGE,
        role=ROLE,
        sagemaker_session=modules_session,
        compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
        output_data_config=OutputDataConfig(s3_output_path=f"s3://{BUCKET}/output"),
    )


@patch("sagemaker.train.model_trainer.TrainingJob")
def test_model_trainer_train_returns_the_created_job(mock_training_job, model_trainer):
    created = TrainingJob(training_job_name="created-job")
    mock_training_job.create.return_value = created

    returned = model_trainer.train(wait=False)

    assert returned is created
    assert model_trainer.latest_training_job is created
    assert model_trainer.latest_training_job.training_job_name == "created-job"


@patch("sagemaker.train.model_trainer.TrainingJob")
def test_model_trainer_train_with_wait_returns_the_job_after_waiting(
    mock_training_job, model_trainer
):
    created = Mock(spec=TrainingJob)
    mock_training_job.create.return_value = created

    returned = model_trainer.train(wait=True, logs=False)

    created.wait.assert_called_once_with(logs=False)
    assert returned is created


def test_model_trainer_latest_training_job_is_none_before_train(model_trainer):
    assert model_trainer.latest_training_job is None


@patch("sagemaker.train.model_trainer.TrainingJob")
def test_model_trainer_dry_run_returns_none_and_creates_nothing(mock_training_job, model_trainer):
    assert model_trainer.train(wait=False, dry_run=True) is None
    mock_training_job.create.assert_not_called()
    assert model_trainer.latest_training_job is None


def test_model_trainer_latest_training_job_setter_attaches_an_existing_job(model_trainer):
    existing = TrainingJob(training_job_name="existing-job")

    model_trainer.latest_training_job = existing

    assert model_trainer.latest_training_job is existing
    assert model_trainer.output_data.endswith("/existing-job/output/output.tar.gz")


def test_model_trainer_latest_training_job_setter_rejects_a_job_name(model_trainer):
    with pytest.raises(TypeError, match="TrainingJob.get"):
        model_trainer.latest_training_job = "existing-job"


class _StubTrainer(BaseTrainer):
    """Minimal concrete trainer: BaseTrainer is abstract on ``train``."""

    def train(self, input_data_config=None, wait=True, logs=True, wait_timeout=None, dry_run=False):
        return None


@patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
def test_base_trainer_latest_training_job_is_none_before_train(mock_get_session):
    mock_get_session.return_value = Mock(sagemaker_config={})

    trainer = _StubTrainer()

    assert trainer.latest_training_job is None


@patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
def test_base_trainer_latest_training_job_setter_round_trips(mock_get_session):
    mock_get_session.return_value = Mock(sagemaker_config={})
    trainer = _StubTrainer()
    existing = TrainingJob(training_job_name="existing-job")

    trainer.latest_training_job = existing

    assert trainer.latest_training_job is existing
    with pytest.raises(TypeError, match="TrainingJob.get"):
        trainer.latest_training_job = "existing-job"


@patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
def test_base_trainer_show_metrics_reads_the_public_handle(mock_get_session):
    mock_get_session.return_value = Mock(sagemaker_config={})
    trainer = _StubTrainer()

    with pytest.raises(ValueError, match="trainer.latest_training_job = TrainingJob.get"):
        trainer.show_metrics()
    with pytest.raises(ValueError, match="trainer.latest_training_job = TrainingJob.get"):
        trainer.stream_logs()


def _serverless_trainer_patches(module):
    return [
        patch("sagemaker.train.common_utils.finetune_utils._get_beta_session"),
        patch(f"{module}.validate_data_path_exists"),
        patch(f"{module}._resolve_model_and_name", return_value=("test-model", "test-model")),
        patch(f"{module}.TrainDefaults.get_role", return_value=ROLE),
        patch(
            f"{module}.TrainDefaults.get_sagemaker_session",
            return_value=Mock(sagemaker_config={}),
        ),
        patch(f"{module}._get_unique_name", return_value="test-job-name"),
        patch(f"{module}._validate_and_resolve_model_package_group", return_value="test-group"),
        patch(f"{module}._create_input_data_config", return_value=[Mock()]),
        patch(f"{module}._convert_input_data_to_channels", return_value=[Mock()]),
        patch(f"{module}._create_output_config", return_value=Mock()),
        patch(f"{module}._create_serverless_config", return_value=Mock()),
        patch(f"{module}._create_mlflow_config", return_value=Mock()),
        patch(f"{module}._create_model_package_config", return_value=Mock()),
    ]


@pytest.mark.parametrize(
    "trainer_cls, module",
    [
        (SFTTrainer, "sagemaker.train.sft_trainer"),
        (DPOTrainer, "sagemaker.train.dpo_trainer"),
    ],
)
def test_serverless_trainers_expose_the_created_job(trainer_cls, module):
    options = Mock()
    options.to_dict.return_value = {"learning_rate": "0.001"}
    created = Mock(spec=TrainingJob)
    created.training_job_name = "test-job-name"
    patches = _serverless_trainer_patches(module) + [
        patch(
            f"{module}._get_fine_tuning_options_and_model_arn", return_value=(options, "arn", False)
        ),
        patch("sagemaker.core.resources.TrainingJob.create", return_value=created),
    ]
    for active in patches:
        active.start()
    try:
        trainer = trainer_cls(
            model="test-model",
            training_type=TrainingType.LORA,
            model_package_group="test-group",
            training_dataset="s3://bucket/train",
        )
        assert trainer.latest_training_job is None

        returned = trainer.train(wait=False)

        assert returned is created
        assert trainer.latest_training_job is created
        assert trainer.latest_training_job.training_job_name == "test-job-name"
    finally:
        for active in patches:
            active.stop()
