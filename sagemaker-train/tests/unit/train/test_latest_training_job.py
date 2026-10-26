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
"""Tests of the public job handle of a trainer.

The ``latest_training_job`` property and its setter, the ``TrainingJob`` that ``train()``
returns, and the ``on_submit`` ordering contract (called once with the accepted job, before
any wait, not for a ``dry_run``) across ``ModelTrainer``, the five serverless trainers, and
the HyperPod path of ``BaseTrainer``.
"""

from __future__ import absolute_import

import json
import re
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import TrainingJob
from sagemaker.core.shapes import OutputDataConfig
from sagemaker.core.training.configs import Compute, HyperPodCompute
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.cpt_trainer import CPTTrainer
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType

ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"
IMAGE = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-algo:latest"
BUCKET = "my-bucket"
REWARD_EVALUATOR = (
    "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/JsonDoc/reward/1.0.0"
)
SETTER_ERROR = re.escape("Use TrainingJob.get(<name>) to attach an existing job.")
ATTACH_HINT = re.escape("trainer.latest_training_job = TrainingJob.get('<job-name>')")
HYPERPOD_IMAGE = "123456789012.dkr.ecr.us-west-2.amazonaws.com/img:latest"
SERVERLESS_TRAINERS = [
    pytest.param((SFTTrainer, "sagemaker.train.sft_trainer", {}), id="sft"),
    pytest.param((DPOTrainer, "sagemaker.train.dpo_trainer", {}), id="dpo"),
    pytest.param(
        (
            RLVRTrainer,
            "sagemaker.train.rlvr_trainer",
            {"custom_reward_function": REWARD_EVALUATOR, "skip_reward_validation": True},
        ),
        id="rlvr",
    ),
    pytest.param((RLAIFTrainer, "sagemaker.train.rlaif_trainer", {}), id="rlaif"),
]


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
    with pytest.raises(TypeError, match=SETTER_ERROR):
        model_trainer.latest_training_job = "existing-job"


class _StubTrainer(BaseTrainer):
    """Minimal concrete trainer: BaseTrainer is abstract on ``train``."""

    def train(
        self,
        input_data_config=None,
        wait=True,
        logs=True,
        wait_timeout=None,
        dry_run=False,
        on_submit=None,
    ):
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
    with pytest.raises(TypeError, match=SETTER_ERROR):
        trainer.latest_training_job = "existing-job"


@patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
def test_base_trainer_show_metrics_reads_the_public_handle(mock_get_session):
    mock_get_session.return_value = Mock(sagemaker_config={})
    trainer = _StubTrainer()

    with pytest.raises(ValueError, match=ATTACH_HINT):
        trainer.show_metrics()
    with pytest.raises(ValueError, match=ATTACH_HINT):
        trainer.stream_logs()


def _serverless_trainer_patches(module, created):
    """Patch the Hub, S3, and service calls of one serverless trainer module.

    Every call ``__init__`` and the serverless ``train()`` path make is patched at the
    name the module imported, so the constructor runs for real and ``train()`` gets to
    ``TrainingJob.create`` without a request.
    """
    options = Mock()
    options.to_dict.return_value = {"learning_rate": "0.001"}
    options.validate_length_constraints.return_value = None
    return [
        patch("sagemaker.train.common_utils.finetune_utils._get_beta_session"),
        patch(f"{module}.validate_data_path_exists"),
        patch(f"{module}.is_multimodal_data", return_value=False),
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
        patch(
            f"{module}._get_fine_tuning_options_and_model_arn", return_value=(options, "arn", False)
        ),
        patch("sagemaker.core.resources.TrainingJob.create", return_value=created),
    ]


@pytest.fixture
def serverless(request):
    """Build one serverless trainer with every patch active for the life of the test.

    Yields ``(trainer, created, order)``. ``created`` is the job ``TrainingJob.create``
    returns, and ``order`` records the ``wait`` call so a test can place ``on_submit``
    against it.
    """
    trainer_cls, module, kwargs = request.param
    created = Mock(spec=TrainingJob)
    created.training_job_name = "test-job-name"
    order = []
    with ExitStack() as stack:
        for active in _serverless_trainer_patches(module, created):
            stack.enter_context(active)
        stack.enter_context(
            patch(
                "sagemaker.train.common_utils.trainer_wait.wait",
                side_effect=lambda job, **_: order.append("wait"),
            )
        )
        trainer = trainer_cls(
            model="test-model",
            training_type=TrainingType.LORA,
            model_package_group="test-group",
            training_dataset="s3://bucket/train",
            **kwargs,
        )
        yield trainer, created, order


@pytest.mark.parametrize("serverless", SERVERLESS_TRAINERS, indirect=True)
def test_serverless_trainers_expose_the_created_job(serverless):
    trainer, created, _ = serverless
    assert trainer.latest_training_job is None

    returned = trainer.train(wait=False)

    assert returned is created
    assert trainer.latest_training_job is created
    assert trainer.latest_training_job.training_job_name == "test-job-name"


@pytest.mark.parametrize("serverless", SERVERLESS_TRAINERS, indirect=True)
def test_serverless_trainers_call_on_submit_before_the_wait(serverless):
    trainer, created, order = serverless

    returned = trainer.train(wait=True, on_submit=lambda job: order.append(job))

    assert order == [created, "wait"]
    assert returned is created
    assert trainer.latest_training_job is created


@pytest.mark.parametrize("serverless", SERVERLESS_TRAINERS, indirect=True)
def test_serverless_trainers_call_on_submit_without_a_wait(serverless):
    trainer, created, order = serverless
    submitted = []

    trainer.train(wait=False, on_submit=submitted.append)

    assert submitted == [created]
    assert order == []


@pytest.mark.parametrize("serverless", SERVERLESS_TRAINERS, indirect=True)
def test_serverless_trainers_skip_on_submit_for_a_dry_run(serverless):
    trainer, _, _ = serverless
    callback = Mock()

    with patch("sagemaker.core.resources.TrainingJob.create") as create:
        assert trainer.train(wait=False, dry_run=True, on_submit=callback) is None

    callback.assert_not_called()
    create.assert_not_called()


@patch("sagemaker.train.model_trainer.TrainingJob")
def test_model_trainer_on_submit_sees_the_job_before_the_wait(mock_training_job, model_trainer):
    created = Mock(spec=TrainingJob)
    created.training_job_name = "created-job"
    mock_training_job.create.return_value = created
    order = []
    created.wait.side_effect = lambda **_: order.append("wait")

    model_trainer.train(wait=True, logs=False, on_submit=lambda job: order.append(job))

    assert order == [created, "wait"]


@patch("sagemaker.train.model_trainer.TrainingJob")
def test_model_trainer_on_submit_runs_without_a_wait(mock_training_job, model_trainer):
    created = Mock(spec=TrainingJob)
    mock_training_job.create.return_value = created
    submitted = []

    returned = model_trainer.train(wait=False, on_submit=submitted.append)

    assert submitted == [created]
    assert returned is created
    created.wait.assert_not_called()


@patch("sagemaker.train.model_trainer.TrainingJob")
def test_model_trainer_dry_run_skips_on_submit(mock_training_job, model_trainer):
    callback = Mock()

    model_trainer.train(wait=False, dry_run=True, on_submit=callback)

    callback.assert_not_called()
    mock_training_job.create.assert_not_called()


@patch("sagemaker.train.model_trainer.TrainingJob")
def test_a_raising_on_submit_keeps_the_job_and_skips_the_wait(mock_training_job, model_trainer):
    """The job is already running, so the handle stays reachable and the error propagates."""
    created = Mock(spec=TrainingJob)
    mock_training_job.create.return_value = created

    def record(job):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        model_trainer.train(wait=True, logs=False, on_submit=record)

    assert model_trainer.latest_training_job is created
    created.wait.assert_not_called()


def _hyperpod_patches():
    """Patch the Hub, IAM, and CLI calls of the HyperPod path, at the names the modules use."""
    return [
        patch("sagemaker.train.common_utils.finetune_utils._get_beta_session"),
        patch("sagemaker.train.sft_trainer._resolve_model_and_name", return_value=("m", "m")),
        patch(
            "sagemaker.train.sft_trainer._get_fine_tuning_options_and_model_arn",
            return_value=(Mock(to_dict=Mock(return_value={})), "arn", False),
        ),
        patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session"),
        patch("sagemaker.train.base_trainer.TrainDefaults.verify_hyperpod_caller_permissions"),
        patch("sagemaker.train.base_trainer.validate_hyperpod_compute"),
        patch("sagemaker.train.base_trainer._get_smhp_replicas_enum", return_value=None),
        patch("sagemaker.train.base_trainer.get_hyperpod_recipe_path", return_value="recipes/test"),
        patch("sagemaker.train.base_trainer.flatten_resolved_recipe", return_value={}),
        patch("sagemaker.train.base_trainer.subprocess"),
    ]


@pytest.fixture
def hyperpod_trainer():
    """An ``SFTTrainer`` on HyperPod compute, built through the real constructor.

    Yields ``(trainer, subprocess)``. The CLI stub answers ``NAME: my-job-123`` to every
    ``hyperpod`` command.
    """
    with ExitStack() as stack:
        mocks = [stack.enter_context(active) for active in _hyperpod_patches()]
        mocks[3].return_value = Mock(sagemaker_config={})
        subprocess = mocks[-1]
        subprocess.run.return_value = SimpleNamespace(stdout="NAME: my-job-123\n", stderr="")
        trainer = SFTTrainer(
            model="amazon.nova-lite-v2",
            compute=HyperPodCompute(
                cluster_name="my-cluster", instance_type="ml.p5.48xlarge", node_count=2
            ),
            training_image=HYPERPOD_IMAGE,
            s3_output_path="s3://bucket/out/",
        )
        stack.enter_context(
            patch.object(trainer, "get_resolved_recipe", return_value={"training_config": {}})
        )
        yield trainer, subprocess


def test_hyperpod_train_returns_the_job_it_hands_to_on_submit(hyperpod_trainer):
    trainer, subprocess = hyperpod_trainer
    submitted = []

    returned = trainer.train(wait=False, on_submit=submitted.append)

    assert isinstance(returned, TrainingJob)
    assert returned.training_job_name == "my-job-123"
    assert submitted == [returned]
    assert trainer.latest_training_job is returned
    start_cmd = subprocess.run.call_args_list[-1].args[0]
    assert start_cmd[:2] == ["hyperpod", "start-job"]
    overrides = json.loads(start_cmd[start_cmd.index("--override-parameters") + 1])
    assert overrides == {"instance_type": "ml.p5.48xlarge", "container": HYPERPOD_IMAGE}


def test_hyperpod_dry_run_returns_none_and_skips_on_submit(hyperpod_trainer):
    trainer, subprocess = hyperpod_trainer
    callback = Mock()

    assert trainer.train(wait=False, dry_run=True, on_submit=callback) is None

    callback.assert_not_called()
    assert trainer.latest_training_job is None
    assert all(call.args[0][1] != "start-job" for call in subprocess.run.call_args_list)


def test_cpt_trainer_train_returns_the_hyperpod_job():
    with ExitStack() as stack:
        stack.enter_context(patch("sagemaker.train.common_utils.finetune_utils._get_beta_session"))
        stack.enter_context(
            patch("sagemaker.train.cpt_trainer._resolve_model_and_name", return_value=("m", "m"))
        )
        trainer = CPTTrainer(
            model="amazon.nova-lite-v2",
            compute=HyperPodCompute(cluster_name="my-cluster", instance_type="ml.p5.48xlarge"),
            training_image=HYPERPOD_IMAGE,
        )
        created = TrainingJob(training_job_name="my-job-123")
        hyperpod = stack.enter_context(
            patch.object(trainer, "_train_hyperpod", return_value=created)
        )
        submitted = []

        returned = trainer.train(wait=False, on_submit=submitted.append)

    assert returned is created
    assert hyperpod.call_args.kwargs["on_submit"] == submitted.append
