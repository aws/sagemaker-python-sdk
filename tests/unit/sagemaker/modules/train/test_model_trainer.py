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
"""ModelTrainer Tests."""
from __future__ import absolute_import

import shutil
import tempfile
import json
import os
import pytest
from unittest.mock import patch, MagicMock, ANY

from sagemaker_core.main.resources import TrainingJob
from sagemaker_core.main.shapes import ResourceConfig

from sagemaker.config import SAGEMAKER, PYTHON_SDK, MODULES
from sagemaker.config.config_schema import (
    MODEL_TRAINER,
    _simple_path,
    TRAINING_JOB_RESOURCE_CONFIG_PATH,
)
from sagemaker.modules import Session
from sagemaker.modules.train.model_trainer import ModelTrainer
from sagemaker.modules.constants import (
    DEFAULT_INSTANCE_TYPE,
    DISTRIBUTED_JSON,
    SOURCE_CODE_JSON,
    TRAIN_SCRIPT,
)
from sagemaker.modules.configs import (
    Compute,
    StoppingCondition,
    RetryStrategy,
    OutputDataConfig,
    SourceCode,
    S3DataSource,
    FileSystemDataSource,
    RemoteDebugConfig,
    TensorBoardOutputConfig,
    InfraCheckConfig,
    SessionChainingConfig,
    InputData,
)
from sagemaker.modules.distributed import Torchrun, SMP, MPI
from sagemaker.modules.templates import EXEUCTE_TORCHRUN_DRIVER, EXECUTE_MPI_DRIVER
from tests.unit import DATA_DIR

DEFAULT_BASE_NAME = "dummy-image-job"
DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_COMPUTE_CONFIG = Compute(instance_type=DEFAULT_INSTANCE_TYPE, instance_count=1)
DEFAULT_OUTPUT_DATA_CONFIG = OutputDataConfig(
    s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}",
    compression_type="GZIP",
    kms_key_id=None,
)
DEFAULT_STOPPING_CONDITION = StoppingCondition(
    max_runtime_in_seconds=3600,
    max_pending_time_in_seconds=None,
    max_wait_time_in_seconds=None,
)
DEFAULT_SOURCE_CODE = SourceCode(
    source_dir=f"{DATA_DIR}/modules/script_mode",
    entry_script="custom_script.py",
)
UNSUPPORTED_SOURCE_CODE = SourceCode(
    entry_script="train.py",
)


@pytest.fixture(scope="module", autouse=True)
def modules_session():
    with patch("sagemaker.modules.Session", spec=Session) as session_mock:
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = DEFAULT_BUCKET
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        yield session_instance


@pytest.fixture
def model_trainer():
    trainer = ModelTrainer(
        training_image=DEFAULT_IMAGE,
        role=DEFAULT_ROLE,
        compute=DEFAULT_COMPUTE_CONFIG,
        stopping_condition=DEFAULT_STOPPING_CONDITION,
        output_data_config=DEFAULT_OUTPUT_DATA_CONFIG,
    )
    return trainer


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "init_params": {},
            "should_throw": True,
        },
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
            },
            "should_throw": False,
        },
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
                "algorithm_name": "dummy-arn",
            },
            "should_throw": True,
        },
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
                "source_code": UNSUPPORTED_SOURCE_CODE,
            },
            "should_throw": True,
        },
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
                "source_code": DEFAULT_SOURCE_CODE,
            },
            "should_throw": False,
        },
    ],
    ids=[
        "no_params",
        "training_image_and_algorithm_name",
        "only_training_image",
        "unsupported_source_code",
        "supported_source_code",
    ],
)
def test_model_trainer_param_validation(test_case, modules_session):
    if test_case["should_throw"]:
        with pytest.raises(ValueError):
            ModelTrainer(**test_case["init_params"], sagemaker_session=modules_session)
    else:
        trainer = ModelTrainer(**test_case["init_params"], sagemaker_session=modules_session)
        assert trainer is not None
        assert trainer.training_image == DEFAULT_IMAGE
        assert trainer.compute == DEFAULT_COMPUTE_CONFIG
        assert trainer.output_data_config == DEFAULT_OUTPUT_DATA_CONFIG
        assert trainer.stopping_condition == DEFAULT_STOPPING_CONDITION
        assert trainer.base_job_name == DEFAULT_BASE_NAME


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_train_with_default_params(mock_training_job, model_trainer):
    model_trainer.train()

    mock_training_job.create.assert_called_once()

    training_job_instance = mock_training_job.create.return_value
    training_job_instance.wait.assert_called_once_with(logs=True)


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
@patch("sagemaker.modules.train.model_trainer.resolve_value_from_config")
def test_train_with_intelligent_defaults(
    mock_resolve_value_from_config, mock_training_job, model_trainer
):
    source_code_path = _simple_path(SAGEMAKER, PYTHON_SDK, MODULES, MODEL_TRAINER, "sourceCode")

    mock_resolve_value_from_config.side_effect = lambda **kwargs: (
        {"command": "echo 'Hello World' && env"}
        if kwargs["config_path"] == source_code_path
        else None
    )

    model_trainer.train()

    mock_training_job.create.assert_called_once()

    training_job_instance = mock_training_job.create.return_value
    training_job_instance.wait.assert_called_once_with(logs=True)


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
@patch("sagemaker.modules.train.model_trainer.resolve_value_from_config")
def test_train_with_intelligent_defaults_training_job_space(
    mock_resolve_value_from_config, mock_training_job, model_trainer
):
    mock_resolve_value_from_config.side_effect = lambda **kwargs: (
        {
            "instanceType": DEFAULT_INSTANCE_TYPE,
            "instanceCount": 1,
            "volumeSizeInGB": 30,
        }
        if kwargs["config_path"] == TRAINING_JOB_RESOURCE_CONFIG_PATH
        else None
    )

    model_trainer.train()

    mock_training_job.create.assert_called_once_with(
        training_job_name=ANY,
        algorithm_specification=ANY,
        hyper_parameters={},
        input_data_config=[],
        resource_config=ResourceConfig(
            volume_size_in_gb=30,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            volume_kms_key_id=None,
            keep_alive_period_in_seconds=None,
            instance_groups=None,
        ),
        vpc_config=None,
        session=ANY,
        role_arn="arn:aws:iam::000000000000:" "role/test-role",
        tags=None,
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=3600,
            max_wait_time_in_seconds=None,
            max_pending_time_in_seconds=None,
        ),
        output_data_config=OutputDataConfig(
            s3_output_path="s3://" "sagemaker-us-west-2" "-000000000000/d" "ummy-image-job",
            kms_key_id=None,
            compression_type="GZIP",
        ),
        checkpoint_config=None,
        environment=None,
        enable_managed_spot_training=None,
        enable_inter_container_traffic_encryption=None,
        enable_network_isolation=None,
        remote_debug_config=None,
        tensor_board_output_config=None,
        retry_strategy=None,
        infra_check_config=None,
        session_chaining_config=None,
    )

    training_job_instance = mock_training_job.create.return_value
    training_job_instance.wait.assert_called_once_with(logs=True)


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
@patch.object(ModelTrainer, "_get_input_data_config")
def test_train_with_input_data_channels(mock_get_input_config, mock_training_job, model_trainer):
    train_data = InputData(channel_name="train", data_source="train/dir")
    test_data = InputData(channel_name="test", data_source="test/dir")
    mock_input_data_config = [train_data, test_data]

    model_trainer.train(input_data_config=mock_input_data_config)

    mock_get_input_config.assert_called_once_with(mock_input_data_config)
    mock_training_job.create.assert_called_once()


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "channel_name": "test",
            "data_source": DATA_DIR,
            "valid": True,
        },
        {
            "channel_name": "test",
            "data_source": f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}-job/input/test",
            "valid": True,
        },
        {
            "channel_name": "test",
            "data_source": S3DataSource(
                s3_data_type="S3Prefix",
                s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}-job/input/test",
                s3_data_distribution_type="FullyReplicated",
            ),
            "valid": True,
        },
        {
            "channel_name": "test",
            "data_source": FileSystemDataSource(
                file_system_id="fs-000000000000",
                file_system_access_mode="ro",
                file_system_type="EFS",
                directory_path="/data/test",
            ),
            "valid": True,
        },
        {
            "channel_name": "test",
            "data_source": "fake/path",
            "valid": False,
        },
    ],
    ids=[
        "valid_local_path",
        "valid_s3_path",
        "valid_s3_data_source",
        "valid_file_system_data_source",
        "invalid_path",
    ],
)
@patch("sagemaker.modules.train.model_trainer.Session.upload_data")
def test_create_input_data_channel(mock_upload_data, model_trainer, test_case):
    expected_s3_uri = f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}-job/input/test"
    mock_upload_data.return_value = expected_s3_uri

    if not test_case["valid"]:
        with pytest.raises(ValueError):
            model_trainer.create_input_data_channel(
                test_case["channel_name"], test_case["data_source"]
            )
    else:
        channel = model_trainer.create_input_data_channel(
            test_case["channel_name"], test_case["data_source"]
        )
        assert channel.channel_name == test_case["channel_name"]

        if isinstance(test_case["data_source"], S3DataSource):
            assert channel.data_source.s3_data_source == test_case["data_source"]
        elif isinstance(test_case["data_source"], FileSystemDataSource):
            assert channel.data_source.file_system_data_source == test_case["data_source"]
        else:
            assert channel.data_source.s3_data_source.s3_uri == expected_s3_uri


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "source_code": DEFAULT_SOURCE_CODE,
            "distributed": Torchrun(),
            "expected_template": EXEUCTE_TORCHRUN_DRIVER,
            "expected_hyperparameters": {},
        },
        {
            "source_code": DEFAULT_SOURCE_CODE,
            "distributed": Torchrun(
                smp=SMP(
                    hybrid_shard_degree=3,
                    sm_activation_offloading=True,
                    allow_empty_shards=True,
                    tensor_parallel_degree=5,
                )
            ),
            "expected_template": EXEUCTE_TORCHRUN_DRIVER,
            "expected_hyperparameters": {
                "mp_parameters": json.dumps(
                    {
                        "hybrid_shard_degree": 3,
                        "sm_activation_offloading": True,
                        "allow_empty_shards": True,
                        "tensor_parallel_degree": 5,
                    }
                ),
            },
        },
        {
            "source_code": DEFAULT_SOURCE_CODE,
            "distributed": MPI(
                custom_mpi_options=["-x", "VAR1", "-x", "VAR2"],
            ),
            "expected_template": EXECUTE_MPI_DRIVER,
            "expected_hyperparameters": {},
        },
    ],
    ids=[
        "torchrun",
        "torchrun_smp",
        "mpi",
    ],
)
@patch("sagemaker.modules.train.model_trainer.TrainingJob")
@patch("sagemaker.modules.train.model_trainer.TemporaryDirectory")
@patch("sagemaker.modules.train.model_trainer.resolve_value_from_config")
def test_train_with_distributed_config(
    mock_resolve_value_from_config,
    mock_tmp_dir,
    mock_training_job,
    test_case,
    request,
    modules_session,
):
    mock_resolve_value_from_config.return_value = None
    modules_session.upload_data.return_value = (
        f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}-job/input/test"
    )

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_dir._cleanup = False
    tmp_dir.cleanup = lambda: None
    mock_tmp_dir.return_value = tmp_dir

    expected_train_script_path = os.path.join(tmp_dir.name, TRAIN_SCRIPT)
    expected_runner_json_path = os.path.join(tmp_dir.name, DISTRIBUTED_JSON)
    expected_source_code_json_path = os.path.join(tmp_dir.name, SOURCE_CODE_JSON)

    try:
        model_trainer = ModelTrainer(
            sagemaker_session=modules_session,
            training_image=DEFAULT_IMAGE,
            source_code=test_case["source_code"],
            distributed=test_case["distributed"],
        )

        model_trainer.train()
        mock_training_job.create.assert_called_once()
        assert mock_training_job.create.call_args.kwargs["hyper_parameters"] == (
            test_case["expected_hyperparameters"]
        )

        assert os.path.exists(expected_train_script_path)
        with open(expected_train_script_path, "r") as f:
            train_script_content = f.read()
            assert test_case["expected_template"] in train_script_content

        assert os.path.exists(expected_runner_json_path)
        with open(expected_runner_json_path, "r") as f:
            runner_json_content = f.read()
            assert test_case["distributed"].model_dump(exclude_none=True) == (
                json.loads(runner_json_content)
            )
        assert os.path.exists(expected_source_code_json_path)
        with open(expected_source_code_json_path, "r") as f:
            source_code_json_content = f.read()
            assert test_case["source_code"].model_dump(exclude_none=True) == (
                json.loads(source_code_json_content)
            )
        assert os.path.exists(expected_source_code_json_path)
        with open(expected_source_code_json_path, "r") as f:
            source_code_json_content = f.read()
            assert test_case["source_code"].model_dump(exclude_none=True) == (
                json.loads(source_code_json_content)
            )
    finally:
        shutil.rmtree(tmp_dir.name)
        assert not os.path.exists(tmp_dir.name)


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_train_stores_created_training_job(mock_training_job, model_trainer):
    mock_training_job.create.return_value = TrainingJob(training_job_name="Created-job")
    model_trainer.train(wait=False)
    assert model_trainer._latest_training_job is not None
    assert model_trainer._latest_training_job == TrainingJob(training_job_name="Created-job")


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_tensorboard_output_config(mock_training_job, modules_session):
    image_uri = DEFAULT_IMAGE
    role = DEFAULT_ROLE
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}",
        local_path="/opt/ml/output/tensorboard",
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        sagemaker_session=modules_session,
        role=role,
    ).with_tensorboard_output_config(tensorboard_output_config)

    assert model_trainer._tensorboard_output_config == tensorboard_output_config

    with patch("sagemaker.modules.train.model_trainer.Session.upload_data") as mock_upload_data:
        mock_upload_data.return_value = "s3://dummy-bucket/dummy-prefix"
        model_trainer.train()

        mock_training_job.create.assert_called_once()
        assert (
            mock_training_job.create.call_args.kwargs["tensor_board_output_config"]
            == tensorboard_output_config
        )


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_retry_strategy(mock_training_job, modules_session):
    image_uri = DEFAULT_IMAGE
    role = DEFAULT_ROLE
    retry_strategy = RetryStrategy(
        maximum_retry_attempts=3,
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        sagemaker_session=modules_session,
        role=role,
    ).with_retry_strategy(retry_strategy)

    assert model_trainer._retry_strategy == retry_strategy

    with patch("sagemaker.modules.train.model_trainer.Session.upload_data") as mock_upload_data:
        mock_upload_data.return_value = "s3://dummy-bucket/dummy-prefix"
        model_trainer.train()

        mock_training_job.create.assert_called_once()
        assert mock_training_job.create.call_args.kwargs["retry_strategy"] == retry_strategy


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_infra_check_config(mock_training_job, modules_session):
    image_uri = DEFAULT_IMAGE
    role = DEFAULT_ROLE
    infra_check_config = InfraCheckConfig(
        enable_infra_check=True,
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        sagemaker_session=modules_session,
        role=role,
    ).with_infra_check_config(infra_check_config)

    assert model_trainer._infra_check_config == infra_check_config

    with patch("sagemaker.modules.train.model_trainer.Session.upload_data") as mock_upload_data:
        mock_upload_data.return_value = "s3://dummy-bucket/dummy-prefix"
        model_trainer.train()

        mock_training_job.create.assert_called_once()
        assert mock_training_job.create.call_args.kwargs["infra_check_config"] == infra_check_config


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_session_chaining_config(mock_training_job, modules_session):
    image_uri = DEFAULT_IMAGE
    role = DEFAULT_ROLE
    session_chaining_config = SessionChainingConfig(
        enable_session_tag_chaining=True,
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        sagemaker_session=modules_session,
        role=role,
    ).with_session_chaining_config(session_chaining_config)

    assert model_trainer._session_chaining_config == session_chaining_config

    with patch("sagemaker.modules.train.model_trainer.Session.upload_data") as mock_upload_data:
        mock_upload_data.return_value = "s3://dummy-bucket/dummy-prefix"
        model_trainer.train()

        mock_training_job.create.assert_called_once()
        assert (
            mock_training_job.create.call_args.kwargs["session_chaining_config"]
            == session_chaining_config
        )


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_remote_debug_config(mock_training_job, modules_session):
    image_uri = DEFAULT_IMAGE
    role = DEFAULT_ROLE
    remote_debug_config = RemoteDebugConfig(
        enable_remote_debug=True,
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        sagemaker_session=modules_session,
        role=role,
    ).with_remote_debug_config(remote_debug_config)

    assert model_trainer._remote_debug_config == remote_debug_config

    with patch("sagemaker.modules.train.model_trainer.Session.upload_data") as mock_upload_data:
        mock_upload_data.return_value = "s3://dummy-bucket/dummy-prefix"
        model_trainer.train()

        mock_training_job.create.assert_called_once()
        assert (
            mock_training_job.create.call_args.kwargs["remote_debug_config"] == remote_debug_config
        )
