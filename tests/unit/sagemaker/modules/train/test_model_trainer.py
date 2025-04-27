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
import yaml
import pytest
from pydantic import ValidationError
from unittest.mock import patch, MagicMock, ANY, mock_open

from sagemaker import image_uris
from sagemaker_core.main.resources import TrainingJob
from sagemaker_core.main.shapes import (
    ResourceConfig,
    VpcConfig,
    AlgorithmSpecification,
)

from sagemaker.config import SAGEMAKER, PYTHON_SDK, MODULES
from sagemaker.config.config_schema import (
    MODEL_TRAINER,
    _simple_path,
    TRAINING_JOB_RESOURCE_CONFIG_PATH,
)
from sagemaker.modules import Session
from sagemaker.modules.train.model_trainer import ModelTrainer, Mode
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
    RemoteDebugConfig,
    TensorBoardOutputConfig,
    InfraCheckConfig,
    SessionChainingConfig,
    InputData,
    Networking,
    TrainingImageConfig,
    TrainingRepositoryAuthConfig,
    CheckpointConfig,
    Tag,
    S3DataSource,
    FileSystemDataSource,
    Channel,
    DataSource,
)
from sagemaker.modules.distributed import Torchrun, SMP, MPI
from sagemaker.modules.train.sm_recipes.utils import _load_recipes_cfg
from sagemaker.modules.templates import EXEUCTE_DISTRIBUTED_DRIVER
from tests.unit import DATA_DIR

DEFAULT_BASE_NAME = "dummy-image-job"
DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_BUCKET_PREFIX = "sample-prefix"
DEFAULT_REGION = "us-west-2"
DEFAULT_SOURCE_DIR = f"{DATA_DIR}/modules/script_mode"
DEFAULT_COMPUTE_CONFIG = Compute(instance_type=DEFAULT_INSTANCE_TYPE, instance_count=1)
DEFAULT_OUTPUT_DATA_CONFIG = OutputDataConfig(
    s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/{DEFAULT_BASE_NAME}",
    compression_type="GZIP",
    kms_key_id=None,
)
DEFAULT_STOPPING_CONDITION = StoppingCondition(
    max_runtime_in_seconds=3600,
    max_pending_time_in_seconds=None,
    max_wait_time_in_seconds=None,
)
DEFAULT_SOURCE_CODE = SourceCode(
    source_dir=DEFAULT_SOURCE_DIR,
    entry_script="custom_script.py",
)
DEFAULT_ENTRYPOINT = ["/bin/bash"]
DEFAULT_ARGUMENTS = [
    "-c",
    (
        "chmod +x /opt/ml/input/data/sm_drivers/sm_train.sh "
        + "&& /opt/ml/input/data/sm_drivers/sm_train.sh"
    ),
]


@pytest.fixture(scope="module", autouse=True)
def modules_session():
    with patch("sagemaker.modules.Session", spec=Session) as session_mock:
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = DEFAULT_BUCKET
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        session_instance.default_bucket_prefix = DEFAULT_BUCKET_PREFIX
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        session_instance.boto_region_name = DEFAULT_REGION
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
                "source_code": SourceCode(
                    entry_script="train.py",
                ),
            },
            "should_throw": True,
        },
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
                "source_code": SourceCode(
                    source_dir="s3://bucket/requirements.txt",
                    entry_script="custom_script.py",
                ),
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
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
                "source_code": SourceCode(
                    source_dir=f"{DEFAULT_SOURCE_DIR}/code.tar.gz",
                    entry_script="custom_script.py",
                ),
            },
            "should_throw": False,
        },
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
                "source_code": SourceCode(
                    source_dir="s3://bucket/code/",
                    entry_script="custom_script.py",
                ),
            },
            "should_throw": False,
        },
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
                "source_code": SourceCode(
                    source_dir="s3://bucket/code/code.tar.gz",
                    entry_script="custom_script.py",
                ),
            },
            "should_throw": False,
        },
    ],
    ids=[
        "no_params",
        "training_image_and_algorithm_name",
        "only_training_image",
        "unsupported_source_code_missing_source_dir",
        "unsupported_source_code_s3_other_file",
        "supported_source_code_local_dir",
        "supported_source_code_local_tar_file",
        "supported_source_code_s3_dir",
        "supported_source_code_s3_tar_file",
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


@pytest.mark.parametrize(
    "default_config",
    [
        {
            "path_name": "sourceCode",
            "config_value": {"command": "echo 'Hello World' && env"},
        },
        {
            "path_name": "compute",
            "config_value": {"volume_size_in_gb": 45},
        },
        {
            "path_name": "networking",
            "config_value": {
                "enable_network_isolation": True,
                "security_group_ids": ["sg-1"],
                "subnets": ["subnet-1"],
            },
        },
        {
            "path_name": "stoppingCondition",
            "config_value": {"max_runtime_in_seconds": 15},
        },
        {
            "path_name": "trainingImageConfig",
            "config_value": {"training_repository_access_mode": "private"},
        },
        {
            "path_name": "outputDataConfig",
            "config_value": {"s3_output_path": "Sample S3 path"},
        },
        {
            "path_name": "checkpointConfig",
            "config_value": {"s3_uri": "sample uri"},
        },
    ],
)
@patch("sagemaker.modules.train.model_trainer.TrainingJob")
@patch("sagemaker.modules.train.model_trainer.resolve_value_from_config")
@patch("sagemaker.modules.train.model_trainer.ModelTrainer.create_input_data_channel")
def test_train_with_intelligent_defaults(
    mock_create_input_data_channel,
    mock_resolve_value_from_config,
    mock_training_job,
    default_config,
    model_trainer,
):
    mock_resolve_value_from_config.side_effect = lambda **kwargs: (
        default_config["config_value"]
        if kwargs["config_path"]
        == _simple_path(SAGEMAKER, PYTHON_SDK, MODULES, MODEL_TRAINER, default_config["path_name"])
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
            training_plan_arn=None,
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
            s3_output_path="s3://"
            "sagemaker-us-west-2"
            "-000000000000/"
            "sample-prefix/"
            "dummy-image-job",
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

    mock_get_input_config.assert_called_once_with(mock_input_data_config, ANY)
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
@patch("sagemaker.modules.train.model_trainer.Session.default_bucket")
def test_create_input_data_channel(mock_default_bucket, mock_upload_data, model_trainer, test_case):
    expected_s3_uri = f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}-job/input/test"
    mock_upload_data.return_value = expected_s3_uri
    mock_default_bucket.return_value = DEFAULT_BUCKET
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
            "expected_template": EXEUCTE_DISTRIBUTED_DRIVER.format(
                driver_name="Torchrun", driver_script="torchrun_driver.py"
            ),
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
            "expected_template": EXEUCTE_DISTRIBUTED_DRIVER.format(
                driver_name="Torchrun", driver_script="torchrun_driver.py"
            ),
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
                mpi_additional_options=["-x", "VAR1", "-x", "VAR2"],
            ),
            "expected_template": EXEUCTE_DISTRIBUTED_DRIVER.format(
                driver_name="MPI", driver_script="mpi_driver.py"
            ),
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
            assert test_case["distributed"].model_dump() == (json.loads(runner_json_content))
        assert os.path.exists(expected_source_code_json_path)
        with open(expected_source_code_json_path, "r") as f:
            source_code_json_content = f.read()
            assert test_case["source_code"].model_dump() == (json.loads(source_code_json_content))
        assert os.path.exists(expected_source_code_json_path)
        with open(expected_source_code_json_path, "r") as f:
            source_code_json_content = f.read()
            assert test_case["source_code"].model_dump() == (json.loads(source_code_json_content))
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


@patch("sagemaker.modules.train.model_trainer._get_unique_name")
@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_model_trainer_full_init(mock_training_job, mock_unique_name, modules_session):
    def mock_upload_data(path, bucket, key_prefix):
        return f"s3://{bucket}/{key_prefix}"

    modules_session.upload_data.side_effect = mock_upload_data

    training_mode = Mode.SAGEMAKER_TRAINING_JOB
    role = DEFAULT_ROLE
    source_code = DEFAULT_SOURCE_CODE
    distributed = Torchrun()
    compute = Compute(
        instance_type=DEFAULT_INSTANCE_TYPE,
        instance_count=1,
        volume_size_in_gb=30,
        volume_kms_key_id="key-id",
        keep_alive_period_in_seconds=3600,
        enable_managed_spot_training=True,
    )
    networking = Networking(
        security_group_ids=["sg-000000000000"],
        subnets=["subnet-000000000000"],
        enable_network_isolation=True,
        enable_inter_container_traffic_encryption=True,
    )
    stopping_condition = DEFAULT_STOPPING_CONDITION
    training_image = DEFAULT_IMAGE
    training_image_config = TrainingImageConfig(
        training_repository_access_mode="Platform",
        training_repository_auth_config=TrainingRepositoryAuthConfig(
            training_repository_credentials_provider_arn="arn:aws:lambda:us-west-2:000000000000:function:dummy-function"
        ),
    )
    output_data_config = DEFAULT_OUTPUT_DATA_CONFIG

    local_input_data = InputData(
        channel_name="train", data_source=f"{DEFAULT_SOURCE_DIR}/data/train"
    )
    s3_data_source_input = InputData(
        channel_name="test",
        data_source=S3DataSource(
            s3_data_type="S3Prefix",
            s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}/data/test",
            s3_data_distribution_type="FullyReplicated",
            attribute_names=["label"],
            instance_group_names=["instance-group"],
        ),
    )
    file_system_input = InputData(
        channel_name="validation",
        data_source=FileSystemDataSource(
            file_system_id="fs-000000000000",
            file_system_access_mode="ro",
            file_system_type="EFS",
            directory_path="/data/validation",
        ),
    )
    input_data_config = [local_input_data, s3_data_source_input, file_system_input]
    checkpoint_config = CheckpointConfig(
        local_path="/opt/ml/checkpoints",
        s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}/checkpoints",
    )
    training_input_mode = "File"
    environment = {"ENV_VAR": "value"}
    hyperparameters = {"key": "value"}
    tags = [Tag(key="key", value="value")]

    model_trainer = ModelTrainer(
        training_mode=training_mode,
        sagemaker_session=modules_session,
        role=role,
        source_code=source_code,
        distributed=distributed,
        compute=compute,
        networking=networking,
        stopping_condition=stopping_condition,
        training_image=training_image,
        training_image_config=training_image_config,
        output_data_config=output_data_config,
        input_data_config=input_data_config,
        checkpoint_config=checkpoint_config,
        training_input_mode=training_input_mode,
        environment=environment,
        hyperparameters=hyperparameters,
        tags=tags,
    )

    assert model_trainer.training_mode == training_mode
    assert model_trainer.sagemaker_session == modules_session
    assert model_trainer.role == role
    assert model_trainer.source_code == source_code
    assert model_trainer.distributed == distributed
    assert model_trainer.compute == compute
    assert model_trainer.networking == networking
    assert model_trainer.stopping_condition == stopping_condition
    assert model_trainer.training_image == training_image
    assert model_trainer.training_image_config == training_image_config
    assert model_trainer.output_data_config == output_data_config
    assert model_trainer.input_data_config == input_data_config
    assert model_trainer.checkpoint_config == checkpoint_config
    assert model_trainer.training_input_mode == training_input_mode
    assert model_trainer.environment == environment
    assert model_trainer.hyperparameters == hyperparameters
    assert model_trainer.tags == tags

    unique_name = "training-job"
    mock_unique_name.return_value = unique_name

    model_trainer.train()

    mock_training_job.create.assert_called_once_with(
        training_job_name=unique_name,
        algorithm_specification=AlgorithmSpecification(
            training_input_mode=training_input_mode,
            training_image=training_image,
            algorithm_name=None,
            container_entrypoint=DEFAULT_ENTRYPOINT,
            container_arguments=DEFAULT_ARGUMENTS,
            training_image_config=training_image_config,
        ),
        hyper_parameters=hyperparameters,
        input_data_config=[
            Channel(
                channel_name=local_input_data.channel_name,
                data_source=DataSource(
                    s3_data_source=S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/{DEFAULT_BASE_NAME}/{unique_name}/input/train",  # noqa: E501
                        s3_data_distribution_type="FullyReplicated",
                    )
                ),
                input_mode="File",
            ),
            Channel(
                channel_name=s3_data_source_input.channel_name,
                data_source=DataSource(s3_data_source=s3_data_source_input.data_source),
            ),
            Channel(
                channel_name=file_system_input.channel_name,
                data_source=DataSource(file_system_data_source=file_system_input.data_source),
            ),
            Channel(
                channel_name="code",
                data_source=DataSource(
                    s3_data_source=S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/{DEFAULT_BASE_NAME}/{unique_name}/input/code",  # noqa: E501
                        s3_data_distribution_type="FullyReplicated",
                    )
                ),
                input_mode="File",
            ),
            Channel(
                channel_name="sm_drivers",
                data_source=DataSource(
                    s3_data_source=S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BUCKET_PREFIX}/{DEFAULT_BASE_NAME}/{unique_name}/input/sm_drivers",  # noqa: E501
                        s3_data_distribution_type="FullyReplicated",
                    ),
                ),
                input_mode="File",
            ),
        ],
        resource_config=ResourceConfig(
            instance_type=compute.instance_type,
            instance_count=compute.instance_count,
            volume_size_in_gb=compute.volume_size_in_gb,
            volume_kms_key_id=compute.volume_kms_key_id,
            keep_alive_period_in_seconds=compute.keep_alive_period_in_seconds,
            instance_groups=None,
            training_plan_arn=None,
        ),
        vpc_config=VpcConfig(
            security_group_ids=networking.security_group_ids,
            subnets=networking.subnets,
        ),
        session=ANY,
        role_arn=role,
        tags=tags,
        stopping_condition=stopping_condition,
        output_data_config=output_data_config,
        checkpoint_config=checkpoint_config,
        environment=environment,
        enable_managed_spot_training=compute.enable_managed_spot_training,
        enable_inter_container_traffic_encryption=(
            networking.enable_inter_container_traffic_encryption
        ),
        enable_network_isolation=networking.enable_network_isolation,
        remote_debug_config=None,
        tensor_board_output_config=None,
        retry_strategy=None,
        infra_check_config=None,
        session_chaining_config=None,
    )


def test_model_trainer_gpu_recipe_full_init(modules_session):
    training_recipe = "training/llama/p4_hf_llama3_70b_seq8k_gpu"
    recipe_overrides = {"run": {"results_dir": "/opt/ml/model"}}
    compute = Compute(instance_type="ml.p4d.24xlarge", instance_count="2")

    gpu_image_cfg = _load_recipes_cfg().get("gpu_image")
    if isinstance(gpu_image_cfg, str):
        expected_training_image = gpu_image_cfg
    else:
        expected_training_image = image_uris.retrieve(
            gpu_image_cfg.get("framework"),
            region=modules_session.boto_region_name,
            version=gpu_image_cfg.get("version"),
            image_scope="training",
            **gpu_image_cfg.get("additional_args"),
        )

    expected_distributed = Torchrun(smp=SMP(random_seed=123456))
    expected_hyperparameters = {"config-path": ".", "config-name": "recipe.yaml"}

    networking = Networking(
        security_group_ids=["sg-000000000000"],
        subnets=["subnet-000000000000"],
        enable_network_isolation=True,
        enable_inter_container_traffic_encryption=True,
    )
    stopping_condition = DEFAULT_STOPPING_CONDITION
    output_data_config = DEFAULT_OUTPUT_DATA_CONFIG
    local_input_data = InputData(
        channel_name="train", data_source=f"{DEFAULT_SOURCE_DIR}/data/train"
    )
    input_data_config = [local_input_data]
    checkpoint_config = CheckpointConfig(
        local_path="/opt/ml/checkpoints",
        s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}/checkpoints",
    )
    training_input_mode = "File"
    environment = {"ENV_VAR": "value"}
    tags = [Tag(key="key", value="value")]
    requirements = f"{DEFAULT_SOURCE_DIR}/requirements.txt"

    model_trainer = ModelTrainer.from_recipe(
        training_recipe=training_recipe,
        recipe_overrides=recipe_overrides,
        compute=compute,
        networking=networking,
        stopping_condition=stopping_condition,
        requirements=requirements,
        output_data_config=output_data_config,
        input_data_config=input_data_config,
        checkpoint_config=checkpoint_config,
        training_input_mode=training_input_mode,
        environment=environment,
        tags=tags,
        sagemaker_session=modules_session,
        role=DEFAULT_ROLE,
        base_job_name=DEFAULT_BASE_NAME,
    )

    assert model_trainer.training_image == expected_training_image
    assert model_trainer.distributed == expected_distributed
    assert model_trainer.hyperparameters == expected_hyperparameters
    assert model_trainer.source_code is not None
    assert model_trainer.source_code.requirements == "requirements.txt"

    assert model_trainer.compute == compute
    assert model_trainer.networking == networking
    assert model_trainer.stopping_condition == stopping_condition
    assert model_trainer.output_data_config == output_data_config
    assert model_trainer.input_data_config == input_data_config
    assert model_trainer.checkpoint_config == checkpoint_config
    assert model_trainer.training_input_mode == training_input_mode
    assert model_trainer.environment == environment
    assert model_trainer.tags == tags


@patch("sagemaker.modules.train.model_trainer._LocalContainer")
@patch("sagemaker.modules.train.model_trainer._get_unique_name")
@patch("sagemaker.modules.local_core.local_container.download_folder")
def test_model_trainer_local_full_init(
    mock_download_folder, mock_unique_name, mock_local_container, modules_session
):
    def mock_upload_data(path, bucket, key_prefix):
        return f"s3://{bucket}/{key_prefix}"

    modules_session.upload_data.side_effect = mock_upload_data
    mock_download_folder.return_value = f"{DEFAULT_SOURCE_DIR}/data/test"
    mock_local_container.train.return_value = None

    training_mode = Mode.LOCAL_CONTAINER
    role = DEFAULT_ROLE
    source_code = DEFAULT_SOURCE_CODE
    distributed = Torchrun()
    compute = Compute(
        instance_type=DEFAULT_INSTANCE_TYPE,
        instance_count=1,
        volume_size_in_gb=30,
        volume_kms_key_id="key-id",
        keep_alive_period_in_seconds=3600,
        enable_managed_spot_training=True,
    )
    networking = Networking(
        security_group_ids=["sg-000000000000"],
        subnets=["subnet-000000000000"],
        enable_network_isolation=True,
        enable_inter_container_traffic_encryption=True,
    )
    stopping_condition = DEFAULT_STOPPING_CONDITION
    training_image = DEFAULT_IMAGE
    training_image_config = TrainingImageConfig(
        training_repository_access_mode="Platform",
        training_repository_auth_config=TrainingRepositoryAuthConfig(
            training_repository_credentials_provider_arn="arn:aws:lambda:us-west-2:000000000000:function:dummy-function"
        ),
    )
    output_data_config = DEFAULT_OUTPUT_DATA_CONFIG

    local_input_data = InputData(
        channel_name="train", data_source=f"{DEFAULT_SOURCE_DIR}/data/train"
    )
    s3_data_source_input = InputData(
        channel_name="test",
        data_source=S3DataSource(
            s3_data_type="S3Prefix",
            s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}/data/test",
            s3_data_distribution_type="FullyReplicated",
            attribute_names=["label"],
            instance_group_names=["instance-group"],
        ),
    )
    file_system_input = InputData(
        channel_name="validation",
        data_source=FileSystemDataSource(
            file_system_id="fs-000000000000",
            file_system_access_mode="ro",
            file_system_type="EFS",
            directory_path="/data/validation",
        ),
    )
    input_data_config = [local_input_data, s3_data_source_input, file_system_input]
    checkpoint_config = CheckpointConfig(
        local_path="/opt/ml/checkpoints",
        s3_uri=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}/checkpoints",
    )
    training_input_mode = "File"
    environment = {"ENV_VAR": "value"}
    hyperparameters = {"key": "value"}
    tags = [Tag(key="key", value="value")]

    local_container_root = os.getcwd()

    model_trainer = ModelTrainer(
        training_mode=training_mode,
        sagemaker_session=modules_session,
        role=role,
        source_code=source_code,
        distributed=distributed,
        compute=compute,
        networking=networking,
        stopping_condition=stopping_condition,
        training_image=training_image,
        training_image_config=training_image_config,
        output_data_config=output_data_config,
        input_data_config=input_data_config,
        checkpoint_config=checkpoint_config,
        training_input_mode=training_input_mode,
        environment=environment,
        hyperparameters=hyperparameters,
        tags=tags,
        local_container_root=local_container_root,
    )

    assert model_trainer.training_mode == training_mode
    assert model_trainer.sagemaker_session == modules_session
    assert model_trainer.role == role
    assert model_trainer.source_code == source_code
    assert model_trainer.distributed == distributed
    assert model_trainer.compute == compute
    assert model_trainer.networking == networking
    assert model_trainer.stopping_condition == stopping_condition
    assert model_trainer.training_image == training_image
    assert model_trainer.training_image_config == training_image_config
    assert model_trainer.output_data_config == output_data_config
    assert model_trainer.input_data_config == input_data_config
    assert model_trainer.checkpoint_config == checkpoint_config
    assert model_trainer.training_input_mode == training_input_mode
    assert model_trainer.environment == environment
    assert model_trainer.hyperparameters == hyperparameters
    assert model_trainer.tags == tags

    unique_name = "training-job"
    mock_unique_name.return_value = unique_name

    model_trainer.train()

    mock_local_container.assert_called_once_with(
        training_job_name=unique_name,
        instance_type=compute.instance_type,
        instance_count=compute.instance_count,
        image=training_image,
        container_root=local_container_root,
        sagemaker_session=modules_session,
        container_entrypoint=DEFAULT_ENTRYPOINT,
        container_arguments=DEFAULT_ARGUMENTS,
        input_data_config=ANY,
        hyper_parameters=hyperparameters,
        environment=environment,
    )


def test_safe_configs():
    # Test extra fails
    with pytest.raises(ValueError):
        SourceCode(entry_point="train.py")
    # Test invalid type fails
    with pytest.raises(ValueError):
        SourceCode(entry_script=1)


@patch("sagemaker.modules.train.model_trainer.TemporaryDirectory")
def test_destructor_cleanup(mock_tmp_dir, modules_session):

    with pytest.raises(ValidationError):
        model_trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            sagemaker_session=modules_session,
            compute="test",
        )
    mock_tmp_dir.cleanup.assert_not_called()

    model_trainer = ModelTrainer(
        training_image=DEFAULT_IMAGE,
        role=DEFAULT_ROLE,
        sagemaker_session=modules_session,
        compute=DEFAULT_COMPUTE_CONFIG,
    )
    model_trainer._temp_recipe_train_dir = mock_tmp_dir
    mock_tmp_dir.assert_not_called()
    del model_trainer
    mock_tmp_dir.cleanup.assert_called_once()


@patch("os.path.exists")
def test_hyperparameters_valid_json(mock_exists, modules_session):
    mock_exists.return_value = True
    expected_hyperparameters = {"param1": "value1", "param2": 2}
    mock_file_open = mock_open(read_data=json.dumps(expected_hyperparameters))

    with patch("builtins.open", mock_file_open):
        model_trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            sagemaker_session=modules_session,
            compute=DEFAULT_COMPUTE_CONFIG,
            hyperparameters="hyperparameters.json",
        )
        assert model_trainer.hyperparameters == expected_hyperparameters
        mock_file_open.assert_called_once_with("hyperparameters.json", "r")
        mock_exists.assert_called_once_with("hyperparameters.json")


@patch("os.path.exists")
def test_hyperparameters_valid_yaml(mock_exists, modules_session):
    mock_exists.return_value = True
    expected_hyperparameters = {"param1": "value1", "param2": 2}
    mock_file_open = mock_open(read_data=yaml.dump(expected_hyperparameters))

    with patch("builtins.open", mock_file_open):
        model_trainer = ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            sagemaker_session=modules_session,
            compute=DEFAULT_COMPUTE_CONFIG,
            hyperparameters="hyperparameters.yaml",
        )
        assert model_trainer.hyperparameters == expected_hyperparameters
        mock_file_open.assert_called_once_with("hyperparameters.yaml", "r")
        mock_exists.assert_called_once_with("hyperparameters.yaml")


def test_hyperparameters_not_exist(modules_session):
    with pytest.raises(ValueError):
        ModelTrainer(
            training_image=DEFAULT_IMAGE,
            role=DEFAULT_ROLE,
            sagemaker_session=modules_session,
            compute=DEFAULT_COMPUTE_CONFIG,
            hyperparameters="nonexistent.json",
        )


@patch("os.path.exists")
def test_hyperparameters_invalid(mock_exists, modules_session):
    mock_exists.return_value = True

    # YAML contents must be a valid mapping
    mock_file_open = mock_open(read_data="- item1\n- item2")
    with patch("builtins.open", mock_file_open):
        with pytest.raises(ValueError, match="Must be a valid JSON or YAML file."):
            ModelTrainer(
                training_image=DEFAULT_IMAGE,
                role=DEFAULT_ROLE,
                sagemaker_session=modules_session,
                compute=DEFAULT_COMPUTE_CONFIG,
                hyperparameters="hyperparameters.yaml",
            )

    # YAML contents must be a valid mapping
    mock_file_open = mock_open(read_data="invalid")
    with patch("builtins.open", mock_file_open):
        with pytest.raises(ValueError, match="Must be a valid JSON or YAML file."):
            ModelTrainer(
                training_image=DEFAULT_IMAGE,
                role=DEFAULT_ROLE,
                sagemaker_session=modules_session,
                compute=DEFAULT_COMPUTE_CONFIG,
                hyperparameters="hyperparameters.yaml",
            )

    # Must be valid YAML
    mock_file_open = mock_open(read_data="* invalid")
    with patch("builtins.open", mock_file_open):
        with pytest.raises(ValueError, match="Must be a valid JSON or YAML file."):
            ModelTrainer(
                training_image=DEFAULT_IMAGE,
                role=DEFAULT_ROLE,
                sagemaker_session=modules_session,
                compute=DEFAULT_COMPUTE_CONFIG,
                hyperparameters="hyperparameters.yaml",
            )
