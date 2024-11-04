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

import pytest
from unittest.mock import patch, MagicMock

from sagemaker.session import Session
from sagemaker.modules.train.model_trainer import ModelTrainer
from sagemaker.modules.constants import DEFAULT_INSTANCE_TYPE
from sagemaker.modules.configs import (
    ResourceConfig,
    StoppingCondition,
    OutputDataConfig,
    SourceCodeConfig,
    S3DataSource,
    FileSystemDataSource,
)
from tests.unit import DATA_DIR

DEFAULT_BASE_NAME = "dummy-image-job"
DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_RESOURCE_CONFIG = ResourceConfig(
    instance_count=1,
    instance_type=DEFAULT_INSTANCE_TYPE,
    volume_size_in_gb=30,
    volume_kms_key_id=None,
    keep_alive_period_in_seconds=None,
    instance_groups=None,
)
DEFAULT_OUTPUT_DATA_CONFIG = OutputDataConfig(
    s3_output_path=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE_NAME}/output",
    compression_type="NONE",
    kms_key_id=None,
)
DEFAULT_STOPPING_CONDITION = StoppingCondition(
    max_runtime_in_seconds=3600,
    max_pending_time_in_seconds=None,
    max_wait_time_in_seconds=None,
)
DEFAULT_SOURCE_CODE_CONFIG = SourceCodeConfig(
    source_dir="test-data",
    entry_point="train.py",
)
UNSUPPORTED_SOURCE_CODE_CONFIG = SourceCodeConfig(
    entry_script="train.py",
)


@pytest.fixture(scope="module", autouse=True)
def modules_session():
    with patch("sagemaker.session.Session", spec=Session) as session_mock:
        session_instance = session_mock.return_value
        session_instance.default_bucket.return_value = DEFAULT_BUCKET
        session_instance.get_caller_identity_arn.return_value = DEFAULT_ROLE
        yield session_instance


@pytest.fixture
def model_trainer():
    trainer = ModelTrainer(
        training_image=DEFAULT_IMAGE,
        role=DEFAULT_ROLE,
        resource_config=MagicMock(spec=ResourceConfig),
        stopping_condition=MagicMock(spec=StoppingCondition),
        output_data_config=MagicMock(spec=OutputDataConfig),
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
                "source_code_config": UNSUPPORTED_SOURCE_CODE_CONFIG,
            },
            "should_throw": True,
        },
        {
            "init_params": {
                "training_image": DEFAULT_IMAGE,
                "source_code_config": DEFAULT_SOURCE_CODE_CONFIG,
            },
            "should_throw": False,
        },
    ],
    ids=[
        "no_params",
        "training_image_and_algorithm_name",
        "only_training_image",
        "unsupported_source_code_config",
        "supported_source_code_config",
    ],
)
def test_model_trainer_param_validation(test_case, modules_session):
    if test_case["should_throw"]:
        with pytest.raises(ValueError):
            ModelTrainer(**test_case["init_params"], session=modules_session)
    else:
        trainer = ModelTrainer(**test_case["init_params"], session=modules_session)
        assert trainer is not None
        assert trainer.training_image == DEFAULT_IMAGE
        assert trainer.resource_config == DEFAULT_RESOURCE_CONFIG
        assert trainer.output_data_config == DEFAULT_OUTPUT_DATA_CONFIG
        assert trainer.stopping_condition == DEFAULT_STOPPING_CONDITION
        assert trainer.base_name == DEFAULT_BASE_NAME


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_train_with_default_params(mock_training_job, model_trainer):
    model_trainer.train()

    mock_training_job.create.assert_called_once()

    training_job_instance = mock_training_job.create.return_value
    training_job_instance.wait.assert_called_once_with(logs=True)


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
@patch.object(ModelTrainer, "_get_input_data_config")
def test_train_with_input_data_channels(mock_get_input_config, mock_training_job, model_trainer):
    mock_input_data_channels = {"train": "train/dir", "test": "test/dir"}

    model_trainer.train(input_data_channels=mock_input_data_channels)

    mock_get_input_config.assert_called_once_with(mock_input_data_channels)
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
