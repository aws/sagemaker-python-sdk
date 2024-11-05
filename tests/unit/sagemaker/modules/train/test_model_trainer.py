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
    ComputeConfig,
    StoppingCondition,
    RetryStrategy,
    OutputDataConfig,
    SourceCodeConfig,
    S3DataSource,
    FileSystemDataSource,
    MetricDefinition,
    DebugHookConfig,
    DebugRuleConfiguration,
    RemoteDebugConfig,
    ProfilerConfig,
    ProfilerRuleConfiguration,
    TensorBoardOutputConfig,
    ExperimentConfig,
    InfraCheckConfig,
    SessionChainingConfig,
    InputData,
)
from tests.unit import DATA_DIR

DEFAULT_BASE_NAME = "dummy-image-job"
DEFAULT_IMAGE = "000000000000.dkr.ecr.us-west-2.amazonaws.com/dummy-image:latest"
DEFAULT_BUCKET = "sagemaker-us-west-2-000000000000"
DEFAULT_ROLE = "arn:aws:iam::000000000000:role/test-role"
DEFAULT_COMPUTE_CONFIG = ComputeConfig(instance_type=DEFAULT_INSTANCE_TYPE, instance_count=1)
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
        session_instance.boto_session = MagicMock(spec="boto3.session.Session")
        yield session_instance


@pytest.fixture
def model_trainer():
    trainer = ModelTrainer(
        training_image=DEFAULT_IMAGE,
        role=DEFAULT_ROLE,
        compute_config=DEFAULT_COMPUTE_CONFIG,
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
        assert trainer.compute_config == DEFAULT_COMPUTE_CONFIG
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


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_metric_settings(mock_training_job, modules_session):
    image_uri = DEFAULT_IMAGE
    role = DEFAULT_ROLE
    metric_definition = MetricDefinition(
        name="test-metric",
        regex="test-regex",
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        session=modules_session,
        role=role,
    ).with_metric_settings(
        enable_sage_maker_metrics_time_series=True, metric_definitions=[metric_definition]
    )

    assert model_trainer._enable_sage_maker_metrics_time_series
    assert model_trainer._metric_definitions == [metric_definition]

    with patch("sagemaker.modules.train.model_trainer.Session.upload_data") as mock_upload_data:
        mock_upload_data.return_value = "s3://dummy-bucket/dummy-prefix"
        model_trainer.train()

        mock_training_job.create.assert_called_once()
        assert mock_training_job.create.call_args.kwargs[
            "algorithm_specification"
        ].metric_definitions == [metric_definition]

        assert mock_training_job.create.call_args.kwargs[
            "algorithm_specification"
        ].enable_sage_maker_metrics_time_series


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_debugger_settings(mock_training_job, modules_session):
    image_uri = DEFAULT_IMAGE
    role = DEFAULT_ROLE

    debug_hook_config = DebugHookConfig(s3_output_path="s3://dummy-bucket/dummy-prefix")
    debug_rule_config = DebugRuleConfiguration(
        rule_configuration_name="rule-name",
        rule_evaluator_image=image_uri,
        rule_parameters={"parameter": "value"},
    )
    remote_debug_config = RemoteDebugConfig(
        enable_remote_debug=True,
    )
    profiler_config = ProfilerConfig(s3_output_path="s3://dummy-bucket/dummy-prefix")
    profiler_rule_config = ProfilerRuleConfiguration(
        rule_configuration_name="rule-name",
        rule_evaluator_image=image_uri,
    )
    tensor_board_output_config = TensorBoardOutputConfig(
        s3_output_path="s3://dummy-bucket/dummy-prefix"
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        session=modules_session,
        role=role,
    ).with_debugger_settings(
        debug_hook_config=debug_hook_config,
        debug_rule_configurations=debug_rule_config,
        remote_debug_config=remote_debug_config,
        profiler_config=profiler_config,
        profiler_rule_configurations=profiler_rule_config,
        tensor_board_output_config=tensor_board_output_config,
    )

    assert model_trainer._debug_hook_config == debug_hook_config
    assert model_trainer._debug_rule_configurations == debug_rule_config
    assert model_trainer._remote_debug_config == remote_debug_config
    assert model_trainer._profiler_config == profiler_config
    assert model_trainer._profiler_rule_configurations == profiler_rule_config
    assert model_trainer._tensor_board_output_config == tensor_board_output_config

    with patch("sagemaker.modules.train.model_trainer.Session.upload_data") as mock_upload_data:
        mock_upload_data.return_value = "s3://dummy-bucket/dummy-prefix"
        model_trainer.train()

        mock_training_job.create.assert_called_once()
        assert mock_training_job.create.call_args.kwargs["debug_hook_config"] == debug_hook_config
        assert (
            mock_training_job.create.call_args.kwargs["debug_rule_configurations"]
            == debug_rule_config
        )
        assert (
            mock_training_job.create.call_args.kwargs["remote_debug_config"] == remote_debug_config
        )
        assert mock_training_job.create.call_args.kwargs["profiler_config"] == profiler_config
        assert (
            mock_training_job.create.call_args.kwargs["profiler_rule_configurations"]
            == profiler_rule_config
        )
        assert (
            mock_training_job.create.call_args.kwargs["tensor_board_output_config"]
            == tensor_board_output_config
        )


@patch("sagemaker.modules.train.model_trainer.TrainingJob")
def test_additional_settings(mock_training_job, modules_session):
    image_uri = DEFAULT_IMAGE
    role = DEFAULT_ROLE

    retry_strategy = RetryStrategy(
        maximum_retry_attempts=3,
    )

    experiment_config = ExperimentConfig(
        experiment_name="experiment-name",
        trial_name="trial-name",
    )
    infra_check_config = InfraCheckConfig(
        enable_infra_check=True,
    )
    session_chaining_config = SessionChainingConfig(
        enable_session_tag_chaining=True,
    )
    model_trainer = ModelTrainer(
        training_image=image_uri,
        session=modules_session,
        role=role,
    ).with_additional_settings(
        retry_strategy=retry_strategy,
        experiment_config=experiment_config,
        infra_check_config=infra_check_config,
        session_chaining_config=session_chaining_config,
    )

    assert model_trainer._retry_strategy == retry_strategy
    assert model_trainer._experiment_config == experiment_config
    assert model_trainer._infra_check_config == infra_check_config
    assert model_trainer._session_chaining_config == session_chaining_config

    with patch("sagemaker.modules.train.model_trainer.Session.upload_data") as mock_upload_data:
        mock_upload_data.return_value = "s3://dummy-bucket/dummy-prefix"
        model_trainer.train()

        mock_training_job.create.assert_called_once()

        assert mock_training_job.create.call_args.kwargs["retry_strategy"] == retry_strategy
        assert mock_training_job.create.call_args.kwargs["experiment_config"] == experiment_config
        assert mock_training_job.create.call_args.kwargs["infra_check_config"] == infra_check_config
        assert (
            mock_training_job.create.call_args.kwargs["session_chaining_config"]
            == session_chaining_config
        )
