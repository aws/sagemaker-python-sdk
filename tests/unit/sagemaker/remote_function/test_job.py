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
from __future__ import absolute_import

import os
import sys

import pytest
from mock import patch, Mock, ANY, mock_open
from mock.mock import MagicMock

from sagemaker.config import load_sagemaker_config
from sagemaker.remote_function.checkpoint_location import CheckpointLocation
from sagemaker.remote_function.core.stored_function import _SerializedData
from sagemaker.session_settings import SessionSettings

from sagemaker.remote_function.spark_config import SparkConfig
from sagemaker.remote_function.custom_file_filter import CustomFileFilter
from sagemaker.remote_function.core.pipeline_variables import Context
from sagemaker.workflow.function_step import DelayedReturn
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline_context import _PipelineConfig
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.utils import sagemaker_timestamp
from sagemaker.workflow.properties import Properties

from tests.unit import DATA_DIR
from sagemaker.remote_function.job import (
    _JobSettings,
    _Job,
    _convert_run_to_json,
    _prepare_and_upload_runtime_scripts,
    _prepare_and_upload_workspace,
    _prepare_and_upload_spark_dependent_files,
    _upload_spark_submit_deps,
    _upload_serialized_spark_configuration,
    _extend_spark_config_to_request,
    _prepare_dependencies_and_pre_execution_scripts,
)

from sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment import (
    set_env,
    safe_serialize,
)


REGION = "us-west-2"
TRAINING_JOB_ARN = "training-job-arn"
IMAGE = "image_uri"
BUCKET = "my-s3-bucket"
S3_URI = f"s3://{BUCKET}/keyprefix"
ROLE_ARN = "my_execution_role_arn"
KMS_KEY_ARN = "kms-key-arn"
DEFAULT_ROLE_ARN = "default_execution_role_arn"
TEST_REGION = "us-west-2"
RUNTIME_SCRIPTS_CHANNEL_NAME = "sagemaker_remote_function_bootstrap"
REMOTE_FUNCTION_WORKSPACE = "sm_rf_user_ws"
SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME = "pre_exec_script_and_dependencies"
HMAC_KEY = "some-hmac-key"

EXPECTED_FUNCTION_URI = S3_URI + "/function.pkl"
EXPECTED_OUTPUT_URI = S3_URI + "/output"
EXPECTED_DEPENDENCIES_URI = S3_URI + "/additional_dependencies/requirements.txt"

# flake8: noqa
EXPECTED_ENV_SINGLE_NODE_CPU = """
export SM_MODEL_DIR='/opt/ml/model'
export SM_INPUT_DIR='/opt/ml/input'
export SM_INPUT_DATA_DIR='/opt/ml/input/data'
export SM_INPUT_CONFIG_DIR='/opt/ml/input/config'
export SM_OUTPUT_DIR='/opt/ml/output'
export SM_OUTPUT_FAILURE='/opt/ml/output/failure'
export SM_OUTPUT_DATA_DIR='/opt/ml/output/data'
export SM_MASTER_ADDR='algo-1'
export SM_MASTER_PORT='7777'
export SM_CURRENT_HOST='algo-1'
export SM_CURRENT_INSTANCE_TYPE='ml.t3.xlarge'
export SM_HOSTS='["algo-1"]'
export SM_NETWORK_INTERFACE_NAME='eth0'
export SM_HOST_COUNT='1'
export SM_CURRENT_HOST_RANK='0'
export SM_NUM_CPUS='4'
export SM_NUM_GPUS='0'
export SM_NUM_NEURONS='0'
export SM_RESOURCE_CONFIG='{"current_host": "algo-1", "hosts": ["algo-1"], "current_group_name": "homogeneousCluster", "current_instance_type": "ml.t3.xlarge", "instance_groups": [{"instance_group_name": "homogeneousCluster", "instance_type": "ml.t3.xlarge", "hosts": ["algo-1"]}], "network_interface_name": "eth0"}'
export SM_NPROC_PER_NODE='4'
export SM_TRAINING_ENV='{"current_host": "algo-1", "current_instance_type": "ml.t3.xlarge", "hosts": ["algo-1"], "host_count": 1, "nproc_per_node": 4, "master_addr": "algo-1", "master_port": 7777, "input_config_dir": "/opt/ml/input/config", "input_data_dir": "/opt/ml/input/data", "input_dir": "/opt/ml/input", "job_name": "test-job", "model_dir": "/opt/ml/model", "network_interface_name": "eth0", "num_cpus": 4, "num_gpus": 0, "num_neurons": 0, "output_data_dir": "/opt/ml/output/data", "resource_config": {"current_host": "algo-1", "hosts": ["algo-1"], "current_group_name": "homogeneousCluster", "current_instance_type": "ml.t3.xlarge", "instance_groups": [{"instance_group_name": "homogeneousCluster", "instance_type": "ml.t3.xlarge", "hosts": ["algo-1"]}], "network_interface_name": "eth0"}}'
export NCCL_SOCKET_IFNAME='eth0'
export NCCL_PROTO='simple'
"""

# flake8: noqa
EXPECTED_ENV_SINGLE_NODE_MULTI_GPUS = """
export SM_MODEL_DIR='/opt/ml/model'
export SM_INPUT_DIR='/opt/ml/input'
export SM_INPUT_DATA_DIR='/opt/ml/input/data'
export SM_INPUT_CONFIG_DIR='/opt/ml/input/config'
export SM_OUTPUT_DIR='/opt/ml/output'
export SM_OUTPUT_FAILURE='/opt/ml/output/failure'
export SM_OUTPUT_DATA_DIR='/opt/ml/output/data'
export SM_MASTER_ADDR='algo-1'
export SM_MASTER_PORT='7777'
export SM_CURRENT_HOST='algo-1'
export SM_CURRENT_INSTANCE_TYPE='ml.g5.12xlarge'
export SM_HOSTS='["algo-1"]'
export SM_NETWORK_INTERFACE_NAME='eth0'
export SM_HOST_COUNT='1'
export SM_CURRENT_HOST_RANK='0'
export SM_NUM_CPUS='48'
export SM_NUM_GPUS='4'
export SM_NUM_NEURONS='0'
export SM_RESOURCE_CONFIG='{"current_host": "algo-1", "hosts": ["algo-1"], "current_group_name": "homogeneousCluster", "current_instance_type": "ml.g5.12xlarge", "instance_groups": [{"instance_group_name": "homogeneousCluster", "instance_type": "ml.g5.12xlarge", "hosts": ["algo-1"]}], "network_interface_name": "eth0"}'
export SM_NPROC_PER_NODE='4'
export SM_TRAINING_ENV='{"current_host": "algo-1", "current_instance_type": "ml.g5.12xlarge", "hosts": ["algo-1"], "host_count": 1, "nproc_per_node": 4, "master_addr": "algo-1", "master_port": 7777, "input_config_dir": "/opt/ml/input/config", "input_data_dir": "/opt/ml/input/data", "input_dir": "/opt/ml/input", "job_name": "test-job", "model_dir": "/opt/ml/model", "network_interface_name": "eth0", "num_cpus": 48, "num_gpus": 4, "num_neurons": 0, "output_data_dir": "/opt/ml/output/data", "resource_config": {"current_host": "algo-1", "hosts": ["algo-1"], "current_group_name": "homogeneousCluster", "current_instance_type": "ml.g5.12xlarge", "instance_groups": [{"instance_group_name": "homogeneousCluster", "instance_type": "ml.g5.12xlarge", "hosts": ["algo-1"]}], "network_interface_name": "eth0"}}'
export NCCL_SOCKET_IFNAME='eth0'
export NCCL_PROTO='simple'
"""

# flake8: noqa
EXPECTED_ENV_MULTI_NODE_MULTI_GPUS = """
export SM_MODEL_DIR='/opt/ml/model'
export SM_INPUT_DIR='/opt/ml/input'
export SM_INPUT_DATA_DIR='/opt/ml/input/data'
export SM_INPUT_CONFIG_DIR='/opt/ml/input/config'
export SM_OUTPUT_DIR='/opt/ml/output'
export SM_OUTPUT_FAILURE='/opt/ml/output/failure'
export SM_OUTPUT_DATA_DIR='/opt/ml/output/data'
export SM_MASTER_ADDR='algo-1'
export SM_MASTER_PORT='7777'
export SM_CURRENT_HOST='algo-1'
export SM_CURRENT_INSTANCE_TYPE='ml.g5.2xlarge'
export SM_HOSTS='["algo-1", "algo-2", "algo-3", "algo-4"]'
export SM_NETWORK_INTERFACE_NAME='eth0'
export SM_HOST_COUNT='4'
export SM_CURRENT_HOST_RANK='0'
export SM_NUM_CPUS='8'
export SM_NUM_GPUS='1'
export SM_NUM_NEURONS='0'
export SM_RESOURCE_CONFIG='{"current_host": "algo-1", "hosts": ["algo-1", "algo-2", "algo-3", "algo-4"], "current_group_name": "homogeneousCluster", "current_instance_type": "ml.g5.2xlarge", "instance_groups": [{"instance_group_name": "homogeneousCluster", "instance_type": "ml.g5.2xlarge", "hosts": ["algo-4", "algo-2", "algo-1", "algo-3"]}], "network_interface_name": "eth0"}'
export SM_NPROC_PER_NODE='1'
export SM_TRAINING_ENV='{"current_host": "algo-1", "current_instance_type": "ml.g5.2xlarge", "hosts": ["algo-1", "algo-2", "algo-3", "algo-4"], "host_count": 4, "nproc_per_node": 1, "master_addr": "algo-1", "master_port": 7777, "input_config_dir": "/opt/ml/input/config", "input_data_dir": "/opt/ml/input/data", "input_dir": "/opt/ml/input", "job_name": "test-job", "model_dir": "/opt/ml/model", "network_interface_name": "eth0", "num_cpus": 8, "num_gpus": 1, "num_neurons": 0, "output_data_dir": "/opt/ml/output/data", "resource_config": {"current_host": "algo-1", "hosts": ["algo-1", "algo-2", "algo-3", "algo-4"], "current_group_name": "homogeneousCluster", "current_instance_type": "ml.g5.2xlarge", "instance_groups": [{"instance_group_name": "homogeneousCluster", "instance_type": "ml.g5.2xlarge", "hosts": ["algo-4", "algo-2", "algo-1", "algo-3"]}], "network_interface_name": "eth0"}}'
export NCCL_SOCKET_IFNAME='eth0'
export NCCL_PROTO='simple'
"""

DESCRIBE_TRAINING_JOB_RESPONSE = {
    "TrainingJobArn": TRAINING_JOB_ARN,
    "TrainingJobStatus": "{}",
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.c4.xlarge",
        "VolumeSizeInGB": 30,
    },
    "OutputDataConfig": {"S3OutputPath": "s3://sagemaker-123/image_uri/output"},
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "sm_training.env")

TEST_JOB_NAME = "my-job-name"
TEST_PIPELINE_NAME = "my-pipeline"
TEST_EXP_NAME = "my-exp-name"
TEST_RUN_NAME = "my-run-name"
TEST_EXP_DISPLAY_NAME = "my-exp-display-name"
TEST_RUN_DISPLAY_NAME = "my-run-display-name"
TEST_TAGS = [{"Key": "some-key", "Value": "some-value"}]

_DEFINITION_CONFIG = PipelineDefinitionConfig(use_custom_job_prefix=False)
MOCKED_PIPELINE_CONFIG = _PipelineConfig(
    TEST_PIPELINE_NAME,
    "test-function-step",
    None,
    "code-hash-0123456789",
    "config-hash-0123456789",
    _DEFINITION_CONFIG,
    sagemaker_timestamp(),
    True,
    True,
    "token-from-pipeline",
)


def mock_get_current_run():
    current_run = Mock()
    current_run.experiment_name = TEST_EXP_NAME
    current_run.run_name = TEST_RUN_NAME
    current_run.experiment_display_name = TEST_EXP_DISPLAY_NAME
    current_run.run_display_name = TEST_RUN_DISPLAY_NAME
    current_run.tags = TEST_TAGS
    return current_run


def describe_training_job_response(job_status):
    return {
        "TrainingJobArn": TRAINING_JOB_ARN,
        "TrainingJobStatus": job_status,
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.c4.xlarge",
            "VolumeSizeInGB": 30,
        },
        "OutputDataConfig": {"S3OutputPath": "s3://sagemaker-123/image_uri/output"},
    }


COMPLETED_TRAINING_JOB = describe_training_job_response("Completed")
INPROGRESS_TRAINING_JOB = describe_training_job_response("InProgress")
CANCELLED_TRAINING_JOB = describe_training_job_response("Stopped")
FAILED_TRAINING_JOB = describe_training_job_response("Failed")


def mock_session():
    session = Mock()
    session.sagemaker_client.create_training_job.return_value = {"TrainingJobArn": TRAINING_JOB_ARN}
    session.sagemaker_client.describe_training_job.return_value = COMPLETED_TRAINING_JOB
    session.settings = SessionSettings()
    session.default_bucket.return_value = BUCKET
    session.expand_role.return_value = ROLE_ARN
    session.boto_region_name = TEST_REGION
    session.sagemaker_config = None
    session._append_sagemaker_config_tags.return_value = []
    session.default_bucket_prefix = None

    return session


def job_function(a, b=1, *, c, d=3):
    return a * b * c * d


def job_function_with_checkpoint(a, checkpoint_1=None, *, b, checkpoint_2=None):
    return a + b


def serialized_data():
    return _SerializedData(func=b"serialized_func", args=b"serialized_args")


@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=DEFAULT_ROLE_ARN)
def test_sagemaker_config_job_settings(get_execution_role, session, secret_token):

    job_settings = _JobSettings(image_uri="image_uri", instance_type="ml.m5.xlarge")
    assert job_settings.image_uri == "image_uri"
    assert job_settings.s3_root_uri == f"s3://{BUCKET}"
    assert job_settings.role == DEFAULT_ROLE_ARN
    assert job_settings.environment_variables == {
        "AWS_DEFAULT_REGION": "us-west-2",
        "REMOTE_FUNCTION_SECRET_KEY": "some-hmac-key",
    }
    assert job_settings.include_local_workdir is False
    assert job_settings.instance_type == "ml.m5.xlarge"


@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch(
    "sagemaker.remote_function.job._JobSettings._get_default_spark_image",
    return_value="some_image_uri",
)
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=DEFAULT_ROLE_ARN)
def test_sagemaker_config_job_settings_with_spark_config(
    get_execution_role, session, mock_get_default_spark_image, secret_token
):

    spark_config = SparkConfig()
    job_settings = _JobSettings(instance_type="ml.m5.xlarge", spark_config=spark_config)

    assert job_settings.image_uri == "some_image_uri"
    assert job_settings.s3_root_uri == f"s3://{BUCKET}"
    assert job_settings.role == DEFAULT_ROLE_ARN
    assert job_settings.environment_variables == {
        "AWS_DEFAULT_REGION": "us-west-2",
        "REMOTE_FUNCTION_SECRET_KEY": "some-hmac-key",
    }
    assert job_settings.include_local_workdir is False
    assert job_settings.instance_type == "ml.m5.xlarge"
    assert job_settings.spark_config == spark_config


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=DEFAULT_ROLE_ARN)
def test_sagemaker_config_job_settings_with_spark_config_and_image_uri(get_execution_role, session):

    spark_config = SparkConfig()

    with pytest.raises(
        ValueError, match="spark_config and image_uri cannot be specified at the same time!"
    ):
        _JobSettings(image_uri="image_uri", instance_type="ml.m5.xlarge", spark_config=spark_config)


def test_sagemaker_config_job_settings_with_not_supported_param_by_spark():
    spark_config = SparkConfig()

    with pytest.raises(ValueError, match="Remote Spark jobs do not support job_conda_env."):
        _JobSettings(
            instance_type="ml.m5.xlarge",
            spark_config=spark_config,
            job_conda_env="conda_env",
        )

    with pytest.raises(
        ValueError, match="Remote Spark jobs do not support automatically capturing dependencies."
    ):
        _JobSettings(
            instance_type="ml.m5.xlarge",
            spark_config=spark_config,
            dependencies="auto_capture",
        )


@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=DEFAULT_ROLE_ARN)
def test_sagemaker_config_job_settings_with_configuration_file(
    get_execution_role, session, secret_token
):
    config_tags = [
        {"Key": "someTagKey", "Value": "someTagValue"},
        {"Key": "someTagKey2", "Value": "someTagValue2"},
    ]
    session().sagemaker_config = load_sagemaker_config(
        additional_config_paths=[os.path.join(DATA_DIR, "remote_function")]
    )
    session()._append_sagemaker_config_tags.return_value = config_tags

    job_settings = _JobSettings(image_uri="image_uri")
    assert job_settings.image_uri == "image_uri"
    assert job_settings.s3_root_uri == f"s3://{BUCKET}"
    assert job_settings.role == DEFAULT_ROLE_ARN
    assert job_settings.tags == config_tags
    assert job_settings.vpc_config == {"Subnets": ["subnet-1234"], "SecurityGroupIds": ["sg123"]}
    assert job_settings.pre_execution_commands == ["command_1", "command_2"]
    assert job_settings.environment_variables == {
        "AWS_DEFAULT_REGION": "us-west-2",
        "REMOTE_FUNCTION_SECRET_KEY": "some-hmac-key",
        "EnvVarKey": "EnvVarValue",
    }
    assert job_settings.job_conda_env == "my_conda_env"
    assert job_settings.include_local_workdir is True
    assert job_settings.custom_file_filter.ignore_name_patterns == ["data", "test"]
    assert job_settings.volume_kms_key == "someVolumeKmsKey"
    assert job_settings.s3_kms_key == "someS3KmsKey"
    assert job_settings.instance_type == "ml.m5.large"
    assert job_settings.enable_network_isolation is False
    assert job_settings.encrypt_inter_container_traffic is True


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=DEFAULT_ROLE_ARN)
def test_sagemaker_config_job_settings_exclusive_pre_exec_cmd_or_script(
    get_execution_role, session
):

    with pytest.raises(
        ValueError,
        match="Only one of pre_execution_commands or pre_execution_script can be specified!",
    ):
        _JobSettings(
            image_uri="image_uri",
            instance_type="ml.m5.xlarge",
            pre_execution_commands=["command_1", "command_2"],
            pre_execution_script="path/to/local/script",
        )

    session().sagemaker_config = load_sagemaker_config(
        additional_config_paths=[os.path.join(DATA_DIR, "remote_function")]
    )

    with pytest.raises(
        ValueError,
        match="Only one of pre_execution_commands or pre_execution_script can be specified!",
    ):
        _JobSettings(
            image_uri="image_uri",
            instance_type="ml.m5.xlarge",
            pre_execution_script="path/to/local/script",
        )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=DEFAULT_ROLE_ARN)
def test_sagemaker_config_job_settings_missing_image_uri(get_execution_role, session):
    session().sagemaker_config = load_sagemaker_config(
        additional_config_paths=[os.path.join(DATA_DIR, "remote_function")]
    )

    py_version = str(sys.version_info[0]) + str(sys.version_info[1])
    if py_version not in ["310", "38"]:
        with pytest.raises(
            ValueError,
            match="Default image is supported only for Python versions 3.8 and 3.10. "
            "If you are using any other python version, you must provide a compatible image_uri.",
        ):
            _JobSettings()
    else:
        job_settings = _JobSettings()
        assert (
            job_settings.image_uri
            == f"236514542706.dkr.ecr.{TEST_REGION}.amazonaws.com/sagemaker-base-python-{py_version}:1.0"
        )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job.get_execution_role", return_value=DEFAULT_ROLE_ARN)
def test_sagemaker_config_job_settings_studio_image_uri(get_execution_role, session, monkeypatch):
    monkeypatch.setenv("SAGEMAKER_INTERNAL_IMAGE_URI", "studio_image_uri")

    session().sagemaker_config = load_sagemaker_config(
        additional_config_paths=[os.path.join(DATA_DIR, "remote_function")]
    )

    job_settings = _JobSettings()
    assert job_settings.image_uri == "studio_image_uri"

    monkeypatch.delenv("SAGEMAKER_INTERNAL_IMAGE_URI")


@patch("sagemaker.experiments._run_context._RunContext.get_current_run", new=mock_get_current_run)
@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace", return_value="some_s3_uri")
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_runtime_scripts", return_value="some_s3_uri"
)
@patch("sagemaker.remote_function.job.RuntimeEnvironmentManager")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_start(
    session,
    mock_stored_function,
    mock_runtime_manager,
    mock_script_upload,
    mock_dependency_upload,
    secret_token,
):

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.m5.large",
        encrypt_inter_container_traffic=True,
    )

    job = _Job.start(job_settings, job_function, func_args=(1, 2), func_kwargs={"c": 3, "d": 4})

    assert job.job_name.startswith("job-function")

    mock_stored_function.assert_called_once_with(
        sagemaker_session=session(),
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        hmac_key=HMAC_KEY,
        s3_kms_key=None,
    )

    mock_stored_function().save.assert_called_once_with(job_function, *(1, 2), **{"c": 3, "d": 4})

    local_dependencies_path = mock_runtime_manager().snapshot()
    mock_python_version = mock_runtime_manager()._current_python_version()
    mock_sagemaker_pysdk_version = mock_runtime_manager()._current_sagemaker_pysdk_version()

    mock_script_upload.assert_called_once_with(
        spark_config=None,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
        use_torchrun=False,
        nproc_per_node=None,
    )

    mock_dependency_upload.assert_called_once_with(
        local_dependencies_path=local_dependencies_path,
        include_local_workdir=True,
        pre_execution_commands=None,
        pre_execution_script_local_path=None,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
        custom_file_filter=None,
    )

    session().sagemaker_client.create_training_job.assert_called_once_with(
        TrainingJobName=job.job_name,
        RoleArn=ROLE_ARN,
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        RetryStrategy={"MaximumRetryAttempts": 1},
        InputDataConfig=[
            dict(
                ChannelName=RUNTIME_SCRIPTS_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_script_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
            dict(
                ChannelName=REMOTE_FUNCTION_WORKSPACE,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_dependency_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
        ],
        OutputDataConfig={"S3OutputPath": f"{S3_URI}/{job.job_name}"},
        AlgorithmSpecification=dict(
            TrainingImage=IMAGE,
            TrainingInputMode="File",
            ContainerEntrypoint=[
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
            ],
            ContainerArguments=[
                "--s3_base_uri",
                f"{S3_URI}/{job.job_name}",
                "--region",
                TEST_REGION,
                "--client_python_version",
                mock_python_version,
                "--client_sagemaker_pysdk_version",
                mock_sagemaker_pysdk_version,
                "--dependency_settings",
                '{"dependency_file": null}',
                "--run_in_context",
                '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}',
            ],
        ),
        ResourceConfig=dict(
            VolumeSizeInGB=30,
            InstanceCount=1,
            InstanceType="ml.m5.large",
            KeepAlivePeriodInSeconds=0,
        ),
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=True,
        EnableManagedSpotTraining=False,
        Environment={"AWS_DEFAULT_REGION": "us-west-2", "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    )


@patch("sagemaker.experiments._run_context._RunContext.get_current_run", new=mock_get_current_run)
@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace", return_value="some_s3_uri")
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_runtime_scripts", return_value="some_s3_uri"
)
@patch("sagemaker.remote_function.job.RuntimeEnvironmentManager")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_start_with_checkpoint_location(
    session,
    mock_stored_function,
    mock_runtime_manager,
    mock_script_upload,
    mock_user_workspace_upload,
    secret_token,
):

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.m5.large",
        encrypt_inter_container_traffic=True,
    )

    input_checkpoint_location = CheckpointLocation("s3://my-bucket/my-checkpoints/")

    job = _Job.start(
        job_settings,
        job_function_with_checkpoint,
        func_args=(1,),
        func_kwargs={"b": 2, "checkpoint_2": input_checkpoint_location},
    )

    assert job.job_name.startswith("job-function-with-checkpoint")

    mock_stored_function.assert_called_once_with(
        sagemaker_session=session(),
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        hmac_key=HMAC_KEY,
        s3_kms_key=None,
    )

    mock_stored_function().save.assert_called_once_with(
        job_function_with_checkpoint, *(1,), **{"b": 2, "checkpoint_2": input_checkpoint_location}
    )

    mock_python_version = mock_runtime_manager()._current_python_version()
    mock_sagemaker_pysdk_version = mock_runtime_manager()._current_sagemaker_pysdk_version()

    session().sagemaker_client.create_training_job.assert_called_once_with(
        TrainingJobName=job.job_name,
        RoleArn=ROLE_ARN,
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        RetryStrategy={"MaximumRetryAttempts": 1},
        CheckpointConfig={
            "LocalPath": "/opt/ml/checkpoints/",
            "S3Uri": "s3://my-bucket/my-checkpoints/",
        },
        InputDataConfig=[
            dict(
                ChannelName=RUNTIME_SCRIPTS_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_script_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
            dict(
                ChannelName=REMOTE_FUNCTION_WORKSPACE,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_user_workspace_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
        ],
        OutputDataConfig={"S3OutputPath": f"{S3_URI}/{job.job_name}"},
        AlgorithmSpecification=dict(
            TrainingImage=IMAGE,
            TrainingInputMode="File",
            ContainerEntrypoint=[
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
            ],
            ContainerArguments=[
                "--s3_base_uri",
                f"{S3_URI}/{job.job_name}",
                "--region",
                TEST_REGION,
                "--client_python_version",
                mock_python_version,
                "--client_sagemaker_pysdk_version",
                mock_sagemaker_pysdk_version,
                "--dependency_settings",
                '{"dependency_file": null}',
                "--run_in_context",
                '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}',
            ],
        ),
        ResourceConfig=dict(
            VolumeSizeInGB=30,
            InstanceCount=1,
            InstanceType="ml.m5.large",
            KeepAlivePeriodInSeconds=0,
        ),
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=True,
        EnableManagedSpotTraining=False,
        Environment={"AWS_DEFAULT_REGION": "us-west-2", "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    )


@patch("sagemaker.remote_function.job._prepare_and_upload_workspace", return_value="some_s3_uri")
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_runtime_scripts", return_value="some_s3_uri"
)
@patch("sagemaker.remote_function.job.RuntimeEnvironmentManager")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_start_with_checkpoint_location_failed_with_multiple_checkpoint_locations_in_args(
    session,
    mock_runtime_manager,
    mock_script_upload,
    mock_dependency_upload,
):

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.m5.large",
        encrypt_inter_container_traffic=True,
    )

    input_checkpoint_location = CheckpointLocation("s3://my-bucket/my-checkpoints/")

    with pytest.raises(
        ValueError,
        match="Remote function cannot have more than one argument of type CheckpointLocation.",
    ):
        _Job.start(
            job_settings,
            job_function_with_checkpoint,
            func_args=(1, input_checkpoint_location),
            func_kwargs={"b": 2, "checkpoint_2": input_checkpoint_location},
        )


@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace", return_value="some_s3_uri")
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_runtime_scripts", return_value="some_s3_uri"
)
@patch("sagemaker.remote_function.job.RuntimeEnvironmentManager")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_start_with_complete_job_settings(
    session,
    mock_stored_function,
    mock_runtime_manager,
    mock_bootstrap_script_upload,
    mock_user_workspace_upload,
    secret_token,
):

    job_settings = _JobSettings(
        dependencies="path/to/dependencies/req.txt",
        pre_execution_script="path/to/script.sh",
        environment_variables={"AWS_DEFAULT_REGION": "us-east-2"},
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        role=ROLE_ARN,
        instance_type="ml.m5.xlarge",
        job_conda_env="conda_env",
        keep_alive_period_in_seconds=120,
        volume_size=120,
        volume_kms_key=KMS_KEY_ARN,
        subnets=["subnet"],
        security_group_ids=["sg"],
    )

    job = _Job.start(job_settings, job_function, func_args=(1, 2), func_kwargs={"c": 3, "d": 4})

    assert job.job_name.startswith("job-function")

    mock_stored_function.assert_called_once_with(
        sagemaker_session=session(),
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        hmac_key=HMAC_KEY,
        s3_kms_key=KMS_KEY_ARN,
    )

    local_dependencies_path = mock_runtime_manager().snapshot()
    mock_python_version = mock_runtime_manager()._current_python_version()
    mock_sagemaker_pysdk_version = mock_runtime_manager()._current_sagemaker_pysdk_version()

    mock_bootstrap_script_upload.assert_called_once_with(
        spark_config=None,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=session(),
        use_torchrun=False,
        nproc_per_node=None,
    )

    mock_user_workspace_upload.assert_called_once_with(
        local_dependencies_path=local_dependencies_path,
        include_local_workdir=False,
        pre_execution_commands=None,
        pre_execution_script_local_path="path/to/script.sh",
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=session(),
        custom_file_filter=None,
    )

    session().sagemaker_client.create_training_job.assert_called_once_with(
        TrainingJobName=job.job_name,
        RoleArn=ROLE_ARN,
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        RetryStrategy={"MaximumRetryAttempts": 1},
        InputDataConfig=[
            dict(
                ChannelName=RUNTIME_SCRIPTS_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_bootstrap_script_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
            dict(
                ChannelName=REMOTE_FUNCTION_WORKSPACE,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_user_workspace_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
        ],
        OutputDataConfig={"S3OutputPath": f"{S3_URI}/{job.job_name}", "KmsKeyId": KMS_KEY_ARN},
        AlgorithmSpecification=dict(
            TrainingImage=IMAGE,
            TrainingInputMode="File",
            ContainerEntrypoint=[
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
            ],
            ContainerArguments=[
                "--s3_base_uri",
                f"{S3_URI}/{job.job_name}",
                "--region",
                TEST_REGION,
                "--client_python_version",
                mock_python_version,
                "--client_sagemaker_pysdk_version",
                mock_sagemaker_pysdk_version,
                "--dependency_settings",
                '{"dependency_file": "req.txt"}',
                "--s3_kms_key",
                KMS_KEY_ARN,
                "--job_conda_env",
                job_settings.job_conda_env,
            ],
        ),
        ResourceConfig=dict(
            VolumeSizeInGB=120,
            InstanceCount=1,
            InstanceType="ml.m5.xlarge",
            VolumeKmsKeyId=KMS_KEY_ARN,
            KeepAlivePeriodInSeconds=120,
        ),
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=False,
        VpcConfig=dict(Subnets=["subnet"], SecurityGroupIds=["sg"]),
        EnableManagedSpotTraining=False,
        Environment={"AWS_DEFAULT_REGION": "us-west-2", "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    )


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@patch("secrets.token_hex", MagicMock(return_value=HMAC_KEY))
@patch(
    "sagemaker.remote_function.job._prepare_dependencies_and_pre_execution_scripts",
    return_value="some_s3_uri",
)
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace", return_value="some_s3_uri")
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_runtime_scripts", return_value="some_s3_uri"
)
@patch("sagemaker.remote_function.job.RuntimeEnvironmentManager")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_get_train_args_under_pipeline_context(
    session,
    mock_stored_function_ctr,
    mock_runtime_manager,
    mock_bootstrap_scripts_upload,
    mock_user_workspace_upload,
    mock_user_dependencies_upload,
):

    from sagemaker.workflow.parameters import ParameterInteger

    mock_stored_function = Mock()
    mock_stored_function_ctr.return_value = mock_stored_function

    function_step = Mock()
    function_step.name = "parent_step"
    func_step_s3_output_prop = Properties(
        step_name=function_step.name, path="OutputDataConfig.S3OutputPath"
    )
    function_step._properties.OutputDataConfig.S3OutputPath = func_step_s3_output_prop

    job_settings = _JobSettings(
        dependencies="path/to/dependencies/req.txt",
        pre_execution_script="path/to/script.sh",
        environment_variables={"AWS_DEFAULT_REGION": "us-east-2"},
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        role=ROLE_ARN,
        instance_type="ml.m5.xlarge",
        job_conda_env="conda_env",
        keep_alive_period_in_seconds=120,
        volume_size=120,
        volume_kms_key=KMS_KEY_ARN,
        subnets=["subnet"],
        security_group_ids=["sg"],
    )

    mocked_serialized_data = serialized_data()
    s3_base_uri = f"{S3_URI}/{TEST_PIPELINE_NAME}"
    train_args = _Job.compile(
        job_settings=job_settings,
        job_name=TEST_JOB_NAME,
        s3_base_uri=s3_base_uri,
        func=job_function,
        func_args=(
            1,
            ParameterInteger(name="b", default_value=2),
            DelayedReturn(function_step, reference_path=("__getitem__", 0)),
        ),
        func_kwargs={
            "c": 3,
            "d": ParameterInteger(name="d", default_value=4),
            "e": DelayedReturn(function_step, reference_path=("__getitem__", 1)),
        },
        serialized_data=mocked_serialized_data,
    )

    mock_stored_function_ctr.assert_called_once_with(
        sagemaker_session=session(),
        s3_base_uri=s3_base_uri,
        hmac_key="token-from-pipeline",
        s3_kms_key=KMS_KEY_ARN,
        context=Context(
            step_name=MOCKED_PIPELINE_CONFIG.step_name,
            func_step_s3_dir=MOCKED_PIPELINE_CONFIG.pipeline_build_time,
        ),
    )
    mock_stored_function.save_pipeline_step_function.assert_called_once_with(mocked_serialized_data)

    local_dependencies_path = mock_runtime_manager().snapshot()
    mock_python_version = mock_runtime_manager()._current_python_version()
    mock_sagemaker_pysdk_version = mock_runtime_manager()._current_sagemaker_pysdk_version()

    mock_bootstrap_scripts_upload.assert_called_once_with(
        spark_config=None,
        s3_base_uri=s3_base_uri,
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=session(),
        use_torchrun=False,
        nproc_per_node=None,
    )

    mock_user_workspace_upload.assert_called_once_with(
        local_dependencies_path=local_dependencies_path,
        include_local_workdir=False,
        pre_execution_commands=None,
        pre_execution_script_local_path="path/to/script.sh",
        s3_base_uri=s3_base_uri,
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=session(),
        custom_file_filter=None,
    )

    mock_user_dependencies_upload.assert_called_once()

    assert train_args == dict(
        TrainingJobName=TEST_JOB_NAME,
        RoleArn=ROLE_ARN,
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        RetryStrategy={"MaximumRetryAttempts": 1},
        InputDataConfig=[
            dict(
                ChannelName=RUNTIME_SCRIPTS_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_bootstrap_scripts_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
            dict(
                ChannelName=SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_user_dependencies_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
            dict(
                ChannelName=MOCKED_PIPELINE_CONFIG.pipeline_build_time,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_user_workspace_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
        ],
        OutputDataConfig={
            "S3OutputPath": Join(
                on="/",
                values=[
                    "s3://my-s3-bucket/keyprefix/my-pipeline",
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "test-function-step",
                    "results",
                ],
            ),
            "KmsKeyId": KMS_KEY_ARN,
        },
        AlgorithmSpecification=dict(
            TrainingImage=IMAGE,
            TrainingInputMode="File",
            ContainerEntrypoint=[
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
            ],
            ContainerArguments=[
                "--s3_base_uri",
                f"{S3_URI}/{TEST_PIPELINE_NAME}",
                "--region",
                TEST_REGION,
                "--client_python_version",
                mock_python_version,
                "--client_sagemaker_pysdk_version",
                mock_sagemaker_pysdk_version,
                "--dependency_settings",
                '{"dependency_file": "req.txt"}',
                "--s3_kms_key",
                KMS_KEY_ARN,
                "--job_conda_env",
                job_settings.job_conda_env,
                "--pipeline_step_name",
                MOCKED_PIPELINE_CONFIG.step_name,
                "--pipeline_execution_id",
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "--func_step_s3_dir",
                MOCKED_PIPELINE_CONFIG.pipeline_build_time,
                "--property_references",
                "Execution.PipelineExecutionId",
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "Parameters.b",
                ParameterInteger(name="b", default_value=2).to_string(),
                "Steps.parent_step.OutputDataConfig.S3OutputPath",
                func_step_s3_output_prop.to_string(),
                "Parameters.d",
                ParameterInteger(name="d", default_value=4).to_string(),
                "Steps.parent_step.OutputDataConfig.S3OutputPath",
                func_step_s3_output_prop.to_string(),
            ],
        ),
        ResourceConfig=dict(
            VolumeSizeInGB=120,
            InstanceCount=1,
            InstanceType="ml.m5.xlarge",
            VolumeKmsKeyId=KMS_KEY_ARN,
            KeepAlivePeriodInSeconds=120,
        ),
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=False,
        VpcConfig=dict(Subnets=["subnet"], SecurityGroupIds=["sg"]),
        EnableManagedSpotTraining=False,
        Environment={
            "AWS_DEFAULT_REGION": "us-west-2",
            "REMOTE_FUNCTION_SECRET_KEY": "token-from-pipeline",
        },
    )


@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch(
    "sagemaker.remote_function.job._JobSettings._get_default_spark_image",
    return_value="some_image_uri",
)
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_spark_dependent_files",
    return_value=tuple(["jars_s3_uri", "py_files_s3_uri", "files_s3_uri", "config_file_s3_uri"]),
)
@patch("sagemaker.experiments._run_context._RunContext.get_current_run", new=mock_get_current_run)
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace", return_value="some_s3_uri")
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_runtime_scripts", return_value="some_s3_uri"
)
@patch("sagemaker.remote_function.job.RuntimeEnvironmentManager")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_start_with_spark(
    session,
    mock_stored_function,
    mock_runtime_manager,
    mock_script_upload,
    mock_dependency_upload,
    mock_spark_dependency_upload,
    mock_get_default_spark_image,
    secrete_token,
):
    spark_config = SparkConfig()
    job_settings = _JobSettings(
        spark_config=spark_config,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.m5.large",
        instance_count=2,
        encrypt_inter_container_traffic=True,
    )

    job = _Job.start(job_settings, job_function, func_args=(1, 2), func_kwargs={"c": 3, "d": 4})

    mock_python_version = mock_runtime_manager()._current_python_version()
    mock_sagemaker_pysdk_version = mock_runtime_manager()._current_sagemaker_pysdk_version()

    assert job.job_name.startswith("job-function")

    assert mock_stored_function.called_once_with(
        sagemaker_session=session(), s3_base_uri=f"{S3_URI}/{job.job_name}", s3_kms_key=None
    )

    mock_script_upload.assert_called_once_with(
        spark_config=spark_config,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
        use_torchrun=False,
        nproc_per_node=None,
    )

    session().sagemaker_client.create_training_job.assert_called_once_with(
        TrainingJobName=job.job_name,
        RoleArn=ROLE_ARN,
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        RetryStrategy={"MaximumRetryAttempts": 1},
        InputDataConfig=[
            dict(
                ChannelName=RUNTIME_SCRIPTS_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_script_upload.return_value,
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            ),
            dict(
                ChannelName=REMOTE_FUNCTION_WORKSPACE,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_dependency_upload.return_value,
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            ),
            dict(
                ChannelName="conf",
                DataSource={
                    "S3DataSource": {
                        "S3Uri": "config_file_s3_uri",
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            ),
        ],
        OutputDataConfig={"S3OutputPath": f"{S3_URI}/{job.job_name}"},
        AlgorithmSpecification=dict(
            TrainingImage="some_image_uri",
            TrainingInputMode="File",
            ContainerEntrypoint=[
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
                "--jars",
                "jars_s3_uri",
                "--py-files",
                "py_files_s3_uri",
                "--files",
                "files_s3_uri",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/spark_app.py",
            ],
            ContainerArguments=[
                "--s3_base_uri",
                f"{S3_URI}/{job.job_name}",
                "--region",
                TEST_REGION,
                "--client_python_version",
                mock_python_version,
                "--client_sagemaker_pysdk_version",
                mock_sagemaker_pysdk_version,
                "--dependency_settings",
                '{"dependency_file": null}',
                "--run_in_context",
                '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}',
            ],
        ),
        ResourceConfig=dict(
            VolumeSizeInGB=30,
            InstanceCount=2,
            InstanceType="ml.m5.large",
            KeepAlivePeriodInSeconds=0,
        ),
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=True,
        EnableManagedSpotTraining=False,
        Environment={"AWS_DEFAULT_REGION": "us-west-2", "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    )


@patch("sagemaker.remote_function.job._prepare_and_upload_runtime_scripts")
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_describe(session, *args):

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        instance_type="ml.m5.large",
    )
    job = _Job.start(job_settings, job_function, func_args=(1, 2), func_kwargs={"c": 3, "d": 4})

    job.describe()
    assert job.describe() == COMPLETED_TRAINING_JOB

    session().sagemaker_client.describe_training_job.assert_called_once()


@patch("sagemaker.remote_function.job._prepare_and_upload_runtime_scripts")
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_stop(session, *args):

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        instance_type="ml.m5.large",
    )
    job = _Job.start(job_settings, job_function, func_args=(1, 2), func_kwargs={"c": 3, "d": 4})

    job.stop()

    session().sagemaker_client.stop_training_job.assert_called_once_with(
        TrainingJobName=job.job_name
    )


@patch("sagemaker.remote_function.job._prepare_and_upload_runtime_scripts")
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace")
@patch("sagemaker.remote_function.job._logs_for_job")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_wait(session, mock_stored_function, mock_logs_for_job, *args):

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        instance_type="ml.m5.large",
    )
    job = _Job.start(job_settings, job_function, func_args=(1, 2), func_kwargs={"c": 3, "d": 4})

    job.wait(timeout=10)

    mock_logs_for_job.assert_called_with(
        sagemaker_session=ANY, job_name=job.job_name, wait=True, timeout=10
    )


@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("shutil.copy2")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_runtime_scripts(session, mock_copy, mock_s3_upload):
    s3_path = _prepare_and_upload_runtime_scripts(
        spark_config=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session(),
    )

    assert s3_path == mock_s3_upload.return_value

    assert mock_copy.call_count == 2
    mock_s3_upload.assert_called_once()


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("shutil.copy2")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_runtime_scripts_under_pipeline_context(
    session, mock_copy, mock_s3_upload
):

    s3_path = _prepare_and_upload_runtime_scripts(
        spark_config=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session(),
    )
    # Bootstrap scripts are uploaded on the first call
    assert s3_path == mock_s3_upload.return_value
    assert mock_copy.call_count == 2
    mock_s3_upload.assert_called_once()

    mock_copy.reset_mock()
    mock_s3_upload.reset_mock()

    s3_path = _prepare_and_upload_runtime_scripts(
        spark_config=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session(),
    )
    # Bootstrap scripts are not uploaded on the second call
    assert s3_path == S3_URI + "/" + RUNTIME_SCRIPTS_CHANNEL_NAME
    assert mock_copy.call_count == 0
    mock_s3_upload.assert_not_called()


@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("sagemaker.remote_function.job.copy_workdir")
@patch("shutil.copy2")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_workspace(session, mock_shutil_copy, mock_copy_workdir, mock_s3_upload):
    s3_path = _prepare_and_upload_workspace(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=["cmd_1", "cmd_2"],
        pre_execution_script_local_path=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
    )

    assert s3_path == mock_s3_upload.return_value

    mock_copy_workdir.assert_called_with(ANY, None)
    mock_shutil_copy.assert_called_with("some/path/to/dependency", ANY)
    mock_s3_upload.assert_called_once_with(
        ANY, S3_URI + "/" + REMOTE_FUNCTION_WORKSPACE, KMS_KEY_ARN, session
    )


@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("shutil.copy2")
@patch("shutil.copytree")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_dependencies_with_custom_filter(
    session, mock_copytree, mock_copy, mock_s3_upload
):
    def custom_file_filter():
        pass

    s3_path = _prepare_and_upload_workspace(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=["cmd_1", "cmd_2"],
        pre_execution_script_local_path=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
        custom_file_filter=custom_file_filter,
    )

    assert s3_path == mock_s3_upload.return_value

    mock_copytree.assert_called_with(os.getcwd(), ANY, ignore=custom_file_filter)


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("sagemaker.remote_function.job.copy_workdir")
@patch("shutil.copy2")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_workspace_under_pipeline_context(
    session, mock_copy, mock_copy_workdir, mock_s3_upload
):
    s3_path = _prepare_and_upload_workspace(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=["cmd_1", "cmd_2"],
        pre_execution_script_local_path=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
    )

    assert s3_path == mock_s3_upload.return_value

    mock_copy_workdir.assert_called_with(ANY, None)
    mock_copy.assert_not_called()
    # upload successful on the first call
    mock_s3_upload.assert_called_once_with(
        ANY,
        f"{S3_URI}/{REMOTE_FUNCTION_WORKSPACE}/{MOCKED_PIPELINE_CONFIG.pipeline_build_time}",
        KMS_KEY_ARN,
        session,
    )

    mock_copy.reset_mock()
    mock_copy_workdir.reset_mock()
    mock_s3_upload.reset_mock()

    s3_path = _prepare_and_upload_workspace(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=["cmd_1", "cmd_2"],
        pre_execution_script_local_path=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
    )

    assert s3_path == (
        f"{S3_URI}/{REMOTE_FUNCTION_WORKSPACE}/" f"{MOCKED_PIPELINE_CONFIG.pipeline_build_time}"
    )

    mock_copy_workdir.assert_not_called()
    mock_copy.assert_not_called()
    # upload is skipped on the second call
    mock_s3_upload.assert_not_called()


@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("sagemaker.remote_function.job.copy_workdir")
@patch("shutil.copy2")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_workspace_with_custom_file_filter(
    session, mock_copy, mock_copy_workdir, mock_s3_upload
):
    s3_path = _prepare_and_upload_workspace(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=["cmd_1", "cmd_2"],
        pre_execution_script_local_path=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
        custom_file_filter=CustomFileFilter(),
    )

    assert s3_path == mock_s3_upload.return_value

    mock_copy_workdir.assert_called_with(ANY, ANY)
    mock_copy.assert_called_with("some/path/to/dependency", ANY)
    mock_s3_upload.assert_called_once_with(
        ANY, S3_URI + "/" + REMOTE_FUNCTION_WORKSPACE, KMS_KEY_ARN, session
    )


@patch("builtins.open", new_callable=mock_open)
@patch("sagemaker.workflow.utilities.hash_object", return_value="updated_hash")
@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("shutil.copy2")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_dependencies_and_pre_execution_scripts(
    session, mock_copy, mock_upload, mock_hash_object, mock_open
):

    tmp_dir = "some_temp_dir"

    s3_path = _prepare_dependencies_and_pre_execution_scripts(
        local_dependencies_path="some/path/to/dependency",
        pre_execution_commands=["cmd_1", "cmd_2"],
        pre_execution_script_local_path=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
        tmp_dir=tmp_dir,
    )

    mock_copy.assert_called_with("some/path/to/dependency", tmp_dir)
    mock_open.assert_called_once()
    mock_upload.assert_not_called()
    mock_hash_object.assert_not_called()
    assert s3_path is None


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
@patch("builtins.open", new_callable=mock_open)
@patch("sagemaker.s3.S3Uploader.upload")
@patch("shutil.copy2")
@patch("sagemaker.workflow.utilities.hash_object", return_value="some_hash")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_dependencies_and_pre_execution_scripts_pipeline_context(
    session, mock_hash_object, mock_copy, mock_upload, mock_open
):

    tmp_dir = "some_temp_dir"

    s3_path = _prepare_dependencies_and_pre_execution_scripts(
        local_dependencies_path="some/path/to/dependency",
        pre_execution_commands=["cmd_1", "cmd_2"],
        pre_execution_script_local_path=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
        tmp_dir=tmp_dir,
    )

    s3_upload_path = (
        f"{S3_URI}/{MOCKED_PIPELINE_CONFIG.step_name}/"
        f"{MOCKED_PIPELINE_CONFIG.pipeline_build_time}/"
        f"pre_exec_script_and_dependencies"
    )

    mock_copy.assert_called_with("some/path/to/dependency", tmp_dir)
    mock_open.assert_called_once()
    mock_upload.assert_called_once_with(tmp_dir, s3_upload_path, KMS_KEY_ARN, session)
    assert s3_path is not None


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_spark_dependent_file_without_spark_config(session):
    assert _prepare_and_upload_spark_dependent_files(
        spark_config=None, s3_base_uri="s3://test-uri", s3_kms_key=None, sagemaker_session=session
    ) == tuple([None, None, None, None])


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch("sagemaker.remote_function.job._upload_serialized_spark_configuration")
@patch("sagemaker.remote_function.job._upload_spark_submit_deps")
def test_prepare_and_upload_spark_dependent_file(
    mock_upload_spark_submit_deps, mock_upload_serialized_spark_configuration, session
):
    spark_config = SparkConfig(
        submit_jars="path_to_jar/foo.jar",
        submit_py_files="path_to_py_file/bar.py",
        submit_files="path_to_file/data.csv",
        spark_event_logs_uri="s3://event_logs_bucket",
        configuration={},
    )

    _prepare_and_upload_spark_dependent_files(
        spark_config=spark_config,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
    )
    mock_upload_spark_submit_deps.assert_any_call(
        "path_to_jar/foo.jar", "sm_rf_spark_jars", S3_URI, KMS_KEY_ARN, session
    )
    mock_upload_spark_submit_deps.assert_any_call(
        "path_to_py_file/bar.py", "sm_rf_spark_py_files", S3_URI, KMS_KEY_ARN, session
    )
    mock_upload_spark_submit_deps.assert_any_call(
        "path_to_file/data.csv", "sm_rf_spark_data_files", S3_URI, KMS_KEY_ARN, session
    )
    mock_upload_serialized_spark_configuration.assert_any_call(S3_URI, KMS_KEY_ARN, {}, session)


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_upload_spark_submit_deps_with_empty_dependencies(session):
    assert (
        _upload_spark_submit_deps(
            submit_deps=None,
            workspace_name="workspace",
            s3_base_uri=S3_URI,
            s3_kms_key=KMS_KEY_ARN,
            sagemaker_session=session,
        )
        is None
    )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_upload_spark_submit_deps_with_invalid_input(session):
    with pytest.raises(ValueError, match="workspace_name or s3_base_uri may not be empty."):
        _upload_spark_submit_deps(
            submit_deps="path/to/deps",
            workspace_name=None,
            s3_base_uri=S3_URI,
            s3_kms_key=KMS_KEY_ARN,
            sagemaker_session=session,
        )

    with pytest.raises(ValueError, match="workspace_name or s3_base_uri may not be empty."):
        _upload_spark_submit_deps(
            submit_deps="path/to/deps",
            workspace_name="workspace",
            s3_base_uri=None,
            s3_kms_key=KMS_KEY_ARN,
            sagemaker_session=session,
        )


@patch("os.path.isfile", return_value=False)
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_upload_spark_submit_deps_with_invalid_file(session, mock_is_file):
    with pytest.raises(
        ValueError, match="submit_deps path path/to/deps is not a valid local file."
    ):
        _upload_spark_submit_deps(
            submit_deps=["path/to/deps"],
            workspace_name="workspace",
            s3_base_uri=S3_URI,
            s3_kms_key=KMS_KEY_ARN,
            sagemaker_session=session,
        )


@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("os.path.isfile", return_value=True)
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_upload_spark_submit_deps(session, mock_is_file, mock_s3_uploader_load):
    assert (
        _upload_spark_submit_deps(
            submit_deps=["path/to/deps.jar", "s3://bucket/test.jar"],
            workspace_name="spark_jars",
            s3_base_uri=S3_URI,
            s3_kms_key=KMS_KEY_ARN,
            sagemaker_session=session,
        )
        == "some_uri,s3://bucket/test.jar"
    )
    mock_s3_uploader_load.assert_called_with(
        local_path="path/to/deps.jar",
        desired_s3_uri="s3://my-s3-bucket/keyprefix/spark_jars",
        kms_key="kms-key-arn",
        sagemaker_session=session,
    )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_upload_serialized_spark_configuration_without_input(session):
    assert _upload_serialized_spark_configuration(S3_URI, KMS_KEY_ARN, None, session) is None


def test_convert_run_to_json():
    run = Mock()
    run.experiment_name = TEST_EXP_NAME
    run.run_name = TEST_RUN_NAME

    assert _convert_run_to_json(run) == (
        '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}'
    )


@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
@patch(
    "sagemaker.remote_function.job._JobSettings._get_default_spark_image",
    return_value="some_image_uri",
)
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_spark_dependent_files",
    return_value=tuple(["jars_s3_uri", "py_files_s3_uri", "files_s3_uri", "config_file_s3_uri"]),
)
def test_extend_spark_config_to_request(
    mock_upload_spark_files, mock_get_default_image, mocked_session
):
    test_request_dict = dict(
        InputDataConfig=[],
        AlgorithmSpecification=dict(
            TrainingImage=IMAGE,
            TrainingInputMode="File",
            ContainerEntrypoint=[
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
            ],
        ),
    )

    job_settings = _JobSettings(
        spark_config=SparkConfig(
            submit_jars=["path/to/jar"],
            submit_py_files=["path/to/py/file"],
            submit_files=["path/to/file"],
            configuration=[],
            spark_event_logs_uri="s3://event/log",
        ),
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        instance_type="ml.m5.large",
    )

    extended_request = _extend_spark_config_to_request(
        test_request_dict, job_settings, "s3://test/base/path"
    )

    assert extended_request == dict(
        AlgorithmSpecification={
            "ContainerEntrypoint": [
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
                "--spark-event-logs-s3-uri",
                "s3://event/log",
                "--jars",
                "jars_s3_uri",
                "--py-files",
                "py_files_s3_uri",
                "--files",
                "files_s3_uri",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/spark_app.py",
            ],
            "TrainingImage": "image_uri",
            "TrainingInputMode": "File",
        },
        InputDataConfig=[
            {
                "ChannelName": "conf",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "config_file_s3_uri",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
    )


@patch("sagemaker.experiments._run_context._RunContext.get_current_run", new=mock_get_current_run)
@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace", return_value="some_s3_uri")
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_runtime_scripts", return_value="some_s3_uri"
)
@patch("sagemaker.remote_function.job.RuntimeEnvironmentManager")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_start_with_torchrun_single_node(
    session,
    mock_stored_function,
    mock_runtime_manager,
    mock_script_upload,
    mock_dependency_upload,
    secret_token,
):

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        include_local_workdir=True,
        instance_type="ml.g5.12xlarge",
        encrypt_inter_container_traffic=True,
        use_torchrun=True,
        nproc_per_node=None,
    )

    job = _Job.start(job_settings, job_function, func_args=(1, 2), func_kwargs={"c": 3, "d": 4})

    assert job.job_name.startswith("job-function")

    mock_stored_function.assert_called_once_with(
        sagemaker_session=session(),
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        hmac_key=HMAC_KEY,
        s3_kms_key=None,
    )

    mock_stored_function().save.assert_called_once_with(job_function, *(1, 2), **{"c": 3, "d": 4})

    local_dependencies_path = mock_runtime_manager().snapshot()
    mock_python_version = mock_runtime_manager()._current_python_version()
    mock_sagemaker_pysdk_version = mock_runtime_manager()._current_sagemaker_pysdk_version()

    mock_script_upload.assert_called_once_with(
        spark_config=None,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
        use_torchrun=True,
        nproc_per_node=None,
    )

    mock_dependency_upload.assert_called_once_with(
        local_dependencies_path=local_dependencies_path,
        include_local_workdir=True,
        pre_execution_commands=None,
        pre_execution_script_local_path=None,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
        custom_file_filter=None,
    )

    session().sagemaker_client.create_training_job.assert_called_once_with(
        TrainingJobName=job.job_name,
        RoleArn=ROLE_ARN,
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        RetryStrategy={"MaximumRetryAttempts": 1},
        InputDataConfig=[
            dict(
                ChannelName=RUNTIME_SCRIPTS_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_script_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
            dict(
                ChannelName=REMOTE_FUNCTION_WORKSPACE,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_dependency_upload.return_value,
                        "S3DataType": "S3Prefix",
                    }
                },
            ),
        ],
        OutputDataConfig={"S3OutputPath": f"{S3_URI}/{job.job_name}"},
        AlgorithmSpecification=dict(
            TrainingImage=IMAGE,
            TrainingInputMode="File",
            ContainerEntrypoint=[
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
            ],
            ContainerArguments=[
                "--s3_base_uri",
                f"{S3_URI}/{job.job_name}",
                "--region",
                TEST_REGION,
                "--client_python_version",
                mock_python_version,
                "--client_sagemaker_pysdk_version",
                mock_sagemaker_pysdk_version,
                "--dependency_settings",
                '{"dependency_file": null}',
                "--run_in_context",
                '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}',
            ],
        ),
        ResourceConfig=dict(
            VolumeSizeInGB=30,
            InstanceCount=1,
            InstanceType="ml.g5.12xlarge",
            KeepAlivePeriodInSeconds=0,
        ),
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=True,
        EnableManagedSpotTraining=False,
        Environment={"AWS_DEFAULT_REGION": "us-west-2", "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    )


@patch("sagemaker.experiments._run_context._RunContext.get_current_run", new=mock_get_current_run)
@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch("sagemaker.remote_function.job._prepare_and_upload_workspace", return_value="some_s3_uri")
@patch(
    "sagemaker.remote_function.job._prepare_and_upload_runtime_scripts", return_value="some_s3_uri"
)
@patch("sagemaker.remote_function.job.RuntimeEnvironmentManager")
@patch("sagemaker.remote_function.job.StoredFunction")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_start_with_torchrun_multi_node(
    session,
    mock_stored_function,
    mock_runtime_manager,
    mock_script_upload,
    mock_dependency_upload,
    secret_token,
):

    job_settings = _JobSettings(
        image_uri=IMAGE,
        s3_root_uri=S3_URI,
        role=ROLE_ARN,
        include_local_workdir=True,
        instance_count=2,
        instance_type="ml.g5.2xlarge",
        encrypt_inter_container_traffic=True,
        use_torchrun=True,
        nproc_per_node=None,
    )

    job = _Job.start(job_settings, job_function, func_args=(1, 2), func_kwargs={"c": 3, "d": 4})

    assert job.job_name.startswith("job-function")

    mock_stored_function.assert_called_once_with(
        sagemaker_session=session(),
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        hmac_key=HMAC_KEY,
        s3_kms_key=None,
    )

    mock_stored_function().save.assert_called_once_with(job_function, *(1, 2), **{"c": 3, "d": 4})

    local_dependencies_path = mock_runtime_manager().snapshot()
    mock_python_version = mock_runtime_manager()._current_python_version()
    mock_sagemaker_pysdk_version = mock_runtime_manager()._current_sagemaker_pysdk_version()

    mock_script_upload.assert_called_once_with(
        spark_config=None,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
        use_torchrun=True,
        nproc_per_node=None,
    )

    mock_dependency_upload.assert_called_once_with(
        local_dependencies_path=local_dependencies_path,
        include_local_workdir=True,
        pre_execution_commands=None,
        pre_execution_script_local_path=None,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
        custom_file_filter=None,
    )

    session().sagemaker_client.create_training_job.assert_called_once_with(
        TrainingJobName=job.job_name,
        RoleArn=ROLE_ARN,
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        RetryStrategy={"MaximumRetryAttempts": 1},
        InputDataConfig=[
            dict(
                ChannelName=RUNTIME_SCRIPTS_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_script_upload.return_value,
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            ),
            dict(
                ChannelName=REMOTE_FUNCTION_WORKSPACE,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": mock_dependency_upload.return_value,
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            ),
        ],
        OutputDataConfig={"S3OutputPath": f"{S3_URI}/{job.job_name}"},
        AlgorithmSpecification=dict(
            TrainingImage=IMAGE,
            TrainingInputMode="File",
            ContainerEntrypoint=[
                "/bin/bash",
                "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
            ],
            ContainerArguments=[
                "--s3_base_uri",
                f"{S3_URI}/{job.job_name}",
                "--region",
                TEST_REGION,
                "--client_python_version",
                mock_python_version,
                "--client_sagemaker_pysdk_version",
                mock_sagemaker_pysdk_version,
                "--dependency_settings",
                '{"dependency_file": null}',
                "--run_in_context",
                '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}',
            ],
        ),
        ResourceConfig=dict(
            VolumeSizeInGB=30,
            InstanceCount=2,
            InstanceType="ml.g5.2xlarge",
            KeepAlivePeriodInSeconds=0,
        ),
        EnableNetworkIsolation=False,
        EnableInterContainerTrafficEncryption=True,
        EnableManagedSpotTraining=False,
        Environment={"AWS_DEFAULT_REGION": "us-west-2", "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    )


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus",
    return_value=4,
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus",
    return_value=0,
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons",
    return_value=0,
)
@patch(
    "sagemaker.modules.train.container_drivers.scripts.environment.safe_serialize",
    side_effect=safe_serialize,
)
def test_set_env_single_node_cpu(
    mock_safe_serialize, mock_num_cpus, mock_num_gpus, mock_num_neurons
):
    with patch.dict(os.environ, {"TRAINING_JOB_NAME": "test-job"}):
        set_env(
            resource_config=dict(
                current_host="algo-1",
                hosts=["algo-1"],
                current_group_name="homogeneousCluster",
                current_instance_type="ml.t3.xlarge",
                instance_groups=[
                    dict(
                        instance_group_name="homogeneousCluster",
                        instance_type="ml.t3.xlarge",
                        hosts=["algo-1"],
                    )
                ],
                network_interface_name="eth0",
            ),
            output_file=OUTPUT_FILE,
        )

        mock_num_cpus.assert_called_once()
        mock_num_gpus.assert_called_once()
        mock_num_neurons.assert_called_once()

        with open(OUTPUT_FILE, "r") as f:
            env_file = f.read().strip()
            expected_env = _remove_extra_lines(EXPECTED_ENV_SINGLE_NODE_CPU)
            env_file = _remove_extra_lines(env_file)

            assert env_file == expected_env
        os.remove(OUTPUT_FILE)
        assert not os.path.exists(OUTPUT_FILE)


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus",
    return_value=48,
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus",
    return_value=4,
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons",
    return_value=0,
)
@patch(
    "sagemaker.modules.train.container_drivers.scripts.environment.safe_serialize",
    side_effect=safe_serialize,
)
def test_set_env_single_node_multi_gpu(
    mock_safe_serialize, mock_num_cpus, mock_num_gpus, mock_num_neurons
):
    with patch.dict(os.environ, {"TRAINING_JOB_NAME": "test-job"}):
        set_env(
            resource_config=dict(
                current_host="algo-1",
                hosts=["algo-1"],
                current_group_name="homogeneousCluster",
                current_instance_type="ml.g5.12xlarge",
                instance_groups=[
                    dict(
                        instance_group_name="homogeneousCluster",
                        instance_type="ml.g5.12xlarge",
                        hosts=["algo-1"],
                    )
                ],
                network_interface_name="eth0",
            ),
            output_file=OUTPUT_FILE,
        )

        mock_num_cpus.assert_called_once()
        mock_num_gpus.assert_called_once()
        mock_num_neurons.assert_called_once()

        with open(OUTPUT_FILE, "r") as f:
            env_file = f.read().strip()
            expected_env = _remove_extra_lines(EXPECTED_ENV_SINGLE_NODE_MULTI_GPUS)
            env_file = _remove_extra_lines(env_file)

            assert env_file == expected_env
        os.remove(OUTPUT_FILE)
        assert not os.path.exists(OUTPUT_FILE)


@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_cpus",
    return_value=8,
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_gpus",
    return_value=1,
)
@patch(
    "sagemaker.remote_function.runtime_environment.bootstrap_runtime_environment.num_neurons",
    return_value=0,
)
@patch(
    "sagemaker.modules.train.container_drivers.scripts.environment.safe_serialize",
    side_effect=safe_serialize,
)
def test_set_env_multi_node_multi_gpu(
    mock_safe_serialize, mock_num_cpus, mock_num_gpus, mock_num_neurons
):
    with patch.dict(os.environ, {"TRAINING_JOB_NAME": "test-job"}):
        set_env(
            resource_config=dict(
                current_host="algo-1",
                hosts=["algo-1", "algo-2", "algo-3", "algo-4"],
                current_group_name="homogeneousCluster",
                current_instance_type="ml.g5.2xlarge",
                instance_groups=[
                    dict(
                        instance_group_name="homogeneousCluster",
                        instance_type="ml.g5.2xlarge",
                        hosts=["algo-4", "algo-2", "algo-1", "algo-3"],
                    )
                ],
                network_interface_name="eth0",
            ),
            output_file=OUTPUT_FILE,
        )

        mock_num_cpus.assert_called_once()
        mock_num_gpus.assert_called_once()
        mock_num_neurons.assert_called_once()

        with open(OUTPUT_FILE, "r") as f:
            env_file = f.read().strip()
            expected_env = _remove_extra_lines(EXPECTED_ENV_MULTI_NODE_MULTI_GPUS)
            env_file = _remove_extra_lines(env_file)

            assert env_file == expected_env
        os.remove(OUTPUT_FILE)
        assert not os.path.exists(OUTPUT_FILE)


def _remove_extra_lines(string):
    """Removes extra blank lines from a string."""
    return "\n".join([line for line in string.splitlines() if line.strip()])
