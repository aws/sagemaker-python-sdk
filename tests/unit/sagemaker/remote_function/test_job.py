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
from mock import patch, Mock, ANY

from sagemaker.config import load_sagemaker_config
from tests.unit import DATA_DIR
from sagemaker.remote_function.job import (
    _JobSettings,
    _Job,
    _convert_run_to_json,
    _prepare_and_upload_runtime_scripts,
    _prepare_and_upload_dependencies,
    _filter_non_python_files,
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
HMAC_KEY = "some-hmac-key"

EXPECTED_FUNCTION_URI = S3_URI + "/function.pkl"
EXPECTED_OUTPUT_URI = S3_URI + "/output"
EXPECTED_DEPENDENCIES_URI = S3_URI + "/additional_dependencies/requirements.txt"

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

TEST_EXP_NAME = "my-exp-name"
TEST_RUN_NAME = "my-run-name"
TEST_EXP_DISPLAY_NAME = "my-exp-display-name"
TEST_RUN_DISPLAY_NAME = "my-run-display-name"
TEST_TAGS = [{"Key": "some-key", "Value": "some-value"}]


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

    session.default_bucket.return_value = BUCKET
    session.expand_role.return_value = ROLE_ARN
    session.boto_region_name = TEST_REGION
    session.sagemaker_config = None
    session._append_sagemaker_config_tags.return_value = []

    return session


def job_function(a, b=1, *, c, d=3):
    return a * b * c * d


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
        "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY,
    }
    assert job_settings.include_local_workdir is False
    assert job_settings.instance_type == "ml.m5.xlarge"


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
        "EnvVarKey": "EnvVarValue",
        "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY,
    }
    assert job_settings.job_conda_env == "my_conda_env"
    assert job_settings.include_local_workdir is True
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
@patch("sagemaker.remote_function.job._prepare_and_upload_dependencies", return_value="some_s3_uri")
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

    assert mock_stored_function.called_once_with(
        sagemaker_session=session(),
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        hmac_key=HMAC_KEY,
        s3_kms_key=None,
    )

    local_dependencies_path = mock_runtime_manager().snapshot()
    mock_python_version = mock_runtime_manager()._current_python_version()

    mock_script_upload.assert_called_once_with(
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
    )

    mock_dependency_upload.assert_called_once_with(
        local_dependencies_path=local_dependencies_path,
        include_local_workdir=True,
        pre_execution_commands=None,
        pre_execution_script_local_path=None,
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=None,
        sagemaker_session=session(),
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
                        "S3Uri": f"{S3_URI}/{job.job_name}/sm_rf_user_ws",
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
        Environment={"AWS_DEFAULT_REGION": "us-west-2", "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    )


@patch("secrets.token_hex", return_value=HMAC_KEY)
@patch("sagemaker.remote_function.job._prepare_and_upload_dependencies", return_value="some_s3_uri")
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
    mock_script_upload,
    mock_dependency_upload,
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

    assert mock_stored_function.called_once_with(
        sagemaker_session=session(),
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        hmac_key=HMAC_KEY,
        s3_kms_key=None,
    )

    local_dependencies_path = mock_runtime_manager().snapshot()
    mock_python_version = mock_runtime_manager()._current_python_version()

    mock_script_upload.assert_called_once_with(
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=session(),
    )

    mock_dependency_upload.assert_called_once_with(
        local_dependencies_path=local_dependencies_path,
        include_local_workdir=False,
        pre_execution_commands=None,
        pre_execution_script_local_path="path/to/script.sh",
        s3_base_uri=f"{S3_URI}/{job.job_name}",
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=session(),
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
                        "S3Uri": f"{S3_URI}/{job.job_name}/sm_rf_user_ws",
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
        Environment={"AWS_DEFAULT_REGION": "us-west-2", "REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    )


@patch("sagemaker.remote_function.job._prepare_and_upload_runtime_scripts")
@patch("sagemaker.remote_function.job._prepare_and_upload_dependencies")
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
@patch("sagemaker.remote_function.job._prepare_and_upload_dependencies")
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
@patch("sagemaker.remote_function.job._prepare_and_upload_dependencies")
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
        boto_session=ANY, job_name=job.job_name, wait=True, timeout=10
    )


@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("shutil.copy2")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_runtime_scripts(session, mock_copy, mock_s3_upload):
    s3_path = _prepare_and_upload_runtime_scripts(
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session(),
    )

    assert s3_path == mock_s3_upload.return_value

    assert mock_copy.call_count == 2
    mock_s3_upload.assert_called_once()


@patch("sagemaker.s3.S3Uploader.upload", return_value="some_uri")
@patch("shutil.copy2")
@patch("shutil.copytree")
@patch("sagemaker.remote_function.job.Session", return_value=mock_session())
def test_prepare_and_upload_dependencies(session, mock_copytree, mock_copy, mock_s3_upload):
    s3_path = _prepare_and_upload_dependencies(
        local_dependencies_path="some/path/to/dependency",
        include_local_workdir=True,
        pre_execution_commands=["cmd_1", "cmd_2"],
        pre_execution_script_local_path=None,
        s3_base_uri=S3_URI,
        s3_kms_key=KMS_KEY_ARN,
        sagemaker_session=session,
    )

    assert s3_path == mock_s3_upload.return_value

    mock_copytree.assert_called_with(os.getcwd(), ANY, ignore=_filter_non_python_files)
    mock_copy.assert_called_with("some/path/to/dependency", ANY)
    mock_s3_upload.assert_called_once_with(
        ANY, S3_URI + "/" + REMOTE_FUNCTION_WORKSPACE, KMS_KEY_ARN, session
    )


def test_convert_run_to_json():
    run = Mock()
    run.experiment_name = TEST_EXP_NAME
    run.run_name = TEST_RUN_NAME

    assert _convert_run_to_json(run) == (
        '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}'
    )
