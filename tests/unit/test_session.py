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

import copy
import datetime
import io
import json
import logging
import os

import pytest
import six
from botocore.exceptions import ClientError
from mock import ANY, MagicMock, Mock, patch, call, mock_open

from .common import _raise_unexpected_client_error
import sagemaker
from sagemaker import TrainingInput, Session, get_execution_role, exceptions
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.explainer import ExplainerConfig
from sagemaker.session import (
    _tuning_job_status,
    _transform_job_status,
    _train_done,
    _wait_until,
    _wait_until_training_done,
    NOTEBOOK_METADATA_FILE,
)
from sagemaker.tuner import WarmStartConfig, WarmStartTypes
from sagemaker.inputs import BatchDataCaptureConfig
from sagemaker.config import MODEL_CONTAINERS_PATH
from sagemaker.utils import update_list_of_dicts_with_values_from_config
from sagemaker.user_agent import (
    SDK_PREFIX,
    STUDIO_PREFIX,
    NOTEBOOK_PREFIX,
)
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from tests.unit import (
    SAGEMAKER_CONFIG_MONITORING_SCHEDULE,
    SAGEMAKER_CONFIG_COMPILATION_JOB,
    SAGEMAKER_CONFIG_EDGE_PACKAGING_JOB,
    SAGEMAKER_CONFIG_ENDPOINT_CONFIG,
    SAGEMAKER_CONFIG_ENDPOINT_ENDPOINT_CONFIG_COMBINED,
    SAGEMAKER_CONFIG_ENDPOINT,
    SAGEMAKER_CONFIG_AUTO_ML,
    SAGEMAKER_CONFIG_AUTO_ML_V2,
    SAGEMAKER_CONFIG_MODEL_PACKAGE,
    SAGEMAKER_CONFIG_FEATURE_GROUP,
    SAGEMAKER_CONFIG_PROCESSING_JOB,
    SAGEMAKER_CONFIG_TRAINING_JOB,
    SAGEMAKER_CONFIG_TRANSFORM_JOB,
    SAGEMAKER_CONFIG_MODEL,
    SAGEMAKER_CONFIG_MODEL_WITH_PRIMARY_CONTAINER,
    SAGEMAKER_CONFIG_SESSION,
    _test_default_bucket_and_prefix_combinations,
    DEFAULT_S3_OBJECT_KEY_PREFIX_NAME,
    DEFAULT_S3_BUCKET_NAME,
)

STATIC_HPs = {"feature_dim": "784"}

SAMPLE_PARAM_RANGES = [{"Name": "mini_batch_size", "MinValue": "10", "MaxValue": "100"}]

ENV_INPUT = {"env_key1": "env_val1", "env_key2": "env_val2", "env_key3": "env_val3"}

REGION = "us-west-2"
STS_ENDPOINT = "sts.us-west-2.amazonaws.com"

RESOURCES = ResourceRequirements(
    requests={
        "num_cpus": 1,  # NumberOfCpuCoresRequired
        "memory": 1024,  # MinMemoryRequiredInMb (required)
        "copies": 1,
    },
    limits={},
)


@pytest.fixture()
def boto_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)

    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.9.69 Python/3.6.5 Linux/4.14.77-70.82.amzn1.x86_64 Botocore/1.12.69 Resource"
    )
    boto_mock.client.return_value = client_mock
    return boto_mock


@patch("boto3.DEFAULT_SESSION")
def test_default_session(boto3_default_session):
    sess = Session()
    assert sess.boto_session is boto3_default_session


@patch("boto3.DEFAULT_SESSION", None)
@patch("boto3.Session")
@patch("boto3.DEFAULT_SESSION", None)
def test_new_session_created(boto3_session):
    # Need to have DEFAULT_SESSION return None as other unit tests can trigger creation of global
    # default boto3 session that will persist and take precedence over boto3.Session()
    sess = Session()
    assert sess.boto_session is boto3_session.return_value


def test_process(boto_session):
    session = Session(boto_session)

    process_request_args = {
        "inputs": [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/container/path/",
                    "S3DataType": "Archive",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "my_dataset",
                "S3Input": {
                    "S3Uri": "s3://path/to/my/dataset/census.csv",
                    "LocalPath": "/container/path/",
                    "S3DataType": "Archive",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "source",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/code/source",
                    "S3DataType": "Archive",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {
            "Outputs": [
                {
                    "OutputName": "output-1",
                    "S3Output": {
                        "S3Uri": "s3://mybucket/current_job_name/output",
                        "LocalPath": "/data/output",
                        "S3UploadMode": "Continuous",
                    },
                },
                {
                    "OutputName": "my_output",
                    "S3Output": {
                        "S3Uri": "s3://uri/",
                        "LocalPath": "/container/path/",
                        "S3UploadMode": "Continuous",
                    },
                },
            ],
            "KmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/kms-key",
        },
        "job_name": "current_job_name",
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 100,
            }
        },
        "stopping_condition": {"MaxRuntimeInSeconds": 3600},
        "app_specification": {
            "ImageUri": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3",
            "ContainerArguments": ["--drop-columns", "'SelfEmployed'"],
            "ContainerEntrypoint": ["python3", "/code/source/sklearn_transformer.py"],
        },
        "environment": {"my_env_variable": 20},
        "network_config": {
            "EnableInterContainerTrafficEncryption": True,
            "EnableNetworkIsolation": True,
            "VpcConfig": {
                "SecurityGroupIds": ["my_security_group_id"],
                "Subnets": ["my_subnet_id"],
            },
        },
        "role_arn": ROLE,
        "tags": [{"Name": "my-tag", "Value": "my-tag-value"}],
        "experiment_config": {"ExperimentName": "AnExperiment"},
    }
    session.process(**process_request_args)

    expected_request = {
        "ProcessingJobName": "current_job_name",
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 100,
            }
        },
        "AppSpecification": {
            "ImageUri": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3",
            "ContainerArguments": ["--drop-columns", "'SelfEmployed'"],
            "ContainerEntrypoint": ["python3", "/code/source/sklearn_transformer.py"],
        },
        "RoleArn": ROLE,
        "ProcessingInputs": [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/container/path/",
                    "S3DataType": "Archive",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "my_dataset",
                "S3Input": {
                    "S3Uri": "s3://path/to/my/dataset/census.csv",
                    "LocalPath": "/container/path/",
                    "S3DataType": "Archive",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "source",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/code/source",
                    "S3DataType": "Archive",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "output-1",
                    "S3Output": {
                        "S3Uri": "s3://mybucket/current_job_name/output",
                        "LocalPath": "/data/output",
                        "S3UploadMode": "Continuous",
                    },
                },
                {
                    "OutputName": "my_output",
                    "S3Output": {
                        "S3Uri": "s3://uri/",
                        "LocalPath": "/container/path/",
                        "S3UploadMode": "Continuous",
                    },
                },
            ],
            "KmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/kms-key",
        },
        "Environment": {"my_env_variable": 20},
        "NetworkConfig": {
            "EnableInterContainerTrafficEncryption": True,
            "EnableNetworkIsolation": True,
            "VpcConfig": {
                "SecurityGroupIds": ["my_security_group_id"],
                "Subnets": ["my_subnet_id"],
            },
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        "Tags": [{"Name": "my-tag", "Value": "my-tag-value"}],
        "ExperimentConfig": {"ExperimentName": "AnExperiment"},
    }

    session.sagemaker_client.create_processing_job.assert_called_with(**expected_request)


def test_create_process_with_sagemaker_config_injection(sagemaker_session):
    processing_job_config = copy.deepcopy(SAGEMAKER_CONFIG_PROCESSING_JOB)
    # deleting RedshiftDatasetDefinition. API can take either RedshiftDatasetDefinition or
    # AthenaDatasetDefinition
    del processing_job_config["SageMaker"]["ProcessingJob"]["ProcessingInputs"][0][
        "DatasetDefinition"
    ]["RedshiftDatasetDefinition"]
    sagemaker_session.sagemaker_config = processing_job_config

    processing_inputs = [
        {
            "InputName": "input-1",
            # No S3Input because the API expects only one of S3Input or DatasetDefinition
            "DatasetDefinition": {
                "AthenaDatasetDefinition": {},
            },
        }
    ]
    output_config = {
        "Outputs": [
            {
                "OutputName": "output-1",
                "S3Output": {
                    "S3Uri": "s3://mybucket/current_job_name/output",
                    "LocalPath": "/data/output",
                    "S3UploadMode": "Continuous",
                },
            },
            {
                "OutputName": "my_output",
                "S3Output": {
                    "S3Uri": "s3://uri/",
                    "LocalPath": "/container/path/",
                    "S3UploadMode": "Continuous",
                },
            },
        ],
    }
    job_name = ("current_job_name",)
    resource_config = {
        "ClusterConfig": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        }
    }
    app_specification = {
        "ImageUri": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3",
        "ContainerArguments": ["--drop-columns", "'SelfEmployed'"],
        "ContainerEntrypoint": ["python3", "/code/source/sklearn_transformer.py"],
    }

    process_request_args = {
        "inputs": processing_inputs,
        "output_config": output_config,
        "job_name": job_name,
        "resources": resource_config,
        "stopping_condition": {"MaxRuntimeInSeconds": 3600},
        "app_specification": app_specification,
        "experiment_config": {"ExperimentName": "AnExperiment"},
    }
    sagemaker_session.process(**process_request_args)
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"][
        "ProcessingResources"
    ]["ClusterConfig"]["VolumeKmsKeyId"]
    expected_output_kms_key_id = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"][
        "ProcessingOutputConfig"
    ]["KmsKeyId"]
    expected_tags = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"]["Tags"]
    expected_role_arn = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"]["RoleArn"]
    expected_vpc_config = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"][
        "NetworkConfig"
    ]["VpcConfig"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"][
        "ProcessingJob"
    ]["NetworkConfig"]["EnableNetworkIsolation"]
    expected_enable_inter_containter_traffic_encryption = SAGEMAKER_CONFIG_PROCESSING_JOB[
        "SageMaker"
    ]["ProcessingJob"]["NetworkConfig"]["EnableInterContainerTrafficEncryption"]
    expected_environment = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"][
        "Environment"
    ]

    expected_request = copy.deepcopy(
        {
            "ProcessingJobName": job_name,
            "ProcessingResources": resource_config,
            "AppSpecification": app_specification,
            "RoleArn": expected_role_arn,
            "ProcessingInputs": processing_inputs,
            "ProcessingOutputConfig": output_config,
            "Environment": expected_environment,
            "NetworkConfig": {
                "VpcConfig": expected_vpc_config,
                "EnableNetworkIsolation": expected_enable_network_isolation,
                "EnableInterContainerTrafficEncryption": expected_enable_inter_containter_traffic_encryption,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
            "Tags": expected_tags,
            "ExperimentConfig": {"ExperimentName": "AnExperiment"},
        }
    )
    expected_request["ProcessingInputs"][0]["DatasetDefinition"] = processing_job_config[
        "SageMaker"
    ]["ProcessingJob"]["ProcessingInputs"][0]["DatasetDefinition"]
    expected_request["ProcessingOutputConfig"]["KmsKeyId"] = expected_output_kms_key_id
    expected_request["ProcessingResources"]["ClusterConfig"][
        "VolumeKmsKeyId"
    ] = expected_volume_kms_key_id
    expected_request["Environment"] = expected_environment

    sagemaker_session.sagemaker_client.create_processing_job.assert_called_with(**expected_request)


def test_default_bucket_with_sagemaker_config(boto_session, client):
    # common kwargs for Session objects
    session_kwargs = {
        "boto_session": boto_session,
        "sagemaker_client": client,
        "sagemaker_runtime_client": client,
        "sagemaker_metrics_client": client,
    }

    # Case 1: Use bucket from sagemaker_config
    session_with_config_bucket = Session(
        default_bucket=None,
        sagemaker_config=SAGEMAKER_CONFIG_SESSION,
        **session_kwargs,
    )
    assert (
        session_with_config_bucket.default_bucket()
        == SAGEMAKER_CONFIG_SESSION["SageMaker"]["PythonSDK"]["Modules"]["Session"][
            "DefaultS3Bucket"
        ]
    )

    # Case 2: Use bucket from user input to Session (even if sagemaker_config has a bucket)
    session_with_user_bucket = Session(
        default_bucket="default-bucket",
        sagemaker_config=SAGEMAKER_CONFIG_SESSION,
        **session_kwargs,
    )
    assert session_with_user_bucket.default_bucket() == "default-bucket"

    # Case 3: Use default bucket of SDK
    session_with_sdk_bucket = Session(
        default_bucket=None,
        sagemaker_config=None,
        **session_kwargs,
    )
    session_with_sdk_bucket.boto_session.client.return_value = Mock(
        get_caller_identity=Mock(return_value={"Account": "111111111"})
    )
    assert session_with_sdk_bucket.default_bucket() == "sagemaker-us-west-2-111111111"


def test_default_bucket_prefix_with_sagemaker_config(boto_session, client):
    # common kwargs for Session objects
    session_kwargs = {
        "boto_session": boto_session,
        "sagemaker_client": client,
        "sagemaker_runtime_client": client,
        "sagemaker_metrics_client": client,
    }

    # Case 1: Use prefix from sagemaker_config
    session_with_config_prefix = Session(
        default_bucket_prefix=None,
        sagemaker_config=SAGEMAKER_CONFIG_SESSION,
        **session_kwargs,
    )
    assert (
        session_with_config_prefix.default_bucket_prefix
        == SAGEMAKER_CONFIG_SESSION["SageMaker"]["PythonSDK"]["Modules"]["Session"][
            "DefaultS3ObjectKeyPrefix"
        ]
    )

    # Case 2: Use prefix from user input to Session (even if sagemaker_config has a prefix)
    session_with_user_prefix = Session(
        default_bucket_prefix="default-prefix",
        sagemaker_config=SAGEMAKER_CONFIG_SESSION,
        **session_kwargs,
    )
    assert session_with_user_prefix.default_bucket_prefix == "default-prefix"

    # Case 3: Neither the user input or config has the prefix
    session_with_no_prefix = Session(
        default_bucket_prefix=None,
        sagemaker_config=None,
        **session_kwargs,
    )
    assert session_with_no_prefix.default_bucket_prefix is None


def mock_exists(filepath_to_mock, exists_result):
    unmocked_exists = os.path.exists

    def side_effect(filepath):
        if filepath == filepath_to_mock:
            return exists_result
        else:
            return unmocked_exists(filepath)

    return Mock(side_effect=side_effect)


def test_get_execution_role():
    session = Mock()
    session.get_caller_identity_arn.return_value = "arn:aws:iam::369233609183:role/SageMakerRole"

    actual = get_execution_role(session)
    assert actual == "arn:aws:iam::369233609183:role/SageMakerRole"


def test_get_execution_role_works_with_service_role():
    session = Mock()
    session.get_caller_identity_arn.return_value = (
        "arn:aws:iam::369233609183:role/service-role/AmazonSageMaker-ExecutionRole-20171129T072388"
    )

    actual = get_execution_role(session)
    assert (
        actual
        == "arn:aws:iam::369233609183:role/service-role/AmazonSageMaker-ExecutionRole-20171129T072388"
    )


def test_get_execution_role_throws_exception_if_arn_is_not_role():
    session = Mock()
    session.get_caller_identity_arn.return_value = "arn:aws:iam::369233609183:user/marcos"

    with pytest.raises(ValueError) as error:
        get_execution_role(session)
    assert "The current AWS identity is not a role" in str(error.value)


def test_get_execution_role_throws_exception_if_arn_is_not_role_with_role_in_name():
    session = Mock()
    session.get_caller_identity_arn.return_value = "arn:aws:iam::369233609183:user/marcos-role"

    with pytest.raises(ValueError) as error:
        get_execution_role(session)
    assert "The current AWS identity is not a role" in str(error.value)


def test_get_execution_role_get_default_role(caplog):
    session = Mock()
    session.get_caller_identity_arn.return_value = "arn:aws:iam::369233609183:user/marcos"

    iam_client = Mock()
    iam_client.get_role.return_value = {"Role": {"Arn": "foo-role"}}
    boto_session = Mock()
    boto_session.client.return_value = iam_client

    session.boto_session = boto_session
    actual = get_execution_role(session, use_default=True)

    iam_client.get_role.assert_called_with(RoleName="AmazonSageMaker-DefaultRole")
    iam_client.attach_role_policy.assert_called_with(
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        RoleName="AmazonSageMaker-DefaultRole",
    )
    assert "Using default role: AmazonSageMaker-DefaultRole" in caplog.text
    assert actual == "foo-role"


def test_get_execution_role_create_default_role(caplog):
    session = Mock()
    session.get_caller_identity_arn.return_value = "arn:aws:iam::369233609183:user/marcos"
    permissions_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": ["sagemaker.amazonaws.com"]},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )
    iam_client = Mock()
    iam_client.exceptions.NoSuchEntityException = Exception
    iam_client.get_role = Mock(side_effect=[Exception(), {"Role": {"Arn": "foo-role"}}])

    boto_session = Mock()
    boto_session.client.return_value = iam_client

    session.boto_session = boto_session

    actual = get_execution_role(session, use_default=True)

    iam_client.create_role.assert_called_with(
        RoleName="AmazonSageMaker-DefaultRole", AssumeRolePolicyDocument=str(permissions_policy)
    )

    iam_client.attach_role_policy.assert_called_with(
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        RoleName="AmazonSageMaker-DefaultRole",
    )

    assert "Created new sagemaker execution role: AmazonSageMaker-DefaultRole" in caplog.text

    assert actual == "foo-role"


@patch(
    "six.moves.builtins.open",
    mock_open(read_data='{"ResourceName": "SageMakerInstance"}'),
)
@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, True))
def test_get_caller_identity_arn_from_describe_notebook_instance(boto_session):
    sess = Session(boto_session)
    expected_role = "arn:aws:iam::369233609183:role/service-role/SageMakerRole-20171129T072388"
    sess.sagemaker_client.describe_notebook_instance.return_value = {"RoleArn": expected_role}

    actual = sess.get_caller_identity_arn()

    assert actual == expected_role
    sess.sagemaker_client.describe_notebook_instance.assert_called_once_with(
        NotebookInstanceName="SageMakerInstance"
    )


@patch(
    "six.moves.builtins.open",
    mock_open(
        read_data='{"ResourceName": "SageMakerInstance", '
        '"DomainId": "d-kbnw5yk6tg8j", '
        '"UserProfileName": "default-1617915559064"}'
    ),
)
@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, True))
def test_get_caller_identity_arn_from_describe_user_profile(boto_session):
    sess = Session(boto_session)
    expected_role = "arn:aws:iam::369233609183:role/service-role/SageMakerRole-20171129T072388"
    sess.sagemaker_client.describe_user_profile.return_value = {
        "UserSettings": {"ExecutionRole": expected_role}
    }

    actual = sess.get_caller_identity_arn()

    assert actual == expected_role
    sess.sagemaker_client.describe_user_profile.assert_called_once_with(
        DomainId="d-kbnw5yk6tg8j",
        UserProfileName="default-1617915559064",
    )


@patch(
    "six.moves.builtins.open",
    mock_open(
        read_data='{"ResourceName": "SageMakerInstance", '
        '"DomainId": "d-kbnw5yk6tg8j", '
        '"UserProfileName": "default-1617915559064"}'
    ),
)
@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, True))
def test_get_caller_identity_arn_from_describe_domain_if_no_user_settings(boto_session):
    sess = Session(boto_session)
    expected_role = "arn:aws:iam::369233609183:role/service-role/SageMakerRole-20171129T072388"
    sess.sagemaker_client.describe_user_profile.return_value = {}
    sess.sagemaker_client.describe_domain.return_value = {
        "DefaultUserSettings": {"ExecutionRole": expected_role}
    }

    actual = sess.get_caller_identity_arn()

    assert actual == expected_role
    sess.sagemaker_client.describe_user_profile.assert_called_once_with(
        DomainId="d-kbnw5yk6tg8j",
        UserProfileName="default-1617915559064",
    )
    sess.sagemaker_client.describe_domain.assert_called_once_with(DomainId="d-kbnw5yk6tg8j")


@patch(
    "six.moves.builtins.open",
    mock_open(
        read_data='{"ResourceName": "SageMakerInstance", '
        '"DomainId": "d-kbnw5yk6tg8j", '
        '"UserProfileName": "default-1617915559064"}'
    ),
)
@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, True))
def test_fallback_to_domain_if_role_unavailable_in_user_settings(boto_session):
    sess = Session(boto_session)
    expected_role = "expected_role"
    sess.sagemaker_client.describe_user_profile.return_value = {
        "DomainId": "d-kbnw5yk6tg8j",
        "UserSettings": {
            "JupyterServerAppSettings": {},
            "KernelGatewayAppSettings": {},
        },
    }

    sess.sagemaker_client.describe_domain.return_value = {
        "DefaultUserSettings": {"ExecutionRole": expected_role}
    }

    actual = sess.get_caller_identity_arn()

    assert actual == expected_role
    sess.sagemaker_client.describe_user_profile.assert_called_once_with(
        DomainId="d-kbnw5yk6tg8j",
        UserProfileName="default-1617915559064",
    )
    sess.sagemaker_client.describe_domain.assert_called_once_with(DomainId="d-kbnw5yk6tg8j")


@patch(
    "six.moves.builtins.open",
    mock_open(
        read_data='{"ResourceName": "SageMakerInstance", '
        '"DomainId": "d-kbnw5yk6tg8j", '
        '"ExecutionRoleArn": "arn:aws:iam::369233609183:role/service-role/SageMakerRole-20171129T072388", '
        '"SpaceName": "space_name"}'
    ),
)
@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, True))
def test_get_caller_identity_arn_from_metadata_file_for_space(boto_session):
    sess = Session(boto_session)
    expected_role = "arn:aws:iam::369233609183:role/service-role/SageMakerRole-20171129T072388"

    actual = sess.get_caller_identity_arn()

    assert actual == expected_role


@patch(
    "six.moves.builtins.open",
    mock_open(
        read_data='{"ResourceName": "SageMakerInstance", '
        '"DomainId": "d-kbnw5yk6tg8j", '
        '"SpaceName": "space_name"}'
    ),
)
@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, True))
def test_get_caller_identity_arn_from_describe_domain_for_space(boto_session):
    sess = Session(boto_session)
    expected_role = "arn:aws:iam::369233609183:role/service-role/SageMakerRole-20171129T072388"
    sess.sagemaker_client.describe_domain.return_value = {
        "DefaultSpaceSettings": {"ExecutionRole": expected_role}
    }

    actual = sess.get_caller_identity_arn()

    assert actual == expected_role
    sess.sagemaker_client.describe_domain.assert_called_once_with(DomainId="d-kbnw5yk6tg8j")


@patch(
    "six.moves.builtins.open",
    mock_open(read_data='{"ResourceName": "SageMakerInstance"}'),
)
@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, True))
@patch("sagemaker.session.sts_regional_endpoint", return_value=STS_ENDPOINT)
def test_get_caller_identity_arn_from_a_role_after_describe_notebook_exception(
    sts_regional_endpoint, boto_session
):
    sess = Session(boto_session)
    exception = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "RecordNotFound"}},
        "Operation",
    )
    sess.sagemaker_client.describe_notebook_instance.side_effect = exception

    arn = (
        "arn:aws:sts::369233609183:assumed-role/SageMakerRole/6d009ef3-5306-49d5-8efc-78db644d8122"
    )
    sess.boto_session.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Arn": arn
    }

    expected_role = "arn:aws:iam::369233609183:role/SageMakerRole"
    sess.boto_session.client("iam").get_role.return_value = {"Role": {"Arn": expected_role}}

    with patch("logging.Logger.debug") as mock_logger:
        actual = sess.get_caller_identity_arn()
        mock_logger.assert_called_once()

    sess.sagemaker_client.describe_notebook_instance.assert_called_once_with(
        NotebookInstanceName="SageMakerInstance"
    )
    assert actual == expected_role


@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, False))
@patch("sagemaker.session.sts_regional_endpoint", return_value=STS_ENDPOINT)
def test_get_caller_identity_arn_from_a_user(sts_regional_endpoint, boto_session):
    sess = Session(boto_session)
    arn = "arn:aws:iam::369233609183:user/mia"
    sess.boto_session.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Arn": arn
    }
    sess.boto_session.client("iam").get_role.return_value = {"Role": {"Arn": arn}}

    actual = sess.get_caller_identity_arn()
    assert actual == "arn:aws:iam::369233609183:user/mia"


@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, False))
@patch("sagemaker.session.sts_regional_endpoint", return_value=STS_ENDPOINT)
def test_get_caller_identity_arn_from_an_user_without_permissions(
    sts_regional_endpoint, boto_session
):
    sess = Session(boto_session)
    arn = "arn:aws:iam::369233609183:user/mia"
    sess.boto_session.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Arn": arn
    }
    sess.boto_session.client("iam").get_role.side_effect = ClientError({}, {})

    with patch("logging.Logger.warning") as mock_logger:
        actual = sess.get_caller_identity_arn()
        assert actual == "arn:aws:iam::369233609183:user/mia"
        mock_logger.assert_called_once()


@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, False))
@patch("sagemaker.session.sts_regional_endpoint", return_value=STS_ENDPOINT)
def test_get_caller_identity_arn_from_a_role(sts_regional_endpoint, boto_session):
    sess = Session(boto_session)
    arn = (
        "arn:aws:sts::369233609183:assumed-role/SageMakerRole/6d009ef3-5306-49d5-8efc-78db644d8122"
    )
    sess.boto_session.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Arn": arn
    }

    expected_role = "arn:aws:iam::369233609183:role/SageMakerRole"
    sess.boto_session.client("iam").get_role.return_value = {"Role": {"Arn": expected_role}}

    actual = sess.get_caller_identity_arn()
    assert actual == expected_role


@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, False))
@patch("sagemaker.session.sts_regional_endpoint", return_value=STS_ENDPOINT)
def test_get_caller_identity_arn_from_an_execution_role(sts_regional_endpoint, boto_session):
    sess = Session(boto_session)
    sts_arn = "arn:aws:sts::369233609183:assumed-role/AmazonSageMaker-ExecutionRole-20171129T072388/SageMaker"
    sess.boto_session.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Arn": sts_arn
    }
    iam_arn = "arn:aws:iam::369233609183:role/AmazonSageMaker-ExecutionRole-20171129T072388"
    sess.boto_session.client("iam").get_role.return_value = {"Role": {"Arn": iam_arn}}

    actual = sess.get_caller_identity_arn()
    assert actual == iam_arn


@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, False))
@patch("sagemaker.session.sts_regional_endpoint", return_value=STS_ENDPOINT)
def test_get_caller_identity_arn_from_a_sagemaker_execution_role_with_iam_client_error(
    sts_regional_endpoint, boto_session
):
    sess = Session(boto_session)
    arn = "arn:aws:sts::369233609183:assumed-role/AmazonSageMaker-ExecutionRole-20171129T072388/SageMaker"
    sess.boto_session.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Arn": arn
    }

    sess.boto_session.client("iam").get_role.side_effect = ClientError({}, {})

    actual = sess.get_caller_identity_arn()
    assert (
        actual
        == "arn:aws:iam::369233609183:role/service-role/AmazonSageMaker-ExecutionRole-20171129T072388"
    )


@patch("os.path.exists", side_effect=mock_exists(NOTEBOOK_METADATA_FILE, False))
@patch("sagemaker.session.sts_regional_endpoint", return_value=STS_ENDPOINT)
def test_get_caller_identity_arn_from_role_with_path(sts_regional_endpoint, boto_session):
    sess = Session(boto_session)
    arn_prefix = "arn:aws:iam::369233609183:role"
    role_name = "name"
    sess.boto_session.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Arn": "/".join([arn_prefix, role_name])
    }

    role_path = "path"
    role_with_path = "/".join([arn_prefix, role_path, role_name])
    sess.boto_session.client("iam").get_role.return_value = {"Role": {"Arn": role_with_path}}

    actual = sess.get_caller_identity_arn()
    assert actual == role_with_path


def test_delete_endpoint(boto_session):
    sess = Session(boto_session)
    sess.delete_endpoint("my_endpoint")

    boto_session.client().delete_endpoint.assert_called_with(EndpointName="my_endpoint")


def test_delete_endpoint_config(boto_session):
    sess = Session(boto_session)
    sess.delete_endpoint_config("my_endpoint_config")

    boto_session.client().delete_endpoint_config.assert_called_with(
        EndpointConfigName="my_endpoint_config"
    )


def test_delete_model(boto_session):
    sess = Session(boto_session)

    model_name = "my_model"
    sess.delete_model(model_name)

    boto_session.client().delete_model.assert_called_with(ModelName=model_name)


def test_user_agent_injected(boto_session):
    assert SDK_PREFIX not in boto_session.client("sagemaker")._client_config.user_agent

    sess = Session(boto_session)

    for client in [
        sess.sagemaker_client,
        sess.sagemaker_runtime_client,
        sess.sagemaker_metrics_client,
    ]:
        assert SDK_PREFIX in client._client_config.user_agent
        assert NOTEBOOK_PREFIX not in client._client_config.user_agent
        assert STUDIO_PREFIX not in client._client_config.user_agent


@patch("sagemaker.user_agent.process_notebook_metadata_file", return_value="ml.t3.medium")
def test_user_agent_injected_with_nbi(
    mock_process_notebook_metadata_file,
    boto_session,
):
    assert SDK_PREFIX not in boto_session.client("sagemaker")._client_config.user_agent

    sess = Session(
        boto_session=boto_session,
    )

    for client in [
        sess.sagemaker_client,
        sess.sagemaker_runtime_client,
        sess.sagemaker_metrics_client,
    ]:
        mock_process_notebook_metadata_file.assert_called()

        assert SDK_PREFIX in client._client_config.user_agent
        assert NOTEBOOK_PREFIX in client._client_config.user_agent
        assert STUDIO_PREFIX not in client._client_config.user_agent


@patch("sagemaker.user_agent.process_studio_metadata_file", return_value="dymmy-app-type")
def test_user_agent_injected_with_studio_app_type(
    mock_process_studio_metadata_file,
    boto_session,
):
    assert SDK_PREFIX not in boto_session.client("sagemaker")._client_config.user_agent

    sess = Session(
        boto_session=boto_session,
    )

    for client in [
        sess.sagemaker_client,
        sess.sagemaker_runtime_client,
        sess.sagemaker_metrics_client,
    ]:
        mock_process_studio_metadata_file.assert_called()

        assert SDK_PREFIX in client._client_config.user_agent
        assert NOTEBOOK_PREFIX not in client._client_config.user_agent
        assert STUDIO_PREFIX in client._client_config.user_agent


def test_training_input_all_defaults():
    prefix = "pre"
    actual = TrainingInput(s3_data=prefix)
    expected = {
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": prefix,
            }
        }
    }
    assert actual.config == expected


def test_training_input_all_arguments():
    prefix = "pre"
    distribution = "FullyReplicated"
    compression = "Gzip"
    content_type = "text/csv"
    record_wrapping = "RecordIO"
    s3_data_type = "Manifestfile"
    input_mode = "Pipe"
    result = TrainingInput(
        s3_data=prefix,
        distribution=distribution,
        compression=compression,
        input_mode=input_mode,
        content_type=content_type,
        record_wrapping=record_wrapping,
        s3_data_type=s3_data_type,
    )
    expected = {
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": distribution,
                "S3DataType": s3_data_type,
                "S3Uri": prefix,
            }
        },
        "CompressionType": compression,
        "ContentType": content_type,
        "RecordWrapperType": record_wrapping,
        "InputMode": input_mode,
    }

    assert result.config == expected


IMAGE = "myimage"
S3_INPUT_URI = "s3://mybucket/data"
DEFAULT_S3_VALIDATION_DATA = "s3://mybucket/invalidation_data"
S3_OUTPUT = "s3://sagemaker-123/output/jobname"
ROLE = "SageMakerRole"
EXPANDED_ROLE = "arn:aws:iam::111111111111:role/ExpandedRole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
MAX_SIZE = 30
MAX_TIME = 3 * 60 * 60
JOB_NAME = "jobname"
TAGS = [{"Key": "some-tag", "Value": "value-for-tag"}]
VPC_CONFIG = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
METRIC_DEFINITONS = [{"Name": "validation-rmse", "Regex": "validation-rmse=(\\d+)"}]
EXPERIMENT_CONFIG = {
    "ExperimentName": "dummyExp",
    "TrialName": "dummyT",
    "TrialComponentDisplayName": "dummyTC",
    "RunName": "dummyRN",
}
MODEL_CLIENT_CONFIG = {"InvocationsMaxRetries": 2, "InvocationsTimeoutInSeconds": 60}

DEFAULT_EXPECTED_TRAIN_JOB_ARGS = {
    "OutputDataConfig": {"S3OutputPath": S3_OUTPUT},
    "RoleArn": EXPANDED_ROLE,
    "ResourceConfig": {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": MAX_SIZE,
    },
    "InputDataConfig": [
        {
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
            "ChannelName": "training",
        }
    ],
    "AlgorithmSpecification": {"TrainingInputMode": "File", "TrainingImage": IMAGE},
    "TrainingJobName": JOB_NAME,
    "StoppingCondition": {"MaxRuntimeInSeconds": MAX_TIME},
    "VpcConfig": VPC_CONFIG,
    "ExperimentConfig": EXPERIMENT_CONFIG,
}

COMPLETED_DESCRIBE_JOB_RESULT = dict(DEFAULT_EXPECTED_TRAIN_JOB_ARGS)
COMPLETED_DESCRIBE_JOB_RESULT.update(
    {"TrainingJobArn": "arn:aws:sagemaker:us-west-2:336:training-job/" + JOB_NAME}
)
COMPLETED_DESCRIBE_JOB_RESULT.update({"TrainingJobStatus": "Completed"})
COMPLETED_DESCRIBE_JOB_RESULT.update(
    {"ModelArtifacts": {"S3ModelArtifacts": S3_OUTPUT + "/model/model.tar.gz"}}
)
# TrainingStartTime and TrainingEndTime are for billable seconds calculation
COMPLETED_DESCRIBE_JOB_RESULT.update(
    {"TrainingStartTime": datetime.datetime(2018, 2, 17, 7, 15, 0, 103000)}
)
COMPLETED_DESCRIBE_JOB_RESULT.update(
    {"TrainingEndTime": datetime.datetime(2018, 2, 17, 7, 19, 34, 953000)}
)

COMPLETED_DESCRIBE_JOB_RESULT_UNCOMPRESSED_S3_MODEL = copy.deepcopy(COMPLETED_DESCRIBE_JOB_RESULT)
COMPLETED_DESCRIBE_JOB_RESULT_UNCOMPRESSED_S3_MODEL["ModelArtifacts"]["S3ModelArtifacts"] = (
    S3_OUTPUT + "/model/prefix"
)
COMPLETED_DESCRIBE_JOB_RESULT_UNCOMPRESSED_S3_MODEL["OutputDataConfig"]["CompressionType"] = "NONE"


STOPPED_DESCRIBE_JOB_RESULT = dict(COMPLETED_DESCRIBE_JOB_RESULT)
STOPPED_DESCRIBE_JOB_RESULT.update({"TrainingJobStatus": "Stopped"})

IN_PROGRESS_DESCRIBE_JOB_RESULT = dict(DEFAULT_EXPECTED_TRAIN_JOB_ARGS)
IN_PROGRESS_DESCRIBE_JOB_RESULT.update({"TrainingJobStatus": "InProgress"})

COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT = {
    "TransformJobStatus": "Completed",
    "ModelName": "some-model",
    "TransformJobName": JOB_NAME,
    "TransformResources": {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
    },
    "TransformEndTime": datetime.datetime(2018, 2, 17, 7, 19, 34, 953000),
    "TransformStartTime": datetime.datetime(2018, 2, 17, 7, 15, 0, 103000),
    "TransformOutput": {
        "AssembleWith": "None",
        "KmsKeyId": "",
        "S3OutputPath": S3_OUTPUT,
    },
    "TransformInput": {
        "CompressionType": "None",
        "ContentType": "text/csv",
        "DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI},
        "SplitType": "Line",
    },
}

STOPPED_DESCRIBE_TRANSFORM_JOB_RESULT = dict(COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT)
STOPPED_DESCRIBE_TRANSFORM_JOB_RESULT.update({"TransformJobStatus": "Stopped"})

IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT = dict(COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT)
IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT.update({"TransformJobStatus": "InProgress"})

SERVERLESS_INFERENCE_CONFIG = {
    "MemorySizeInMB": 2048,
    "MaxConcurrency": 2,
}


@pytest.fixture()
def sagemaker_session():
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("sts", endpoint_url=STS_ENDPOINT).get_caller_identity.return_value = {
        "Account": "123"
    }
    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())
    ims.expand_role = Mock(return_value=EXPANDED_ROLE)

    return ims


def test_train_pack_to_request(sagemaker_session):
    in_config = [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        }
    ]

    out_config = {"S3OutputPath": S3_OUTPUT}

    resource_config = {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": MAX_SIZE,
    }

    stop_cond = {"MaxRuntimeInSeconds": MAX_TIME}

    sagemaker_session.train(
        image_uri=IMAGE,
        input_mode="File",
        input_config=in_config,
        role=EXPANDED_ROLE,
        job_name=JOB_NAME,
        output_config=out_config,
        resource_config=resource_config,
        hyperparameters=None,
        stop_condition=stop_cond,
        tags=None,
        vpc_config=VPC_CONFIG,
        metric_definitions=None,
        experiment_config=EXPERIMENT_CONFIG,
        enable_sagemaker_metrics=None,
    )

    assert sagemaker_session.sagemaker_client.method_calls[0] == (
        "create_training_job",
        (),
        DEFAULT_EXPECTED_TRAIN_JOB_ARGS,
    )


SAMPLE_STOPPING_CONDITION = {"MaxRuntimeInSeconds": MAX_TIME}

RESOURCE_CONFIG = {
    "InstanceCount": INSTANCE_COUNT,
    "InstanceType": INSTANCE_TYPE,
    "VolumeSizeInGB": MAX_SIZE,
}

SAMPLE_INPUT = [
    {
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": S3_INPUT_URI,
            }
        },
        "ChannelName": "training",
    }
]

SAMPLE_OUTPUT = {"S3OutputPath": S3_OUTPUT}

SAMPLE_OBJECTIVE = {"Type": "Maximize", "MetricName": "val-score"}
SAMPLE_OBJECTIVE_2 = {"Type": "Maximize", "MetricName": "value-score"}

SAMPLE_METRIC_DEF = [{"Name": "train:progress", "Regex": "regex-1"}]
SAMPLE_METRIC_DEF_2 = [{"Name": "value-score", "Regex": "regex-2"}]

STATIC_HPs = {"feature_dim": "784"}
STATIC_HPs_2 = {"gamma": "0.1"}

SAMPLE_PARAM_RANGES = [{"Name": "mini_batch_size", "MinValue": "10", "MaxValue": "100"}]
SAMPLE_PARAM_RANGES_2 = [{"Name": "kernel", "Values": ["rbf", "sigmoid"]}]

SAMPLE_TUNING_JOB_REQUEST = {
    "HyperParameterTuningJobName": "dummy-tuning-1",
    "HyperParameterTuningJobConfig": {
        "Strategy": "Bayesian",
        "HyperParameterTuningJobObjective": SAMPLE_OBJECTIVE,
        "ResourceLimits": {
            "MaxNumberOfTrainingJobs": 100,
            "MaxParallelTrainingJobs": 5,
        },
        "ParameterRanges": SAMPLE_PARAM_RANGES,
        "TrainingJobEarlyStoppingType": "Off",
        "RandomSeed": 0,
    },
    "TrainingJobDefinition": {
        "StaticHyperParameters": STATIC_HPs,
        "AlgorithmSpecification": {
            "TrainingImage": "dummy-image-1",
            "TrainingInputMode": "File",
            "MetricDefinitions": SAMPLE_METRIC_DEF,
        },
        "RoleArn": EXPANDED_ROLE,
        "InputDataConfig": SAMPLE_INPUT,
        "OutputDataConfig": SAMPLE_OUTPUT,
        "ResourceConfig": RESOURCE_CONFIG,
        "StoppingCondition": SAMPLE_STOPPING_CONDITION,
        "Environment": ENV_INPUT,
    },
}

SAMPLE_MULTI_ALGO_TUNING_JOB_REQUEST = {
    "HyperParameterTuningJobName": "dummy-tuning-1",
    "HyperParameterTuningJobConfig": {
        "Strategy": "Bayesian",
        "ResourceLimits": {
            "MaxNumberOfTrainingJobs": 100,
            "MaxParallelTrainingJobs": 5,
        },
        "TrainingJobEarlyStoppingType": "Off",
    },
    "TrainingJobDefinitions": [
        {
            "DefinitionName": "estimator_1",
            "TuningObjective": SAMPLE_OBJECTIVE,
            "HyperParameterRanges": SAMPLE_PARAM_RANGES,
            "StaticHyperParameters": STATIC_HPs,
            "AlgorithmSpecification": {
                "TrainingImage": "dummy-image-1",
                "TrainingInputMode": "File",
                "MetricDefinitions": SAMPLE_METRIC_DEF,
            },
            "RoleArn": EXPANDED_ROLE,
            "InputDataConfig": SAMPLE_INPUT,
            "OutputDataConfig": SAMPLE_OUTPUT,
            "ResourceConfig": RESOURCE_CONFIG,
            "StoppingCondition": SAMPLE_STOPPING_CONDITION,
            "Environment": ENV_INPUT,
        },
        {
            "DefinitionName": "estimator_2",
            "TuningObjective": SAMPLE_OBJECTIVE_2,
            "HyperParameterRanges": SAMPLE_PARAM_RANGES_2,
            "StaticHyperParameters": STATIC_HPs_2,
            "AlgorithmSpecification": {
                "TrainingImage": "dummy-image-2",
                "TrainingInputMode": "File",
                "MetricDefinitions": SAMPLE_METRIC_DEF_2,
            },
            "RoleArn": EXPANDED_ROLE,
            "InputDataConfig": SAMPLE_INPUT,
            "OutputDataConfig": SAMPLE_OUTPUT,
            "ResourceConfig": RESOURCE_CONFIG,
            "StoppingCondition": SAMPLE_STOPPING_CONDITION,
            "Environment": ENV_INPUT,
        },
    ],
}

SAMPLE_HYPERBAND_STRATEGY_CONFIG = {
    "HyperbandStrategyConfig": {
        "MinResource": 1,
        "MaxResource": 10,
    }
}


@pytest.mark.parametrize(
    "warm_start_type, parents",
    [
        ("IdenticalDataAndAlgorithm", {"p1", "p2", "p3"}),
        ("TransferLearning", {"p1", "p2", "p3"}),
    ],
)
def test_tune_warm_start(sagemaker_session, warm_start_type, parents):
    def assert_create_tuning_job_request(**kwrags):
        assert (
            kwrags["HyperParameterTuningJobConfig"]
            == SAMPLE_TUNING_JOB_REQUEST["HyperParameterTuningJobConfig"]
        )
        assert kwrags["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert kwrags["TrainingJobDefinition"] == SAMPLE_TUNING_JOB_REQUEST["TrainingJobDefinition"]
        assert kwrags["WarmStartConfig"] == {
            "WarmStartType": warm_start_type,
            "ParentHyperParameterTuningJobs": [
                {"HyperParameterTuningJobName": parent} for parent in parents
            ],
        }

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = (
        assert_create_tuning_job_request
    )
    sagemaker_session.tune(
        job_name="dummy-tuning-1",
        strategy="Bayesian",
        random_seed=0,
        objective_type="Maximize",
        objective_metric_name="val-score",
        max_jobs=100,
        max_parallel_jobs=5,
        parameter_ranges=SAMPLE_PARAM_RANGES,
        static_hyperparameters=STATIC_HPs,
        image_uri="dummy-image-1",
        input_mode="File",
        metric_definitions=SAMPLE_METRIC_DEF,
        role=EXPANDED_ROLE,
        input_config=SAMPLE_INPUT,
        output_config=SAMPLE_OUTPUT,
        resource_config=RESOURCE_CONFIG,
        stop_condition=SAMPLE_STOPPING_CONDITION,
        tags=None,
        warm_start_config=WarmStartConfig(
            warm_start_type=WarmStartTypes(warm_start_type), parents=parents
        ).to_input_req(),
        environment=ENV_INPUT,
    )


def test_create_tuning_job_without_training_config_or_list(sagemaker_session):
    with pytest.raises(
        ValueError,
        match="Either training_config or training_config_list should be provided.",
    ):
        sagemaker_session.create_tuning_job(
            job_name="dummy-tuning-1",
            tuning_config={
                "strategy": "Bayesian",
                "objective_type": "Maximize",
                "objective_metric_name": "val-score",
                "max_jobs": 100,
                "max_parallel_jobs": 5,
                "parameter_ranges": SAMPLE_PARAM_RANGES,
            },
        )


def test_create_tuning_job_with_both_training_config_and_list(sagemaker_session):
    with pytest.raises(
        ValueError,
        match="Only one of training_config and training_config_list should be provided.",
    ):
        sagemaker_session.create_tuning_job(
            job_name="dummy-tuning-1",
            tuning_config={
                "strategy": "Bayesian",
                "objective_type": "Maximize",
                "objective_metric_name": "val-score",
                "max_jobs": 100,
                "max_parallel_jobs": 5,
                "parameter_ranges": SAMPLE_PARAM_RANGES,
            },
            training_config={
                "static_hyperparameters": STATIC_HPs,
                "image_uri": "dummy-image-1",
            },
            training_config_list=[
                {
                    "static_hyperparameters": STATIC_HPs,
                    "image_uri": "dummy-image-1",
                    "estimator_name": "estimator_1",
                },
                {
                    "static_hyperparameters": STATIC_HPs_2,
                    "image_uri": "dummy-image-2",
                    "estimator_name": "estimator_2",
                },
            ],
        )


def test_create_tuning_job(sagemaker_session):
    def assert_create_tuning_job_request(**kwrags):
        assert (
            kwrags["HyperParameterTuningJobConfig"]
            == SAMPLE_TUNING_JOB_REQUEST["HyperParameterTuningJobConfig"]
        )
        assert kwrags["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert kwrags["TrainingJobDefinition"] == SAMPLE_TUNING_JOB_REQUEST["TrainingJobDefinition"]
        assert "TrainingJobDefinitions" not in kwrags
        assert kwrags.get("WarmStartConfig", None) is None

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = (
        assert_create_tuning_job_request
    )
    sagemaker_session.create_tuning_job(
        job_name="dummy-tuning-1",
        tuning_config={
            "strategy": "Bayesian",
            "objective_type": "Maximize",
            "objective_metric_name": "val-score",
            "max_jobs": 100,
            "max_parallel_jobs": 5,
            "parameter_ranges": SAMPLE_PARAM_RANGES,
            "random_seed": 0,
        },
        training_config={
            "static_hyperparameters": STATIC_HPs,
            "image_uri": "dummy-image-1",
            "input_mode": "File",
            "metric_definitions": SAMPLE_METRIC_DEF,
            "role": EXPANDED_ROLE,
            "input_config": SAMPLE_INPUT,
            "output_config": SAMPLE_OUTPUT,
            "resource_config": RESOURCE_CONFIG,
            "stop_condition": SAMPLE_STOPPING_CONDITION,
            "environment": ENV_INPUT,
        },
        tags=None,
        warm_start_config=None,
    )


def test_create_tuning_job_multi_algo(sagemaker_session):
    def assert_create_tuning_job_request(**kwrags):
        expected_tuning_config = SAMPLE_MULTI_ALGO_TUNING_JOB_REQUEST[
            "HyperParameterTuningJobConfig"
        ]
        assert kwrags["HyperParameterTuningJobConfig"] == expected_tuning_config
        assert kwrags["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert "TrainingJobDefinition" not in kwrags
        assert (
            kwrags["TrainingJobDefinitions"]
            == SAMPLE_MULTI_ALGO_TUNING_JOB_REQUEST["TrainingJobDefinitions"]
        )
        assert kwrags.get("WarmStartConfig", None) is None

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = (
        assert_create_tuning_job_request
    )
    sagemaker_session.create_tuning_job(
        job_name="dummy-tuning-1",
        tuning_config={"strategy": "Bayesian", "max_jobs": 100, "max_parallel_jobs": 5},
        training_config_list=[
            {
                "static_hyperparameters": STATIC_HPs,
                "image_uri": "dummy-image-1",
                "input_mode": "File",
                "metric_definitions": SAMPLE_METRIC_DEF,
                "role": EXPANDED_ROLE,
                "input_config": SAMPLE_INPUT,
                "output_config": SAMPLE_OUTPUT,
                "resource_config": RESOURCE_CONFIG,
                "stop_condition": SAMPLE_STOPPING_CONDITION,
                "estimator_name": "estimator_1",
                "objective_type": "Maximize",
                "objective_metric_name": "val-score",
                "parameter_ranges": SAMPLE_PARAM_RANGES,
                "environment": ENV_INPUT,
            },
            {
                "static_hyperparameters": STATIC_HPs_2,
                "image_uri": "dummy-image-2",
                "input_mode": "File",
                "metric_definitions": SAMPLE_METRIC_DEF_2,
                "role": EXPANDED_ROLE,
                "input_config": SAMPLE_INPUT,
                "output_config": SAMPLE_OUTPUT,
                "resource_config": RESOURCE_CONFIG,
                "stop_condition": SAMPLE_STOPPING_CONDITION,
                "estimator_name": "estimator_2",
                "objective_type": "Maximize",
                "objective_metric_name": "value-score",
                "parameter_ranges": SAMPLE_PARAM_RANGES_2,
                "environment": ENV_INPUT,
            },
        ],
        tags=None,
        warm_start_config=None,
    )


def test_tune(sagemaker_session):
    def assert_create_tuning_job_request(**kwrags):
        assert (
            kwrags["HyperParameterTuningJobConfig"]
            == SAMPLE_TUNING_JOB_REQUEST["HyperParameterTuningJobConfig"]
        )
        assert kwrags["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert kwrags["TrainingJobDefinition"] == SAMPLE_TUNING_JOB_REQUEST["TrainingJobDefinition"]
        assert kwrags.get("WarmStartConfig", None) is None

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = (
        assert_create_tuning_job_request
    )
    sagemaker_session.tune(
        job_name="dummy-tuning-1",
        strategy="Bayesian",
        random_seed=0,
        objective_type="Maximize",
        objective_metric_name="val-score",
        max_jobs=100,
        max_parallel_jobs=5,
        parameter_ranges=SAMPLE_PARAM_RANGES,
        static_hyperparameters=STATIC_HPs,
        image_uri="dummy-image-1",
        input_mode="File",
        metric_definitions=SAMPLE_METRIC_DEF,
        role=EXPANDED_ROLE,
        input_config=SAMPLE_INPUT,
        output_config=SAMPLE_OUTPUT,
        resource_config=RESOURCE_CONFIG,
        stop_condition=SAMPLE_STOPPING_CONDITION,
        tags=None,
        warm_start_config=None,
        environment=ENV_INPUT,
    )


def test_tune_with_strategy_config(sagemaker_session):
    def assert_create_tuning_job_request(**kwrags):
        assert (
            kwrags["HyperParameterTuningJobConfig"]["StrategyConfig"]["HyperbandStrategyConfig"][
                "MinResource"
            ]
            == SAMPLE_HYPERBAND_STRATEGY_CONFIG["HyperbandStrategyConfig"]["MinResource"]
        )
        assert (
            kwrags["HyperParameterTuningJobConfig"]["StrategyConfig"]["HyperbandStrategyConfig"][
                "MaxResource"
            ]
            == SAMPLE_HYPERBAND_STRATEGY_CONFIG["HyperbandStrategyConfig"]["MaxResource"]
        )

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = (
        assert_create_tuning_job_request
    )
    sagemaker_session.tune(
        job_name="dummy-tuning-1",
        strategy="Bayesian",
        objective_type="Maximize",
        objective_metric_name="val-score",
        max_jobs=100,
        max_parallel_jobs=5,
        parameter_ranges=SAMPLE_PARAM_RANGES,
        static_hyperparameters=STATIC_HPs,
        image_uri="dummy-image-1",
        input_mode="File",
        metric_definitions=SAMPLE_METRIC_DEF,
        role=EXPANDED_ROLE,
        input_config=SAMPLE_INPUT,
        output_config=SAMPLE_OUTPUT,
        resource_config=RESOURCE_CONFIG,
        stop_condition=SAMPLE_STOPPING_CONDITION,
        tags=None,
        warm_start_config=None,
        strategy_config=SAMPLE_HYPERBAND_STRATEGY_CONFIG,
        environment=ENV_INPUT,
    )


def test_tune_with_encryption_flag(sagemaker_session):
    def assert_create_tuning_job_request(**kwrags):
        assert (
            kwrags["HyperParameterTuningJobConfig"]
            == SAMPLE_TUNING_JOB_REQUEST["HyperParameterTuningJobConfig"]
        )
        assert kwrags["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert kwrags["TrainingJobDefinition"]["EnableInterContainerTrafficEncryption"] is True
        assert kwrags.get("WarmStartConfig", None) is None

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = (
        assert_create_tuning_job_request
    )
    sagemaker_session.tune(
        job_name="dummy-tuning-1",
        strategy="Bayesian",
        random_seed=0,
        objective_type="Maximize",
        objective_metric_name="val-score",
        max_jobs=100,
        max_parallel_jobs=5,
        parameter_ranges=SAMPLE_PARAM_RANGES,
        static_hyperparameters=STATIC_HPs,
        image_uri="dummy-image-1",
        input_mode="File",
        metric_definitions=SAMPLE_METRIC_DEF,
        role=EXPANDED_ROLE,
        input_config=SAMPLE_INPUT,
        output_config=SAMPLE_OUTPUT,
        resource_config=RESOURCE_CONFIG,
        stop_condition=SAMPLE_STOPPING_CONDITION,
        tags=None,
        warm_start_config=None,
        encrypt_inter_container_traffic=True,
    )


def test_tune_with_spot_and_checkpoints(sagemaker_session):
    def assert_create_tuning_job_request(**kwargs):
        assert (
            kwargs["HyperParameterTuningJobConfig"]
            == SAMPLE_TUNING_JOB_REQUEST["HyperParameterTuningJobConfig"]
        )
        assert kwargs["HyperParameterTuningJobName"] == "dummy-tuning-1"
        assert kwargs["TrainingJobDefinition"]["EnableManagedSpotTraining"] is True
        assert (
            kwargs["TrainingJobDefinition"]["CheckpointConfig"]["S3Uri"]
            == "s3://mybucket/checkpoints/"
        )
        assert (
            kwargs["TrainingJobDefinition"]["CheckpointConfig"]["LocalPath"] == "/tmp/checkpoints"
        )
        assert kwargs.get("WarmStartConfig", None) is None

    sagemaker_session.sagemaker_client.create_hyper_parameter_tuning_job.side_effect = (
        assert_create_tuning_job_request
    )
    sagemaker_session.tune(
        job_name="dummy-tuning-1",
        strategy="Bayesian",
        random_seed=0,
        objective_type="Maximize",
        objective_metric_name="val-score",
        max_jobs=100,
        max_parallel_jobs=5,
        parameter_ranges=SAMPLE_PARAM_RANGES,
        static_hyperparameters=STATIC_HPs,
        image_uri="dummy-image-1",
        input_mode="File",
        metric_definitions=SAMPLE_METRIC_DEF,
        role=EXPANDED_ROLE,
        input_config=SAMPLE_INPUT,
        output_config=SAMPLE_OUTPUT,
        resource_config=RESOURCE_CONFIG,
        stop_condition=SAMPLE_STOPPING_CONDITION,
        tags=None,
        warm_start_config=None,
        use_spot_instances=True,
        checkpoint_s3_uri="s3://mybucket/checkpoints/",
        checkpoint_local_path="/tmp/checkpoints",
    )


def test_stop_tuning_job(sagemaker_session):
    sms = sagemaker_session
    sms.sagemaker_client.stop_hyper_parameter_tuning_job = Mock(
        name="stop_hyper_parameter_tuning_job"
    )

    sagemaker_session.stop_tuning_job(JOB_NAME)
    sms.sagemaker_client.stop_hyper_parameter_tuning_job.assert_called_once_with(
        HyperParameterTuningJobName=JOB_NAME
    )


def test_stop_tuning_job_client_error_already_stopped(sagemaker_session):
    sms = sagemaker_session
    exception = ClientError({"Error": {"Code": "ValidationException"}}, "Operation")
    sms.sagemaker_client.stop_hyper_parameter_tuning_job = Mock(
        name="stop_hyper_parameter_tuning_job", side_effect=exception
    )
    sagemaker_session.stop_tuning_job(JOB_NAME)

    sms.sagemaker_client.stop_hyper_parameter_tuning_job.assert_called_once_with(
        HyperParameterTuningJobName=JOB_NAME
    )


def test_stop_tuning_job_client_error(sagemaker_session):
    error_response = {"Error": {"Code": "MockException", "Message": "MockMessage"}}
    operation = "Operation"
    exception = ClientError(error_response, operation)

    sms = sagemaker_session
    sms.sagemaker_client.stop_hyper_parameter_tuning_job = Mock(
        name="stop_hyper_parameter_tuning_job", side_effect=exception
    )

    with pytest.raises(ClientError) as e:
        sagemaker_session.stop_tuning_job(JOB_NAME)

    sms.sagemaker_client.stop_hyper_parameter_tuning_job.assert_called_once_with(
        HyperParameterTuningJobName=JOB_NAME
    )
    assert (
        "An error occurred (MockException) when calling the Operation operation: MockMessage"
        in str(e)
    )


def test_train_with_sagemaker_config_injection_custom_profiler_config(sagemaker_session):
    """
    Tests disableProfiler property is overridden by custom profiler_config with property set
    :param sagemaker_session:
    :return: None
    """
    custom_config = copy.deepcopy(SAGEMAKER_CONFIG_TRAINING_JOB)
    custom_config["SageMaker"]["TrainingJob"]["ProfilerConfig"]["DisableProfiler"] = True

    sagemaker_session.sagemaker_config = custom_config
    in_config = [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        }
    ]

    out_config = {"S3OutputPath": S3_OUTPUT}

    resource_config = {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": MAX_SIZE,
    }

    stop_cond = {"MaxRuntimeInSeconds": MAX_TIME}
    RETRY_STRATEGY = {"MaximumRetryAttempts": 2}
    hyperparameters = {"foo": "bar"}
    TRAINING_IMAGE_CONFIG = {
        "TrainingRepositoryAccessMode": "Vpc",
        "TrainingRepositoryAuthConfig": {
            "TrainingRepositoryCredentialsProviderArn": "arn:aws:lambda:us-west-2:1234567897:function:test"
        },
    }

    profiler_config = {"DisableProfiler": False}
    sagemaker_session.train(
        image_uri=IMAGE,
        input_mode="File",
        input_config=in_config,
        job_name=JOB_NAME,
        output_config=out_config,
        resource_config=resource_config,
        profiler_config=profiler_config,
        hyperparameters=hyperparameters,
        stop_condition=stop_cond,
        metric_definitions=METRIC_DEFINITONS,
        use_spot_instances=True,
        checkpoint_s3_uri="s3://mybucket/checkpoints/",
        checkpoint_local_path="/tmp/checkpoints",
        enable_sagemaker_metrics=True,
        retry_strategy=RETRY_STRATEGY,
        training_image_config=TRAINING_IMAGE_CONFIG,
    )

    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]

    assert actual_train_args["ProfilerConfig"]["DisableProfiler"] is False


def test_train_with_sagemaker_config_injection_empty_profiler_config(sagemaker_session):
    """
    Tests disableProfiler property is injected into a profiler_config object when the property does not exist explicitly
    :param sagemaker_session:
    :return: None
    """
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    in_config = [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        }
    ]

    out_config = {"S3OutputPath": S3_OUTPUT}

    resource_config = {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": MAX_SIZE,
    }

    stop_cond = {"MaxRuntimeInSeconds": MAX_TIME}
    RETRY_STRATEGY = {"MaximumRetryAttempts": 2}
    hyperparameters = {"foo": "bar"}
    TRAINING_IMAGE_CONFIG = {
        "TrainingRepositoryAccessMode": "Vpc",
        "TrainingRepositoryAuthConfig": {
            "TrainingRepositoryCredentialsProviderArn": "arn:aws:lambda:us-west-2:1234567897:function:test"
        },
    }

    profiler_config = {}
    sagemaker_session.train(
        image_uri=IMAGE,
        input_mode="File",
        input_config=in_config,
        job_name=JOB_NAME,
        output_config=out_config,
        resource_config=resource_config,
        profiler_config=profiler_config,
        hyperparameters=hyperparameters,
        stop_condition=stop_cond,
        metric_definitions=METRIC_DEFINITONS,
        use_spot_instances=True,
        checkpoint_s3_uri="s3://mybucket/checkpoints/",
        checkpoint_local_path="/tmp/checkpoints",
        enable_sagemaker_metrics=True,
        retry_strategy=RETRY_STRATEGY,
        training_image_config=TRAINING_IMAGE_CONFIG,
    )

    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]

    expected_disable_profiler_attribute = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]["DisableProfiler"]
    assert (
        actual_train_args["ProfilerConfig"]["DisableProfiler"]
        == expected_disable_profiler_attribute
    )


def test_train_with_sagemaker_config_injection_partial_profiler_config(sagemaker_session):
    """
    Tests disableProfiler property is injected into a profiler_config object when the property does not exist explicitly
    :param sagemaker_session:
    :return: None
    """
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    in_config = [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        }
    ]

    out_config = {"S3OutputPath": S3_OUTPUT}

    resource_config = {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": MAX_SIZE,
    }

    stop_cond = {"MaxRuntimeInSeconds": MAX_TIME}
    RETRY_STRATEGY = {"MaximumRetryAttempts": 2}
    hyperparameters = {"foo": "bar"}
    TRAINING_IMAGE_CONFIG = {
        "TrainingRepositoryAccessMode": "Vpc",
        "TrainingRepositoryAuthConfig": {
            "TrainingRepositoryCredentialsProviderArn": "arn:aws:lambda:us-west-2:1234567897:function:test"
        },
    }

    profiler_config = {
        "ProfilingIntervalInMilliseconds": 1000,
        "S3OutputPath": "s3://mybucket/myoutputbucket",
        "ProfilingParameters": {},
    }
    sagemaker_session.train(
        image_uri=IMAGE,
        input_mode="File",
        input_config=in_config,
        job_name=JOB_NAME,
        output_config=out_config,
        resource_config=resource_config,
        profiler_config=profiler_config,
        hyperparameters=hyperparameters,
        stop_condition=stop_cond,
        metric_definitions=METRIC_DEFINITONS,
        use_spot_instances=True,
        checkpoint_s3_uri="s3://mybucket/checkpoints/",
        checkpoint_local_path="/tmp/checkpoints",
        enable_sagemaker_metrics=True,
        retry_strategy=RETRY_STRATEGY,
        training_image_config=TRAINING_IMAGE_CONFIG,
    )

    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]

    expected_disable_profiler_attribute = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]["DisableProfiler"]
    assert (
        actual_train_args["ProfilerConfig"]["DisableProfiler"]
        == expected_disable_profiler_attribute
    )
    assert actual_train_args["ProfilerConfig"]["ProfilingIntervalInMilliseconds"] == 1000
    assert actual_train_args["ProfilerConfig"]["S3OutputPath"] == "s3://mybucket/myoutputbucket"


def test_update_training_job_without_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.update_training_job(job_name="MyTestJob")
    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]
    assert actual_train_args["TrainingJobName"] == "MyTestJob"


def test_update_training_job_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB
    sagemaker_session.update_training_job(job_name="MyTestJob")
    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]
    expected_disable_profiler_attribute = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]["DisableProfiler"]
    assert (
        actual_train_args["ProfilerConfig"]["DisableProfiler"]
        == expected_disable_profiler_attribute
    )


def test_update_training_job_with_remote_debug_config(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB
    sagemaker_session.update_training_job(
        job_name="MyTestJob", remote_debug_config={"EnableRemoteDebug": False}
    )
    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]
    assert not actual_train_args["RemoteDebugConfig"]["EnableRemoteDebug"]


def test_train_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    in_config = [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        }
    ]

    out_config = {"S3OutputPath": S3_OUTPUT}

    resource_config = {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": MAX_SIZE,
    }

    stop_cond = {"MaxRuntimeInSeconds": MAX_TIME}
    RETRY_STRATEGY = {"MaximumRetryAttempts": 2}
    hyperparameters = {"foo": "bar"}
    TRAINING_IMAGE_CONFIG = {
        "TrainingRepositoryAccessMode": "Vpc",
        "TrainingRepositoryAuthConfig": {
            "TrainingRepositoryCredentialsProviderArn": "arn:aws:lambda:us-west-2:1234567897:function:test"
        },
    }
    INFRA_CHECK_CONFIG = {"EnableInfraCheck": True}
    CONTAINER_ENTRY_POINT = ["bin/bash", "test.sh"]
    CONTAINER_ARGUMENTS = ["--arg1", "value1", "--arg2", "value2"]

    sagemaker_session.train(
        image_uri=IMAGE,
        input_mode="File",
        input_config=in_config,
        job_name=JOB_NAME,
        output_config=out_config,
        resource_config=resource_config,
        hyperparameters=hyperparameters,
        stop_condition=stop_cond,
        metric_definitions=METRIC_DEFINITONS,
        use_spot_instances=True,
        checkpoint_s3_uri="s3://mybucket/checkpoints/",
        checkpoint_local_path="/tmp/checkpoints",
        enable_sagemaker_metrics=True,
        retry_strategy=RETRY_STRATEGY,
        training_image_config=TRAINING_IMAGE_CONFIG,
        container_entry_point=CONTAINER_ENTRY_POINT,
        container_arguments=CONTAINER_ARGUMENTS,
        infra_check_config=INFRA_CHECK_CONFIG,
    )

    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]

    expected_volume_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ResourceConfig"
    ]["VolumeKmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "OutputDataConfig"
    ]["KmsKeyId"]
    expected_vpc_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["VpcConfig"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "EnableNetworkIsolation"
    ]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"][
        "TrainingJob"
    ]["EnableInterContainerTrafficEncryption"]
    expected_tags = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["Tags"]
    expected_environment = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["Environment"]
    expected_profiler_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]

    assert actual_train_args["VpcConfig"] == expected_vpc_config
    assert actual_train_args["HyperParameters"] == hyperparameters
    assert actual_train_args["Tags"] == expected_tags
    assert actual_train_args["AlgorithmSpecification"]["MetricDefinitions"] == METRIC_DEFINITONS
    assert actual_train_args["AlgorithmSpecification"]["EnableSageMakerMetricsTimeSeries"] is True
    assert (
        actual_train_args["EnableInterContainerTrafficEncryption"]
        == expected_enable_inter_container_traffic_encryption
    )
    assert actual_train_args["EnableNetworkIsolation"] == expected_enable_network_isolation
    assert actual_train_args["EnableManagedSpotTraining"] is True
    assert actual_train_args["CheckpointConfig"]["S3Uri"] == "s3://mybucket/checkpoints/"
    assert actual_train_args["CheckpointConfig"]["LocalPath"] == "/tmp/checkpoints"
    assert actual_train_args["Environment"] == expected_environment
    assert actual_train_args["RetryStrategy"] == RETRY_STRATEGY
    assert (
        actual_train_args["AlgorithmSpecification"]["TrainingImageConfig"] == TRAINING_IMAGE_CONFIG
    )
    assert (
        actual_train_args["AlgorithmSpecification"]["ContainerEntrypoint"] == CONTAINER_ENTRY_POINT
    )
    assert actual_train_args["AlgorithmSpecification"]["ContainerArguments"] == CONTAINER_ARGUMENTS
    assert actual_train_args["InfraCheckConfig"] == INFRA_CHECK_CONFIG
    assert actual_train_args["RoleArn"] == expected_role_arn
    assert actual_train_args["ResourceConfig"] == {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": MAX_SIZE,
        "VolumeKmsKeyId": expected_volume_kms_key_id,
    }
    assert actual_train_args["ProfilerConfig"] == expected_profiler_config
    assert actual_train_args["OutputDataConfig"] == {
        "S3OutputPath": S3_OUTPUT,
        "KmsKeyId": expected_kms_key_id,
    }


def test_train_with_sagemaker_config_injection_no_kms_support(sagemaker_session):
    """Tests that when the instance type for train does not support KMS, no KMS key is used."""

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    in_config = [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        }
    ]

    out_config = {"S3OutputPath": S3_OUTPUT}

    resource_config = {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": "ml.g5.12xlarge",
    }

    stop_cond = {"MaxRuntimeInSeconds": MAX_TIME}
    RETRY_STRATEGY = {"MaximumRetryAttempts": 2}
    hyperparameters = {"foo": "bar"}
    TRAINING_IMAGE_CONFIG = {
        "TrainingRepositoryAccessMode": "Vpc",
        "TrainingRepositoryAuthConfig": {
            "TrainingRepositoryCredentialsProviderArn": "arn:aws:lambda:us-west-2:1234567897:function:test"
        },
    }

    sagemaker_session.train(
        image_uri=IMAGE,
        input_mode="File",
        input_config=in_config,
        job_name=JOB_NAME,
        output_config=out_config,
        resource_config=resource_config,
        hyperparameters=hyperparameters,
        stop_condition=stop_cond,
        metric_definitions=METRIC_DEFINITONS,
        use_spot_instances=True,
        checkpoint_s3_uri="s3://mybucket/checkpoints/",
        checkpoint_local_path="/tmp/checkpoints",
        enable_sagemaker_metrics=True,
        environment=ENV_INPUT,
        retry_strategy=RETRY_STRATEGY,
        training_image_config=TRAINING_IMAGE_CONFIG,
    )

    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]

    expected_role_arn = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "OutputDataConfig"
    ]["KmsKeyId"]
    expected_vpc_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["VpcConfig"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "EnableNetworkIsolation"
    ]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"][
        "TrainingJob"
    ]["EnableInterContainerTrafficEncryption"]
    expected_tags = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["Tags"]
    expected_profiler_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]

    assert actual_train_args["VpcConfig"] == expected_vpc_config
    assert actual_train_args["HyperParameters"] == hyperparameters
    assert actual_train_args["Tags"] == expected_tags
    assert actual_train_args["AlgorithmSpecification"]["MetricDefinitions"] == METRIC_DEFINITONS
    assert actual_train_args["AlgorithmSpecification"]["EnableSageMakerMetricsTimeSeries"] is True
    assert (
        actual_train_args["EnableInterContainerTrafficEncryption"]
        == expected_enable_inter_container_traffic_encryption
    )
    assert actual_train_args["EnableNetworkIsolation"] == expected_enable_network_isolation
    assert actual_train_args["EnableManagedSpotTraining"] is True
    assert actual_train_args["CheckpointConfig"]["S3Uri"] == "s3://mybucket/checkpoints/"
    assert actual_train_args["CheckpointConfig"]["LocalPath"] == "/tmp/checkpoints"
    assert actual_train_args["Environment"] == ENV_INPUT
    assert actual_train_args["RetryStrategy"] == RETRY_STRATEGY
    assert (
        actual_train_args["AlgorithmSpecification"]["TrainingImageConfig"] == TRAINING_IMAGE_CONFIG
    )
    assert actual_train_args["RoleArn"] == expected_role_arn
    assert actual_train_args["ResourceConfig"] == {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": "ml.g5.12xlarge",
    }
    assert actual_train_args["ProfilerConfig"] == expected_profiler_config
    assert actual_train_args["OutputDataConfig"] == {
        "S3OutputPath": S3_OUTPUT,
        "KmsKeyId": expected_kms_key_id,
    }


def test_train_pack_to_request_with_optional_params(sagemaker_session):
    in_config = [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        }
    ]

    out_config = {"S3OutputPath": S3_OUTPUT}

    resource_config = {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": MAX_SIZE,
    }

    stop_cond = {"MaxRuntimeInSeconds": MAX_TIME}
    RETRY_STRATEGY = {"MaximumRetryAttempts": 2}
    hyperparameters = {"foo": "bar"}
    TRAINING_IMAGE_CONFIG = {
        "TrainingRepositoryAccessMode": "Vpc",
        "TrainingRepositoryAuthConfig": {
            "TrainingRepositoryCredentialsProviderArn": "arn:aws:lambda:us-west-2:1234567897:function:test"
        },
    }
    CONTAINER_ENTRY_POINT = ["bin/bash", "test.sh"]
    CONTAINER_ARGUMENTS = ["--arg1", "value1", "--arg2", "value2"]
    remote_debug_config = {"EnableRemoteDebug": True}

    sagemaker_session.train(
        image_uri=IMAGE,
        input_mode="File",
        input_config=in_config,
        role=EXPANDED_ROLE,
        job_name=JOB_NAME,
        output_config=out_config,
        resource_config=resource_config,
        vpc_config=VPC_CONFIG,
        hyperparameters=hyperparameters,
        stop_condition=stop_cond,
        tags=TAGS,
        metric_definitions=METRIC_DEFINITONS,
        encrypt_inter_container_traffic=True,
        use_spot_instances=True,
        checkpoint_s3_uri="s3://mybucket/checkpoints/",
        checkpoint_local_path="/tmp/checkpoints",
        enable_sagemaker_metrics=True,
        environment=ENV_INPUT,
        retry_strategy=RETRY_STRATEGY,
        training_image_config=TRAINING_IMAGE_CONFIG,
        container_entry_point=CONTAINER_ENTRY_POINT,
        container_arguments=CONTAINER_ARGUMENTS,
        remote_debug_config=remote_debug_config,
    )

    _, _, actual_train_args = sagemaker_session.sagemaker_client.method_calls[0]

    assert actual_train_args["VpcConfig"] == VPC_CONFIG
    assert actual_train_args["HyperParameters"] == hyperparameters
    assert actual_train_args["Tags"] == TAGS
    assert actual_train_args["AlgorithmSpecification"]["MetricDefinitions"] == METRIC_DEFINITONS
    assert actual_train_args["AlgorithmSpecification"]["EnableSageMakerMetricsTimeSeries"] is True
    assert actual_train_args["EnableInterContainerTrafficEncryption"] is True
    assert actual_train_args["EnableManagedSpotTraining"] is True
    assert actual_train_args["CheckpointConfig"]["S3Uri"] == "s3://mybucket/checkpoints/"
    assert actual_train_args["CheckpointConfig"]["LocalPath"] == "/tmp/checkpoints"
    assert actual_train_args["Environment"] == ENV_INPUT
    assert actual_train_args["RetryStrategy"] == RETRY_STRATEGY
    assert (
        actual_train_args["AlgorithmSpecification"]["TrainingImageConfig"] == TRAINING_IMAGE_CONFIG
    )
    assert (
        actual_train_args["AlgorithmSpecification"]["ContainerEntrypoint"] == CONTAINER_ENTRY_POINT
    )
    assert actual_train_args["AlgorithmSpecification"]["ContainerArguments"] == CONTAINER_ARGUMENTS
    assert actual_train_args["RemoteDebugConfig"]["EnableRemoteDebug"]


def test_create_transform_job_with_sagemaker_config_injection(sagemaker_session):
    # Config to test injection for
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRANSFORM_JOB

    model_name = "my-model"
    in_config = {
        "CompressionType": "None",
        "ContentType": "text/csv",
        "SplitType": "None",
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI}},
    }
    out_config = {"S3OutputPath": S3_OUTPUT}
    resource_config = {"InstanceCount": INSTANCE_COUNT, "InstanceType": INSTANCE_TYPE}
    data_processing = {"OutputFilter": "$", "InputFilter": "$", "JoinSource": "Input"}
    data_capture_config = BatchDataCaptureConfig(destination_s3_uri="s3://test")

    # important to deepcopy, otherwise the original dicts are modified when we add expected params
    expected_args = copy.deepcopy(
        {
            "TransformJobName": JOB_NAME,
            "ModelName": model_name,
            "TransformInput": in_config,
            "TransformOutput": out_config,
            "TransformResources": resource_config,
            "DataProcessing": data_processing,
            "DataCaptureConfig": data_capture_config._to_request_dict(),
            "Tags": TAGS,
        }
    )
    # The following parameters should be fetched from config
    expected_tags = SAGEMAKER_CONFIG_TRANSFORM_JOB["SageMaker"]["TransformJob"]["Tags"]
    expected_args["DataCaptureConfig"]["KmsKeyId"] = SAGEMAKER_CONFIG_TRANSFORM_JOB["SageMaker"][
        "TransformJob"
    ]["DataCaptureConfig"]["KmsKeyId"]
    expected_args["TransformOutput"]["KmsKeyId"] = SAGEMAKER_CONFIG_TRANSFORM_JOB["SageMaker"][
        "TransformJob"
    ]["TransformOutput"]["KmsKeyId"]
    expected_args["TransformResources"]["VolumeKmsKeyId"] = SAGEMAKER_CONFIG_TRANSFORM_JOB[
        "SageMaker"
    ]["TransformJob"]["TransformResources"]["VolumeKmsKeyId"]
    expected_args["Environment"] = SAGEMAKER_CONFIG_TRANSFORM_JOB["SageMaker"]["TransformJob"][
        "Environment"
    ]

    # make sure the original dicts were not modified before config injection
    assert "KmsKeyId" not in in_config
    assert "KmsKeyId" not in out_config
    assert "VolumeKmsKeyId" not in resource_config

    # injection should happen during this method
    sagemaker_session.transform(
        job_name=JOB_NAME,
        model_name=model_name,
        strategy=None,
        max_concurrent_transforms=None,
        max_payload=None,
        env=None,
        input_config=in_config,
        output_config=out_config,
        resource_config=resource_config,
        experiment_config=None,
        model_client_config=None,
        tags=expected_tags,
        data_processing=data_processing,
        batch_data_capture_config=data_capture_config,
    )

    _, _, actual_args = sagemaker_session.sagemaker_client.method_calls[0]
    assert actual_args == expected_args


def test_transform_pack_to_request(sagemaker_session):
    model_name = "my-model"

    in_config = {
        "CompressionType": "None",
        "ContentType": "text/csv",
        "SplitType": "None",
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI}},
    }

    out_config = {"S3OutputPath": S3_OUTPUT}

    resource_config = {"InstanceCount": INSTANCE_COUNT, "InstanceType": INSTANCE_TYPE}

    data_processing = {"OutputFilter": "$", "InputFilter": "$", "JoinSource": "Input"}

    expected_args = {
        "TransformJobName": JOB_NAME,
        "ModelName": model_name,
        "TransformInput": in_config,
        "TransformOutput": out_config,
        "TransformResources": resource_config,
        "DataProcessing": data_processing,
    }

    sagemaker_session.transform(
        job_name=JOB_NAME,
        model_name=model_name,
        strategy=None,
        max_concurrent_transforms=None,
        max_payload=None,
        env=None,
        input_config=in_config,
        output_config=out_config,
        resource_config=resource_config,
        experiment_config=None,
        model_client_config=None,
        tags=None,
        data_processing=data_processing,
        batch_data_capture_config=None,
    )

    _, _, actual_args = sagemaker_session.sagemaker_client.method_calls[0]
    assert actual_args == expected_args


def test_transform_pack_to_request_with_optional_params(sagemaker_session):
    strategy = "strategy"
    max_concurrent_transforms = 1
    max_payload = 0
    env = {"FOO": "BAR"}

    batch_data_capture_config = BatchDataCaptureConfig(
        destination_s3_uri="test_uri",
        kms_key_id="",
        generate_inference_id=False,
    )

    sagemaker_session.transform(
        job_name=JOB_NAME,
        model_name="my-model",
        strategy=strategy,
        max_concurrent_transforms=max_concurrent_transforms,
        env=env,
        max_payload=max_payload,
        input_config={},
        output_config={},
        resource_config={},
        experiment_config=EXPERIMENT_CONFIG,
        model_client_config=MODEL_CLIENT_CONFIG,
        tags=TAGS,
        data_processing=None,
        batch_data_capture_config=batch_data_capture_config,
    )

    _, _, actual_args = sagemaker_session.sagemaker_client.method_calls[0]
    assert actual_args["BatchStrategy"] == strategy
    assert actual_args["MaxConcurrentTransforms"] == max_concurrent_transforms
    assert actual_args["MaxPayloadInMB"] == max_payload
    assert actual_args["Environment"] == env
    assert actual_args["Tags"] == TAGS
    assert actual_args["ExperimentConfig"] == EXPERIMENT_CONFIG
    assert actual_args["ModelClientConfig"] == MODEL_CLIENT_CONFIG
    assert actual_args["DataCaptureConfig"] == batch_data_capture_config._to_request_dict()


@patch("sys.stdout", new_callable=io.BytesIO if six.PY2 else io.StringIO)
def test_color_wrap(bio):
    color_wrap = sagemaker.logs.ColorWrap()
    color_wrap(0, "hi there")
    assert bio.getvalue() == "hi there\n"


class MockBotoException(ClientError):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


DEFAULT_LOG_STREAMS = {"logStreams": [{"logStreamName": JOB_NAME + "/xxxxxxxxx"}]}
LIFECYCLE_LOG_STREAMS = [
    MockBotoException("ResourceNotFoundException"),
    DEFAULT_LOG_STREAMS,
    DEFAULT_LOG_STREAMS,
    DEFAULT_LOG_STREAMS,
    DEFAULT_LOG_STREAMS,
    DEFAULT_LOG_STREAMS,
    DEFAULT_LOG_STREAMS,
]

DEFAULT_LOG_EVENTS = [
    {"nextForwardToken": None, "events": [{"timestamp": 1, "message": "hi there #1"}]},
    {"nextForwardToken": None, "events": []},
]
STREAM_LOG_EVENTS = [
    {"nextForwardToken": None, "events": [{"timestamp": 1, "message": "hi there #1"}]},
    {"nextForwardToken": None, "events": []},
    {
        "nextForwardToken": None,
        "events": [
            {"timestamp": 1, "message": "hi there #1"},
            {"timestamp": 2, "message": "hi there #2"},
        ],
    },
    {"nextForwardToken": None, "events": []},
    {
        "nextForwardToken": None,
        "events": [
            {"timestamp": 2, "message": "hi there #2"},
            {"timestamp": 2, "message": "hi there #2a"},
            {"timestamp": 3, "message": "hi there #3"},
        ],
    },
    {"nextForwardToken": None, "events": []},
]


@pytest.fixture()
def boto_session_complete():
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("logs").describe_log_streams.return_value = DEFAULT_LOG_STREAMS
    boto_mock.client("logs").get_log_events.side_effect = DEFAULT_LOG_EVENTS
    boto_mock.client("sagemaker").describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    boto_mock.client(
        "sagemaker"
    ).describe_transform_job.return_value = COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT
    return boto_mock


@pytest.fixture()
def sagemaker_session_complete(boto_session_complete):
    ims = sagemaker.Session(boto_session=boto_session_complete, sagemaker_client=MagicMock())
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.sagemaker_client.describe_transform_job.return_value = (
        COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT
    )
    return ims


@pytest.fixture()
def boto_session_stopped():
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("logs").describe_log_streams.return_value = DEFAULT_LOG_STREAMS
    boto_mock.client("logs").get_log_events.side_effect = DEFAULT_LOG_EVENTS
    boto_mock.client("sagemaker").describe_training_job.return_value = STOPPED_DESCRIBE_JOB_RESULT
    boto_mock.client(
        "sagemaker"
    ).describe_transform_job.return_value = STOPPED_DESCRIBE_TRANSFORM_JOB_RESULT
    return boto_mock


@pytest.fixture()
def sagemaker_session_stopped(boto_session_stopped):
    ims = sagemaker.Session(boto_session=boto_session_stopped, sagemaker_client=MagicMock())
    ims.sagemaker_client.describe_training_job.return_value = STOPPED_DESCRIBE_JOB_RESULT
    ims.sagemaker_client.describe_transform_job.return_value = STOPPED_DESCRIBE_TRANSFORM_JOB_RESULT
    return ims


@pytest.fixture()
def boto_session_ready_lifecycle():
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("logs").describe_log_streams.return_value = DEFAULT_LOG_STREAMS
    boto_mock.client("logs").get_log_events.side_effect = STREAM_LOG_EVENTS
    boto_mock.client("sagemaker").describe_training_job.side_effect = [
        IN_PROGRESS_DESCRIBE_JOB_RESULT,
        IN_PROGRESS_DESCRIBE_JOB_RESULT,
        COMPLETED_DESCRIBE_JOB_RESULT,
    ]
    boto_mock.client("sagemaker").describe_transform_job.side_effect = [
        IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT,
        IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT,
        COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT,
    ]
    return boto_mock


@pytest.fixture()
def sagemaker_session_ready_lifecycle(boto_session_ready_lifecycle):
    ims = sagemaker.Session(boto_session=boto_session_ready_lifecycle, sagemaker_client=MagicMock())
    ims.sagemaker_client.describe_training_job.side_effect = [
        IN_PROGRESS_DESCRIBE_JOB_RESULT,
        IN_PROGRESS_DESCRIBE_JOB_RESULT,
        COMPLETED_DESCRIBE_JOB_RESULT,
    ]
    ims.sagemaker_client.describe_transform_job.side_effect = [
        IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT,
        IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT,
        COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT,
    ]
    return ims


@pytest.fixture()
def boto_session_full_lifecycle():
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("logs").describe_log_streams.side_effect = LIFECYCLE_LOG_STREAMS
    boto_mock.client("logs").get_log_events.side_effect = STREAM_LOG_EVENTS
    boto_mock.client("sagemaker").describe_training_job.side_effect = [
        IN_PROGRESS_DESCRIBE_JOB_RESULT,
        IN_PROGRESS_DESCRIBE_JOB_RESULT,
        COMPLETED_DESCRIBE_JOB_RESULT,
    ]
    boto_mock.client("sagemaker").sagemaker_client.describe_transform_job.side_effect = [
        IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT,
        IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT,
        COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT,
    ]
    return boto_mock


@pytest.fixture()
def sagemaker_session_full_lifecycle(boto_session_full_lifecycle):
    ims = sagemaker.Session(boto_session=boto_session_full_lifecycle, sagemaker_client=MagicMock())
    ims.sagemaker_client.describe_training_job.side_effect = [
        IN_PROGRESS_DESCRIBE_JOB_RESULT,
        IN_PROGRESS_DESCRIBE_JOB_RESULT,
        COMPLETED_DESCRIBE_JOB_RESULT,
    ]
    ims.sagemaker_client.describe_transform_job.side_effect = [
        IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT,
        IN_PROGRESS_DESCRIBE_TRANSFORM_JOB_RESULT,
        COMPLETED_DESCRIBE_TRANSFORM_JOB_RESULT,
    ]
    return ims


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_job_no_wait(cw, sagemaker_session_complete):
    ims = sagemaker_session_complete
    ims.logs_for_job(JOB_NAME)
    ims.sagemaker_client.describe_training_job.assert_called_once_with(TrainingJobName=JOB_NAME)
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_job_no_wait_stopped_job(cw, sagemaker_session_stopped):
    ims = sagemaker_session_stopped
    ims.logs_for_job(JOB_NAME)
    ims.sagemaker_client.describe_training_job.assert_called_once_with(TrainingJobName=JOB_NAME)
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_job_wait_on_completed(cw, sagemaker_session_complete):
    ims = sagemaker_session_complete
    ims.logs_for_job(JOB_NAME, wait=True, poll=0)
    assert ims.sagemaker_client.describe_training_job.call_args_list == [
        call(TrainingJobName=JOB_NAME)
    ]
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_job_wait_on_stopped(cw, sagemaker_session_stopped):
    ims = sagemaker_session_stopped
    ims.logs_for_job(JOB_NAME, wait=True, poll=0)
    assert ims.sagemaker_client.describe_training_job.call_args_list == [
        call(TrainingJobName=JOB_NAME)
    ]
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_job_no_wait_on_running(cw, sagemaker_session_ready_lifecycle):
    ims = sagemaker_session_ready_lifecycle
    ims.logs_for_job(JOB_NAME)
    assert ims.sagemaker_client.describe_training_job.call_args_list == [
        call(TrainingJobName=JOB_NAME)
    ]
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
@patch("time.time", side_effect=[0, 30, 60, 90, 120, 150, 180])
def test_logs_for_job_full_lifecycle(time, cw, sagemaker_session_full_lifecycle):
    ims = sagemaker_session_full_lifecycle
    ims.logs_for_job(JOB_NAME, wait=True, poll=0)
    assert (
        ims.sagemaker_client.describe_training_job.call_args_list
        == [call(TrainingJobName=JOB_NAME)] * 3
    )
    assert cw().call_args_list == [
        call(0, "hi there #1"),
        call(0, "hi there #2"),
        call(0, "hi there #2a"),
        call(0, "hi there #3"),
    ]


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_transform_job_no_wait(cw, sagemaker_session_complete):
    ims = sagemaker_session_complete
    ims.logs_for_transform_job(JOB_NAME)
    ims.sagemaker_client.describe_transform_job.assert_called_once_with(TransformJobName=JOB_NAME)
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_transform_job_no_wait_stopped_job(cw, sagemaker_session_stopped):
    ims = sagemaker_session_stopped
    ims.logs_for_transform_job(JOB_NAME)
    ims.sagemaker_client.describe_transform_job.assert_called_once_with(TransformJobName=JOB_NAME)
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_transform_job_wait_on_completed(cw, sagemaker_session_complete):
    ims = sagemaker_session_complete
    ims.logs_for_transform_job(JOB_NAME, wait=True, poll=0)
    assert ims.sagemaker_client.describe_transform_job.call_args_list == [
        call(TransformJobName=JOB_NAME)
    ]
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_transform_job_wait_on_stopped(cw, sagemaker_session_stopped):
    ims = sagemaker_session_stopped
    ims.logs_for_transform_job(JOB_NAME, wait=True, poll=0)
    assert ims.sagemaker_client.describe_transform_job.call_args_list == [
        call(TransformJobName=JOB_NAME)
    ]
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
def test_logs_for_transform_job_no_wait_on_running(cw, sagemaker_session_ready_lifecycle):
    ims = sagemaker_session_ready_lifecycle
    ims.logs_for_transform_job(JOB_NAME)
    assert ims.sagemaker_client.describe_transform_job.call_args_list == [
        call(TransformJobName=JOB_NAME)
    ]
    cw().assert_called_with(0, "hi there #1")


@patch("sagemaker.logs.ColorWrap")
@patch("time.time", side_effect=[0, 30, 60, 90, 120, 150, 180])
def test_logs_for_transform_job_full_lifecycle(time, cw, sagemaker_session_full_lifecycle):
    ims = sagemaker_session_full_lifecycle
    ims.logs_for_transform_job(JOB_NAME, wait=True, poll=0)
    assert (
        ims.sagemaker_client.describe_transform_job.call_args_list
        == [call(TransformJobName=JOB_NAME)] * 3
    )
    assert cw().call_args_list == [
        call(0, "hi there #1"),
        call(0, "hi there #2"),
        call(0, "hi there #2a"),
        call(0, "hi there #3"),
    ]


MODEL_NAME = "some-model"
CONTAINERS = [
    {
        "Image": "mi-1",
        "ModelDataUrl": "s3://bucket/model_1.tar.gz",
        "Framework": "TENSORFLOW",
        "FrameworkVersion": "2.9",
        "NearestModelName": "resnet50",
        "ModelInput": {
            "DataInputConfig": '{"input_1":[1,224,224,3]}',
        },
    },
    {"Image": "mi-2", "ModelDataUrl": "s3://bucket/model_2.tar.gz"},
]
PRIMARY_CONTAINER = {
    "Environment": {},
    "Image": IMAGE,
    "ModelDataUrl": "s3://sagemaker-123/output/jobname/model/model.tar.gz",
}
PRIMARY_CONTAINER_WITH_COMPRESSED_S3_MODEL = {
    "Environment": {},
    "Image": IMAGE,
    "ModelDataSource": {
        "S3DataSource": {
            "S3Uri": "s3://sagemaker-123/output/jobname/model/model.tar.gz",
            "S3DataType": "S3Object",
            "CompressionType": "Gzip",
        }
    },
}
PRIMARY_CONTAINER_WITH_UNCOMPRESSED_S3_MODEL = {
    "Environment": {},
    "Image": IMAGE,
    "ModelDataSource": {
        "S3DataSource": {
            # expect model data URI has trailing forward slash appended
            "S3Uri": "s3://sagemaker-123/output/jobname/model/prefix/",
            "S3DataType": "S3Prefix",
            "CompressionType": "None",
        }
    },
}


def test_create_model_with_sagemaker_config_injection_with_primary_container(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_MODEL_WITH_PRIMARY_CONTAINER

    sagemaker_session.expand_role = Mock(
        name="expand_role", side_effect=lambda role_name: role_name
    )
    model = sagemaker_session.create_model(
        MODEL_NAME,
    )
    expected_role_arn = SAGEMAKER_CONFIG_MODEL_WITH_PRIMARY_CONTAINER["SageMaker"]["Model"][
        "ExecutionRoleArn"
    ]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_MODEL_WITH_PRIMARY_CONTAINER["SageMaker"][
        "Model"
    ]["EnableNetworkIsolation"]
    expected_vpc_config = SAGEMAKER_CONFIG_MODEL_WITH_PRIMARY_CONTAINER["SageMaker"]["Model"][
        "VpcConfig"
    ]
    expected_tags = SAGEMAKER_CONFIG_MODEL_WITH_PRIMARY_CONTAINER["SageMaker"]["Model"]["Tags"]
    expected_primary_container = SAGEMAKER_CONFIG_MODEL_WITH_PRIMARY_CONTAINER["SageMaker"][
        "Model"
    ]["PrimaryContainer"]
    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(
        ModelName=MODEL_NAME,
        ExecutionRoleArn=expected_role_arn,
        PrimaryContainer=expected_primary_container,
        Tags=expected_tags,
        VpcConfig=expected_vpc_config,
        EnableNetworkIsolation=expected_enable_network_isolation,
    )


def test_create_model_with_sagemaker_config_injection_with_container_defs(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_MODEL

    sagemaker_session.expand_role = Mock(
        name="expand_role", side_effect=lambda role_name: role_name
    )
    model = sagemaker_session.create_model(
        MODEL_NAME,
        container_defs=CONTAINERS,
    )
    expected_role_arn = SAGEMAKER_CONFIG_MODEL["SageMaker"]["Model"]["ExecutionRoleArn"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_MODEL["SageMaker"]["Model"][
        "EnableNetworkIsolation"
    ]
    expected_vpc_config = SAGEMAKER_CONFIG_MODEL["SageMaker"]["Model"]["VpcConfig"]
    expected_tags = SAGEMAKER_CONFIG_MODEL["SageMaker"]["Model"]["Tags"]
    update_list_of_dicts_with_values_from_config(
        CONTAINERS, MODEL_CONTAINERS_PATH, sagemaker_session=sagemaker_session
    )

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(
        ModelName=MODEL_NAME,
        ExecutionRoleArn=expected_role_arn,
        Containers=CONTAINERS,
        Tags=expected_tags,
        VpcConfig=expected_vpc_config,
        EnableNetworkIsolation=expected_enable_network_isolation,
    )


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_model(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE,
        ModelName=MODEL_NAME,
        PrimaryContainer=PRIMARY_CONTAINER,
    )


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_model_with_tags(expand_container_def, sagemaker_session):
    tags = [{"Key": "TagtestKey", "Value": "TagtestValue"}]
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER, tags=tags)

    assert model == MODEL_NAME
    tags = [{"Value": "TagtestValue", "Key": "TagtestKey"}]
    sagemaker_session.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE,
        ModelName=MODEL_NAME,
        PrimaryContainer=PRIMARY_CONTAINER,
        Tags=tags,
    )


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_model_with_primary_container(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, container_defs=PRIMARY_CONTAINER)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE,
        ModelName=MODEL_NAME,
        PrimaryContainer=PRIMARY_CONTAINER,
    )


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_model_with_both(expand_container_def, sagemaker_session):
    with pytest.raises(ValueError):
        sagemaker_session.create_model(
            MODEL_NAME,
            ROLE,
            container_defs=PRIMARY_CONTAINER,
            primary_container=PRIMARY_CONTAINER,
        )


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_pipeline_model(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, container_defs=CONTAINERS)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE, ModelName=MODEL_NAME, Containers=CONTAINERS
    )


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_model_vpc_config(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER, VPC_CONFIG)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE,
        ModelName=MODEL_NAME,
        PrimaryContainer=PRIMARY_CONTAINER,
        VpcConfig=VPC_CONFIG,
    )


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_pipeline_model_vpc_config(expand_container_def, sagemaker_session):
    model = sagemaker_session.create_model(MODEL_NAME, ROLE, CONTAINERS, VPC_CONFIG)

    assert model == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE,
        ModelName=MODEL_NAME,
        Containers=CONTAINERS,
        VpcConfig=VPC_CONFIG,
    )


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_model_already_exists(expand_container_def, sagemaker_session, caplog):
    error_response = {
        "Error": {
            "Code": "ValidationException",
            "Message": "Cannot create already existing model",
        }
    }
    exception = ClientError(error_response, "Operation")
    sagemaker_session.sagemaker_client.create_model.side_effect = exception

    model = sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER)
    assert model == MODEL_NAME

    expected_warning = (
        "sagemaker",
        logging.WARNING,
        "Using already existing model: {}".format(MODEL_NAME),
    )
    assert expected_warning in caplog.record_tuples


@patch("sagemaker.session._expand_container_def", return_value=PRIMARY_CONTAINER)
def test_create_model_failure(expand_container_def, sagemaker_session):
    error_message = "this is expected"
    sagemaker_session.sagemaker_client.create_model.side_effect = RuntimeError(error_message)

    with pytest.raises(RuntimeError) as e:
        sagemaker_session.create_model(MODEL_NAME, ROLE, PRIMARY_CONTAINER)

    assert error_message in str(e)


def test_create_model_from_job(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME)

    assert (
        call(TrainingJobName=JOB_NAME) in ims.sagemaker_client.describe_training_job.call_args_list
    )
    ims.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE,
        ModelName=JOB_NAME,
        PrimaryContainer=PRIMARY_CONTAINER_WITH_COMPRESSED_S3_MODEL,
        VpcConfig=VPC_CONFIG,
    )


def test_create_model_from_job_with_tags(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME, tags=TAGS)

    assert (
        call(TrainingJobName=JOB_NAME) in ims.sagemaker_client.describe_training_job.call_args_list
    )
    ims.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE,
        ModelName=JOB_NAME,
        PrimaryContainer=PRIMARY_CONTAINER_WITH_COMPRESSED_S3_MODEL,
        VpcConfig=VPC_CONFIG,
        Tags=TAGS,
    )


def test_create_model_from_job_uncompressed_s3_model(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = (
        COMPLETED_DESCRIBE_JOB_RESULT_UNCOMPRESSED_S3_MODEL
    )
    ims.create_model_from_job(JOB_NAME)

    assert (
        call(TrainingJobName=JOB_NAME) in ims.sagemaker_client.describe_training_job.call_args_list
    )
    ims.sagemaker_client.create_model.assert_called_with(
        ExecutionRoleArn=EXPANDED_ROLE,
        ModelName=JOB_NAME,
        PrimaryContainer=PRIMARY_CONTAINER_WITH_UNCOMPRESSED_S3_MODEL,
        VpcConfig=VPC_CONFIG,
    )


def test_create_model_from_job_uncompressed_s3_model_unrecognized_compression_type(
    sagemaker_session,
):
    ims = sagemaker_session
    job_desc = copy.deepcopy(COMPLETED_DESCRIBE_JOB_RESULT_UNCOMPRESSED_S3_MODEL)
    job_desc["OutputDataConfig"]["CompressionType"] = "JUNK"
    ims.sagemaker_client.describe_training_job.return_value = job_desc

    with pytest.raises(
        ValueError, match='Unrecognized training job output data compression type "JUNK"'
    ):
        ims.create_model_from_job(JOB_NAME)

    assert (
        call(TrainingJobName=JOB_NAME) in ims.sagemaker_client.describe_training_job.call_args_list
    )

    ims.sagemaker_client.create_model.assert_not_called()


def test_create_edge_packaging_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_EDGE_PACKAGING_JOB

    output_config = {"S3OutputLocation": S3_OUTPUT}

    sagemaker_session.package_model_for_edge(
        output_config,
    )
    expected_role_arn = SAGEMAKER_CONFIG_EDGE_PACKAGING_JOB["SageMaker"]["EdgePackagingJob"][
        "RoleArn"
    ]
    expected_kms_key_id = SAGEMAKER_CONFIG_EDGE_PACKAGING_JOB["SageMaker"]["EdgePackagingJob"][
        "OutputConfig"
    ]["KmsKeyId"]
    expected_tags = SAGEMAKER_CONFIG_EDGE_PACKAGING_JOB["SageMaker"]["EdgePackagingJob"]["Tags"]
    expected_resource_key = SAGEMAKER_CONFIG_EDGE_PACKAGING_JOB["SageMaker"]["EdgePackagingJob"][
        "ResourceKey"
    ]
    sagemaker_session.sagemaker_client.create_edge_packaging_job.assert_called_with(
        RoleArn=expected_role_arn,  # provided from config
        OutputConfig={
            "S3OutputLocation": S3_OUTPUT,  # provided as param
            "KmsKeyId": expected_kms_key_id,  # fetched from config
        },
        ModelName=None,
        ModelVersion=None,
        EdgePackagingJobName=None,
        CompilationJobName=None,
        ResourceKey=expected_resource_key,
        Tags=expected_tags,
    )


def test_create_monitoring_schedule_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_MONITORING_SCHEDULE

    monitoring_output_config = {"MonitoringOutputs": [{"S3Output": {"S3Uri": S3_OUTPUT}}]}

    sagemaker_session.create_monitoring_schedule(
        JOB_NAME,
        schedule_expression=None,
        statistics_s3_uri=None,
        constraints_s3_uri=None,
        monitoring_inputs=[],
        monitoring_output_config=monitoring_output_config,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        volume_size_in_gb=4,
        image_uri="someimageuri",
        network_config={"VpcConfig": {"SecurityGroupIds": ["sg-asparam"]}},
    )
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_MONITORING_SCHEDULE["SageMaker"][
        "MonitoringSchedule"
    ]["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["MonitoringResources"][
        "ClusterConfig"
    ][
        "VolumeKmsKeyId"
    ]
    expected_role_arn = SAGEMAKER_CONFIG_MONITORING_SCHEDULE["SageMaker"]["MonitoringSchedule"][
        "MonitoringScheduleConfig"
    ]["MonitoringJobDefinition"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_MONITORING_SCHEDULE["SageMaker"]["MonitoringSchedule"][
        "MonitoringScheduleConfig"
    ]["MonitoringJobDefinition"]["MonitoringOutputConfig"]["KmsKeyId"]
    expected_subnets = SAGEMAKER_CONFIG_MONITORING_SCHEDULE["SageMaker"]["MonitoringSchedule"][
        "MonitoringScheduleConfig"
    ]["MonitoringJobDefinition"]["NetworkConfig"]["VpcConfig"]["Subnets"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_MONITORING_SCHEDULE["SageMaker"][
        "MonitoringSchedule"
    ]["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["NetworkConfig"][
        "EnableNetworkIsolation"
    ]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_MONITORING_SCHEDULE[
        "SageMaker"
    ]["MonitoringSchedule"]["MonitoringScheduleConfig"]["MonitoringJobDefinition"]["NetworkConfig"][
        "EnableInterContainerTrafficEncryption"
    ]
    expected_environment = SAGEMAKER_CONFIG_MONITORING_SCHEDULE["SageMaker"]["MonitoringSchedule"][
        "MonitoringScheduleConfig"
    ]["MonitoringJobDefinition"]["Environment"]
    sagemaker_session.sagemaker_client.create_monitoring_schedule.assert_called_with(
        MonitoringScheduleName=JOB_NAME,
        MonitoringScheduleConfig={
            "MonitoringJobDefinition": {
                "Environment": expected_environment,
                "MonitoringInputs": [],
                "MonitoringResources": {
                    "ClusterConfig": {
                        "InstanceCount": 1,  # provided as param
                        "InstanceType": "ml.m4.xlarge",  # provided as param
                        "VolumeSizeInGB": 4,  # provided as param
                        "VolumeKmsKeyId": expected_volume_kms_key_id,  # Fetched from config
                    }
                },
                "MonitoringAppSpecification": {"ImageUri": "someimageuri"},  # provided as param
                "RoleArn": expected_role_arn,  # Fetched from config
                "MonitoringOutputConfig": {
                    "MonitoringOutputs": [  # provided as param
                        {"S3Output": {"S3Uri": "s3://sagemaker-123/output/jobname"}}
                    ],
                    "KmsKeyId": expected_kms_key_id,  # fetched from config
                },
                "NetworkConfig": {
                    "VpcConfig": {
                        "Subnets": expected_subnets,  # fetched from config
                        "SecurityGroupIds": ["sg-asparam"],  # provided as param
                    },
                    # The following are fetched from config
                    "EnableNetworkIsolation": expected_enable_network_isolation,
                    "EnableInterContainerTrafficEncryption": expected_enable_inter_container_traffic_encryption,
                },
            },
        },
        Tags=TAGS,  # fetched from config
    )


def test_compile_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_COMPILATION_JOB

    sagemaker_session.compile_model(
        input_model_config={},
        output_model_config={"S3OutputLocation": "s3://test"},
        job_name="TestJob",
    )
    expected_role_arn = SAGEMAKER_CONFIG_COMPILATION_JOB["SageMaker"]["CompilationJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_COMPILATION_JOB["SageMaker"]["CompilationJob"][
        "OutputConfig"
    ]["KmsKeyId"]
    expected_vpc_config = SAGEMAKER_CONFIG_COMPILATION_JOB["SageMaker"]["CompilationJob"][
        "VpcConfig"
    ]
    expected_tags = SAGEMAKER_CONFIG_COMPILATION_JOB["SageMaker"]["CompilationJob"]["Tags"]
    sagemaker_session.sagemaker_client.create_compilation_job.assert_called_with(
        InputConfig={},
        OutputConfig={"S3OutputLocation": "s3://test", "KmsKeyId": expected_kms_key_id},
        RoleArn=expected_role_arn,
        StoppingCondition=None,
        CompilationJobName="TestJob",
        VpcConfig=expected_vpc_config,
        Tags=expected_tags,
    )


def test_create_model_from_job_with_image(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME, image_uri="some-image")
    [create_model_call] = ims.sagemaker_client.create_model.call_args_list
    assert dict(create_model_call[1]["PrimaryContainer"])["Image"] == "some-image"


def test_create_model_from_job_with_container_def(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(
        JOB_NAME,
        image_uri="some-image",
        model_data_url="some-data",
        env={"a": "b"},
    )
    [create_model_call] = ims.sagemaker_client.create_model.call_args_list
    c_def = create_model_call[1]["PrimaryContainer"]
    assert c_def["Image"] == "some-image"
    assert c_def["ModelDataUrl"] == "some-data"
    assert c_def["Environment"] == {"a": "b"}


def test_create_model_from_job_with_vpc_config_override(sagemaker_session):
    vpc_config_override = {"Subnets": ["foo", "bar"], "SecurityGroupIds": ["baz"]}

    ims = sagemaker_session
    ims.sagemaker_client.describe_training_job.return_value = COMPLETED_DESCRIBE_JOB_RESULT
    ims.create_model_from_job(JOB_NAME, vpc_config_override=vpc_config_override)
    assert ims.sagemaker_client.create_model.call_args[1]["VpcConfig"] == vpc_config_override

    ims.create_model_from_job(JOB_NAME, vpc_config_override=None)
    assert "VpcConfig" not in ims.sagemaker_client.create_model.call_args[1]


def test_endpoint_from_production_variants(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "p299.4096xlarge"),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    sagemaker_session.endpoint_from_production_variants("some-endpoint", pvs)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint", EndpointName="some-endpoint", Tags=[]
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint", ProductionVariants=pvs
    )


def test_create_endpoint_config_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_CONFIG

    data_capture_config_dict = {"DestinationS3Uri": "s3://test"}

    # This method does not support ASYNC_INFERENCE_CONFIG or multiple PRODUCTION_VARIANTS
    sagemaker_session.create_endpoint_config(
        name="endpoint-test",
        initial_instance_count=1,
        instance_type="ml.p2.xlarge",
        model_name="simple-model",
        data_capture_config_dict=data_capture_config_dict,
    )
    expected_data_capture_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["DataCaptureConfig"]["KmsKeyId"]
    expected_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "KmsKeyId"
    ]
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"]["Tags"]

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="endpoint-test",
        ProductionVariants=[
            {
                "ModelName": "simple-model",
                "VariantName": "AllTraffic",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.p2.xlarge",
            }
        ],
        DataCaptureConfig={
            "DestinationS3Uri": "s3://test",
            "KmsKeyId": expected_data_capture_kms_key_id,
        },
        KmsKeyId=expected_kms_key_id,
        Tags=expected_tags,
    )


def test_create_endpoint_config_with_sagemaker_config_injection_no_kms_support(
    sagemaker_session,
):
    """Tests that when no production variant instance supports KMS, no KMS key is used."""

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_CONFIG

    data_capture_config_dict = {"DestinationS3Uri": "s3://test"}

    # This method does not support ASYNC_INFERENCE_CONFIG or multiple PRODUCTION_VARIANTS
    sagemaker_session.create_endpoint_config(
        name="endpoint-test",
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        model_name="simple-model",
        data_capture_config_dict=data_capture_config_dict,
    )
    expected_data_capture_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["DataCaptureConfig"]["KmsKeyId"]

    expected_tags = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"]["Tags"]

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="endpoint-test",
        ProductionVariants=[
            {
                "ModelName": "simple-model",
                "VariantName": "AllTraffic",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.g5.xlarge",
            }
        ],
        DataCaptureConfig={
            "DestinationS3Uri": "s3://test",
            "KmsKeyId": expected_data_capture_kms_key_id,
        },
        Tags=expected_tags,
    )


def test_create_endpoint_config_from_existing_with_sagemaker_config_injection(
    sagemaker_session,
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_CONFIG

    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "ml.p2.xlarge"),
        sagemaker.production_variant("C", "ml.p2.xlarge"),
    ]
    # Add DestinationS3Uri to only one production variant
    pvs[0]["CoreDumpConfig"] = {"DestinationS3Uri": "s3://test"}
    existing_endpoint_arn = "arn:aws:sagemaker:us-west-2:123412341234:endpoint-config/foo"
    existing_endpoint_name = "foo"
    new_endpoint_name = "new-foo"
    sagemaker_session.sagemaker_client.describe_endpoint_config.return_value = {
        "ProductionVariants": [sagemaker.production_variant("A", "ml.m4.xlarge")],
        "EndpointConfigArn": existing_endpoint_arn,
        "AsyncInferenceConfig": {},
    }
    sagemaker_session.sagemaker_client.list_tags.return_value = {"Tags": []}

    sagemaker_session.create_endpoint_config_from_existing(
        existing_endpoint_name, new_endpoint_name, new_production_variants=pvs
    )

    expected_production_variant_0_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["ProductionVariants"][0]["CoreDumpConfig"]["KmsKeyId"]
    expected_inference_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "AsyncInferenceConfig"
    ]["OutputConfig"]["KmsKeyId"]
    expected_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "KmsKeyId"
    ]
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"]["Tags"]

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName=new_endpoint_name,
        ProductionVariants=[
            {
                "CoreDumpConfig": {
                    "KmsKeyId": expected_production_variant_0_kms_key_id,
                    "DestinationS3Uri": pvs[0]["CoreDumpConfig"]["DestinationS3Uri"],
                },
                **sagemaker.production_variant("A", "ml.p2.xlarge"),
            },
            {
                # Merge shouldn't happen because input for this index doesn't have DestinationS3Uri
                **sagemaker.production_variant("B", "ml.p2.xlarge"),
            },
            sagemaker.production_variant("C", "ml.p2.xlarge"),
        ],
        KmsKeyId=expected_kms_key_id,  # from config
        Tags=expected_tags,  # from config
        AsyncInferenceConfig={"OutputConfig": {"KmsKeyId": expected_inference_kms_key_id}},
    )


def test_create_endpoint_config_from_existing_with_sagemaker_config_injection_partial_kms_support(
    sagemaker_session,
):
    """Tests that when some production variant instances supports KMS, a KMS key is used.

    Note that this will result in an Exception being thrown at runtime. However, this is a preferable
    experience than not using KMS keys when one could be used, from a security perspective.
    """

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_CONFIG

    pvs = [
        sagemaker.production_variant("A", "ml.g5.2xlarge"),
        sagemaker.production_variant("B", "ml.p2.xlarge"),
        sagemaker.production_variant("C", "ml.p2.xlarge"),
    ]
    # Add DestinationS3Uri to only one production variant
    pvs[0]["CoreDumpConfig"] = {"DestinationS3Uri": "s3://test"}
    existing_endpoint_arn = "arn:aws:sagemaker:us-west-2:123412341234:endpoint-config/foo"
    existing_endpoint_name = "foo"
    new_endpoint_name = "new-foo"
    sagemaker_session.sagemaker_client.describe_endpoint_config.return_value = {
        "ProductionVariants": [sagemaker.production_variant("A", "ml.m4.xlarge")],
        "EndpointConfigArn": existing_endpoint_arn,
        "AsyncInferenceConfig": {},
    }
    sagemaker_session.sagemaker_client.list_tags.return_value = {"Tags": []}

    sagemaker_session.create_endpoint_config_from_existing(
        existing_endpoint_name, new_endpoint_name, new_production_variants=pvs
    )

    expected_production_variant_0_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["ProductionVariants"][0]["CoreDumpConfig"]["KmsKeyId"]
    expected_inference_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "AsyncInferenceConfig"
    ]["OutputConfig"]["KmsKeyId"]
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "KmsKeyId"
    ]
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"]["Tags"]

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName=new_endpoint_name,
        ProductionVariants=[
            {
                "CoreDumpConfig": {
                    "KmsKeyId": expected_production_variant_0_kms_key_id,
                    "DestinationS3Uri": pvs[0]["CoreDumpConfig"]["DestinationS3Uri"],
                },
                **sagemaker.production_variant("A", "ml.g5.2xlarge"),
            },
            {
                # Merge shouldn't happen because input for this index doesn't have DestinationS3Uri
                **sagemaker.production_variant("B", "ml.p2.xlarge"),
            },
            sagemaker.production_variant("C", "ml.p2.xlarge"),
        ],
        KmsKeyId=expected_volume_kms_key_id,  # from config
        Tags=expected_tags,  # from config
        AsyncInferenceConfig={"OutputConfig": {"KmsKeyId": expected_inference_kms_key_id}},
    )


def test_create_endpoint_config_from_existing_with_sagemaker_config_injection_no_kms_support(
    sagemaker_session,
):
    """Tests that when no production variant instance supports KMS, no KMS key is used."""

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_CONFIG

    pvs = [
        sagemaker.production_variant("A", "ml.g5.2xlarge"),
        sagemaker.production_variant("B", "ml.g5.xlarge"),
        sagemaker.production_variant("C", "ml.g5.xlarge"),
    ]
    # Add DestinationS3Uri to only one production variant
    pvs[0]["CoreDumpConfig"] = {"DestinationS3Uri": "s3://test"}
    existing_endpoint_arn = "arn:aws:sagemaker:us-west-2:123412341234:endpoint-config/foo"
    existing_endpoint_name = "foo"
    new_endpoint_name = "new-foo"
    sagemaker_session.sagemaker_client.describe_endpoint_config.return_value = {
        "ProductionVariants": [sagemaker.production_variant("A", "ml.m4.xlarge")],
        "EndpointConfigArn": existing_endpoint_arn,
        "AsyncInferenceConfig": {},
    }
    sagemaker_session.sagemaker_client.list_tags.return_value = {"Tags": []}

    sagemaker_session.create_endpoint_config_from_existing(
        existing_endpoint_name, new_endpoint_name, new_production_variants=pvs
    )

    expected_production_variant_0_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["ProductionVariants"][0]["CoreDumpConfig"]["KmsKeyId"]
    expected_inference_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "AsyncInferenceConfig"
    ]["OutputConfig"]["KmsKeyId"]

    expected_tags = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"]["Tags"]

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName=new_endpoint_name,
        ProductionVariants=[
            {
                "CoreDumpConfig": {
                    "KmsKeyId": expected_production_variant_0_kms_key_id,
                    "DestinationS3Uri": pvs[0]["CoreDumpConfig"]["DestinationS3Uri"],
                },
                **sagemaker.production_variant("A", "ml.g5.2xlarge"),
            },
            {
                # Merge shouldn't happen because input for this index doesn't have DestinationS3Uri
                **sagemaker.production_variant("B", "ml.g5.xlarge"),
            },
            sagemaker.production_variant("C", "ml.g5.xlarge"),
        ],
        Tags=expected_tags,  # from config
        AsyncInferenceConfig={"OutputConfig": {"KmsKeyId": expected_inference_kms_key_id}},
    )


def test_endpoint_from_production_variants_with_sagemaker_config_injection(
    sagemaker_session,
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_CONFIG

    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value={"EndpointStatus": "InService"}
    )
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "ml.p2.xlarge"),
        sagemaker.production_variant("C", "ml.p2.xlarge"),
    ]
    # Add DestinationS3Uri to only one production variant
    pvs[0]["CoreDumpConfig"] = {"DestinationS3Uri": "s3://test"}
    sagemaker_session.endpoint_from_production_variants(
        "some-endpoint",
        pvs,
        data_capture_config_dict={},
        async_inference_config_dict=AsyncInferenceConfig()._to_request_dict(),
    )
    expected_data_capture_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["DataCaptureConfig"]["KmsKeyId"]
    expected_inference_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "AsyncInferenceConfig"
    ]["OutputConfig"]["KmsKeyId"]
    expected_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "KmsKeyId"
    ]
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"]["Tags"]

    expected_async_inference_config_dict = AsyncInferenceConfig()._to_request_dict()
    expected_async_inference_config_dict["OutputConfig"]["KmsKeyId"] = expected_inference_kms_key_id
    expected_pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "ml.p2.xlarge"),
        sagemaker.production_variant("C", "ml.p2.xlarge"),
    ]
    # Add DestinationS3Uri, KmsKeyId to only one production variant
    expected_production_variant_0_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["ProductionVariants"][0]["CoreDumpConfig"]["KmsKeyId"]
    expected_pvs[0]["CoreDumpConfig"] = {
        "DestinationS3Uri": "s3://test",
        "KmsKeyId": expected_production_variant_0_kms_key_id,
    }
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint",
        ProductionVariants=expected_pvs,
        Tags=expected_tags,  # from config
        KmsKeyId=expected_kms_key_id,  # from config
        AsyncInferenceConfig=expected_async_inference_config_dict,
        DataCaptureConfig={"KmsKeyId": expected_data_capture_kms_key_id},
    )
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint",
        EndpointName="some-endpoint",
        Tags=[],
    )


def test_endpoint_from_production_variants_with_sagemaker_config_injection_partial_kms_support(
    sagemaker_session,
):
    """Tests that when some production variants support KMS, a KMS key is applied from config.

    Note that this will result in an Exception being thrown at runtime. However, this is a preferable
    experience than not using KMS keys when one could be used, from a security perspective.
    """

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_CONFIG

    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value={"EndpointStatus": "InService"}
    )
    pvs = [
        sagemaker.production_variant("A", "ml.g5.xlarge"),
        sagemaker.production_variant("B", "ml.p2.xlarge"),
        sagemaker.production_variant("C", "ml.p2.xlarge"),
    ]
    # Add DestinationS3Uri to only one production variant
    pvs[0]["CoreDumpConfig"] = {"DestinationS3Uri": "s3://test"}
    sagemaker_session.endpoint_from_production_variants(
        "some-endpoint",
        pvs,
        data_capture_config_dict={},
        async_inference_config_dict=AsyncInferenceConfig()._to_request_dict(),
    )
    expected_data_capture_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["DataCaptureConfig"]["KmsKeyId"]
    expected_inference_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "AsyncInferenceConfig"
    ]["OutputConfig"]["KmsKeyId"]
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "KmsKeyId"
    ]
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"]["Tags"]

    expected_async_inference_config_dict = AsyncInferenceConfig()._to_request_dict()
    expected_async_inference_config_dict["OutputConfig"]["KmsKeyId"] = expected_inference_kms_key_id
    expected_pvs = [
        sagemaker.production_variant("A", "ml.g5.xlarge"),
        sagemaker.production_variant("B", "ml.p2.xlarge"),
        sagemaker.production_variant("C", "ml.p2.xlarge"),
    ]
    # Add DestinationS3Uri, KmsKeyId to only one production variant
    expected_production_variant_0_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["ProductionVariants"][0]["CoreDumpConfig"]["KmsKeyId"]
    expected_pvs[0]["CoreDumpConfig"] = {
        "DestinationS3Uri": "s3://test",
        "KmsKeyId": expected_production_variant_0_kms_key_id,
    }
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint",
        ProductionVariants=expected_pvs,
        Tags=expected_tags,  # from config
        KmsKeyId=expected_volume_kms_key_id,  # from config
        AsyncInferenceConfig=expected_async_inference_config_dict,
        DataCaptureConfig={"KmsKeyId": expected_data_capture_kms_key_id},
    )
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint",
        EndpointName="some-endpoint",
        Tags=[],
    )


def test_endpoint_from_production_variants_with_sagemaker_config_injection_no_kms_support(
    sagemaker_session,
):
    """Tests that when no production variants support KMS, no KMS key is applied from config."""

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_CONFIG

    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value={"EndpointStatus": "InService"}
    )
    pvs = [
        sagemaker.production_variant("A", "ml.g5.xlarge"),
        sagemaker.production_variant("B", "ml.g5.xlarge"),
        sagemaker.production_variant("C", "ml.g5.xlarge"),
    ]
    # Add DestinationS3Uri to only one production variant
    pvs[0]["CoreDumpConfig"] = {"DestinationS3Uri": "s3://test"}
    sagemaker_session.endpoint_from_production_variants(
        "some-endpoint",
        pvs,
        data_capture_config_dict={},
        async_inference_config_dict=AsyncInferenceConfig()._to_request_dict(),
    )
    expected_data_capture_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["DataCaptureConfig"]["KmsKeyId"]
    expected_inference_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"][
        "AsyncInferenceConfig"
    ]["OutputConfig"]["KmsKeyId"]

    expected_tags = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"]["EndpointConfig"]["Tags"]

    expected_async_inference_config_dict = AsyncInferenceConfig()._to_request_dict()
    expected_async_inference_config_dict["OutputConfig"]["KmsKeyId"] = expected_inference_kms_key_id
    expected_pvs = [
        sagemaker.production_variant("A", "ml.g5.xlarge"),
        sagemaker.production_variant("B", "ml.g5.xlarge"),
        sagemaker.production_variant("C", "ml.g5.xlarge"),
    ]
    # Add DestinationS3Uri, KmsKeyId to only one production variant
    expected_production_variant_0_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_CONFIG["SageMaker"][
        "EndpointConfig"
    ]["ProductionVariants"][0]["CoreDumpConfig"]["KmsKeyId"]
    expected_pvs[0]["CoreDumpConfig"] = {
        "DestinationS3Uri": "s3://test",
        "KmsKeyId": expected_production_variant_0_kms_key_id,
    }
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint",
        ProductionVariants=expected_pvs,
        Tags=expected_tags,  # from config
        AsyncInferenceConfig=expected_async_inference_config_dict,
        DataCaptureConfig={"KmsKeyId": expected_data_capture_kms_key_id},
    )
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint",
        EndpointName="some-endpoint",
        Tags=[],
    )


def test_create_endpoint_config_with_tags_list(sagemaker_session):
    tags = [{"Key": "TagtestKey", "Value": "TagtestValue"}]

    sagemaker_session.create_endpoint_config(
        name="endpoint-test",
        initial_instance_count=1,
        instance_type="local",
        model_name="simple-model",
        tags=tags,
    )

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="endpoint-test", ProductionVariants=ANY, Tags=tags
    )


def test_create_endpoint_config_with_tags_dict(sagemaker_session):
    tags = {"TagtestKey": "TagtestValue"}
    call_tags = [{"Key": "TagtestKey", "Value": "TagtestValue"}]

    sagemaker_session.create_endpoint_config(
        name="endpoint-test",
        initial_instance_count=1,
        instance_type="local",
        model_name="simple-model",
        tags=tags,
    )

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="endpoint-test", ProductionVariants=ANY, Tags=call_tags
    )


def test_create_endpoint_config_with_explainer_config(sagemaker_session):
    explainer_config = ExplainerConfig

    sagemaker_session.create_endpoint_config(
        name="endpoint-test",
        model_name="simple-model",
        initial_instance_count=1,
        instance_type="local",
        explainer_config_dict=explainer_config,
    )

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="endpoint-test",
        ProductionVariants=ANY,
        Tags=ANY,
        ExplainerConfig=explainer_config,
    )


def test_endpoint_from_production_variants_with_tags(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "p299.4096xlarge"),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    tags = [{"ModelName": "TestModel"}]
    sagemaker_session.endpoint_from_production_variants("some-endpoint", pvs, tags)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint", EndpointName="some-endpoint", Tags=tags
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint", ProductionVariants=pvs, Tags=tags
    )


def test_endpoint_from_production_variants_with_combined_sagemaker_config_injection_tags(
    sagemaker_session,
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT_ENDPOINT_CONFIG_COMBINED

    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "p299.4096xlarge"),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    expected_endpoint_tags = SAGEMAKER_CONFIG_ENDPOINT_ENDPOINT_CONFIG_COMBINED["SageMaker"][
        "Endpoint"
    ]["Tags"]
    expected_endpoint_config_tags = SAGEMAKER_CONFIG_ENDPOINT_ENDPOINT_CONFIG_COMBINED["SageMaker"][
        "EndpointConfig"
    ]["Tags"]
    expected_endpoint_config_kms_key_id = SAGEMAKER_CONFIG_ENDPOINT_ENDPOINT_CONFIG_COMBINED[
        "SageMaker"
    ]["EndpointConfig"]["KmsKeyId"]
    sagemaker_session.endpoint_from_production_variants("some-endpoint", pvs)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint",
        EndpointName="some-endpoint",
        Tags=expected_endpoint_tags,
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint",
        ProductionVariants=pvs,
        Tags=expected_endpoint_config_tags,
        KmsKeyId=expected_endpoint_config_kms_key_id,
    )


def test_endpoint_from_production_variants_with_sagemaker_config_injection_tags(
    sagemaker_session,
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT

    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "p299.4096xlarge"),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT["SageMaker"]["Endpoint"]["Tags"]
    sagemaker_session.endpoint_from_production_variants("some-endpoint", pvs)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint",
        EndpointName="some-endpoint",
        Tags=expected_tags,
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint", ProductionVariants=pvs
    )


def test_endpoint_from_production_variants_with_accelerator_type(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge", accelerator_type=ACCELERATOR_TYPE),
        sagemaker.production_variant("B", "p299.4096xlarge", accelerator_type=ACCELERATOR_TYPE),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    tags = [{"ModelName": "TestModel"}]
    sagemaker_session.endpoint_from_production_variants("some-endpoint", pvs, tags)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint", EndpointName="some-endpoint", Tags=tags
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint", ProductionVariants=pvs, Tags=tags
    )


def test_endpoint_from_production_variants_with_accelerator_type_sagemaker_config_injection_tags(
    sagemaker_session,
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT

    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge", accelerator_type=ACCELERATOR_TYPE),
        sagemaker.production_variant("B", "p299.4096xlarge", accelerator_type=ACCELERATOR_TYPE),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT["SageMaker"]["Endpoint"]["Tags"]
    sagemaker_session.endpoint_from_production_variants("some-endpoint", pvs)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint",
        EndpointName="some-endpoint",
        Tags=expected_tags,
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint", ProductionVariants=pvs
    )


def test_endpoint_from_production_variants_with_serverless_inference_config(
    sagemaker_session,
):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant(
            "A", "ml.p2.xlarge", serverless_inference_config=SERVERLESS_INFERENCE_CONFIG
        ),
        sagemaker.production_variant(
            "B",
            "p299.4096xlarge",
            serverless_inference_config=SERVERLESS_INFERENCE_CONFIG,
        ),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    tags = [{"ModelName": "TestModel"}]
    sagemaker_session.endpoint_from_production_variants("some-endpoint", pvs, tags)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint", EndpointName="some-endpoint", Tags=tags
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint", ProductionVariants=pvs, Tags=tags
    )


def test_endpoint_from_production_variants_with_serverless_inference_config_sagemaker_config_injection_tags(
    sagemaker_session,
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT

    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant(
            "A", "ml.p2.xlarge", serverless_inference_config=SERVERLESS_INFERENCE_CONFIG
        ),
        sagemaker.production_variant(
            "B",
            "p299.4096xlarge",
            serverless_inference_config=SERVERLESS_INFERENCE_CONFIG,
        ),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT["SageMaker"]["Endpoint"]["Tags"]
    sagemaker_session.endpoint_from_production_variants("some-endpoint", pvs)
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint",
        EndpointName="some-endpoint",
        Tags=expected_tags,
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint", ProductionVariants=pvs
    )


def test_endpoint_from_production_variants_with_async_config(sagemaker_session):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "p299.4096xlarge"),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    sagemaker_session.endpoint_from_production_variants(
        "some-endpoint",
        pvs,
        async_inference_config_dict=AsyncInferenceConfig()._to_request_dict(),
    )
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint", EndpointName="some-endpoint", Tags=[]
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint",
        ProductionVariants=pvs,
        AsyncInferenceConfig=AsyncInferenceConfig()._to_request_dict(),
    )


def test_endpoint_from_production_variants_with_async_config_sagemaker_config_injection_tags(
    sagemaker_session,
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_ENDPOINT

    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "p299.4096xlarge"),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    expected_tags = SAGEMAKER_CONFIG_ENDPOINT["SageMaker"]["Endpoint"]["Tags"]
    sagemaker_session.endpoint_from_production_variants(
        "some-endpoint",
        pvs,
        async_inference_config_dict=AsyncInferenceConfig()._to_request_dict(),
    )
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint",
        EndpointName="some-endpoint",
        Tags=expected_tags,
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint",
        ProductionVariants=pvs,
        AsyncInferenceConfig=AsyncInferenceConfig()._to_request_dict(),
    )


def test_endpoint_from_production_variants_with_clarify_explainer_config(
    sagemaker_session,
):
    ims = sagemaker_session
    ims.sagemaker_client.describe_endpoint = Mock(return_value={"EndpointStatus": "InService"})
    pvs = [
        sagemaker.production_variant("A", "ml.p2.xlarge"),
        sagemaker.production_variant("B", "p299.4096xlarge"),
    ]
    ex = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Could not find your thing",
            }
        },
        "b",
    )
    ims.sagemaker_client.describe_endpoint_config = Mock(side_effect=ex)
    sagemaker_session.endpoint_from_production_variants(
        "some-endpoint",
        pvs,
        explainer_config_dict=ExplainerConfig,
    )
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_with(
        EndpointConfigName="some-endpoint", EndpointName="some-endpoint", Tags=[]
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName="some-endpoint",
        ProductionVariants=pvs,
        ExplainerConfig=ExplainerConfig,
    )


def test_update_endpoint_succeed(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value={"EndpointStatus": "InService"}
    )
    endpoint_name = "some-endpoint"
    endpoint_config = "some-endpoint-config"
    returned_endpoint_name = sagemaker_session.update_endpoint(endpoint_name, endpoint_config)
    assert returned_endpoint_name == endpoint_name


def test_update_endpoint_no_wait(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value={"EndpointStatus": "Updating"}
    )
    endpoint_name = "some-endpoint"
    endpoint_config = "some-endpoint-config"
    returned_endpoint_name = sagemaker_session.update_endpoint(
        endpoint_name, endpoint_config, wait=False
    )
    assert returned_endpoint_name == endpoint_name


def test_update_endpoint_non_existing_endpoint(sagemaker_session):
    error = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Could not find entity"}},
        "foo",
    )
    expected_error_message = (
        "Endpoint with name 'non-existing-endpoint' does not exist; "
        "please use an existing endpoint name"
    )
    sagemaker_session.sagemaker_client.describe_endpoint = Mock(side_effect=error)
    with pytest.raises(ValueError, match=expected_error_message):
        sagemaker_session.update_endpoint("non-existing-endpoint", "non-existing-config")


def test_create_endpoint_config_from_existing(sagemaker_session):
    pvs = [sagemaker.production_variant("A", "ml.m4.xlarge")]
    tags = [{"Key": "aws:cloudformation:stackname", "Value": "this-tag-should-be-ignored"}]
    existing_endpoint_arn = "arn:aws:sagemaker:us-west-2:123412341234:endpoint-config/foo"
    kms_key = "kms"
    sagemaker_session.sagemaker_client.describe_endpoint_config.return_value = {
        "Tags": tags,
        "ProductionVariants": pvs,
        "EndpointConfigArn": existing_endpoint_arn,
        "KmsKeyId": kms_key,
    }
    sagemaker_session.sagemaker_client.list_tags.return_value = {"Tags": tags}

    existing_endpoint_name = "foo"
    new_endpoint_name = "new-foo"
    sagemaker_session.create_endpoint_config_from_existing(
        existing_endpoint_name, new_endpoint_name
    )

    sagemaker_session.sagemaker_client.describe_endpoint_config.assert_called_with(
        EndpointConfigName=existing_endpoint_name
    )
    sagemaker_session.sagemaker_client.list_tags.assert_called_with(
        ResourceArn=existing_endpoint_arn, MaxResults=50
    )
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName=new_endpoint_name, ProductionVariants=pvs, KmsKeyId=kms_key
    )


def test_create_endpoint_config_from_existing_with_explainer_config(sagemaker_session):
    pvs = [sagemaker.production_variant("A", "ml.m4.xlarge")]
    tags = [{"Key": "aws:cloudformation:stackname", "Value": "this-tag-should-be-ignored"}]
    existing_endpoint_arn = "arn:aws:sagemaker:us-west-2:123412341234:endpoint-config/foo"
    kms_key = "kms"
    existing_explainer_config = ExplainerConfig
    sagemaker_session.sagemaker_client.describe_endpoint_config.return_value = {
        "Tags": tags,
        "ProductionVariants": pvs,
        "EndpointConfigArn": existing_endpoint_arn,
        "KmsKeyId": kms_key,
        "ExplainerConfig": existing_explainer_config,
    }
    sagemaker_session.sagemaker_client.list_tags.return_value = {"Tags": tags}

    existing_endpoint_name = "foo"
    new_endpoint_name = "new-foo"
    sagemaker_session.create_endpoint_config_from_existing(
        existing_endpoint_name, new_endpoint_name
    )

    sagemaker_session.sagemaker_client.describe_endpoint_config.assert_called_with(
        EndpointConfigName=existing_endpoint_name
    )

    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_with(
        EndpointConfigName=new_endpoint_name,
        ProductionVariants=pvs,
        KmsKeyId=kms_key,
        ExplainerConfig=existing_explainer_config,
    )


@patch("time.sleep")
def test_wait_for_tuning_job(sleep, sagemaker_session):
    hyperparameter_tuning_job_desc = {"HyperParameterTuningJobStatus": "Completed"}
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job",
        return_value=hyperparameter_tuning_job_desc,
    )

    result = sagemaker_session.wait_for_tuning_job(JOB_NAME)
    assert result["HyperParameterTuningJobStatus"] == "Completed"


def test_tune_job_status(sagemaker_session):
    hyperparameter_tuning_job_desc = {"HyperParameterTuningJobStatus": "Completed"}
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job",
        return_value=hyperparameter_tuning_job_desc,
    )

    result = _tuning_job_status(sagemaker_session.sagemaker_client, JOB_NAME)

    assert result["HyperParameterTuningJobStatus"] == "Completed"


def test_tune_job_status_none(sagemaker_session):
    hyperparameter_tuning_job_desc = {"HyperParameterTuningJobStatus": "InProgress"}
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job = Mock(
        name="describe_hyper_parameter_tuning_job",
        return_value=hyperparameter_tuning_job_desc,
    )

    result = _tuning_job_status(sagemaker_session.sagemaker_client, JOB_NAME)

    assert result is None


@patch("time.sleep")
def test_wait_for_transform_job_completed(sleep, sagemaker_session):
    transform_job_desc = {"TransformJobStatus": "Completed"}
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(
        name="describe_transform_job", return_value=transform_job_desc
    )

    assert sagemaker_session.wait_for_transform_job(JOB_NAME)["TransformJobStatus"] == "Completed"


@patch("time.sleep")
def test_wait_for_transform_job_in_progress(sleep, sagemaker_session):
    transform_job_desc_in_progress = {"TransformJobStatus": "InProgress"}
    transform_job_desc_in_completed = {"TransformJobStatus": "Completed"}
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(
        name="describe_transform_job",
        side_effect=[transform_job_desc_in_progress, transform_job_desc_in_completed],
    )

    assert (
        sagemaker_session.wait_for_transform_job(JOB_NAME, 1)["TransformJobStatus"] == "Completed"
    )
    assert 2 == sagemaker_session.sagemaker_client.describe_transform_job.call_count


def test_transform_job_status(sagemaker_session):
    transform_job_desc = {"TransformJobStatus": "Completed"}
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(
        name="describe_transform_job", return_value=transform_job_desc
    )

    result = _transform_job_status(sagemaker_session.sagemaker_client, JOB_NAME)
    assert result["TransformJobStatus"] == "Completed"


def test_transform_job_status_none(sagemaker_session):
    transform_job_desc = {"TransformJobStatus": "InProgress"}
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(
        name="describe_transform_job", return_value=transform_job_desc
    )

    result = _transform_job_status(sagemaker_session.sagemaker_client, JOB_NAME)
    assert result is None


def test_train_done_completed(sagemaker_session):
    training_job_desc = {"TrainingJobStatus": "Completed"}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=training_job_desc
    )

    actual_job_desc, training_finished = _train_done(
        sagemaker_session.sagemaker_client, JOB_NAME, None
    )

    assert actual_job_desc["TrainingJobStatus"] == "Completed"
    assert training_finished is True


def test_train_done_in_progress(sagemaker_session):
    training_job_desc = {"TrainingJobStatus": "InProgress"}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=training_job_desc
    )

    actual_job_desc, training_finished = _train_done(
        sagemaker_session.sagemaker_client, JOB_NAME, None
    )

    assert actual_job_desc["TrainingJobStatus"] == "InProgress"
    assert training_finished is False


@patch("time.sleep", return_value=None)
def test_wait_until_training_done_raises_other_exception(patched_sleep):
    response = {"Error": {"Code": "ValidationException", "Message": "Could not access entity."}}
    mock_func = Mock(
        name="describe_training_job",
        side_effect=ClientError(error_response=response, operation_name="foo"),
    )
    desc = "dummy"
    with pytest.raises(ClientError) as error:
        _wait_until_training_done(mock_func, desc)

    mock_func.assert_called_once()
    assert "ValidationException" in str(error)


@patch("time.sleep", return_value=None)
def test_wait_until_training_done_tag_propagation(patched_sleep):
    response = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "Could not access entity.",
        }
    }
    side_effect_iter = [ClientError(error_response=response, operation_name="foo")] * 3
    side_effect_iter.append(("result", "result"))
    mock_func = Mock(name="describe_training_job", side_effect=side_effect_iter)
    desc = "dummy"
    result = _wait_until_training_done(mock_func, desc)
    assert result == "result"
    assert mock_func.call_count == 4


@patch("time.sleep", return_value=None)
def test_wait_until_training_done_fail_access_denied_after_5_mins(patched_sleep):
    response = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "Could not access entity.",
        }
    }
    side_effect_iter = [ClientError(error_response=response, operation_name="foo")] * 70
    mock_func = Mock(name="describe_training_job", side_effect=side_effect_iter)
    desc = "dummy"
    with pytest.raises(ClientError) as error:
        _wait_until_training_done(mock_func, desc)

    # mock_func should be retried 300(elapsed time)/5(default poll delay) = 60 times
    assert mock_func.call_count == 61
    assert "AccessDeniedException" in str(error)


@patch("time.sleep", return_value=None)
def test_wait_until_raises_other_exception(patched_sleep):
    mock_func = Mock(name="describe_training_job", side_effect=_raise_unexpected_client_error)
    with pytest.raises(ClientError) as error:
        _wait_until(mock_func)

    mock_func.assert_called_once()
    assert "ValidationException" in str(error)


@patch("time.sleep", return_value=None)
def test_wait_until_tag_propagation(patched_sleep):
    response = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "Could not access entity.",
        }
    }
    side_effect_iter = [ClientError(error_response=response, operation_name="foo")] * 3
    side_effect_iter.append("result")
    mock_func = Mock(name="describe_training_job", side_effect=side_effect_iter)
    result = _wait_until(mock_func)
    assert result == "result"
    assert mock_func.call_count == 4


@patch("time.sleep", return_value=None)
def test_wait_until_fail_access_denied_after_5_mins(patched_sleep):
    response = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "Could not access entity.",
        }
    }
    side_effect_iter = [ClientError(error_response=response, operation_name="foo")] * 70
    mock_func = Mock(name="describe_training_job", side_effect=side_effect_iter)
    with pytest.raises(ClientError) as error:
        _wait_until(mock_func)

    # mock_func should be retried 300(elapsed time)/5(default poll delay) = 60 times
    assert mock_func.call_count == 61
    assert "AccessDeniedException" in str(error)


DEFAULT_EXPECTED_AUTO_ML_JOB_ARGS = {
    "AutoMLJobName": JOB_NAME,
    "InputDataConfig": [
        {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI}},
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
        }
    ],
    "OutputDataConfig": {"S3OutputPath": S3_OUTPUT},
    "AutoMLJobConfig": {
        "CompletionCriteria": {
            "MaxCandidates": 10,
            "MaxAutoMLJobRuntimeInSeconds": 36000,
            "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
        }
    },
    "RoleArn": EXPANDED_ROLE,
    "GenerateCandidateDefinitionsOnly": False,
}

COMPLETE_EXPECTED_AUTO_ML_JOB_ARGS = {
    "AutoMLJobName": JOB_NAME,
    "InputDataConfig": [
        {
            "ChannelType": "training",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
        },
        {
            "ChannelType": "validation",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": DEFAULT_S3_VALIDATION_DATA,
                }
            },
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
        },
    ],
    "OutputDataConfig": {"S3OutputPath": S3_OUTPUT},
    "ProblemType": "Regression",
    "AutoMLJobObjective": {"Type": "type", "MetricName": "metric-name"},
    "AutoMLJobConfig": {
        "CandidateGenerationConfig": {"FeatureSpecificationS3Uri": "s3://mybucket/features.json"},
        "Mode": "ENSEMBLING",
        "CompletionCriteria": {
            "MaxCandidates": 10,
            "MaxAutoMLJobRuntimeInSeconds": 36000,
            "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
        },
        "SecurityConfig": {
            "VolumeKmsKeyId": "volume-kms-key-id-string",
            "EnableInterContainerTrafficEncryption": False,
            "VpcConfig": {
                "SecurityGroupIds": ["security-group-id"],
                "Subnets": ["subnet"],
            },
        },
    },
    "RoleArn": EXPANDED_ROLE,
    "GenerateCandidateDefinitionsOnly": True,
    "Tags": ["tag"],
}

DEFAULT_EXPECTED_AUTO_ML_V2_JOB_ARGS = {
    "AutoMLJobName": JOB_NAME,
    "AutoMLJobInputDataConfig": [
        {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI}},
        }
    ],
    "AutoMLProblemTypeConfig": {
        "TabularJobConfig": {
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
            "CompletionCriteria": {
                "MaxCandidates": 10,
                "MaxAutoMLJobRuntimeInSeconds": 36000,
                "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
            },
            "GenerateCandidateDefinitionsOnly": False,
        }
    },
    "OutputDataConfig": {"S3OutputPath": S3_OUTPUT},
    "RoleArn": EXPANDED_ROLE,
}

COMPLETE_EXPECTED_AUTO_ML_V2_JOB_ARGS = {
    "AutoMLJobName": JOB_NAME,
    "AutoMLJobInputDataConfig": [
        {
            "ChannelType": "training",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        },
        {
            "ChannelType": "validation",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": DEFAULT_S3_VALIDATION_DATA,
                }
            },
        },
    ],
    "OutputDataConfig": {"S3OutputPath": S3_OUTPUT},
    "AutoMLProblemTypeConfig": {
        "TabularJobConfig": {
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
            "CompletionCriteria": {
                "MaxCandidates": 10,
                "MaxAutoMLJobRuntimeInSeconds": 36000,
                "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
            },
            "GenerateCandidateDefinitionsOnly": False,
            "ProblemType": "Regression",
            "FeatureSpecificationS3Uri": "s3://mybucket/features.json",
            "Mode": "ENSEMBLING",
        },
    },
    "AutoMLJobObjective": {"Type": "type", "MetricName": "metric-name"},
    "SecurityConfig": {
        "VolumeKmsKeyId": "volume-kms-key-id-string",
        "EnableInterContainerTrafficEncryption": False,
        "VpcConfig": {
            "SecurityGroupIds": ["security-group-id"],
            "Subnets": ["subnet"],
        },
    },
    "RoleArn": EXPANDED_ROLE,
    "Tags": ["tag"],
}


COMPLETE_EXPECTED_LIST_CANDIDATES_ARGS = {
    "AutoMLJobName": JOB_NAME,
    "StatusEquals": "Completed",
    "SortOrder": "Descending",
    "SortBy": "Status",
    "MaxResults": 10,
}


def test_auto_ml_pack_to_request(sagemaker_session):
    input_config = [
        {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI}},
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
        }
    ]

    output_config = {"S3OutputPath": S3_OUTPUT}

    auto_ml_job_config = {
        "CompletionCriteria": {
            "MaxCandidates": 10,
            "MaxAutoMLJobRuntimeInSeconds": 36000,
            "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
        }
    }

    job_name = JOB_NAME
    role = EXPANDED_ROLE

    sagemaker_session.auto_ml(input_config, output_config, auto_ml_job_config, role, job_name)
    sagemaker_session.sagemaker_client.create_auto_ml_job.assert_called_with(
        AutoMLJobName=DEFAULT_EXPECTED_AUTO_ML_JOB_ARGS["AutoMLJobName"],
        InputDataConfig=DEFAULT_EXPECTED_AUTO_ML_JOB_ARGS["InputDataConfig"],
        OutputDataConfig=DEFAULT_EXPECTED_AUTO_ML_JOB_ARGS["OutputDataConfig"],
        AutoMLJobConfig=DEFAULT_EXPECTED_AUTO_ML_JOB_ARGS["AutoMLJobConfig"],
        RoleArn=DEFAULT_EXPECTED_AUTO_ML_JOB_ARGS["RoleArn"],
        GenerateCandidateDefinitionsOnly=False,
    )


def test_auto_ml_v2_pack_to_request(sagemaker_session):
    input_config = [
        {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI}},
        }
    ]

    output_config = {"S3OutputPath": S3_OUTPUT}

    problem_config = {
        "TabularJobConfig": {
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
            "CompletionCriteria": {
                "MaxCandidates": 10,
                "MaxAutoMLJobRuntimeInSeconds": 36000,
                "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
            },
            "GenerateCandidateDefinitionsOnly": False,
        }
    }

    job_name = JOB_NAME
    role = EXPANDED_ROLE

    sagemaker_session.create_auto_ml_v2(
        input_config=input_config,
        job_name=job_name,
        problem_config=problem_config,
        output_config=output_config,
        role=role,
    )

    sagemaker_session.sagemaker_client.create_auto_ml_job_v2.assert_called_with(
        AutoMLJobName=DEFAULT_EXPECTED_AUTO_ML_V2_JOB_ARGS["AutoMLJobName"],
        AutoMLJobInputDataConfig=DEFAULT_EXPECTED_AUTO_ML_V2_JOB_ARGS["AutoMLJobInputDataConfig"],
        OutputDataConfig=DEFAULT_EXPECTED_AUTO_ML_V2_JOB_ARGS["OutputDataConfig"],
        AutoMLProblemTypeConfig=DEFAULT_EXPECTED_AUTO_ML_V2_JOB_ARGS["AutoMLProblemTypeConfig"],
        RoleArn=DEFAULT_EXPECTED_AUTO_ML_V2_JOB_ARGS["RoleArn"],
    )


def test_create_auto_ml_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_AUTO_ML

    input_config = [
        {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI}},
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
        }
    ]

    output_config = {"S3OutputPath": S3_OUTPUT}

    auto_ml_job_config = {
        "CompletionCriteria": {
            "MaxCandidates": 10,
            "MaxAutoMLJobRuntimeInSeconds": 36000,
            "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
        }
    }

    job_name = JOB_NAME
    sagemaker_session.auto_ml(input_config, output_config, auto_ml_job_config, job_name=job_name)
    expected_call_args = copy.deepcopy(DEFAULT_EXPECTED_AUTO_ML_JOB_ARGS)
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"][
        "AutoMLJobConfig"
    ]["SecurityConfig"]["VolumeKmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"]["OutputDataConfig"][
        "KmsKeyId"
    ]
    expected_vpc_config = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"]["AutoMLJobConfig"][
        "SecurityConfig"
    ]["VpcConfig"]
    expected_tags = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"]["AutoMLJob"]["Tags"]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_AUTO_ML["SageMaker"][
        "AutoMLJob"
    ]["AutoMLJobConfig"]["SecurityConfig"]["EnableInterContainerTrafficEncryption"]
    expected_call_args["OutputDataConfig"]["KmsKeyId"] = expected_kms_key_id
    expected_call_args["RoleArn"] = expected_role_arn
    expected_call_args["AutoMLJobConfig"]["SecurityConfig"] = {
        "EnableInterContainerTrafficEncryption": expected_enable_inter_container_traffic_encryption
    }
    expected_call_args["AutoMLJobConfig"]["SecurityConfig"]["VpcConfig"] = expected_vpc_config
    expected_call_args["AutoMLJobConfig"]["SecurityConfig"][
        "VolumeKmsKeyId"
    ] = expected_volume_kms_key_id
    sagemaker_session.sagemaker_client.create_auto_ml_job.assert_called_with(
        AutoMLJobName=expected_call_args["AutoMLJobName"],
        InputDataConfig=expected_call_args["InputDataConfig"],
        OutputDataConfig=expected_call_args["OutputDataConfig"],
        AutoMLJobConfig=expected_call_args["AutoMLJobConfig"],
        RoleArn=expected_call_args["RoleArn"],
        GenerateCandidateDefinitionsOnly=False,
        Tags=expected_tags,
    )


def test_create_auto_ml_v2_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_AUTO_ML_V2

    input_config = [
        {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": S3_INPUT_URI}},
        }
    ]

    output_config = {"S3OutputPath": S3_OUTPUT}

    problem_config = {
        "TabularJobConfig": {
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
            "CompletionCriteria": {
                "MaxCandidates": 10,
                "MaxAutoMLJobRuntimeInSeconds": 36000,
                "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
            },
            "GenerateCandidateDefinitionsOnly": False,
        }
    }

    job_name = JOB_NAME
    sagemaker_session.create_auto_ml_v2(
        input_config=input_config,
        job_name=job_name,
        problem_config=problem_config,
        output_config=output_config,
    )

    expected_call_args = copy.deepcopy(DEFAULT_EXPECTED_AUTO_ML_V2_JOB_ARGS)
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_AUTO_ML_V2["SageMaker"]["AutoMLJobV2"][
        "SecurityConfig"
    ]["VolumeKmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_AUTO_ML_V2["SageMaker"]["AutoMLJobV2"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_AUTO_ML_V2["SageMaker"]["AutoMLJobV2"][
        "OutputDataConfig"
    ]["KmsKeyId"]
    expected_vpc_config = SAGEMAKER_CONFIG_AUTO_ML_V2["SageMaker"]["AutoMLJobV2"]["SecurityConfig"][
        "VpcConfig"
    ]
    expected_tags = SAGEMAKER_CONFIG_AUTO_ML_V2["SageMaker"]["AutoMLJobV2"]["Tags"]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_AUTO_ML_V2["SageMaker"][
        "AutoMLJobV2"
    ]["SecurityConfig"]["EnableInterContainerTrafficEncryption"]
    expected_call_args["OutputDataConfig"]["KmsKeyId"] = expected_kms_key_id
    expected_call_args["RoleArn"] = expected_role_arn
    expected_call_args["SecurityConfig"] = {
        "EnableInterContainerTrafficEncryption": expected_enable_inter_container_traffic_encryption
    }
    expected_call_args["SecurityConfig"]["VpcConfig"] = expected_vpc_config
    expected_call_args["SecurityConfig"]["VolumeKmsKeyId"] = expected_volume_kms_key_id
    sagemaker_session.sagemaker_client.create_auto_ml_job_v2.assert_called_with(
        AutoMLJobName=expected_call_args["AutoMLJobName"],
        AutoMLJobInputDataConfig=expected_call_args["AutoMLJobInputDataConfig"],
        OutputDataConfig=expected_call_args["OutputDataConfig"],
        AutoMLProblemTypeConfig=expected_call_args["AutoMLProblemTypeConfig"],
        RoleArn=expected_call_args["RoleArn"],
        Tags=expected_tags,
    )


def test_auto_ml_pack_to_request_with_optional_args(sagemaker_session):
    input_config = [
        {
            "ChannelType": "training",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
        },
        {
            "ChannelType": "validation",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": DEFAULT_S3_VALIDATION_DATA,
                }
            },
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
        },
    ]

    output_config = {"S3OutputPath": S3_OUTPUT}

    auto_ml_job_config = {
        "CandidateGenerationConfig": {"FeatureSpecificationS3Uri": "s3://mybucket/features.json"},
        "Mode": "ENSEMBLING",
        "CompletionCriteria": {
            "MaxCandidates": 10,
            "MaxAutoMLJobRuntimeInSeconds": 36000,
            "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
        },
        "SecurityConfig": {
            "VolumeKmsKeyId": "volume-kms-key-id-string",
            "EnableInterContainerTrafficEncryption": False,
            "VpcConfig": {
                "SecurityGroupIds": ["security-group-id"],
                "Subnets": ["subnet"],
            },
        },
    }

    job_name = JOB_NAME
    role = EXPANDED_ROLE

    sagemaker_session.auto_ml(
        input_config,
        output_config,
        auto_ml_job_config,
        role,
        job_name,
        problem_type="Regression",
        job_objective={"Type": "type", "MetricName": "metric-name"},
        generate_candidate_definitions_only=True,
        tags=["tag"],
    )

    assert sagemaker_session.sagemaker_client.method_calls[0] == (
        "create_auto_ml_job",
        (),
        COMPLETE_EXPECTED_AUTO_ML_JOB_ARGS,
    )


def test_auto_ml_v2_pack_to_request_with_optional_args(sagemaker_session):
    input_config = [
        {
            "ChannelType": "training",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": S3_INPUT_URI,
                }
            },
        },
        {
            "ChannelType": "validation",
            "CompressionType": "Gzip",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": DEFAULT_S3_VALIDATION_DATA,
                }
            },
        },
    ]

    output_config = {"S3OutputPath": S3_OUTPUT}

    problem_config = {
        "TabularJobConfig": {
            "TargetAttributeName": "y",
            "SampleWeightAttributeName": "sampleWeight",
            "CompletionCriteria": {
                "MaxCandidates": 10,
                "MaxAutoMLJobRuntimeInSeconds": 36000,
                "MaxRuntimePerTrainingJobInSeconds": 3600 * 2,
            },
            "GenerateCandidateDefinitionsOnly": False,
            "FeatureSpecificationS3Uri": "s3://mybucket/features.json",
            "Mode": "ENSEMBLING",
            "ProblemType": "Regression",
        }
    }

    security_config = {
        "VolumeKmsKeyId": "volume-kms-key-id-string",
        "EnableInterContainerTrafficEncryption": False,
        "VpcConfig": {
            "SecurityGroupIds": ["security-group-id"],
            "Subnets": ["subnet"],
        },
    }

    job_name = JOB_NAME
    role = EXPANDED_ROLE

    sagemaker_session.create_auto_ml_v2(
        input_config=input_config,
        job_name=job_name,
        problem_config=problem_config,
        output_config=output_config,
        role=role,
        security_config=security_config,
        job_objective={"Type": "type", "MetricName": "metric-name"},
        tags=["tag"],
    )

    assert sagemaker_session.sagemaker_client.method_calls[0] == (
        "create_auto_ml_job_v2",
        (),
        COMPLETE_EXPECTED_AUTO_ML_V2_JOB_ARGS,
    )


def test_list_candidates_for_auto_ml_job_default(sagemaker_session):
    sagemaker_session.list_candidates(job_name=JOB_NAME)
    sagemaker_session.sagemaker_client.list_candidates_for_auto_ml_job.assert_called_once()
    sagemaker_session.sagemaker_client.list_candidates_for_auto_ml_job.assert_called_with(
        AutoMLJobName=JOB_NAME
    )


def test_list_candidates_for_auto_ml_job_with_optional_args(sagemaker_session):
    sagemaker_session.list_candidates(
        job_name=JOB_NAME,
        status_equals="Completed",
        sort_order="Descending",
        sort_by="Status",
        max_results=10,
    )
    sagemaker_session.sagemaker_client.list_candidates_for_auto_ml_job.assert_called_once()
    sagemaker_session.sagemaker_client.list_candidates_for_auto_ml_job.assert_called_with(
        **COMPLETE_EXPECTED_LIST_CANDIDATES_ARGS
    )


def test_describe_tuning_job(sagemaker_session):
    job_name = "hyper-parameter-tuning"
    sagemaker_session.describe_tuning_job(job_name=job_name)
    sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job.assert_called_with(
        HyperParameterTuningJobName=job_name
    )


@pytest.fixture
def pipeline_empty_definition():
    return '{"Version": "2020-12-01", ' '"Metadata": {}, ' '"Parameters": [], ' '"Steps": []}'


@pytest.fixture
def pipeline_role_arn():
    return "my:pipeline:role:arn"


def test_describe_model(sagemaker_session):
    model_name = "sagemaker-model-name"
    sagemaker_session.describe_model(name=model_name)
    sagemaker_session.sagemaker_client.describe_model.assert_called_with(ModelName=model_name)


def test_create_model_package_from_containers(sagemaker_session):
    model_package_name = "sagemaker-model-package"
    sagemaker_session.create_model_package_from_containers(model_package_name=model_package_name)
    sagemaker_session.sagemaker_client.create_model_package.assert_called_once()


def test_create_model_package_from_containers_cross_account_mpg_name(sagemaker_session):
    mpg_name = "arn:aws:sagemaker:us-east-1:215995503607:model-package-group/stage-dev"
    content_types = ["text/csv"]
    response_types = ["text/csv"]
    sagemaker_session.create_model_package_from_containers(
        model_package_group_name=mpg_name,
        content_types=content_types,
        response_types=response_types,
    )
    sagemaker_session.sagemaker_client.create_model_package.assert_called_once()


def test_create_mpg_from_containers_cross_account_mpg_name(sagemaker_session):
    mpg_name = "arn:aws:sagemaker:us-east-1:215995503607:model-package-group/stage-dev"
    content_types = ["text/csv"]
    response_types = ["text/csv"]
    with pytest.raises(AssertionError) as error:
        sagemaker_session.create_model_package_from_containers(
            model_package_group_name=mpg_name,
            content_types=content_types,
            response_types=response_types,
        )
        sagemaker_session.sagemaker_client.create_model_package_group.assert_called_once()
        assert (
            "Expected 'create_model_package_group' to have been called once. "
            "Called 0 times." == str(error)
        )


def test_create_model_package_from_containers_name_conflict(sagemaker_session):
    model_package_name = "sagemaker-model-package"
    model_package_group_name = "sagemaker-model-package-group"
    with pytest.raises(ValueError) as error:
        sagemaker_session.create_model_package_from_containers(
            model_package_name=model_package_name,
            model_package_group_name=model_package_group_name,
        )
        assert (
            "model_package_name and model_package_group_name cannot be present at the same "
            "time." == str(error)
        )


def test_create_model_package_from_containers_incomplete_args(sagemaker_session):
    model_package_name = "sagemaker-model-package"
    containers = ["dummy-container"]
    with pytest.raises(ValueError) as error:
        sagemaker_session.create_model_package_from_containers(
            model_package_name=model_package_name,
            containers=containers,
        )
        assert (
            "content_types and response_types "
            "must be provided if containers is present." == str(error)
        )


def test_create_model_package_from_containers_without_model_package_group_name(
    sagemaker_session,
):
    model_package_name = "sagemaker-model-package"
    containers = ["dummy-container"]
    content_types = ["application/json"]
    response_types = ["application/json"]
    with pytest.raises(ValueError) as error:
        sagemaker_session.create_model_package_from_containers(
            model_package_name=model_package_name,
            containers=containers,
            content_types=content_types,
            response_types=response_types,
        )
        assert (
            "inference_inferences and transform_instances "
            "must be provided if model_package_group_name is not present." == str(error)
        )


def test_create_model_package_with_sagemaker_config_injection(sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_MODEL_PACKAGE

    skip_model_validation = "All"
    model_package_name = "sagemaker-model-package"
    containers = [{"Image": "dummy-container"}]
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarget"]
    model_metrics = {
        "Bias": {
            "ContentType": "content-type",
            "S3Uri": "s3://...",
        }
    }
    drift_check_baselines = {
        "Bias": {
            "ConfigFile": {
                "ContentType": "content-type",
                "S3Uri": "s3://...",
            }
        }
    }
    validation_profiles = [
        {
            "ProfileName": "profileName",
            "TransformJobDefinition": {"TransformOutput": {"S3OutputPath": "s3://test"}},
        }
    ]
    validation_specification = {"ValidationProfiles": validation_profiles}

    metadata_properties = {
        "CommitId": "test-commit-id",
        "Repository": "test-repository",
        "GeneratedBy": "sagemaker-python-sdk",
        "ProjectId": "unit-test",
    }
    marketplace_cert = (True,)
    approval_status = ("Approved",)
    description = "description"
    customer_metadata_properties = {"key1": "value1"}
    domain = "COMPUTER_VISION"
    task = "IMAGE_CLASSIFICATION"
    sample_payload_url = "s3://test-bucket/model"
    sagemaker_session.create_model_package_from_containers(
        containers=containers,
        content_types=content_types,
        response_types=response_types,
        inference_instances=inference_instances,
        transform_instances=transform_instances,
        model_package_name=model_package_name,
        model_metrics=model_metrics,
        metadata_properties=metadata_properties,
        marketplace_cert=marketplace_cert,
        approval_status=approval_status,
        description=description,
        drift_check_baselines=drift_check_baselines,
        customer_metadata_properties=customer_metadata_properties,
        domain=domain,
        sample_payload_url=sample_payload_url,
        task=task,
        validation_specification=validation_specification,
        skip_model_validation=skip_model_validation,
    )
    expected_kms_key_id = SAGEMAKER_CONFIG_MODEL_PACKAGE["SageMaker"]["ModelPackage"][
        "ValidationSpecification"
    ]["ValidationProfiles"][0]["TransformJobDefinition"]["TransformOutput"]["KmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_MODEL_PACKAGE["SageMaker"]["ModelPackage"][
        "ValidationSpecification"
    ]["ValidationRole"]
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_MODEL_PACKAGE["SageMaker"]["ModelPackage"][
        "ValidationSpecification"
    ]["ValidationProfiles"][0]["TransformJobDefinition"]["TransformResources"]["VolumeKmsKeyId"]
    expected_environment = SAGEMAKER_CONFIG_MODEL_PACKAGE["SageMaker"]["ModelPackage"][
        "ValidationSpecification"
    ]["ValidationProfiles"][0]["TransformJobDefinition"]["Environment"]
    expected_containers_environment = SAGEMAKER_CONFIG_MODEL_PACKAGE["SageMaker"]["ModelPackage"][
        "InferenceSpecification"
    ]["Containers"][0]["Environment"]
    expected_args = copy.deepcopy(
        {
            "ModelPackageName": model_package_name,
            "InferenceSpecification": {
                "Containers": containers,
                "SupportedContentTypes": content_types,
                "SupportedResponseMIMETypes": response_types,
                "SupportedRealtimeInferenceInstanceTypes": inference_instances,
                "SupportedTransformInstanceTypes": transform_instances,
            },
            "ModelPackageDescription": description,
            "ModelMetrics": model_metrics,
            "MetadataProperties": metadata_properties,
            "CertifyForMarketplace": marketplace_cert,
            "ModelApprovalStatus": approval_status,
            "DriftCheckBaselines": drift_check_baselines,
            "CustomerMetadataProperties": customer_metadata_properties,
            "Domain": domain,
            "SamplePayloadUrl": sample_payload_url,
            "Task": task,
            "ValidationSpecification": validation_specification,
            "SkipModelValidation": skip_model_validation,
        }
    )
    expected_args["ValidationSpecification"]["ValidationRole"] = expected_role_arn
    expected_args["ValidationSpecification"]["ValidationProfiles"][0]["TransformJobDefinition"][
        "TransformResources"
    ]["VolumeKmsKeyId"] = expected_volume_kms_key_id
    expected_args["ValidationSpecification"]["ValidationProfiles"][0]["TransformJobDefinition"][
        "TransformOutput"
    ]["KmsKeyId"] = expected_kms_key_id
    expected_args["ValidationSpecification"]["ValidationProfiles"][0]["TransformJobDefinition"][
        "Environment"
    ] = expected_environment
    expected_args["InferenceSpecification"]["Containers"][0][
        "Environment"
    ] = expected_containers_environment

    sagemaker_session.sagemaker_client.create_model_package.assert_called_with(**expected_args)


def test_create_model_package_from_containers_all_args(sagemaker_session):
    model_package_name = "sagemaker-model-package"
    containers = ["dummy-container"]
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarget"]
    model_metrics = {
        "Bias": {
            "ContentType": "content-type",
            "S3Uri": "s3://...",
        }
    }
    drift_check_baselines = {
        "Bias": {
            "ConfigFile": {
                "ContentType": "content-type",
                "S3Uri": "s3://...",
            }
        }
    }

    metadata_properties = {
        "CommitId": "test-commit-id",
        "Repository": "test-repository",
        "GeneratedBy": "sagemaker-python-sdk",
        "ProjectId": "unit-test",
    }
    skip_model_validation = "All"
    marketplace_cert = (True,)
    approval_status = ("Approved",)
    description = "description"
    customer_metadata_properties = {"key1": "value1"}
    domain = "COMPUTER_VISION"
    task = "IMAGE_CLASSIFICATION"
    sample_payload_url = "s3://test-bucket/model"
    sagemaker_session.create_model_package_from_containers(
        containers=containers,
        content_types=content_types,
        response_types=response_types,
        inference_instances=inference_instances,
        transform_instances=transform_instances,
        model_package_name=model_package_name,
        model_metrics=model_metrics,
        metadata_properties=metadata_properties,
        marketplace_cert=marketplace_cert,
        approval_status=approval_status,
        description=description,
        drift_check_baselines=drift_check_baselines,
        customer_metadata_properties=customer_metadata_properties,
        domain=domain,
        sample_payload_url=sample_payload_url,
        task=task,
        skip_model_validation=skip_model_validation,
    )
    expected_args = {
        "ModelPackageName": model_package_name,
        "InferenceSpecification": {
            "Containers": containers,
            "SupportedContentTypes": content_types,
            "SupportedResponseMIMETypes": response_types,
            "SupportedRealtimeInferenceInstanceTypes": inference_instances,
            "SupportedTransformInstanceTypes": transform_instances,
        },
        "ModelPackageDescription": description,
        "ModelMetrics": model_metrics,
        "MetadataProperties": metadata_properties,
        "CertifyForMarketplace": marketplace_cert,
        "ModelApprovalStatus": approval_status,
        "DriftCheckBaselines": drift_check_baselines,
        "CustomerMetadataProperties": customer_metadata_properties,
        "Domain": domain,
        "SamplePayloadUrl": sample_payload_url,
        "Task": task,
        "SkipModelValidation": skip_model_validation,
    }
    sagemaker_session.sagemaker_client.create_model_package.assert_called_with(**expected_args)


def test_create_model_package_from_containers_without_instance_types(sagemaker_session):
    model_package_group_name = "sagemaker-model-package-group-name-1.0"
    containers = ["dummy-container"]
    content_types = ["application/json"]
    response_types = ["application/json"]
    model_metrics = {
        "Bias": {
            "ContentType": "content-type",
            "S3Uri": "s3://...",
        }
    }
    drift_check_baselines = {
        "Bias": {
            "ConfigFile": {
                "ContentType": "content-type",
                "S3Uri": "s3://...",
            }
        }
    }

    metadata_properties = {
        "CommitId": "test-commit-id",
        "Repository": "test-repository",
        "GeneratedBy": "sagemaker-python-sdk",
        "ProjectId": "unit-test",
    }
    skip_model_validation = "All"
    marketplace_cert = (True,)
    approval_status = ("Approved",)
    description = "description"
    customer_metadata_properties = {"key1": "value1"}
    sagemaker_session.create_model_package_from_containers(
        containers=containers,
        content_types=content_types,
        response_types=response_types,
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        metadata_properties=metadata_properties,
        marketplace_cert=marketplace_cert,
        approval_status=approval_status,
        description=description,
        drift_check_baselines=drift_check_baselines,
        customer_metadata_properties=customer_metadata_properties,
        skip_model_validation=skip_model_validation,
    )
    expected_args = {
        "ModelPackageGroupName": model_package_group_name,
        "InferenceSpecification": {
            "Containers": containers,
            "SupportedContentTypes": content_types,
            "SupportedResponseMIMETypes": response_types,
        },
        "ModelPackageDescription": description,
        "ModelMetrics": model_metrics,
        "MetadataProperties": metadata_properties,
        "CertifyForMarketplace": marketplace_cert,
        "ModelApprovalStatus": approval_status,
        "DriftCheckBaselines": drift_check_baselines,
        "CustomerMetadataProperties": customer_metadata_properties,
        "SkipModelValidation": skip_model_validation,
    }
    sagemaker_session.sagemaker_client.create_model_package.assert_called_with(**expected_args)


def test_create_model_package_from_containers_with_one_instance_types(
    sagemaker_session,
):
    model_package_group_name = "sagemaker-model-package-group-name-1.0"
    containers = ["dummy-container"]
    content_types = ["application/json"]
    response_types = ["application/json"]
    transform_instances = ["ml.m5.xlarge"]
    model_metrics = {
        "Bias": {
            "ContentType": "content-type",
            "S3Uri": "s3://...",
        }
    }
    drift_check_baselines = {
        "Bias": {
            "ConfigFile": {
                "ContentType": "content-type",
                "S3Uri": "s3://...",
            }
        }
    }

    metadata_properties = {
        "CommitId": "test-commit-id",
        "Repository": "test-repository",
        "GeneratedBy": "sagemaker-python-sdk",
        "ProjectId": "unit-test",
    }
    skip_model_validation = "All"
    marketplace_cert = (True,)
    approval_status = ("Approved",)
    description = "description"
    customer_metadata_properties = {"key1": "value1"}
    sagemaker_session.create_model_package_from_containers(
        containers=containers,
        content_types=content_types,
        response_types=response_types,
        transform_instances=transform_instances,
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        metadata_properties=metadata_properties,
        marketplace_cert=marketplace_cert,
        approval_status=approval_status,
        description=description,
        drift_check_baselines=drift_check_baselines,
        customer_metadata_properties=customer_metadata_properties,
        skip_model_validation=skip_model_validation,
    )
    expected_args = {
        "ModelPackageGroupName": model_package_group_name,
        "InferenceSpecification": {
            "Containers": containers,
            "SupportedContentTypes": content_types,
            "SupportedResponseMIMETypes": response_types,
            "SupportedTransformInstanceTypes": transform_instances,
        },
        "ModelPackageDescription": description,
        "ModelMetrics": model_metrics,
        "MetadataProperties": metadata_properties,
        "CertifyForMarketplace": marketplace_cert,
        "ModelApprovalStatus": approval_status,
        "DriftCheckBaselines": drift_check_baselines,
        "CustomerMetadataProperties": customer_metadata_properties,
        "SkipModelValidation": skip_model_validation,
    }
    sagemaker_session.sagemaker_client.create_model_package.assert_called_with(**expected_args)


@pytest.fixture
def feature_group_dummy_definitions():
    return [{"FeatureName": "feature1", "FeatureType": "String"}]


def test_feature_group_create_with_sagemaker_config_injection(
    sagemaker_session, feature_group_dummy_definitions
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_FEATURE_GROUP

    sagemaker_session.create_feature_group(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=feature_group_dummy_definitions,
        offline_store_config={"S3StorageConfig": {"S3Uri": "s3://test"}},
    )
    expected_offline_store_kms_key_id = SAGEMAKER_CONFIG_FEATURE_GROUP["SageMaker"]["FeatureGroup"][
        "OfflineStoreConfig"
    ]["S3StorageConfig"]["KmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_FEATURE_GROUP["SageMaker"]["FeatureGroup"]["RoleArn"]
    expected_online_store_kms_key_id = SAGEMAKER_CONFIG_FEATURE_GROUP["SageMaker"]["FeatureGroup"][
        "OnlineStoreConfig"
    ]["SecurityConfig"]["KmsKeyId"]
    expected_tags = SAGEMAKER_CONFIG_FEATURE_GROUP["SageMaker"]["FeatureGroup"]["Tags"]
    expected_request = {
        "FeatureGroupName": "MyFeatureGroup",
        "RecordIdentifierFeatureName": "feature1",
        "EventTimeFeatureName": "feature2",
        "FeatureDefinitions": feature_group_dummy_definitions,
        "RoleArn": expected_role_arn,
        "OnlineStoreConfig": {
            "SecurityConfig": {"KmsKeyId": expected_online_store_kms_key_id},
            "EnableOnlineStore": True,
        },
        "OfflineStoreConfig": {
            "S3StorageConfig": {
                "KmsKeyId": expected_offline_store_kms_key_id,
                "S3Uri": "s3://test",
            }
        },
        "Tags": expected_tags,
    }
    sagemaker_session.sagemaker_client.create_feature_group.assert_called_with(**expected_request)


def test_feature_group_create(sagemaker_session, feature_group_dummy_definitions):
    sagemaker_session.create_feature_group(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=feature_group_dummy_definitions,
        role_arn="dummy_role",
    )
    assert sagemaker_session.sagemaker_client.create_feature_group.called_with(
        FeatureGroupName="MyFeatureGroup",
        RecordIdentifierFeatureName="feature1",
        EventTimeFeatureName="feature2",
        FeatureDefinitions=feature_group_dummy_definitions,
        RoleArn="dummy_role",
    )


def test_feature_group_delete(sagemaker_session):
    sagemaker_session.delete_feature_group(feature_group_name="MyFeatureGroup")
    assert sagemaker_session.sagemaker_client.delete_feature_group.called_with(
        FeatureGroupName="MyFeatureGroup",
    )


def test_feature_group_describe(sagemaker_session):
    sagemaker_session.describe_feature_group(feature_group_name="MyFeatureGroup")
    assert sagemaker_session.sagemaker_client.describe_feature_group.called_with(
        FeatureGroupName="MyFeatureGroup",
    )


def test_feature_group_feature_additions_update(sagemaker_session, feature_group_dummy_definitions):
    sagemaker_session.update_feature_group(
        feature_group_name="MyFeatureGroup",
        feature_additions=feature_group_dummy_definitions,
    )
    assert sagemaker_session.sagemaker_client.update_feature_group.called_with(
        FeatureGroupName="MyFeatureGroup",
        FeatureAdditions=feature_group_dummy_definitions,
    )


def test_feature_group_online_store_config_update(sagemaker_session):
    os_conf_update = {"TtlDuration": {"Unit": "Seconds", "Value": 123}}
    sagemaker_session.update_feature_group(
        feature_group_name="MyFeatureGroup",
        online_store_config=os_conf_update,
    )
    assert sagemaker_session.sagemaker_client.update_feature_group.called_with(
        FeatureGroupName="MyFeatureGroup", OnlineStoreConfig=os_conf_update
    )


def test_feature_group_throughput_config_update(sagemaker_session):
    tp_update = {
        "ThroughputMode": "Provisioned",
        "ProvisionedReadCapacityUnits": 123,
        "ProvisionedWriteCapacityUnits": 456,
    }
    sagemaker_session.update_feature_group(
        feature_group_name="MyFeatureGroup",
        throughput_config=tp_update,
    )
    assert sagemaker_session.sagemaker_client.update_feature_group.called_with(
        FeatureGroupName="MyFeatureGroup", ThroughputConfig=tp_update
    )


def test_feature_metadata_update(sagemaker_session):
    parameter_additions = [
        {
            "key": "TestKey",
            "value": "TestValue",
        }
    ]
    parameter_removals = ["TestKey"]

    sagemaker_session.update_feature_metadata(
        feature_group_name="TestFeatureGroup",
        feature_name="TestFeature",
        description="TestDescription",
        parameter_additions=parameter_additions,
        parameter_removals=parameter_removals,
    )
    assert sagemaker_session.sagemaker_client.update_feature_group.called_with(
        feature_group_name="TestFeatureGroup",
        FeatureName="TestFeature",
        Description="TestDescription",
        ParameterAdditions=parameter_additions,
        ParameterRemovals=parameter_removals,
    )
    sagemaker_session.update_feature_metadata(
        feature_group_name="TestFeatureGroup",
        feature_name="TestFeature",
    )
    assert sagemaker_session.sagemaker_client.update_feature_group.called_with(
        feature_group_name="TestFeatureGroup",
        FeatureName="TestFeature",
    )


def test_feature_metadata_describe(sagemaker_session):
    sagemaker_session.describe_feature_metadata(
        feature_group_name="MyFeatureGroup", feature_name="TestFeature"
    )
    assert sagemaker_session.sagemaker_client.describe_feature_metadata.called_with(
        FeatureGroupName="MyFeatureGroup", FeatureName="TestFeature"
    )


def test_list_feature_groups(sagemaker_session):
    expected_list_feature_groups_args = {
        "NameContains": "MyFeatureGroup",
        "FeatureGroupStatusEquals": "Created",
        "OfflineStoreStatusEquals": "Active",
        "CreationTimeAfter": datetime.datetime(2020, 12, 1),
        "CreationTimeBefore": datetime.datetime(2022, 7, 1),
        "SortOrder": "Ascending",
        "SortBy": "Name",
        "MaxResults": 50,
        "NextToken": "token",
    }
    sagemaker_session.list_feature_groups(
        name_contains="MyFeatureGroup",
        feature_group_status_equals="Created",
        offline_store_status_equals="Active",
        creation_time_after=datetime.datetime(2020, 12, 1),
        creation_time_before=datetime.datetime(2022, 7, 1),
        sort_order="Ascending",
        sort_by="Name",
        max_results=50,
        next_token="token",
    )
    assert sagemaker_session.sagemaker_client.list_feature_groups.called_once()
    assert sagemaker_session.sagemaker_client.list_feature_groups.called_with(
        **expected_list_feature_groups_args
    )


@pytest.fixture()
def sagemaker_session_with_fs_runtime_client():
    boto_mock = MagicMock(name="boto_session")
    sagemaker_session = sagemaker.Session(
        boto_session=boto_mock, sagemaker_featurestore_runtime_client=MagicMock()
    )
    return sagemaker_session


def test_feature_group_put_record(sagemaker_session_with_fs_runtime_client):
    sagemaker_session_with_fs_runtime_client.put_record(
        feature_group_name="MyFeatureGroup",
        record=[{"FeatureName": "feature1", "ValueAsString": "value1"}],
    )
    fs_client_mock = sagemaker_session_with_fs_runtime_client.sagemaker_featurestore_runtime_client

    assert fs_client_mock.put_record.called_with(
        FeatureGroupName="MyFeatureGroup",
        record=[{"FeatureName": "feature1", "ValueAsString": "value1"}],
    )


def test_feature_group_put_record_with_ttl_and_target_stores(
    sagemaker_session_with_fs_runtime_client,
):
    sagemaker_session_with_fs_runtime_client.put_record(
        feature_group_name="MyFeatureGroup",
        record=[{"FeatureName": "feature1", "ValueAsString": "value1"}],
        ttl_duration={"Unit": "Seconds", "Value": 123},
        target_stores=["OnlineStore", "OfflineStore"],
    )
    fs_client_mock = sagemaker_session_with_fs_runtime_client.sagemaker_featurestore_runtime_client
    assert fs_client_mock.put_record.called_with(
        FeatureGroupName="MyFeatureGroup",
        record=[{"FeatureName": "feature1", "ValueAsString": "value1"}],
        target_stores=["OnlineStore", "OfflineStore"],
        ttl_duration={"Unit": "Seconds", "Value": 123},
    )


def test_start_query_execution(sagemaker_session):
    athena_mock = Mock()
    sagemaker_session.boto_session.client(
        "athena", region_name=sagemaker_session.boto_region_name
    ).return_value = athena_mock
    sagemaker_session.start_query_execution(
        catalog="catalog",
        database="database",
        query_string="query",
        output_location="s3://results",
    )
    assert athena_mock.start_query_execution.called_once_with(
        QueryString="query",
        QueryExecutionContext={"Catalog": "catalog", "Database": "database"},
        OutputLocation="s3://results",
    )


def test_get_query_execution(sagemaker_session):
    athena_mock = Mock()
    sagemaker_session.boto_session.client(
        "athena", region_name=sagemaker_session.boto_region_name
    ).return_value = athena_mock
    sagemaker_session.get_query_execution(query_execution_id="query_id")
    assert athena_mock.get_query_execution.called_with(QueryExecutionId="query_id")


def test_download_athena_query_result(sagemaker_session):
    sagemaker_session.s3_client = Mock()
    sagemaker_session.download_athena_query_result(
        bucket="bucket",
        prefix="prefix",
        query_execution_id="query_id",
        filename="filename",
    )
    assert sagemaker_session.s3_client.download_file.called_with(
        Bucket="bucket",
        Key="prefix/query_id.csv",
        Filename="filename",
    )


def test_update_monitoring_alert(sagemaker_session):
    sagemaker_session.update_monitoring_alert(
        monitoring_schedule_name="schedule-name",
        monitoring_alert_name="alert-name",
        data_points_to_alert=1,
        evaluation_period=1,
    )
    assert sagemaker_session.sagemaker_client.update_monitoring_alert.called_with(
        MonitoringScheduleName="schedule-name",
        MonitoringAlertName="alert-name",
        DatapointsToAlert=1,
        EvaluationPeriod=1,
    )


def test_list_monitoring_alerts(sagemaker_session):
    sagemaker_session.list_monitoring_alerts(
        monitoring_schedule_name="schedule-name",
        next_token="next_token",
        max_results=100,
    )
    assert sagemaker_session.sagemaker_client.list_monitoring_alerts.called_with(
        MonitoringScheduleName="schedule-name",
        NextToken="next_token",
        MaxResults=100,
    )


def test_list_monitoring_alert_history(sagemaker_session):
    sagemaker_session.list_monitoring_alert_history(
        monitoring_schedule_name="schedule-name",
        monitoring_alert_name="alert-name",
        sort_by="CreationTime",
        sort_order="Descending",
        next_token="next_token",
        max_results=100,
        status_equals="InAlert",
        creation_time_before="creation_time_before",
        creation_time_after="creation_time_after",
    )
    assert sagemaker_session.sagemaker_client.list_monitoring_alerts.called_with(
        MonitoringScheduleName="schedule-name",
        MonitoringAlertName="alert-name",
        SortBy="CreationTime",
        SortOrder="Descending",
        NextToken="next_token",
        MaxResults=100,
        CreationTimeBefore="creation_time_before",
        CreationTimeAfter="creation_time_after",
        StatusEquals="InAlert",
    )


@patch("sagemaker.session.Session.get_query_execution")
def test_wait_for_athena_query(query_execution, sagemaker_session):
    query_execution.return_value = {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}
    sagemaker_session.wait_for_athena_query(query_execution_id="query_id")
    assert query_execution.called_with(query_execution_id="query_id")


def test_search(sagemaker_session):
    expected_search_args = {
        "Resource": "FeatureGroup",
        "SearchExpression": {
            "Filters": [
                {
                    "Name": "FeatureGroupName",
                    "Value": "MyFeatureGroup",
                    "Operator": "Contains",
                }
            ],
            "Operator": "And",
        },
        "SortBy": "Name",
        "SortOrder": "Ascending",
        "NextToken": "token",
        "MaxResults": 50,
    }
    sagemaker_session.search(
        resource="FeatureGroup",
        search_expression={
            "Filters": [
                {
                    "Name": "FeatureGroupName",
                    "Value": "MyFeatureGroup",
                    "Operator": "Contains",
                }
            ],
            "Operator": "And",
        },
        sort_by="Name",
        sort_order="Ascending",
        next_token="token",
        max_results=50,
    )
    assert sagemaker_session.sagemaker_client.search.called_once()
    assert sagemaker_session.sagemaker_client.search.called_with(**expected_search_args)


def test_batch_get_record(sagemaker_session):
    expected_batch_get_record_args = {
        "Identifiers": [
            {
                "FeatureGroupName": "name",
                "RecordIdentifiersValueAsString": ["identifier"],
                "FeatureNames": ["feature_1"],
            }
        ]
    }
    sagemaker_session.batch_get_record(
        identifiers=[
            {
                "FeatureGroupName": "name",
                "RecordIdentifiersValueAsString": ["identifier"],
                "FeatureNames": ["feature_1"],
            }
        ]
    )
    assert sagemaker_session.sagemaker_client.batch_get_record.called_once()
    assert sagemaker_session.sagemaker_client.batch_get_record.called_with(
        **expected_batch_get_record_args
    )


def test_batch_get_record_expiration_time_response(sagemaker_session):
    expected_batch_get_record_args = {
        "Identifiers": [
            {
                "FeatureGroupName": "name",
                "RecordIdentifiersValueAsString": ["identifier"],
                "FeatureNames": ["feature_1"],
            }
        ],
        "ExpirationTimeResponse": "Disabled",
    }
    sagemaker_session.batch_get_record(
        identifiers=[
            {
                "FeatureGroupName": "name",
                "RecordIdentifiersValueAsString": ["identifier"],
                "FeatureNames": ["feature_1"],
            }
        ],
        expiration_time_response="Disabled",
    )
    assert sagemaker_session.sagemaker_client.batch_get_record.called_once()
    assert sagemaker_session.sagemaker_client.batch_get_record.called_with(
        **expected_batch_get_record_args
    )


IR_USER_JOB_NAME = "custom-job-name"
IR_JOB_NAME = "SMPYTHONSDK-sample-unique-uuid"
IR_ADVANCED_JOB = "Advanced"
IR_ROLE_ARN = "arn:aws:iam::123456789123:role/service-role/AmazonSageMaker-ExecutionRole-UnitTest"
IR_SAMPLE_PAYLOAD_URL = "s3://sagemaker-us-west-2-123456789123/payload/payload.tar.gz"
IR_SUPPORTED_CONTENT_TYPES = ["text/csv"]
IR_MODEL_PACKAGE_VERSION_ARN = (
    "arn:aws:sagemaker:us-west-2:123456789123:model-package/unit-test-package-version/1"
)
IR_MODEL_NAME = "MODEL_NAME"
IR_NEAREST_MODEL_NAME = "xgboost"
IR_SUPPORTED_INSTANCE_TYPES = ["ml.c5.xlarge", "ml.c5.2xlarge"]
IR_FRAMEWORK = "XGBOOST"
IR_FRAMEWORK_VERSION = "1.2.0"
IR_NEAREST_MODEL_NAME = "xgboost"
IR_JOB_DURATION_IN_SECONDS = 7200
IR_ENDPOINT_CONFIGURATIONS = [
    {
        "EnvironmentParameterRanges": {
            "CategoricalParameterRanges": [{"Name": "OMP_NUM_THREADS", "Value": ["2", "4", "10"]}]
        },
        "InferenceSpecificationName": "unit-test-specification",
        "InstanceType": "ml.c5.xlarge",
    }
]
IR_TRAFFIC_PATTERN = {
    "Phases": [{"DurationInSeconds": 120, "InitialNumberOfUsers": 1, "SpawnRate": 1}],
    "TrafficType": "PHASES",
}
IR_STOPPING_CONDITIONS = {
    "MaxInvocations": 300,
    "ModelLatencyThresholds": [{"Percentile": "P95", "ValueInMilliseconds": 100}],
}
IR_RESOURCE_LIMIT = {"MaxNumberOfTests": 10, "MaxParallelOfTests": 1}


def create_inference_recommendations_job_default_happy_response():
    return {
        "JobName": IR_USER_JOB_NAME,
        "JobType": "Default",
        "RoleArn": IR_ROLE_ARN,
        "InputConfig": {
            "ContainerConfig": {
                "Domain": "MACHINE_LEARNING",
                "Task": "OTHER",
                "Framework": IR_FRAMEWORK,
                "PayloadConfig": {
                    "SamplePayloadUrl": IR_SAMPLE_PAYLOAD_URL,
                    "SupportedContentTypes": IR_SUPPORTED_CONTENT_TYPES,
                },
                "FrameworkVersion": IR_FRAMEWORK_VERSION,
                "NearestModelName": IR_NEAREST_MODEL_NAME,
                "SupportedInstanceTypes": IR_SUPPORTED_INSTANCE_TYPES,
            },
            "ModelPackageVersionArn": IR_MODEL_PACKAGE_VERSION_ARN,
        },
        "JobDescription": "#python-sdk-create",
        "Tags": [{"Key": "ClientType", "Value": "PythonSDK-RightSize"}],
    }


def create_inference_recommendations_job_default_model_name_happy_response():
    return {
        "JobName": IR_USER_JOB_NAME,
        "JobType": "Default",
        "RoleArn": IR_ROLE_ARN,
        "InputConfig": {
            "ContainerConfig": {
                "Domain": "MACHINE_LEARNING",
                "Task": "OTHER",
                "Framework": IR_FRAMEWORK,
                "PayloadConfig": {
                    "SamplePayloadUrl": IR_SAMPLE_PAYLOAD_URL,
                    "SupportedContentTypes": IR_SUPPORTED_CONTENT_TYPES,
                },
                "FrameworkVersion": IR_FRAMEWORK_VERSION,
                "NearestModelName": IR_NEAREST_MODEL_NAME,
                "SupportedInstanceTypes": IR_SUPPORTED_INSTANCE_TYPES,
            },
            "ModelName": IR_MODEL_NAME,
        },
        "JobDescription": "#python-sdk-create",
        "Tags": [{"Key": "ClientType", "Value": "PythonSDK-RightSize"}],
    }


def create_inference_recommendations_job_advanced_happy_response():
    base_advanced_job_response = create_inference_recommendations_job_default_happy_response()

    base_advanced_job_response["JobName"] = IR_JOB_NAME
    base_advanced_job_response["JobType"] = IR_ADVANCED_JOB
    base_advanced_job_response["StoppingConditions"] = IR_STOPPING_CONDITIONS
    base_advanced_job_response["InputConfig"]["JobDurationInSeconds"] = IR_JOB_DURATION_IN_SECONDS
    base_advanced_job_response["InputConfig"]["EndpointConfigurations"] = IR_ENDPOINT_CONFIGURATIONS
    base_advanced_job_response["InputConfig"]["TrafficPattern"] = IR_TRAFFIC_PATTERN
    base_advanced_job_response["InputConfig"]["ResourceLimit"] = IR_RESOURCE_LIMIT

    return base_advanced_job_response


def create_inference_recommendations_job_advanced_model_name_happy_response():
    base_advanced_job_response = (
        create_inference_recommendations_job_default_model_name_happy_response()
    )

    base_advanced_job_response["JobName"] = IR_JOB_NAME
    base_advanced_job_response["JobType"] = IR_ADVANCED_JOB
    base_advanced_job_response["StoppingConditions"] = IR_STOPPING_CONDITIONS
    base_advanced_job_response["InputConfig"]["JobDurationInSeconds"] = IR_JOB_DURATION_IN_SECONDS
    base_advanced_job_response["InputConfig"]["EndpointConfigurations"] = IR_ENDPOINT_CONFIGURATIONS
    base_advanced_job_response["InputConfig"]["TrafficPattern"] = IR_TRAFFIC_PATTERN
    base_advanced_job_response["InputConfig"]["ResourceLimit"] = IR_RESOURCE_LIMIT

    return base_advanced_job_response


def test_create_inference_recommendations_job_default_happy(sagemaker_session):
    job_name = sagemaker_session.create_inference_recommendations_job(
        role=IR_ROLE_ARN,
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        model_package_version_arn=IR_MODEL_PACKAGE_VERSION_ARN,
        framework=IR_FRAMEWORK,
        framework_version=IR_FRAMEWORK_VERSION,
        nearest_model_name=IR_NEAREST_MODEL_NAME,
        supported_instance_types=IR_SUPPORTED_INSTANCE_TYPES,
        job_name=IR_USER_JOB_NAME,
    )

    sagemaker_session.sagemaker_client.create_inference_recommendations_job.assert_called_with(
        **create_inference_recommendations_job_default_happy_response()
    )

    assert IR_USER_JOB_NAME == job_name


@patch("uuid.uuid4", MagicMock(return_value="sample-unique-uuid"))
def test_create_inference_recommendations_job_advanced_happy(sagemaker_session):
    job_name = sagemaker_session.create_inference_recommendations_job(
        role=IR_ROLE_ARN,
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        model_package_version_arn=IR_MODEL_PACKAGE_VERSION_ARN,
        framework=IR_FRAMEWORK,
        framework_version=IR_FRAMEWORK_VERSION,
        nearest_model_name=IR_NEAREST_MODEL_NAME,
        supported_instance_types=IR_SUPPORTED_INSTANCE_TYPES,
        endpoint_configurations=IR_ENDPOINT_CONFIGURATIONS,
        traffic_pattern=IR_TRAFFIC_PATTERN,
        stopping_conditions=IR_STOPPING_CONDITIONS,
        resource_limit=IR_RESOURCE_LIMIT,
        job_type=IR_ADVANCED_JOB,
        job_duration_in_seconds=IR_JOB_DURATION_IN_SECONDS,
    )

    sagemaker_session.sagemaker_client.create_inference_recommendations_job.assert_called_with(
        **create_inference_recommendations_job_advanced_happy_response()
    )

    assert IR_JOB_NAME == job_name


def test_create_inference_recommendations_job_default_model_name_happy(
    sagemaker_session,
):
    job_name = sagemaker_session.create_inference_recommendations_job(
        role=IR_ROLE_ARN,
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        model_name=IR_MODEL_NAME,
        model_package_version_arn=None,
        framework=IR_FRAMEWORK,
        framework_version=IR_FRAMEWORK_VERSION,
        nearest_model_name=IR_NEAREST_MODEL_NAME,
        supported_instance_types=IR_SUPPORTED_INSTANCE_TYPES,
        job_name=IR_USER_JOB_NAME,
    )

    sagemaker_session.sagemaker_client.create_inference_recommendations_job.assert_called_with(
        **create_inference_recommendations_job_default_model_name_happy_response()
    )

    assert IR_USER_JOB_NAME == job_name


@patch("uuid.uuid4", MagicMock(return_value="sample-unique-uuid"))
def test_create_inference_recommendations_job_advanced_model_name_happy(
    sagemaker_session,
):
    job_name = sagemaker_session.create_inference_recommendations_job(
        role=IR_ROLE_ARN,
        sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
        supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
        model_name=IR_MODEL_NAME,
        model_package_version_arn=None,
        framework=IR_FRAMEWORK,
        framework_version=IR_FRAMEWORK_VERSION,
        nearest_model_name=IR_NEAREST_MODEL_NAME,
        supported_instance_types=IR_SUPPORTED_INSTANCE_TYPES,
        endpoint_configurations=IR_ENDPOINT_CONFIGURATIONS,
        traffic_pattern=IR_TRAFFIC_PATTERN,
        stopping_conditions=IR_STOPPING_CONDITIONS,
        resource_limit=IR_RESOURCE_LIMIT,
        job_type=IR_ADVANCED_JOB,
        job_duration_in_seconds=IR_JOB_DURATION_IN_SECONDS,
    )

    sagemaker_session.sagemaker_client.create_inference_recommendations_job.assert_called_with(
        **create_inference_recommendations_job_advanced_model_name_happy_response()
    )

    assert IR_JOB_NAME == job_name


def test_create_inference_recommendations_job_missing_model_name_and_pkg(
    sagemaker_session,
):
    with pytest.raises(
        ValueError,
        match="Please provide either model_name or model_package_version_arn.",
    ):
        sagemaker_session.create_inference_recommendations_job(
            role=IR_ROLE_ARN,
            sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
            supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
            model_name=None,
            model_package_version_arn=None,
            framework=IR_FRAMEWORK,
            framework_version=IR_FRAMEWORK_VERSION,
            nearest_model_name=IR_NEAREST_MODEL_NAME,
            supported_instance_types=IR_SUPPORTED_INSTANCE_TYPES,
            job_name=IR_USER_JOB_NAME,
        )


def test_create_inference_recommendations_job_provided_model_name_and_pkg(
    sagemaker_session,
):
    with pytest.raises(
        ValueError,
        match="Please provide either model_name or model_package_version_arn.",
    ):
        sagemaker_session.create_inference_recommendations_job(
            role=IR_ROLE_ARN,
            sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
            supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
            model_name=IR_MODEL_NAME,
            model_package_version_arn=IR_MODEL_PACKAGE_VERSION_ARN,
            framework=IR_FRAMEWORK,
            framework_version=IR_FRAMEWORK_VERSION,
            nearest_model_name=IR_NEAREST_MODEL_NAME,
            supported_instance_types=IR_SUPPORTED_INSTANCE_TYPES,
            job_name=IR_USER_JOB_NAME,
        )


def test_create_inference_recommendations_job_propogate_validation_exception(
    sagemaker_session,
):
    validation_exception_message = (
        "Failed to describe model due to validation failure with following error: test_error"
    )

    validation_exception = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": validation_exception_message,
            }
        },
        "create_inference_recommendations_job",
    )

    sagemaker_session.sagemaker_client.create_inference_recommendations_job.side_effect = (
        validation_exception
    )

    with pytest.raises(ClientError) as error:
        sagemaker_session.create_inference_recommendations_job(
            role=IR_ROLE_ARN,
            sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
            supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
            model_package_version_arn=IR_MODEL_PACKAGE_VERSION_ARN,
            framework=IR_FRAMEWORK,
            framework_version=IR_FRAMEWORK_VERSION,
            nearest_model_name=IR_NEAREST_MODEL_NAME,
            supported_instance_types=IR_SUPPORTED_INSTANCE_TYPES,
        )

    assert "ValidationException" in str(error)


def test_create_inference_recommendations_job_propogate_other_exception(
    sagemaker_session,
):
    access_denied_exception_message = "Access is not allowed for the caller."

    access_denied_exception = ClientError(
        {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": access_denied_exception_message,
            }
        },
        "create_inference_recommendations_job",
    )

    sagemaker_session.sagemaker_client.create_inference_recommendations_job.side_effect = (
        access_denied_exception
    )

    with pytest.raises(ClientError) as error:
        sagemaker_session.create_inference_recommendations_job(
            role=IR_ROLE_ARN,
            sample_payload_url=IR_SAMPLE_PAYLOAD_URL,
            supported_content_types=IR_SUPPORTED_CONTENT_TYPES,
            model_package_version_arn=IR_MODEL_PACKAGE_VERSION_ARN,
            framework=IR_FRAMEWORK,
            framework_version=IR_FRAMEWORK_VERSION,
            nearest_model_name=IR_NEAREST_MODEL_NAME,
            supported_instance_types=IR_SUPPORTED_INSTANCE_TYPES,
        )

    assert "AccessDeniedException" in str(error)


DEFAULT_LOG_EVENTS_INFERENCE_RECOMMENDER = [
    MockBotoException("ResourceNotFoundException"),
    {"nextForwardToken": None, "events": [{"timestamp": 1, "message": "hi there #1"}]},
    {"nextForwardToken": None, "events": [{"timestamp": 2, "message": "hi there #2"}]},
    {"nextForwardToken": None, "events": [{"timestamp": 3, "message": "hi there #3"}]},
    {"nextForwardToken": None, "events": [{"timestamp": 4, "message": "hi there #4"}]},
]

FLUSH_LOG_EVENTS_INFERENCE_RECOMMENDER = [
    MockBotoException("ResourceNotFoundException"),
    {"nextForwardToken": None, "events": [{"timestamp": 1, "message": "hi there #1"}]},
    {"nextForwardToken": None, "events": [{"timestamp": 2, "message": "hi there #2"}]},
    {"nextForwardToken": None, "events": []},
    {"nextForwardToken": None, "events": [{"timestamp": 3, "message": "hi there #3"}]},
    {"nextForwardToken": None, "events": []},
    {"nextForwardToken": None, "events": [{"timestamp": 4, "message": "hi there #4"}]},
]

INFERENCE_RECOMMENDATIONS_DESC_STATUS_PENDING = {"Status": "PENDING"}
INFERENCE_RECOMMENDATIONS_DESC_STATUS_IN_PROGRESS = {"Status": "IN_PROGRESS"}
INFERENCE_RECOMMENDATIONS_DESC_STATUS_COMPLETED = {"Status": "COMPLETED"}


@pytest.fixture()
def sm_session_inference_recommender():
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("logs").get_log_events.side_effect = DEFAULT_LOG_EVENTS_INFERENCE_RECOMMENDER

    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())

    ims.sagemaker_client.describe_inference_recommendations_job.side_effect = [
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_PENDING,
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_IN_PROGRESS,
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_COMPLETED,
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_COMPLETED,
    ]

    return ims


@pytest.fixture()
def sm_session_inference_recommender_flush():
    boto_mock = MagicMock(name="boto_session")
    boto_mock.client("logs").get_log_events.side_effect = FLUSH_LOG_EVENTS_INFERENCE_RECOMMENDER

    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())

    ims.sagemaker_client.describe_inference_recommendations_job.side_effect = [
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_PENDING,
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_IN_PROGRESS,
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_IN_PROGRESS,
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_COMPLETED,
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_COMPLETED,
        INFERENCE_RECOMMENDATIONS_DESC_STATUS_COMPLETED,
    ]

    return ims


@patch("time.sleep")
def test_wait_for_inference_recommendations_job_completed(sleep, sm_session_inference_recommender):
    assert (
        sm_session_inference_recommender.wait_for_inference_recommendations_job(
            JOB_NAME, log_level="Quiet"
        )["Status"]
        == "COMPLETED"
    )

    assert (
        4
        == sm_session_inference_recommender.sagemaker_client.describe_inference_recommendations_job.call_count
    )
    assert 3 == sleep.call_count
    sleep.assert_has_calls([call(120), call(120), call(120)])


def test_wait_for_inference_recommendations_job_failed(sagemaker_session):
    inference_recommendations_desc_status_failed = {
        "Status": "FAILED",
        "FailureReason": "Mock Failure Reason",
    }

    sagemaker_session.sagemaker_client.describe_inference_recommendations_job = Mock(
        name="describe_inference_recommendations_job",
        return_value=inference_recommendations_desc_status_failed,
    )

    with pytest.raises(exceptions.UnexpectedStatusException) as error:
        sagemaker_session.wait_for_inference_recommendations_job(JOB_NAME)

    assert "Mock Failure Reason" in str(error)


@patch("builtins.print")
@patch("time.sleep")
def test_wait_for_inference_recommendations_job_completed_verbose(
    sleep, mock_print, sm_session_inference_recommender
):
    assert (
        sm_session_inference_recommender.wait_for_inference_recommendations_job(
            JOB_NAME, log_level="Verbose"
        )["Status"]
        == "COMPLETED"
    )
    assert (
        4
        == sm_session_inference_recommender.sagemaker_client.describe_inference_recommendations_job.call_count
    )

    assert (
        5 == sm_session_inference_recommender.boto_session.client("logs").get_log_events.call_count
    )

    assert 3 == sleep.call_count
    sleep.assert_has_calls([call(10), call(60), call(60)])

    assert 8 == mock_print.call_count


@patch("builtins.print")
@patch("time.sleep")
def test_wait_for_inference_recommendations_job_flush_completed(
    sleep, mock_print, sm_session_inference_recommender_flush
):
    assert (
        sm_session_inference_recommender_flush.wait_for_inference_recommendations_job(
            JOB_NAME, log_level="Verbose"
        )["Status"]
        == "COMPLETED"
    )
    assert (
        6
        == sm_session_inference_recommender_flush.sagemaker_client.describe_inference_recommendations_job.call_count
    )

    assert (
        7
        == sm_session_inference_recommender_flush.boto_session.client(
            "logs"
        ).get_log_events.call_count
    )

    assert 5 == sleep.call_count
    sleep.assert_has_calls([call(10), call(60), call(60), call(60), call(60)])

    assert 8 == mock_print.call_count


def test_wait_for_inference_recommendations_job_invalid_log_level(sagemaker_session):
    with pytest.raises(ValueError) as error:
        sagemaker_session.wait_for_inference_recommendations_job(
            JOB_NAME, log_level="invalid_log_level"
        )

    assert "log_level must be either Quiet or Verbose" in str(error)


def test_append_sagemaker_config_tags(sagemaker_session):
    tags_base = [
        {"Key": "tagkey4", "Value": "000"},
        {"Key": "tagkey5", "Value": "000"},
    ]
    tags_duplicate = [
        {"Key": "tagkey1", "Value": "000"},
        {"Key": "tagkey2", "Value": "000"},
    ]
    tags_none = None
    tags_empty = []

    # Helper to sort the lists so that the test is not dependent on order
    def sort(tags):
        return tags.sort(key=lambda tag: tag["Key"])

    config_tag_value = [
        {"Key": "tagkey1", "Value": "tagvalue1"},
        {"Key": "tagkey2", "Value": "tagvalue2"},
        {"Key": "tagkey3", "Value": "tagvalue3"},
    ]
    sagemaker_session.sagemaker_config = {"SchemaVersion": "1.0"}
    sagemaker_session.sagemaker_config.update(
        {"SageMaker": {"ProcessingJob": {"Tags": config_tag_value}}}
    )
    config_key_path = "SageMaker.ProcessingJob.Tags"

    base_case = sagemaker_session._append_sagemaker_config_tags(tags_base, config_key_path)
    assert sort(base_case) == sort(
        [
            {"Key": "tagkey1", "Value": "tagvalue1"},
            {"Key": "tagkey2", "Value": "tagvalue2"},
            {"Key": "tagkey3", "Value": "tagvalue3"},
            {"Key": "tagkey4", "Value": "000"},
            {"Key": "tagkey5", "Value": "000"},
        ]
    )

    duplicate_case = sagemaker_session._append_sagemaker_config_tags(
        tags_duplicate, config_key_path
    )
    assert sort(duplicate_case) == sort(
        [
            {"Key": "tagkey1", "Value": "000"},
            {"Key": "tagkey2", "Value": "000"},
            {"Key": "tagkey3", "Value": "tagvalue3"},
        ]
    )

    none_case = sagemaker_session._append_sagemaker_config_tags(tags_none, config_key_path)
    assert sort(none_case) == sort(
        [
            {"Key": "tagkey1", "Value": "tagvalue1"},
            {"Key": "tagkey2", "Value": "tagvalue2"},
            {"Key": "tagkey3", "Value": "tagvalue3"},
        ]
    )

    empty_case = sagemaker_session._append_sagemaker_config_tags(tags_empty, config_key_path)
    assert sort(empty_case) == sort(
        [
            {"Key": "tagkey1", "Value": "tagvalue1"},
            {"Key": "tagkey2", "Value": "tagvalue2"},
            {"Key": "tagkey3", "Value": "tagvalue3"},
        ]
    )

    sagemaker_session.sagemaker_config.update(
        {"SageMaker": {"TrainingJob": {"Tags": config_tag_value}}}
    )
    config_tags_none = sagemaker_session._append_sagemaker_config_tags(tags_base, config_key_path)
    assert sort(config_tags_none) == sort(
        [
            {"Key": "tagkey4", "Value": "000"},
            {"Key": "tagkey5", "Value": "000"},
        ]
    )

    sagemaker_session.sagemaker_config.update(
        {"SageMaker": {"ProcessingJob": {"Tags": config_tag_value}}}
    )
    config_tags_empty = sagemaker_session._append_sagemaker_config_tags(
        tags_base, "DUMMY.CONFIG.PATH"
    )
    assert sort(config_tags_empty) == sort(
        [
            {"Key": "tagkey4", "Value": "000"},
            {"Key": "tagkey5", "Value": "000"},
        ]
    )


@pytest.mark.parametrize(
    (
        "file_path, user_input_params, "
        "expected__without_user_input__with_default_bucket_and_default_prefix, "
        "expected__without_user_input__with_default_bucket_only, "
        "expected__with_user_input__with_default_bucket_and_prefix, "
        "expected__with_user_input__with_default_bucket_only"
    ),
    [
        # Group with just bucket as user input
        (
            "some/local/path/model.gz",
            {"bucket": "input-bucket"},
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/data/model.gz",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/data/model.gz",
            "s3://input-bucket/data/model.gz",
            "s3://input-bucket/data/model.gz",
        ),
        (
            "some/local/path/dir",
            {"bucket": "input-bucket"},
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/data/dir",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/data/dir",
            "s3://input-bucket/data/dir",
            "s3://input-bucket/data/dir",
        ),
        # Group with both bucket and prefix as user input
        (
            "some/local/path/model.gz",
            {"bucket": "input-bucket", "key_prefix": "input-prefix"},
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/data/model.gz",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/data/model.gz",
            "s3://input-bucket/input-prefix/model.gz",
            "s3://input-bucket/input-prefix/model.gz",
        ),
        (
            "some/local/path/dir",
            {"bucket": "input-bucket", "key_prefix": "input-prefix"},
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/data/dir",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/data/dir",
            "s3://input-bucket/input-prefix/dir",
            "s3://input-bucket/input-prefix/dir",
        ),
        # Group with just prefix as user input
        (
            "some/local/path/model.gz",
            {"key_prefix": "input-prefix"},
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/data/model.gz",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/data/model.gz",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/input-prefix/model.gz",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/input-prefix/model.gz",
        ),
        (
            "some/local/path/dir",
            {"key_prefix": "input-prefix"},
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/data/dir",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/data/dir",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/input-prefix/dir",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/input-prefix/dir",
        ),
        (
            "some/local/path/dir",
            {"key_prefix": "input-prefix/longer/path/"},
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/data/dir",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/data/dir",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/input-prefix/longer/path/dir",
            f"s3://{DEFAULT_S3_BUCKET_NAME}/input-prefix/longer/path/dir",
        ),
    ],
)
def test_upload_data_default_bucket_and_prefix_combinations(
    sagemaker_session,
    file_path,
    user_input_params,
    expected__without_user_input__with_default_bucket_and_default_prefix,
    expected__without_user_input__with_default_bucket_only,
    expected__with_user_input__with_default_bucket_and_prefix,
    expected__with_user_input__with_default_bucket_only,
):
    sagemaker_session.s3_resource = Mock()
    sagemaker_session._default_bucket = DEFAULT_S3_BUCKET_NAME

    session_with_bucket_and_prefix = copy.deepcopy(sagemaker_session)
    session_with_bucket_and_prefix.default_bucket_prefix = DEFAULT_S3_OBJECT_KEY_PREFIX_NAME

    session_with_bucket_and_no_prefix = copy.deepcopy(sagemaker_session)
    session_with_bucket_and_no_prefix.default_bucket_prefix = None

    actual, expected = _test_default_bucket_and_prefix_combinations(
        session_with_bucket_and_prefix=session_with_bucket_and_prefix,
        session_with_bucket_and_no_prefix=session_with_bucket_and_no_prefix,
        function_with_user_input=(lambda sess: sess.upload_data(file_path, **user_input_params)),
        function_without_user_input=(lambda sess: sess.upload_data(file_path)),
        expected__without_user_input__with_default_bucket_and_default_prefix=(
            expected__without_user_input__with_default_bucket_and_default_prefix
        ),
        expected__without_user_input__with_default_bucket_only=expected__without_user_input__with_default_bucket_only,
        expected__with_user_input__with_default_bucket_and_prefix=(
            expected__with_user_input__with_default_bucket_and_prefix
        ),
        expected__with_user_input__with_default_bucket_only=expected__with_user_input__with_default_bucket_only,
    )
    assert actual == expected


def test_is_inference_component_based_endpoint_affirmative(sagemaker_session):

    describe_endpoint_response = {"EndpointConfigName": "some-endpoint-config"}
    describe_endpoint_config_response = {
        "ExecutionRoleArn": "some-role-arn",
        "ProductionVariants": [{"VariantName": "AllTraffic"}],
    }

    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value=describe_endpoint_response
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config = Mock(
        return_value=describe_endpoint_config_response
    )

    assert sagemaker_session.is_inference_component_based_endpoint("endpoint-name")
    sagemaker_session.sagemaker_client.describe_endpoint.assert_called_once_with(
        EndpointName="endpoint-name"
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config.assert_called_once_with(
        EndpointConfigName="some-endpoint-config"
    )


def test_is_inference_component_based_endpoint_negative_no_role(sagemaker_session):

    describe_endpoint_response = {"EndpointConfigName": "some-endpoint-config"}
    describe_endpoint_config_response = {
        "ProductionVariants": [{"VariantName": "AllTraffic"}],
    }

    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value=describe_endpoint_response
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config = Mock(
        return_value=describe_endpoint_config_response
    )

    assert not sagemaker_session.is_inference_component_based_endpoint("endpoint-name")
    sagemaker_session.sagemaker_client.describe_endpoint.assert_called_once_with(
        EndpointName="endpoint-name"
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config.assert_called_once_with(
        EndpointConfigName="some-endpoint-config"
    )


def test_is_inference_component_based_endpoint_positive_multiple_variants(sagemaker_session):

    describe_endpoint_response = {"EndpointConfigName": "some-endpoint-config"}
    describe_endpoint_config_response = {
        "ExecutionRoleArn": "some-role-arn",
        "ProductionVariants": [{"VariantName": "AllTraffic1"}, {"VariantName": "AllTraffic2"}],
    }

    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value=describe_endpoint_response
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config = Mock(
        return_value=describe_endpoint_config_response
    )

    assert sagemaker_session.is_inference_component_based_endpoint("endpoint-name")
    sagemaker_session.sagemaker_client.describe_endpoint.assert_called_once_with(
        EndpointName="endpoint-name"
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config.assert_called_once_with(
        EndpointConfigName="some-endpoint-config"
    )


def test_is_inference_component_based_endpoint_negative_no_variants(sagemaker_session):

    describe_endpoint_response = {"EndpointConfigName": "some-endpoint-config"}
    describe_endpoint_config_response = {
        "ExecutionRoleArn": "some-role-arn",
        "ProductionVariants": [],
    }

    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value=describe_endpoint_response
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config = Mock(
        return_value=describe_endpoint_config_response
    )

    assert not sagemaker_session.is_inference_component_based_endpoint("endpoint-name")
    sagemaker_session.sagemaker_client.describe_endpoint.assert_called_once_with(
        EndpointName="endpoint-name"
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config.assert_called_once_with(
        EndpointConfigName="some-endpoint-config"
    )


def test_is_inference_component_based_endpoint_negative_model_name_present(sagemaker_session):

    describe_endpoint_response = {"EndpointConfigName": "some-endpoint-config"}
    describe_endpoint_config_response = {
        "ExecutionRoleArn": "some-role-arn",
        "ProductionVariants": [
            {"VariantName": "AllTraffic", "ModelName": "blah"},
            {"VariantName": "AllTraffic1"},
        ],
    }

    sagemaker_session.sagemaker_client.describe_endpoint = Mock(
        return_value=describe_endpoint_response
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config = Mock(
        return_value=describe_endpoint_config_response
    )

    assert not sagemaker_session.is_inference_component_based_endpoint("endpoint-name")
    sagemaker_session.sagemaker_client.describe_endpoint.assert_called_once_with(
        EndpointName="endpoint-name"
    )
    sagemaker_session.sagemaker_client.describe_endpoint_config.assert_called_once_with(
        EndpointConfigName="some-endpoint-config"
    )


def test_create_inference_component(sagemaker_session):
    tags = [{"Key": "TagtestKey", "Value": "TagtestValue"}]
    sagemaker_session.sagemaker_client.describe_inference_component = Mock(
        return_value={"InferenceComponentStatus": "InService"}
    )
    model_name = "test_model"
    inference_component_name = "test_inference_component"
    endpoint_name = "test_endpoint"

    resources = RESOURCES.get_compute_resource_requirements()
    specification = {"ModelName": model_name, "ComputeResourceRequirements": resources}
    runtime_config = {"CopyCount": RESOURCES.copy_count}

    request = {
        "InferenceComponentName": inference_component_name,
        "EndpointName": endpoint_name,
        "VariantName": "AllTraffic",
        "Specification": specification,
        "RuntimeConfig": runtime_config,
        "Tags": tags,
    }

    sagemaker_session.create_inference_component(
        inference_component_name=inference_component_name,
        endpoint_name=endpoint_name,
        variant_name="AllTraffic",
        specification=specification,
        runtime_config=runtime_config,
        tags=tags,
    )

    sagemaker_session.sagemaker_client.create_inference_component.assert_called_with(**request)


def test_delete_inference_component(boto_session):
    sess = Session(boto_session)

    inference_component_name = "my_inference_component"

    sess.delete_inference_component(inference_component_name)

    boto_session.client().delete_inference_component.assert_called_with(
        InferenceComponentName=inference_component_name
    )


def test_describe_inference_component(sagemaker_session):
    inference_component_name = "sagemaker_inference_component_name"

    sagemaker_session.describe_inference_component(
        inference_component_name=inference_component_name
    )

    sagemaker_session.sagemaker_client.describe_inference_component.assert_called_with(
        InferenceComponentName=inference_component_name
    )


def test_list_and_paginate_inference_component_names_associated_with_endpoint(sagemaker_session):
    endpoint_name = "test-endpoint"

    sagemaker_session.list_inference_components = Mock()
    sagemaker_session.list_inference_components.side_effect = [
        {
            "InferenceComponents": [
                {
                    "CreationTime": 1234,
                    "EndpointArn": "endpointarn1",
                    "EndpointName": "endpointname1",
                    "InferenceComponentArn": "icarn1",
                    "InferenceComponentName": "icname1",
                    "InferenceComponentStatus": "icstatus1",
                    "LastModifiedTime": 5678,
                    "VariantName": "blah1",
                }
            ],
            "NextToken": "1234",
        },
        {
            "InferenceComponents": [
                {
                    "CreationTime": 12345,
                    "EndpointArn": "endpointarn2",
                    "EndpointName": "endpointname2",
                    "InferenceComponentArn": "icarn2",
                    "InferenceComponentName": "icname2",
                    "InferenceComponentStatus": "icstatus2",
                    "LastModifiedTime": 56789,
                    "VariantName": "blah2",
                }
            ]
        },
    ]
    assert [
        "icname1",
        "icname2",
    ] == sagemaker_session.list_and_paginate_inference_component_names_associated_with_endpoint(
        endpoint_name
    )

    sagemaker_session.list_inference_components.assert_has_calls(
        [
            call(endpoint_name_equals="test-endpoint", next_token=None),
            call(endpoint_name_equals="test-endpoint", next_token="1234"),
        ]
    )


def test_list_inference_components(sagemaker_session):
    endpoint_name = "test-endpoint"
    variant_name = "test-variant-name"

    sagemaker_session.list_inference_components(
        endpoint_name_equals=endpoint_name,
        variant_name_equals=variant_name,
        sort_by="Status",
        sort_order="Ascending",
        max_results="5",
        name_contains="model",
        status_equals="InService",
    )

    request = {
        "EndpointNameEquals": endpoint_name,
        "VariantNameEquals": variant_name,
        "SortBy": "Status",
        "SortOrder": "Ascending",
        "MaxResults": "5",
        "NameContains": "model",
        "StatusEquals": "InService",
    }

    sagemaker_session.sagemaker_client.list_inference_components.assert_called_with(**request)


def test_update_inference_component(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_inference_component = Mock(
        return_value={"InferenceComponentStatus": "InService"}
    )
    inference_component_name = "test_inference_component"

    model_name = "test_model"
    inference_component_name = "test_inference_component"

    resources = RESOURCES.get_compute_resource_requirements()
    specification = {"ModelName": model_name, "ComputeResourceRequirements": resources}
    runtime_config = {"CopyCount": RESOURCES.copy_count}

    request = {
        "InferenceComponentName": inference_component_name,
        "Specification": specification,
        "RuntimeConfig": runtime_config,
    }

    sagemaker_session.update_inference_component(
        inference_component_name=inference_component_name,
        specification=specification,
        runtime_config=runtime_config,
    )

    sagemaker_session.sagemaker_client.update_inference_component.assert_called_with(**request)


@patch("os.makedirs")
def test_download_data_with_only_directory(makedirs, sagemaker_session):
    sagemaker_session.s3_client = Mock()
    sagemaker_session.s3_client.list_objects_v2 = Mock(
        return_value={
            "Contents": [
                {
                    "Key": "foo/bar/",
                    "Size": 0,
                }
            ]
        }
    )
    sagemaker_session.download_data(path=".", bucket="foo-bucket")

    makedirs.assert_called_with("./foo/bar", exist_ok=True)
    sagemaker_session.s3_client.download_file.assert_not_called()


@patch("os.makedirs")
def test_download_data_with_only_file(makedirs, sagemaker_session):
    sagemaker_session.s3_client = Mock()
    sagemaker_session.s3_client.list_objects_v2 = Mock(
        return_value={
            "Contents": [
                {
                    "Key": "foo/bar/mode.tar.gz",
                    "Size": 100,
                }
            ]
        }
    )
    sagemaker_session.download_data(path=".", bucket="foo-bucket")

    makedirs.assert_called_with("./foo/bar", exist_ok=True)
    sagemaker_session.s3_client.download_file.assert_called_with(
        Bucket="foo-bucket",
        Key="foo/bar/mode.tar.gz",
        Filename="./foo/bar/mode.tar.gz",
        ExtraArgs=None,
    )


@patch("os.makedirs")
def test_download_data_with_file_and_directory(makedirs, sagemaker_session):
    sagemaker_session.s3_client = Mock()
    sagemaker_session.s3_client.list_objects_v2 = Mock(
        return_value={
            "Contents": [
                {
                    "Key": "foo/bar/",
                    "Size": 0,
                },
                {
                    "Key": "foo/bar/mode.tar.gz",
                    "Size": 100,
                },
            ]
        }
    )
    sagemaker_session.download_data(path=".", bucket="foo-bucket")

    makedirs.assert_called_with("./foo/bar", exist_ok=True)
    makedirs.assert_has_calls([call("./foo/bar", exist_ok=True), call("./foo/bar", exist_ok=True)])
    sagemaker_session.s3_client.download_file.assert_called_with(
        Bucket="foo-bucket",
        Key="foo/bar/mode.tar.gz",
        Filename="./foo/bar/mode.tar.gz",
        ExtraArgs=None,
    )
