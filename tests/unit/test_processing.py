# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
from mock import Mock, patch

from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.network import NetworkConfig

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session_mock.upload_data = Mock(
        name="upload_data", return_value="mocked_s3_uri_from_upload_data"
    )
    session_mock.download_data = Mock(name="download_data")
    return session_mock


def test_sklearn(sagemaker_session):
    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=ROLE,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    with patch("os.path.isfile", return_value=True):
        sklearn_processor.run(
            command=["python3"],
            code="/local/path/to/sklearn_transformer.py",
            inputs=[
                ProcessingInput(source="/local/path/to/my/dataset/census.csv", destination="/data/")
            ],
        )

    expected_args = {
        "inputs": [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/data/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {"Outputs": []},
        "job_name": sklearn_processor._current_job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3",
            "ContainerEntrypoint": ["python3", "/input/code/sklearn_transformer.py"],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
    }
    sagemaker_session.process.assert_called_with(**expected_args)


def test_sklearn_with_no_inputs(sagemaker_session):
    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=ROLE,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    with patch("os.path.isfile", return_value=True):
        sklearn_processor.run(command=["python3"], code="/local/path/to/sklearn_transformer.py")

    expected_args = {
        "inputs": [
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            }
        ],
        "output_config": {"Outputs": []},
        "job_name": sklearn_processor._current_job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3",
            "ContainerEntrypoint": ["python3", "/input/code/sklearn_transformer.py"],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
    }
    sagemaker_session.process.assert_called_with(**expected_args)


def test_sklearn_with_all_customizations(sagemaker_session):
    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=ROLE,
        instance_type="ml.m4.xlarge",
        py_version="py3",
        volume_size_in_gb=100,
        volume_kms_key=None,
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )

    with patch("os.path.isdir", return_value=True):
        sklearn_processor.run(
            command=["python3"],
            code="/local/path/to/code",
            script_name="sklearn_transformer.py",
            inputs=[
                ProcessingInput(
                    source="/local/path/to/my/sklearn_transformer.py",
                    destination="/container/path/",
                ),
                ProcessingInput(
                    source="s3://path/to/my/dataset/census.csv",
                    destination="/container/path/",
                    input_name="my_dataset",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="FullyReplicated",
                    s3_compression_type="None",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source="/container/path/",
                    destination="s3://uri/",
                    output_name="my_output",
                    s3_upload_mode="EndOfJob",
                )
            ],
            arguments=["--drop-columns", "'SelfEmployed'"],
            wait=True,
            logs=False,
            job_name="my_job_name",
        )

    expected_args = {
        "inputs": [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/container/path/",
                    "S3DataType": "S3Prefix",
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
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {
            "Outputs": [
                {
                    "OutputName": "my_output",
                    "S3Output": {
                        "S3Uri": "s3://uri/",
                        "LocalPath": "/container/path/",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ],
            "KmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/kms-key",
        },
        "job_name": sklearn_processor._current_job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 100,
            }
        },
        "stopping_condition": {"MaxRuntimeInSeconds": 3600},
        "app_specification": {
            "ImageUri": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3",
            "ContainerArguments": ["--drop-columns", "'SelfEmployed'"],
            "ContainerEntrypoint": ["python3", "/input/code/sklearn_transformer.py"],
        },
        "environment": {"my_env_variable": "my_env_variable_value"},
        "network_config": {
            "EnableInterContainerTrafficEncryption": True,
            "EnableNetworkIsolation": True,
            "VpcConfig": {
                "SecurityGroupIds": ["my_security_group_id"],
                "Subnets": ["my_subnet_id"],
            },
        },
        "role_arn": ROLE,
        "tags": [{"Key": "my-tag", "Value": "my-tag-value"}],
    }
    sagemaker_session.process.assert_called_with(**expected_args)


def test_byo_container_with_script_processor(sagemaker_session):
    script_processor = ScriptProcessor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    with patch("os.path.isfile", return_value=True):
        script_processor.run(
            command=["python3"],
            code="/local/path/to/sklearn_transformer.py",
            inputs=[
                ProcessingInput(source="/local/path/to/my/dataset/census.csv", destination="/data/")
            ],
        )

    expected_args = {
        "inputs": [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/data/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {"Outputs": []},
        "job_name": script_processor._current_job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": CUSTOM_IMAGE_URI,
            "ContainerEntrypoint": ["python3", "/input/code/sklearn_transformer.py"],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
    }
    sagemaker_session.process.assert_called_with(**expected_args)


def test_byo_container_with_custom_script(sagemaker_session):
    custom_processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        entrypoint="sklearn_transformer.py",
        sagemaker_session=sagemaker_session,
    )

    custom_processor.run(
        inputs=[
            ProcessingInput(source="/local/path/to/my/dataset/census.csv", destination="/data/")
        ],
        arguments=["CensusTract", "County"],
    )

    expected_args = {
        "inputs": [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/data/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            }
        ],
        "output_config": {"Outputs": []},
        "job_name": custom_processor._current_job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": CUSTOM_IMAGE_URI,
            "ContainerArguments": ["CensusTract", "County"],
            "ContainerEntrypoint": "sklearn_transformer.py",
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
    }
    sagemaker_session.process.assert_called_with(**expected_args)


def test_byo_container_with_baked_in_script(sagemaker_session):
    custom_processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    custom_processor.run(
        inputs=[
            ProcessingInput(source="/local/path/to/my/sklearn_transformer", destination="/code/")
        ],
        arguments=["CensusTract", "County"],
    )

    expected_args = {
        "inputs": [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/code/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            }
        ],
        "output_config": {"Outputs": []},
        "job_name": custom_processor._current_job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": CUSTOM_IMAGE_URI,
            "ContainerArguments": ["CensusTract", "County"],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
    }
    sagemaker_session.process.assert_called_with(**expected_args)
