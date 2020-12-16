# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from mock import Mock, patch, MagicMock

from sagemaker.dataset_definition.inputs import (
    S3Input,
    DatasetDefinition,
    AthenaDatasetDefinition,
    RedshiftDatasetDefinition,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
    ProcessingJob,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.network import NetworkConfig
from sagemaker.processing import FeatureStoreOutput

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
ECR_HOSTNAME = "ecr.us-west-2.amazonaws.com"
CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = MagicMock(
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
    session_mock.expand_role.return_value = ROLE
    session_mock.describe_processing_job = MagicMock(
        name="describe_processing_job", return_value=_get_describe_response_inputs_and_ouputs()
    )
    return session_mock


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_sklearn_processor_with_required_parameters(
    exists_mock, isfile_mock, botocore_resolver, sagemaker_session, sklearn_version
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = SKLearnProcessor(
        role=ROLE,
        instance_type="ml.m4.xlarge",
        framework_version=sklearn_version,
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name)

    sklearn_image_uri = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-py3"
    ).format(sklearn_version)
    expected_args["app_specification"]["ImageUri"] = sklearn_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_sklearn_with_all_parameters(
    exists_mock, isfile_mock, botocore_resolver, sklearn_version, sagemaker_session
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = SKLearnProcessor(
        role=ROLE,
        framework_version=sklearn_version,
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
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

    processor.run(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)
    sklearn_image_uri = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-py3"
    ).format(sklearn_version)
    expected_args["app_specification"]["ImageUri"] = sklearn_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=False)
def test_script_processor_errors_with_nonexistent_local_code(exists_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)
    with pytest.raises(ValueError):
        processor.run(code="/local/path/to/processing_code.py")


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=False)
def test_script_processor_errors_with_code_directory(exists_mock, isfile_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)
    with pytest.raises(ValueError):
        processor.run(code="/local/path/to/code")


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_errors_with_invalid_code_url_scheme(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    with pytest.raises(ValueError):
        processor.run(code="hdfs:///path/to/processing_code.py")


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_absolute_local_path(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name)

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_relative_local_path(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name)
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_relative_local_path_with_directories(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="path/to/processing_code.py")
    expected_args = _get_expected_args(processor._current_job_name)
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_file_code_url_scheme(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="file:///path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name)
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_s3_code_url(exists_mock, isfile_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="s3://bucket/path/to/processing_code.py")

    expected_args = _get_expected_args(
        processor._current_job_name, "s3://bucket/path/to/processing_code.py"
    )
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_with_one_input(exists_mock, isfile_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)
    processor.run(
        code="/local/path/to/processing_code.py",
        inputs=[
            ProcessingInput(source="/local/path/to/my/dataset/census.csv", destination="/data/")
        ],
    )

    expected_args = _get_expected_args(processor._current_job_name)
    expected_args["inputs"].insert(0, _get_data_input())

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_with_required_parameters(exists_mock, isfile_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name)
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_with_all_parameters(exists_mock, isfile_mock, sagemaker_session):
    processor = ScriptProcessor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        command=["python3"],
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
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

    processor.run(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)

    sagemaker_session.process.assert_called_with(**expected_args)
    assert "my_job_name" in processor._current_job_name


def test_processor_with_required_parameters(sagemaker_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    processor.run()

    expected_args = _get_expected_args(processor._current_job_name)
    del expected_args["app_specification"]["ContainerEntrypoint"]
    expected_args["inputs"] = []

    sagemaker_session.process.assert_called_with(**expected_args)


def test_processor_with_missing_network_config_parameters(sagemaker_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        network_config=NetworkConfig(enable_network_isolation=True),
    )

    processor.run()

    expected_args = _get_expected_args(processor._current_job_name)
    del expected_args["app_specification"]["ContainerEntrypoint"]
    expected_args["inputs"] = []
    expected_args["network_config"] = {"EnableNetworkIsolation": True}

    sagemaker_session.process.assert_called_with(**expected_args)


def test_processor_with_encryption_parameter_in_network_config(sagemaker_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        network_config=NetworkConfig(encrypt_inter_container_traffic=False),
    )

    processor.run()

    expected_args = _get_expected_args(processor._current_job_name)
    del expected_args["app_specification"]["ContainerEntrypoint"]
    expected_args["inputs"] = []
    expected_args["network_config"] = {
        "EnableNetworkIsolation": False,
        "EnableInterContainerTrafficEncryption": False,
    }

    sagemaker_session.process.assert_called_with(**expected_args)


def test_processor_with_all_parameters(sagemaker_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        entrypoint=["python3", "/opt/ml/processing/input/code/processing_code.py"],
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="processor_base_name",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
    )

    processor.run(
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)
    # Drop the "code" input from expected values.
    expected_args["inputs"] = expected_args["inputs"][:-1]

    sagemaker_session.process.assert_called_with(**expected_args)


def test_processing_job_from_processing_arn(sagemaker_session):
    processing_job = ProcessingJob.from_processing_arn(
        sagemaker_session=sagemaker_session,
        processing_job_arn="arn:aws:sagemaker:dummy-region:dummy-account-number:processing-job/dummy-job-name",
    )

    assert isinstance(processing_job, ProcessingJob)
    assert [
        processing_input._to_request_dict() for processing_input in processing_job.inputs
    ] == _get_describe_response_inputs_and_ouputs()["ProcessingInputs"]
    assert [
        processing_output._to_request_dict() for processing_output in processing_job.outputs
    ] == _get_describe_response_inputs_and_ouputs()["ProcessingOutputConfig"]["Outputs"]
    assert (
        processing_job.output_kms_key
        == _get_describe_response_inputs_and_ouputs()["ProcessingOutputConfig"]["KmsKeyId"]
    )


def test_extend_processing_args(sagemaker_session):
    inputs = []
    outputs = []

    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        network_config=NetworkConfig(encrypt_inter_container_traffic=False),
    )

    extended_inputs, extended_outputs = processor._extend_processing_args([], [])

    assert extended_inputs == inputs
    assert extended_outputs == outputs


def _get_script_processor(sagemaker_session):
    return ScriptProcessor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        command=["python3"],
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )


def _get_expected_args(job_name, code_s3_uri="mocked_s3_uri_from_upload_data"):
    return {
        "inputs": [
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": code_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            }
        ],
        "output_config": {"Outputs": []},
        "job_name": job_name,
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
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/processing_code.py"],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
        "experiment_config": None,
    }


def _get_data_input():
    data_input = {
        "InputName": "input-1",
        "AppManaged": False,
        "S3Input": {
            "S3Uri": "mocked_s3_uri_from_upload_data",
            "LocalPath": "/data/",
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
            "S3CompressionType": "None",
        },
    }
    return data_input


def _get_data_inputs_all_parameters():
    return [
        ProcessingInput(
            source="s3://path/to/my/dataset/census.csv",
            destination="/container/path/",
            input_name="my_dataset",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
            s3_data_distribution_type="FullyReplicated",
            s3_compression_type="None",
        ),
        ProcessingInput(
            input_name="s3_input",
            s3_input=S3Input(
                s3_uri="s3://path/to/my/dataset/census.csv",
                local_path="/container/path/",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            ),
        ),
        ProcessingInput(
            input_name="redshift_dataset_definition",
            app_managed=True,
            dataset_definition=DatasetDefinition(
                local_path="/opt/ml/processing/input/dd",
                data_distribution_type="FullyReplicated",
                input_mode="File",
                redshift_dataset_definition=RedshiftDatasetDefinition(
                    cluster_id="cluster_id",
                    database="database",
                    db_user="db_user",
                    query_string="query_string",
                    cluster_role_arn="cluster_role_arn",
                    output_s3_uri="output_s3_uri",
                    kms_key_id="kms_key_id",
                    output_format="CSV",
                    output_compression="SNAPPY",
                ),
            ),
        ),
        ProcessingInput(
            input_name="athena_dataset_definition",
            app_managed=True,
            dataset_definition=DatasetDefinition(
                local_path="/opt/ml/processing/input/dd",
                data_distribution_type="FullyReplicated",
                input_mode="File",
                athena_dataset_definition=AthenaDatasetDefinition(
                    catalog="catalog",
                    database="database",
                    work_group="workgroup",
                    query_string="query_string",
                    output_s3_uri="output_s3_uri",
                    kms_key_id="kms_key_id",
                    output_format="AVRO",
                    output_compression="ZLIB",
                ),
            ),
        ),
    ]


def _get_data_outputs_all_parameters():
    return [
        ProcessingOutput(
            source="/container/path/",
            destination="s3://uri/",
            output_name="my_output",
            s3_upload_mode="EndOfJob",
        ),
        ProcessingOutput(
            output_name="feature_store_output",
            app_managed=True,
            feature_store_output=FeatureStoreOutput(feature_group_name="FeatureGroupName"),
        ),
    ]


def _get_expected_args_all_parameters(job_name):
    return {
        "inputs": [
            {
                "InputName": "my_dataset",
                "AppManaged": False,
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
                "InputName": "s3_input",
                "AppManaged": False,
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
                "InputName": "redshift_dataset_definition",
                "AppManaged": True,
                "DatasetDefinition": {
                    "DataDistributionType": "FullyReplicated",
                    "InputMode": "File",
                    "LocalPath": "/opt/ml/processing/input/dd",
                    "RedshiftDatasetDefinition": {
                        "ClusterId": "cluster_id",
                        "Database": "database",
                        "DbUser": "db_user",
                        "QueryString": "query_string",
                        "ClusterRoleArn": "cluster_role_arn",
                        "OutputS3Uri": "output_s3_uri",
                        "KmsKeyId": "kms_key_id",
                        "OutputFormat": "CSV",
                        "OutputCompression": "SNAPPY",
                    },
                },
            },
            {
                "InputName": "athena_dataset_definition",
                "AppManaged": True,
                "DatasetDefinition": {
                    "DataDistributionType": "FullyReplicated",
                    "InputMode": "File",
                    "LocalPath": "/opt/ml/processing/input/dd",
                    "AthenaDatasetDefinition": {
                        "Catalog": "catalog",
                        "Database": "database",
                        "QueryString": "query_string",
                        "OutputS3Uri": "output_s3_uri",
                        "WorkGroup": "workgroup",
                        "KmsKeyId": "kms_key_id",
                        "OutputFormat": "AVRO",
                        "OutputCompression": "ZLIB",
                    },
                },
            },
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": "mocked_s3_uri_from_upload_data",
                    "LocalPath": "/opt/ml/processing/input/code",
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
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": "s3://uri/",
                        "LocalPath": "/container/path/",
                        "S3UploadMode": "EndOfJob",
                    },
                },
                {
                    "OutputName": "feature_store_output",
                    "AppManaged": True,
                    "FeatureStoreOutput": {"FeatureGroupName": "FeatureGroupName"},
                },
            ],
            "KmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        },
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 100,
                "VolumeKmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
            }
        },
        "stopping_condition": {"MaxRuntimeInSeconds": 3600},
        "app_specification": {
            "ImageUri": "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri",
            "ContainerArguments": ["--drop-columns", "'SelfEmployed'"],
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/processing_code.py"],
        },
        "environment": {"my_env_variable": "my_env_variable_value"},
        "network_config": {
            "EnableNetworkIsolation": True,
            "EnableInterContainerTrafficEncryption": True,
            "VpcConfig": {
                "SecurityGroupIds": ["my_security_group_id"],
                "Subnets": ["my_subnet_id"],
            },
        },
        "role_arn": ROLE,
        "tags": [{"Key": "my-tag", "Value": "my-tag-value"}],
        "experiment_config": {"ExperimentName": "AnExperiment"},
    }


def _get_describe_response_inputs_and_ouputs():
    return {
        "ProcessingInputs": _get_expected_args_all_parameters(None)["inputs"],
        "ProcessingOutputConfig": _get_expected_args_all_parameters(None)["output_config"],
    }
