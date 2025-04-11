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
import pytest
from mock import MagicMock


@pytest.fixture()
def base_config_with_schema():
    return {"SchemaVersion": "1.0"}


@pytest.fixture()
def valid_vpc_config():
    return {"SecurityGroupIds": ["sg123"], "Subnets": ["subnet-1234"]}


@pytest.fixture()
def valid_iam_role_arn():
    return "arn:aws:iam::555555555555:role/IMRole"


@pytest.fixture()
def valid_tags():
    return [{"Key": "tag1", "Value": "tagValue1"}]


@pytest.fixture()
def valid_session_config():
    return {
        "DefaultS3Bucket": "sagemaker-python-sdk-test-bucket",
        "DefaultS3ObjectKeyPrefix": "test-prefix",
    }


@pytest.fixture()
def valid_estimator_config():
    return {
        "DebugHookConfig": False,
    }


@pytest.fixture()
def valid_environment_config():
    return {
        "var1": "value1",
        "var2": "value2",
    }


@pytest.fixture()
def valid_containers_config(valid_environment_config):
    return [{"Environment": valid_environment_config}]


@pytest.fixture()
def valid_feature_group_config(valid_iam_role_arn):
    security_storage_config = {"KmsKeyId": "kmskeyid1"}
    s3_storage_config = {"KmsKeyId": "kmskeyid2"}
    online_store_config = {"SecurityConfig": security_storage_config}
    offline_store_config = {"S3StorageConfig": s3_storage_config}
    return {
        "OnlineStoreConfig": online_store_config,
        "OfflineStoreConfig": offline_store_config,
        "RoleArn": valid_iam_role_arn,
    }


@pytest.fixture()
def valid_edge_packaging_config(valid_iam_role_arn):
    return {
        "OutputConfig": {"KmsKeyId": "kmskeyid1"},
        "RoleArn": valid_iam_role_arn,
        "ResourceKey": "kmskeyid1",
    }


@pytest.fixture()
def valid_model_config(
    valid_iam_role_arn, valid_vpc_config, valid_environment_config, valid_containers_config
):
    return {
        "Containers": valid_containers_config,
        "EnableNetworkIsolation": True,
        "ExecutionRoleArn": valid_iam_role_arn,
        "PrimaryContainer": {"Environment": valid_environment_config},
        "VpcConfig": valid_vpc_config,
    }


@pytest.fixture()
def valid_model_package_config(
    valid_iam_role_arn, valid_environment_config, valid_containers_config
):
    inference_specification = {
        "Containers": valid_containers_config,
    }
    transform_job_definition = {
        "Environment": valid_environment_config,
        "TransformOutput": {"KmsKeyId": "kmskeyid1"},
        "TransformResources": {"VolumeKmsKeyId": "volumekmskeyid1"},
    }
    validation_specification = {
        "ValidationProfiles": [{"TransformJobDefinition": transform_job_definition}],
        "ValidationRole": valid_iam_role_arn,
    }
    return {
        "InferenceSpecification": inference_specification,
        "ValidationSpecification": validation_specification,
    }


@pytest.fixture()
def valid_processing_job_config(valid_iam_role_arn, valid_vpc_config, valid_environment_config):
    network_config = {"EnableNetworkIsolation": True, "VpcConfig": valid_vpc_config}
    dataset_definition = {
        "AthenaDatasetDefinition": {"KmsKeyId": "kmskeyid1"},
        "RedshiftDatasetDefinition": {
            "KmsKeyId": "kmskeyid2",
            "ClusterRoleArn": valid_iam_role_arn,
        },
    }
    return {
        "Environment": valid_environment_config,
        "NetworkConfig": network_config,
        "ProcessingInputs": [{"DatasetDefinition": dataset_definition}],
        "ProcessingOutputConfig": {"KmsKeyId": "kmskeyid3"},
        "ProcessingResources": {"ClusterConfig": {"VolumeKmsKeyId": "volumekmskeyid1"}},
        "RoleArn": valid_iam_role_arn,
    }


@pytest.fixture()
def valid_training_job_config(valid_iam_role_arn, valid_vpc_config, valid_environment_config):
    return {
        "EnableNetworkIsolation": True,
        "Environment": valid_environment_config,
        "OutputDataConfig": {"KmsKeyId": "kmskeyid1"},
        "ResourceConfig": {"VolumeKmsKeyId": "volumekmskeyid1"},
        "ProfilerConfig": {"DisableProfiler": False},
        "RoleArn": valid_iam_role_arn,
        "VpcConfig": valid_vpc_config,
    }


@pytest.fixture()
def valid_pipeline_config(valid_iam_role_arn):
    return {"RoleArn": valid_iam_role_arn}


@pytest.fixture()
def valid_compilation_job_config(valid_iam_role_arn, valid_vpc_config):
    return {
        "OutputConfig": {"KmsKeyId": "kmskeyid1"},
        "RoleArn": valid_iam_role_arn,
        "VpcConfig": valid_vpc_config,
    }


@pytest.fixture()
def valid_transform_job_config(valid_environment_config):
    return {
        "DataCaptureConfig": {"KmsKeyId": "kmskeyid1"},
        "Environment": valid_environment_config,
        "TransformOutput": {"KmsKeyId": "kmskeyid2"},
        "TransformResources": {"VolumeKmsKeyId": "volumekmskeyid1"},
    }


@pytest.fixture()
def valid_automl_config(valid_iam_role_arn, valid_vpc_config):
    return {
        "AutoMLJobConfig": {
            "SecurityConfig": {"VolumeKmsKeyId": "volumekmskeyid1", "VpcConfig": valid_vpc_config}
        },
        "OutputDataConfig": {"KmsKeyId": "kmskeyid1"},
        "RoleArn": valid_iam_role_arn,
    }


@pytest.fixture()
def valid_endpointconfig_config():
    return {
        "AsyncInferenceConfig": {"OutputConfig": {"KmsKeyId": "kmskeyid1"}},
        "DataCaptureConfig": {"KmsKeyId": "kmskeyid2"},
        "KmsKeyId": "kmskeyid3",
        "ProductionVariants": [{"CoreDumpConfig": {"KmsKeyId": "kmskeyid4"}}],
    }


@pytest.fixture()
def valid_endpoint_config(valid_tags):
    return {"Tags": valid_tags}


@pytest.fixture()
def valid_monitoring_schedule_config(
    valid_iam_role_arn, valid_vpc_config, valid_environment_config
):
    network_config = {"EnableNetworkIsolation": True, "VpcConfig": valid_vpc_config}
    return {
        "MonitoringScheduleConfig": {
            "MonitoringJobDefinition": {
                "Environment": valid_environment_config,
                "MonitoringOutputConfig": {"KmsKeyId": "kmskeyid1"},
                "MonitoringResources": {"ClusterConfig": {"VolumeKmsKeyId": "volumekmskeyid1"}},
                "NetworkConfig": network_config,
                "RoleArn": valid_iam_role_arn,
            }
        }
    }


@pytest.fixture()
def valid_remote_function_config(valid_iam_role_arn, valid_tags, valid_vpc_config):
    return {
        "Dependencies": "./requirements.txt",
        "EnvironmentVariables": {"var1": "value1", "var2": "value2"},
        "ImageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/myimage:latest",
        "IncludeLocalWorkDir": True,
        "InstanceType": "ml.m5.xlarge",
        "JobCondaEnvironment": "some_conda_env",
        "RoleArn": valid_iam_role_arn,
        "S3KmsKeyId": "kmskeyid1",
        "S3RootUri": "s3://my-bucket/key",
        "Tags": valid_tags,
        "VolumeKmsKeyId": "kmskeyid2",
        "VpcConfig": valid_vpc_config,
    }


@pytest.fixture()
def valid_config_with_all_the_scopes(
    valid_session_config,
    valid_feature_group_config,
    valid_monitoring_schedule_config,
    valid_endpoint_config,
    valid_endpointconfig_config,
    valid_automl_config,
    valid_transform_job_config,
    valid_compilation_job_config,
    valid_pipeline_config,
    valid_model_config,
    valid_model_package_config,
    valid_processing_job_config,
    valid_training_job_config,
    valid_edge_packaging_config,
    valid_remote_function_config,
    valid_estimator_config,
):
    return {
        "PythonSDK": {
            "Modules": {
                "Estimator": valid_estimator_config,
                "RemoteFunction": valid_remote_function_config,
                "Session": valid_session_config,
            }
        },
        "FeatureGroup": valid_feature_group_config,
        "MonitoringSchedule": valid_monitoring_schedule_config,
        "Endpoint": valid_endpoint_config,
        "EndpointConfig": valid_endpointconfig_config,
        "AutoMLJob": valid_automl_config,
        "TransformJob": valid_transform_job_config,
        "CompilationJob": valid_compilation_job_config,
        "Pipeline": valid_pipeline_config,
        "Model": valid_model_config,
        "ModelPackage": valid_model_package_config,
        "ProcessingJob": valid_processing_job_config,
        "TrainingJob": valid_training_job_config,
        "EdgePackagingJob": valid_edge_packaging_config,
    }


@pytest.fixture()
def s3_resource_mock():
    return MagicMock(name="s3")


@pytest.fixture()
def get_data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data", "config")


@pytest.fixture()
def base_local_mode_config():
    return {
        "local": {
            "local_code": True,
            "region_name": "",
            "serving_port": 8080,
            "container_config": {"shm_size": "128M"},
        }
    }
