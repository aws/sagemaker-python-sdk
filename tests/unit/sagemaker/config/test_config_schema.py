# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import
from jsonschema import validate, exceptions
import pytest
from sagemaker.config.config_schema import SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA


def _validate_config(base_config_with_schema, sagemaker_config):
    config = base_config_with_schema
    config["SageMaker"] = sagemaker_config
    validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_valid_schema_version(base_config_with_schema):
    validate(base_config_with_schema, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_invalid_schema_version():
    config = {"SchemaVersion": "99.0"}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)

    # Also test missing schema version.
    config = {}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_valid_config_with_all_the_features(
    base_config_with_schema, valid_config_with_all_the_scopes
):
    _validate_config(base_config_with_schema, valid_config_with_all_the_scopes)


def test_feature_group_schema(base_config_with_schema, valid_feature_group_config):
    _validate_config(base_config_with_schema, {"FeatureGroup": valid_feature_group_config})


def test_valid_edge_packaging_job_schema(base_config_with_schema, valid_edge_packaging_config):
    _validate_config(base_config_with_schema, {"EdgePackagingJob": valid_edge_packaging_config})


def test_valid_training_job_schema(base_config_with_schema, valid_training_job_config):
    _validate_config(base_config_with_schema, {"TrainingJob": valid_training_job_config})


def test_valid_processing_job_schema(base_config_with_schema, valid_processing_job_config):
    _validate_config(base_config_with_schema, {"ProcessingJob": valid_processing_job_config})


def test_valid_model_package_schema(base_config_with_schema, valid_model_package_config):
    _validate_config(base_config_with_schema, {"ModelPackage": valid_model_package_config})


def test_valid_model_schema(base_config_with_schema, valid_model_config):
    _validate_config(base_config_with_schema, {"Model": valid_model_config})


def test_valid_pipeline_schema(base_config_with_schema, valid_pipeline_config):
    _validate_config(base_config_with_schema, {"Pipeline": valid_pipeline_config})


def test_valid_compilation_job_schema(base_config_with_schema, valid_compilation_job_config):
    _validate_config(base_config_with_schema, {"CompilationJob": valid_compilation_job_config})


def test_valid_transform_job_schema(base_config_with_schema, valid_transform_job_config):
    _validate_config(base_config_with_schema, {"TransformJob": valid_transform_job_config})


def test_valid_automl_schema(base_config_with_schema, valid_automl_config):
    _validate_config(base_config_with_schema, {"AutoMLJob": valid_automl_config})


def test_valid_endpoint_config_schema(base_config_with_schema, valid_endpointconfig_config):
    _validate_config(base_config_with_schema, {"EndpointConfig": valid_endpointconfig_config})


def test_valid_monitoring_schedule_schema(
    base_config_with_schema, valid_monitoring_schedule_config
):
    _validate_config(
        base_config_with_schema, {"MonitoringSchedule": valid_monitoring_schedule_config}
    )


def test_valid_remote_function_schema(base_config_with_schema, valid_remote_function_config):
    _validate_config(
        base_config_with_schema, {"PythonSDK": {"Modules": valid_remote_function_config}}
    )


def test_tags_with_invalid_schema(base_config_with_schema, valid_edge_packaging_config):
    edge_packaging_config = valid_edge_packaging_config.copy()
    edge_packaging_config["Tags"] = [{"Key": "somekey"}]
    config = base_config_with_schema
    config["SageMaker"] = {"EdgePackagingJob": edge_packaging_config}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)
    edge_packaging_config["Tags"] = [{"Value": "somekey"}]
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_tags_with_valid_schema(base_config_with_schema, valid_edge_packaging_config):
    edge_packaging_config = valid_edge_packaging_config.copy()
    edge_packaging_config["Tags"] = [{"Key": "somekey", "Value": "somevalue"}]
    config = base_config_with_schema
    config["SageMaker"] = {"EdgePackagingJob": edge_packaging_config}
    validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_invalid_training_job_schema(base_config_with_schema, valid_iam_role_arn, valid_vpc_config):
    # Changing key names
    training_job_config = {
        "EnableNetworkIsolation1": True,
        "OutputDataConfig1": {"KmsKeyId": "somekmskey"},
        "ResourceConfig1": {"VolumeKmsKeyId": "somevolumekmskey"},
        "RoleArn1": valid_iam_role_arn,
        "VpcConfig1": valid_vpc_config,
    }
    config = base_config_with_schema
    config["SageMaker"] = {"TrainingJob": training_job_config}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_invalid_edge_packaging_job_schema(base_config_with_schema, valid_iam_role_arn):
    # Using invalid keys
    edge_packaging_job_config = {
        "OutputConfig1": {"KmsKeyId": "somekeyid"},
        "RoleArn1": valid_iam_role_arn,
    }
    config = base_config_with_schema
    config["SageMaker"] = {"EdgePackagingJob": edge_packaging_job_config}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_invalid_feature_group_schema(base_config_with_schema):
    s3_storage_config = {"KmsKeyId": "somekmskeyid"}
    security_storage_config = {"KmsKeyId": "someotherkmskeyid"}
    # Online store doesn't have S3StorageConfig and similarly
    # Offline store doesn't have SecurityConfig
    online_store_config = {"S3StorageConfig": security_storage_config}
    offline_store_config = {"SecurityConfig": s3_storage_config}
    feature_group_config = {
        "OnlineStoreConfig": online_store_config,
        "OfflineStoreConfig": offline_store_config,
    }
    config = base_config_with_schema
    config["SageMaker"] = {"FeatureGroup": feature_group_config}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_valid_custom_parameters_schema(base_config_with_schema):
    config = base_config_with_schema
    config["CustomParameters"] = {
        "custom_key": "custom_value",
        "CustomKey": "CustomValue",
        "custom key": "custom value",
        "custom-key": "custom-value",
        "custom0123 key0123": "custom0123 value0123",
    }
    validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_invalid_custom_parameters_schema(base_config_with_schema):
    config = base_config_with_schema

    config["CustomParameters"] = {"^&": "custom_value"}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)

    config["CustomParameters"] = {"custom_key": 476}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)

    config["CustomParameters"] = {"custom_key": {"custom_key": "custom_value"}}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)


def test_invalid_s3uri_schema(base_config_with_schema):
    config = base_config_with_schema

    config["SageMaker"] = {"PythonSDK": {"Modules": {"RemoteFunction": {"S3RootUri": "bad_regex"}}}}
    with pytest.raises(exceptions.ValidationError):
        validate(config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)

@pytest.mark.parametrize(
    "bucket_name",
    [
        "docexamplebucket1",
        "log-delivery-march-2020",
        "my-hosted-content",
        "docexamplewebsite.com",
        "www.docexamplewebsite.com",
        "my.example.s3.bucket",
    ],
)
def test_session_s3_bucket_schema(base_config_with_schema, bucket_name):
    config = {"PythonSDK": {"Modules": {"Session": {"S3Bucket": bucket_name}}}}
    _validate_config(base_config_with_schema, config)


@pytest.mark.parametrize(
    "invalid_bucket_name",
    [
        "ab",
        "this-is-sixty-four-characters-total-which-is-one-above-the-limit",
        "UPPERCASE-LETTERS",
        "special_characters",
        "special-characters@",
        ".dot-at-the-beginning",
        "-dash-at-the-beginning",
        "dot-at-the-end.",
        "dash-at-the-end-",
    ],
)
def test_invalid_session_s3_bucket_schema(base_config_with_schema, invalid_bucket_name):
    with pytest.raises(exceptions.ValidationError):
        test_session_s3_bucket_schema(base_config_with_schema, invalid_bucket_name)
