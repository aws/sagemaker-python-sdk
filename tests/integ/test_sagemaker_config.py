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
import pathlib
import tempfile

import pytest
import yaml
from botocore.config import Config

from sagemaker import (
    PipelineModel,
    image_uris,
    Model,
    Predictor,
    Session,
)
from sagemaker.config import load_sagemaker_config
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.s3 import S3Uploader
from sagemaker.sparkml import SparkMLModel
from sagemaker.utils import sagemaker_timestamp

from tests.integ import DATA_DIR
from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ.test_inference_pipeline import SCHEMA
from tests.integ.timeout import timeout_and_delete_endpoint_by_name

S3_KEY_PREFIX = "integ-test-sagemaker_config"
ENDPOINT_CONFIG_TAGS = [
    {"Key": "SagemakerConfigUsed", "Value": "Yes"},
    {"Key": "ConfigOperation", "Value": "EndpointConfig"},
]
MODEL_TAGS = [
    {"Key": "SagemakerConfigUsed", "Value": "Yes"},
    {"Key": "ConfigOperation", "Value": "Model"},
]
CONFIG_DATA_DIR = os.path.join(DATA_DIR, "config")


@pytest.fixture(scope="session")
def role_arn(sagemaker_session):
    iam_client = sagemaker_session.boto_session.client("iam")
    return iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]


@pytest.fixture(scope="session")
def kms_key_arn(sagemaker_session):
    return get_or_create_kms_key(sagemaker_session=sagemaker_session)


@pytest.fixture()
def expected_merged_config():
    expected_merged_config_file_path = os.path.join(
        CONFIG_DATA_DIR, "expected_output_config_after_merge.yaml"
    )
    with open(expected_merged_config_file_path, "r") as f:
        return yaml.safe_load(f.read())


@pytest.fixture(scope="module")
def s3_uri_prefix(sagemaker_session):
    # Note: not using unique_name_from_base() here because the config contents are expected to
    # change very rarely (if ever), so rather than writing new files and deleting them every time
    # we can just use the same S3 paths
    s3_uri_prefix = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        S3_KEY_PREFIX,
    )
    return s3_uri_prefix


@pytest.fixture(scope="session")
def sagemaker_session_with_dynamically_generated_sagemaker_config(
    role_arn,
    kms_key_arn,
    sagemaker_client_config,
    sagemaker_runtime_config,
    boto_session,
    sagemaker_metrics_config,
):
    # This config needs to be dynamically generated so it can include the specific infra parameters
    # created/reused for the Integ tests
    config_as_dict = {
        "SchemaVersion": "1.0",
        "SageMaker": {
            "PythonSDK": {
                "Modules": {
                    "Session": {
                        "DefaultS3ObjectKeyPrefix": S3_KEY_PREFIX,
                        # S3Bucket is omitted for now, because the tests support one S3 bucket at
                        # the moment and it would be hard to validate injection of this parameter
                        # if we use the same bucket that the rest of the tests are.
                    },
                },
            },
            "EndpointConfig": {
                "AsyncInferenceConfig": {"OutputConfig": {"KmsKeyId": kms_key_arn}},
                "DataCaptureConfig": {"KmsKeyId": kms_key_arn},
                "KmsKeyId": kms_key_arn,
                "Tags": ENDPOINT_CONFIG_TAGS,
            },
            "Model": {
                "EnableNetworkIsolation": True,
                "ExecutionRoleArn": role_arn,
                "Tags": MODEL_TAGS,
                # VpcConfig is omitted for now, more info inside test
                # test_sagemaker_config_cross_context_injection
            },
        },
    }

    dynamic_sagemaker_config_yaml_path = os.path.join(
        tempfile.gettempdir(), "dynamic_sagemaker_config.yaml"
    )

    # write to yaml file, and avoid references and anchors
    yaml.Dumper.ignore_aliases = lambda *args: True
    with open(pathlib.Path(dynamic_sagemaker_config_yaml_path), "w") as f:
        yaml.dump(config_as_dict, f, sort_keys=False, default_flow_style=False)

    # other Session inputs (same as sagemaker_session fixture)
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )
    runtime_client = (
        boto_session.client("sagemaker-runtime", **sagemaker_runtime_config)
        if sagemaker_runtime_config
        else None
    )
    metrics_client = (
        boto_session.client("sagemaker-metrics", **sagemaker_metrics_config)
        if sagemaker_metrics_config
        else None
    )

    session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        sagemaker_metrics_client=metrics_client,
        sagemaker_config=load_sagemaker_config([dynamic_sagemaker_config_yaml_path]),
    )

    return session


def test_config_download_from_s3_and_merge(
    sagemaker_session,
    kms_key_arn,
    s3_uri_prefix,
    expected_merged_config,
):
    config_file_1_local_path = os.path.join(CONFIG_DATA_DIR, "sample_config_for_merge.yaml")
    config_file_2_local_path = os.path.join(
        CONFIG_DATA_DIR, "sample_additional_config_for_merge.yaml"
    )

    with open(config_file_1_local_path, "r") as f:
        config_file_1_as_yaml = f.read()
    s3_uri_config_1 = os.path.join(s3_uri_prefix, "config_1.yaml")

    # Upload S3 files in case they dont already exist
    S3Uploader.upload_string_as_file_body(
        body=config_file_1_as_yaml,
        desired_s3_uri=s3_uri_config_1,
        kms_key=kms_key_arn,
        sagemaker_session=sagemaker_session,
    )

    # Set env variable so load_sagemaker_config can construct an S3 resource in the right region
    previous_env_value = os.getenv("AWS_DEFAULT_REGION")
    os.environ["AWS_DEFAULT_REGION"] = sagemaker_session.boto_session.region_name

    # The thing being tested.
    sagemaker_config = load_sagemaker_config(
        additional_config_paths=[s3_uri_config_1, config_file_2_local_path]
    )

    # Reset the env variable to what it was before (if it was set before)
    os.unsetenv("AWS_DEFAULT_REGION")
    if previous_env_value is not None:
        os.environ["AWS_DEFAULT_REGION"] = previous_env_value

    assert sagemaker_config == expected_merged_config


@pytest.mark.slow_test
def test_sagemaker_config_cross_context_injection(
    sagemaker_session_with_dynamically_generated_sagemaker_config,
    role_arn,
    kms_key_arn,
    s3_uri_prefix,
    cpu_instance_type,
    alternative_cpu_instance_type,
):
    # This tests injection from the sagemaker_config, specifically for one scenario where a method
    # call (deploy of PipelineModel) leads to injections from separate
    # Model, EndpointConfig, and Endpoint configs.

    sagemaker_session = sagemaker_session_with_dynamically_generated_sagemaker_config
    name = "test-sm-config-pipeline-deploy-{}".format(sagemaker_timestamp())
    test_tags = [
        {
            "Key": "Test",
            "Value": "test_sagemaker_config_cross_context_injection",
        },
    ]
    data_capture_s3_uri = os.path.join(s3_uri_prefix, "model-monitor", "data-capture")

    sparkml_data_path = os.path.join(DATA_DIR, "sparkml_model")
    xgboost_data_path = os.path.join(DATA_DIR, "xgboost_model")
    sparkml_model_data = sagemaker_session.upload_data(
        path=os.path.join(sparkml_data_path, "mleap_model.tar.gz"),
        key_prefix="sparkml/model",
    )
    xgb_model_data = sagemaker_session.upload_data(
        path=os.path.join(xgboost_data_path, "xgb_model.tar.gz"),
        key_prefix="xgboost/model",
    )

    with timeout_and_delete_endpoint_by_name(name, sagemaker_session):

        # Create classes
        sparkml_model = SparkMLModel(
            model_data=sparkml_model_data,
            env={"SAGEMAKER_SPARKML_SCHEMA": SCHEMA},
            sagemaker_session=sagemaker_session,
        )
        xgb_image = image_uris.retrieve(
            "xgboost", sagemaker_session.boto_region_name, version="1", image_scope="inference"
        )
        xgb_model = Model(
            model_data=xgb_model_data,
            image_uri=xgb_image,
            sagemaker_session=sagemaker_session,
        )
        pipeline_model = PipelineModel(
            models=[sparkml_model, xgb_model],
            predictor_cls=Predictor,
            sagemaker_session=sagemaker_session,
            name=name,
        )

        # Basic check before any API calls that config parameters were injected. Not included:
        # - VpcConfig: The VPC created by the test suite today (via get_or_create_vpc_resources)
        #              creates two subnets in the same AZ. However, CreateEndpoint fails if it
        #              does not have at least two AZs. TODO: Can explore either creating a new
        #              VPC or modifying the existing one, so that it can be included in the
        #              config too
        # - Tags: By design. These are injected before the API call, not inside the Model classes
        assert [
            sparkml_model.role,
            xgb_model.role,
            pipeline_model.role,
            sparkml_model.enable_network_isolation(),
            xgb_model.enable_network_isolation(),
            pipeline_model.enable_network_isolation,  # This is not a function in PipelineModel
            sagemaker_session.default_bucket_prefix,
        ] == [role_arn, role_arn, role_arn, True, True, True, S3_KEY_PREFIX]

        # First mutating API call where sagemaker_config values should be injected in
        predictor = pipeline_model.deploy(
            1,
            alternative_cpu_instance_type,
            endpoint_name=name,
            data_capture_config=DataCaptureConfig(
                True,
                sagemaker_session=sagemaker_session,
            ),
            tags=test_tags,
        )
        endpoint_1 = sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=name)

        # Second mutating API call where sagemaker_config values should be injected in
        predictor.update_endpoint(initial_instance_count=1, instance_type=cpu_instance_type)
        endpoint_2 = sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=name)

        # Call remaining describe APIs to fetch info that we will validate against
        model = sagemaker_session.sagemaker_client.describe_model(ModelName=name)
        endpoint_config_1_name = endpoint_1["EndpointConfigName"]
        endpoint_config_2_name = endpoint_2["EndpointConfigName"]
        endpoint_config_1 = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_1_name
        )
        endpoint_config_2 = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_2_name
        )
        model_tags = sagemaker_session.sagemaker_client.list_tags(ResourceArn=model["ModelArn"])
        endpoint_1_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=endpoint_1["EndpointArn"]
        )
        endpoint_2_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=endpoint_2["EndpointArn"]
        )
        endpoint_config_1_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=endpoint_config_1["EndpointConfigArn"]
        )
        endpoint_config_2_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=endpoint_config_2["EndpointConfigArn"]
        )

        # Remove select key-values from the Describe API outputs that we do not need to compare
        # (things that may keep changing over time, and ARNs.)
        # Still leaving in more than just the sagemaker_config injected fields so we can verify that
        # the injection has not overwritten anything it shouldn't.
        for key in ["Containers", "CreationTime", "ModelArn", "ResponseMetadata"]:
            model.pop(key)

        for key in ["EndpointArn", "CreationTime", "LastModifiedTime", "ResponseMetadata"]:
            endpoint_1.pop(key)
            endpoint_2.pop(key)
        del endpoint_1["ProductionVariants"][0]["DeployedImages"]
        del endpoint_2["ProductionVariants"][0]["DeployedImages"]

        for key in ["EndpointConfigArn", "CreationTime", "ResponseMetadata"]:
            endpoint_config_1.pop(key)
            endpoint_config_2.pop(key)

        for key in ["ResponseMetadata"]:
            model_tags.pop(key)
            endpoint_1_tags.pop(key)
            endpoint_2_tags.pop(key)
            endpoint_config_1_tags.pop(key)
            endpoint_config_2_tags.pop(key)

        # Expected parameters for these objects
        expected_model = {
            "ModelName": name,
            "InferenceExecutionConfig": {"Mode": "Serial"},
            "ExecutionRoleArn": role_arn,  # from sagemaker_config
            "EnableNetworkIsolation": True,  # from sagemaker_config
            "DeploymentRecommendation": {"RecommendationStatus": "NOT_APPLICABLE"},
        }

        expected_endpoint_1 = {
            "EndpointName": name,
            "EndpointConfigName": endpoint_config_1_name,
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "CurrentWeight": 1.0,
                    "DesiredWeight": 1.0,
                    "CurrentInstanceCount": 1,
                    "DesiredInstanceCount": 1,
                }
            ],
            "DataCaptureConfig": {
                "EnableCapture": True,
                "CaptureStatus": "Started",
                "CurrentSamplingPercentage": 20,
                "DestinationS3Uri": data_capture_s3_uri,
                "KmsKeyId": kms_key_arn,  # from sagemaker_config
            },
            "EndpointStatus": "InService",
        }

        expected_endpoint_2 = {
            **expected_endpoint_1,
            "EndpointConfigName": endpoint_config_2_name,
        }

        expected_endpoint_config_1 = {
            "EndpointConfigName": endpoint_config_1_name,
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": name,
                    "InitialInstanceCount": 1,
                    "InstanceType": alternative_cpu_instance_type,
                    "InitialVariantWeight": 1.0,
                    "VolumeSizeInGB": 4,
                }
            ],
            "DataCaptureConfig": {
                "EnableCapture": True,
                "InitialSamplingPercentage": 20,
                "DestinationS3Uri": data_capture_s3_uri,
                "KmsKeyId": kms_key_arn,  # from sagemaker_config
                "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
                "CaptureContentTypeHeader": {
                    "CsvContentTypes": ["text/csv"],
                    "JsonContentTypes": ["application/json"],
                },
            },
            "KmsKeyId": kms_key_arn,  # from sagemaker_config
            "EnableNetworkIsolation": False,
        }

        expected_endpoint_config_2 = {
            **expected_endpoint_config_1,
            "EndpointConfigName": endpoint_config_2_name,
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": name,
                    "InitialInstanceCount": 1,
                    "InstanceType": cpu_instance_type,
                    "InitialVariantWeight": 1.0,
                    "VolumeSizeInGB": 16,
                }
            ],
            "EnableNetworkIsolation": False,
        }

        # TODO: Update expected tags for endpoints if injection behavior is changed
        expected_model_tags = {"Tags": MODEL_TAGS}
        expected_endpoint_1_tags = {"Tags": test_tags + ENDPOINT_CONFIG_TAGS}
        expected_endpoint_2_tags = {"Tags": test_tags + ENDPOINT_CONFIG_TAGS}
        expected_endpoint_config_1_tags = {"Tags": test_tags + ENDPOINT_CONFIG_TAGS}
        expected_endpoint_config_2_tags = {"Tags": test_tags + ENDPOINT_CONFIG_TAGS}

        # Doing the comparison in this way simplifies debugging failures for this test,
        # because all the values can be compared and checked together at once, rather than having
        # to run the test repeatedly to get through 10 separate comparisons one at a time
        assert [
            model,
            endpoint_1,
            endpoint_2,
            endpoint_config_1,
            endpoint_config_2,
            set(model_tags),
            set(endpoint_1_tags),
            set(endpoint_2_tags),
            set(endpoint_config_1_tags),
            set(endpoint_config_2_tags),
        ] == [
            expected_model,
            expected_endpoint_1,
            expected_endpoint_2,
            expected_endpoint_config_1,
            expected_endpoint_config_2,
            set(expected_model_tags),
            set(expected_endpoint_1_tags),
            set(expected_endpoint_2_tags),
            set(expected_endpoint_config_1_tags),
            set(expected_endpoint_config_2_tags),
        ]

    # Finally delete the model. (Endpoints should be deleted by the
    # timeout_and_delete_endpoint_by_name above )
    pipeline_model.delete_model()
    with pytest.raises(Exception) as exception:
        sagemaker_session.sagemaker_client.describe_model(ModelName=pipeline_model.name)
        assert "Could not find model" in str(exception.value)
