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

import sagemaker
import tests.integ
import tests.integ.timeout
from sagemaker.model_monitor import DataCaptureConfig, NetworkConfig
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.utils import unique_name_from_base
from tests.integ.retry import retries

ROLE = "SageMakerRole"
SKLEARN_FRAMEWORK = "scikit-learn"

INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.m5.xlarge"
VOLUME_SIZE_IN_GB = 20
MAX_RUNTIME_IN_SECONDS = 2 * 60 * 60
ENVIRONMENT = {"env_key_1": "env_value_1"}
TAGS = [{"Key": "tag_key_1", "Value": "tag_value_1"}]
NETWORK_CONFIG = NetworkConfig(enable_network_isolation=True)

CUSTOM_SAMPLING_PERCENTAGE = 10
CUSTOM_CAPTURE_OPTIONS = ["REQUEST"]
CUSTOM_CSV_CONTENT_TYPES = ["text/csvtype1", "text/csvtype2"]
CUSTOM_JSON_CONTENT_TYPES = ["application/jsontype1", "application/jsontype2"]


def test_enabling_data_capture_on_endpoint_shows_correct_data_capture_status(
    sagemaker_session, tensorflow_inference_latest_version
):
    endpoint_name = unique_name_from_base("sagemaker-tensorflow-serving")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = TensorFlowModel(
            model_data=model_data,
            role=ROLE,
            framework_version=tensorflow_inference_latest_version,
            sagemaker_session=sagemaker_session,
        )
        predictor = model.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            endpoint_name=endpoint_name,
        )

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=predictor.endpoint_name
        )

        endpoint_config_desc = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )

        assert endpoint_config_desc.get("DataCaptureConfig") is None

        predictor.enable_data_capture()

        # Wait for endpoint to finish updating
        # Endpoint update takes ~7min. 25 retries * 60s sleeps = 25min timeout
        for _ in retries(
            max_retry_count=25,
            exception_message_prefix="Waiting for 'InService' endpoint status",
            seconds_to_sleep=60,
        ):
            new_endpoint = sagemaker_session.sagemaker_client.describe_endpoint(
                EndpointName=predictor.endpoint_name
            )
            if new_endpoint["EndpointStatus"] == "InService":
                break

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=predictor.endpoint_name
        )

        endpoint_config_desc = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )

        assert endpoint_config_desc["DataCaptureConfig"]["EnableCapture"]


def test_disabling_data_capture_on_endpoint_shows_correct_data_capture_status(
    sagemaker_session, tensorflow_inference_latest_version
):
    endpoint_name = unique_name_from_base("sagemaker-tensorflow-serving")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = TensorFlowModel(
            model_data=model_data,
            role=ROLE,
            framework_version=tensorflow_inference_latest_version,
            sagemaker_session=sagemaker_session,
        )
        destination_s3_uri = os.path.join(
            "s3://", sagemaker_session.default_bucket(), endpoint_name, "custom"
        )
        predictor = model.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            endpoint_name=endpoint_name,
            data_capture_config=DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=CUSTOM_SAMPLING_PERCENTAGE,
                destination_s3_uri=destination_s3_uri,
                capture_options=CUSTOM_CAPTURE_OPTIONS,
                csv_content_types=CUSTOM_CSV_CONTENT_TYPES,
                json_content_types=CUSTOM_JSON_CONTENT_TYPES,
                sagemaker_session=sagemaker_session,
            ),
        )

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=predictor.endpoint_name
        )

        endpoint_config_desc = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )

        assert endpoint_config_desc["DataCaptureConfig"]["EnableCapture"]
        assert (
            endpoint_config_desc["DataCaptureConfig"]["InitialSamplingPercentage"]
            == CUSTOM_SAMPLING_PERCENTAGE
        )
        assert endpoint_config_desc["DataCaptureConfig"]["CaptureOptions"] == [
            {"CaptureMode": "Input"}
        ]
        assert (
            endpoint_config_desc["DataCaptureConfig"]["CaptureContentTypeHeader"]["CsvContentTypes"]
            == CUSTOM_CSV_CONTENT_TYPES
        )
        assert (
            endpoint_config_desc["DataCaptureConfig"]["CaptureContentTypeHeader"][
                "JsonContentTypes"
            ]
            == CUSTOM_JSON_CONTENT_TYPES
        )

        predictor.disable_data_capture()

        # Wait for endpoint to finish updating
        # Endpoint update takes ~7min. 25 retries * 60s sleeps = 25min timeout
        for _ in retries(
            max_retry_count=25,
            exception_message_prefix="Waiting for 'InService' endpoint status",
            seconds_to_sleep=60,
        ):
            new_endpoint = sagemaker_session.sagemaker_client.describe_endpoint(
                EndpointName=predictor.endpoint_name
            )
            if new_endpoint["EndpointStatus"] == "InService":
                break

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=predictor.endpoint_name
        )

        endpoint_config_desc = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )

        assert not endpoint_config_desc["DataCaptureConfig"]["EnableCapture"]


def test_updating_data_capture_on_endpoint_shows_correct_data_capture_status(
    sagemaker_session, tensorflow_inference_latest_version
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-tensorflow-serving")
    model_data = sagemaker_session.upload_data(
        path=os.path.join(tests.integ.DATA_DIR, "tensorflow-serving-test-model.tar.gz"),
        key_prefix="tensorflow-serving/models",
    )
    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        model = TensorFlowModel(
            model_data=model_data,
            role=ROLE,
            framework_version=tensorflow_inference_latest_version,
            sagemaker_session=sagemaker_session,
        )
        destination_s3_uri = os.path.join(
            "s3://", sagemaker_session.default_bucket(), endpoint_name, "custom"
        )
        predictor = model.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            endpoint_name=endpoint_name,
        )

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=predictor.endpoint_name
        )

        endpoint_config_desc = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )

        assert endpoint_config_desc.get("DataCaptureConfig") is None

        predictor.update_data_capture_config(
            data_capture_config=DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=CUSTOM_SAMPLING_PERCENTAGE,
                destination_s3_uri=destination_s3_uri,
                capture_options=CUSTOM_CAPTURE_OPTIONS,
                csv_content_types=CUSTOM_CSV_CONTENT_TYPES,
                json_content_types=CUSTOM_JSON_CONTENT_TYPES,
                sagemaker_session=sagemaker_session,
            )
        )

        # Wait for endpoint to finish updating
        # Endpoint update takes ~7min. 25 retries * 60s sleeps = 25min timeout
        for _ in retries(
            max_retry_count=25,
            exception_message_prefix="Waiting for 'InService' endpoint status",
            seconds_to_sleep=60,
        ):
            new_endpoint = sagemaker_session.sagemaker_client.describe_endpoint(
                EndpointName=predictor.endpoint_name
            )
            if new_endpoint["EndpointStatus"] == "InService":
                break

        endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=predictor.endpoint_name
        )

        endpoint_config_desc = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_desc["EndpointConfigName"]
        )

        assert endpoint_config_desc["DataCaptureConfig"]["EnableCapture"]
        assert (
            endpoint_config_desc["DataCaptureConfig"]["InitialSamplingPercentage"]
            == CUSTOM_SAMPLING_PERCENTAGE
        )
        assert endpoint_config_desc["DataCaptureConfig"]["CaptureOptions"] == [
            {"CaptureMode": "Input"}
        ]
        assert (
            endpoint_config_desc["DataCaptureConfig"]["CaptureContentTypeHeader"]["CsvContentTypes"]
            == CUSTOM_CSV_CONTENT_TYPES
        )
        assert (
            endpoint_config_desc["DataCaptureConfig"]["CaptureContentTypeHeader"][
                "JsonContentTypes"
            ]
            == CUSTOM_JSON_CONTENT_TYPES
        )
