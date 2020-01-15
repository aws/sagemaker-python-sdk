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
from mock import Mock  # , patch, mock_open, MagicMock

# from mock import patch

# from sagemaker.model_monitor import ModelMonitor
from sagemaker.model_monitor import DefaultModelMonitor

# from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor, ScriptProcessor
# from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.network import NetworkConfig

# from sagemaker.processing import ProcessingJob

REGION = "us-west-2"
BUCKET_NAME = "mybucket"

ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.m5.10xlarge"
VOLUME_SIZE_IN_GB = 2
VOLUME_KMS_KEY = "volume-kms-key"
OUTPUT_KMS_KEY = "output-kms-key"
MAX_RUNTIME_IN_SECONDS = 3
BASE_JOB_NAME = "base-job-name"
ENV_KEY_1 = "env_key_1"
ENV_VALUE_1 = "env_key_1"
ENVIRONMENT = {ENV_KEY_1: ENV_VALUE_1}
TAG_KEY_1 = "tag_key_1"
TAG_VALUE_1 = "tag_value_1"
TAGS = [{"Key": TAG_KEY_1, "Value": TAG_VALUE_1}]
NETWORK_CONFIG = NetworkConfig(enable_network_isolation=True)
ENABLE_CLOUDWATCH_METRICS = True

BASELINE_DATASET_PATH = "/my/local/path/baseline.csv"
PREPROCESSOR_PATH = "/my/local/path/preprocessor.py"
POSTPROCESSOR_PATH = "/my/local/path/postprocessor.py"
OUTPUT_S3_URI = "s3://output-s3-uri/"


CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"


# TODO-reinvent-2019: Continue to flesh these out.
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


# @patch("sagemaker.processing.Processor")
# @patch("sagemaker.processing.Processor.run")
# @patch("sagemaker.processing.Processor.processing_job", return_value=ProcessingJob(
#     sagemaker_session=sagemaker_session,
#     inputs="",
#     outputs=""
# ))
def test_default_model_monitor_suggest_baseline(sagemaker_session):
    my_default_monitor = DefaultModelMonitor(
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=VOLUME_SIZE_IN_GB,
        volume_kms_key=VOLUME_KMS_KEY,
        output_kms_key=OUTPUT_KMS_KEY,
        max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS,
        base_job_name=BASE_JOB_NAME,
        sagemaker_session=sagemaker_session,
        env=ENVIRONMENT,
        tags=TAGS,
        network_config=NETWORK_CONFIG,
    )

    my_default_monitor.suggest_baseline(
        baseline_dataset=BASELINE_DATASET_PATH,
        dataset_format=DatasetFormat.csv(header=False),
        record_preprocessor_script=PREPROCESSOR_PATH,
        post_analytics_processor_script=POSTPROCESSOR_PATH,
        output_s3_uri=OUTPUT_S3_URI,
        wait=False,
        logs=False,
    )

    assert my_default_monitor.role == ROLE
    assert my_default_monitor.instance_count == INSTANCE_COUNT
    assert my_default_monitor.instance_type == INSTANCE_TYPE
    assert my_default_monitor.volume_size_in_gb == VOLUME_SIZE_IN_GB
    assert my_default_monitor.volume_kms_key == VOLUME_KMS_KEY
    assert my_default_monitor.output_kms_key == OUTPUT_KMS_KEY
    assert my_default_monitor.max_runtime_in_seconds == MAX_RUNTIME_IN_SECONDS
    assert my_default_monitor.base_job_name == BASE_JOB_NAME
    assert my_default_monitor.sagemaker_session == sagemaker_session
    assert my_default_monitor.tags == TAGS
    assert my_default_monitor.network_config == NETWORK_CONFIG

    assert BASE_JOB_NAME in my_default_monitor.latest_baselining_job_name
    assert my_default_monitor.latest_baselining_job_name != BASE_JOB_NAME

    assert my_default_monitor.env[ENV_KEY_1] == ENV_VALUE_1

    # processor().run.assert_called_once(
    #
    # )
