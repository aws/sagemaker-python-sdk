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

from sagemaker.model_monitor import DataCaptureConfig

DEFAULT_ENABLE_CAPTURE = True
DEFAULT_SAMPLING_PERCENTAGE = 20
DEFAULT_BUCKET_NAME = "default-bucket"
DEFAULT_DESTINATION_S3_URI = "s3://" + DEFAULT_BUCKET_NAME + "/model-monitor/data-capture"
DEFAULT_KMS_KEY_ID = None
DEFAULT_CAPTURE_MODES = ["REQUEST", "RESPONSE"]
DEFAULT_CSV_CONTENT_TYPES = ["text/csv"]
DEFAULT_JSON_CONTENT_TYPES = ["application/json"]

NON_DEFAULT_ENABLE_CAPTURE = False
NON_DEFAULT_CAPTURE_STATUS = "STOPPED"
NON_DEFAULT_SAMPLING_PERCENTAGE = 97
NON_DEFAULT_DESTINATION_S3_URI = "s3://uri/"
NON_DEFAULT_KMS_KEY_ID = "my_kms_key_id"
NON_DEFAULT_CAPTURE_MODES = ["RESPONSE"]
NON_DEFAULT_CSV_CONTENT_TYPES = ["custom/csv-format"]
NON_DEFAULT_JSON_CONTENT_TYPES = ["custom/json-format"]


def test_to_request_dict_returns_correct_params_when_non_defaults_provided():
    data_capture_config = DataCaptureConfig(
        enable_capture=NON_DEFAULT_ENABLE_CAPTURE,
        sampling_percentage=NON_DEFAULT_SAMPLING_PERCENTAGE,
        destination_s3_uri=NON_DEFAULT_DESTINATION_S3_URI,
        kms_key_id=NON_DEFAULT_KMS_KEY_ID,
        csv_content_types=NON_DEFAULT_CSV_CONTENT_TYPES,
        json_content_types=NON_DEFAULT_JSON_CONTENT_TYPES,
    )

    assert data_capture_config.enable_capture == NON_DEFAULT_ENABLE_CAPTURE
    assert data_capture_config.sampling_percentage == NON_DEFAULT_SAMPLING_PERCENTAGE
    assert data_capture_config.destination_s3_uri == NON_DEFAULT_DESTINATION_S3_URI
    assert data_capture_config.kms_key_id == NON_DEFAULT_KMS_KEY_ID
    assert data_capture_config.csv_content_types == NON_DEFAULT_CSV_CONTENT_TYPES
    assert data_capture_config.json_content_types == NON_DEFAULT_JSON_CONTENT_TYPES


def test_to_request_dict_returns_correct_default_params_when_optionals_not_provided():
    data_capture_config = DataCaptureConfig(
        enable_capture=DEFAULT_ENABLE_CAPTURE, destination_s3_uri=DEFAULT_DESTINATION_S3_URI
    )

    assert data_capture_config.enable_capture == DEFAULT_ENABLE_CAPTURE
    assert data_capture_config.sampling_percentage == DEFAULT_SAMPLING_PERCENTAGE
    assert data_capture_config.destination_s3_uri == DEFAULT_DESTINATION_S3_URI
    assert data_capture_config.kms_key_id == DEFAULT_KMS_KEY_ID
    assert data_capture_config.csv_content_types == DEFAULT_CSV_CONTENT_TYPES
    assert data_capture_config.json_content_types == DEFAULT_JSON_CONTENT_TYPES
