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

from sagemaker.core import constants


def test_script_param_name():
    """Test SCRIPT_PARAM_NAME constant."""
    assert constants.SCRIPT_PARAM_NAME == "sagemaker_program"


def test_dir_param_name():
    """Test DIR_PARAM_NAME constant."""
    assert constants.DIR_PARAM_NAME == "sagemaker_submit_directory"


def test_container_log_level_param_name():
    """Test CONTAINER_LOG_LEVEL_PARAM_NAME constant."""
    assert constants.CONTAINER_LOG_LEVEL_PARAM_NAME == "sagemaker_container_log_level"


def test_job_name_param_name():
    """Test JOB_NAME_PARAM_NAME constant."""
    assert constants.JOB_NAME_PARAM_NAME == "sagemaker_job_name"


def test_model_server_workers_param_name():
    """Test MODEL_SERVER_WORKERS_PARAM_NAME constant."""
    assert constants.MODEL_SERVER_WORKERS_PARAM_NAME == "sagemaker_model_server_workers"


def test_sagemaker_region_param_name():
    """Test SAGEMAKER_REGION_PARAM_NAME constant."""
    assert constants.SAGEMAKER_REGION_PARAM_NAME == "sagemaker_region"


def test_sagemaker_output_location():
    """Test SAGEMAKER_OUTPUT_LOCATION constant."""
    assert constants.SAGEMAKER_OUTPUT_LOCATION == "sagemaker_s3_output"


def test_neo_allowed_frameworks():
    """Test NEO_ALLOWED_FRAMEWORKS constant."""
    expected_frameworks = {
        "mxnet", "tensorflow", "keras", "pytorch", "onnx", "xgboost", "tflite"
    }
    assert constants.NEO_ALLOWED_FRAMEWORKS == expected_frameworks
    assert isinstance(constants.NEO_ALLOWED_FRAMEWORKS, set)


def test_neo_allowed_frameworks_contains_expected():
    """Test that NEO_ALLOWED_FRAMEWORKS contains all expected frameworks."""
    assert "mxnet" in constants.NEO_ALLOWED_FRAMEWORKS
    assert "tensorflow" in constants.NEO_ALLOWED_FRAMEWORKS
    assert "keras" in constants.NEO_ALLOWED_FRAMEWORKS
    assert "pytorch" in constants.NEO_ALLOWED_FRAMEWORKS
    assert "onnx" in constants.NEO_ALLOWED_FRAMEWORKS
    assert "xgboost" in constants.NEO_ALLOWED_FRAMEWORKS
    assert "tflite" in constants.NEO_ALLOWED_FRAMEWORKS


def test_neo_allowed_frameworks_count():
    """Test that NEO_ALLOWED_FRAMEWORKS has the expected number of frameworks."""
    assert len(constants.NEO_ALLOWED_FRAMEWORKS) == 7


def test_all_exports():
    """Test that __all__ contains all expected exports."""
    expected_exports = [
        "SCRIPT_PARAM_NAME",
        "DIR_PARAM_NAME",
        "CONTAINER_LOG_LEVEL_PARAM_NAME",
        "JOB_NAME_PARAM_NAME",
        "MODEL_SERVER_WORKERS_PARAM_NAME",
        "SAGEMAKER_REGION_PARAM_NAME",
        "SAGEMAKER_OUTPUT_LOCATION",
        "NEO_ALLOWED_FRAMEWORKS",
    ]
    assert constants.__all__ == expected_exports


def test_constants_are_strings():
    """Test that all parameter name constants are strings."""
    assert isinstance(constants.SCRIPT_PARAM_NAME, str)
    assert isinstance(constants.DIR_PARAM_NAME, str)
    assert isinstance(constants.CONTAINER_LOG_LEVEL_PARAM_NAME, str)
    assert isinstance(constants.JOB_NAME_PARAM_NAME, str)
    assert isinstance(constants.MODEL_SERVER_WORKERS_PARAM_NAME, str)
    assert isinstance(constants.SAGEMAKER_REGION_PARAM_NAME, str)
    assert isinstance(constants.SAGEMAKER_OUTPUT_LOCATION, str)
