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

import pytest
from mock import patch, Mock, ANY
from sagemaker.remote_function import invoke_function
from sagemaker.remote_function.errors import SerializationError
from sagemaker.remote_function.core.pipeline_variables import Context
from sagemaker.utils import sagemaker_timestamp

TEST_REGION = "us-west-2"
TEST_S3_BASE_URI = "s3://my-bucket/"
TEST_S3_KMS_KEY = "my-kms-key"
TEST_RUN_IN_CONTEXT = '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}'
TEST_STEP_NAME = "training-step"
TEST_EXECUTION_ID = "some-execution-id"
FUNC_STEP_S3_DIR = sagemaker_timestamp()


def mock_args():
    return [
        "--region",
        TEST_REGION,
        "--s3_base_uri",
        TEST_S3_BASE_URI,
        "--s3_kms_key",
        TEST_S3_KMS_KEY,
    ]


def mock_args_with_run_in_context():
    return [
        "--region",
        TEST_REGION,
        "--s3_base_uri",
        TEST_S3_BASE_URI,
        "--s3_kms_key",
        TEST_S3_KMS_KEY,
        "--run_in_context",
        TEST_RUN_IN_CONTEXT,
    ]


def mock_args_with_execution_and_step():
    return [
        "--s3_base_uri",
        TEST_S3_BASE_URI,
        "--s3_kms_key",
        TEST_S3_KMS_KEY,
        "--pipeline_execution_id",
        TEST_EXECUTION_ID,
        "--pipeline_step_name",
        TEST_STEP_NAME,
        "--property_references",
        "Parameter.a",
        "1.0",
        "Parameter.b",
        "2.0",
        "--region",
        TEST_REGION,
        "--func_step_s3_dir",
        FUNC_STEP_S3_DIR,
    ]


def mock_session():
    return Mock()


@patch("sagemaker.remote_function.invoke_function._load_run_object")
@patch("sys.exit")
@patch("sagemaker.remote_function.core.stored_function.StoredFunction.load_and_invoke")
@patch(
    "sagemaker.remote_function.invoke_function._get_sagemaker_session",
    return_value=mock_session(),
)
def test_main_success(_get_sagemaker_session, load_and_invoke, _exit_process, _load_run_object):
    invoke_function.main(mock_args())

    _get_sagemaker_session.assert_called_with(TEST_REGION)
    load_and_invoke.assert_called()
    _load_run_object.assert_not_called()
    _exit_process.assert_called_with(0)


@patch("sagemaker.remote_function.invoke_function._load_run_object")
@patch("sys.exit")
@patch("sagemaker.remote_function.core.stored_function.StoredFunction.load_and_invoke")
@patch(
    "sagemaker.remote_function.invoke_function._get_sagemaker_session",
    return_value=mock_session(),
)
def test_main_success_with_run(
    _get_sagemaker_session, load_and_invoke, _exit_process, _load_run_object
):
    invoke_function.main(mock_args_with_run_in_context())

    _get_sagemaker_session.assert_called_with(TEST_REGION)
    load_and_invoke.assert_called()
    _load_run_object.assert_called_once_with(TEST_RUN_IN_CONTEXT, _get_sagemaker_session())
    _exit_process.assert_called_with(0)


@patch("sagemaker.remote_function.invoke_function._load_run_object")
@patch("sys.exit")
@patch("sagemaker.remote_function.core.stored_function.StoredFunction")
@patch(
    "sagemaker.remote_function.invoke_function._get_sagemaker_session",
    return_value=mock_session(),
)
@pytest.mark.parametrize(
    "args",
    [
        (mock_args_with_execution_and_step(), False),
        (mock_args_with_execution_and_step() + ["--serialize_output_to_json", "false"], False),
        (mock_args_with_execution_and_step() + ["--serialize_output_to_json", "False"], False),
        (mock_args_with_execution_and_step() + ["--serialize_output_to_json", "true"], True),
        (mock_args_with_execution_and_step() + ["--serialize_output_to_json", "True"], True),
    ],
)
def test_main_success_with_pipeline_context(
    _get_sagemaker_session, mock_stored_function, _exit_process, _load_run_object, args
):

    args_input, expected_serialize_output_to_json = args
    invoke_function.main(args_input)

    _get_sagemaker_session.assert_called_with(TEST_REGION)
    mock_stored_function.assert_called_with(
        sagemaker_session=ANY,
        s3_base_uri=TEST_S3_BASE_URI,
        s3_kms_key=TEST_S3_KMS_KEY,
        context=Context(
            execution_id=TEST_EXECUTION_ID,
            step_name=TEST_STEP_NAME,
            property_references={
                "Parameter.a": "1.0",
                "Parameter.b": "2.0",
            },
            serialize_output_to_json=expected_serialize_output_to_json,
            func_step_s3_dir=FUNC_STEP_S3_DIR,
        ),
    )
    _load_run_object.assert_not_called()
    _exit_process.assert_called_with(0)


@patch("sagemaker.remote_function.invoke_function._load_run_object")
@patch("sagemaker.remote_function.invoke_function.handle_error")
@patch("sys.exit")
@patch("sagemaker.remote_function.core.stored_function.StoredFunction.load_and_invoke")
@patch(
    "sagemaker.remote_function.invoke_function._get_sagemaker_session",
    return_value=mock_session(),
)
def test_main_failure(
    _get_sagemaker_session, load_and_invoke, _exit_process, handle_error, _load_run_object
):
    ser_err = SerializationError("some failure reason")
    load_and_invoke.side_effect = ser_err
    handle_error.return_value = 1

    invoke_function.main(mock_args())

    _get_sagemaker_session.assert_called_with(TEST_REGION)
    load_and_invoke.assert_called()
    _load_run_object.assert_not_called()
    handle_error.assert_called_with(
        error=ser_err,
        sagemaker_session=_get_sagemaker_session(),
        s3_base_uri=TEST_S3_BASE_URI,
        s3_kms_key=TEST_S3_KMS_KEY,
    )
    _exit_process.assert_called_with(1)


@patch("sagemaker.remote_function.invoke_function._load_run_object")
@patch("sagemaker.remote_function.invoke_function.handle_error")
@patch("sys.exit")
@patch("sagemaker.remote_function.core.stored_function.StoredFunction.load_and_invoke")
@patch(
    "sagemaker.remote_function.invoke_function._get_sagemaker_session",
    return_value=mock_session(),
)
def test_main_failure_with_step(
    _get_sagemaker_session, load_and_invoke, _exit_process, handle_error, _load_run_object
):
    ser_err = SerializationError("some failure reason")
    load_and_invoke.side_effect = ser_err
    handle_error.return_value = 1

    invoke_function.main(mock_args_with_execution_and_step())

    _get_sagemaker_session.assert_called_with(TEST_REGION)
    load_and_invoke.assert_called()
    _load_run_object.assert_not_called()
    s3_uri = TEST_S3_BASE_URI + TEST_EXECUTION_ID + "/" + TEST_STEP_NAME
    handle_error.assert_called_with(
        error=ser_err,
        sagemaker_session=_get_sagemaker_session(),
        s3_base_uri=s3_uri,
        s3_kms_key=TEST_S3_KMS_KEY,
    )
    _exit_process.assert_called_with(1)
