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
from mock import patch, Mock
from sagemaker.remote_function import invoke_function
from sagemaker.remote_function.errors import SerializationError

TEST_REGION = "us-west-2"
TEST_S3_BASE_URI = "s3://my-bucket/"
TEST_S3_KMS_KEY = "my-kms-key"
TEST_RUN_IN_CONTEXT = '{"experiment_name": "my-exp-name", "run_name": "my-run-name"}'
TEST_HMAC_KEY = "some-hmac-key"


def mock_args():
    args = Mock()
    args.region = TEST_REGION
    args.s3_base_uri = TEST_S3_BASE_URI
    args.s3_kms_key = TEST_S3_KMS_KEY
    args.run_in_context = None

    return args


def mock_args_with_run_in_context():
    args = Mock()
    args.region = TEST_REGION
    args.s3_base_uri = TEST_S3_BASE_URI
    args.s3_kms_key = TEST_S3_KMS_KEY
    args.run_in_context = TEST_RUN_IN_CONTEXT

    return args


def mock_session():
    return Mock()


@patch("sagemaker.remote_function.invoke_function._parse_agrs", new=mock_args)
@patch("sagemaker.remote_function.invoke_function._load_run_object")
@patch("sys.exit")
@patch("sagemaker.remote_function.core.stored_function.StoredFunction.load_and_invoke")
@patch(
    "sagemaker.remote_function.invoke_function._get_sagemaker_session",
    return_value=mock_session(),
)
def test_main_success(_get_sagemaker_session, load_and_invoke, _exit_process, _load_run_object):
    os.environ["REMOTE_FUNCTION_SECRET_KEY"] = TEST_HMAC_KEY
    invoke_function.main()

    _get_sagemaker_session.assert_called_with(TEST_REGION)
    load_and_invoke.assert_called()
    _load_run_object.assert_not_called()
    _exit_process.assert_called_with(0)


@patch("sagemaker.remote_function.invoke_function._parse_agrs", new=mock_args_with_run_in_context)
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
    os.environ["REMOTE_FUNCTION_SECRET_KEY"] = TEST_HMAC_KEY
    invoke_function.main()

    _get_sagemaker_session.assert_called_with(TEST_REGION)
    load_and_invoke.assert_called()
    _load_run_object.assert_called_once_with(TEST_RUN_IN_CONTEXT, _get_sagemaker_session())
    _exit_process.assert_called_with(0)


@patch("sagemaker.remote_function.invoke_function._parse_agrs", new=mock_args)
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
    os.environ["REMOTE_FUNCTION_SECRET_KEY"] = TEST_HMAC_KEY
    ser_err = SerializationError("some failure reason")
    load_and_invoke.side_effect = ser_err
    handle_error.return_value = 1

    invoke_function.main()

    _get_sagemaker_session.assert_called_with(TEST_REGION)
    load_and_invoke.assert_called()
    _load_run_object.assert_not_called()
    handle_error.assert_called_with(
        error=ser_err,
        sagemaker_session=_get_sagemaker_session(),
        s3_base_uri=TEST_S3_BASE_URI,
        s3_kms_key=TEST_S3_KMS_KEY,
        hmac_key=TEST_HMAC_KEY,
    )
    _exit_process.assert_called_with(1)
