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
import errno
from mock import patch, Mock, mock_open

from sagemaker.remote_function.errors import SerializationError, handle_error

TEST_S3_BASE_URI = "s3://my-bucket/"
TEST_S3_KMS_KEY = "my-kms-key"
TEST_HMAC_KEY = "some-hmac-key"


class _InvalidErrorNumberException(Exception):
    def __init__(self, *args, **kwargs):  # real signature unknown
        self.errno = "invalid"


@pytest.fixture()
def sagemaker_session():
    return Mock()


@pytest.mark.parametrize(
    "error, expected_exit_code, error_string",
    [
        (
            SerializationError("some failure reason"),
            1,
            "SerializationError('some failure reason')",
        ),
        (
            FileNotFoundError(errno.ENOENT, "No such file or directory"),
            errno.ENOENT,
            "FileNotFoundError(2, 'No such file or directory')",
        ),
        (
            Exception("No such file or directory"),
            1,
            "Exception('No such file or directory')",
        ),
        (
            _InvalidErrorNumberException("No such file or directory"),
            1,
            "_InvalidErrorNumberException('No such file or directory')",
        ),
    ],
)
@patch("sagemaker.remote_function.client.serialization.serialize_exception_to_s3")
@patch("builtins.open", new_callable=mock_open())
@patch("os.path.exists", return_value=False)
def test_handle_error(
    exists,
    mock_open_file,
    serialize_exception_to_s3,
    sagemaker_session,
    error,
    expected_exit_code,
    error_string,
):
    err = error
    exit_code = handle_error(
        error=err,
        sagemaker_session=sagemaker_session,
        s3_base_uri=TEST_S3_BASE_URI,
        s3_kms_key=TEST_S3_KMS_KEY,
        hmac_key=TEST_HMAC_KEY,
    )

    assert exit_code == expected_exit_code
    exists.assert_called_once_with("/opt/ml/output/failure")
    mock_open_file.assert_called_with("/opt/ml/output/failure", "w")
    mock_open_file.return_value.__enter__().write.assert_called_with(error_string)
    serialize_exception_to_s3.assert_called_with(
        exc=err,
        sagemaker_session=sagemaker_session,
        s3_uri=TEST_S3_BASE_URI + "exception",
        hmac_key=TEST_HMAC_KEY,
        s3_kms_key=TEST_S3_KMS_KEY,
    )
