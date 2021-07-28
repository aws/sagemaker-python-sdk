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

from mock import MagicMock, Mock
import pytest

import sagemaker

UPLOAD_DATA_TESTS_FILE_DIR = "upload_data_tests"
SINGLE_FILE_NAME = "file1.py"
BODY = 'print("test")'
DESTINATION_DATA_TESTS_FILE = os.path.join(UPLOAD_DATA_TESTS_FILE_DIR, SINGLE_FILE_NAME)
BUCKET_NAME = "mybucket"
AES_ENCRYPTION_ENABLED = {"ServerSideEncryption": "AES256"}


@pytest.fixture()
def sagemaker_session():
    boto_mock = MagicMock(name="boto_session")
    client_mock = MagicMock()
    client_mock.get_caller_identity.return_value = {
        "UserId": "mock_user_id",
        "Account": "012345678910",
        "Arn": "arn:aws:iam::012345678910:user/mock-user",
    }
    boto_mock.client.return_value = client_mock
    ims = sagemaker.Session(boto_session=boto_mock)
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    return ims


def test_upload_string_file(sagemaker_session):
    result_s3_uri = sagemaker_session.upload_string_as_file_body(
        body=BODY, bucket=BUCKET_NAME, key=DESTINATION_DATA_TESTS_FILE
    )

    uploaded_files_with_args = [
        kwargs
        for name, args, kwargs in sagemaker_session.boto_session.mock_calls
        if name == "resource().Object().put"
    ]

    assert result_s3_uri == "s3://{}/{}".format(BUCKET_NAME, DESTINATION_DATA_TESTS_FILE)
    assert len(uploaded_files_with_args) == 1
    kwargs = uploaded_files_with_args[0]
    assert kwargs["Body"] == BODY


def test_upload_aes_encrypted_string_file(sagemaker_session):
    result_s3_uri = sagemaker_session.upload_string_as_file_body(
        body=BODY,
        bucket=BUCKET_NAME,
        key=DESTINATION_DATA_TESTS_FILE,
        kms_key=AES_ENCRYPTION_ENABLED,
    )

    uploaded_files_with_args = [
        kwargs
        for name, args, kwargs in sagemaker_session.boto_session.mock_calls
        if name == "resource().Object().put"
    ]

    assert result_s3_uri == "s3://{}/{}".format(BUCKET_NAME, DESTINATION_DATA_TESTS_FILE)
    assert len(uploaded_files_with_args) == 1
    kwargs = uploaded_files_with_args[0]
    assert kwargs["Body"] == BODY
    assert kwargs["SSEKMSKeyId"] == AES_ENCRYPTION_ENABLED
