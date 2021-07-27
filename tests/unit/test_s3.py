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
import pytest
from mock import Mock

from sagemaker import s3

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
CURRENT_JOB_NAME = "currentjobname"
SOURCE_NAME = "source"
KMS_KEY = "kmskey"


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
    session_mock.upload_data = Mock(name="upload_data", return_value="s3_uri_to_uploaded_data")
    session_mock.download_data = Mock(name="download_data")
    return session_mock


def test_upload(sagemaker_session, caplog):
    desired_s3_uri = os.path.join("s3://", BUCKET_NAME, CURRENT_JOB_NAME, SOURCE_NAME)
    s3.S3Uploader.upload(
        local_path="/path/to/app.jar",
        desired_s3_uri=desired_s3_uri,
        sagemaker_session=sagemaker_session,
    )
    sagemaker_session.upload_data.assert_called_with(
        path="/path/to/app.jar",
        bucket=BUCKET_NAME,
        key_prefix=os.path.join(CURRENT_JOB_NAME, SOURCE_NAME),
        extra_args=None,
    )


def test_upload_with_kms_key(sagemaker_session):
    desired_s3_uri = os.path.join("s3://", BUCKET_NAME, CURRENT_JOB_NAME, SOURCE_NAME)
    s3.S3Uploader.upload(
        local_path="/path/to/app.jar",
        desired_s3_uri=desired_s3_uri,
        kms_key=KMS_KEY,
        sagemaker_session=sagemaker_session,
    )
    sagemaker_session.upload_data.assert_called_with(
        path="/path/to/app.jar",
        bucket=BUCKET_NAME,
        key_prefix=os.path.join(CURRENT_JOB_NAME, SOURCE_NAME),
        extra_args={"SSEKMSKeyId": KMS_KEY, "ServerSideEncryption": "aws:kms"},
    )


def test_download(sagemaker_session):
    s3_uri = os.path.join("s3://", BUCKET_NAME, CURRENT_JOB_NAME, SOURCE_NAME)
    s3.S3Downloader.download(
        s3_uri=s3_uri, local_path="/path/for/download/", sagemaker_session=sagemaker_session
    )
    sagemaker_session.download_data.assert_called_with(
        path="/path/for/download/",
        bucket=BUCKET_NAME,
        key_prefix=os.path.join(CURRENT_JOB_NAME, SOURCE_NAME),
        extra_args=None,
    )


def test_download_with_kms_key(sagemaker_session):
    s3_uri = os.path.join("s3://", BUCKET_NAME, CURRENT_JOB_NAME, SOURCE_NAME)
    s3.S3Downloader.download(
        s3_uri=s3_uri,
        local_path="/path/for/download/",
        kms_key=KMS_KEY,
        sagemaker_session=sagemaker_session,
    )
    sagemaker_session.download_data.assert_called_with(
        path="/path/for/download/",
        bucket=BUCKET_NAME,
        key_prefix=os.path.join(CURRENT_JOB_NAME, SOURCE_NAME),
        extra_args={"SSECustomerKey": KMS_KEY},
    )


def test_parse_s3_url():
    bucket, key_prefix = s3.parse_s3_url("s3://bucket/code_location")
    assert "bucket" == bucket
    assert "code_location" == key_prefix


def test_parse_s3_url_fail():
    with pytest.raises(ValueError) as error:
        s3.parse_s3_url("t3://code_location")
    assert "Expecting 's3' scheme" in str(error)


def test_path_join():
    test_cases = (
        ("foo/bar", ("foo", "bar")),
        ("foo/bar", ("foo/", "bar")),
        ("foo/bar", ("/foo/", "bar")),
        ("s3://foo/bar", ("s3://", "foo", "bar")),
        ("s3://foo/bar", ("s3://", "/foo", "bar")),
        ("s3://foo/bar", ("s3://foo", "bar")),
    )

    for expected, args in test_cases:
        assert expected == s3.s3_path_join(*args)
