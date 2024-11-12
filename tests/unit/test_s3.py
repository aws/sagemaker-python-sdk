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

SESSION_MOCK_WITH_PREFIX = Mock(
    default_bucket=Mock(return_value="session_bucket"), default_bucket_prefix="session_prefix"
)
SESSION_MOCK_WITHOUT_PREFIX = Mock(
    default_bucket=Mock(return_value="session_bucket"), default_bucket_prefix=None
)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        default_bucket_prefix=None,
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
        callback=None,
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
        callback=None,
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


@pytest.mark.parametrize(
    "input_url, expected_bucket, expected_prefix",
    [
        ("s3://bucket/code_location", "bucket", "code_location"),
        ("s3://bucket/code_location/sub_location", "bucket", "code_location/sub_location"),
        ("s3://bucket/code_location/sub_location/", "bucket", "code_location/sub_location/"),
        ("s3://bucket/", "bucket", ""),
        ("s3://bucket", "bucket", ""),
    ],
)
def test_parse_s3_url(input_url, expected_bucket, expected_prefix):
    bucket, key_prefix = s3.parse_s3_url(input_url)
    assert bucket == expected_bucket
    assert key_prefix == expected_prefix


def test_parse_s3_url_fail():
    with pytest.raises(ValueError) as error:
        s3.parse_s3_url("t3://code_location")
    assert "Expecting 's3' scheme" in str(error)


@pytest.mark.parametrize(
    "expected_output, input_args",
    [
        # simple cases
        ("foo", ["foo"]),
        ("foo/bar", ("foo", "bar")),
        ("foo/bar", ("foo/", "bar")),
        ("foo/bar", ("/foo/", "bar")),
        # ----------------
        # cases with s3://
        ("s3://foo/bar", ("s3://", "foo", "bar")),
        ("s3://foo/bar", ("s3://", "/foo", "bar")),
        ("s3://foo/bar", ("s3://foo", "bar")),
        ("s3://foo/bar/baz", ("s3://", "foo/bar/", "baz/")),
        ("s3:", ["s3:"]),
        ("s3:", ["s3:/"]),
        ("s3://", ["s3://"]),
        ("s3://", (["s3:////"])),
        ("s3:", ("/", "s3://")),
        ("s3://", ("s3://", "/")),
        ("s/3/:", ("s", "3", ":", "/", "/")),
        # ----------------
        # cases with empty or None
        ("", []),
        ("s3://foo/bar", ("s3://", "", "foo", "", "bar", "")),
        ("s3://foo/bar", ("s3://", None, "foo", None, "bar", None)),
        ("foo", (None, "foo")),
        ("", ("", "", "")),
        ("", ("")),
        ("", ([None])),
        ("", (None, None, None)),
        # ----------------
        # cases with trailing slash
        ("", ["/"]),
        ("", ["/////"]),
        ("foo", ["foo/"]),
        ("foo", ["foo/////"]),
        ("foo/bar", ("foo", "bar/")),
        ("foo/bar", ("foo/", "bar/")),
        ("foo/bar", ("/foo/", "bar/")),
        # ----------------
        # cases with leading slashes
        # (os.path.join and pathlib.PurePosixPath discard anything before the last leading slash)
        ("foo/bar", ("/foo", "bar/")),
        ("foo/bar", ("/////foo/", "bar/")),
        ("foo", ("/", "foo")),
        ("s3://foo/bar/baz", ("s3://", "foo", "/bar", "baz")),
        ("s3://foo/bar/baz", ("s3://", "foo", "/bar", "/baz")),
        # ----------------
        # cases with multiple slashes (note: multiple slashes are allowed by S3)
        # (pathlib.PurePosixPath collapses multiple slashes to one)
        ("s3://foo/bar/baz", ("s3://", "foo////bar/////", "baz/")),
        ("s3://foo/bar/baz", ("s3://", "foo////bar/", "/////baz/")),
        # ----------------
        # cases with a dot
        # (pathlib.PurePosixPath collapses some single dots)
        ("f.oo/bar", ("f.oo", "bar")),
        ("foo/.bar", ("foo", ".bar")),
        ("foo/.bar", ("foo", "/.bar")),
        ("foo./bar", ("foo.", "bar")),
        ("foo/./bar", ("foo/.", "bar")),
        ("foo/./bar", ("foo/./", "bar")),
        ("foo/./bar", ["foo/./bar"]),
        (
            "s3://foo/..././bar/..../.././baz",
            ("s3://", "foo//..././bar/", "..../.././/baz/"),
        ),
        # ----------------
        # cases with 2 dots
        ("f..oo/bar", ("f..oo", "bar")),
        ("foo/..bar", ("foo", "..bar")),
        ("foo/..bar", ("foo", "/..bar")),
        ("foo../bar", ("foo..", "bar")),
        ("foo/../bar", ("foo/..", "bar")),
        ("foo/../bar", ("foo/../", "bar")),
        ("foo/../bar", ["foo/../bar"]),
    ],
)
def test_path_join(expected_output, input_args):
    assert s3.s3_path_join(*input_args) == expected_output


@pytest.mark.parametrize(
    "expected_output, input_args",
    [
        ("foo/", ["foo"]),
        ("foo/", ["foo///"]),
        ("foo/bar/", ("foo", "bar")),
        ("foo/bar/", ("foo/", "bar")),
        ("foo/bar/", ("/foo/", "bar")),
        ("s3://foo/bar/", ("s3://", "foo", "bar")),
        ("s3://foo/bar/", ("s3://", "/foo", "bar")),
        ("s3://foo/bar/", ("s3://foo", "bar")),
        ("s3://foo/bar/baz/", ("s3://", "foo/bar/", "baz/")),
        ("s3://foo/bar/", ("s3://", "", "foo", "", "bar", "")),
        ("s3://foo/bar/", ("s3://", None, "foo", None, "bar", None)),
        ("foo/", (None, "foo")),
        ("", ("", "", "")),
        ("", ("")),
        ("", ("/")),
        ("", ("///")),
        ("", ([None])),
        ("", (None, None, None)),
        ("s3:/", ["s3:"]),
        ("s3:/", ["s3:/"]),
        ("s3://", ["s3://"]),
        ("s3://", (["s3:////"])),
        ("s3:/", ("/", "s3://")),
        ("s3://", ("s3://", "/")),
        ("s/3/:/", ("s", "3", ":", "/", "/")),
    ],
)
def test_s3_path_join_with_end_slash(expected_output, input_args):
    assert s3.s3_path_join(*input_args, with_end_slash=True) == expected_output


@pytest.mark.parametrize(
    "input_bucket, input_prefix, input_session, expected_bucket, expected_prefix",
    [
        ("input-bucket", None, None, "input-bucket", None),
        ("input-bucket", "input-prefix", None, "input-bucket", "input-prefix"),
        ("input-bucket", None, SESSION_MOCK_WITH_PREFIX, "input-bucket", None),
        ("input-bucket", "input-prefix", SESSION_MOCK_WITH_PREFIX, "input-bucket", "input-prefix"),
        (None, None, SESSION_MOCK_WITH_PREFIX, "session_bucket", "session_prefix"),
        (None, None, SESSION_MOCK_WITHOUT_PREFIX, "session_bucket", ""),
        (
            None,
            "input-prefix",
            SESSION_MOCK_WITH_PREFIX,
            "session_bucket",
            "session_prefix/input-prefix",
        ),
        (None, "input-prefix", SESSION_MOCK_WITHOUT_PREFIX, "session_bucket", "input-prefix"),
    ],
)
def test_determine_bucket_and_prefix(
    input_bucket, input_prefix, input_session, expected_bucket, expected_prefix
):

    actual_bucket, actual_prefix = s3.determine_bucket_and_prefix(
        bucket=input_bucket, key_prefix=input_prefix, sagemaker_session=input_session
    )

    assert (actual_bucket == expected_bucket) and (actual_prefix == expected_prefix)
