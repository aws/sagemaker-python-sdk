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
"""``S3Downloader`` reads a byte range, streams lines, and returns ``HeadObject`` metadata.

Before this change ``S3Downloader.read_bytes`` had no ``byte_range`` parameter and downloaded
the whole object, and the class had no ``iter_lines`` and no ``head``. Fifteen SageMaker agent
skill scripts split ``s3://`` URIs by hand, eight called ``get_object(Range=...)``, and four
called ``head_object`` directly, each without the ``ExpectedBucketOwner`` spot check.
"""

from __future__ import absolute_import

from datetime import datetime, timezone

import pytest
from mock import Mock

from sagemaker.core.s3 import client as s3

REGION = "us-west-2"
S3_URI = "s3://mybucket/prefix/data.jsonl"
ACCOUNT_ID = "123456789012"


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
    session_mock.s3_client = Mock(name="s3_client")
    session_mock._get_account_id_if_default_bucket = Mock(return_value=None)
    return session_mock


def _get_object_response(payload: bytes):
    body = Mock(name="Body")
    body.read = Mock(return_value=payload)
    body.iter_lines = Mock(return_value=iter(payload.splitlines()))
    return {"Body": body}


def test_read_bytes_with_range_sends_inclusive_range_header(sagemaker_session):
    sagemaker_session.s3_client.get_object.return_value = _get_object_response(b"abc")

    result = s3.S3Downloader.read_bytes(S3_URI, sagemaker_session, byte_range=(0, 2))

    assert result == b"abc"
    sagemaker_session.s3_client.get_object.assert_called_once_with(
        Bucket="mybucket", Key="prefix/data.jsonl", Range="bytes=0-2"
    )
    sagemaker_session.s3_resource.Bucket.assert_not_called()


def test_read_bytes_with_range_passes_expected_bucket_owner(sagemaker_session):
    sagemaker_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
    sagemaker_session.s3_client.get_object.return_value = _get_object_response(b"x")

    s3.S3Downloader.read_bytes(S3_URI, sagemaker_session, byte_range=(10, 1033))

    sagemaker_session._get_account_id_if_default_bucket.assert_called_once_with("mybucket")
    sagemaker_session.s3_client.get_object.assert_called_once_with(
        Bucket="mybucket",
        Key="prefix/data.jsonl",
        Range="bytes=10-1033",
        ExpectedBucketOwner=ACCOUNT_ID,
    )


def test_read_bytes_without_range_downloads_the_whole_object(sagemaker_session):
    def _download_fileobj(key, fileobj, ExtraArgs):
        fileobj.write(b"whole object")

    bucket_mock = sagemaker_session.s3_resource.Bucket.return_value
    bucket_mock.download_fileobj.side_effect = _download_fileobj

    result = s3.S3Downloader.read_bytes(S3_URI, sagemaker_session)

    assert result == b"whole object"
    bucket_mock.download_fileobj.assert_called_once()
    assert bucket_mock.download_fileobj.call_args.kwargs == {"ExtraArgs": None}
    sagemaker_session.s3_client.get_object.assert_not_called()


@pytest.mark.parametrize("byte_range", [(-1, 10), (5, 4), (-3, -1)])
def test_read_bytes_rejects_bad_range(sagemaker_session, byte_range):
    with pytest.raises(ValueError) as error:
        s3.S3Downloader.read_bytes(S3_URI, sagemaker_session, byte_range=byte_range)

    assert "byte_range" in str(error.value)
    sagemaker_session.s3_client.get_object.assert_not_called()


def test_iter_lines_streams_and_decodes_each_line(sagemaker_session):
    response = _get_object_response("first\nzweite Zeile \u00e4\nthird".encode("utf-8"))
    sagemaker_session.s3_client.get_object.return_value = response

    lines = s3.S3Downloader.iter_lines(S3_URI, sagemaker_session)

    assert list(lines) == ["first", "zweite Zeile \u00e4", "third"]
    response["Body"].iter_lines.assert_called_once_with()
    response["Body"].read.assert_not_called()
    sagemaker_session.s3_client.get_object.assert_called_once_with(
        Bucket="mybucket", Key="prefix/data.jsonl"
    )


def test_iter_lines_uses_the_given_encoding_and_owner_check(sagemaker_session):
    sagemaker_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
    response = _get_object_response("caf\u00e9\n".encode("latin-1"))
    sagemaker_session.s3_client.get_object.return_value = response

    lines = s3.S3Downloader.iter_lines(S3_URI, sagemaker_session, encoding="latin-1")

    assert list(lines) == ["caf\u00e9"]
    sagemaker_session.s3_client.get_object.assert_called_once_with(
        Bucket="mybucket", Key="prefix/data.jsonl", ExpectedBucketOwner=ACCOUNT_ID
    )


def test_head_maps_head_object_fields(sagemaker_session):
    last_modified = datetime(2026, 10, 23, 12, 0, 0, tzinfo=timezone.utc)
    sagemaker_session.s3_client.head_object.return_value = {
        "ContentLength": 2048,
        "LastModified": last_modified,
        "ETag": '"d41d8cd98f00b204e9800998ecf8427e"',
        "ContentType": "application/jsonlines",
        "Metadata": {},
    }

    head = s3.S3Downloader.head(S3_URI, sagemaker_session)

    assert head == s3.S3ObjectHead(
        content_length=2048,
        last_modified=last_modified,
        etag='"d41d8cd98f00b204e9800998ecf8427e"',
        content_type="application/jsonlines",
    )
    sagemaker_session.s3_client.head_object.assert_called_once_with(
        Bucket="mybucket", Key="prefix/data.jsonl"
    )


def test_head_without_content_type_and_with_owner_check(sagemaker_session):
    sagemaker_session._get_account_id_if_default_bucket.return_value = ACCOUNT_ID
    sagemaker_session.s3_client.head_object.return_value = {
        "ContentLength": 0,
        "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "ETag": '"etag"',
    }

    head = s3.S3Downloader.head(S3_URI, sagemaker_session)

    assert head.content_type is None
    assert head.content_length == 0
    sagemaker_session.s3_client.head_object.assert_called_once_with(
        Bucket="mybucket", Key="prefix/data.jsonl", ExpectedBucketOwner=ACCOUNT_ID
    )


def test_read_primitives_fall_back_to_a_boto_client_without_s3_client(sagemaker_session):
    sagemaker_session.s3_client = None
    boto_client = sagemaker_session.boto_session.client.return_value
    boto_client.get_object.return_value = _get_object_response(b"ab")

    result = s3.S3Downloader.read_bytes(S3_URI, sagemaker_session, byte_range=(0, 1))

    assert result == b"ab"
    sagemaker_session.boto_session.client.assert_called_once_with("s3", region_name=REGION)
