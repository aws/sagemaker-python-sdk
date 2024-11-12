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

import io
import os
import uuid

import pytest

from sagemaker.s3 import S3Uploader
from sagemaker.s3 import S3Downloader

from tests.integ.kms_utils import get_or_create_kms_key


TMP_BASE_PATH = "/tmp"


@pytest.fixture(scope="module")
def s3_files_kms_key(sagemaker_session):
    return get_or_create_kms_key(sagemaker_session=sagemaker_session)


def test_s3_uploader_and_downloader_reads_files_when_given_file_name_uris(
    sagemaker_session, s3_files_kms_key
):
    my_uuid = str(uuid.uuid4())

    file_1_body = "First File Body {}.".format(my_uuid)
    file_1_name = "first_file_{}.txt".format(my_uuid)
    file_2_body = "Second File Body {}.".format(my_uuid)
    file_2_name = "second_file_{}.txt".format(my_uuid)

    base_s3_uri = os.path.join(
        "s3://", sagemaker_session.default_bucket(), "integ-test-test-s3-list", my_uuid
    )
    file_1_s3_uri = os.path.join(base_s3_uri, file_1_name)
    file_2_s3_uri = os.path.join(base_s3_uri, file_2_name)

    S3Uploader.upload_string_as_file_body(
        body=file_1_body,
        desired_s3_uri=file_1_s3_uri,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    S3Uploader.upload_string_as_file_body(
        body=file_2_body,
        desired_s3_uri=file_2_s3_uri,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    s3_uris = S3Downloader.list(s3_uri=base_s3_uri, sagemaker_session=sagemaker_session)

    assert file_1_name in s3_uris[0]
    assert file_2_name in s3_uris[1]

    assert file_1_body == S3Downloader.read_file(
        s3_uri=s3_uris[0], sagemaker_session=sagemaker_session
    )
    assert file_2_body == S3Downloader.read_file(
        s3_uri=s3_uris[1], sagemaker_session=sagemaker_session
    )


def test_s3_uploader_and_downloader_downloads_files_when_given_file_name_uris(
    sagemaker_session, s3_files_kms_key
):
    my_uuid = str(uuid.uuid4())

    file_1_body = "First File Body {}.".format(my_uuid)
    file_1_name = "first_file_{}.txt".format(my_uuid)
    file_2_body = "Second File Body {}.".format(my_uuid)
    file_2_name = "second_file_{}.txt".format(my_uuid)

    base_s3_uri = os.path.join(
        "s3://", sagemaker_session.default_bucket(), "integ-test-test-s3-list", my_uuid
    )
    file_1_s3_uri = os.path.join(base_s3_uri, file_1_name)
    file_2_s3_uri = os.path.join(base_s3_uri, file_2_name)

    S3Uploader.upload_string_as_file_body(
        body=file_1_body,
        desired_s3_uri=file_1_s3_uri,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    S3Uploader.upload_string_as_file_body(
        body=file_2_body,
        desired_s3_uri=file_2_s3_uri,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    s3_uris = S3Downloader.list(s3_uri=base_s3_uri, sagemaker_session=sagemaker_session)

    assert file_1_name in s3_uris[0]
    assert file_2_name in s3_uris[1]

    S3Downloader.download(
        s3_uri=s3_uris[0], local_path=TMP_BASE_PATH, sagemaker_session=sagemaker_session
    )
    S3Downloader.download(
        s3_uri=s3_uris[1], local_path=TMP_BASE_PATH, sagemaker_session=sagemaker_session
    )

    with open(os.path.join(TMP_BASE_PATH, file_1_name), "r") as f:
        assert file_1_body == f.read()

    with open(os.path.join(TMP_BASE_PATH, file_2_name), "r") as f:
        assert file_2_body == f.read()


def test_s3_uploader_and_downloader_downloads_files_when_given_directory_uris_with_files(
    sagemaker_session, s3_files_kms_key
):
    my_uuid = str(uuid.uuid4())

    file_1_body = "First File Body {}.".format(my_uuid)
    file_1_name = "first_file_{}.txt".format(my_uuid)
    file_2_body = "Second File Body {}.".format(my_uuid)
    file_2_name = "second_file_{}.txt".format(my_uuid)

    base_s3_uri = os.path.join(
        "s3://", sagemaker_session.default_bucket(), "integ-test-test-s3-list", my_uuid
    )
    file_1_s3_uri = os.path.join(base_s3_uri, file_1_name)
    file_2_s3_uri = os.path.join(base_s3_uri, file_2_name)

    S3Uploader.upload_string_as_file_body(
        body=file_1_body,
        desired_s3_uri=file_1_s3_uri,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    S3Uploader.upload_string_as_file_body(
        body=file_2_body,
        desired_s3_uri=file_2_s3_uri,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    s3_uris = S3Downloader.list(s3_uri=base_s3_uri, sagemaker_session=sagemaker_session)

    assert file_1_name in s3_uris[0]
    assert file_2_name in s3_uris[1]

    assert file_1_body == S3Downloader.read_file(
        s3_uri=s3_uris[0], sagemaker_session=sagemaker_session
    )
    assert file_2_body == S3Downloader.read_file(
        s3_uri=s3_uris[1], sagemaker_session=sagemaker_session
    )

    S3Downloader.download(
        s3_uri=base_s3_uri, local_path=TMP_BASE_PATH, sagemaker_session=sagemaker_session
    )

    with open(os.path.join(TMP_BASE_PATH, file_1_name), "r") as f:
        assert file_1_body == f.read()

    with open(os.path.join(TMP_BASE_PATH, file_2_name), "r") as f:
        assert file_2_body == f.read()


def test_s3_uploader_and_downloader_downloads_files_when_given_directory_uris_with_directory(
    sagemaker_session, s3_files_kms_key
):
    my_uuid = str(uuid.uuid4())
    my_inner_directory_uuid = str(uuid.uuid4())

    file_1_body = "First File Body {}.".format(my_uuid)
    file_1_name = "first_file_{}.txt".format(my_uuid)
    file_2_body = "Second File Body {}.".format(my_uuid)
    file_2_name = "second_file_{}.txt".format(my_uuid)

    base_s3_uri = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-test-s3-list",
        my_uuid,
        my_inner_directory_uuid,
    )
    file_1_s3_uri = os.path.join(base_s3_uri, file_1_name)
    file_2_s3_uri = os.path.join(base_s3_uri, file_2_name)

    S3Uploader.upload_string_as_file_body(
        body=file_1_body,
        desired_s3_uri=file_1_s3_uri,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    S3Uploader.upload_string_as_file_body(
        body=file_2_body,
        desired_s3_uri=file_2_s3_uri,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    s3_uris = S3Downloader.list(s3_uri=base_s3_uri, sagemaker_session=sagemaker_session)

    assert file_1_name in s3_uris[0]
    assert file_2_name in s3_uris[1]

    assert file_1_body == S3Downloader.read_file(
        s3_uri=s3_uris[0], sagemaker_session=sagemaker_session
    )
    assert file_2_body == S3Downloader.read_file(
        s3_uri=s3_uris[1], sagemaker_session=sagemaker_session
    )

    s3_directory_with_directory_underneath = os.path.join(
        "s3://", sagemaker_session.default_bucket(), "integ-test-test-s3-list", my_uuid
    )

    S3Downloader.download(
        s3_uri=s3_directory_with_directory_underneath,
        local_path=TMP_BASE_PATH,
        sagemaker_session=sagemaker_session,
    )

    with open(os.path.join(TMP_BASE_PATH, my_inner_directory_uuid, file_1_name), "r") as f:
        assert file_1_body == f.read()

    with open(os.path.join(TMP_BASE_PATH, my_inner_directory_uuid, file_2_name), "r") as f:
        assert file_2_body == f.read()


def test_upload_and_read_bytes(sagemaker_session, s3_files_kms_key):
    my_uuid = str(uuid.uuid4())
    base_s3_uri = os.path.join(
        "s3://", sagemaker_session.default_bucket(), "integ-test-test-upload-read-bytes", my_uuid
    )

    body = bytes(my_uuid, "utf-8")

    S3Uploader.upload_bytes(
        body,
        s3_uri=os.path.join(base_s3_uri, "from_bytes"),
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    S3Uploader.upload_bytes(
        io.BytesIO(body),
        s3_uri=os.path.join(base_s3_uri, "from_bytes_io"),
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    assert body == S3Downloader.read_bytes(
        s3_uri=os.path.join(base_s3_uri, "from_bytes"), sagemaker_session=sagemaker_session
    )

    assert body == S3Downloader.read_bytes(
        s3_uri=os.path.join(base_s3_uri, "from_bytes_io"), sagemaker_session=sagemaker_session
    )
