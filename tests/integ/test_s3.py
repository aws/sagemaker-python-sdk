# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import uuid

import pytest

from sagemaker.s3 import S3Uploader
from sagemaker.s3 import S3Downloader

from tests.integ.kms_utils import get_or_create_kms_key


@pytest.fixture(scope="module")
def s3_files_kms_key(sagemaker_session):
    return get_or_create_kms_key(sagemaker_session=sagemaker_session)


def test_statistics_object_creation_from_s3_uri_with_customizations(
    sagemaker_session, s3_files_kms_key
):
    file_1_body = "First File Body."
    file_1_name = "first_file.txt"
    file_2_body = "Second File Body."
    file_2_name = "second_file.txt"

    my_uuid = str(uuid.uuid4())

    base_s3_uri = os.path.join(
        "s3://", sagemaker_session.default_bucket(), "integ-test-test-s3-list", my_uuid
    )
    file_1_s3_uri = os.path.join(base_s3_uri, file_1_name)
    file_2_s3_uri = os.path.join(base_s3_uri, file_2_name)

    S3Uploader.upload_string_as_file_body(
        body=file_1_body,
        desired_s3_uri=file_1_s3_uri,
        kms_key=s3_files_kms_key,
        session=sagemaker_session,
    )

    S3Uploader.upload_string_as_file_body(
        body=file_2_body,
        desired_s3_uri=file_2_s3_uri,
        kms_key=s3_files_kms_key,
        session=sagemaker_session,
    )

    s3_uris = S3Downloader.list(s3_uri=base_s3_uri, session=sagemaker_session)

    assert file_1_name in s3_uris[0]
    assert file_2_name in s3_uris[1]

    assert file_1_body == S3Downloader.read_file(s3_uri=s3_uris[0], session=sagemaker_session)
    assert file_2_body == S3Downloader.read_file(s3_uri=s3_uris[1], session=sagemaker_session)
