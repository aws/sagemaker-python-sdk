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

from six.moves.urllib.parse import urlparse

from tests.integ import DATA_DIR

AES_ENCRYPTION_ENABLED = {"ServerSideEncryption": "AES256"}


def test_upload_data_absolute_file(sagemaker_session):
    """Test the method ``Session.upload_data`` can upload one encrypted file to S3 bucket"""
    data_path = os.path.join(DATA_DIR, "upload_data_tests", "file1.py")
    uploaded_file = sagemaker_session.upload_data(data_path, extra_args=AES_ENCRYPTION_ENABLED)
    parsed_url = urlparse(uploaded_file)
    s3_client = sagemaker_session.boto_session.client("s3")
    head = s3_client.head_object(Bucket=parsed_url.netloc, Key=parsed_url.path.lstrip("/"))
    assert head["ServerSideEncryption"] == "AES256"


def test_upload_data_absolute_dir(sagemaker_session):
    """Test the method ``Session.upload_data`` can upload encrypted objects to S3 bucket"""
    data_path = os.path.join(DATA_DIR, "upload_data_tests", "nested_dir")
    uploaded_dir = sagemaker_session.upload_data(data_path, extra_args=AES_ENCRYPTION_ENABLED)
    parsed_url = urlparse(uploaded_dir)
    s3_bucket = parsed_url.netloc
    s3_prefix = parsed_url.path.lstrip("/")
    s3_client = sagemaker_session.boto_session.client("s3")
    for file in os.listdir(data_path):
        s3_key = "{}/{}".format(s3_prefix, file)
        head = s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
        assert head["ServerSideEncryption"] == "AES256"
