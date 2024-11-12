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
import re
import tarfile

import boto3
from six.moves.urllib.parse import urlparse

from sagemaker.utils import custom_extractall_tarfile


def assert_s3_files_exist(sagemaker_session, s3_url, files):
    parsed_url = urlparse(s3_url)
    region = sagemaker_session.boto_region_name
    s3 = boto3.client("s3", region_name=region)
    contents = s3.list_objects_v2(Bucket=parsed_url.netloc, Prefix=parsed_url.path.lstrip("/"))[
        "Contents"
    ]
    for f in files:
        found = [x["Key"] for x in contents if x["Key"].endswith(f)]
        if not found:
            raise ValueError("File {} is not found under {}".format(f, s3_url))


def assert_s3_file_patterns_exist(sagemaker_session, s3_url, file_patterns):
    parsed_url = urlparse(s3_url)
    region = sagemaker_session.boto_region_name
    s3 = boto3.client("s3", region_name=region)
    contents = s3.list_objects_v2(Bucket=parsed_url.netloc, Prefix=parsed_url.path.lstrip("/"))[
        "Contents"
    ]
    for pattern in file_patterns:
        search_pattern = re.compile(pattern)
        found = [x["Key"] for x in contents if search_pattern.search(x["Key"])]
        if not found:
            raise ValueError("File {} is not found under {}".format(pattern, s3_url))


def extract_files_from_s3(s3_url, tmpdir, sagemaker_session):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource("s3", region_name=sagemaker_session.boto_region_name)

    model = os.path.join(tmpdir, "model")
    s3.Bucket(parsed_url.netloc).download_file(parsed_url.path.lstrip("/"), model)

    with tarfile.open(model, "r") as tar_file:
        custom_extractall_tarfile(tar_file, tmpdir)
