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

import boto3
from six.moves.urllib.parse import urlparse


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
