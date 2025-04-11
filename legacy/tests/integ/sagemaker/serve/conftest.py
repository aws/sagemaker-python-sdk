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
import os
import boto3
import sagemaker
import sagemaker_core.helper.session_helper as core_session

DEFAULT_REGION = "us-west-2"


@pytest.fixture(scope="module")
def mb_sagemaker_session():
    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION
        region_manual_set = True
    else:
        region_manual_set = True

    boto_session = boto3.Session(region_name=os.environ["AWS_DEFAULT_REGION"])
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    yield sagemaker_session

    if region_manual_set and "AWS_DEFAULT_REGION" in os.environ:
        del os.environ["AWS_DEFAULT_REGION"]


@pytest.fixture(scope="module")
def mb_sagemaker_core_session():
    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION
        region_manual_set = True
    else:
        region_manual_set = True

    boto_session = boto3.Session(region_name=os.environ["AWS_DEFAULT_REGION"])
    sagemaker_session = core_session.Session(boto_session=boto_session)

    yield sagemaker_session

    if region_manual_set and "AWS_DEFAULT_REGION" in os.environ:
        del os.environ["AWS_DEFAULT_REGION"]
