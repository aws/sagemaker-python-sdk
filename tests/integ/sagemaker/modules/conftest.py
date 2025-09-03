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
"""This module contains code to test image builder"""
from __future__ import absolute_import

import pytest

import os
import boto3
from sagemaker.modules import Session

DEFAULT_REGION = "us-west-2"


@pytest.fixture(scope="module")
def modules_sagemaker_session():
    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION
        region_manual_set = True
    else:
        region_manual_set = False

    boto_session = boto3.Session(region_name=os.environ["AWS_DEFAULT_REGION"])
    sagemaker_session = Session(boto_session=boto_session)

    yield sagemaker_session

    if region_manual_set and "AWS_DEFAULT_REGION" in os.environ:
        del os.environ["AWS_DEFAULT_REGION"]
