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
import json

import boto3
from botocore.config import Config
from sagemaker import Session

DEFAULT_REGION = "us-west-2"

@pytest.fixture(scope="module")
def modules_boto_session(request):
    config = request.config.getoption("--boto-config")
    if config:
        return boto3.Session(**json.loads(config))
    else:
        return boto3.Session(region_name=DEFAULT_REGION)

@pytest.fixture(scope="module")
def modules_sagemaker_session(request, modules_boto_session):
    sagemaker_client = (
        modules_boto_session.client(
            "sagemaker", 
            config=Config(retries={"max_attempts": 10, "mode": "standard"})
        )
    )
    return Session(boto_session=modules_boto_session)
