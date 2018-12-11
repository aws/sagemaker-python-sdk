# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging
import os
import sys

import boto3

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TRAINING_DEFAULT_TIMEOUT_MINUTES = 20
TUNING_DEFAULT_TIMEOUT_MINUTES = 20
TRANSFORM_DEFAULT_TIMEOUT_MINUTES = 20
PYTHON_VERSION = 'py' + str(sys.version_info.major)
HOSTING_NO_P2_REGIONS = ['ca-central-1', 'eu-west-2', 'us-west-1']
HOSTING_SCARCE_P2_REGIONS = HOSTING_NO_P2_REGIONS + ['eu-central-1']
HOSTING_NO_P3_REGIONS = ['ap-southeast-1', 'ap-southeast-2', 'ap-south-1', 'ca-central-1',
                         'eu-west-2', 'us-west-1']

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)


def test_region():
    return os.environ.get('TEST_AWS_REGION_NAME', boto3.session.Session().region_name)
