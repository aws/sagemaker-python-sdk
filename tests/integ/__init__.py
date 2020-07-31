# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import boto3

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAINING_DEFAULT_TIMEOUT_MINUTES = 20
TUNING_DEFAULT_TIMEOUT_MINUTES = 20
TRANSFORM_DEFAULT_TIMEOUT_MINUTES = 20
AUTO_ML_DEFAULT_TIMEMOUT_MINUTES = 60

# these regions have some p2 and p3 instances, but not enough for continuous testing
HOSTING_NO_P2_REGIONS = [
    "ap-east-1",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "sa-east-1",
    "us-west-1",
]
HOSTING_NO_P3_REGIONS = [
    "ap-east-1",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "sa-east-1",
    "us-west-1",
]
TRAINING_NO_P2_REGIONS = [
    "ap-east-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "me-south-1",
    "sa-east-1",
    "us-west-1",
]

# EI is currently only supported in the following regions
# regions were derived from https://aws.amazon.com/machine-learning/elastic-inference/pricing/
EI_SUPPORTED_REGIONS = [
    "ap-northeast-1",
    "ap-northeast-2",
    "eu-west-1",
    "us-east-1",
    "us-east-2",
    "us-west-2",
]

NO_LDA_REGIONS = ["eu-west-3", "eu-north-1", "sa-east-1", "ap-east-1", "me-south-1"]
NO_MARKET_PLACE_REGIONS = ["eu-west-3", "eu-north-1", "sa-east-1", "ap-east-1", "me-south-1"]
NO_AUTO_ML_REGIONS = ["sa-east-1", "me-south-1", "ap-east-1", "eu-west-3"]
NO_MODEL_MONITORING_REGIONS = ["me-south-1"]

EFS_TEST_ENABLED_REGION = []

logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)


def test_region():
    return os.environ.get("TEST_AWS_REGION_NAME", boto3.session.Session().region_name)
