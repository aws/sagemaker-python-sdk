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

import logging
import os

import boto3

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAINING_DEFAULT_TIMEOUT_MINUTES = 40
TUNING_DEFAULT_TIMEOUT_MINUTES = 40
TRANSFORM_DEFAULT_TIMEOUT_MINUTES = 40
AUTO_ML_DEFAULT_TIMEMOUT_MINUTES = 60
AUTO_ML_V2_DEFAULT_WAITING_TIME_MINUTES = 10
MODEL_CARD_DEFAULT_TIMEOUT_MINUTES = 10

# these regions have some p2 and p3 instances, but not enough for continuous testing
HOSTING_NO_P2_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "sa-east-1",
    "us-west-1",
]
HOSTING_NO_P3_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "sa-east-1",
    "us-west-1",
]
TRAINING_NO_P2_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-south-1",  # not enough capacity
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "me-south-1",
    "sa-east-1",
    "us-west-1",
]
TRAINING_NO_P3_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-southeast-1",  # it has p3, but not enough
    "ap-southeast-2",  # it has p3, but not enough
    "ca-central-1",  # it has p3, but not enough
    "eu-central-1",  # it has p3, but not enough
    "eu-north-1",
    "eu-west-2",  # it has p3, but not enough
    "eu-west-3",
    "eu-south-1",
    "me-south-1",
    "sa-east-1",
    "us-west-1",
    "ap-northeast-1",  # it has p3, but not enough
    "ap-south-1",
    "ap-northeast-2",  # it has p3, but not enough
    "us-east-2",  # it has p3, but not enough
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

RL_SUPPORTED_REGIONS = (
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
)

NO_LDA_REGIONS = [
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "af-south-1",
    "eu-south-1",
]
NO_MARKET_PLACE_REGIONS = [
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "af-south-1",
    "eu-south-1",
]
NO_AUTO_ML_REGIONS = [
    "eu-west-3",
    "af-south-1",
    "eu-south-1",
]
NO_CANVAS_REGIONS = [
    "ca-central-1",
    "eu-north-1",
    "eu-west-2",
    "sa-east-1",
    "us-west-1",
]
NO_MODEL_MONITORING_REGIONS = ["me-south-1", "af-south-1", "eu-south-1"]
DRIFT_CHECK_BASELINES_SUPPORTED_REGIONS = [
    "us-east-2",
    "ca-central-1",
    "me-south-1",
    "us-west-2",
    "ap-east-1",
    "ap-northeast-2",
    "ap-southeast-2",
    "eu-west-2",
    "us-east-1",
]
EDGE_PACKAGING_SUPPORTED_REGIONS = [
    "us-east-2",
    "us-west-2",
    "us-east-1",
    "eu-west-1",
    "ap-northeast-1",
    "eu-central-1",
]

TRAINING_COMPILER_SUPPORTED_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-south-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "me-south-1",
    "sa-east-1",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]

INFERENCE_COMPONENT_SUPPORTED_REGIONS = [
    "ap-south-1",
    "us-west-2",
    "ca-central-1",
    "us-east-1",
    "us-east-2",
    "ap-northeast-2",
    "eu-west-2",
    "ap-southeast-2",
    "eu-west-1",
    "ap-northeast-1",
    "eu-central-1",
    "eu-north-1",
    "ap-southeast-1",
    "sa-east-1",
    "me-central-1",
    "ap-southeast-3",
]

# Data parallelism need to be tested with p3.16xlarge.
# The instance type is expensive and not supported in all the regions.
# Limiting the test to run in IAD and CMH
DATA_PARALLEL_TESTING_REGIONS = ["us-east-2", "us-east-1"]


EFS_TEST_ENABLED_REGION = []

logging.getLogger("boto3").setLevel(logging.INFO)
logging.getLogger("botocore").setLevel(logging.INFO)


def test_region():
    return os.environ.get("TEST_AWS_REGION_NAME", boto3.session.Session().region_name)
