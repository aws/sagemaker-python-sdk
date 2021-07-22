# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest

from sagemaker.serverless import LambdaModel
from sagemaker.utils import unique_name_from_base

# See tests/data/serverless for the image source code.
ACCOUNT_ID = 142577830533
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/serverless-integ-test:latest"
ROLE = f"arn:aws:iam::{ACCOUNT_ID}:role/lambda_basic_execution"
URL = "https://sagemaker-integ-tests-data.s3.us-east-1.amazonaws.com/cat.jpeg"


def test_lambda():
    client = boto3.client("lambda")
    if client.get_caller_identity().get("Account") != ACCOUNT_ID:
        pytest.skip("The container image is private to the CI account.")

    model = LambdaModel(image_uri=IMAGE_URI, role=ROLE, client=client)

    predictor = model.deploy(
        unique_name_from_base("my-lambda-function"), timeout=60, memory_size=4092
    )
    prediction = predictor.predict({"url": URL})

    assert prediction == {"class": "tabby"}

    model.delete_model()
    predictor.delete_predictor()
