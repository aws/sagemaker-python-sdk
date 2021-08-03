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

from sagemaker.serverless import LambdaModel
from sagemaker.utils import unique_name_from_base

URL = "https://sagemaker-integ-tests-data.s3.us-east-1.amazonaws.com/cat.jpeg"
REPOSITORY_NAME = "my-lambda-repository"
ROLE_NAME = "LambdaExecutionRole"


@pytest.fixture(name="image_uri", scope="module")
def fixture_image_uri(account, region):
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{REPOSITORY_NAME}:latest"


@pytest.fixture(name="role", scope="module")
def fixture_role(account):
    return f"arn:aws:iam::{account}:role/{ROLE_NAME}"


@pytest.fixture(name="client")
def fixture_client(boto_session):
    return boto_session.client("lambda")


def test_lambda(image_uri, role, client):
    model = LambdaModel(image_uri=image_uri, role=role, client=client)

    predictor = model.deploy(
        unique_name_from_base("my-lambda-function"), timeout=60, memory_size=4092
    )
    prediction = predictor.predict({"url": URL})

    assert prediction == {"class": "tabby"}

    model.delete_model()
    predictor.delete_predictor()
