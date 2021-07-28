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
import subprocess

import boto3
import botocore
import pytest

from sagemaker.serverless import LambdaModel
from sagemaker.utils import unique_name_from_base

from tests.integ import DATA_DIR

URL = "https://sagemaker-integ-tests-data.s3.us-east-1.amazonaws.com/cat.jpeg"

IMAGE_NAME = "my-lambda-function"
REPOSITORY_NAME = "my-lambda-repository"
BUILD_CONTEXT = os.path.join(DATA_DIR, "serverless")

ROLE_NAME = "LambdaExecutionRole"
POLICY_DOCUMENT = '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
POLICY_ARN = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"


@pytest.fixture(name="tag", scope="module")
def fixture_tag():
    return unique_name_from_base("tag")


@pytest.fixture(name="image_uri", scope="module")
def fixture_image_uri(account, region, tag, boto_session):
    client = boto_session.client("ecr")
    try:
        client.describe_repositories(repositoryNames=[REPOSITORY_NAME])["repositories"]
    except client.exceptions.RepositoryNotFoundException:
        client.create_repository(repositoryName=REPOSITORY_NAME)

    process = subprocess.Popen(["aws", "ecr", "get-login-password", "--region", region], stdout=subprocess.PIPE)
    subprocess.check_call(["docker", "login", "--username", "AWS", "--password-stdin", f"{account}.dkr.ecr.{region}.amazonaws.com"], stdin=process.stdout)
    process.wait()

    subprocess.check_call(["docker", "build", "-t", IMAGE_NAME, BUILD_CONTEXT])

    image_uri = f"{account}.dkr.ecr.us-west-2.amazonaws.com/{REPOSITORY_NAME}:{tag}"
    subprocess.check_call(["docker", "tag", f"{IMAGE_NAME}:latest", image_uri])
    subprocess.check_call(["docker", "push", image_uri])

    yield image_uri

    client.batch_delete_image(repositoryName=REPOSITORY_NAME, imageIds=[{"imageTag": tag}])
    subprocess.check_call(["docker", "rmi", IMAGE_NAME, image_uri])


@pytest.fixture(name="role", scope="module")
def fixture_role(boto_session):
    client = boto_session.client("iam")
    try:
        response = client.get_role(RoleName=ROLE_NAME)
        return response["Role"]["Arn"]
    except client.exceptions.NoSuchEntityException:
        response = client.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=POLICY_DOCUMENT)
        client.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=POLICY_ARN)
        return response["Role"]["Arn"]


def test_lambda(image_uri, role, boto_session):
    client = boto_session.client("lambda")
    model = LambdaModel(image_uri=image_uri, role=role, client=client)

    predictor = model.deploy(
        unique_name_from_base("my-lambda-function"), timeout=60, memory_size=4092
    )
    prediction = predictor.predict({"url": URL})

    assert prediction == {"class": "tabby"}

    model.delete_model()
    predictor.delete_predictor()
