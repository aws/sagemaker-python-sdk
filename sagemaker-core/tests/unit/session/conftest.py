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

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_boto_session():
    """Common mock boto3 session fixture"""
    mock_session = Mock()
    mock_session.region_name = "us-west-2"
    mock_session.client.return_value = Mock()
    mock_session.resource.return_value = Mock()
    return mock_session


@pytest.fixture
def mock_sagemaker_client():
    """Mock SageMaker client fixture"""
    return Mock()


@pytest.fixture
def mock_s3_client():
    """Mock S3 client fixture"""
    return Mock()


@pytest.fixture
def mock_sts_client():
    """Mock STS client fixture"""
    mock_client = Mock()
    mock_client.get_caller_identity.return_value = {"Account": "123456789012"}
    return mock_client


@pytest.fixture
def sample_model_package_args():
    """Sample model package arguments for testing"""
    return {
        "model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test",
        "containers": [{"Image": "test-image"}],
        "content_types": ["application/json"],
        "response_types": ["application/json"],
        "inference_instances": ["ml.m5.large"],
        "transform_instances": ["ml.m5.large"],
    }


@pytest.fixture
def sample_production_variant():
    """Sample production variant for testing"""
    return {
        "ModelName": "test-model",
        "VariantName": "AllTraffic",
        "InitialVariantWeight": 1,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.large",
    }
