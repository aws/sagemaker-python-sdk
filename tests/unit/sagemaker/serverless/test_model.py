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

from mock import Mock
import pytest

from sagemaker.serverless import LambdaModel

IMAGE_URI = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-lambda-image:latest"
ROLE = "arn:aws:iam::123456789012:role/MyLambdaExecutionRole"


@pytest.fixture
def mock_client():
    return Mock()


@pytest.mark.parametrize("wait", [False, True])
def test_deploy(mock_client, wait):
    model = LambdaModel(IMAGE_URI, ROLE, client=mock_client)
    mock_client.create_function = Mock(return_value={"State": "Pending"})
    mock_client.get_function_configuration = Mock(return_value={"State": "Active"})

    function_name, timeout, memory_size = "my-function", 3, 128
    predictor = model.deploy(function_name, timeout=timeout, memory_size=memory_size, wait=wait)

    mock_client.create_function.assert_called_once()
    _, kwargs = mock_client.create_function.call_args
    assert kwargs["FunctionName"] == function_name
    assert kwargs["PackageType"] == "Image"
    assert kwargs["Timeout"] == timeout
    assert kwargs["MemorySize"] == memory_size
    assert kwargs["Role"] == ROLE
    assert kwargs["Code"] == {"ImageUri": IMAGE_URI}

    assert predictor.function_name == function_name


def test_destroy():
    model = LambdaModel(IMAGE_URI, ROLE, client=mock_client)
    model.destroy()  # NOTE: This method is a no-op.
