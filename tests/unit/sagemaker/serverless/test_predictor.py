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

import io
from mock import Mock
import pytest

from sagemaker.serverless import LambdaPredictor

FUNCTION_NAME = "my-function"


@pytest.fixture
def mock_client():
    return Mock()


def test_predict(mock_client):
    mock_client.invoke = Mock(
        return_value={
            "StatusCode": 200,
            "Payload": io.BytesIO(b'{"class": "cat"}'),
            "ResponseMetadata": {"HTTPHeaders": {"content-type": "application/json"}},
        }
    )
    predictor = LambdaPredictor(FUNCTION_NAME, mock_client)

    prediction = predictor.predict({"url": "https://images.com/cat.jpg"})

    mock_client.invoke.assert_called_once
    _, kwargs = mock_client.invoke.call_args
    assert kwargs["FunctionName"] == FUNCTION_NAME

    assert prediction == {"class": "cat"}


def test_delete_endpoint(mock_client):
    predictor = LambdaPredictor(FUNCTION_NAME, client=mock_client)

    predictor.delete_endpoint()

    mock_client.delete_function.assert_called_once()
    _, kwargs = mock_client.delete_function.call_args
    assert kwargs["FunctionName"] == FUNCTION_NAME


def test_content_type(mock_client):
    predictor = LambdaPredictor(FUNCTION_NAME, client=mock_client)
    assert predictor.content_type == "application/json"


def test_accept(mock_client):
    predictor = LambdaPredictor(FUNCTION_NAME, client=mock_client)
    assert predictor.accept == ("application/json",)


def test_function_name(mock_client):
    predictor = LambdaPredictor(FUNCTION_NAME, client=mock_client)
    assert predictor.function_name == FUNCTION_NAME
