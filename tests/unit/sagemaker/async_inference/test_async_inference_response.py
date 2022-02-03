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
from mock import Mock
from botocore.exceptions import ClientError
from sagemaker.predictor import Predictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.async_inference import AsyncInferenceResponse
from sagemaker.exceptions import ObjectNotExistedError, UnexpectedClientError

DEFAULT_OUTPUT_PATH = "s3://some-output-path/object-name"
ENDPOINT_NAME = "some-endpoint-name"
RETURN_VALUE = 0


def empty_s3_client():
    s3_client = Mock(name="s3-client")

    client_other_error = ClientError(
        error_response={"Error": {"Code": "SomeOtherError", "Message": "some-error-message"}},
        operation_name="client-other-error",
    )

    client_error = ClientError(
        error_response={"Error": {"Code": "NoSuchKey"}},
        operation_name="async-inference-response-test",
    )

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=RETURN_VALUE)
    response_body.close = Mock("close", return_value=None)

    s3_client.get_object = Mock(
        name="get_object",
        side_effect=[client_other_error, client_error, {"Body": response_body}],
    )

    return s3_client


def empty_deserializer():
    deserializer = Mock(name="deserializer")
    deserializer.deserialize = Mock(name="deserialize", return_value=RETURN_VALUE)
    return deserializer


def test_init_():
    predictor_async = AsyncPredictor(Predictor(ENDPOINT_NAME))
    async_inference_response = AsyncInferenceResponse(
        output_path=DEFAULT_OUTPUT_PATH,
        predictor_async=predictor_async,
    )
    assert async_inference_response.output_path == DEFAULT_OUTPUT_PATH


def test_get_result():
    predictor_async = AsyncPredictor(Predictor(ENDPOINT_NAME))
    predictor_async.s3_client = empty_s3_client()
    async_inference_response = AsyncInferenceResponse(
        output_path=DEFAULT_OUTPUT_PATH,
        predictor_async=predictor_async,
    )

    with pytest.raises(UnexpectedClientError):
        async_inference_response.get_result()

    with pytest.raises(ObjectNotExistedError, match="Inference could still be running"):
        async_inference_response.get_result()

    result = async_inference_response.get_result()
    assert async_inference_response._result == result
    assert result == RETURN_VALUE


def test_wrong_waiter_config_object():
    predictor_async = AsyncPredictor(Predictor(ENDPOINT_NAME))
    async_inference_response = AsyncInferenceResponse(
        output_path=DEFAULT_OUTPUT_PATH,
        predictor_async=predictor_async,
    )

    with pytest.raises(
        ValueError,
        match="waiter_config should be a WaiterConfig object",
    ):
        async_inference_response.get_result(waiter_config={})
