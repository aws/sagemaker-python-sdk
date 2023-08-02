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
from sagemaker.exceptions import (
    AsyncInferenceModelError,
    ObjectNotExistedError,
    UnexpectedClientError,
)

DEFAULT_OUTPUT_PATH = "s3://some-output-path/object-name"
DEFAULT_FAILURE_PATH = "s3://some-failure-path/object-name"
ENDPOINT_NAME = "some-endpoint-name"
RETURN_VALUE = 0


def empty_s3_client():
    """
    Returns a mocked S3 client with the `get_object` method overridden
    to raise different exceptions based on the input.

    Exceptions raised:
    - `ClientError` with code "NoSuchKey"
    - `AsyncInferenceModelError`
    - `ObjectNotExistedError`
    - `ClientError` with code "SomeOtherError"
    - `UnexpectedClientError`

    """
    s3_client = Mock(name="s3-client")

    client_error_no_such_key = ClientError(
        error_response={"Error": {"Code": "NoSuchKey"}},
        operation_name="async-inference-response-test",
    )

    async_error = AsyncInferenceModelError("some error message")

    object_error = ObjectNotExistedError("some error message", DEFAULT_OUTPUT_PATH)

    client_error_other = ClientError(
        error_response={"Error": {"Code": "SomeOtherError", "Message": "some error message"}},
        operation_name="async-inference-response-test",
    )

    unexpected_error = UnexpectedClientError("some error message")

    s3_client.get_object = Mock(
        name="get_object",
        side_effect=[
            client_error_no_such_key,
            async_error,
            object_error,
            client_error_other,
            unexpected_error,
        ],
    )
    return s3_client


def empty_s3_client_to_verify_exceptions_for_null_failure_path():
    """
    Returns a mocked S3 client with the `get_object` method overridden
    to raise different exceptions based on the input.

    Exceptions raised:
    - `ObjectNotExistedError`
    - `UnexpectedClientError`

    """
    s3_client = Mock(name="s3-client")

    object_error = ObjectNotExistedError("Inference could still be running", DEFAULT_OUTPUT_PATH)

    unexpected_error = UnexpectedClientError("some error message")

    s3_client.get_object = Mock(
        name="get_object",
        side_effect=[
            object_error,
            unexpected_error,
        ],
    )
    return s3_client


def mock_s3_client():
    """
    This function returns a mocked S3 client object that has a get_object method with a side_effect
    that returns a dictionary with a Body key that points to a mocked response body object.
    """
    s3_client = Mock(name="s3-client")
    response_body = Mock("body")
    response_body.read = Mock("read", return_value=RETURN_VALUE)
    response_body.close = Mock("close", return_value=None)
    s3_client.get_object = Mock(
        name="get_object",
        side_effect=[
            {"Body": response_body},
        ],
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
        failure_path=DEFAULT_FAILURE_PATH,
    )
    assert async_inference_response.output_path == DEFAULT_OUTPUT_PATH
    assert async_inference_response.failure_path == DEFAULT_FAILURE_PATH


def test_wrong_waiter_config_object():
    predictor_async = AsyncPredictor(Predictor(ENDPOINT_NAME))
    async_inference_response = AsyncInferenceResponse(
        output_path=DEFAULT_OUTPUT_PATH,
        predictor_async=predictor_async,
        failure_path=DEFAULT_FAILURE_PATH,
    )

    with pytest.raises(
        ValueError,
        match="waiter_config should be a WaiterConfig object",
    ):
        async_inference_response.get_result(waiter_config={})


def test_get_result_success():
    """
    verifies that the result is returned correctly if no errors occur.
    """
    # Initialize AsyncInferenceResponse
    predictor_async = AsyncPredictor(Predictor(ENDPOINT_NAME))
    predictor_async.s3_client = mock_s3_client()
    async_inference_response = AsyncInferenceResponse(
        output_path=DEFAULT_OUTPUT_PATH,
        predictor_async=predictor_async,
        failure_path=DEFAULT_FAILURE_PATH,
    )

    result = async_inference_response.get_result()
    assert async_inference_response._result == result
    assert result == RETURN_VALUE


def test_get_result_verify_exceptions():
    """
    Verifies that get_result method raises the expected exception
    when an error occurs while fetching the result.
    """
    # Initialize AsyncInferenceResponse
    predictor_async = AsyncPredictor(Predictor(ENDPOINT_NAME))
    predictor_async.s3_client = empty_s3_client()
    async_inference_response = AsyncInferenceResponse(
        output_path=DEFAULT_OUTPUT_PATH,
        predictor_async=predictor_async,
        failure_path=DEFAULT_FAILURE_PATH,
    )

    # Test AsyncInferenceModelError
    with pytest.raises(AsyncInferenceModelError, match="Model returned error: some error message"):
        async_inference_response.get_result()

    # Test ObjectNotExistedError
    with pytest.raises(
        ObjectNotExistedError,
        match=f"Object not exist at {DEFAULT_OUTPUT_PATH}. some error message",
    ):
        async_inference_response.get_result()

    # Test UnexpectedClientError
    with pytest.raises(
        UnexpectedClientError, match="Encountered unexpected client error: some error message"
    ):
        async_inference_response.get_result()


def test_get_result_with_null_failure_path():
    """
    verifies that the result is returned correctly if no errors occur.
    """
    # Initialize AsyncInferenceResponse
    predictor_async = AsyncPredictor(Predictor(ENDPOINT_NAME))
    predictor_async.s3_client = mock_s3_client()
    async_inference_response = AsyncInferenceResponse(
        output_path=DEFAULT_OUTPUT_PATH, predictor_async=predictor_async, failure_path=None
    )

    result = async_inference_response.get_result()
    assert async_inference_response._result == result
    assert result == RETURN_VALUE


def test_get_result_verify_exceptions_with_null_failure_path():
    """
    Verifies that get_result method raises the expected exception
    when an error occurs while fetching the result.
    """
    # Initialize AsyncInferenceResponse
    predictor_async = AsyncPredictor(Predictor(ENDPOINT_NAME))
    predictor_async.s3_client = empty_s3_client_to_verify_exceptions_for_null_failure_path()
    async_inference_response = AsyncInferenceResponse(
        output_path=DEFAULT_OUTPUT_PATH,
        predictor_async=predictor_async,
        failure_path=None,
    )

    # Test ObjectNotExistedError
    with pytest.raises(
        ObjectNotExistedError,
        match=f"Object not exist at {DEFAULT_OUTPUT_PATH}. Inference could still be running",
    ):
        async_inference_response.get_result()

    # Test UnexpectedClientError
    with pytest.raises(
        UnexpectedClientError, match="Encountered unexpected client error: some error message"
    ):
        async_inference_response.get_result()
