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
from sagemaker.async_inference.waiter_config import WaiterConfig
from sagemaker.predictor import Predictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.exceptions import AsyncInferenceModelError, PollingTimeoutError

ENDPOINT = "mxnet_endpoint"
BUCKET_NAME = "mxnet_endpoint"
DEFAULT_CONTENT_TYPE = "application/octet-stream"
CSV_CONTENT_TYPE = "text/csv"
DEFAULT_ACCEPT = "*/*"
RETURN_VALUE = 0
CSV_RETURN_VALUE = "1,2,3\r\n"
PRODUCTION_VARIANT_1 = "PRODUCTION_VARIANT_1"
INFERENCE_ID = "inference-id"
ASYNC_OUTPUT_LOCATION = "s3://some-output-path/object-name"
ASYNC_FAILURE_LOCATION = "s3://some-failure-path/object-name"
ASYNC_INPUT_LOCATION = "s3://some-input-path/object-name"
ASYNC_CHECK_PERIOD = 1
ASYNC_PREDICTOR = "async-predictor"
DUMMY_DATA = [0, 1, 2, 3]

ENDPOINT_DESC = {"EndpointArn": "foo", "EndpointConfigName": ENDPOINT}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}

DEFAULT_WAITER_CONFIG = WaiterConfig(max_attempts=2, delay=2)  # set max_attempts=2


def empty_sagemaker_session():
    ims = Mock(
        name="sagemaker_session",
        default_bucket_prefix=None,
    )
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name="sagemaker_runtime")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    ims.sagemaker_runtime_client.invoke_endpoint_async = Mock(
        name="invoke_endpoint_async",
        return_value={
            "OutputLocation": ASYNC_OUTPUT_LOCATION,
            "FailureLocation": ASYNC_FAILURE_LOCATION,
        },
    )

    async_inference_model_error = AsyncInferenceModelError(message="some error from model")

    polling_timeout_error = PollingTimeoutError(
        message="Inference could still be running",
        output_path=ASYNC_OUTPUT_LOCATION,
        seconds=DEFAULT_WAITER_CONFIG.delay * DEFAULT_WAITER_CONFIG.max_attempts,
    )

    ims.s3_client = Mock(name="s3_client")
    ims.s3_client.get_object = Mock(
        name="get_object",
        side_effect=[async_inference_model_error, polling_timeout_error],
    )

    ims.s3_client.put_object = Mock(name="put_object")

    return ims


def empty_sagemaker_session_with_null_failure_path():
    ims = Mock(name="sagemaker_session")
    ims.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    ims.sagemaker_runtime_client = Mock(name="sagemaker_runtime")
    ims.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    ims.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)

    ims.sagemaker_runtime_client.invoke_endpoint_async = Mock(
        name="invoke_endpoint_async",
        return_value={
            "OutputLocation": ASYNC_OUTPUT_LOCATION,
        },
    )

    polling_timeout_error = PollingTimeoutError(
        message="Inference could still be running",
        output_path=ASYNC_OUTPUT_LOCATION,
        seconds=DEFAULT_WAITER_CONFIG.delay * DEFAULT_WAITER_CONFIG.max_attempts,
    )

    ims.s3_client = Mock(name="s3_client")
    ims.s3_client.get_object = Mock(
        name="get_object",
        side_effect=[polling_timeout_error],
    )

    ims.s3_client.put_object = Mock(name="put_object")

    return ims


def empty_predictor():
    predictor = Mock(name="predictor")
    predictor.update_endpoint = Mock(name="update_endpoint")
    predictor.delete_endpoint = Mock(name="delete_endpoint")
    predictor.delete_model = Mock(name="delete_model")
    predictor.enable_data_capture = Mock(name="enable_data_capture")
    predictor.disable_data_capture = Mock(name="disable_data_capture")
    predictor.update_data_capture_config = Mock(name="update_data_capture_config")
    predictor.list_monitor = Mock(name="list_monitor")
    predictor.endpoint_context = Mock(name="endpoint_context")

    return predictor


def test_async_predict_call_pass_through():
    sagemaker_session = empty_sagemaker_session()
    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))

    result = predictor_async.predict_async(input_path=ASYNC_INPUT_LOCATION)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called
    assert sagemaker_session.sagemaker_client.describe_endpoint.not_called
    assert sagemaker_session.sagemaker_client.describe_endpoint_config.not_called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "InputLocation": ASYNC_INPUT_LOCATION,
        "EndpointName": ENDPOINT,
    }

    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.call_args
    assert kwargs == expected_request_args
    assert result.output_path == ASYNC_OUTPUT_LOCATION
    assert result.failure_path == ASYNC_FAILURE_LOCATION


def test_async_predict_call_with_data():
    sagemaker_session = empty_sagemaker_session()
    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))
    predictor_async.name = ASYNC_PREDICTOR
    data = DUMMY_DATA

    result = predictor_async.predict_async(data=data)
    assert sagemaker_session.s3_client.put_object.called

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called
    assert sagemaker_session.sagemaker_client.describe_endpoint.not_called
    assert sagemaker_session.sagemaker_client.describe_endpoint_config.not_called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "InputLocation": predictor_async._input_path,
        "EndpointName": ENDPOINT,
    }

    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.call_args
    assert kwargs == expected_request_args
    assert result.output_path == ASYNC_OUTPUT_LOCATION


def test_async_predict_call_with_data_and_input_path():
    sagemaker_session = empty_sagemaker_session()
    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))
    predictor_async.name = ASYNC_PREDICTOR
    data = DUMMY_DATA

    result = predictor_async.predict_async(data=data, input_path=ASYNC_INPUT_LOCATION)
    assert sagemaker_session.s3_client.put_object.called

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called
    assert sagemaker_session.sagemaker_client.describe_endpoint.not_called
    assert sagemaker_session.sagemaker_client.describe_endpoint_config.not_called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "InputLocation": ASYNC_INPUT_LOCATION,
        "EndpointName": ENDPOINT,
    }

    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.call_args
    assert kwargs == expected_request_args
    assert result.output_path == ASYNC_OUTPUT_LOCATION
    assert result.failure_path == ASYNC_FAILURE_LOCATION


def test_async_predict_call_with_data_and_input_and_null_failure_path():
    sagemaker_session = empty_sagemaker_session_with_null_failure_path()
    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))
    predictor_async.name = ASYNC_PREDICTOR
    data = DUMMY_DATA

    result = predictor_async.predict_async(data=data, input_path=ASYNC_INPUT_LOCATION)
    assert sagemaker_session.s3_client.put_object.called

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called
    assert sagemaker_session.sagemaker_client.describe_endpoint.not_called
    assert sagemaker_session.sagemaker_client.describe_endpoint_config.not_called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "InputLocation": ASYNC_INPUT_LOCATION,
        "EndpointName": ENDPOINT,
    }

    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.call_args
    assert kwargs == expected_request_args
    assert result.output_path == ASYNC_OUTPUT_LOCATION
    assert result.failure_path is None


def test_async_predict_call_verify_exceptions():
    sagemaker_session = empty_sagemaker_session()
    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))

    input_location = "s3://some-input-path"
    with pytest.raises(
        AsyncInferenceModelError, match="Model returned error: some error from model"
    ):
        predictor_async.predict(input_path=input_location)

    with pytest.raises(
        PollingTimeoutError,
        match=f"No result at {ASYNC_OUTPUT_LOCATION} after polling for "
        f"{DEFAULT_WAITER_CONFIG.delay*DEFAULT_WAITER_CONFIG.max_attempts}"
        f" seconds. Inference could still be running",
    ):
        predictor_async.predict(input_path=input_location, waiter_config=DEFAULT_WAITER_CONFIG)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called
    assert sagemaker_session.s3_client.get_object.called
    assert sagemaker_session.sagemaker_client.describe_endpoint.not_called
    assert sagemaker_session.sagemaker_client.describe_endpoint_config.not_called


def test_async_predict_call_verify_exceptions_with_null_failure_path():
    sagemaker_session = empty_sagemaker_session_with_null_failure_path()
    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))

    input_location = "s3://some-input-path"

    with pytest.raises(
        PollingTimeoutError,
        match=f"No result at {ASYNC_OUTPUT_LOCATION} after polling for "
        f"{DEFAULT_WAITER_CONFIG.delay*DEFAULT_WAITER_CONFIG.max_attempts}"
        f" seconds. Inference could still be running",
    ):
        predictor_async.predict(input_path=input_location, waiter_config=DEFAULT_WAITER_CONFIG)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called
    assert sagemaker_session.s3_client.get_object.called
    assert sagemaker_session.sagemaker_client.describe_endpoint.not_called
    assert sagemaker_session.sagemaker_client.describe_endpoint_config.not_called


def test_async_predict_call_pass_through_output_failure_paths():
    sagemaker_session = empty_sagemaker_session()

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=RETURN_VALUE)
    response_body.close = Mock("close", return_value=None)

    sagemaker_session.s3_client = Mock(name="s3_client")
    sagemaker_session.s3_client.get_object = Mock(
        name="get_object",
        return_value={"Body": response_body},
    )
    sagemaker_session.s3_client.put_object = Mock(name="put_object")

    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))

    sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async = Mock(
        name="invoke_endpoint_async",
        return_value={
            "OutputLocation": ASYNC_OUTPUT_LOCATION,
            "FailureLocation": ASYNC_FAILURE_LOCATION,
        },
    )

    input_location = "s3://some-input-path"

    result = predictor_async.predict(
        input_path=input_location,
    )

    assert result == RETURN_VALUE
    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called
    assert sagemaker_session.s3_client.get_waiter.assert_called_once_with("object_exists")
    assert sagemaker_session.sagemaker_client.describe_endpoint.not_called
    assert sagemaker_session.sagemaker_client.describe_endpoint_config.not_called


def test_async_predict_call_pass_through_with_null_failure_path():
    sagemaker_session = empty_sagemaker_session_with_null_failure_path()

    response_body = Mock("body")
    response_body.read = Mock("read", return_value=RETURN_VALUE)
    response_body.close = Mock("close", return_value=None)

    sagemaker_session.s3_client = Mock(name="s3_client")
    sagemaker_session.s3_client.get_object = Mock(
        name="get_object",
        return_value={"Body": response_body},
    )
    sagemaker_session.s3_client.put_object = Mock(name="put_object")

    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))

    sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async = Mock(
        name="invoke_endpoint_async",
        return_value={
            "OutputLocation": ASYNC_OUTPUT_LOCATION,
        },
    )

    input_location = "s3://some-input-path"

    result = predictor_async.predict(
        input_path=input_location,
    )

    assert result == RETURN_VALUE
    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called
    assert sagemaker_session.s3_client.get_waiter.assert_called_once_with("object_exists")
    assert sagemaker_session.sagemaker_client.describe_endpoint.not_called
    assert sagemaker_session.sagemaker_client.describe_endpoint_config.not_called


def test_predict_async_call_invalid_input():
    sagemaker_session = empty_sagemaker_session()
    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))

    with pytest.raises(
        ValueError,
        match="Please provide input data or input Amazon S3 location to use async prediction",
    ):
        predictor_async.predict_async()

    with pytest.raises(
        ValueError,
        match="Please provide input data or input Amazon S3 location to use async prediction",
    ):
        predictor_async.predict()


def test_predict_call_with_inference_id():
    sagemaker_session = empty_sagemaker_session()
    predictor_async = AsyncPredictor(Predictor(ENDPOINT, sagemaker_session))

    input_location = "s3://some-input-path"
    result = predictor_async.predict_async(input_path=input_location, inference_id=INFERENCE_ID)

    assert sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.called

    expected_request_args = {
        "Accept": DEFAULT_ACCEPT,
        "InputLocation": input_location,
        "EndpointName": ENDPOINT,
        "InferenceId": INFERENCE_ID,
    }

    call_args, kwargs = sagemaker_session.sagemaker_runtime_client.invoke_endpoint_async.call_args
    assert kwargs == expected_request_args

    assert result.output_path == ASYNC_OUTPUT_LOCATION
    assert result.failure_path == ASYNC_FAILURE_LOCATION


def test_update_endpoint_no_args():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.update_endpoint()
    predictor.update_endpoint.assert_called_with(
        initial_instance_count=None,
        instance_type=None,
        accelerator_type=None,
        model_name=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
        wait=True,
    )


def test_update_endpoint_all_args():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.update_endpoint()

    new_instance_count = 2
    new_instance_type = "ml.c4.xlarge"
    new_accelerator_type = "ml.eia1.medium"
    new_model_name = "new-model"
    new_tags = [{"Key": "foo", "Value": "bar"}]
    new_kms_key = "new-key"
    new_data_capture_config_dict = {}

    predictor_async.update_endpoint(
        initial_instance_count=new_instance_count,
        instance_type=new_instance_type,
        accelerator_type=new_accelerator_type,
        model_name=new_model_name,
        tags=new_tags,
        kms_key=new_kms_key,
        data_capture_config_dict=new_data_capture_config_dict,
        wait=False,
    )

    predictor.update_endpoint.assert_called_with(
        initial_instance_count=new_instance_count,
        instance_type=new_instance_type,
        accelerator_type=new_accelerator_type,
        model_name=new_model_name,
        tags=new_tags,
        kms_key=new_kms_key,
        data_capture_config_dict=new_data_capture_config_dict,
        wait=False,
    )


def test_delete_endpoint_with_config():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.delete_endpoint()
    predictor.delete_endpoint.assert_called_with(True)


def test_delete_endpoint_only():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.delete_endpoint(delete_endpoint_config=False)
    predictor.delete_endpoint.assert_called_with(False)


def test_delete_model():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.delete_model()
    predictor.delete_model.assert_called_with()


def test_enable_data_capture():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.enable_data_capture()
    predictor.enable_data_capture.assert_called_with()


def test_disable_data_capture():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.disable_data_capture()
    predictor.disable_data_capture.assert_called_with()


def test_update_data_capture_config():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    data_capture_config = Mock(name="data_capture_config")
    predictor_async.update_data_capture_config(data_capture_config=data_capture_config)
    predictor.update_data_capture_config.assert_called_with(data_capture_config)


def test_endpoint_context():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.endpoint_context()
    predictor.endpoint_context.assert_called_with()


def test_list_monitors():
    predictor = empty_predictor()
    predictor_async = AsyncPredictor(predictor=predictor)

    predictor_async.list_monitors()
    predictor.list_monitors.assert_called_with()
