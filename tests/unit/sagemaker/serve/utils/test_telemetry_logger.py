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
import unittest
from unittest.mock import Mock, patch
from sagemaker.serve import Mode, ModelServer
from sagemaker.serve.utils.telemetry_logger import (
    _send_telemetry,
    _capture_telemetry,
    _construct_url,
)
from sagemaker.serve.utils.exceptions import ModelBuilderException, LocalModelOutOfMemoryException
from sagemaker.user_agent import SDK_VERSION

MOCK_SESSION = Mock()
MOCK_FUNC_NAME = "Mock.deploy"
MOCK_DJL_CONTAINER = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/" "djl-inference:0.25.0-deepspeed0.11.0-cu118"
)
MOCK_TGI_CONTAINER = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04"
)
MOCK_HUGGINGFACE_ID = "meta-llama/Llama-2-7b-hf"
MOCK_EXCEPTION = LocalModelOutOfMemoryException("mock raise ex")
MOCK_ENDPOINT_ARN = "arn:aws:sagemaker:us-west-2:123456789012:endpoint/test"


class ModelBuilderMock:
    def __init__(self):
        self.serve_settings = Mock()
        self.sagemaker_session = MOCK_SESSION

    @_capture_telemetry(MOCK_FUNC_NAME)
    def mock_deploy(self, mock_exception_func=None):
        if mock_exception_func:
            mock_exception_func()


class TestTelemetryLogger(unittest.TestCase):
    @patch("sagemaker.serve.utils.telemetry_logger._requests_helper")
    @patch("sagemaker.serve.utils.telemetry_logger._get_accountId")
    def test_log_sucessfully(self, mocked_get_accountId, mocked_request_helper):
        MOCK_SESSION.boto_session.region_name = "ap-south-1"
        mocked_get_accountId.return_value = "testAccountId"
        _send_telemetry("someStatus", 1, MOCK_SESSION)
        mocked_request_helper.assert_called_with(
            "https://dev-exp-t-ap-south-1.s3.ap-south-1.amazonaws.com/"
            "telemetry?x-accountId=testAccountId&x-mode=1&x-status=someStatus",
            2,
        )

    @patch("sagemaker.serve.utils.telemetry_logger._get_accountId")
    def test_log_handle_exception(self, mocked_get_accountId):
        mocked_get_accountId.side_effect = Exception("Internal error")
        _send_telemetry("someStatus", 1, MOCK_SESSION)
        self.assertRaises(Exception)

    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    def test_capture_telemetry_decorator_djl_success(self, mock_send_telemetry):
        mock_model_builder = ModelBuilderMock()
        mock_model_builder.serve_settings.telemetry_opt_out = False
        mock_model_builder.image_uri = MOCK_DJL_CONTAINER
        mock_model_builder.model = MOCK_HUGGINGFACE_ID
        mock_model_builder.mode = Mode.LOCAL_CONTAINER
        mock_model_builder.model_server = ModelServer.DJL_SERVING
        mock_model_builder.sagemaker_session.endpoint_arn = MOCK_ENDPOINT_ARN

        mock_model_builder.mock_deploy()

        args = mock_send_telemetry.call_args.args
        latency = str(args[5]).split("latency=")[1]
        expected_extra_str = (
            f"{MOCK_FUNC_NAME}"
            "&x-modelServer=4"
            "&x-imageTag=djl-inference:0.25.0-deepspeed0.11.0-cu118"
            f"&x-sdkVersion={SDK_VERSION}"
            f"&x-modelName={MOCK_HUGGINGFACE_ID}"
            f"&x-endpointArn={MOCK_ENDPOINT_ARN}"
            f"&x-latency={latency}"
        )

        mock_send_telemetry.assert_called_once_with(
            "1", 2, MOCK_SESSION, None, None, expected_extra_str
        )

    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    def test_capture_telemetry_decorator_tgi_success(self, mock_send_telemetry):
        mock_model_builder = ModelBuilderMock()
        mock_model_builder.serve_settings.telemetry_opt_out = False
        mock_model_builder.image_uri = MOCK_TGI_CONTAINER
        mock_model_builder.model = MOCK_HUGGINGFACE_ID
        mock_model_builder.mode = Mode.LOCAL_CONTAINER
        mock_model_builder.model_server = ModelServer.TGI
        mock_model_builder.sagemaker_session.endpoint_arn = MOCK_ENDPOINT_ARN

        mock_model_builder.mock_deploy()

        args = mock_send_telemetry.call_args.args
        latency = str(args[5]).split("latency=")[1]
        expected_extra_str = (
            f"{MOCK_FUNC_NAME}"
            "&x-modelServer=6"
            "&x-imageTag=huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04"
            f"&x-sdkVersion={SDK_VERSION}"
            f"&x-modelName={MOCK_HUGGINGFACE_ID}"
            f"&x-endpointArn={MOCK_ENDPOINT_ARN}"
            f"&x-latency={latency}"
        )

        mock_send_telemetry.assert_called_once_with(
            "1", 2, MOCK_SESSION, None, None, expected_extra_str
        )

    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    def test_capture_telemetry_decorator_no_call_when_disabled(self, mock_send_telemetry):
        mock_model_builder = ModelBuilderMock()
        mock_model_builder.serve_settings.telemetry_opt_out = True
        mock_model_builder.image_uri = MOCK_DJL_CONTAINER
        mock_model_builder.model = MOCK_HUGGINGFACE_ID
        mock_model_builder.model_server = ModelServer.DJL_SERVING

        mock_model_builder.mock_deploy()

        assert not mock_send_telemetry.called

    @patch("sagemaker.serve.utils.telemetry_logger._send_telemetry")
    def test_capture_telemetry_decorator_handle_exception_success(self, mock_send_telemetry):
        mock_model_builder = ModelBuilderMock()
        mock_model_builder.serve_settings.telemetry_opt_out = False
        mock_model_builder.image_uri = MOCK_DJL_CONTAINER
        mock_model_builder.model = MOCK_HUGGINGFACE_ID
        mock_model_builder.mode = Mode.LOCAL_CONTAINER
        mock_model_builder.model_server = ModelServer.DJL_SERVING
        mock_model_builder.sagemaker_session.endpoint_arn = MOCK_ENDPOINT_ARN

        mock_exception = Mock()
        mock_exception_obj = MOCK_EXCEPTION
        mock_exception.side_effect = mock_exception_obj

        with self.assertRaises(ModelBuilderException) as _:
            mock_model_builder.mock_deploy(mock_exception)

        args = mock_send_telemetry.call_args.args
        latency = str(args[5]).split("latency=")[1]
        expected_extra_str = (
            f"{MOCK_FUNC_NAME}"
            "&x-modelServer=4"
            "&x-imageTag=djl-inference:0.25.0-deepspeed0.11.0-cu118"
            f"&x-sdkVersion={SDK_VERSION}"
            f"&x-modelName={MOCK_HUGGINGFACE_ID}"
            f"&x-endpointArn={MOCK_ENDPOINT_ARN}"
            f"&x-latency={latency}"
        )

        mock_send_telemetry.assert_called_once_with(
            "0",
            2,
            MOCK_SESSION,
            str(mock_exception_obj),
            mock_exception_obj.__class__.__name__,
            expected_extra_str,
        )

    def test_construct_url_with_failure_reason_and_extra_info(self):
        mock_accountId = "12345678910"
        mock_mode = Mode.LOCAL_CONTAINER
        mock_status = "0"
        mock_failure_reason = str(MOCK_EXCEPTION)
        mock_failure_type = MOCK_EXCEPTION.__class__.__name__
        mock_extra_info = "mock_extra_info"
        mock_region = "us-west-2"

        ret_url = _construct_url(
            accountId=mock_accountId,
            mode=mock_mode,
            status=mock_status,
            failure_reason=mock_failure_reason,
            failure_type=mock_failure_type,
            extra_info=mock_extra_info,
            region=mock_region,
        )

        expected_base_url = (
            f"https://dev-exp-t-{mock_region}.s3.{mock_region}.amazonaws.com/telemetry?"
            f"x-accountId={mock_accountId}"
            f"&x-mode={mock_mode}"
            f"&x-status={mock_status}"
            f"&x-failureReason={mock_failure_reason}"
            f"&x-failureType={mock_failure_type}"
            f"&x-extra={mock_extra_info}"
        )
        self.assertEquals(ret_url, expected_base_url)
