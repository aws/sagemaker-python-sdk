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
"""Tests for granular telemetry: TelemetryParamType, _extract_telemetry_params, _classify_error."""
from __future__ import absolute_import
import unittest
from unittest.mock import Mock, patch

from sagemaker.core.telemetry.telemetry_logging import (
    TelemetryParamType,
    _extract_telemetry_params,
    _attr_to_key,
    _classify_error,
    _telemetry_emitter,
)
from sagemaker.core.telemetry.constants import Feature


class TestAttrToKey(unittest.TestCase):
    """Tests for _attr_to_key helper."""

    def test_simple_attribute(self):
        assert _attr_to_key("networking") == "networking"

    def test_snake_case(self):
        assert _attr_to_key("training_type") == "trainingType"

    def test_leading_underscore(self):
        assert _attr_to_key("_model_name") == "modelName"

    def test_multiple_underscores(self):
        assert _attr_to_key("kms_key_id") == "kmsKeyId"

    def test_double_leading_underscore(self):
        assert _attr_to_key("__private_attr") == "privateAttr"


class TestExtractTelemetryParams(unittest.TestCase):
    """Tests for _extract_telemetry_params."""

    def _make_instance(self, **attrs):
        instance = Mock()
        for k, v in attrs.items():
            setattr(instance, k, v)
        return instance

    def test_returns_empty_when_no_params(self):
        instance = self._make_instance()
        assert _extract_telemetry_params(instance, {}, None) == ""

    def test_attr_value_emits_value(self):
        instance = self._make_instance(_model_name="llama-3-8b", training_type="LORA")
        result = _extract_telemetry_params(instance, {}, [
            ("_model_name", TelemetryParamType.ATTR_VALUE),
            ("training_type", TelemetryParamType.ATTR_VALUE),
        ])
        assert "&x-modelName=llama-3-8b" in result
        assert "&x-trainingType=LORA" in result

    def test_attr_value_skips_none(self):
        instance = self._make_instance(_model_name=None)
        result = _extract_telemetry_params(instance, {}, [
            ("_model_name", TelemetryParamType.ATTR_VALUE),
        ])
        assert "modelName" not in result

    def test_attr_exists_true(self):
        instance = self._make_instance(networking={"subnets": ["subnet-1"]})
        result = _extract_telemetry_params(instance, {}, [
            ("networking", TelemetryParamType.ATTR_EXISTS),
        ])
        assert "&x-hasNetworking=true" in result

    def test_attr_exists_false(self):
        instance = self._make_instance(networking=None)
        result = _extract_telemetry_params(instance, {}, [
            ("networking", TelemetryParamType.ATTR_EXISTS),
        ])
        assert "&x-hasNetworking=false" in result

    def test_attr_call_emits_return_value(self):
        instance = self._make_instance()
        instance._is_model_customization = Mock(return_value=True)
        result = _extract_telemetry_params(instance, {}, [
            ("_is_model_customization", TelemetryParamType.ATTR_CALL),
        ])
        assert "&x-isModelCustomization=True" in result
        instance._is_model_customization.assert_called_once()

    def test_attr_call_skips_on_exception(self):
        instance = self._make_instance()
        instance._is_model_customization = Mock(side_effect=RuntimeError("boom"))
        result = _extract_telemetry_params(instance, {}, [
            ("_is_model_customization", TelemetryParamType.ATTR_CALL),
        ])
        assert "isModelCustomization" not in result

    def test_kwarg_value_emits_value(self):
        instance = self._make_instance()
        result = _extract_telemetry_params(instance, {"instance_type": "ml.g5.2xlarge"}, [
            ("instance_type", TelemetryParamType.KWARG_VALUE),
        ])
        assert "&x-instanceType=ml.g5.2xlarge" in result

    def test_kwarg_value_skips_none(self):
        instance = self._make_instance()
        result = _extract_telemetry_params(instance, {"instance_type": None}, [
            ("instance_type", TelemetryParamType.KWARG_VALUE),
        ])
        assert "instanceType" not in result

    def test_kwarg_exists_true(self):
        instance = self._make_instance()
        result = _extract_telemetry_params(instance, {"update_endpoint": True}, [
            ("update_endpoint", TelemetryParamType.KWARG_EXISTS),
        ])
        assert "&x-hasUpdateEndpoint=true" in result

    def test_kwarg_exists_false(self):
        instance = self._make_instance()
        result = _extract_telemetry_params(instance, {}, [
            ("update_endpoint", TelemetryParamType.KWARG_EXISTS),
        ])
        assert "&x-hasUpdateEndpoint=false" in result

    def test_mixed_params(self):
        instance = self._make_instance(
            _model_name="llama-3",
            networking={"vpc": True},
            kms_key_id=None,
        )
        result = _extract_telemetry_params(instance, {"wait": True}, [
            ("_model_name", TelemetryParamType.ATTR_VALUE),
            ("networking", TelemetryParamType.ATTR_EXISTS),
            ("kms_key_id", TelemetryParamType.ATTR_EXISTS),
            ("wait", TelemetryParamType.KWARG_EXISTS),
        ])
        assert "&x-modelName=llama-3" in result
        assert "&x-hasNetworking=true" in result
        assert "&x-hasKmsKeyId=false" in result
        assert "&x-hasWait=true" in result

    def test_attr_type_emits_class_name(self):
        class HyperPodCompute:
            pass
        instance = self._make_instance(compute=HyperPodCompute())
        result = _extract_telemetry_params(instance, {}, [
            ("compute", TelemetryParamType.ATTR_TYPE),
        ])
        assert "&x-computeType=HyperPodCompute" in result

    def test_attr_type_skips_none(self):
        instance = self._make_instance(compute=None)
        result = _extract_telemetry_params(instance, {}, [
            ("compute", TelemetryParamType.ATTR_TYPE),
        ])
        assert "compute" not in result


class TestClassifyError(unittest.TestCase):
    """Tests for _classify_error."""

    def test_value_error_type(self):
        assert _classify_error(ValueError("anything")) == "validation_error"

    def test_type_error_type(self):
        assert _classify_error(TypeError("bad arg")) == "validation_error"

    def test_timeout_error_type(self):
        assert _classify_error(TimeoutError("timed out")) == "timeout_error"

    def test_connection_error_type(self):
        assert _classify_error(ConnectionError("refused")) == "network_error"

    def test_aws_validation_exception(self):
        e = Exception("ValidationException")
        e.response = {"ResponseMetadata": {"HTTPStatusCode": 400}}
        assert _classify_error(e) == "validation_error"

    def test_aws_access_denied(self):
        e = Exception("AccessDeniedException")
        e.response = {"ResponseMetadata": {"HTTPStatusCode": 403}}
        assert _classify_error(e) == "auth_error"

    def test_aws_resource_not_found(self):
        e = Exception("ResourceNotFoundException")
        e.response = {"ResponseMetadata": {"HTTPStatusCode": 404}}
        assert _classify_error(e) == "resource_not_found"

    def test_aws_throttling(self):
        e = Exception("ThrottlingException")
        e.response = {"ResponseMetadata": {"HTTPStatusCode": 429}}
        assert _classify_error(e) == "throttling_error"

    def test_aws_service_error(self):
        e = Exception("InternalServerError")
        e.response = {"ResponseMetadata": {"HTTPStatusCode": 500}}
        assert _classify_error(e) == "service_error"

    def test_eula_value_error_classified_as_validation(self):
        assert _classify_error(ValueError("accept_eula=True required")) == "validation_error"

    def test_unclassified_returns_class_name(self):
        assert _classify_error(RuntimeError("something unexpected")) == "runtimeerror"


class TestTelemetryEmitterWithParams(unittest.TestCase):
    """Tests for _telemetry_emitter decorator with telemetry_params."""

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_emitter_includes_telemetry_params_on_success(
        self, mock_resolve_config, mock_send_telemetry
    ):
        mock_resolve_config.return_value = False

        class FakeTrainer:
            sagemaker_session = Mock()
            sagemaker_session.sagemaker_config = None
            sagemaker_session.local_mode = False
            _model_name = "llama-3-8b"
            training_type = "LORA"
            networking = None

            @_telemetry_emitter(
                feature=Feature.MODEL_CUSTOMIZATION,
                func_name="FakeTrainer.train",
                telemetry_params=[
                    ("_model_name", TelemetryParamType.ATTR_VALUE),
                    ("training_type", TelemetryParamType.ATTR_VALUE),
                    ("networking", TelemetryParamType.ATTR_EXISTS),
                ],
            )
            def train(self):
                return "job_arn"

        trainer = FakeTrainer()
        trainer.train()

        extra = mock_send_telemetry.call_args.args[5]
        assert "x-modelName=llama-3-8b" in extra
        assert "x-trainingType=LORA" in extra
        assert "x-hasNetworking=false" in extra

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_emitter_includes_error_category_on_failure(
        self, mock_resolve_config, mock_send_telemetry
    ):
        mock_resolve_config.return_value = False

        class FakeTrainer:
            sagemaker_session = Mock()
            sagemaker_session.sagemaker_config = None
            sagemaker_session.local_mode = False

            @_telemetry_emitter(
                feature=Feature.MODEL_CUSTOMIZATION,
                func_name="FakeTrainer.train",
            )
            def train(self):
                raise ValueError("Invalid model name")

        trainer = FakeTrainer()
        with self.assertRaises(ValueError):
            trainer.train()

        extra = mock_send_telemetry.call_args.args[5]
        assert "x-errorCategory=validation_error" in extra

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_emitter_includes_kwarg_params(
        self, mock_resolve_config, mock_send_telemetry
    ):
        mock_resolve_config.return_value = False

        class FakeBuilder:
            sagemaker_session = Mock()
            sagemaker_session.sagemaker_config = None
            sagemaker_session.local_mode = False

            @_telemetry_emitter(
                feature=Feature.MODEL_CUSTOMIZATION,
                func_name="FakeBuilder.deploy",
                telemetry_params=[
                    ("update_endpoint", TelemetryParamType.KWARG_EXISTS),
                    ("instance_type", TelemetryParamType.KWARG_VALUE),
                ],
            )
            def deploy(self, update_endpoint=False, instance_type=None):
                return "endpoint_arn"

        builder = FakeBuilder()
        builder.deploy(update_endpoint=True, instance_type="ml.g5.2xlarge")

        extra = mock_send_telemetry.call_args.args[5]
        assert "x-hasUpdateEndpoint=true" in extra
        assert "x-instanceType=ml.g5.2xlarge" in extra


if __name__ == "__main__":
    unittest.main()
