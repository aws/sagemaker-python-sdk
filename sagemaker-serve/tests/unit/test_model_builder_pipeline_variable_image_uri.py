"""Unit tests for ModelBuilder with a pipeline-variable image_uri (issue #5760)."""

import unittest
from unittest.mock import Mock, patch

import boto3

from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession, _ModelStepArguments
from sagemaker.serve.model_builder import ModelBuilder

ROLE_ARN = "arn:aws:iam::123456789012:role/TestRole"
TELEMETRY_MODULE = "sagemaker.core.telemetry.telemetry_logging"
NOVA_1P_IMAGE = "708977205387.dkr.ecr.us-east-1.amazonaws.com/nova-inference:latest"


def _mock_session():
    session = Mock()
    session.boto_region_name = "us-west-2"
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = None
    session.config = {}
    session.sagemaker_config = {}
    return session


class TestBuildValidationsWithPipelineVariableImageUri(unittest.TestCase):
    """_build_validations must not slice a pipeline-variable image_uri."""

    def test_image_only_pipeline_variable_is_passthrough(self):
        builder = ModelBuilder(
            image_uri=ParameterString(name="ServingImageUri"),
            role_arn=ROLE_ARN,
            sagemaker_session=_mock_session(),
        )

        builder._build_validations()

        self.assertTrue(builder._passthrough)

    def test_pipeline_variable_with_model_requires_model_server(self):
        builder = ModelBuilder(
            model=Mock(),
            image_uri=ParameterString(name="ServingImageUri"),
            role_arn=ROLE_ARN,
            sagemaker_session=_mock_session(),
        )

        with self.assertRaises(ValueError) as context:
            builder._build_validations()

        self.assertIn("Model_server must be set", str(context.exception))


@patch("sagemaker.serve.model_builder.resolve_nested_dict_value_from_config")
@patch("sagemaker.serve.model_builder.resolve_value_from_config")
@patch.object(ModelBuilder, "_init_sagemaker_session_if_does_not_exist")
@patch.object(ModelBuilder, "_prepare_container_def")
class TestCreateSageMakerModelNetworkIsolation(unittest.TestCase):
    """The Nova network-isolation check must tolerate a pipeline-variable image."""

    def _run(self, image_uri, mock_prepare, mock_resolve, mock_resolve_nested):
        mock_prepare.return_value = {"Image": image_uri, "Environment": {}}
        mock_resolve.side_effect = lambda value, *args, **kwargs: value
        mock_resolve_nested.side_effect = lambda value, *args, **kwargs: value

        session = Mock(spec=PipelineSession)
        builder = ModelBuilder(
            image_uri=image_uri,
            role_arn=ROLE_ARN,
            sagemaker_session=_mock_session(),
        )
        builder.sagemaker_session = session
        builder.model_name = "test-model"

        builder._create_sagemaker_model()

        session.create_model.assert_called_once()
        return session.create_model.call_args.kwargs

    def test_pipeline_variable_image_does_not_raise(
        self, mock_prepare, mock_init, mock_resolve, mock_resolve_nested
    ):
        image_uri = ParameterString(name="ServingImageUri")

        kwargs = self._run(image_uri, mock_prepare, mock_resolve, mock_resolve_nested)

        self.assertIs(kwargs["container_defs"]["Image"], image_uri)
        self.assertFalse(kwargs["enable_network_isolation"])

    def test_nova_1p_string_image_still_enables_network_isolation(
        self, mock_prepare, mock_init, mock_resolve, mock_resolve_nested
    ):
        kwargs = self._run(NOVA_1P_IMAGE, mock_prepare, mock_resolve, mock_resolve_nested)

        self.assertTrue(kwargs["enable_network_isolation"])


@patch(f"{TELEMETRY_MODULE}.resolve_value_from_config", return_value=False)
@patch(f"{TELEMETRY_MODULE}._send_telemetry_request")
@patch("sagemaker.serve.model_builder.resolve_and_validate_role", return_value=ROLE_ARN)
@patch.object(PipelineSession, "default_bucket", return_value="test-bucket")
class TestBuildWithPipelineVariableImageUri(unittest.TestCase):
    """End-to-end build() under a PipelineSession, as reported in #5760."""

    def test_build_keeps_pipeline_variable_in_create_model_request(self, *_mocks):
        image_uri = ParameterString(name="ServingImageUri")
        session = PipelineSession(
            boto_session=boto3.Session(
                region_name="us-west-2",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
            )
        )
        builder = ModelBuilder(image_uri=image_uri, role_arn=ROLE_ARN, sagemaker_session=session)

        step_args = builder.build()

        self.assertIsInstance(step_args, _ModelStepArguments)
        self.assertIs(step_args.create_model_request["PrimaryContainer"]["Image"], image_uri)


if __name__ == "__main__":
    unittest.main()
