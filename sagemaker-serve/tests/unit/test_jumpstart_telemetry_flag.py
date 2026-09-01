"""Unit tests for the JumpStart flag on ModelBuilder build and deploy telemetry.

The flag lets usage analytics separate JumpStart deployments from other
``model_builder.build`` and ``model_builder.deploy`` events.
"""

import unittest
from unittest.mock import Mock, patch

from sagemaker.core.resources import Endpoint, Model
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer

TELEMETRY_MODULE = "sagemaker.core.telemetry.telemetry_logging"
JUMPSTART_MODEL_ID = "huggingface-llm-falcon-7b-bf16"


def _telemetry_extra(mock_send_telemetry):
    """Return the extra info string of the last telemetry request."""
    return mock_send_telemetry.call_args.args[5]


@patch(f"{TELEMETRY_MODULE}.resolve_value_from_config", return_value=False)
@patch(f"{TELEMETRY_MODULE}._send_telemetry_request")
class TestJumpStartTelemetryFlag(unittest.TestCase):
    """Tests for the x-isJumpstartModelId telemetry param."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.local_mode = False
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"

        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client

        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    def _make_builder(self, model):
        """Create a ModelBuilder that reports no built model."""
        builder = ModelBuilder(
            model=model,
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE,
        )
        builder.built_model = None
        return builder

    @patch("sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder")
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_serve_setting")
    def test_build_emits_true_for_jumpstart_model_id(
        self,
        mock_serve_setting,
        mock_build_single,
        mock_is_jumpstart,
        mock_send_telemetry,
        mock_resolve_config,
    ):
        """build() emits the flag as True for a JumpStart model ID."""
        mock_serve_setting.return_value = Mock()
        mock_build_single.return_value = Mock(spec=Model)
        mock_is_jumpstart.return_value = True

        self._make_builder(JUMPSTART_MODEL_ID).build()

        assert "&x-isJumpstartModelId=True" in _telemetry_extra(mock_send_telemetry)

    @patch("sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder")
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_serve_setting")
    def test_build_emits_false_for_other_model(
        self,
        mock_serve_setting,
        mock_build_single,
        mock_is_jumpstart,
        mock_send_telemetry,
        mock_resolve_config,
    ):
        """build() emits the flag as False for a model that is not from JumpStart."""
        mock_serve_setting.return_value = Mock()
        mock_build_single.return_value = Mock(spec=Model)
        mock_is_jumpstart.return_value = False

        self._make_builder(Mock()).build()

        assert "&x-isJumpstartModelId=False" in _telemetry_extra(mock_send_telemetry)

    @patch("sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.model_builder.ModelBuilder._deploy")
    def test_deploy_emits_true_for_jumpstart_model_id(
        self, mock_deploy, mock_is_jumpstart, mock_send_telemetry, mock_resolve_config
    ):
        """deploy() emits the flag as True for a JumpStart model ID."""
        mock_deploy.return_value = Mock(spec=Endpoint)
        mock_is_jumpstart.return_value = True

        builder = self._make_builder(JUMPSTART_MODEL_ID)
        builder.built_model = Mock(spec=Model)
        builder.instance_type = "ml.g5.2xlarge"
        builder.deploy(endpoint_name="test-endpoint", wait=False)

        assert "&x-isJumpstartModelId=True" in _telemetry_extra(mock_send_telemetry)

    @patch("sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.model_builder.ModelBuilder._deploy")
    def test_deploy_emits_false_for_other_model(
        self, mock_deploy, mock_is_jumpstart, mock_send_telemetry, mock_resolve_config
    ):
        """deploy() emits the flag as False for a model that is not from JumpStart."""
        mock_deploy.return_value = Mock(spec=Endpoint)
        mock_is_jumpstart.return_value = False

        builder = self._make_builder(Mock())
        builder.built_model = Mock(spec=Model)
        builder.instance_type = "ml.g5.2xlarge"
        builder.deploy(endpoint_name="test-endpoint", wait=False)

        assert "&x-isJumpstartModelId=False" in _telemetry_extra(mock_send_telemetry)

    @patch("sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id")
    @patch("sagemaker.serve.model_builder.ModelBuilder._deploy")
    def test_deploy_emits_the_flag_before_the_latency_param(
        self, mock_deploy, mock_is_jumpstart, mock_send_telemetry, mock_resolve_config
    ):
        """The flag arrives before x-latency, so a greedy field parser reads it."""
        mock_deploy.return_value = Mock(spec=Endpoint)
        mock_is_jumpstart.return_value = True

        builder = self._make_builder(JUMPSTART_MODEL_ID)
        builder.built_model = Mock(spec=Model)
        builder.instance_type = "ml.g5.2xlarge"
        builder.deploy(endpoint_name="test-endpoint", wait=False)

        extra = _telemetry_extra(mock_send_telemetry)
        assert extra.index("&x-isJumpstartModelId=") < extra.index("&x-latency=")


if __name__ == "__main__":
    unittest.main()
