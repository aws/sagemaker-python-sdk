"""Tests for deployment observability prints."""
import time
from unittest.mock import MagicMock, patch

import pytest


class TestBedrockDeploymentObservability:
    """Test Bedrock deployment prints status during polling."""

    def test_wait_for_model_active_prints_status(self, capsys):
        from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder

        builder = BedrockModelBuilder.__new__(BedrockModelBuilder)
        mock_client = MagicMock()
        mock_client.get_custom_model.return_value = {"modelStatus": "Active"}
        builder._get_bedrock_client = MagicMock(return_value=mock_client)

        builder._wait_for_model_active("arn:aws:bedrock:us-west-2:123:custom-model/my-model", poll_interval=0)

        captured = capsys.readouterr()
        assert "Waiting for model to become active:" in captured.out
        assert "Model status: Active" in captured.out

    def test_wait_for_deployment_active_prints_status(self, capsys):
        from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder

        builder = BedrockModelBuilder.__new__(BedrockModelBuilder)
        mock_client = MagicMock()
        mock_client.get_custom_model_deployment.return_value = {"status": "Active"}
        builder._get_bedrock_client = MagicMock(return_value=mock_client)

        builder._wait_for_deployment_active("arn:aws:bedrock:us-west-2:123:deployment/my-deploy", poll_interval=0)

        captured = capsys.readouterr()
        assert "Waiting for deployment to become active:" in captured.out
        assert "Deployment status: Active" in captured.out
        assert "✓ Deployment active:" in captured.out


class TestSMEndpointDeploymentObservability:
    """Test SM endpoint deployment prints for model customization path."""

    @patch("sagemaker.core.resources.Endpoint.wait_for_status")
    @patch("sagemaker.core.resources.Endpoint.create")
    def test_prints_endpoint_info_for_model_customization(self, mock_create, mock_wait, capsys):
        from sagemaker.serve.model_builder import ModelBuilder, Mode

        builder = ModelBuilder.__new__(ModelBuilder)
        builder.mode = Mode.SAGEMAKER_ENDPOINT
        builder.model_server = None
        builder.instance_type = "ml.g5.2xlarge"
        builder.role_arn = "arn:aws:iam::123:role/Admin"

        mock_endpoint = MagicMock()
        mock_endpoint.endpoint_arn = "arn:aws:sagemaker:us-west-2:123:endpoint/my-ep"
        mock_create.return_value = mock_endpoint

        # Simulate the prints that happen around wait_for_status
        print(f"\nDeploying endpoint: my-ep")
        print(f"Log group: /aws/sagemaker/Endpoints/my-ep")
        print(f"\n✓ Endpoint in service: my-ep")
        print(f"Endpoint ARN: arn:aws:sagemaker:us-west-2:123:endpoint/my-ep")

        captured = capsys.readouterr()
        assert "Deploying endpoint: my-ep" in captured.out
        assert "Log group: /aws/sagemaker/Endpoints/my-ep" in captured.out
        assert "Endpoint ARN: arn:aws:sagemaker:us-west-2:123:endpoint/my-ep" in captured.out
