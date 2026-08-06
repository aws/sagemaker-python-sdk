"""Unit tests for execution role permission validation for evaluation jobs."""

import pytest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError

from sagemaker.train.common_utils.role_permission_validator import (
    validate_evaluation_role_permissions,
)


class TestValidateEvaluationRolePermissions:
    """Tests for validate_evaluation_role_permissions."""

    def _make_session(self, iam_client):
        session = MagicMock()
        session.boto_session.client.return_value = iam_client
        return session

    def _mock_simulate_all_allowed(self, iam_client):
        """All actions return allowed."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": action, "EvalDecision": "allowed"}
                    for action in [
                        "bedrock:CreateEvaluationJob",
                        "bedrock:GetEvaluationJob",
                        "sagemaker-mlflow:GetExperimentByName",
                        "sagemaker-mlflow:CreateExperiment",
                        "sagemaker-mlflow:CreateRun",
                        "sagemaker-mlflow:LogBatch",
                    ]
                ]
            }
        ]
        iam_client.get_paginator.return_value = paginator

    def _mock_simulate_with_denied(self, iam_client, denied_actions):
        """Return denied for specified actions, allowed for the rest."""
        def paginate_side_effect(**kwargs):
            results = [
                {"EvalActionName": a, "EvalDecision": "implicitDeny" if a in denied_actions else "allowed"}
                for a in kwargs["ActionNames"]
            ]
            return [{"EvaluationResults": results}]

        paginator = MagicMock()
        paginator.paginate.side_effect = paginate_side_effect
        iam_client.get_paginator.return_value = paginator

    def _mock_role_trusts_bedrock(self, iam_client, trusts=True):
        """Mock _role_trusts_service result via get_role."""
        services = ["sagemaker.amazonaws.com"]
        if trusts:
            services.append("bedrock.amazonaws.com")
        iam_client.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": services},
                        "Action": "sts:AssumeRole",
                    }],
                }
            }
        }

    # --- Bedrock permission tests ---

    def test_passes_when_all_valid(self):
        iam_client = MagicMock()
        self._mock_simulate_all_allowed(iam_client)
        self._mock_role_trusts_bedrock(iam_client, trusts=True)
        session = self._make_session(iam_client)

        validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    def test_raises_when_create_evaluation_job_denied(self):
        iam_client = MagicMock()
        self._mock_simulate_with_denied(iam_client, ["bedrock:CreateEvaluationJob"])
        session = self._make_session(iam_client)

        with pytest.raises(ValueError, match="bedrock:CreateEvaluationJob"):
            validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    def test_raises_when_get_evaluation_job_denied(self):
        iam_client = MagicMock()
        self._mock_simulate_with_denied(iam_client, ["bedrock:GetEvaluationJob"])
        session = self._make_session(iam_client)

        with pytest.raises(ValueError, match="bedrock:GetEvaluationJob"):
            validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    def test_raises_when_both_bedrock_actions_denied(self):
        iam_client = MagicMock()
        self._mock_simulate_with_denied(iam_client, ["bedrock:CreateEvaluationJob", "bedrock:GetEvaluationJob"])
        session = self._make_session(iam_client)

        with pytest.raises(ValueError, match="bedrock:CreateEvaluationJob"):
            validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    # --- Trust policy tests ---

    def test_raises_when_trust_missing_bedrock(self):
        iam_client = MagicMock()
        self._mock_simulate_all_allowed(iam_client)
        self._mock_role_trusts_bedrock(iam_client, trusts=False)
        session = self._make_session(iam_client)

        with pytest.raises(ValueError, match="bedrock.amazonaws.com"):
            validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    # --- MLflow permission tests ---

    def test_raises_when_mlflow_denied_and_enabled(self):
        iam_client = MagicMock()
        self._mock_simulate_with_denied(iam_client, ["sagemaker-mlflow:GetExperimentByName"])
        self._mock_role_trusts_bedrock(iam_client, trusts=True)
        session = self._make_session(iam_client)

        with pytest.raises(ValueError, match="sagemaker-mlflow:GetExperimentByName"):
            validate_evaluation_role_permissions(
                "arn:aws:iam::123456789012:role/MyRole", session, mlflow_enabled=True
            )

    def test_skips_mlflow_check_when_not_enabled(self):
        iam_client = MagicMock()
        # Bedrock allowed, but MLflow would be denied - doesn't matter since not enabled
        self._mock_simulate_with_denied(iam_client, ["sagemaker-mlflow:GetExperimentByName"])
        self._mock_role_trusts_bedrock(iam_client, trusts=True)
        session = self._make_session(iam_client)

        # Should not raise since mlflow_enabled=False (default)
        validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    # --- Graceful degradation tests ---

    def test_warns_when_simulate_access_denied(self):
        iam_client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": ""}}, "SimulatePrincipalPolicy"
        )
        iam_client.get_paginator.return_value = paginator
        # Trust check also needs to handle gracefully
        iam_client.get_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": ""}}, "GetRole"
        )
        session = self._make_session(iam_client)

        # Should not raise
        validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    def test_warns_when_get_role_access_denied(self):
        iam_client = MagicMock()
        self._mock_simulate_all_allowed(iam_client)
        iam_client.get_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": ""}}, "GetRole"
        )
        session = self._make_session(iam_client)

        # Should not raise
        validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    def test_skips_when_session_is_none(self):
        with patch("boto3.Session", side_effect=Exception("no creds")):
            validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", None)

    # --- Error message quality tests ---

    def test_error_includes_doc_link(self):
        iam_client = MagicMock()
        self._mock_simulate_with_denied(iam_client, ["bedrock:CreateEvaluationJob"])
        session = self._make_session(iam_client)

        with pytest.raises(ValueError, match="model-customize-open-weight-prereq"):
            validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    def test_error_suggests_managed_policy(self):
        iam_client = MagicMock()
        self._mock_simulate_with_denied(iam_client, ["bedrock:CreateEvaluationJob"])
        session = self._make_session(iam_client)

        with pytest.raises(ValueError, match="AmazonSageMakerModelCustomizationCoreAccess"):
            validate_evaluation_role_permissions("arn:aws:iam::123456789012:role/MyRole", session)

    def test_mlflow_error_mentions_tracking_enabled(self):
        iam_client = MagicMock()
        self._mock_simulate_with_denied(iam_client, ["sagemaker-mlflow:CreateRun"])
        self._mock_role_trusts_bedrock(iam_client, trusts=True)
        session = self._make_session(iam_client)

        with pytest.raises(ValueError, match="MLflow experiment tracking is enabled"):
            validate_evaluation_role_permissions(
                "arn:aws:iam::123456789012:role/MyRole", session, mlflow_enabled=True
            )
