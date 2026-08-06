"""Unit tests for the 'evaluation' role type in iam_role_resolver."""

import pytest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError

from sagemaker.core.helper.iam_role_resolver import (
    resolve_and_validate_role,
    RoleValidationError,
    _evaluate_permissions,
    _role_trusts_service,
    _get_smoke_test_actions,
    _expected_trust_services,
)


class TestEvaluationRoleType:
    """Tests for the 'evaluation' role type configuration."""

    def test_evaluation_role_type_exists(self):
        """The 'evaluation' role type should be recognized."""
        from sagemaker.core.helper.iam_policies import IAM_POLICY_CONFIG
        assert "evaluation" in IAM_POLICY_CONFIG

    def test_evaluation_trust_includes_bedrock(self):
        """Evaluation role type should require bedrock.amazonaws.com trust."""
        expected = _expected_trust_services("evaluation")
        assert "bedrock.amazonaws.com" in expected

    def test_evaluation_trust_includes_sagemaker(self):
        """Evaluation role type should require sagemaker.amazonaws.com trust."""
        expected = _expected_trust_services("evaluation")
        assert "sagemaker.amazonaws.com" in expected

    def test_evaluation_smoke_actions_include_bedrock(self):
        """Smoke test actions should include Bedrock evaluation actions."""
        actions = _get_smoke_test_actions("evaluation")
        assert "bedrock:CreateEvaluationJob" in actions
        assert "bedrock:GetEvaluationJob" in actions

    def test_evaluation_smoke_actions_include_bedrock_invoke(self):
        """Smoke test actions should include Bedrock invoke actions."""
        actions = _get_smoke_test_actions("evaluation")
        assert "bedrock:InvokeModel" in actions

    def test_resolve_raises_when_bedrock_permissions_denied(self):
        """Should raise RoleValidationError when Bedrock permissions are denied."""
        mock_iam = MagicMock()

        # Simulate: bedrock:CreateEvaluationJob denied
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": "bedrock:CreateEvaluationJob", "EvalDecision": "implicitDeny"},
                    {"EvalActionName": "bedrock:GetEvaluationJob", "EvalDecision": "allowed"},
                    {"EvalActionName": "bedrock:InvokeModel", "EvalDecision": "allowed"},
                    {"EvalActionName": "bedrock:InvokeModelWithResponseStream", "EvalDecision": "allowed"},
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator

        verdict, denied = _evaluate_permissions(mock_iam, "arn:aws:iam::123456789012:role/MyRole", "evaluation")
        assert verdict is False
        assert "bedrock:CreateEvaluationJob" in denied

    def test_resolve_passes_when_all_allowed(self):
        """Should pass when all evaluation permissions are allowed."""
        mock_iam = MagicMock()

        actions = _get_smoke_test_actions("evaluation")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": a, "EvalDecision": "allowed"} for a in actions
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator

        verdict, denied = _evaluate_permissions(mock_iam, "arn:aws:iam::123456789012:role/MyRole", "evaluation")
        assert verdict is True
        assert denied == []

    def test_trust_check_fails_without_bedrock(self):
        """Should return False when trust policy lacks bedrock.amazonaws.com."""
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }],
                }
            }
        }

        result = _role_trusts_service(mock_iam, "arn:aws:iam::123456789012:role/MyRole", "evaluation")
        assert result is False

    def test_trust_check_passes_with_bedrock(self):
        """Should return True when trust policy includes both services."""
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": ["sagemaker.amazonaws.com", "bedrock.amazonaws.com"]},
                        "Action": "sts:AssumeRole",
                    }],
                }
            }
        }

        result = _role_trusts_service(mock_iam, "arn:aws:iam::123456789012:role/MyRole", "evaluation")
        assert result is True

    def test_resolve_and_validate_raises_on_trust_failure(self):
        """Full resolve_and_validate_role should raise when trust fails."""
        with patch("sagemaker.core.helper.iam_role_resolver._get_boto_session") as mock_session:
            mock_boto = MagicMock()
            mock_session.return_value = mock_boto

            mock_iam = MagicMock()
            mock_boto.client.return_value = mock_iam

            # Role exists
            mock_iam.get_role.return_value = {
                "Role": {
                    "Arn": "arn:aws:iam::123456789012:role/MyRole",
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [{
                            "Effect": "Allow",
                            "Principal": {"Service": "sagemaker.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }],
                    }
                }
            }

            # Permissions all allowed
            actions = _get_smoke_test_actions("evaluation")
            paginator = MagicMock()
            paginator.paginate.return_value = [
                {"EvaluationResults": [{"EvalActionName": a, "EvalDecision": "allowed"} for a in actions]}
            ]
            mock_iam.get_paginator.return_value = paginator

            with pytest.raises(RoleValidationError, match="bedrock.amazonaws.com"):
                resolve_and_validate_role(
                    provided_role="arn:aws:iam::123456789012:role/MyRole",
                    role_type="evaluation",
                )

    def test_resolve_and_validate_passes_with_correct_role(self):
        """Full resolve_and_validate_role should pass with correct permissions and trust."""
        with patch("sagemaker.core.helper.iam_role_resolver._get_boto_session") as mock_session:
            mock_boto = MagicMock()
            mock_session.return_value = mock_boto

            mock_iam = MagicMock()
            mock_boto.client.return_value = mock_iam

            # Role exists with correct trust
            mock_iam.get_role.return_value = {
                "Role": {
                    "Arn": "arn:aws:iam::123456789012:role/MyRole",
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [{
                            "Effect": "Allow",
                            "Principal": {"Service": ["sagemaker.amazonaws.com", "bedrock.amazonaws.com"]},
                            "Action": "sts:AssumeRole",
                        }],
                    }
                }
            }

            # All permissions allowed
            actions = _get_smoke_test_actions("evaluation")
            paginator = MagicMock()
            paginator.paginate.return_value = [
                {"EvaluationResults": [{"EvalActionName": a, "EvalDecision": "allowed"} for a in actions]}
            ]
            mock_iam.get_paginator.return_value = paginator

            result = resolve_and_validate_role(
                provided_role="arn:aws:iam::123456789012:role/MyRole",
                role_type="evaluation",
            )
            assert result == "arn:aws:iam::123456789012:role/MyRole"
