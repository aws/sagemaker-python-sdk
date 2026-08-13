"""Unit tests for the 'evaluation' role type in iam_role_resolver."""

from unittest.mock import MagicMock, patch

from sagemaker.core.helper.iam_role_resolver import (
    resolve_and_validate_role,
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
        assert "model_eval" in IAM_POLICY_CONFIG

    def test_evaluation_trust_includes_sagemaker(self):
        """Evaluation role type should require sagemaker.amazonaws.com trust."""
        expected = _expected_trust_services("model_eval")
        assert expected == {"sagemaker.amazonaws.com"}

    def test_evaluation_trust_does_not_require_bedrock(self):
        """Evaluation role type should NOT require bedrock.amazonaws.com trust.

        The serverless evaluation backend runs as the SageMaker execution role
        and calls Bedrock APIs using the role's own credentials. Bedrock does
        not need to assume the role, so trust is not required.
        """
        expected = _expected_trust_services("model_eval")
        bedrock_service = "bedrock" + ".amazonaws.com"
        assert bedrock_service not in expected

    def test_evaluation_smoke_actions_include_bedrock(self):
        """Smoke test actions should include Bedrock evaluation actions."""
        actions = _get_smoke_test_actions("model_eval")
        assert "bedrock:CreateEvaluationJob" in actions
        assert "bedrock:GetEvaluationJob" in actions

    def test_evaluation_smoke_actions_include_bedrock_invoke(self):
        """Smoke test actions should include Bedrock invoke actions."""
        actions = _get_smoke_test_actions("model_eval")
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

        verdict, denied = _evaluate_permissions(mock_iam, "arn:aws:iam::123456789012:role/MyRole", "model_eval")
        assert verdict is False
        assert "bedrock:CreateEvaluationJob" in denied

    def test_resolve_passes_when_all_allowed(self):
        """Should pass when all evaluation permissions are allowed."""
        mock_iam = MagicMock()

        actions = _get_smoke_test_actions("model_eval")
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": a, "EvalDecision": "allowed"} for a in actions
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator

        verdict, denied = _evaluate_permissions(mock_iam, "arn:aws:iam::123456789012:role/MyRole", "model_eval")
        assert verdict is True
        assert denied == []

    def test_trust_check_passes_with_sagemaker(self):
        """Should pass when trust policy includes sagemaker.amazonaws.com."""
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

        result = _role_trusts_service(mock_iam, "arn:aws:iam::123456789012:role/MyRole", "model_eval")
        assert result is True

    def test_resolve_and_validate_passes_with_sagemaker_trust(self):
        """Full resolve_and_validate_role should pass with sagemaker trust only."""
        with patch("sagemaker.core.helper.iam_role_resolver._get_boto_session") as mock_session:
            mock_boto = MagicMock()
            mock_session.return_value = mock_boto

            mock_iam = MagicMock()
            mock_boto.client.return_value = mock_iam

            # Role exists with sagemaker trust only (no bedrock needed)
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

            # All permissions allowed
            actions = _get_smoke_test_actions("model_eval")
            paginator = MagicMock()
            paginator.paginate.return_value = [
                {"EvaluationResults": [{"EvalActionName": a, "EvalDecision": "allowed"} for a in actions]}
            ]
            mock_iam.get_paginator.return_value = paginator

            result = resolve_and_validate_role(
                provided_role="arn:aws:iam::123456789012:role/MyRole",
                role_type="model_eval",
            )
            assert result == "arn:aws:iam::123456789012:role/MyRole"

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
            actions = _get_smoke_test_actions("model_eval")
            paginator = MagicMock()
            paginator.paginate.return_value = [
                {"EvaluationResults": [{"EvalActionName": a, "EvalDecision": "allowed"} for a in actions]}
            ]
            mock_iam.get_paginator.return_value = paginator

            result = resolve_and_validate_role(
                provided_role="arn:aws:iam::123456789012:role/MyRole",
                role_type="model_eval",
            )
            assert result == "arn:aws:iam::123456789012:role/MyRole"
