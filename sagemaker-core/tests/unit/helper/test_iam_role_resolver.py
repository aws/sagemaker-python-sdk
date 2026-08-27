"""Unit tests for the read-only IAM role resolver (validate, never create)."""
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.core.helper.iam_role_resolver import (
    RoleAutoCreationError,
    RoleValidationError,
    resolve_and_validate_role,
    verify_hyperpod_connect_permissions,
    HYPERPOD_CLI_CONNECT_ACTIONS,
    _load_policy_config,
    _get_required_actions,
    _get_smoke_test_actions,
    _replace_placeholders,
    _get_boto_session,
    _simulate_denied_actions,
    _role_trusts_service,
    _trusted_services_in_document,
    _expected_trust_services,
)


def _make_session(caller_arn, account="123456789012"):
    """Build session/iam/sts mocks with the given caller identity."""
    mock_session = MagicMock()
    mock_iam = MagicMock()
    mock_sts = MagicMock()

    def client_factory(service, **kwargs):
        return mock_iam if service == "iam" else mock_sts

    mock_session.boto_session.client.side_effect = client_factory
    mock_sts.get_caller_identity.return_value = {"Arn": caller_arn, "Account": account}
    return mock_session, mock_iam, mock_sts


def _paginator_allowing(actions):
    """A simulate paginator that reports every action as allowed."""
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"EvaluationResults": [{"EvalActionName": a, "EvalDecision": "allowed"} for a in actions]}
    ]
    return paginator


def _trusted_doc():
    """A trust document that allows the SageMaker service to assume the role."""
    return {
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ]
    }


class TestResolveAndValidateRole:
    """Tests for resolve_and_validate_role() — read-only, never creates."""

    def test_explicit_role_arn_validated_and_returned(self):
        """A provided full ARN is validated (perms + trust) and returned."""
        arn = "arn:aws:iam::123456789012:role/MyRole"
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/Other/sess"
        )
        mock_iam.get_role.return_value = {"Role": {"AssumeRolePolicyDocument": _trusted_doc()}}
        mock_iam.get_paginator.return_value = _paginator_allowing(["s3:GetObject"])

        result = resolve_and_validate_role(
            provided_role=arn, role_type="training", sagemaker_session=mock_session
        )
        assert result == arn

    def test_explicit_role_arn_non_commercial_partition_returned(self):
        """A GovCloud/China role ARN is recognized and validated."""
        for arn in (
            "arn:aws-us-gov:iam::123456789012:role/MyRole",
            "arn:aws-cn:iam::123456789012:role/MyRole",
        ):
            mock_session, mock_iam, _ = _make_session(
                "arn:aws-us-gov:sts::123456789012:assumed-role/Other/sess"
            )
            mock_iam.get_role.return_value = {
                "Role": {"AssumeRolePolicyDocument": _trusted_doc()}
            }
            mock_iam.get_paginator.return_value = _paginator_allowing(["s3:GetObject"])
            assert (
                resolve_and_validate_role(
                    provided_role=arn,
                    role_type="hyperpod",
                    sagemaker_session=mock_session,
                )
                == arn
            )

    def test_explicit_role_name_resolved_to_arn(self):
        """A provided role name is looked up, validated, and its ARN returned."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/Other/sess"
        )
        mock_iam.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/MyRole",
                "AssumeRolePolicyDocument": _trusted_doc(),
            }
        }
        mock_iam.get_paginator.return_value = _paginator_allowing(["s3:GetObject"])

        result = resolve_and_validate_role(
            provided_role="MyRole",
            role_type="training",
            sagemaker_session=mock_session,
        )
        assert result == "arn:aws:iam::123456789012:role/MyRole"
        mock_iam.create_role.assert_not_called()

    def test_provided_role_lacking_permissions_raises(self):
        """A provided role (e.g. a read-only role) missing required perms raises."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/Other/sess"
        )
        # The provided role ARN is used directly (no get_role lookup needed).
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": "s3:PutObject", "EvalDecision": "implicitDeny"}
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator
        mock_iam.get_role.return_value = {
            "Role": {"AssumeRolePolicyDocument": _trusted_doc()}
        }

        with pytest.raises(RoleValidationError) as exc:
            resolve_and_validate_role(
                provided_role="arn:aws:iam::123456789012:role/ReadOnly",
                role_type="training",
                sagemaker_session=mock_session,
            )
        assert "s3:PutObject" in str(exc.value)
        mock_iam.create_role.assert_not_called()

    def test_explicit_role_name_not_found_raises(self):
        """If user provides a role name that doesn't exist, raise ValueError."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/Other/sess"
        )
        no_such_entity = type("NoSuchEntityException", (ClientError,), {})
        mock_iam.exceptions.NoSuchEntityException = no_such_entity
        mock_iam.get_role.side_effect = no_such_entity(
            {"Error": {"Code": "NoSuchEntity", "Message": "not found"}}, "GetRole"
        )

        with pytest.raises(ValueError, match="does not exist"):
            resolve_and_validate_role(
                provided_role="NonExistent",
                role_type="training",
                sagemaker_session=mock_session,
            )

    def test_caller_role_with_sufficient_permissions_returned(self):
        """A caller role that validates (perms allowed + trusted) is returned."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/NotebookRole/session-name"
        )
        mock_iam.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/NotebookRole",
                "AssumeRolePolicyDocument": _trusted_doc(),
            }
        }
        mock_iam.get_paginator.return_value = _paginator_allowing(["s3:GetObject"])

        result = resolve_and_validate_role(
            provided_role=None,
            role_type="training",
            sagemaker_session=mock_session,
        )
        assert result == "arn:aws:iam::123456789012:role/NotebookRole"
        mock_iam.create_role.assert_not_called()
        mock_iam.create_policy.assert_not_called()
        mock_iam.attach_role_policy.assert_not_called()

    def test_denied_permission_raises_validation_error_listing_actions(self):
        """A definitively denied *-resource action raises RoleValidationError listing it."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/WeakRole/sess"
        )
        mock_iam.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/WeakRole",
                "AssumeRolePolicyDocument": _trusted_doc(),
            }
        }
        # cloudwatch:PutMetricData is a *-resource action, so it is part of the
        # smoke-test gate; denying it must block.
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": "cloudwatch:PutMetricData", "EvalDecision": "implicitDeny"}
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator

        with pytest.raises(RoleValidationError) as exc:
            resolve_and_validate_role(
                provided_role=None,
                role_type="training",
                sagemaker_session=mock_session,
            )
        assert "cloudwatch:PutMetricData" in str(exc.value)

    def test_scoped_actions_excluded_from_validation_gate(self):
        """Resource-scoped actions (mlflow/hub/model-package) must NOT be simulated.

        Regression for the false-positive bug: simulating a resource-scoped action
        without ResourceArns returns implicitDeny and would wrongly block a usable
        role (including jobs that never use MLflow). Only *-resource smoke-test
        actions are simulated; the scoped ones never reach the paginator.
        """
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/NotebookRole/sess"
        )
        mock_iam.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/NotebookRole",
                "AssumeRolePolicyDocument": _trusted_doc(),
            }
        }
        captured = {}

        def paginate(**kwargs):
            captured["actions"] = kwargs.get("ActionNames", [])
            return [
                {
                    "EvaluationResults": [
                        {"EvalActionName": a, "EvalDecision": "allowed"}
                        for a in kwargs.get("ActionNames", [])
                    ]
                }
            ]

        paginator = MagicMock()
        paginator.paginate.side_effect = paginate
        mock_iam.get_paginator.return_value = paginator

        resolve_and_validate_role(
            provided_role=None, role_type="training", sagemaker_session=mock_session
        )

        simulated = set(captured["actions"])
        # Scoped actions are excluded from the gate...
        assert not any(a.startswith("sagemaker-mlflow:") for a in simulated)
        assert "sagemaker:DescribeHubContent" not in simulated
        assert "s3:GetObject" not in simulated  # S3 is scoped to S3_PLACEHOLDER
        # Repository-level ECR actions are scoped and excluded from the gate.
        assert "ecr:BatchGetImage" not in simulated
        assert "ecr:GetDownloadUrlForLayer" not in simulated
        assert "ecr:BatchCheckLayerAvailability" not in simulated
        # ...but *-resource smoke-test actions are included.
        assert "cloudwatch:PutMetricData" in simulated
        assert "ecr:GetAuthorizationToken" in simulated
        mock_iam.create_role.assert_not_called()

    def test_unverifiable_permissions_returns_role_with_warning(self, caplog):
        """When the caller can't self-simulate, return the role and warn (don't raise)."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/NotebookRole/sess"
        )
        mock_iam.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/NotebookRole",
                "AssumeRolePolicyDocument": _trusted_doc(),
            }
        }
        paginator = MagicMock()
        paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "SimulatePrincipalPolicy"
        )
        mock_iam.get_paginator.return_value = paginator

        with caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            result = resolve_and_validate_role(
                provided_role=None,
                role_type="training",
                sagemaker_session=mock_session,
            )

        assert result == "arn:aws:iam::123456789012:role/NotebookRole"
        mock_iam.create_role.assert_not_called()
        mock_iam.create_policy.assert_not_called()
        assert any(
            r.levelno == logging.WARNING and "Could not verify permissions" in r.getMessage()
            for r in caplog.records
        )

    def test_untrusted_role_raises_validation_error(self):
        """A role whose trust policy excludes the service raises RoleValidationError."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/Admin/sess"
        )
        mock_iam.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/Admin",
                # Trusted by the account root, not by the SageMaker service.
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "arn:aws:iam::123456789012:root"},
                            "Action": "sts:AssumeRole",
                        }
                    ]
                },
            }
        }
        mock_iam.get_paginator.return_value = _paginator_allowing(["s3:GetObject"])

        with pytest.raises(RoleValidationError) as exc:
            resolve_and_validate_role(
                provided_role=None,
                role_type="training",
                sagemaker_session=mock_session,
            )
        assert "trust policy" in str(exc.value)
        assert "sagemaker.amazonaws.com" in str(exc.value)

    def test_no_resolvable_caller_role_raises(self):
        """An IAM user / root (no backing role) raises RoleValidationError."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:iam::123456789012:user/dev-user"
        )
        with pytest.raises(RoleValidationError) as exc:
            resolve_and_validate_role(
                provided_role=None,
                role_type="training",
                sagemaker_session=mock_session,
            )
        assert "No IAM role could be resolved" in str(exc.value)
        mock_iam.create_role.assert_not_called()

    def test_invalid_role_type_raises(self):
        """Invalid role_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid role_type"):
            resolve_and_validate_role(provided_role=None, role_type="invalid")

    def test_never_creates_or_mutates_iam(self):
        """Across success and failure paths, no IAM write API is ever called."""
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/NotebookRole/sess"
        )
        mock_iam.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/NotebookRole",
                "AssumeRolePolicyDocument": _trusted_doc(),
            }
        }
        mock_iam.get_paginator.return_value = _paginator_allowing(["s3:GetObject"])

        resolve_and_validate_role(
            provided_role=None, role_type="training", sagemaker_session=mock_session
        )

        for write_method in (
            "create_role",
            "create_policy",
            "attach_role_policy",
            "put_role_policy",
            "tag_role",
            "create_policy_version",
        ):
            getattr(mock_iam, write_method).assert_not_called()

    def test_hyperpod_role_type_is_accepted(self):
        """The 'hyperpod' role type validates and returns an explicit ARN."""
        arn = "arn:aws:iam::123456789012:role/MyHyperPodRole"
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/Other/sess"
        )
        mock_iam.get_role.return_value = {"Role": {"AssumeRolePolicyDocument": _trusted_doc()}}
        mock_iam.get_paginator.return_value = _paginator_allowing(["s3:GetObject"])
        result = resolve_and_validate_role(
            provided_role=arn, role_type="hyperpod", sagemaker_session=mock_session
        )
        assert result == arn

    def test_get_boto_session_uses_provided_session(self):
        """When a sagemaker_session is given, its boto_session is reused."""
        mock_session = MagicMock()
        assert _get_boto_session(mock_session) is mock_session.boto_session

    def test_get_boto_session_falls_back_to_core_session(self):
        """Without a session, it falls back to a SageMaker core Session, not boto3."""
        with patch("sagemaker.core.helper.session_helper.Session") as mock_session_cls:
            result = _get_boto_session(None)
        assert result is mock_session_cls.return_value.boto_session


class TestResolverDoesNotExposeWriteApi:
    """The resolver module must not re-introduce auto-creation."""

    def test_resolve_or_create_role_removed(self):
        import sagemaker.core.helper.iam_role_resolver as r

        assert not hasattr(r, "resolve_or_create_role")

    def test_no_write_helpers_on_resolver(self):
        import sagemaker.core.helper.iam_role_resolver as r

        for name in (
            "_ensure_policies_attached",
            "_ensure_role_tagged",
            "_update_policy_document",
            "_ensure_hub_content_permission_on_reused_role",
        ):
            assert not hasattr(r, name), f"{name} must not live on the read-only resolver"


class TestPolicyConfig:
    """Tests for policy configuration loading (pure data)."""

    def test_load_policy_config_has_all_types(self):
        config = _load_policy_config()
        for role_type in (
            "training",
            "serving",
            "pipeline",
            "feature_store",
            "bedrock",
            "hyperpod",
        ):
            assert role_type in config

    def test_each_type_has_required_fields(self):
        config = _load_policy_config()
        for role_type in (
            "training",
            "serving",
            "pipeline",
            "feature_store",
            "bedrock",
            "hyperpod",
        ):
            assert "role_name" in config[role_type]
            assert "trust_policy" in config[role_type]
            assert "policies" in config[role_type]
            assert len(config[role_type]["policies"]) > 0

    def test_bedrock_role_trusts_bedrock_service(self):
        config = _load_policy_config()
        principal = config["bedrock"]["trust_policy"]["Statement"][0]["Principal"]
        assert principal["Service"] == "bedrock.amazonaws.com"

    def test_hyperpod_role_trusts_sagemaker_service(self):
        config = _load_policy_config()
        principal = config["hyperpod"]["trust_policy"]["Statement"][0]["Principal"]
        assert principal["Service"] == "sagemaker.amazonaws.com"

    def test_hyperpod_job_role_excludes_cluster_connect_permissions(self):
        """The job execution role must NOT carry caller-side CLI connect actions."""
        actions = set(_get_required_actions("hyperpod"))
        assert "sagemaker:DescribeCluster" not in actions
        assert "eks:DescribeCluster" not in actions
        assert "eks:AccessKubernetesApi" not in actions

    def test_hyperpod_job_role_has_runtime_permissions(self):
        actions = set(_get_required_actions("hyperpod"))
        assert "s3:GetObject" in actions
        assert "ecr:BatchGetImage" in actions
        assert "logs:CreateLogStream" in actions

    def test_training_role_has_describe_hub_content(self):
        """The training execution role must carry hub + hub-content permissions."""
        config = _load_policy_config()
        actions = set(_get_required_actions("training"))
        assert {
            "sagemaker:DescribeHubContent",
            "sagemaker:ListHubContents",
            "sagemaker:ListHubs",
            "sagemaker:DescribeHub",
        }.issubset(actions)
        stmt = config["training"]["policies"]["hub_content_policy"]["Statement"][0]
        assert stmt["Resource"] == [
            "arn:aws:sagemaker:*:*:hub/*",
            "arn:aws:sagemaker:*:*:hub-content/*",
        ]

    def test_training_role_has_model_package_permissions(self):
        config = _load_policy_config()
        actions = set(_get_required_actions("training"))
        assert {
            "sagemaker:CreateModelPackageGroup",
            "sagemaker:DescribeModelPackageGroup",
            "sagemaker:CreateModelPackage",
        }.issubset(actions)
        stmt = config["training"]["policies"]["model_package_policy"]["Statement"][0]
        assert stmt["Resource"] == [
            "arn:aws:sagemaker:*:*:model-package-group/*",
            "arn:aws:sagemaker:*:*:model-package/*",
        ]

    def test_training_role_has_evaluation_lineage_permissions(self):
        """Evaluation runs as the training execution role and records lineage +
        launches/tags a training job, so the training role carries CreateAction
        (lineage) and AddTags on training-job/* — the two actions the eval
        pipeline failed on.
        """
        config = _load_policy_config()
        actions = set(_get_required_actions("training"))
        assert {
            "sagemaker:CreateAction",
            "sagemaker:CreateArtifact",
            "sagemaker:AddAssociation",
        }.issubset(actions)
        statements = config["training"]["policies"]["evaluation_policy"]["Statement"]
        lineage_stmt, training_job_stmt = statements[0], statements[1]
        # Lineage actions scoped to action/artifact/context resources.
        assert lineage_stmt["Resource"] == [
            "arn:aws:sagemaker:*:*:action/*",
            "arn:aws:sagemaker:*:*:artifact/*",
            "arn:aws:sagemaker:*:*:context/*",
        ]
        # AddTags on the training job the eval pipeline launches (the failing call).
        assert training_job_stmt["Resource"] == "arn:aws:sagemaker:*:*:training-job/*"
        assert "sagemaker:AddTags" in training_job_stmt["Action"]
        assert "sagemaker:CreateTrainingJob" in training_job_stmt["Action"]

    def test_training_role_has_mlflow_permissions(self):
        """Training/eval jobs log to managed MLflow as the execution role, so the
        training role carries the sagemaker-mlflow actions (aligned to the Nova
        Forge SDK MLflowSageMaker policy).
        """
        config = _load_policy_config()
        actions = set(_get_required_actions("training"))
        assert {
            "sagemaker-mlflow:CreateRun",
            "sagemaker-mlflow:LogMetric",
            "sagemaker-mlflow:LogModel",
            "sagemaker-mlflow:CreateExperiment",
            "sagemaker-mlflow:CreateRegisteredModel",
            "sagemaker-mlflow:LogInputs",
        }.issubset(actions)
        statements = config["training"]["policies"]["mlflow_policy"]["Statement"]
        data_plane, control_plane = statements[0], statements[1]
        # Both statements scoped to MLflow resources only (app + classic server).
        for stmt in (data_plane, control_plane):
            assert stmt["Resource"] == [
                "arn:aws:sagemaker:*:*:mlflow-app/*",
                "arn:aws:sagemaker:*:*:mlflow-tracking-server/*",
            ]
        # Statement 1: data-plane logging via the sagemaker-mlflow namespace.
        assert all(a.startswith("sagemaker-mlflow:") for a in data_plane["Action"])
        # Statement 2: control-plane describe to resolve the tracking endpoint
        # from the provided MLflow ARN (both app and classic tracking server).
        assert set(control_plane["Action"]) == {
            "sagemaker:DescribeMlflowApp",
            "sagemaker:DescribeMlflowTrackingServer",
        }

    def test_mlflow_permissions_on_execution_role_types_only(self):
        """MLflow permissions belong on the job execution-role types that log to
        MLflow — training (covers eval) and hyperpod. serving/pipeline/
        feature_store/bedrock must not carry them.
        """
        for role_type in ("training", "hyperpod"):
            actions = set(_get_required_actions(role_type))
            assert any(
                a.startswith("sagemaker-mlflow:") for a in actions
            ), f"{role_type} should have sagemaker-mlflow permissions"
        for role_type in ("serving", "pipeline", "feature_store", "bedrock"):
            actions = set(_get_required_actions(role_type))
            assert not any(
                a.startswith("sagemaker-mlflow:") for a in actions
            ), f"{role_type} should not have sagemaker-mlflow permissions"

    def test_hyperpod_role_has_mlflow_permissions(self):
        """HyperPod fine-tuning jobs log to managed MLflow as the execution role
        (the MLflow tracking URI is injected into the HyperPod recipe), so the
        hyperpod role mirrors training's mlflow_policy.
        """
        config = _load_policy_config()
        statements = config["hyperpod"]["policies"]["mlflow_policy"]["Statement"]
        data_plane, control_plane = statements[0], statements[1]
        assert all(a.startswith("sagemaker-mlflow:") for a in data_plane["Action"])
        assert set(control_plane["Action"]) == {
            "sagemaker:DescribeMlflowApp",
            "sagemaker:DescribeMlflowTrackingServer",
        }
        for stmt in (data_plane, control_plane):
            assert stmt["Resource"] == [
                "arn:aws:sagemaker:*:*:mlflow-app/*",
                "arn:aws:sagemaker:*:*:mlflow-tracking-server/*",
            ]

    def test_hyperpod_role_has_lambda_invoke(self):
        """RLVR runs on HyperPod and can use a Lambda reward function the job
        invokes as the execution role, so the hyperpod role carries
        lambda:InvokeFunction (mirrors training).
        """
        actions = set(_get_required_actions("hyperpod"))
        assert "lambda:InvokeFunction" in actions

    def test_hyperpod_job_role_has_describe_hub_content(self):
        config = _load_policy_config()
        actions = set(_get_required_actions("hyperpod"))
        assert {
            "sagemaker:DescribeHubContent",
            "sagemaker:ListHubContents",
            "sagemaker:ListHubs",
            "sagemaker:DescribeHub",
        }.issubset(actions)
        stmt = config["hyperpod"]["policies"]["hub_content_policy"]["Statement"][0]
        assert stmt["Resource"] == [
            "arn:aws:sagemaker:*:*:hub/*",
            "arn:aws:sagemaker:*:*:hub-content/*",
        ]

    def test_all_role_types_have_source_account_placeholder(self):
        """Every (service-trusted) role's trust policy carries aws:SourceAccount."""
        config = _load_policy_config()
        for role_type in ("training", "serving", "pipeline", "feature_store", "bedrock"):
            statement = config[role_type]["trust_policy"]["Statement"][0]
            condition = statement.get("Condition", {})
            assert (
                condition.get("StringEquals", {}).get("aws:SourceAccount")
                == "ACCOUNT_PLACEHOLDER"
            ), f"{role_type} trust policy missing aws:SourceAccount placeholder"


class TestEcrPolicyScopedCorrectly:
    """Regression tests for ECR policy scoping (V2287033493).

    Repository-level ECR actions (BatchGetImage, GetDownloadUrlForLayer,
    BatchCheckLayerAvailability) must be scoped to repository ARNs, NOT
    Resource: "*". Only GetAuthorizationToken is account-level and belongs
    under "*". When all four are under "*", _get_smoke_test_actions includes
    the repo-level ones and SimulatePrincipalPolicy (called without ResourceArns)
    returns implicitDeny for roles with least-privilege ECR policies — a false
    positive that blocks deploys/training/pipelines.
    """

    ECR_REPO_ACTIONS = {
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability",
    }

    @pytest.mark.parametrize("role_type", ["training", "serving", "hyperpod"])
    def test_ecr_repo_actions_not_in_smoke_test(self, role_type):
        """Repository-level ECR actions must NOT appear in the smoke test set."""
        smoke_actions = set(_get_smoke_test_actions(role_type))
        overlap = smoke_actions & self.ECR_REPO_ACTIONS
        assert not overlap, (
            f"role_type={role_type}: repo-level ECR actions {overlap} should not be "
            f"in smoke test (would cause false implicitDeny)"
        )

    @pytest.mark.parametrize("role_type", ["training", "serving", "hyperpod"])
    def test_ecr_get_authorization_token_in_smoke_test(self, role_type):
        """GetAuthorizationToken is account-level and SHOULD be in the smoke test."""
        smoke_actions = set(_get_smoke_test_actions(role_type))
        assert "ecr:GetAuthorizationToken" in smoke_actions

    @pytest.mark.parametrize("role_type", ["training", "serving", "hyperpod"])
    def test_ecr_repo_actions_scoped_to_repository_arn(self, role_type):
        """Repository-level ECR actions must have a repository/* resource scope."""
        config = _load_policy_config()
        ecr_stmts = config[role_type]["policies"]["ecr_policy"]["Statement"]
        # Find the statement containing BatchGetImage
        repo_stmt = None
        for stmt in ecr_stmts:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            if "ecr:BatchGetImage" in actions:
                repo_stmt = stmt
                break
        assert repo_stmt is not None, "No statement with ecr:BatchGetImage found"
        resource = repo_stmt["Resource"]
        assert resource == "arn:aws:ecr:*:*:repository/*", (
            f"Expected repository/* scope, got: {resource}"
        )

    @pytest.mark.parametrize("role_type", ["training", "serving", "hyperpod"])
    def test_ecr_get_authorization_token_resource_is_wildcard(self, role_type):
        """GetAuthorizationToken must remain under Resource: '*'."""
        config = _load_policy_config()
        ecr_stmts = config[role_type]["policies"]["ecr_policy"]["Statement"]
        auth_stmt = None
        for stmt in ecr_stmts:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            if "ecr:GetAuthorizationToken" in actions:
                auth_stmt = stmt
                break
        assert auth_stmt is not None, "No statement with ecr:GetAuthorizationToken found"
        assert auth_stmt["Resource"] == "*"

    @pytest.mark.parametrize("role_type", ["training", "serving", "hyperpod"])
    def test_all_ecr_actions_still_in_required_actions(self, role_type):
        """All four ECR actions must remain in the full required actions list."""
        all_actions = set(_get_required_actions(role_type))
        expected = self.ECR_REPO_ACTIONS | {"ecr:GetAuthorizationToken"}
        assert expected.issubset(all_actions), (
            f"Missing ECR actions from required set: {expected - all_actions}"
        )

    def test_least_privilege_ecr_role_passes_validation(self):
        """A role with ECR permissions scoped to specific repos must not be blocked.

        This is the actual customer scenario from V2287033493: the customer's role
        grants BatchGetImage/GetDownloadUrlForLayer/BatchCheckLayerAvailability on
        specific repository ARNs, not '*'. The smoke test should only simulate
        GetAuthorizationToken (which the customer grants on '*'), so the validation
        should pass.
        """
        mock_session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/MyTrainingRole/sess"
        )
        mock_iam.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/MyTrainingRole",
                "AssumeRolePolicyDocument": _trusted_doc(),
            }
        }

        # Simulate: all smoke-test actions are allowed (customer has them on *)
        captured = {}

        def paginate(**kwargs):
            captured["actions"] = kwargs.get("ActionNames", [])
            return [
                {
                    "EvaluationResults": [
                        {"EvalActionName": a, "EvalDecision": "allowed"}
                        for a in kwargs["ActionNames"]
                    ]
                }
            ]

        paginator = MagicMock()
        paginator.paginate.side_effect = paginate
        mock_iam.get_paginator.return_value = paginator

        # Should succeed without raising
        result = resolve_and_validate_role(
            provided_role=None, role_type="training", sagemaker_session=mock_session
        )
        assert result == "arn:aws:iam::123456789012:role/MyTrainingRole"

        # Verify repo-level ECR actions were NOT simulated
        simulated = set(captured["actions"])
        assert "ecr:BatchGetImage" not in simulated
        assert "ecr:GetDownloadUrlForLayer" not in simulated
        assert "ecr:BatchCheckLayerAvailability" not in simulated
        # But GetAuthorizationToken WAS simulated
        assert "ecr:GetAuthorizationToken" in simulated


class TestReplacePlaceholders:
    """Tests for the pure-data _replace_placeholders() helper."""

    def test_s3_single_bucket(self):
        policies = {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "S3_PLACEHOLDER"}],
            }
        }
        result = _replace_placeholders(policies, s3_resource="my-bucket", kms_resource="*")
        assert result["s3_policy"]["Statement"][0]["Resource"] == [
            "arn:aws:s3:::my-bucket",
            "arn:aws:s3:::my-bucket/*",
        ]

    def test_s3_wildcard(self):
        policies = {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "S3_PLACEHOLDER"}],
            }
        }
        result = _replace_placeholders(policies, s3_resource="*", kms_resource="*")
        assert result["s3_policy"]["Statement"][0]["Resource"] == "*"

    def test_kms_with_account(self):
        policies = {
            "kms_policy": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["kms:Encrypt"], "Resource": "KMS_PLACEHOLDER"}],
            }
        }
        result = _replace_placeholders(
            policies, s3_resource="*", kms_resource="abc-123", account_id="123456789012"
        )
        assert result["kms_policy"]["Statement"][0]["Resource"] == [
            "arn:aws:kms:*:123456789012:key/abc-123"
        ]

    def test_s3_list(self):
        policies = {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "S3_PLACEHOLDER"}],
            }
        }
        result = _replace_placeholders(
            policies, s3_resource=["dataset-bucket", "output-bucket"], kms_resource="*"
        )
        assert result["s3_policy"]["Statement"][0]["Resource"] == [
            "arn:aws:s3:::dataset-bucket",
            "arn:aws:s3:::dataset-bucket/*",
            "arn:aws:s3:::output-bucket",
            "arn:aws:s3:::output-bucket/*",
        ]

    def test_passrole_scoped_to_account(self):
        policies = {
            "iam_passrole_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": ["iam:PassRole"], "Resource": "IAM_PASSROLE_PLACEHOLDER"}
                ],
            }
        }
        result = _replace_placeholders(
            policies, s3_resource="*", kms_resource="*", account_id="123456789012"
        )
        resource = result["iam_passrole_policy"]["Statement"][0]["Resource"]
        assert resource == "arn:aws:iam::123456789012:role/SageMaker-AutoRole-*"
        assert not resource.endswith("role/*")

    def test_rewrites_static_arn_partition(self):
        policies = {
            "cloudwatch_logs_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["logs:PutLogEvents"],
                        "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
                    }
                ],
            }
        }
        result = _replace_placeholders(
            policies, s3_resource="*", kms_resource="*", partition="aws-us-gov"
        )
        assert (
            result["cloudwatch_logs_policy"]["Statement"][0]["Resource"]
            == "arn:aws-us-gov:logs:*:*:log-group:/aws/sagemaker/*"
        )

    def test_list_with_wildcard_collapses(self):
        policies = {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "S3_PLACEHOLDER"}],
            }
        }
        result = _replace_placeholders(
            policies, s3_resource=["my-bucket", "*"], kms_resource="*"
        )
        assert result["s3_policy"]["Statement"][0]["Resource"] == "*"


class TestSimulateDeniedActions:
    """Tests for the shared _simulate_denied_actions() helper."""

    def _paginated_iam(self, decisions):
        mock_iam = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": a, "EvalDecision": d} for a, d in decisions
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator
        return mock_iam

    def test_returns_empty_when_all_allowed(self):
        mock_iam = self._paginated_iam(
            [("s3:GetObject", "allowed"), ("s3:PutObject", "allowed")]
        )
        denied = _simulate_denied_actions(
            mock_iam, "arn:aws:iam::123456789012:role/R", ["s3:GetObject", "s3:PutObject"]
        )
        assert denied == []

    def test_returns_denied_subset(self):
        mock_iam = self._paginated_iam(
            [("s3:GetObject", "allowed"), ("eks:DescribeCluster", "implicitDeny")]
        )
        denied = _simulate_denied_actions(
            mock_iam, "arn:aws:iam::123456789012:role/R",
            ["s3:GetObject", "eks:DescribeCluster"],
        )
        assert denied == ["eks:DescribeCluster"]

    def test_aggregates_across_pages(self):
        mock_iam = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"EvaluationResults": [{"EvalActionName": "a", "EvalDecision": "allowed"}]},
            {"EvaluationResults": [{"EvalActionName": "b", "EvalDecision": "implicitDeny"}]},
        ]
        mock_iam.get_paginator.return_value = paginator
        denied = _simulate_denied_actions(mock_iam, "arn:aws:iam::1:role/R", ["a", "b"])
        assert denied == ["b"]


class TestVerifyHyperPodConnectPermissions:
    """Tests for verify_hyperpod_connect_permissions() (caller-side CLI perms)."""

    def test_all_connect_actions_allowed_returns_true(self):
        session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/CallerRole/session"
        )
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/CallerRole"}
        }
        mock_iam.get_paginator.return_value = _paginator_allowing(
            HYPERPOD_CLI_CONNECT_ACTIONS
        )
        assert verify_hyperpod_connect_permissions(sagemaker_session=session) is True

    def test_denied_connect_action_returns_false_and_warns(self, caplog):
        session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/CallerRole/session"
        )
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/CallerRole"}
        }
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": "sagemaker:DescribeCluster", "EvalDecision": "allowed"},
                    {"EvalActionName": "eks:DescribeCluster", "EvalDecision": "implicitDeny"},
                    {"EvalActionName": "eks:AccessKubernetesApi", "EvalDecision": "allowed"},
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator

        with caplog.at_level(logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"):
            result = verify_hyperpod_connect_permissions(
                sagemaker_session=session, cluster_name="prod-cluster"
            )

        assert result is False
        assert any(
            "eks:DescribeCluster" in r.getMessage() and "prod-cluster" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_iam_user_caller_returns_none(self):
        session, mock_iam, _ = _make_session("arn:aws:iam::123456789012:user/dev-user")
        assert verify_hyperpod_connect_permissions(sagemaker_session=session) is None
        mock_iam.get_paginator.assert_not_called()

    def test_simulate_access_denied_returns_none(self):
        session, mock_iam, _ = _make_session(
            "arn:aws:sts::123456789012:assumed-role/CallerRole/session"
        )
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/CallerRole"}
        }
        paginator = MagicMock()
        paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no simulate"}},
            "SimulatePrincipalPolicy",
        )
        mock_iam.get_paginator.return_value = paginator
        assert verify_hyperpod_connect_permissions(sagemaker_session=session) is None


class TestRoleTrustsService:
    """Tests for the trust-policy check helpers."""

    def test_expected_trust_services_training(self):
        assert _expected_trust_services("training") == {"sagemaker.amazonaws.com"}

    def test_expected_trust_services_multi_principal(self):
        assert _expected_trust_services("feature_store") == {
            "sagemaker.amazonaws.com",
            "scheduler.amazonaws.com",
        }

    def test_trusted_services_collects_service_principals(self):
        doc = _trusted_doc()
        assert _trusted_services_in_document(doc) == {"sagemaker.amazonaws.com"}

    def test_trusted_services_ignores_non_assume_and_deny(self):
        doc = {
            "Statement": [
                {
                    "Effect": "Deny",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                },
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "events.amazonaws.com"},
                    "Action": "sts:TagSession",
                },
            ]
        }
        assert _trusted_services_in_document(doc) == set()

    def test_role_trusts_service_true(self):
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {"Role": {"AssumeRolePolicyDocument": _trusted_doc()}}
        assert _role_trusts_service(
            mock_iam, "arn:aws:iam::123456789012:role/MyRole", "training"
        ) is True

    def test_role_trusts_service_false_for_admin_role(self):
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "arn:aws:iam::123456789012:root"},
                            "Action": "sts:AssumeRole",
                        }
                    ]
                }
            }
        }
        assert _role_trusts_service(
            mock_iam, "arn:aws:iam::123456789012:role/Admin", "training"
        ) is False

    def test_role_trusts_service_url_encoded_document(self):
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {"AssumeRolePolicyDocument": json.dumps(_trusted_doc())}
        }
        assert _role_trusts_service(
            mock_iam, "arn:aws:iam::123456789012:role/MyRole", "training"
        ) is True

    def test_role_trusts_service_none_when_document_missing(self):
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {"Role": {}}
        assert _role_trusts_service(
            mock_iam, "arn:aws:iam::123456789012:role/MyRole", "training"
        ) is None

    def test_role_trusts_service_none_on_access_denied(self):
        mock_iam = MagicMock()
        mock_iam.get_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": ""}}, "GetRole"
        )
        assert _role_trusts_service(
            mock_iam, "arn:aws:iam::123456789012:role/MyRole", "training"
        ) is None


class TestBackwardCompatibleExceptions:
    """RoleAutoCreationError stays importable from the resolver path."""

    def test_role_auto_creation_error_importable(self):
        assert issubclass(RoleAutoCreationError, Exception)
