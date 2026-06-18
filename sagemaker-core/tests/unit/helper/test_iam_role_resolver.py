"""Unit tests for IAM role auto-creation resolver."""
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.core.helper.iam_role_resolver import (
    RoleAutoCreationError,
    resolve_or_create_role,
    verify_hyperpod_connect_permissions,
    HYPERPOD_CLI_CONNECT_ACTIONS,
    _load_policy_config,
    _get_required_actions,
    _replace_placeholders,
    _get_attached_policy_names,
    _ensure_policies_attached,
    _get_boto_session,
    _simulate_denied_actions,
    _MAX_POLICY_VERSIONS,
    _build_role_tags,
    _scope_trust_policy_to_account,
    _ensure_role_tagged,
    _SDK_ROLE_TAGS,
    _is_wildcard_scope,
    _role_uses_placeholder,
)


class TestResolveOrCreateRole:
    """Tests for resolve_or_create_role()."""

    def test_explicit_role_arn_returned_directly(self):
        """If user provides a full ARN, return it without IAM calls."""
        arn = "arn:aws:iam::123456789012:role/MyRole"
        result = resolve_or_create_role(provided_role=arn, role_type="training")
        assert result == arn

    def test_explicit_role_arn_non_commercial_partition_returned_directly(self):
        """A GovCloud/China role ARN is recognized and returned without IAM calls."""
        for arn in (
            "arn:aws-us-gov:iam::123456789012:role/MyRole",
            "arn:aws-cn:iam::123456789012:role/MyRole",
        ):
            assert resolve_or_create_role(provided_role=arn, role_type="hyperpod") == arn

    def test_explicit_role_name_resolved_to_arn(self):
        """If user provides a role name, look it up via IAM."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_session.boto_session.client.return_value = mock_iam
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/MyRole"}
        }

        result = resolve_or_create_role(
            provided_role="MyRole",
            role_type="training",
            sagemaker_session=mock_session,
        )
        assert result == "arn:aws:iam::123456789012:role/MyRole"
        mock_iam.get_role.assert_called_once_with(RoleName="MyRole")

    def test_explicit_role_name_not_found_raises(self):
        """If user provides a role name that doesn't exist, raise ValueError."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_session.boto_session.client.return_value = mock_iam

        no_such_entity = type("NoSuchEntityException", (ClientError,), {})
        mock_iam.exceptions.NoSuchEntityException = no_such_entity
        mock_iam.get_role.side_effect = no_such_entity(
            {"Error": {"Code": "NoSuchEntity", "Message": "not found"}}, "GetRole"
        )

        with pytest.raises(ValueError, match="does not exist"):
            resolve_or_create_role(
                provided_role="NonExistent",
                role_type="training",
                sagemaker_session=mock_session,
            )

    def test_caller_identity_is_role_returned(self):
        """If caller identity is an assumed role with sufficient permissions, reuse it.

        STS get_caller_identity returns an assumed-role ARN; the resolver resolves
        the canonical IAM role ARN via get_role (which also handles role paths) and
        reuses it when the permission simulation says it is sufficient.
        """
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            if service == "iam":
                return mock_iam
            return mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/NotebookRole/session-name",
            "Account": "123456789012",
        }
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/NotebookRole"}
        }
        # Permission simulation reports all required actions are allowed.
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"EvaluationResults": [{"EvalActionName": "s3:GetObject", "EvalDecision": "allowed"}]}
        ]
        mock_iam.get_paginator.return_value = mock_paginator

        result = resolve_or_create_role(
            provided_role=None,
            role_type="training",
            sagemaker_session=mock_session,
        )
        assert result == "arn:aws:iam::123456789012:role/NotebookRole"
        mock_iam.create_role.assert_not_called()

    def test_auto_creates_role_when_not_exists(self, caplog):
        """If role doesn't exist, create it with policies."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            if service == "iam":
                return mock_iam
            return mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/dev-user",
            "Account": "123456789012",
        }
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException(
            {"Error": {"Code": "NoSuchEntity"}}, "GetRole"
        )
        mock_iam.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (ClientError,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException(
            {"Error": {"Code": "NoSuchEntity", "Message": ""}}, "GetRole"
        )
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/test"}
        }

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            result = resolve_or_create_role(
                provided_role=None,
                role_type="training",
                sagemaker_session=mock_session,
            )

        assert "SageMaker-AutoRole-Training" in result
        mock_iam.create_role.assert_called_once()
        # The newly created role is surfaced to the user at WARNING level.
        assert any(
            r.levelno == logging.WARNING
            and "created a new IAM role" in r.getMessage()
            and "SageMaker-AutoRole-Training" in r.getMessage()
            for r in caplog.records
        )

    def test_reuses_existing_role(self):
        """If role already exists, reuse it without creating."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            if service == "iam":
                return mock_iam
            return mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/dev-user",
            "Account": "123456789012",
        }
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [
                {"PolicyName": "sagemaker-autorole-training-s3_policy"},
                {"PolicyName": "sagemaker-autorole-training-ecr_policy"},
                {"PolicyName": "sagemaker-autorole-training-cloudwatch_logs_policy"},
                {"PolicyName": "sagemaker-autorole-training-cloudwatch_metric_policy"},
                {"PolicyName": "sagemaker-autorole-training-ec2_policy"},
                {"PolicyName": "sagemaker-autorole-training-kms_policy"},
            ]
        }

        result = resolve_or_create_role(
            provided_role=None,
            role_type="training",
            sagemaker_session=mock_session,
        )

        assert result == "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"
        mock_iam.create_role.assert_not_called()

    def test_access_denied_raises_role_auto_creation_error(self):
        """If IAM permissions insufficient, raise clear error."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            if service == "iam":
                return mock_iam
            return mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/dev-user",
            "Account": "123456789012",
        }

        no_such_entity = type("NoSuchEntityException", (ClientError,), {})
        mock_iam.exceptions.NoSuchEntityException = no_such_entity
        mock_iam.get_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "not authorized"}},
            "GetRole",
        )

        with pytest.raises(RoleAutoCreationError, match="Cannot auto-create"):
            resolve_or_create_role(
                provided_role=None,
                role_type="training",
                sagemaker_session=mock_session,
            )

    def test_invalid_role_type_raises(self):
        """Invalid role_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid role_type"):
            resolve_or_create_role(provided_role=None, role_type="invalid")

    def test_hyperpod_role_type_is_accepted(self):
        """The 'hyperpod' role type resolves an explicit ARN without IAM calls."""
        arn = "arn:aws:iam::123456789012:role/MyHyperPodRole"
        result = resolve_or_create_role(provided_role=arn, role_type="hyperpod")
        assert result == arn

    def test_auto_creates_hyperpod_role_when_not_exists(self, caplog):
        """If the HyperPod auto-role doesn't exist, create it with its policies."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            if service == "iam":
                return mock_iam
            return mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/dev-user",
            "Account": "123456789012",
        }
        mock_iam.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (ClientError,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException(
            {"Error": {"Code": "NoSuchEntity", "Message": ""}}, "GetRole"
        )
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-HyperPod"}
        }
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/test"}
        }

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            result = resolve_or_create_role(
                provided_role=None,
                role_type="hyperpod",
                sagemaker_session=mock_session,
            )

        assert "SageMaker-AutoRole-HyperPod" in result
        mock_iam.create_role.assert_called_once()
        # The HyperPod role is trusted by the SageMaker service principal.
        trust_doc = json.loads(
            mock_iam.create_role.call_args.kwargs["AssumeRolePolicyDocument"]
        )
        assert (
            trust_doc["Statement"][0]["Principal"]["Service"] == "sagemaker.amazonaws.com"
        )

    def test_get_boto_session_uses_provided_session(self):
        """When a sagemaker_session is given, its boto_session is reused."""
        mock_session = MagicMock()
        assert _get_boto_session(mock_session) is mock_session.boto_session

    def test_get_boto_session_falls_back_to_core_session(self):
        """Without a session, it falls back to a SageMaker core Session, not boto3."""
        with patch("sagemaker.core.helper.session_helper.Session") as mock_session_cls:
            result = _get_boto_session(None)
        assert result is mock_session_cls.return_value.boto_session


class TestPolicyConfig:
    """Tests for policy configuration loading."""

    def test_load_policy_config_has_all_types(self):
        """Config file contains all expected role types."""
        config = _load_policy_config()
        assert "training" in config
        assert "serving" in config
        assert "pipeline" in config
        assert "feature_store" in config
        assert "bedrock" in config
        assert "hyperpod" in config

    def test_each_type_has_required_fields(self):
        """Each role type has role_name, trust_policy, and policies."""
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
        """The bedrock role type is trusted by the Bedrock service principal."""
        config = _load_policy_config()
        principal = config["bedrock"]["trust_policy"]["Statement"][0]["Principal"]
        assert principal["Service"] == "bedrock.amazonaws.com"

    def test_hyperpod_role_trusts_sagemaker_service(self):
        """The hyperpod job role runs as a SageMaker execution role on the cluster."""
        config = _load_policy_config()
        principal = config["hyperpod"]["trust_policy"]["Statement"][0]["Principal"]
        assert principal["Service"] == "sagemaker.amazonaws.com"

    def test_hyperpod_job_role_excludes_cluster_connect_permissions(self):
        """The job execution role must NOT carry caller-side CLI connect actions.

        The HyperPod CLI (connect-cluster / start-job) runs as the caller, not as
        the SageMaker-trusted job role, so baking eks:* / sagemaker:DescribeCluster
        onto this role would be dead weight on an identity that can't use them.
        Those permissions are verified on the caller via
        verify_hyperpod_connect_permissions() instead.
        """
        actions = set(_get_required_actions("hyperpod"))
        assert "sagemaker:DescribeCluster" not in actions
        assert "eks:DescribeCluster" not in actions
        assert "eks:AccessKubernetesApi" not in actions

    def test_hyperpod_job_role_has_runtime_permissions(self):
        """The hyperpod job role carries the S3/ECR/CloudWatch job-runtime actions."""
        actions = set(_get_required_actions("hyperpod"))
        assert "s3:GetObject" in actions
        assert "ecr:BatchGetImage" in actions
        assert "logs:CreateLogStream" in actions

    def test_training_role_has_describe_hub_content(self):
        """The training execution role must carry sagemaker:DescribeHubContent.

        Fine-tuning jobs reference a base model via a hub-content ARN, and the
        SageMaker service reads that hub content as this execution role when it
        creates the training job. Without DescribeHubContent the service fails
        CreateTrainingJob with an "Access denied to hub content" error.
        """
        config = _load_policy_config()
        actions = set(_get_required_actions("training"))
        assert "sagemaker:DescribeHubContent" in actions
        # Scoped to hub-content resources, not "*".
        stmt = config["training"]["policies"]["hub_content_policy"]["Statement"][0]
        assert stmt["Resource"] == "arn:aws:sagemaker:*:*:hub-content/*"

    def test_hyperpod_job_role_has_describe_hub_content(self):
        """The HyperPod job role also carries sagemaker:DescribeHubContent so it can
        resolve the base model's hub content.
        """
        config = _load_policy_config()
        actions = set(_get_required_actions("hyperpod"))
        assert "sagemaker:DescribeHubContent" in actions
        stmt = config["hyperpod"]["policies"]["hub_content_policy"]["Statement"][0]
        assert stmt["Resource"] == "arn:aws:sagemaker:*:*:hub-content/*"

    def test_replace_placeholders_s3(self):
        """S3 placeholder gets replaced with actual bucket ARN."""
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

    def test_replace_placeholders_wildcard(self):
        """Wildcard s3_resource results in '*' resource."""
        policies = {
            "s3_policy": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "S3_PLACEHOLDER"}],
            }
        }
        result = _replace_placeholders(policies, s3_resource="*", kms_resource="*")
        assert result["s3_policy"]["Statement"][0]["Resource"] == "*"

    def test_replace_placeholders_kms(self):
        """KMS placeholder gets replaced with key ARN."""
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

    def test_replace_placeholders_kms_account_defaults_to_wildcard(self):
        """Without an account_id, the KMS ARN account segment is '*'."""
        policies = {
            "kms_policy": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["kms:Encrypt"], "Resource": "KMS_PLACEHOLDER"}],
            }
        }
        result = _replace_placeholders(policies, s3_resource="*", kms_resource="abc-123")
        assert result["kms_policy"]["Statement"][0]["Resource"] == ["arn:aws:kms:*:*:key/abc-123"]

    def test_replace_placeholders_s3_list(self):
        """A list of S3 buckets expands to bucket + object ARNs for each."""
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

    def test_replace_placeholders_kms_list(self):
        """A list of KMS key IDs expands to a key ARN for each."""
        policies = {
            "kms_policy": {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": ["kms:Encrypt"], "Resource": "KMS_PLACEHOLDER"}],
            }
        }
        result = _replace_placeholders(
            policies, s3_resource="*", kms_resource=["key-1", "key-2"], account_id="123456789012"
        )
        assert result["kms_policy"]["Statement"][0]["Resource"] == [
            "arn:aws:kms:*:123456789012:key/key-1",
            "arn:aws:kms:*:123456789012:key/key-2",
        ]

    def test_replace_placeholders_passrole_scoped_to_account(self):
        """iam:PassRole placeholder scopes to roles in the caller's account."""
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
        assert (
            result["iam_passrole_policy"]["Statement"][0]["Resource"]
            == "arn:aws:iam::123456789012:role/SageMaker-AutoRole-*"
        )

    def test_replace_placeholders_rewrites_static_arn_partition(self):
        """Literal arn:aws: ARNs are rewritten to the caller's partition."""
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

    def test_replace_placeholders_static_arn_unchanged_for_aws_partition(self):
        """The default commercial partition leaves static ARNs untouched."""
        policies = {
            "p": {
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
        result = _replace_placeholders(policies, s3_resource="*", kms_resource="*")
        assert (
            result["p"]["Statement"][0]["Resource"]
            == "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
        )

    def test_replace_placeholders_list_with_wildcard_collapses(self):
        """A list containing '*' collapses to the '*' wildcard."""
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


class TestEnsurePoliciesAttached:
    """Tests for _ensure_policies_attached() policy create/update/attach behavior."""

    DOC_V1 = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "*"}],
    }
    DOC_V2 = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": "*"}
        ],
    }

    def _make_iam(self):
        mock_iam = MagicMock()
        mock_iam.exceptions.EntityAlreadyExistsException = type(
            "EntityAlreadyExistsException", (ClientError,), {}
        )
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
        return mock_iam

    def test_creates_and_attaches_new_policy(self):
        """A policy that does not yet exist is created and attached."""
        mock_iam = self._make_iam()

        _ensure_policies_attached(
            mock_iam, "MyRole", {"s3_policy": self.DOC_V1}, account_id="123456789012"
        )

        mock_iam.create_policy.assert_called_once()
        mock_iam.attach_role_policy.assert_called_once_with(
            RoleName="MyRole",
            PolicyArn="arn:aws:iam::123456789012:policy/MyRole-s3_policy",
        )
        mock_iam.create_policy_version.assert_not_called()

    def test_existing_policy_up_to_date_is_not_updated(self):
        """An existing policy whose document matches is left untouched."""
        mock_iam = self._make_iam()
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            {"Error": {"Code": "EntityAlreadyExists"}}, "CreatePolicy"
        )
        mock_iam.get_policy.return_value = {"Policy": {"DefaultVersionId": "v1"}}
        mock_iam.get_policy_version.return_value = {
            "PolicyVersion": {"Document": self.DOC_V1}
        }
        # Already attached, so no re-attach expected.
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [{"PolicyName": "myrole-s3_policy"}]
        }

        _ensure_policies_attached(
            mock_iam, "MyRole", {"s3_policy": self.DOC_V1}, account_id="123456789012"
        )

        mock_iam.create_policy_version.assert_not_called()
        mock_iam.attach_role_policy.assert_not_called()

    def test_existing_policy_drifted_is_updated(self):
        """An existing policy whose document drifted gets a new default version."""
        mock_iam = self._make_iam()
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            {"Error": {"Code": "EntityAlreadyExists"}}, "CreatePolicy"
        )
        mock_iam.get_policy.return_value = {"Policy": {"DefaultVersionId": "v1"}}
        mock_iam.get_policy_version.return_value = {
            "PolicyVersion": {"Document": self.DOC_V1}
        }
        mock_iam.list_policy_versions.return_value = {
            "Versions": [{"VersionId": "v1", "IsDefaultVersion": True}]
        }

        _ensure_policies_attached(
            mock_iam, "MyRole", {"s3_policy": self.DOC_V2}, account_id="123456789012"
        )

        mock_iam.create_policy_version.assert_called_once_with(
            PolicyArn="arn:aws:iam::123456789012:policy/MyRole-s3_policy",
            PolicyDocument=json.dumps(self.DOC_V2),
            SetAsDefault=True,
        )
        mock_iam.delete_policy_version.assert_not_called()

    def test_drifted_policy_prunes_oldest_version_at_limit(self):
        """When the version limit is reached, the oldest non-default version is pruned."""
        mock_iam = self._make_iam()
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            {"Error": {"Code": "EntityAlreadyExists"}}, "CreatePolicy"
        )
        mock_iam.get_policy.return_value = {"Policy": {"DefaultVersionId": "v5"}}
        mock_iam.get_policy_version.return_value = {
            "PolicyVersion": {"Document": self.DOC_V1}
        }
        mock_iam.list_policy_versions.return_value = {
            "Versions": [
                {"VersionId": f"v{i}", "IsDefaultVersion": (i == 5)}
                for i in range(1, _MAX_POLICY_VERSIONS + 1)
            ]
        }

        _ensure_policies_attached(
            mock_iam, "MyRole", {"s3_policy": self.DOC_V2}, account_id="123456789012"
        )

        mock_iam.delete_policy_version.assert_called_once_with(
            PolicyArn="arn:aws:iam::123456789012:policy/MyRole-s3_policy",
            VersionId="v1",
        )
        mock_iam.create_policy_version.assert_called_once()

    def test_created_policy_logs_warning(self, caplog):
        """Creating a policy emits a WARNING so the user is not surprised."""
        mock_iam = self._make_iam()

        with caplog.at_level(logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"):
            _ensure_policies_attached(
                mock_iam, "MyRole", {"s3_policy": self.DOC_V1}, account_id="123456789012"
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("created" in r.getMessage() and "MyRole-s3_policy" in r.getMessage() for r in warnings)

    def test_updated_policy_logs_warning(self, caplog):
        """Updating a drifted policy emits a WARNING."""
        mock_iam = self._make_iam()
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            {"Error": {"Code": "EntityAlreadyExists"}}, "CreatePolicy"
        )
        mock_iam.get_policy.return_value = {"Policy": {"DefaultVersionId": "v1"}}
        mock_iam.get_policy_version.return_value = {"PolicyVersion": {"Document": self.DOC_V1}}
        mock_iam.list_policy_versions.return_value = {
            "Versions": [{"VersionId": "v1", "IsDefaultVersion": True}]
        }

        with caplog.at_level(logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"):
            _ensure_policies_attached(
                mock_iam, "MyRole", {"s3_policy": self.DOC_V2}, account_id="123456789012"
            )

        assert any("updated" in r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING)

    def test_up_to_date_policy_logs_no_warning(self, caplog):
        """A policy that is already current and attached emits no WARNING."""
        mock_iam = self._make_iam()
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            {"Error": {"Code": "EntityAlreadyExists"}}, "CreatePolicy"
        )
        mock_iam.get_policy.return_value = {"Policy": {"DefaultVersionId": "v1"}}
        mock_iam.get_policy_version.return_value = {"PolicyVersion": {"Document": self.DOC_V1}}
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [{"PolicyName": "myrole-s3_policy"}]
        }

        with caplog.at_level(logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"):
            _ensure_policies_attached(
                mock_iam, "MyRole", {"s3_policy": self.DOC_V1}, account_id="123456789012"
            )

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]


class TestSimulateDeniedActions:
    """Tests for the shared _simulate_denied_actions() helper."""

    def _paginated_iam(self, decisions):
        """Build a mock IAM client whose simulate paginator yields `decisions`."""
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
        """Denied actions on a later page are not missed (no truncation)."""
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

    def _make_session(self, caller_arn, account="123456789012"):
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            return mock_iam if service == "iam" else mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {"Arn": caller_arn, "Account": account}
        return mock_session, mock_iam, mock_sts

    def test_all_connect_actions_allowed_returns_true(self):
        """When the caller role allows every connect action, return True."""
        session, mock_iam, _ = self._make_session(
            "arn:aws:sts::123456789012:assumed-role/CallerRole/session"
        )
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/CallerRole"}
        }
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": a, "EvalDecision": "allowed"}
                    for a in HYPERPOD_CLI_CONNECT_ACTIONS
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator

        assert verify_hyperpod_connect_permissions(sagemaker_session=session) is True

    def test_denied_connect_action_returns_false_and_warns(self, caplog):
        """A denied connect action returns False and warns the user."""
        session, mock_iam, _ = self._make_session(
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
        """An IAM user (no backing role) cannot be simulated → None, no raise."""
        session, mock_iam, _ = self._make_session(
            "arn:aws:iam::123456789012:user/dev-user"
        )
        assert verify_hyperpod_connect_permissions(sagemaker_session=session) is None
        mock_iam.get_paginator.assert_not_called()

    def test_simulate_access_denied_returns_none(self):
        """If the caller cannot self-simulate, return None rather than raising."""
        session, mock_iam, _ = self._make_session(
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


class TestTrustPolicySourceAccount:
    """Tests for the aws:SourceAccount confused-deputy mitigation (#1)."""

    def test_all_role_types_have_source_account_placeholder(self):
        """Every role's trust policy carries an aws:SourceAccount condition placeholder."""
        config = _load_policy_config()
        for role_type in ("training", "serving", "pipeline", "feature_store", "bedrock"):
            statement = config[role_type]["trust_policy"]["Statement"][0]
            condition = statement.get("Condition", {})
            assert (
                condition.get("StringEquals", {}).get("aws:SourceAccount")
                == "ACCOUNT_PLACEHOLDER"
            ), f"{role_type} trust policy missing aws:SourceAccount placeholder"

    def test_scope_trust_policy_substitutes_account_id(self):
        """ACCOUNT_PLACEHOLDER is replaced with the caller's account ID."""
        config = _load_policy_config()
        scoped = _scope_trust_policy_to_account(
            config["training"]["trust_policy"], "123456789012"
        )
        condition = scoped["Statement"][0]["Condition"]["StringEquals"]
        assert condition["aws:SourceAccount"] == "123456789012"

    def test_scope_trust_policy_does_not_mutate_input(self):
        """Substitution operates on a copy; the config constant is untouched."""
        config = _load_policy_config()
        original = config["training"]["trust_policy"]
        _scope_trust_policy_to_account(original, "123456789012")
        assert (
            original["Statement"][0]["Condition"]["StringEquals"]["aws:SourceAccount"]
            == "ACCOUNT_PLACEHOLDER"
        )

    def test_created_role_trust_policy_is_account_scoped(self):
        """The trust policy passed to create_role has the real account ID substituted."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            return mock_iam if service == "iam" else mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/dev-user",
            "Account": "123456789012",
        }
        mock_iam.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (ClientError,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException(
            {"Error": {"Code": "NoSuchEntity", "Message": ""}}, "GetRole"
        )
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            resolve_or_create_role(
                provided_role=None, role_type="training", sagemaker_session=mock_session
            )

        _, kwargs = mock_iam.create_role.call_args
        trust_doc = json.loads(kwargs["AssumeRolePolicyDocument"])
        condition = trust_doc["Statement"][0]["Condition"]["StringEquals"]
        assert condition["aws:SourceAccount"] == "123456789012"


class TestPassRoleScoping:
    """Tests for narrowing iam:PassRole to SDK auto-roles (#2)."""

    def test_passrole_scoped_to_auto_role_prefix(self):
        """iam:PassRole resolves to the SageMaker-AutoRole-* prefix, not all roles."""
        policies = {
            "iam_passrole_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["iam:PassRole"],
                        "Resource": "IAM_PASSROLE_PLACEHOLDER",
                    }
                ],
            }
        }
        result = _replace_placeholders(
            policies, s3_resource="*", kms_resource="*", account_id="123456789012"
        )
        resource = result["iam_passrole_policy"]["Statement"][0]["Resource"]
        assert resource == "arn:aws:iam::123456789012:role/SageMaker-AutoRole-*"
        assert not resource.endswith("role/*")

    def test_pipeline_passrole_resource_is_scoped(self):
        """The pipeline role's real PassRole statement resolves to the auto-role prefix."""
        config = _load_policy_config()
        resolved = _replace_placeholders(
            config["pipeline"]["policies"],
            s3_resource="*",
            kms_resource="*",
            account_id="123456789012",
        )
        passrole = resolved["iam_passrole_policy"]["Statement"][0]
        assert passrole["Resource"] == (
            "arn:aws:iam::123456789012:role/SageMaker-AutoRole-*"
        )
        # The PassedToService guard is preserved alongside the narrowed resource.
        assert (
            passrole["Condition"]["StringEquals"]["iam:PassedToService"]
            == "sagemaker.amazonaws.com"
        )


class TestRoleOwnershipTagging:
    """Tests for SDK ownership tagging of auto-created roles (#4)."""

    def test_build_role_tags_includes_ownership_and_type(self):
        """Tag list carries the ownership markers plus the role type."""
        tags = {t["Key"]: t["Value"] for t in _build_role_tags("training")}
        assert tags["CreatedBy"] == "sagemaker-python-sdk"
        assert tags["RoleType"] == "training"
        for key in _SDK_ROLE_TAGS:
            assert key in tags

    def test_create_role_is_tagged(self):
        """A freshly created role is tagged via create_role's Tags arg (Action 1)."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            return mock_iam if service == "iam" else mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/dev-user",
            "Account": "123456789012",
        }
        mock_iam.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (ClientError,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException(
            {"Error": {"Code": "NoSuchEntity", "Message": ""}}, "GetRole"
        )
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            resolve_or_create_role(
                provided_role=None, role_type="training", sagemaker_session=mock_session
            )

        _, kwargs = mock_iam.create_role.call_args
        tag_keys = {t["Key"] for t in kwargs["Tags"]}
        assert "CreatedBy" in tag_keys

    def test_ensure_role_tagged_applies_missing_tags(self):
        """An untagged existing role gets the SDK tags applied (Action 2)."""
        mock_iam = MagicMock()
        mock_iam.list_role_tags.return_value = {"Tags": []}

        _ensure_role_tagged(mock_iam, "SageMaker-AutoRole-Training", "training")

        mock_iam.tag_role.assert_called_once()
        _, kwargs = mock_iam.tag_role.call_args
        assert kwargs["RoleName"] == "SageMaker-AutoRole-Training"
        assert {"Key": "CreatedBy", "Value": "sagemaker-python-sdk"} in kwargs["Tags"]

    def test_ensure_role_tagged_noop_when_already_tagged(self):
        """A role that already carries the tags is not re-tagged."""
        mock_iam = MagicMock()
        mock_iam.list_role_tags.return_value = {
            "Tags": [
                {"Key": "CreatedBy", "Value": "sagemaker-python-sdk"},
                {"Key": "RoleType", "Value": "training"},
            ]
        }

        _ensure_role_tagged(mock_iam, "SageMaker-AutoRole-Training", "training")

        mock_iam.tag_role.assert_not_called()

    def test_ensure_role_tagged_swallows_client_error(self):
        """Tagging failures are non-fatal and do not raise."""
        mock_iam = MagicMock()
        mock_iam.list_role_tags.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "ListRoleTags"
        )

        # Should not raise.
        _ensure_role_tagged(mock_iam, "SageMaker-AutoRole-Training", "training")
        mock_iam.tag_role.assert_not_called()

    def test_reused_role_is_tag_checked(self):
        """Reusing an existing sufficient auto-role triggers a tag check (Action 2)."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            return mock_iam if service == "iam" else mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        # Caller is a user (not a role), so resolution falls through to the auto-role.
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:user/dev-user",
            "Account": "123456789012",
        }
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [
                {"PolicyName": "sagemaker-autorole-training-s3_policy"},
                {"PolicyName": "sagemaker-autorole-training-ecr_policy"},
                {"PolicyName": "sagemaker-autorole-training-cloudwatch_logs_policy"},
                {"PolicyName": "sagemaker-autorole-training-cloudwatch_metric_policy"},
                {"PolicyName": "sagemaker-autorole-training-ec2_policy"},
                {"PolicyName": "sagemaker-autorole-training-kms_policy"},
            ]
        }
        mock_iam.list_role_tags.return_value = {"Tags": []}

        resolve_or_create_role(
            provided_role=None, role_type="training", sagemaker_session=mock_session
        )

        mock_iam.list_role_tags.assert_called()
        mock_iam.tag_role.assert_called_once()


class TestBroadPermissionWarnings:
    """Tests for broad-permission warnings on auto-created roles (#3 and #6).

    AppSec accepted the broad S3/KMS ("*") grants and the read-only Glue catalog
    grant on the feature_store role, on the condition that the SDK warns the user
    about them. These tests assert those WARNINGs are emitted (and suppressed when
    the scope is actually narrowed).
    """

    def _make_auto_create_mocks(self, role_type, caller_arn=None):
        """Build session/iam/sts mocks that drive a fresh auto-role creation path."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            return mock_iam if service == "iam" else mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        # A plain IAM user (not a role) so resolution falls through to the auto-role.
        mock_sts.get_caller_identity.return_value = {
            "Arn": caller_arn or "arn:aws:iam::123456789012:user/dev-user",
            "Account": "123456789012",
        }
        mock_iam.exceptions.NoSuchEntityException = type(
            "NoSuchEntityException", (ClientError,), {}
        )
        mock_iam.get_role.side_effect = mock_iam.exceptions.NoSuchEntityException(
            {"Error": {"Code": "NoSuchEntity", "Message": ""}}, "GetRole"
        )
        config = _load_policy_config()
        mock_iam.create_role.return_value = {
            "Role": {
                "Arn": f"arn:aws:iam::123456789012:role/{config[role_type]['role_name']}"
            }
        }
        mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
        return mock_session, mock_iam

    def test_is_wildcard_scope(self):
        """Wildcard detection matches a bare '*' or any list containing '*'."""
        assert _is_wildcard_scope("*") is True
        assert _is_wildcard_scope(["*"]) is True
        assert _is_wildcard_scope(["my-bucket", "*"]) is True
        assert _is_wildcard_scope("my-bucket") is False
        assert _is_wildcard_scope(["a", "b"]) is False

    def test_role_uses_placeholder(self):
        """Placeholder detection reflects which resource placeholders a role uses."""
        config = _load_policy_config()
        assert _role_uses_placeholder(config["training"], "S3_PLACEHOLDER") is True
        assert _role_uses_placeholder(config["training"], "KMS_PLACEHOLDER") is True
        # The serving role has no KMS policy.
        assert _role_uses_placeholder(config["serving"], "KMS_PLACEHOLDER") is False

    def test_wildcard_s3_emits_warning(self, caplog):
        """Default s3_resource='*' warns about account-wide S3 access."""
        mock_session, mock_iam = self._make_auto_create_mocks("training")
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            resolve_or_create_role(
                provided_role=None, role_type="training", sagemaker_session=mock_session
            )
        assert any(
            r.levelno == logging.WARNING
            and "ALL S3 buckets" in r.getMessage()
            for r in caplog.records
        )

    def test_wildcard_kms_emits_warning(self, caplog):
        """Default kms_resource='*' warns about account-wide KMS access."""
        mock_session, mock_iam = self._make_auto_create_mocks("training")
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            resolve_or_create_role(
                provided_role=None, role_type="training", sagemaker_session=mock_session
            )
        assert any(
            r.levelno == logging.WARNING and "ALL KMS keys" in r.getMessage()
            for r in caplog.records
        )

    def test_scoped_s3_suppresses_s3_warning(self, caplog):
        """Passing explicit buckets suppresses the broad-S3 warning."""
        mock_session, mock_iam = self._make_auto_create_mocks("training")
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            resolve_or_create_role(
                provided_role=None,
                role_type="training",
                sagemaker_session=mock_session,
                s3_resource="my-bucket",
                kms_resource="my-key",
            )
        assert not any("ALL S3 buckets" in r.getMessage() for r in caplog.records)
        assert not any("ALL KMS keys" in r.getMessage() for r in caplog.records)

    def test_serving_role_does_not_warn_about_kms(self, caplog):
        """The serving role has no KMS policy, so no KMS warning is emitted."""
        mock_session, mock_iam = self._make_auto_create_mocks("serving")
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            resolve_or_create_role(
                provided_role=None, role_type="serving", sagemaker_session=mock_session
            )
        assert not any("ALL KMS keys" in r.getMessage() for r in caplog.records)

    def test_feature_store_role_warns_about_glue(self, caplog):
        """The feature_store role warns about account-wide Glue catalog read access."""
        mock_session, mock_iam = self._make_auto_create_mocks("feature_store")
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            resolve_or_create_role(
                provided_role=None,
                role_type="feature_store",
                sagemaker_session=mock_session,
            )
        assert any(
            r.levelno == logging.WARNING
            and "Glue catalog metadata" in r.getMessage()
            for r in caplog.records
        )

    def test_non_feature_store_role_does_not_warn_about_glue(self, caplog):
        """Only the feature_store role carries the Glue grant / warning."""
        mock_session, mock_iam = self._make_auto_create_mocks("training")
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            resolve_or_create_role(
                provided_role=None, role_type="training", sagemaker_session=mock_session
            )
        assert not any("Glue catalog metadata" in r.getMessage() for r in caplog.records)
