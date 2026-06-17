"""Unit tests for IAM role auto-creation resolver."""
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.core.helper.iam_role_resolver import (
    RoleAutoCreationError,
    resolve_or_create_role,
    _load_policy_config,
    _replace_placeholders,
    _get_attached_policy_names,
    _ensure_policies_attached,
    _get_boto_session,
    _MAX_POLICY_VERSIONS,
)


class TestResolveOrCreateRole:
    """Tests for resolve_or_create_role()."""

    def test_explicit_role_arn_returned_directly(self):
        """If user provides a full ARN, return it without IAM calls."""
        arn = "arn:aws:iam::123456789012:role/MyRole"
        result = resolve_or_create_role(provided_role=arn, role_type="training")
        assert result == arn

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

    def test_each_type_has_required_fields(self):
        """Each role type has role_name, trust_policy, and policies."""
        config = _load_policy_config()
        for role_type in ("training", "serving", "pipeline", "feature_store", "bedrock"):
            assert "role_name" in config[role_type]
            assert "trust_policy" in config[role_type]
            assert "policies" in config[role_type]
            assert len(config[role_type]["policies"]) > 0

    def test_bedrock_role_trusts_bedrock_service(self):
        """The bedrock role type is trusted by the Bedrock service principal."""
        config = _load_policy_config()
        principal = config["bedrock"]["trust_policy"]["Statement"][0]["Principal"]
        assert principal["Service"] == "bedrock.amazonaws.com"

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
            == "arn:aws:iam::123456789012:role/*"
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
