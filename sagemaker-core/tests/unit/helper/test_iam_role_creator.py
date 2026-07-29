"""Unit tests for IamRoleResolver (explicit, opt-in IAM role creation)."""
import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.core.helper.iam_role_resolver import IamRoleResolver, _MAX_POLICY_VERSIONS
from sagemaker.core.helper.iam_role_resolver import RoleAutoCreationError

ALL_ROLE_TYPES = ("training", "serving", "pipeline", "feature_store", "bedrock", "hyperpod")


def _make_resolver(account="123456789012", caller="arn:aws:iam::123456789012:user/dev"):
    """Build an IamRoleResolver wired to mock iam/sts clients."""
    mock_session = MagicMock()
    mock_iam = MagicMock()
    mock_sts = MagicMock()

    def client_factory(service, **kwargs):
        return mock_iam if service == "iam" else mock_sts

    mock_session.boto_session.client.side_effect = client_factory
    mock_sts.get_caller_identity.return_value = {"Arn": caller, "Account": account}
    mock_iam.exceptions.EntityAlreadyExistsException = type(
        "EntityAlreadyExistsException", (ClientError,), {}
    )
    mock_iam.list_attached_role_policies.return_value = {"AttachedPolicies": []}
    with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
        resolver = IamRoleResolver(sagemaker_session=mock_session)
    return resolver, mock_iam


class TestCreateExecutionRole:
    def test_creates_role_with_trust_and_tags(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.create_policy.return_value = {"Policy": {"Arn": "arn:aws:iam::1:policy/p"}}

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            arn = resolver.create_execution_role(role_type="training")

        assert arn.endswith("SageMaker-AutoRole-Training")
        mock_iam.create_role.assert_called_once()
        _, kwargs = mock_iam.create_role.call_args
        # Trust principal is the SageMaker service, account-scoped.
        trust = json.loads(kwargs["AssumeRolePolicyDocument"])
        assert trust["Statement"][0]["Principal"]["Service"] == "sagemaker.amazonaws.com"
        assert (
            trust["Statement"][0]["Condition"]["StringEquals"]["aws:SourceAccount"]
            == "123456789012"
        )
        # SDK ownership tags are applied at create time.
        tag_keys = {t["Key"] for t in kwargs["Tags"]}
        assert {"CreatedBy", "RoleType"}.issubset(tag_keys)

    def test_attaches_all_policies_for_role_type(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.create_policy.return_value = {"Policy": {"Arn": "arn:aws:iam::1:policy/p"}}

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            resolver.create_execution_role(role_type="training")

        from sagemaker.core.helper.iam_policies import IAM_POLICY_CONFIG

        expected = set(IAM_POLICY_CONFIG["training"]["policies"].keys())
        created = {
            c.kwargs["PolicyName"].split("SageMaker-AutoRole-Training-")[-1]
            for c in mock_iam.create_policy.call_args_list
        }
        assert expected.issubset(created)
        assert mock_iam.attach_role_policy.call_count == len(expected)

    @pytest.mark.parametrize("role_type", ALL_ROLE_TYPES)
    def test_create_for_each_role_type(self, role_type):
        resolver, mock_iam = _make_resolver()
        from sagemaker.core.helper.iam_policies import IAM_POLICY_CONFIG

        role_name = IAM_POLICY_CONFIG[role_type]["role_name"]
        mock_iam.create_role.return_value = {
            "Role": {"Arn": f"arn:aws:iam::123456789012:role/{role_name}"}
        }
        mock_iam.create_policy.return_value = {"Policy": {"Arn": "arn:aws:iam::1:policy/p"}}

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            arn = resolver.create_execution_role(role_type=role_type)
        assert arn.endswith(role_name)
        mock_iam.create_role.assert_called_once()

    def test_idempotent_when_role_exists(self):
        """If the role already exists, reuse it and (idempotently) refresh policies."""
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            {"Error": {"Code": "EntityAlreadyExists"}}, "CreateRole"
        )
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.list_role_tags.return_value = {"Tags": []}
        mock_iam.create_policy.return_value = {"Policy": {"Arn": "arn:aws:iam::1:policy/p"}}

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            arn = resolver.create_execution_role(role_type="training")

        assert arn.endswith("SageMaker-AutoRole-Training")
        # Reused role gets ownership tags ensured.
        mock_iam.tag_role.assert_called_once()

    def test_update_if_exists_false_skips_policy_attach(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            resolver.create_execution_role(role_type="training", update_if_exists=False)
        mock_iam.create_policy.assert_not_called()
        mock_iam.attach_role_policy.assert_not_called()

    def test_drifted_policy_creates_new_version(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        # Every create_policy says "already exists"; documents have drifted.
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            {"Error": {"Code": "EntityAlreadyExists"}}, "CreatePolicy"
        )
        mock_iam.get_policy.return_value = {"Policy": {"DefaultVersionId": "v1"}}
        mock_iam.get_policy_version.return_value = {
            "PolicyVersion": {"Document": {"different": "doc"}}
        }
        mock_iam.list_policy_versions.return_value = {
            "Versions": [{"VersionId": "v1", "IsDefaultVersion": True}]
        }

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            resolver.create_execution_role(role_type="training")

        assert mock_iam.create_policy_version.called

    def test_drift_prunes_oldest_version_at_limit(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.create_policy.side_effect = mock_iam.exceptions.EntityAlreadyExistsException(
            {"Error": {"Code": "EntityAlreadyExists"}}, "CreatePolicy"
        )
        mock_iam.get_policy.return_value = {"Policy": {"DefaultVersionId": "v5"}}
        mock_iam.get_policy_version.return_value = {
            "PolicyVersion": {"Document": {"different": "doc"}}
        }
        mock_iam.list_policy_versions.return_value = {
            "Versions": [
                {"VersionId": f"v{i}", "IsDefaultVersion": (i == 5)}
                for i in range(1, _MAX_POLICY_VERSIONS + 1)
            ]
        }

        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            resolver.create_execution_role(role_type="training")

        assert mock_iam.delete_policy_version.called

    def test_access_denied_raises_role_auto_creation_error(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "not authorized"}}, "CreateRole"
        )
        with pytest.raises(RoleAutoCreationError, match="Cannot create IAM role"):
            with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
                resolver.create_execution_role(role_type="training")

    def test_invalid_role_type_raises(self):
        resolver, _ = _make_resolver()
        with pytest.raises(ValueError, match="Invalid role_type"):
            resolver.create_execution_role(role_type="invalid")

    def test_custom_role_name(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/my-custom-role"}
        }
        mock_iam.create_policy.return_value = {"Policy": {"Arn": "arn:aws:iam::1:policy/p"}}
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"):
            resolver.create_execution_role(role_type="training", role_name="my-custom-role")
        _, kwargs = mock_iam.create_role.call_args
        assert kwargs["RoleName"] == "my-custom-role"

    def test_wildcard_s3_emits_warning(self, caplog):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.create_policy.return_value = {"Policy": {"Arn": "arn:aws:iam::1:policy/p"}}
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            resolver.create_execution_role(role_type="training")  # default s3="*"
        assert any("ALL S3 buckets" in r.getMessage() for r in caplog.records)

    def test_scoped_s3_suppresses_warning(self, caplog):
        resolver, mock_iam = _make_resolver()
        mock_iam.create_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/SageMaker-AutoRole-Training"}
        }
        mock_iam.create_policy.return_value = {"Policy": {"Arn": "arn:aws:iam::1:policy/p"}}
        with patch("sagemaker.core.helper.iam_role_resolver.time.sleep"), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            resolver.create_execution_role(
                role_type="training", s3_resource="my-bucket", kms_resource="my-key"
            )
        assert not any("ALL S3 buckets" in r.getMessage() for r in caplog.records)


class TestDeleteExecutionRole:
    def test_detaches_deletes_policies_and_role(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [
                {
                    "PolicyName": "SageMaker-AutoRole-Training-s3_policy",
                    "PolicyArn": "arn:aws:iam::123456789012:policy/SageMaker-AutoRole-Training-s3_policy",
                }
            ]
        }
        mock_iam.list_policy_versions.return_value = {
            "Versions": [{"VersionId": "v1", "IsDefaultVersion": True}]
        }

        resolver.delete_execution_role(role_type="training")

        mock_iam.detach_role_policy.assert_called_once()
        mock_iam.delete_policy.assert_called_once()
        mock_iam.delete_role.assert_called_once_with(RoleName="SageMaker-AutoRole-Training")

    def test_missing_role_is_noop(self):
        resolver, mock_iam = _make_resolver()
        mock_iam.list_attached_role_policies.side_effect = ClientError(
            {"Error": {"Code": "NoSuchEntity"}}, "ListAttachedRolePolicies"
        )
        # Should not raise.
        resolver.delete_execution_role(role_type="training")
        mock_iam.delete_role.assert_not_called()

    def test_does_not_delete_foreign_policies(self):
        """Policies not created by the SDK (no role-name prefix) are detached but not deleted."""
        resolver, mock_iam = _make_resolver()
        mock_iam.list_attached_role_policies.return_value = {
            "AttachedPolicies": [
                {
                    "PolicyName": "SomeCustomerManagedPolicy",
                    "PolicyArn": "arn:aws:iam::123456789012:policy/SomeCustomerManagedPolicy",
                }
            ]
        }
        resolver.delete_execution_role(role_type="training")
        mock_iam.detach_role_policy.assert_called_once()
        mock_iam.delete_policy.assert_not_called()


class TestGetRequiredActions:
    def test_returns_actions_for_role_type(self):
        resolver, _ = _make_resolver()
        actions = resolver.get_required_actions("training")
        assert "sagemaker:DescribeHubContent" in actions

    def test_invalid_role_type_raises(self):
        resolver, _ = _make_resolver()
        with pytest.raises(ValueError, match="Invalid role_type"):
            resolver.get_required_actions("nope")
