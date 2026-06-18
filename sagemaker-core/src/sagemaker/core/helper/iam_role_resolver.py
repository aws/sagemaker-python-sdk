"""IAM role auto-creation for SageMaker Python SDK.

Provides idempotent role resolution: if no role is specified, the SDK checks
whether the default/caller role has sufficient permissions. Only creates a new
role if the existing one lacks the required permissions for the operation.
"""
from __future__ import absolute_import

import copy
import json
import logging
import time
from typing import List, Optional, Set, Union
from urllib.parse import unquote

from botocore.exceptions import ClientError

from sagemaker.core.helper.iam_policies import IAM_POLICY_CONFIG

logger = logging.getLogger(__name__)

ROLE_TYPES = ("training", "serving", "pipeline", "feature_store", "bedrock", "hyperpod")

_IAM_PROPAGATION_DELAY_SECONDS = 10

# An IAM managed policy can hold at most five versions.
_MAX_POLICY_VERSIONS = 5

_REQUIRED_IAM_ACTIONS = (
    "iam:CreateRole",
    "iam:CreatePolicy",
    "iam:AttachRolePolicy",
    "iam:GetRole",
    "iam:TagRole",
)

# Permissions the HyperPod CLI flow needs on the *caller* identity — the local
# principal that runs `hyperpod connect-cluster` and `hyperpod start-job`. The CLI
# resolves the HyperPod cluster (sagemaker:DescribeCluster), reads its EKS
# orchestrator and refreshes the kubeconfig (eks:DescribeCluster, via
# `aws eks update-kubeconfig`), and submits jobs through the Kubernetes API the
# cluster fronts (eks:AccessKubernetesApi). These are deliberately NOT attached to
# the auto-created job execution role: that role is trusted by sagemaker.amazonaws.com
# and assumed by the job on the cluster, whereas these actions must be held by the
# caller running the CLI. See verify_hyperpod_connect_permissions().
HYPERPOD_CLI_CONNECT_ACTIONS = (
    "sagemaker:DescribeCluster",
    "eks:DescribeCluster",
    "eks:AccessKubernetesApi",
)

# Ownership tags applied to every role the SDK auto-creates, so the roles are
# auditable and clearly attributable to the SDK. ``RoleType`` (filled in per
# role) records which workload the role was created for.
_SDK_ROLE_TAGS = {
    "CreatedBy": "sagemaker-python-sdk",
}


class RoleAutoCreationError(Exception):
    """Raised when the SDK cannot auto-create an IAM role due to insufficient permissions."""


def _trust_principal_for(role_type: str) -> str:
    """Return a human-readable trust principal for a role type's remediation text."""
    statement = IAM_POLICY_CONFIG[role_type]["trust_policy"]["Statement"][0]
    service = statement["Principal"]["Service"]
    if isinstance(service, (list, tuple)):
        return ", ".join(service)
    return service


def _raise_auto_creation_error(
    role_name: str, original_error: ClientError, role_type: str = "training"
) -> None:
    """Raise a RoleAutoCreationError with actionable remediation guidance."""
    raise RoleAutoCreationError(
        f"Cannot auto-create IAM role '{role_name}'.\n\n"
        f"Your IAM identity needs: {', '.join(_REQUIRED_IAM_ACTIONS)}\n\n"
        f"Alternatively, create a role manually and pass it via the 'role' parameter.\n"
        f"Required trust principal: {_trust_principal_for(role_type)}\n\n"
        f"Original error: {original_error}"
    ) from original_error


def _load_policy_config():
    """Return the policy configuration constant."""
    return IAM_POLICY_CONFIG


def _partition_from_arn(arn: str) -> str:
    """Extract the AWS partition (aws, aws-cn, aws-us-gov) from an ARN."""
    parts = arn.split(":")
    return parts[1] if len(parts) > 1 and parts[1] else "aws"


def _build_role_tags(role_type: str) -> List[dict]:
    """Build the SDK ownership tag list (IAM TagList form) for a role type."""
    tags = dict(_SDK_ROLE_TAGS)
    tags["RoleType"] = role_type
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def _scope_trust_policy_to_account(trust_policy: dict, account_id: str) -> dict:
    """Substitute ACCOUNT_PLACEHOLDER in trust-policy conditions with the account ID.

    The ``aws:SourceAccount`` condition restricts which account's service requests
    may assume the role, preventing the cross-service confused-deputy problem.
    """
    trust_policy = copy.deepcopy(trust_policy)
    for statement in trust_policy.get("Statement", []):
        condition = statement.get("Condition", {})
        for operator_values in condition.values():
            if "aws:SourceAccount" in operator_values:
                operator_values["aws:SourceAccount"] = account_id
    return trust_policy


def _ensure_role_tagged(iam_client, role_name: str, role_type: str) -> None:
    """Idempotently ensure an existing auto-role carries the SDK ownership tags.

    Roles created before tagging was introduced (or via a non-create path such as
    a concurrent EntityAlreadyExists fallback) may be missing the tags. ``TagRole``
    is idempotent, so re-applying identical tags is a no-op. Tagging failures are
    non-fatal: the role is still usable, so we log and continue.
    """
    try:
        existing = {
            t["Key"] for t in iam_client.list_role_tags(RoleName=role_name).get("Tags", [])
        }
        desired = _build_role_tags(role_type)
        missing = [t for t in desired if t["Key"] not in existing]
        if missing:
            iam_client.tag_role(RoleName=role_name, Tags=desired)
            logger.info(
                "Applied SDK ownership tags to existing role '%s'.", role_name
            )
    except ClientError as e:
        logger.info(
            "Could not verify/apply ownership tags on role '%s': %s", role_name, e
        )


def _is_wildcard_scope(resource: Union[str, List[str]]) -> bool:
    """Return True if a resource scope resolves to the account-wide "*" wildcard.

    Mirrors the collapsing behavior of ``_expand_s3_resource``/``_expand_kms_resource``:
    a bare ``"*"`` or any list that contains ``"*"`` means account-wide access.
    """
    if isinstance(resource, str):
        return resource == "*"
    return "*" in resource


def _role_uses_placeholder(role_config: dict, placeholder: str) -> bool:
    """Return True if any of the role's policy statements uses ``placeholder`` as its Resource."""
    for policy_doc in role_config["policies"].values():
        for statement in policy_doc.get("Statement", []):
            if statement.get("Resource") == placeholder:
                return True
    return False


def _warn_broad_permissions(
    role_type: str,
    role_config: dict,
    s3_resource: Union[str, List[str]],
    kms_resource: Union[str, List[str]],
) -> None:
    """Warn when the auto-role will be granted account-wide ("*") resource access.

    These broad grants are accepted by design — the SDK cannot always know the
    exact buckets, keys, or Glue catalog a job needs at role-creation time — but
    they are surfaced at WARNING so the account owner is aware and can scope them
    down (via ``s3_resource``/``kms_resource``) or supply their own role.
    """
    if _role_uses_placeholder(role_config, "S3_PLACEHOLDER") and _is_wildcard_scope(
        s3_resource
    ):
        logger.warning(
            "SageMaker Python SDK is granting the auto-created '%s' role access to "
            "ALL S3 buckets in your account (s3 resource scope is '*'). To restrict "
            "this, pass s3_resource=<bucket name(s)> when resolving the role, or supply "
            "your own least-privilege role via the 'role' parameter.",
            role_type,
        )
    if _role_uses_placeholder(role_config, "KMS_PLACEHOLDER") and _is_wildcard_scope(
        kms_resource
    ):
        logger.warning(
            "SageMaker Python SDK is granting the auto-created '%s' role access to "
            "ALL KMS keys in your account (kms resource scope is '*'). To restrict "
            "this, pass kms_resource=<key id(s)> when resolving the role, or supply "
            "your own least-privilege role via the 'role' parameter.",
            role_type,
        )
    if role_type == "feature_store":
        # The feature_store role grants read-only glue:GetTable/GetDatabase/
        # GetPartitions on "*" (see iam_policies.py). The specific catalog,
        # databases, and tables are not known at role-creation time, so this
        # broad read-only metadata grant is accepted by design — surface it.
        logger.warning(
            "SageMaker Python SDK is granting the auto-created 'feature_store' role "
            "read-only access to ALL AWS Glue catalog metadata in your account "
            "(glue:GetTable/GetDatabase/GetPartitions on '*'). This metadata grant is "
            "required because the specific Glue databases/tables are not known at "
            "role-creation time. Supply your own least-privilege role via the 'role' "
            "parameter to scope this down."
        )


def _get_attached_policy_names(iam_client, role_name: str) -> Set[str]:
    """Return the set of policy names already attached to a role."""
    attached = iam_client.list_attached_role_policies(RoleName=role_name)
    return {p["PolicyName"].lower() for p in attached.get("AttachedPolicies", [])}


def _expand_s3_resource(s3_resource, partition: str = "aws"):
    """Expand one or more S3 bucket names into bucket + object ARNs.

    Accepts a single bucket name or a list of bucket names. Returns "*" when
    the value is the "*" wildcard (or a list containing it).
    """
    buckets = [s3_resource] if isinstance(s3_resource, str) else list(s3_resource)
    if "*" in buckets:
        return "*"
    arns = []
    for bucket in buckets:
        arns.append(f"arn:{partition}:s3:::{bucket}")
        arns.append(f"arn:{partition}:s3:::{bucket}/*")
    return arns


def _expand_kms_resource(kms_resource, partition: str = "aws", account_id: str = "*"):
    """Expand one or more KMS key IDs into key ARNs.

    Accepts a single key ID or a list of key IDs. Returns "*" when the value
    is the "*" wildcard (or a list containing it).

    The region segment is intentionally left as "*": a SageMaker job may use a
    KMS key in a different region from the one creating the role (e.g. cross-region
    artifacts), and the key's region isn't reliably known at policy-creation time.
    The account segment is scoped to the caller's account when available.
    """
    keys = [kms_resource] if isinstance(kms_resource, str) else list(kms_resource)
    if "*" in keys:
        return "*"
    return [f"arn:{partition}:kms:*:{account_id}:key/{key}" for key in keys]


def _apply_partition(resource, partition: str):
    """Rewrite the partition segment of literal ``arn:aws:`` resource ARNs.

    Policy documents hardcode the commercial ``arn:aws:`` partition for static
    ARNs (e.g. CloudWatch log groups). In GovCloud / China / ISO partitions those
    would never match, so normalize them to the caller's partition. Accepts a
    single ARN string or a list of them; non-ARN values (e.g. ``"*"``) pass
    through unchanged.
    """
    if partition == "aws":
        return resource

    def _rewrite(value):
        if isinstance(value, str) and value.startswith("arn:aws:"):
            # Replace the "aws" partition token only; keep the ":service:..."
            # remainder intact.
            return "arn:" + partition + value[len("arn:aws"):]
        return value

    if isinstance(resource, list):
        return [_rewrite(v) for v in resource]
    return _rewrite(resource)


def _replace_placeholders(
    policies: dict, s3_resource, kms_resource, partition: str = "aws", account_id: str = "*"
) -> dict:
    """Replace placeholder values in policy documents with actual resource ARNs.

    ``s3_resource`` and ``kms_resource`` each accept a single name/ID, a list of
    them, or the "*" wildcard. Literal ``arn:aws:`` ARNs in the policy documents
    are rewritten to the caller's partition.
    """
    policies = copy.deepcopy(policies)
    for policy_doc in policies.values():
        for statement in policy_doc.get("Statement", []):
            resource = statement.get("Resource")
            if resource == "S3_PLACEHOLDER":
                statement["Resource"] = _expand_s3_resource(s3_resource, partition)
            elif resource == "KMS_PLACEHOLDER":
                statement["Resource"] = _expand_kms_resource(
                    kms_resource, partition, account_id
                )
            elif resource == "IAM_PASSROLE_PLACEHOLDER":
                # Scope iam:PassRole to the SDK's own auto-created roles in the
                # caller's account (rather than all roles), so this role can only
                # pass least-privilege SageMaker-AutoRole-* roles and never an
                # arbitrary (e.g. highly privileged) role in the account. The
                # PassedToService=sagemaker condition further restricts the target
                # service. Region is omitted because IAM is a global service.
                statement["Resource"] = (
                    f"arn:{partition}:iam::{account_id}:role/SageMaker-AutoRole-*"
                )
            else:
                # Static ARNs in the config hardcode the commercial partition;
                # normalize them to the caller's partition (no-op for "aws").
                statement["Resource"] = _apply_partition(resource, partition)
    return policies


def _get_required_actions(role_type: str) -> List[str]:
    """Extract the list of required IAM actions for a given role type from policy config."""
    config = _load_policy_config()
    role_config = config[role_type]
    actions = []
    for policy_doc in role_config["policies"].values():
        for statement in policy_doc.get("Statement", []):
            stmt_actions = statement.get("Action", [])
            if isinstance(stmt_actions, str):
                stmt_actions = [stmt_actions]
            actions.extend(stmt_actions)
    return actions


def _simulate_denied_actions(iam_client, role_arn: str, actions: List[str]) -> List[str]:
    """Return the subset of ``actions`` that ``role_arn`` is NOT allowed to perform.

    Wraps the paginated ``iam:SimulatePrincipalPolicy`` call so both the role-
    resolution path and the HyperPod caller-side check share one implementation.
    An empty list means every action is allowed. The pagination matters: a
    truncated first page must not produce a false "all allowed" verdict.

    Raises ClientError on failures the caller must interpret (e.g. AccessDenied
    when the principal can't self-simulate, NoSuchEntity when the role is gone).
    """
    evaluation_results = []
    paginator = iam_client.get_paginator("simulate_principal_policy")
    for page in paginator.paginate(PolicySourceArn=role_arn, ActionNames=actions):
        evaluation_results.extend(page.get("EvaluationResults", []))

    return [
        result["EvalActionName"]
        for result in evaluation_results
        if result["EvalDecision"] != "allowed"
    ]


def _role_has_sufficient_permissions(
    iam_client, role_arn: str, role_type: str
) -> Optional[bool]:
    """Check whether a role has the permissions required for the given operation.

    Uses iam:SimulatePrincipalPolicy to test whether the role can perform the
    required actions.

    Returns:
        True  — all required actions are allowed.
        False — at least one required action is explicitly denied.
        None  — could not be determined (e.g. the caller lacks
                iam:SimulatePrincipalPolicy). Callers decide how to treat this.
    """
    required_actions = _get_required_actions(role_type)
    if not required_actions:
        return True

    try:
        denied = _simulate_denied_actions(iam_client, role_arn, required_actions)
        if denied:
            logger.info(
                "Role '%s' is missing permissions for: %s",
                role_arn,
                ", ".join(denied[:5]) + ("..." if len(denied) > 5 else ""),
            )
            return False
        return True

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AccessDenied", "AccessDeniedException"):
            # Cannot simulate — verdict is unknown.
            logger.info(
                "Cannot simulate policies for '%s' (access denied); "
                "permission verdict unknown.",
                role_arn,
            )
            return None
        if error_code in ("NoSuchEntity", "NoSuchEntityException"):
            return False
        raise


def _policy_document_matches(iam_client, policy_arn: str, desired_document: dict) -> bool:
    """Return True if the policy's default version already matches the desired document."""
    try:
        policy = iam_client.get_policy(PolicyArn=policy_arn)
        default_version_id = policy["Policy"]["DefaultVersionId"]
        version = iam_client.get_policy_version(
            PolicyArn=policy_arn, VersionId=default_version_id
        )
        current_document = version["PolicyVersion"]["Document"]
    except ClientError:
        # If we cannot read the current document, assume it needs syncing.
        return False
    # IAM may URL-encode the stored document; normalize both sides before comparing.
    if isinstance(current_document, str):
        current_document = json.loads(unquote(current_document))
    return current_document == desired_document


def _update_policy_document(iam_client, policy_arn: str, policy_document: dict) -> None:
    """Create a new default version of an existing policy, pruning old versions if needed.

    A managed policy can hold at most five versions, so the oldest non-default
    version is deleted before adding a new one when the limit is reached.
    """
    versions = iam_client.list_policy_versions(PolicyArn=policy_arn).get("Versions", [])
    if len(versions) >= _MAX_POLICY_VERSIONS:
        oldest = min(
            (v for v in versions if not v["IsDefaultVersion"]),
            key=lambda v: v["VersionId"],
        )
        iam_client.delete_policy_version(
            PolicyArn=policy_arn, VersionId=oldest["VersionId"]
        )

    iam_client.create_policy_version(
        PolicyArn=policy_arn,
        PolicyDocument=json.dumps(policy_document),
        SetAsDefault=True,
    )


def _ensure_policies_attached(
    iam_client, role_name: str, policies: dict, account_id: str, partition: str = "aws"
) -> None:
    """Create, update, and attach the role's least-privilege policies.

    Idempotent on policy *content*, not just name: if a managed policy already
    exists but its document has drifted from the current definition, a new default
    version is created so existing roles pick up permission changes.
    """
    already_attached = _get_attached_policy_names(iam_client, role_name)

    created, updated, attached = [], [], []
    for policy_name, policy_document in policies.items():
        managed_policy_name = f"{role_name}-{policy_name}"
        policy_arn = f"arn:{partition}:iam::{account_id}:policy/{managed_policy_name}"

        try:
            iam_client.create_policy(
                PolicyName=managed_policy_name,
                PolicyDocument=json.dumps(policy_document),
            )
            created.append(managed_policy_name)
        except iam_client.exceptions.EntityAlreadyExistsException:
            # Policy already exists; update its document only if it has drifted.
            if _policy_document_matches(iam_client, policy_arn, policy_document):
                logger.info("Policy %s is up to date", managed_policy_name)
            else:
                _update_policy_document(iam_client, policy_arn, policy_document)
                updated.append(managed_policy_name)

        if managed_policy_name.lower() not in already_attached:
            iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            attached.append(managed_policy_name)

    # Surface account-mutating changes at WARNING so the user is not surprised
    # by policies the SDK created/updated/attached on their behalf.
    if created:
        logger.warning(
            "SageMaker Python SDK created %d IAM managed %s in your AWS account and "
            "attached %s to role '%s': %s",
            len(created),
            "policy" if len(created) == 1 else "policies",
            "it" if len(created) == 1 else "them",
            role_name,
            ", ".join(created),
        )
    if updated:
        logger.warning(
            "SageMaker Python SDK updated %d existing IAM managed %s on role '%s' "
            "to the latest least-privilege document: %s",
            len(updated),
            "policy" if len(updated) == 1 else "policies",
            role_name,
            ", ".join(updated),
        )
    # Pre-existing policies re-attached (created elsewhere) but not freshly created.
    reattached = [name for name in attached if name not in created]
    if reattached:
        logger.warning(
            "SageMaker Python SDK attached %d existing IAM managed %s to role '%s': %s",
            len(reattached),
            "policy" if len(reattached) == 1 else "policies",
            role_name,
            ", ".join(reattached),
        )


def resolve_or_create_role(
    provided_role: Optional[str],
    role_type: str,
    sagemaker_session=None,
    s3_resource: Union[str, List[str]] = "*",
    kms_resource: Union[str, List[str]] = "*",
) -> str:
    """Resolve an existing role or auto-create one with least-privilege policies.

    Resolution order:
        1. Explicit role provided → validate it exists → return ARN.
        2. Caller identity is already a role → check permissions → return if sufficient.
        3. Default auto-role exists → check permissions → return if sufficient.
        4. Auto-create role with correct policies → return ARN.
        5. Raise RoleAutoCreationError if IAM permissions are insufficient.

    A new role is only created when the existing/default role does NOT have the
    permissions required for the operation.

    Args:
        provided_role: User-supplied role name or ARN. If set, used directly.
        role_type: One of "training", "serving", "pipeline", "feature_store",
            "bedrock", "hyperpod".
        sagemaker_session: SageMaker session (used to get boto session).
        s3_resource: S3 bucket name (or list of bucket names) to scope policies.
            Defaults to "*" (no scoping). A "*" scope grants the auto-created role
            access to all S3 buckets in the account and emits a WARNING.
        kms_resource: KMS key ID (or list of key IDs) to scope policies.
            Defaults to "*" (no scoping). A "*" scope grants the auto-created role
            access to all KMS keys in the account and emits a WARNING.

    Returns:
        IAM role ARN.

    Raises:
        RoleAutoCreationError: If auto-creation fails due to IAM permissions.
        ValueError: If provided_role doesn't exist or role_type is invalid.
    """
    if role_type not in ROLE_TYPES:
        raise ValueError(f"Invalid role_type '{role_type}'. Must be one of: {ROLE_TYPES}")

    if provided_role:
        return _resolve_explicit_role(provided_role, sagemaker_session)

    return _auto_resolve_role(role_type, sagemaker_session, s3_resource, kms_resource)


def verify_hyperpod_connect_permissions(
    sagemaker_session=None, cluster_name: Optional[str] = None
) -> Optional[bool]:
    """Verify the caller can drive the HyperPod CLI; warn (don't block) if not.

    The HyperPod CLI (``connect-cluster`` + ``start-job``) runs locally under the
    *caller's* credentials, not under the job execution role. So unlike the job
    role — which the SDK can auto-create — these permissions must already be held
    by whoever runs the trainer. This function simulates them on the caller
    identity and surfaces actionable guidance when they are missing.

    It logs a WARNING rather than raising so the SDK does not hard-fail in the
    common case where the caller cannot self-simulate (e.g. an execution role
    without ``iam:SimulatePrincipalPolicy``); the CLI itself will still produce a
    precise error if a permission is truly absent. Callers that want to fail fast
    can inspect the return value.

    Args:
        sagemaker_session: SageMaker session (used to get the boto session).
        cluster_name: Optional HyperPod cluster name, included in guidance.

    Returns:
        True  — all connect actions are allowed for the caller.
        False — at least one connect action is denied.
        None  — could not be determined (caller is not a role, or cannot simulate).
    """
    boto_session = _get_boto_session(sagemaker_session)
    sts_client = boto_session.client("sts")
    iam_client = boto_session.client("iam")

    caller_identity = sts_client.get_caller_identity()
    caller_arn = caller_identity["Arn"]
    account_id = caller_identity["Account"]
    partition = _partition_from_arn(caller_arn)

    caller_role_arn = _resolve_caller_role_arn(iam_client, caller_arn, account_id, partition)
    if not caller_role_arn:
        # IAM user / root: nothing to simulate against a role. Leave it to the CLI.
        logger.info(
            "Could not resolve a caller role to verify HyperPod connect "
            "permissions; the HyperPod CLI will validate access at submit time."
        )
        return None

    try:
        denied = _simulate_denied_actions(
            iam_client, caller_role_arn, list(HYPERPOD_CLI_CONNECT_ACTIONS)
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AccessDenied", "AccessDeniedException"):
            logger.info(
                "Cannot simulate HyperPod connect permissions for '%s' (access "
                "denied); the HyperPod CLI will validate access at submit time.",
                caller_role_arn,
            )
            return None
        raise

    if denied:
        cluster_hint = f" for cluster '{cluster_name}'" if cluster_name else ""
        logger.warning(
            "Your identity '%s' is missing IAM permissions the HyperPod CLI needs"
            "%s: %s. The job execution role was resolved successfully, but "
            "connecting to and submitting jobs on the cluster runs as your own "
            "credentials. Grant these actions to your identity (scoped to the "
            "cluster and its EKS orchestrator) to avoid CLI failures.",
            caller_role_arn,
            cluster_hint,
            ", ".join(denied),
        )
        return False

    logger.info(
        "Caller '%s' has the HyperPod CLI connect permissions.", caller_role_arn
    )
    return True


def _get_boto_session(sagemaker_session):
    """Return the boto session, falling back to a SageMaker core Session.

    Uses ``sagemaker.core.helper.session_helper.Session`` (rather than boto3
    directly) so the fallback inherits the SDK's config and region defaults.
    Imported lazily to avoid a circular import at module load.
    """
    if sagemaker_session:
        return sagemaker_session.boto_session
    from sagemaker.core.helper.session_helper import Session

    return Session().boto_session


def _resolve_explicit_role(provided_role: str, sagemaker_session=None) -> str:
    """Validate and return ARN for an explicitly provided role."""
    # Accept any partition's IAM role ARN (aws, aws-cn, aws-us-gov, aws-iso-*),
    # not just the commercial "aws" partition.
    if provided_role.startswith("arn:") and ":iam::" in provided_role:
        return provided_role

    boto_session = _get_boto_session(sagemaker_session)
    iam_client = boto_session.client("iam")
    try:
        response = iam_client.get_role(RoleName=provided_role)
        return response["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchEntity", "NoSuchEntityException"):
            raise ValueError(
                f"IAM role '{provided_role}' does not exist. "
                f"Create it first or omit the role parameter to auto-create one."
            ) from e
        raise


def _resolve_caller_role_arn(
    iam_client, caller_arn: str, account_id: str, partition: str
) -> Optional[str]:
    """Resolve the canonical IAM role ARN backing the current caller identity.

    Handles assumed-role ARNs (the STS get_caller_identity format used in Studio,
    notebooks, and on EC2) as well as direct role ARNs. Returns None if the caller
    is an IAM user, the root account, or the role cannot be resolved.

    Resolving via get_role (rather than string-building the ARN) ensures roles with
    a path such as ``/service-role/`` produce the correct ARN.
    """
    if ":assumed-role/" in caller_arn:
        role_name = caller_arn.split("/")[1]
    elif ":role/" in caller_arn:
        role_name = caller_arn.split(":role/")[1].split("/")[-1]
    else:
        # IAM user or root — no backing role to reuse.
        return None

    try:
        return iam_client.get_role(RoleName=role_name)["Role"]["Arn"]
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("NoSuchEntity", "NoSuchEntityException"):
            return None
        if error_code in ("AccessDenied", "AccessDeniedException"):
            # Can't introspect the caller role; fall back to a string-built ARN
            # so the downstream permission simulation can still try.
            return f"arn:{partition}:iam::{account_id}:role/{role_name}"
        raise


def _auto_resolve_role(
    role_type: str,
    sagemaker_session,
    s3_resource: Union[str, List[str]],
    kms_resource: Union[str, List[str]],
) -> str:
    """Resolve a role through permission checks, only creating if necessary.

    Steps:
        1. If caller is a role with sufficient permissions → use it.
        2. If default auto-role exists with sufficient permissions → use it.
        3. Otherwise, create/update the auto-role with required policies.
    """
    boto_session = _get_boto_session(sagemaker_session)
    sts_client = boto_session.client("sts")
    iam_client = boto_session.client("iam")

    caller_identity = sts_client.get_caller_identity()
    caller_arn = caller_identity["Arn"]
    account_id = caller_identity["Account"]
    partition = _partition_from_arn(caller_arn)

    # Step 1: If the caller is (assuming) a role, check whether it already has
    # sufficient permissions and reuse it — this is the common case in Studio,
    # notebooks, and on EC2. STS get_caller_identity returns an assumed-role ARN
    # (arn:aws:sts::ACCT:assumed-role/ROLE_NAME/SESSION), so we resolve the
    # canonical IAM role ARN via get_role (which also handles role paths).
    caller_role_arn = _resolve_caller_role_arn(iam_client, caller_arn, account_id, partition)
    if caller_role_arn:
        has_perms = _role_has_sufficient_permissions(iam_client, caller_role_arn, role_type)
        # Reuse the caller's role when it is sufficient OR when we cannot determine
        # (has_perms is None) — a SageMaker execution role commonly lacks
        # iam:SimulatePrincipalPolicy, and creating a brand-new role in that case
        # would be a regression from the prior get_execution_role() behavior.
        if has_perms is not False:
            reason = "has sufficient permissions" if has_perms else "assumed usable (unverifiable)"
            logger.info("Caller role '%s' %s for %s. Using it.", caller_role_arn, reason, role_type)
            return caller_role_arn
        logger.info(
            "Caller role '%s' lacks permissions for %s. Will use/create dedicated role.",
            caller_role_arn,
            role_type,
        )

    # Step 2: Check if default auto-role already exists with sufficient permissions
    config = _load_policy_config()
    role_config = config[role_type]
    auto_role_name = role_config["role_name"]

    # We are committing to the dedicated SDK auto-role (the caller's own role was
    # unusable or absent). Warn about any account-wide ("*") grants this role will
    # carry so the account owner is aware and can scope them down.
    _warn_broad_permissions(role_type, role_config, s3_resource, kms_resource)

    try:
        existing_role = iam_client.get_role(RoleName=auto_role_name)
        auto_role_arn = existing_role["Role"]["Arn"]

        if _role_has_sufficient_permissions(iam_client, auto_role_arn, role_type) is True:
            logger.info(
                "Existing role '%s' has sufficient permissions. Reusing.",
                auto_role_name,
            )
            # Ensure pre-existing auto-roles carry the SDK ownership tags, even when
            # they were created before tagging was introduced.
            _ensure_role_tagged(iam_client, auto_role_name, role_type)
            return auto_role_arn
        # Role exists but is missing (or we cannot verify) some policies — since we
        # own this auto-role, attach the missing ones. Attachment is idempotent.
        logger.info(
            "Role '%s' exists but lacks some permissions. Attaching missing policies.",
            auto_role_name,
        )
        policies = _replace_placeholders(
            role_config["policies"], s3_resource, kms_resource, partition, account_id
        )
        _ensure_policies_attached(iam_client, auto_role_name, policies, account_id, partition)
        _ensure_role_tagged(iam_client, auto_role_name, role_type)
        logger.info("Using role: %s", auto_role_arn)
        return auto_role_arn

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code not in ("NoSuchEntity", "NoSuchEntityException"):
            if error_code in ("AccessDenied", "AccessDeniedException", "UnauthorizedAccess"):
                _raise_auto_creation_error(auto_role_name, e, role_type)
            raise

    # Step 3: Role doesn't exist — create it.
    trust_policy = _scope_trust_policy_to_account(role_config["trust_policy"], account_id)
    policies = _replace_placeholders(
        role_config["policies"], s3_resource, kms_resource, partition, account_id
    )

    try:
        logger.info("Creating role '%s'...", auto_role_name)
        try:
            response = iam_client.create_role(
                RoleName=auto_role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=(
                    f"Auto-created by SageMaker Python SDK for {role_type} workloads. "
                    f"Safe to delete if no longer needed."
                ),
                Tags=_build_role_tags(role_type),
            )
            role_arn = response["Role"]["Arn"]
            # Surface the new role at WARNING so the user is not surprised by an
            # IAM role the SDK created on their behalf.
            logger.warning(
                "SageMaker Python SDK created a new IAM role '%s' (%s) in your AWS "
                "account for %s workloads. It is safe to delete if no longer needed.",
                auto_role_name,
                role_arn,
                role_type,
            )
        except iam_client.exceptions.EntityAlreadyExistsException:
            # Another caller created the role concurrently (or a prior partial
            # create left it behind). Reuse it — policy attachment below is idempotent.
            logger.info("Role '%s' already exists, reusing.", auto_role_name)
            role_arn = iam_client.get_role(RoleName=auto_role_name)["Role"]["Arn"]
            # The concurrently/previously created role may lack ownership tags.
            _ensure_role_tagged(iam_client, auto_role_name, role_type)

        _ensure_policies_attached(iam_client, auto_role_name, policies, account_id, partition)

        logger.info("Waiting %ds for IAM propagation...", _IAM_PROPAGATION_DELAY_SECONDS)
        time.sleep(_IAM_PROPAGATION_DELAY_SECONDS)
        logger.info("Using role: %s", role_arn)
        return role_arn

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AccessDenied", "AccessDeniedException", "UnauthorizedAccess"):
            _raise_auto_creation_error(auto_role_name, e, role_type)
        raise
