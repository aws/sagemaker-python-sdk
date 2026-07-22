"""IAM role resolution, validation, and (opt-in) creation for the SageMaker Python SDK.

The default path is **read-only**: :func:`resolve_and_validate_role` resolves the
role to use (an explicitly provided role, or the caller's own identity) and
*validates* that it has the permissions and trust policy required for the
operation. It never creates, mutates, attaches, or tags IAM roles or policies.

Creating a managed least-privilege execution role is an explicit, opt-in action
exposed via the :class:`IamRoleResolver` class at the bottom of this module —
the only code path here that writes IAM. Auto-creation was removed from the
default path because mutating a customer's IAM account as a side effect of an
ordinary SDK call is an elevation-of-privilege risk.
"""
from __future__ import absolute_import

import json
import logging
import time
from typing import List, Optional, Set, Tuple, Union
from urllib.parse import unquote

from botocore.exceptions import ClientError

from sagemaker.core.helper.iam_policies import IAM_POLICY_CONFIG

logger = logging.getLogger(__name__)

ROLE_TYPES = ("training", "serving", "pipeline", "feature_store", "bedrock", "hyperpod")

# Permissions the HyperPod CLI flow needs on the *caller* identity — the local
# principal that runs `hyperpod connect-cluster` and `hyperpod start-job`. The CLI
# resolves the HyperPod cluster (sagemaker:DescribeCluster), reads its EKS
# orchestrator and refreshes the kubeconfig (eks:DescribeCluster, via
# `aws eks update-kubeconfig`), and submits jobs through the Kubernetes API the
# cluster fronts (eks:AccessKubernetesApi). These are deliberately NOT part of the
# job execution role: that role is trusted by sagemaker.amazonaws.com and assumed
# by the job on the cluster, whereas these actions must be held by the caller
# running the CLI. See verify_hyperpod_connect_permissions().
HYPERPOD_CLI_CONNECT_ACTIONS = (
    "sagemaker:DescribeCluster",
    "eks:DescribeCluster",
    "eks:AccessKubernetesApi",
)

# Actions the *caller* must have to orchestrate Pipeline-based evaluations
# directly. These actions must be held by whoever calls evaluator.evaluate(),
# NOT by the job execution role (which is covered by role_type="training").
# See verify_evaluation_caller_permissions().
from sagemaker.core.helper.iam_policies import EVALUATION_CALLER_ACTIONS


class RoleValidationError(Exception):
    """Raised when the resolved IAM role lacks the permissions/trust an operation needs.

    The message lists the specific missing permissions (or required actions) and
    the two ways to fix it: grant them to the role, or create a dedicated
    least-privilege role via ``IamRoleResolver().create_execution_role(...)``.
    """


class RoleAutoCreationError(Exception):
    """Raised when explicit role creation fails due to insufficient IAM permissions."""


# ---------------------------------------------------------------------------
# Pure-data helpers (no AWS calls, no mutation).
# ---------------------------------------------------------------------------
def _load_policy_config():
    """Return the policy configuration constant."""
    return IAM_POLICY_CONFIG


def _partition_from_arn(arn: str) -> str:
    """Extract the AWS partition (aws, aws-cn, aws-us-gov) from an ARN."""
    parts = arn.split(":")
    return parts[1] if len(parts) > 1 and parts[1] else "aws"


def _trust_principal_for(role_type: str) -> str:
    """Return a human-readable trust principal for a role type's remediation text."""
    statement = IAM_POLICY_CONFIG[role_type]["trust_policy"]["Statement"][0]
    service = statement["Principal"]["Service"]
    if isinstance(service, (list, tuple)):
        return ", ".join(service)
    return service


def _get_required_actions(role_type: str) -> List[str]:
    """Extract the full list of IAM actions a role type needs from policy config.

    This is the complete set (used to show users what to grant). It is NOT what the
    permission *gate* simulates against — see ``_get_smoke_test_actions`` for why.
    """
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


def _statement_resource_is_wildcard(resource) -> bool:
    """Return True if a statement's Resource is the account-wide ``*`` wildcard."""
    if isinstance(resource, str):
        return resource == "*"
    if isinstance(resource, list):
        return "*" in resource
    return False


def _get_smoke_test_actions(role_type: str) -> List[str]:
    """Actions used to *gate* role validation (a definitive deny here blocks reuse).

    We deliberately validate only against actions whose policy ``Resource`` is the
    ``*`` wildcard. ``iam:SimulatePrincipalPolicy`` is called without ResourceArns
    (we don't know the caller's concrete buckets/keys/MLflow servers up front), so a
    resource-*scoped* action (e.g. ``sagemaker-mlflow:LogMetric`` on
    ``mlflow-app/*``, or S3 on a specific bucket) would evaluate against an implicit
    ``*`` and come back ``implicitDeny`` even for a perfectly good role — a false
    positive that would wrongly block training/eval jobs (including ones that never
    use MLflow). Gating only on ``*``-resource actions keeps validation a meaningful
    "is this a usable SageMaker execution role?" smoke test without false denials.
    The full action list is still surfaced in the error message for remediation.
    """
    config = _load_policy_config()
    role_config = config[role_type]
    actions = []
    for policy_doc in role_config["policies"].values():
        for statement in policy_doc.get("Statement", []):
            if not _statement_resource_is_wildcard(statement.get("Resource")):
                continue
            stmt_actions = statement.get("Action", [])
            if isinstance(stmt_actions, str):
                stmt_actions = [stmt_actions]
            actions.extend(stmt_actions)
    return actions


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
    import copy

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


# ---------------------------------------------------------------------------
# Boto session / caller resolution
# ---------------------------------------------------------------------------
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
                f"IAM role '{provided_role}' does not exist. Create it first "
                f"(see IamRoleResolver().create_execution_role) or pass an existing role."
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


# ---------------------------------------------------------------------------
# Read-only permission / trust validation
# ---------------------------------------------------------------------------
def _simulate_denied_actions(iam_client, role_arn: str, actions: List[str]) -> List[str]:
    """Return the subset of ``actions`` that ``role_arn`` is NOT allowed to perform.

    Wraps the paginated ``iam:SimulatePrincipalPolicy`` call so both the role-
    validation path and the HyperPod caller-side check share one implementation.
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


def _evaluate_permissions(
    iam_client, role_arn: str, role_type: str
) -> Tuple[Optional[bool], List[str]]:
    """Evaluate whether a role has the required permissions for an operation.

    Only ``*``-resource "smoke test" actions are simulated (see
    ``_get_smoke_test_actions``) so resource-scoped actions don't produce false
    denials when simulated without ResourceArns.

    Returns a ``(verdict, denied_actions)`` tuple:
        verdict True  — all gated actions are allowed (denied_actions empty).
        verdict False — at least one gated action is denied (denied_actions lists them).
        verdict None  — could not be determined, e.g. the caller lacks
                        iam:SimulatePrincipalPolicy (denied_actions empty).
    """
    required_actions = _get_smoke_test_actions(role_type)
    if not required_actions:
        return True, []

    try:
        denied = _simulate_denied_actions(iam_client, role_arn, required_actions)
        if denied:
            logger.info(
                "Role '%s' is missing permissions for: %s",
                role_arn,
                ", ".join(denied[:5]) + ("..." if len(denied) > 5 else ""),
            )
            return False, denied
        return True, []

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AccessDenied", "AccessDeniedException"):
            # Cannot simulate — verdict is unknown.
            logger.info(
                "Cannot simulate policies for '%s' (access denied); "
                "permission verdict unknown.",
                role_arn,
            )
            return None, []
        if error_code in ("NoSuchEntity", "NoSuchEntityException"):
            return False, []
        raise


def _role_has_sufficient_permissions(
    iam_client, role_arn: str, role_type: str
) -> Optional[bool]:
    """Return True/False/None for whether a role has the required permissions.

    Thin wrapper over :func:`_evaluate_permissions` that drops the denied-action
    detail. Retained for callers/tests that only need the verdict.
    """
    verdict, _ = _evaluate_permissions(iam_client, role_arn, role_type)
    return verdict


def _expected_trust_services(role_type: str) -> Set[str]:
    """Return the service principals that must be able to assume a role of this type.

    Derived from the role type's trust policy in the policy config (e.g.
    ``sagemaker.amazonaws.com`` for training). These are the principals
    SageMaker assumes the role *as* when running the workload.
    """
    statement = IAM_POLICY_CONFIG[role_type]["trust_policy"]["Statement"][0]
    service = statement["Principal"]["Service"]
    if isinstance(service, (list, tuple)):
        return set(service)
    return {service}


def _trusted_services_in_document(trust_document: dict) -> Set[str]:
    """Collect the service principals allowed to ``sts:AssumeRole`` in a trust policy.

    Only ``Allow`` statements whose action grants ``sts:AssumeRole`` (directly or
    via ``sts:*`` / ``*``) contribute. Conditions are intentionally ignored: their
    presence does not change which service principals the policy is written for.
    """
    trusted: Set[str] = set()
    for statement in trust_document.get("Statement", []):
        if statement.get("Effect") != "Allow":
            continue
        actions = statement.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        if not any(action in ("sts:AssumeRole", "sts:*", "*") for action in actions):
            continue
        principal = statement.get("Principal", {})
        if not isinstance(principal, dict):
            continue
        service = principal.get("Service")
        if service is None:
            continue
        if isinstance(service, str):
            trusted.add(service)
        else:
            trusted.update(service)
    return trusted


def _role_trusts_service(iam_client, role_arn: str, role_type: str) -> Optional[bool]:
    """Check whether a role's trust policy lets the role type's service assume it.

    A role that grants every required *permission* is still unusable if its trust
    policy does not allow the service principal (e.g. ``sagemaker.amazonaws.com``)
    to assume it — ``CreateTrainingJob`` then fails at the API with
    "Could not assume role". This complements ``_role_has_sufficient_permissions``,
    which only inspects the role's permission policies, not its trust policy.

    Returns:
        True  — every required service principal may assume the role.
        False — at least one required service principal is not trusted.
        None  — could not be determined (trust document unreadable, or the caller
                lacks permission to read it). Callers should not block on None.
    """
    expected = _expected_trust_services(role_type)
    if not expected:
        return True

    role_name = role_arn.split(":role/")[1].split("/")[-1]
    try:
        role = iam_client.get_role(RoleName=role_name)["Role"]
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AccessDenied", "AccessDeniedException"):
            return None
        if error_code in ("NoSuchEntity", "NoSuchEntityException"):
            return False
        raise

    trust_document = role.get("AssumeRolePolicyDocument")
    if not trust_document:
        # Trust policy not returned — verdict unknown; don't block reuse.
        return None
    # IAM may URL-encode the stored document; normalize before inspecting.
    if isinstance(trust_document, str):
        trust_document = json.loads(unquote(trust_document))

    trusted = _trusted_services_in_document(trust_document)
    return expected.issubset(trusted)


def _build_validation_error_message(
    role_arn: Optional[str],
    role_type: str,
    missing_actions: Optional[List[str]] = None,
    trust_failed: bool = False,
) -> str:
    """Build the actionable message for :class:`RoleValidationError`.

    Lists the specific missing actions when known (falling back to the full
    required-action list), the required trust principal on a trust failure, and
    the two remediation paths: grant the permissions, or create a dedicated role.
    """
    principal = _trust_principal_for(role_type)
    if role_arn:
        lines = [f"IAM role '{role_arn}' cannot be used for '{role_type}' workloads."]
    else:
        lines = [
            "No IAM role could be resolved from your caller identity for "
            f"'{role_type}' workloads (you may be running as an IAM user or root)."
        ]

    if trust_failed:
        lines.append(
            f"Its trust policy does not allow '{principal}' to assume it, so "
            f"SageMaker cannot use it (the API would fail with 'Could not assume role')."
        )

    if missing_actions:
        lines.append("Missing permissions: " + ", ".join(sorted(set(missing_actions))))
    else:
        lines.append(
            "Required permissions: "
            + ", ".join(sorted(set(_get_required_actions(role_type))))
        )

    lines += [
        "",
        "To fix this, either:",
        f"  1. Grant the permissions above (and ensure '{principal}' is a trusted "
        "principal) on a role, then pass it via the interface's 'role'/'role_arn' "
        "parameter. If you cannot modify or create IAM roles, ask your "
        "administrator to do this; or",
        "  2. Create a dedicated least-privilege role and pass its ARN:",
        "       from sagemaker.core.helper import IamRoleResolver",
        f"       role_arn = IamRoleResolver().create_execution_role(role_type='{role_type}')",
    ]
    return "\n".join(lines)


def resolve_and_validate_role(
    provided_role: Optional[str],
    role_type: str,
    sagemaker_session=None,
) -> str:
    """Resolve the role to use and validate it (read-only; does not mutate IAM).

    Resolution:
        1. ``provided_role`` given → resolve it to an ARN (must exist).
        2. Otherwise → resolve the caller's own identity role.

    The resolved role is then VALIDATED (read-only, via iam:SimulatePrincipalPolicy
    + trust inspection):
        * permissions allowed AND trusted → return the ARN.
        * a required permission is definitively denied → raise RoleValidationError.
        * the trust policy definitively excludes the service → raise RoleValidationError.
        * permissions cannot be verified (caller lacks iam:SimulatePrincipalPolicy,
          the common Studio/notebook case) → return the ARN with a WARNING.

    Args:
        provided_role: User-supplied role name or ARN. If set, used directly.
        role_type: One of ROLE_TYPES.
        sagemaker_session: SageMaker session (used to get the boto session).

    Returns:
        IAM role ARN.

    Raises:
        RoleValidationError: The resolved role lacks required permissions/trust,
            or no role could be resolved.
        ValueError: ``provided_role`` doesn't exist, or ``role_type`` is invalid.
    """
    if role_type not in ROLE_TYPES:
        raise ValueError(f"Invalid role_type '{role_type}'. Must be one of: {ROLE_TYPES}")

    boto_session = _get_boto_session(sagemaker_session)
    iam_client = boto_session.client("iam")

    if provided_role:
        role_arn = _resolve_explicit_role(provided_role, sagemaker_session)
    else:
        sts_client = boto_session.client("sts")
        caller_identity = sts_client.get_caller_identity()
        caller_arn = caller_identity["Arn"]
        account_id = caller_identity["Account"]
        partition = _partition_from_arn(caller_arn)
        role_arn = _resolve_caller_role_arn(iam_client, caller_arn, account_id, partition)
        if not role_arn:
            raise RoleValidationError(_build_validation_error_message(None, role_type))

    # Permission check (definitive denial blocks; unverifiable warns).
    verdict, denied = _evaluate_permissions(iam_client, role_arn, role_type)
    if verdict is False:
        raise RoleValidationError(
            _build_validation_error_message(role_arn, role_type, missing_actions=denied)
        )

    # Trust check (only a definitive "not trusted" blocks).
    trusts_service = _role_trusts_service(iam_client, role_arn, role_type)
    if trusts_service is False:
        raise RoleValidationError(
            _build_validation_error_message(role_arn, role_type, trust_failed=True)
        )

    if verdict is None:
        logger.warning(
            "Could not verify permissions for role '%s' (caller lacks "
            "iam:SimulatePrincipalPolicy). Proceeding with it. If the operation "
            "later fails with an access-denied error, ensure the role has the "
            "required permissions for '%s' (see "
            "IamRoleResolver().get_required_actions('%s')) or create a dedicated "
            "role via IamRoleResolver().create_execution_role(role_type='%s').",
            role_arn,
            role_type,
            role_type,
            role_type,
        )
    else:
        logger.info("Role '%s' validated for %s. Using it.", role_arn, role_type)
    return role_arn


def verify_hyperpod_connect_permissions(
    sagemaker_session=None, cluster_name: Optional[str] = None
) -> Optional[bool]:
    """Verify the caller can drive the HyperPod CLI; warn (don't block) if not.

    The HyperPod CLI (``connect-cluster`` + ``start-job``) runs locally under the
    *caller's* credentials, not under the job execution role. This function
    simulates the connect actions on the caller identity and surfaces actionable
    guidance when they are missing.

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


def verify_evaluation_caller_permissions(
    sagemaker_session=None,
) -> Optional[bool]:
    """Verify the caller can orchestrate SageMaker Pipeline-based evaluations.

    The evaluate module submits work via SageMaker Pipelines — creating, updating,
    starting, and describing pipelines and their executions. These actions run under
    the *caller's* credentials (the notebook user, Lambda, or CI role), NOT under
    the job execution role passed to the pipeline. This function simulates the
    required pipeline-orchestration actions on the caller identity and raises
    :class:`RoleValidationError` when they are missing.

    Args:
        sagemaker_session: SageMaker session (used to get the boto session).

    Returns:
        True  — all evaluation caller actions are allowed.
        None  — could not be determined (caller is not a role, or cannot simulate).

    Raises:
        RoleValidationError: If permissions are definitively denied.
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
        logger.info(
            "Could not resolve a caller role to verify evaluation pipeline "
            "permissions; errors will surface at pipeline creation time."
        )
        return None

    try:
        denied = _simulate_denied_actions(
            iam_client, caller_role_arn, list(EVALUATION_CALLER_ACTIONS)
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AccessDenied", "AccessDeniedException"):
            logger.info(
                "Cannot simulate evaluation caller permissions for '%s' (access "
                "denied to iam:SimulatePrincipalPolicy); errors will surface at "
                "pipeline creation time.",
                caller_role_arn,
            )
            return None
        raise

    if denied:
        message = (
            f"Your identity '{caller_role_arn}' is missing IAM permissions required "
            f"to orchestrate SageMaker Pipeline-based evaluations: "
            f"{', '.join(denied)}. "
            f"The evaluation execution role was resolved successfully, but creating "
            f"and starting the evaluation pipeline runs as YOUR credentials. "
            f"Grant these actions to your identity (scoped to "
            f"arn:{partition}:sagemaker:*:{account_id}:pipeline/*) or use the "
            f"AmazonSageMakerFullAccess managed policy."
        )
        raise RoleValidationError(message)

    logger.info(
        "Caller '%s' has the evaluation pipeline orchestration permissions.",
        caller_role_arn,
    )
    return True


# ---------------------------------------------------------------------------
# Opt-in IAM execution-role creation.
#
# Everything above is read-only. The IamRoleResolver class below is the ONLY
# place in the SDK that writes IAM (create/attach/tag/delete). Default SDK paths
# never instantiate it; provisioning a managed least-privilege role is a
# deliberate action the user takes by calling IamRoleResolver().create_execution_role(...).
# ---------------------------------------------------------------------------
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

# Ownership tags applied to every role the SDK creates, so the roles are
# auditable and clearly attributable to the SDK. ``RoleType`` (filled in per
# role) records which workload the role was created for.
_SDK_ROLE_TAGS = {
    "CreatedBy": "sagemaker-python-sdk",
}


class IamRoleResolver:
    """Create and manage SageMaker execution roles (explicit, opt-in).

    Instantiate with an optional SageMaker session and call
    :meth:`create_execution_role` to provision a managed least-privilege role for
    a workload type, or :meth:`delete_execution_role` to remove one. This class is
    the only place in the SDK that writes IAM.

    Example::

        from sagemaker.core.helper import IamRoleResolver

        resolver = IamRoleResolver()
        role_arn = resolver.create_execution_role(
            role_type="training", s3_resource="my-bucket"
        )
        # ... pass role_arn into SFTTrainer(role=role_arn) / ModelBuilder(...) etc.
    """

    def __init__(self, sagemaker_session=None):
        self._sagemaker_session = sagemaker_session
        self._boto_session = _get_boto_session(sagemaker_session)
        self._iam_client = self._boto_session.client("iam")
        self._sts_client = self._boto_session.client("sts")

    # -- public API ---------------------------------------------------------
    def get_required_actions(self, role_type: str) -> List[str]:
        """Return the IAM actions a role of ``role_type`` needs (read-only preview)."""
        self._validate_role_type(role_type)
        return _get_required_actions(role_type)

    def create_execution_role(
        self,
        role_type: str,
        *,
        role_name: Optional[str] = None,
        s3_resource: Union[str, List[str]] = "*",
        kms_resource: Union[str, List[str]] = "*",
        update_if_exists: bool = True,
    ) -> str:
        """Create (or update) a managed least-privilege execution role.

        Args:
            role_type: One of ROLE_TYPES.
            role_name: Role name to create. Defaults to the well-known
                ``SageMaker-AutoRole-<Type>`` name from the policy config.
            s3_resource: S3 bucket name(s) to scope S3 policies. "*" (default)
                grants account-wide S3 access and emits a WARNING.
            kms_resource: KMS key id(s) to scope KMS policies. "*" (default)
                grants account-wide KMS access and emits a WARNING.
            update_if_exists: If the role already exists, attach/refresh its
                policies (idempotent). If False and the role exists, return its ARN
                without modifying policies.

        Returns:
            The created/existing role ARN.

        Raises:
            RoleAutoCreationError: The caller lacks the IAM permissions to create
                the role/policies.
            ValueError: ``role_type`` is invalid.
        """
        self._validate_role_type(role_type)
        role_config = IAM_POLICY_CONFIG[role_type]
        target_role_name = role_name or role_config["role_name"]

        caller_identity = self._sts_client.get_caller_identity()
        account_id = caller_identity["Account"]
        partition = _partition_from_arn(caller_identity["Arn"])

        self._warn_broad_permissions(role_type, role_config, s3_resource, kms_resource)

        policies = _replace_placeholders(
            role_config["policies"], s3_resource, kms_resource, partition, account_id
        )
        trust_policy = self._scope_trust_policy_to_account(
            role_config["trust_policy"], account_id
        )

        try:
            role_arn = self._create_or_get_role(
                target_role_name, trust_policy, role_type
            )
            if update_if_exists:
                self._ensure_policies_attached(
                    target_role_name, policies, account_id, partition
                )
            logger.info("Waiting %ds for IAM propagation...", _IAM_PROPAGATION_DELAY_SECONDS)
            time.sleep(_IAM_PROPAGATION_DELAY_SECONDS)
            logger.info("Using role: %s", role_arn)
            return role_arn
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("AccessDenied", "AccessDeniedException", "UnauthorizedAccess"):
                self._raise_auto_creation_error(target_role_name, e, role_type)
            raise

    def delete_execution_role(
        self, role_type: str, *, role_name: Optional[str] = None
    ) -> None:
        """Delete a role created by :meth:`create_execution_role` and its policies.

        Idempotent and best-effort: detaches and deletes the SDK-managed policies,
        then deletes the role. Missing entities are ignored so re-running is safe.
        Useful for notebook/test cleanup.
        """
        self._validate_role_type(role_type)
        target_role_name = role_name or IAM_POLICY_CONFIG[role_type]["role_name"]
        iam = self._iam_client

        try:
            attached = iam.list_attached_role_policies(RoleName=target_role_name).get(
                "AttachedPolicies", []
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") in (
                "NoSuchEntity",
                "NoSuchEntityException",
            ):
                return
            raise

        for policy in attached:
            policy_arn = policy["PolicyArn"]
            try:
                iam.detach_role_policy(RoleName=target_role_name, PolicyArn=policy_arn)
            except ClientError:
                pass
            # Only delete policies the SDK created for this role (prefix match),
            # never customer-managed/AWS-managed policies attached separately.
            if policy["PolicyName"].startswith(f"{target_role_name}-"):
                self._delete_policy_and_versions(policy_arn)

        try:
            iam.delete_role(RoleName=target_role_name)
            logger.info("Deleted role '%s'.", target_role_name)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") not in (
                "NoSuchEntity",
                "NoSuchEntityException",
            ):
                raise

    # -- internal write helpers --------------------------------------------
    @staticmethod
    def _validate_role_type(role_type: str) -> None:
        if role_type not in ROLE_TYPES:
            raise ValueError(
                f"Invalid role_type '{role_type}'. Must be one of: {ROLE_TYPES}"
            )

    @staticmethod
    def _build_role_tags(role_type: str) -> List[dict]:
        """Build the SDK ownership tag list (IAM TagList form) for a role type."""
        tags = dict(_SDK_ROLE_TAGS)
        tags["RoleType"] = role_type
        return [{"Key": k, "Value": v} for k, v in tags.items()]

    @staticmethod
    def _scope_trust_policy_to_account(trust_policy: dict, account_id: str) -> dict:
        """Substitute ACCOUNT_PLACEHOLDER in trust-policy conditions with the account ID.

        The ``aws:SourceAccount`` condition restricts which account's service
        requests may assume the role, preventing the cross-service confused-deputy
        problem.
        """
        import copy

        trust_policy = copy.deepcopy(trust_policy)
        for statement in trust_policy.get("Statement", []):
            condition = statement.get("Condition", {})
            for operator_values in condition.values():
                if "aws:SourceAccount" in operator_values:
                    operator_values["aws:SourceAccount"] = account_id
        return trust_policy

    def _raise_auto_creation_error(
        self, role_name: str, original_error: ClientError, role_type: str
    ) -> None:
        """Raise a RoleAutoCreationError with actionable remediation guidance."""
        raise RoleAutoCreationError(
            f"Cannot create IAM role '{role_name}'.\n\n"
            f"Your IAM identity needs: {', '.join(_REQUIRED_IAM_ACTIONS)}\n\n"
            f"Alternatively, create a role manually and pass it via the 'role' parameter.\n"
            f"Required trust principal: {_trust_principal_for(role_type)}\n\n"
            f"Original error: {original_error}"
        ) from original_error

    def _create_or_get_role(
        self, role_name: str, trust_policy: dict, role_type: str
    ) -> str:
        """Create the role, or reuse it if it already exists. Returns the ARN."""
        iam = self._iam_client
        try:
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=(
                    f"Created by SageMaker Python SDK for {role_type} workloads. "
                    f"Safe to delete if no longer needed."
                ),
                Tags=self._build_role_tags(role_type),
            )
            role_arn = response["Role"]["Arn"]
            logger.warning(
                "SageMaker Python SDK created a new IAM role '%s' (%s) in your AWS "
                "account for %s workloads. It is safe to delete if no longer needed.",
                role_name,
                role_arn,
                role_type,
            )
            return role_arn
        except iam.exceptions.EntityAlreadyExistsException:
            # Created concurrently or pre-existing — reuse it (policy attachment
            # below is idempotent). Ensure ownership tags are present.
            logger.info("Role '%s' already exists, reusing.", role_name)
            role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
            self._ensure_role_tagged(role_name, role_type)
            return role_arn

    def _ensure_role_tagged(self, role_name: str, role_type: str) -> None:
        """Idempotently ensure a role carries the SDK ownership tags."""
        iam = self._iam_client
        try:
            existing = {
                t["Key"] for t in iam.list_role_tags(RoleName=role_name).get("Tags", [])
            }
            desired = self._build_role_tags(role_type)
            missing = [t for t in desired if t["Key"] not in existing]
            if missing:
                iam.tag_role(RoleName=role_name, Tags=desired)
                logger.info("Applied SDK ownership tags to role '%s'.", role_name)
        except ClientError as e:
            logger.info(
                "Could not verify/apply ownership tags on role '%s': %s", role_name, e
            )

    def _get_attached_policy_names(self, role_name: str) -> Set[str]:
        """Return the set of policy names already attached to a role (lowercased)."""
        attached = self._iam_client.list_attached_role_policies(RoleName=role_name)
        return {p["PolicyName"].lower() for p in attached.get("AttachedPolicies", [])}

    def _policy_document_matches(self, policy_arn: str, desired_document: dict) -> bool:
        """Return True if the policy's default version matches the desired document."""
        iam = self._iam_client
        try:
            policy = iam.get_policy(PolicyArn=policy_arn)
            default_version_id = policy["Policy"]["DefaultVersionId"]
            version = iam.get_policy_version(
                PolicyArn=policy_arn, VersionId=default_version_id
            )
            current_document = version["PolicyVersion"]["Document"]
        except ClientError:
            return False
        if isinstance(current_document, str):
            current_document = json.loads(unquote(current_document))
        return current_document == desired_document

    def _update_policy_document(self, policy_arn: str, policy_document: dict) -> None:
        """Create a new default policy version, pruning the oldest if at the limit."""
        iam = self._iam_client
        versions = iam.list_policy_versions(PolicyArn=policy_arn).get("Versions", [])
        if len(versions) >= _MAX_POLICY_VERSIONS:
            oldest = min(
                (v for v in versions if not v["IsDefaultVersion"]),
                key=lambda v: v["VersionId"],
            )
            iam.delete_policy_version(PolicyArn=policy_arn, VersionId=oldest["VersionId"])
        iam.create_policy_version(
            PolicyArn=policy_arn,
            PolicyDocument=json.dumps(policy_document),
            SetAsDefault=True,
        )

    def _delete_policy_and_versions(self, policy_arn: str) -> None:
        """Delete a managed policy and all its non-default versions (best-effort)."""
        iam = self._iam_client
        try:
            versions = iam.list_policy_versions(PolicyArn=policy_arn).get("Versions", [])
            for v in versions:
                if not v["IsDefaultVersion"]:
                    iam.delete_policy_version(PolicyArn=policy_arn, VersionId=v["VersionId"])
            iam.delete_policy(PolicyArn=policy_arn)
        except ClientError:
            pass

    def _ensure_policies_attached(
        self, role_name: str, policies: dict, account_id: str, partition: str = "aws"
    ) -> None:
        """Create, update, and attach the role's least-privilege policies.

        Idempotent on policy *content*: if a managed policy already exists but its
        document has drifted, a new default version is created so the role picks up
        the change.
        """
        iam = self._iam_client
        already_attached = self._get_attached_policy_names(role_name)

        created, updated, attached = [], [], []
        for policy_name, policy_document in policies.items():
            managed_policy_name = f"{role_name}-{policy_name}"
            policy_arn = f"arn:{partition}:iam::{account_id}:policy/{managed_policy_name}"

            try:
                iam.create_policy(
                    PolicyName=managed_policy_name,
                    PolicyDocument=json.dumps(policy_document),
                )
                created.append(managed_policy_name)
            except iam.exceptions.EntityAlreadyExistsException:
                if self._policy_document_matches(policy_arn, policy_document):
                    logger.info("Policy %s is up to date", managed_policy_name)
                else:
                    self._update_policy_document(policy_arn, policy_document)
                    updated.append(managed_policy_name)

            if managed_policy_name.lower() not in already_attached:
                iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
                attached.append(managed_policy_name)

        if created:
            logger.warning(
                "SageMaker Python SDK created %d IAM managed %s in your AWS account "
                "and attached %s to role '%s': %s",
                len(created),
                "policy" if len(created) == 1 else "policies",
                "it" if len(created) == 1 else "them",
                role_name,
                ", ".join(created),
            )
        if updated:
            logger.warning(
                "SageMaker Python SDK updated %d existing IAM managed %s on role "
                "'%s' to the latest least-privilege document: %s",
                len(updated),
                "policy" if len(updated) == 1 else "policies",
                role_name,
                ", ".join(updated),
            )
        reattached = [name for name in attached if name not in created]
        if reattached:
            logger.warning(
                "SageMaker Python SDK attached %d existing IAM managed %s to role "
                "'%s': %s",
                len(reattached),
                "policy" if len(reattached) == 1 else "policies",
                role_name,
                ", ".join(reattached),
            )

    def _warn_broad_permissions(
        self,
        role_type: str,
        role_config: dict,
        s3_resource: Union[str, List[str]],
        kms_resource: Union[str, List[str]],
    ) -> None:
        """Warn when the role will be granted account-wide ("*") resource access."""
        if self._role_uses_placeholder(role_config, "S3_PLACEHOLDER") and self._is_wildcard_scope(
            s3_resource
        ):
            logger.warning(
                "SageMaker Python SDK is granting the '%s' role access to ALL S3 "
                "buckets in your account (s3 resource scope is '*'). To restrict "
                "this, pass s3_resource=<bucket name(s)>.",
                role_type,
            )
        if self._role_uses_placeholder(role_config, "KMS_PLACEHOLDER") and self._is_wildcard_scope(
            kms_resource
        ):
            logger.warning(
                "SageMaker Python SDK is granting the '%s' role access to ALL KMS "
                "keys in your account (kms resource scope is '*'). To restrict this, "
                "pass kms_resource=<key id(s)>.",
                role_type,
            )
        if role_type == "feature_store":
            logger.warning(
                "SageMaker Python SDK is granting the 'feature_store' role read-only "
                "access to ALL AWS Glue catalog metadata in your account "
                "(glue:GetTable/GetDatabase/GetPartitions on '*'). This metadata grant "
                "is required because the specific Glue databases/tables are not known "
                "at role-creation time."
            )

    @staticmethod
    def _is_wildcard_scope(resource: Union[str, List[str]]) -> bool:
        """Return True if a resource scope resolves to the account-wide "*" wildcard."""
        if isinstance(resource, str):
            return resource == "*"
        return "*" in resource

    @staticmethod
    def _role_uses_placeholder(role_config: dict, placeholder: str) -> bool:
        """Return True if any policy statement uses ``placeholder`` as its Resource."""
        for policy_doc in role_config["policies"].values():
            for statement in policy_doc.get("Statement", []):
                if statement.get("Resource") == placeholder:
                    return True
        return False
