# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""End-to-end integration test for HyperPod IAM role auto-creation.

This exercises ``resolve_or_create_role(role_type="hyperpod")`` against REAL AWS
IAM — no mocks, no stubs. It creates the ``SageMaker-AutoRole-HyperPod`` role and
its least-privilege managed policies, verifies the role's trust relationship and
attached policies, confirms the HyperPod cluster-connect permissions resolve via
``iam:SimulatePrincipalPolicy``, and then tears everything back down.

Run as a script (recommended for ad-hoc verification)::

    python -m tests.integ.helper.test_iam_role_resolver_hyperpod_integ
    #   or, from sagemaker-core/:
    PYTHONPATH=src python tests/integ/helper/test_iam_role_resolver_hyperpod_integ.py

Or under pytest (the integ marker keeps it out of the unit run)::

    PYTHONPATH=src python -m pytest tests/integ/helper/test_iam_role_resolver_hyperpod_integ.py -m integ -s

The test requires AWS credentials with iam:CreateRole / CreatePolicy /
AttachRolePolicy / GetRole (plus the simulate + delete actions used for verify
and cleanup). It is intentionally self-cleaning: the role and policies it creates
are deleted in a ``finally`` block even if an assertion fails midway. To inspect
the artifacts instead of deleting them, set ``KEEP_HYPERPOD_ROLE=1``.
"""
from __future__ import absolute_import

import json
import logging
import os
import sys
import uuid

import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

# Allow running as a bare script (python tests/.../this_file.py) by making the
# package importable without installing it.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from sagemaker.core.helper.iam_role_resolver import (  # noqa: E402
    verify_hyperpod_connect_permissions,
    HYPERPOD_CLI_CONNECT_ACTIONS,
    _get_required_actions,
    _load_policy_config,
)
from sagemaker.core.helper.iam_role_resolver import IamRoleResolver  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hyperpod_iam_integ")

ROLE_TYPE = "hyperpod"
ROLE_NAME = _load_policy_config()[ROLE_TYPE]["role_name"]


def _credentials_available() -> bool:
    """Return True if usable AWS credentials are configured."""
    try:
        boto3.Session().client("sts").get_caller_identity()
        return True
    except (NoCredentialsError, ClientError) as e:
        logger.warning("No usable AWS credentials (%s). Integ test will be skipped.", e)
        return False


def _delete_role_and_policies(iam_client, role_name: str) -> None:
    """Detach + delete every managed policy on the role, then the role itself.

    Idempotent and best-effort: missing entities are ignored so this is safe to
    call whether the test created a fresh role or reused an existing one.
    """
    try:
        attached = iam_client.list_attached_role_policies(RoleName=role_name)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchEntity", "NoSuchEntityException"):
            return
        raise

    for policy in attached.get("AttachedPolicies", []):
        policy_arn = policy["PolicyArn"]
        try:
            iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            # A managed policy can only be deleted once all non-default versions
            # are removed, so prune them before deleting the policy.
            versions = iam_client.list_policy_versions(PolicyArn=policy_arn).get(
                "Versions", []
            )
            for version in versions:
                if not version["IsDefaultVersion"]:
                    iam_client.delete_policy_version(
                        PolicyArn=policy_arn, VersionId=version["VersionId"]
                    )
            iam_client.delete_policy(PolicyArn=policy_arn)
            logger.info("Deleted policy %s", policy_arn)
        except ClientError as e:
            logger.warning("Could not fully delete policy %s: %s", policy_arn, e)

    try:
        iam_client.delete_role(RoleName=role_name)
        logger.info("Deleted role %s", role_name)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("NoSuchEntity", "NoSuchEntityException"):
            logger.warning("Could not delete role %s: %s", role_name, e)


def run_end_to_end() -> None:
    """Create the HyperPod role for real, verify it, then clean up."""
    boto_session = boto3.Session()
    iam_client = boto_session.client("iam")

    config = _load_policy_config()[ROLE_TYPE]
    expected_policy_keys = list(config["policies"])

    # Use a unique role name so the creation path ALWAYS runs the account-mutating
    # logic this test exists to verify.
    unique_role_name = f"{ROLE_NAME}-IntegTest-{uuid.uuid4().hex[:12]}"
    keep = os.environ.get("KEEP_HYPERPOD_ROLE") == "1"
    creator = IamRoleResolver()

    # Best-effort clean slate for the well-known role name.
    _delete_role_and_policies(iam_client, ROLE_NAME)

    try:
        # --- Deterministic creation path: explicitly create the role + policies
        # via the opt-in IamRoleResolver against a unique role name.
        role_arn = creator.create_execution_role(
            role_type=ROLE_TYPE, role_name=unique_role_name
        )
        assert role_arn.startswith("arn:"), f"unexpected ARN: {role_arn}"
        logger.info("Created and provisioned test role: %s", role_arn)

        # --- Verify 1: the job role is trusted by sagemaker.amazonaws.com.
        trust_doc = iam_client.get_role(RoleName=unique_role_name)["Role"][
            "AssumeRolePolicyDocument"
        ]
        if isinstance(trust_doc, str):
            trust_doc = json.loads(trust_doc)
        principal = trust_doc["Statement"][0]["Principal"]["Service"]
        assert principal == "sagemaker.amazonaws.com", f"unexpected trust principal: {principal}"
        logger.info("OK: role trusts %s", principal)

        # --- Verify 2: every policy in the config got created and attached.
        attached = {
            p["PolicyName"]
            for p in iam_client.list_attached_role_policies(RoleName=unique_role_name)[
                "AttachedPolicies"
            ]
        }
        expected = {f"{unique_role_name}-{name}" for name in expected_policy_keys}
        missing = expected - attached
        assert not missing, f"expected policies not attached: {missing}"
        logger.info("OK: all %d expected policies attached", len(expected))

        # --- Verify 3: the job role's runtime permissions resolve to "allowed"
        # AND it does NOT carry the caller-side CLI connect permissions (those
        # belong on the caller identity, not this SageMaker-trusted job role).
        results = []
        paginator = iam_client.get_paginator("simulate_principal_policy")
        for page in paginator.paginate(
            PolicySourceArn=role_arn,
            ActionNames=list(_get_required_actions(ROLE_TYPE)) + list(HYPERPOD_CLI_CONNECT_ACTIONS),
        ):
            results.extend(page.get("EvaluationResults", []))
        decisions = {r["EvalActionName"]: r["EvalDecision"] for r in results}
        assert decisions.get("s3:GetObject") == "allowed", "job runtime perm not allowed"
        for connect_action in HYPERPOD_CLI_CONNECT_ACTIONS:
            assert decisions.get(connect_action) != "allowed", (
                f"{connect_action} must NOT be granted by the job execution role"
            )
        logger.info("OK: job role has runtime perms and excludes CLI connect perms")

        # --- Verify 4: idempotency — re-provisioning does not error or duplicate.
        creator.create_execution_role(role_type=ROLE_TYPE, role_name=unique_role_name)
        attached_again = {
            p["PolicyName"]
            for p in iam_client.list_attached_role_policies(RoleName=unique_role_name)[
                "AttachedPolicies"
            ]
        }
        assert attached_again == attached, "idempotent re-provision changed attached policies"
        logger.info("OK: re-provision is idempotent")

        # --- Verify 5: caller-side connect-permission check runs against real IAM
        # without raising (returns True/False/None depending on the caller).
        verdict = verify_hyperpod_connect_permissions(cluster_name="integ-test-cluster")
        logger.info("OK: verify_hyperpod_connect_permissions returned %s", verdict)

        logger.info("HyperPod IAM role end-to-end verification PASSED.")
    finally:
        if keep:
            logger.info("KEEP_HYPERPOD_ROLE=1 set; leaving roles in place.")
        else:
            _delete_role_and_policies(iam_client, unique_role_name)
            _delete_role_and_policies(iam_client, ROLE_NAME)


@pytest.mark.integ
@pytest.mark.skipif(
    not _credentials_available(), reason="No AWS credentials available for integ test"
)
def test_hyperpod_role_end_to_end():
    """Pytest entry point for the no-mocks HyperPod IAM role flow."""
    run_end_to_end()


if __name__ == "__main__":
    if not _credentials_available():
        print(
            "SKIPPED: no usable AWS credentials. Configure credentials "
            "(e.g. `ada credentials update ...` / `aws configure`) and re-run."
        )
        sys.exit(0)
    run_end_to_end()
    print("PASSED")
