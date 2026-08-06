"""Pre-validation for execution role permissions required by evaluation jobs.

Validates that the execution role has the permissions and trust relationships
needed to run model evaluation jobs, providing actionable error messages at
submission time instead of failing 30+ minutes later during execution.

All validation is best-effort: if the caller lacks iam:SimulatePrincipalPolicy
(common in Studio/notebooks), a warning is logged and execution proceeds.

This module reuses the IAM simulation helpers from iam_role_resolver.py.
"""

import logging
from typing import List, Optional

from botocore.exceptions import ClientError

from sagemaker.core.helper.iam_role_resolver import (
    _simulate_denied_actions,
    _role_trusts_service,
)

logger = logging.getLogger(__name__)

_PREREQS_DOC_URL = (
    "https://docs.aws.amazon.com/sagemaker/latest/dg/"
    "model-customize-open-weight-prereq.html"
)

# Bedrock actions required on the execution role for evaluation jobs
_BEDROCK_EVAL_ACTIONS = [
    "bedrock:CreateEvaluationJob",
    "bedrock:GetEvaluationJob",
]

# MLflow actions required when experiment tracking is enabled
_MLFLOW_ACTIONS = [
    "sagemaker-mlflow:GetExperimentByName",
    "sagemaker-mlflow:CreateExperiment",
    "sagemaker-mlflow:CreateRun",
    "sagemaker-mlflow:LogBatch",
]


def validate_evaluation_role_permissions(
    role_arn: str,
    sagemaker_session,
    mlflow_enabled: bool = False,
) -> None:
    """Validate that the execution role has permissions required for evaluation.

    Checks that the execution role can:
      1. Call bedrock:CreateEvaluationJob and bedrock:GetEvaluationJob.
      2. Be assumed by bedrock.amazonaws.com (trust relationship).
      3. (If mlflow_enabled) Call sagemaker-mlflow:* APIs for experiment tracking.

    Uses the same IAM simulation infrastructure as resolve_and_validate_role().
    Best-effort: warns and proceeds if caller cannot simulate.

    Args:
        role_arn: The resolved execution role ARN.
        sagemaker_session: SageMaker session (used to get boto session).
        mlflow_enabled: Whether MLflow tracking is configured.

    Raises:
        ValueError: If the role definitively lacks required permissions or trust.
    """
    iam_client = _get_iam_client(sagemaker_session)
    if iam_client is None:
        return

    # 1. Check Bedrock permissions
    _check_permissions(
        iam_client,
        role_arn,
        _BEDROCK_EVAL_ACTIONS,
        error_context=(
            "Model evaluation requires these permissions on the execution role "
            "to create and monitor Bedrock evaluation jobs."
        ),
    )

    # 2. Check trust relationship for bedrock.amazonaws.com
    _check_trust(iam_client, role_arn)

    # 3. Check MLflow permissions if tracking is enabled
    if mlflow_enabled:
        _check_permissions(
            iam_client,
            role_arn,
            _MLFLOW_ACTIONS,
            error_context=(
                "MLflow experiment tracking is enabled (mlflow_resource_arn was provided), "
                "but the execution role cannot access MLflow APIs."
            ),
        )

    logger.info("Execution role '%s' validated for evaluation.", role_arn)


def _check_permissions(
    iam_client,
    role_arn: str,
    actions: List[str],
    error_context: str,
) -> None:
    """Check that the role has the specified permissions via SimulatePrincipalPolicy."""
    try:
        denied = _simulate_denied_actions(iam_client, role_arn, actions)
        if denied:
            raise ValueError(
                f"IAM role '{role_arn}' is missing required permissions: "
                f"{', '.join(denied)}. "
                f"{error_context} "
                f"To fix this, attach the 'AmazonSageMakerModelCustomizationCoreAccess' "
                f"managed policy to your role, or add an inline policy granting "
                f"the missing actions. "
                f"See: {_PREREQS_DOC_URL}"
            )
    except ClientError as e:
        if _is_access_denied(e):
            logger.warning(
                "Could not verify permissions for role '%s' (caller lacks "
                "iam:SimulatePrincipalPolicy). If the job fails with "
                "AccessDeniedException, ensure the role has: %s",
                role_arn,
                ", ".join(actions),
            )
        elif _is_no_such_entity(e):
            pass  # Role gone; let it fail downstream with clearer error
        else:
            raise
    except ValueError:
        raise
    except Exception as e:
        logger.info("Permission check failed unexpectedly: %s; skipping.", e)


def _check_trust(iam_client, role_arn: str) -> None:
    """Check that the role trusts bedrock.amazonaws.com."""
    try:
        trusts_bedrock = _role_trusts_service(iam_client, role_arn, "bedrock")
        if trusts_bedrock is False:
            raise ValueError(
                f"IAM role '{role_arn}' trust policy does not include "
                f"'bedrock.amazonaws.com'. Model evaluation requires "
                f"Bedrock to assume your execution role to run the evaluation job. "
                f"Add the following trust statement to your role:\n"
                f'{{"Effect": "Allow", "Principal": {{"Service": '
                f'"bedrock.amazonaws.com"}}, "Action": "sts:AssumeRole"}}\n'
                f"See: {_PREREQS_DOC_URL}"
            )
    except ClientError as e:
        if _is_access_denied(e):
            logger.warning(
                "Could not verify trust policy for role '%s'. If the evaluation "
                "fails with 'Could not assume role', ensure bedrock.amazonaws.com "
                "is in the role's trust policy.",
                role_arn,
            )
        elif _is_no_such_entity(e):
            pass
        else:
            raise
    except ValueError:
        raise
    except Exception as e:
        logger.info("Trust check failed unexpectedly: %s; skipping.", e)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_iam_client(sagemaker_session):
    """Get an IAM client from the session, or None if unavailable."""
    import boto3

    try:
        if sagemaker_session and hasattr(sagemaker_session, 'boto_session'):
            return sagemaker_session.boto_session.client("iam")
        return boto3.Session().client("iam")
    except Exception:
        logger.info("Could not create IAM client for role validation; skipping.")
        return None


def _is_access_denied(error) -> bool:
    code = error.response.get("Error", {}).get("Code", "")
    return code in ("AccessDenied", "AccessDeniedException")


def _is_no_such_entity(error) -> bool:
    code = error.response.get("Error", {}).get("Code", "")
    return code in ("NoSuchEntity", "NoSuchEntityException")
