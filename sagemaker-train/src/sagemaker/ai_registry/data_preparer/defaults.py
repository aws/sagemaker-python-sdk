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
"""Schema-driven intelligent defaults for DataPreparer."""

from __future__ import absolute_import

import copy
import logging
import re
import time
from typing import Any, Callable, Dict, Optional

from sagemaker.ai_registry.data_preparer.constants import (
    DATA_PREPARER_GLOBAL_PATH,
    MAX_JOB_NAME_LENGTH,
    JobCategory,
    data_preparer_job_path,
)
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerConfigurationError,
)
from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)

# Smithy JobName pattern
_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](-*[A-Za-z0-9]){0,62}$")

# Inserts an underscore at every PascalCase boundary so e.g.
#   "OutputS3Uri" → "Output_S3_Uri" → "output_s3_uri"
# Matches the convention in _schema_resolver._pascal_to_snake; duplicated
# here to avoid a cross-module import cycle.
_PASCAL_TO_SNAKE_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Counter for sub-millisecond collision avoidance
_job_name_counter = 0


def _fallback_role_arn(
    category: JobCategory, session: Session, payload: Dict[str, Any], config_mgr=None
) -> Optional[str]:
    """Well-known fallback for RoleArn: TrainingJob config → get_execution_role.

    Per Req 4.5 + design §1.3.5: if get_execution_role fails to resolve a role,
    raise DataPreparerConfigurationError rather than swallowing the exception so
    callers get actionable context.
    """
    if config_mgr is not None:
        role = config_mgr.resolve_value_from_config(
            config_path="SageMaker.TrainingJob.RoleArn",
            sagemaker_session=session,
        )
        if role:
            return role
    try:
        from sagemaker.core.helper.session_helper import get_execution_role

        return get_execution_role(session)
    except Exception as e:
        raise DataPreparerConfigurationError(
            "Could not resolve RoleArn. Provide role= on DataPreparer, set "
            "SageMaker.TrainingJob.RoleArn in SageMakerConfig, or ensure "
            "get_execution_role() is available."
        ) from e


def _fallback_default_s3_uri(
    category: JobCategory, session: Session, payload: Dict[str, Any], config_mgr=None
) -> Optional[str]:
    """Well-known fallback for the schema's output-location property.

    Registered under both spellings of that property: SDG and DataQualityEvaluation
    v1 call it ``OutputDataConfig.OutputS3Uri``, while DataTransformation and
    DataQualityEvaluation v2 call it ``OutputDataConfig.S3OutputPath``. The
    resolved value is identical either way — only the schema field name differs.
    """
    from sagemaker.train.utils import _default_s3_uri

    base_job_name = DataPreparerDefaults.get_base_job_name(payload, category)
    return _default_s3_uri(session, f"DataPreparer/{category.value}/{base_job_name}")


def _fallback_hub_name(
    category: JobCategory, session: Session, payload: Dict[str, Any], config_mgr=None
) -> Optional[str]:
    """Well-known fallback for OutputDataConfig.HubName.

    NOTE: Removed from _WELL_KNOWN_FALLBACKS per Req 4.4 (SHALL NOT apply well-known
    fallbacks for any path other than RoleArn and OutputDataConfig.OutputS3Uri).
    Kept as a private helper in case the hardcoded fallback SDG schema needs it for
    bootstrap before Hub publication; callers must invoke it explicitly.
    """
    from sagemaker.ai_registry.air_hub import AIRHub

    return AIRHub.get_hub_name()


# Only registered well-known fallbacks per design §1.3.4 / Req 4.4.
#
# Req 4.4 enumerates RoleArn and OutputDataConfig.OutputS3Uri, which was the
# complete set of output-location spellings when only SDG and DQ v1 existed.
# DataTransformation and DQ v2 publish the same concept as
# OutputDataConfig.S3OutputPath, so it is registered here too: tier-3 lookup is
# exact-path match (see _walk_and_fill), meaning the OutputS3Uri entry alone
# silently gives those categories no default and forces every caller to pass an
# output path by hand. This widens the spelling, not the set of defaulted concepts.
_WELL_KNOWN_FALLBACKS: Dict[str, Callable] = {
    "RoleArn": _fallback_role_arn,
    "OutputDataConfig.OutputS3Uri": _fallback_default_s3_uri,
    "OutputDataConfig.S3OutputPath": _fallback_default_s3_uri,
}


def _resolve_existing(target: Dict[str, Any], prop_name: str) -> Any:
    """Return the user-supplied value for ``prop_name``.

    Checks both the PascalCase schema name and its snake_case alias. If found under the
    snake_case alias, rename the key in ``target`` to the PascalCase
    schema name so downstream serialization, recursion, and fallback
    application all observe a single canonical key.

    Returns ``None`` if neither is present.
    """
    if prop_name in target:
        return target[prop_name]
    # Convert PascalCase schema name to its snake_case alias (e.g.
    # OutputS3Uri → output_s3_uri). The alias generation here mirrors
    # _pascal_to_snake in _schema_resolver, but we keep it local to
    # avoid an import cycle.
    snake = _PASCAL_TO_SNAKE_RE.sub("_", prop_name).lower()
    snake = "".join(c if c.isalnum() or c == "_" else "_" for c in snake)
    if snake in target:
        # Promote snake-cased key to PascalCase in-place so the rest of
        # _walk_and_fill (and downstream serialization) treats it as
        # already-set under the canonical name. This avoids a second
        # snake/Pascal mismatch later in the pipeline.
        target[prop_name] = target.pop(snake)
        return target[prop_name]
    return None


class DataPreparerDefaults:
    """Static helpers for resolving intelligent defaults, mirroring TrainDefaults."""

    @staticmethod
    def get_sagemaker_session(session: Optional[Session] = None) -> Session:
        """Return the session, constructing a default if None."""
        if session is None:
            session = Session()
            logger.info("SageMaker session not provided. Using default Session.")
        return session

    @staticmethod
    def get_role(sagemaker_session: Session, config_mgr=None) -> Optional[str]:
        """Resolve the envelope-level RoleArn through the full fallback chain.

        Mirrors the intelligent-defaults ladder in §1.5 for the `RoleArn`
        property but operates at the CreateJob envelope layer (not inside the
        JobConfigDocument schema, since RoleArn is envelope-level).

        Order: Modules.DataPreparer.GlobalDefaults.RoleArn → well-known fallback
        (SageMaker.TrainingJob.RoleArn → get_execution_role). Returns None if
        none resolves; caller must handle (raise DataPreparerConfigurationError).
        """
        if config_mgr is None:
            from sagemaker.core.config.config_manager import SageMakerConfig

            config_mgr = SageMakerConfig()

        role = config_mgr.resolve_value_from_config(
            config_path=f"{DATA_PREPARER_GLOBAL_PATH}.RoleArn",
            sagemaker_session=sagemaker_session,
        )
        if role:
            logger.info("Applied default from %s.RoleArn", DATA_PREPARER_GLOBAL_PATH)
            return role

        # Delegate to the registered well-known fallback (TrainingJob.RoleArn → get_execution_role)
        return _fallback_role_arn(None, sagemaker_session, {}, config_mgr)

    @staticmethod
    def get_base_job_name(payload: Dict[str, Any], category: JobCategory) -> str:
        """Derive a base job name from the payload or category."""
        # Try well-known name fields from the payload
        for key in ("DatasetName", "EvaluationName", "dataset_name", "evaluation_name"):
            if key in payload and payload[key]:
                return str(payload[key])
        return f"{category.value}-job"

    @staticmethod
    def resolve_job_name(
        job_name: Optional[str],
        base_job_name: Optional[str],
        category: JobCategory,
        user_payload: Dict[str, Any],
    ) -> str:
        """Resolve the final JobName per the Req 5 ladder.

        Priority: job_name > base_job_name+ts > payload name+ts > category-job-ts.
        """
        if job_name:
            if not _JOB_NAME_PATTERN.match(job_name):
                raise DataPreparerConfigurationError(
                    f"job_name '{job_name}' does not match the Smithy JobName pattern "
                    f"'^[A-Za-z0-9](-*[A-Za-z0-9]){{0,62}}$'"
                )
            return job_name

        if base_job_name:
            base = base_job_name
        else:
            base = DataPreparerDefaults.get_base_job_name(user_payload, category)

        global _job_name_counter
        _job_name_counter += 1
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        ms = f"{int(time.time() * 1000) % 1000:03d}"
        suffix = f"-{timestamp}-{ms}-{_job_name_counter}"

        # Truncate base to fit within MAX_JOB_NAME_LENGTH
        max_base_len = MAX_JOB_NAME_LENGTH - len(suffix)
        if max_base_len < 1:
            max_base_len = 1
        base = base[:max_base_len]
        # Ensure no trailing hyphen after truncation
        base = base.rstrip("-")
        # Guard against empty base (e.g. all-hyphen input)
        if not base:
            base = "j"

        resolved = f"{base}{suffix}"
        if not _JOB_NAME_PATTERN.match(resolved):
            raise DataPreparerConfigurationError(
                f"Derived job name '{resolved}' does not match the Smithy JobName pattern."
            )
        return resolved

    @staticmethod
    def build_job_arn(category: JobCategory, job_name: str, session: Session) -> str:
        """Build the JobArn from category, job_name, region, and account."""
        region = session.boto_region_name
        account = session.account_id()
        return f"arn:aws:sagemaker:{region}:{account}:job/{category.value}/{job_name}"

    @staticmethod
    def get_customer_details(session: Session) -> Dict[str, Any]:
        """Build CustomerDetails with AccountId only."""
        return {"AccountId": session.account_id()}

    @staticmethod
    def resolve_into_payload(
        user_payload: Dict[str, Any],
        schema: Dict[str, Any],
        category: JobCategory,
        sagemaker_session: Optional[Session] = None,
        config_mgr=None,
    ) -> Dict[str, Any]:
        """Walk the schema and fill absent paths from config/fallbacks.

        Returns a new dict (deep-copied from user_payload) so user_payload is not mutated.
        """
        merged = copy.deepcopy(user_payload)
        properties = schema.get("properties", {})

        DataPreparerDefaults._walk_and_fill(
            merged,
            properties,
            data_preparer_job_path(category),
            DATA_PREPARER_GLOBAL_PATH,
            category,
            sagemaker_session,
            config_mgr,
            merged,
        )
        return merged

    @staticmethod
    def _walk_and_fill(
        target: Dict[str, Any],
        properties: Dict[str, Any],
        category_path: str,
        global_path: str,
        category: JobCategory,
        session: Optional[Session],
        config_mgr,
        root_payload: Dict[str, Any],
        path_prefix: str = "",
    ) -> None:
        """Recursively walk schema properties and fill defaults."""
        for prop_name, prop_schema in properties.items():
            current_path = f"{path_prefix}.{prop_name}" if path_prefix else prop_name
            # Resolve the user-supplied value under either the PascalCase
            # schema name OR its snake_case alias. If the user passed
            # snake_case (e.g. output_s3_uri), the helper renames the key
            # to PascalCase in-place so this property is treated as
            # already-set everywhere downstream.
            existing_value = _resolve_existing(target, prop_name)

            if prop_schema.get("type") == "object" and "properties" in prop_schema:
                # Whether the scaffolding below created this dict, captured before
                # that assignment overwrites the signal. Only scaffolding may be
                # pruned: a caller-supplied empty object is a request, not an absent
                # value. `EvaluationConfig.Deterministic: {}` is how a caller asks a
                # DataQualityEvaluation job for the default metric set -- the service
                # selects defaults precisely when `Metrics` is omitted -- so pruning
                # it turned a valid request into a silent no-op that ran no metrics.
                sdk_created = existing_value is None
                # Recurse into nested objects only if user hasn't set the entire subtree
                if existing_value is None:
                    target[prop_name] = {}
                    existing_value = target[prop_name]
                if isinstance(existing_value, dict):
                    DataPreparerDefaults._walk_and_fill(
                        existing_value,
                        prop_schema["properties"],
                        category_path,
                        global_path,
                        category,
                        session,
                        config_mgr,
                        root_payload,
                        path_prefix=current_path,
                    )
                    # Remove empty nested dicts that we created and never filled. A
                    # caller's own empty dict is preserved (see sdk_created above).
                    if sdk_created and not existing_value:
                        del target[prop_name]
                continue

            # Leaf property: skip if user already set it
            if existing_value is not None:
                continue

            logger.debug("Default for '%s' not set by user; searching config paths", prop_name)

            # Tier 1: per-category config
            value = None
            if config_mgr is not None:
                value = config_mgr.resolve_value_from_config(
                    config_path=f"{category_path}.{current_path}",
                    sagemaker_session=session,
                )
            if value is not None:
                logger.info("Applied default from %s.%s", category_path, current_path)
                target[prop_name] = value
                continue

            # Tier 2: global config
            if config_mgr is not None:
                value = config_mgr.resolve_value_from_config(
                    config_path=f"{global_path}.{current_path}",
                    sagemaker_session=session,
                )
            if value is not None:
                logger.info("Applied default from %s.%s", global_path, current_path)
                target[prop_name] = value
                continue

            # Tier 3: well-known fallbacks (exact path match only)
            if current_path in _WELL_KNOWN_FALLBACKS and session is not None:
                fallback_fn = _WELL_KNOWN_FALLBACKS[current_path]
                value = fallback_fn(category, session, root_payload, config_mgr)
                if value is not None:
                    logger.info("Applied well-known fallback for '%s'", current_path)
                    target[prop_name] = value
