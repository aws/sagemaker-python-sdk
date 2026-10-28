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
"""DatasetJobConfig — single dynamic-schema config for both JobCategories."""

from __future__ import absolute_import

import difflib
import json
import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.ai_registry.data_preparer._case_conversion import (
    _normalize_keys_to_pascal,
    _snake_to_pascal,
)
from sagemaker.ai_registry.data_preparer._schema_resolver import (
    _SchemaResolver,
    _major,
    _pascal_to_snake,
)
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.defaults import DataPreparerDefaults
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerConfigurationError,
    DataPreparerSchemaValidationError,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.shapes import LabelingOptions
from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)


def _find_key(mapping: Dict[str, Any], pascal_name: str) -> Optional[str]:
    """Return the key in *mapping* that means *pascal_name*, or ``None``.

    Payload keys reach these helpers as the caller typed them — snake_case or
    PascalCase — because PascalCase normalization only runs at the end of
    ``_resolve_and_validate``. ``_snake_to_pascal`` is idempotent on PascalCase,
    so comparing through it matches either form.
    """
    for key in mapping:
        if _snake_to_pascal(key) == pascal_name:
            return key
    return None


def _promote_singular_sample_generation_model(payload: Dict[str, Any]) -> None:
    """Rewrite a v1-style singular ``SampleGenerationModelConfig`` into the v2 list.

    The one field SDG v2 drops is ``GenerationConfig.ModelConfig``'s singular
    ``SampleGenerationModelConfig``; v2 takes the plural
    ``SampleGenerationModelConfigs`` list instead. Before the ceiling bump an
    unpinned ``generate()`` resolved to v1 and the singular key was the way to
    ask for single-model generation, so without this shim the bump would turn
    working single-model calls into a ``CreateJob`` rejection. Rewriting here
    keeps them working against v2 unchanged.

    Mutates *payload* in place. A no-op when the payload has no
    ``GenerationConfig.ModelConfig`` object or no singular key.

    Raises:
        DataPreparerConfigurationError: If both the singular and the plural key
            are set, since which models to use would be ambiguous.
    """
    gen_key = _find_key(payload, "GenerationConfig")
    if gen_key is None:
        return
    generation_config = payload[gen_key]
    if not isinstance(generation_config, dict):
        return

    model_key = _find_key(generation_config, "ModelConfig")
    if model_key is None:
        return
    model_config = generation_config[model_key]
    if not isinstance(model_config, dict):
        return

    singular_key = _find_key(model_config, "SampleGenerationModelConfig")
    if singular_key is None:
        return

    plural_key = _find_key(model_config, "SampleGenerationModelConfigs")
    if plural_key is not None:
        raise DataPreparerConfigurationError(
            "GenerationConfig.ModelConfig sets both SampleGenerationModelConfig and "
            "SampleGenerationModelConfigs. SyntheticDataGeneration v2 takes only "
            "SampleGenerationModelConfigs; set only that one."
        )

    # Note on both branches: this rewrite only covers a value passed to the job.
    # The same key set in a SageMaker config file is resolved against the v2
    # schema, which no longer declares it, so that path is never consulted and
    # has to be renamed by hand.
    singular = model_config.pop(singular_key)
    if singular is None:
        logger.warning(
            "GenerationConfig.ModelConfig.SampleGenerationModelConfig is deprecated and was "
            "set to null, so it is being dropped. SyntheticDataGeneration v2 takes the "
            "SampleGenerationModelConfigs list; set that instead. A value set at the same "
            "path in a SageMaker config file is not picked up under v2 either."
        )
        return

    logger.warning(
        "GenerationConfig.ModelConfig.SampleGenerationModelConfig is deprecated: "
        "SyntheticDataGeneration v2 takes the SampleGenerationModelConfigs list. "
        "Sending it as a one-element list for this job; set "
        "SampleGenerationModelConfigs directly instead. Note this rewrite only "
        "covers a value passed to the job: the same key set in a SageMaker config "
        "file is resolved against the v2 schema, which no longer declares it, so "
        "that path is not picked up at all and must be renamed."
    )
    model_config["SampleGenerationModelConfigs"] = [singular]


def _insert_dataset_options(
    payload: Dict[str, Any],
    options: LabelingOptions,
    category: Optional[JobCategory],
    version: str,
) -> None:
    """Serialize typed *options* into ``payload``'s ``GenerationConfig.DatasetOptions``.

    Mutates *payload* in place, replacing its ``GenerationConfig`` with a new
    dict so a ``generation_config`` the caller still holds a reference to is left
    untouched.

    Raises:
        DataPreparerConfigurationError: If the job category is not
            SyntheticDataGeneration, if *version* resolved below 2.x, if
            ``generation_config`` is set to something other than a mapping, or if
            ``DatasetOptions`` is already set there.
    """
    if category not in (None, JobCategory.SYNTHETIC_DATA_GENERATION):
        raise DataPreparerConfigurationError(
            f"dataset_options is only valid for {JobCategory.SYNTHETIC_DATA_GENERATION.value} "
            f"jobs, not {category.value}. Auto-labeling runs as part of a generate() job."
        )

    # DatasetOptions only exists from v2. A v1-pinned request would otherwise send
    # it as an unmodelled extra and be rejected by the service instead of here.
    if _major(version) < 2:
        raise DataPreparerConfigurationError(
            f"dataset_options requires a SyntheticDataGeneration 2.x schema, but this job "
            f"resolved {version}. Drop schema_version='{version}' to take the current "
            "default, or pin schema_version='2.0.0'."
        )

    gen_key = _find_key(payload, "GenerationConfig")
    generation_config: Dict[str, Any] = {}
    if gen_key is not None:
        existing = payload[gen_key]
        if not isinstance(existing, dict):
            raise DataPreparerConfigurationError(
                "generation_config must be a mapping to carry dataset_options, "
                f"got {type(existing).__name__}."
            )
        generation_config = existing

    if _find_key(generation_config, "DatasetOptions") is not None:
        raise DataPreparerConfigurationError(
            "DatasetOptions is set twice: once as a typed dataset_options and once as a raw "
            "GenerationConfig.DatasetOptions dict. Pass one or the other, not both."
        )

    payload[gen_key or "GenerationConfig"] = {
        **generation_config,
        "DatasetOptions": options.to_dataset_options(),
    }


class DatasetJobConfig(BaseModel):
    """Config for a DataPreparer job, dynamically validated against the Hub schema.

    Usage patterns:
    - Dict input: DatasetJobConfig(dataset_name="my-dataset", generation_config={
      "DatasetSpecification": {"SpecS3Uri": "s3://..."}, "NumSamples": 500})
    - Typed input: DatasetJobConfig(output_data_config=OutputDataConfig(...))
    - Template: DatasetJobConfig.from_template(JobCategory.SYNTHETIC_DATA_GENERATION)
    - Auto-labeling: pass a typed options class as ``dataset_options`` and it
      serializes into ``GenerationConfig.DatasetOptions``::

          from sagemaker.ai_registry import (
              ClassificationLabelingOptions,
              EvaluationData,
              Scoring,
          )

          DatasetJobConfig(
              dataset_name="tickets-labeled",
              generation_config={
                  "InputDataS3Uri": "s3://bucket/tickets/",
                  "DatasetSpecification": {"SpecS3Uri": "s3://bucket/label-spec.md"},
              },
              dataset_options=ClassificationLabelingOptions(
                  text_fields=["subject", "body"],
                  evaluation_data=EvaluationData(
                      s3_uri="s3://bucket/gold.jsonl", label_field="label"
                  ),
                  scoring=Scoring(target=0.90),
              ),
          )

      See :mod:`sagemaker.ai_registry.data_preparer.auto_labeling.shapes` for the task
      classes. A raw ``generation_config={"DatasetOptions": {...}}`` dict still
      works, but setting both raises.

    job_category is typically inferred from the calling method.
    Schema resolution is LAZY — happens inside DataPreparer.generate/evaluate.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",
    )

    job_category: Optional[JobCategory] = None
    schema_version: Optional[str] = None
    sagemaker_session: Optional[Session] = None
    dataset_options: Optional[LabelingOptions] = None

    _resolved_schema_version: Optional[str] = PrivateAttr(default=None)
    _resolved_schema: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _dynamic_model: Optional[Type[BaseModel]] = PrivateAttr(default=None)
    _validated_payload: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _serialized_document: Optional[str] = PrivateAttr(default=None)

    def _resolve_and_validate(
        self,
        sagemaker_session: Optional[Session] = None,
        config_mgr=None,
    ) -> None:
        """Fetch Hub schema, build dynamic model, merge defaults, validate, freeze."""
        # Idempotency guard: once resolved, do not re-resolve
        if self._serialized_document is not None:
            return

        session = DataPreparerDefaults.get_sagemaker_session(
            sagemaker_session or self.sagemaker_session
        )
        # Lazy-construct a SageMakerConfig if the caller didn't pass one so the
        # intelligent-defaults walk still fires (Req 4.1–4.11).
        if config_mgr is None:
            from sagemaker.core.config.config_manager import SageMakerConfig

            config_mgr = SageMakerConfig()

        version, schema = _SchemaResolver.get_schema(
            self.job_category, self.schema_version, session
        )

        user_payload = dict(self.model_extra or {})

        # Typed auto-labeling options become GenerationConfig.DatasetOptions before
        # the defaults walk, so the resulting payload is the one the schema sees.
        if self.dataset_options is not None:
            _insert_dataset_options(user_payload, self.dataset_options, self.job_category, version)

        # Schema-driven intelligent defaulting walk (Req 4.1–4.11). Resolves any
        # schema property not set by the user from:
        #   1. Modules.DataPreparer.Job.<Category>.<path>
        #   2. Modules.DataPreparer.GlobalDefaults.<path>
        #   3. _WELL_KNOWN_FALLBACKS (RoleArn, OutputDataConfig.OutputS3Uri)
        #   4. Schema-declared default
        merged_payload = DataPreparerDefaults.resolve_into_payload(
            user_payload,
            schema,
            self.job_category,
            sagemaker_session=session,
            config_mgr=config_mgr,
        )

        # HubName on OutputDataConfig for SDG:
        # 1. Customer explicitly set it → respect it (already in merged_payload)
        # 2. Hub exists in account → auto-inject the deterministic name
        # 3. Hub doesn't exist → omit it, service will auto-create
        if self.job_category == JobCategory.SYNTHETIC_DATA_GENERATION:
            odc = merged_payload.get("OutputDataConfig")
            if isinstance(odc, dict) and not odc.get("HubName"):
                hub_name = AIRHub.get_hub_name()
                try:
                    client = session.sagemaker_client
                    client.describe_hub(HubName=hub_name)
                    odc["HubName"] = hub_name
                except Exception:
                    pass  # hub doesn't exist; omit and let service auto-create

        # SDG v2 replaced the singular model-config object with a list. Rewrite a
        # v1-style single-model request so the ceiling bump (see constants.py)
        # doesn't break callers who were relying on the old key.
        if self.job_category == JobCategory.SYNTHETIC_DATA_GENERATION and _major(version) >= 2:
            _promote_singular_sample_generation_model(merged_payload)

        # Build dynamic model first so we can use it for key normalization
        dynamic_model = _SchemaResolver.build_dynamic_model(
            self.job_category, version, schema
        )

        # Pass merged payload to dynamic model — it accepts both
        # snake_case and PascalCase via populate_by_name=True
        try:
            validated = dynamic_model(**merged_payload)
        except ValidationError as e:
            raise _rewrite_validation_error(
                e, self.job_category, version, dynamic_model
            ) from e

        # Check for template sentinels that were not replaced
        dumped = validated.model_dump(by_alias=True, exclude_none=True)
        for key, value in dumped.items():
            if value == "<SET_ME>":
                raise DataPreparerConfigurationError(
                    f"Template placeholder '<SET_ME>' found in field '{key}'. "
                    "Please set a real value before submitting."
                )

        import copy

        self._resolved_schema_version = version
        self._resolved_schema = copy.deepcopy(schema)
        self._dynamic_model = dynamic_model
        self._validated_payload = validated.model_dump(by_alias=True, exclude_none=True)
        # Normalize all keys to PascalCase before serialization. This handles:
        # 1. model_extra nested dicts that bypass Pydantic alias processing
        # 2. Any snake_case keys that leaked through the dynamic model
        # The conversion is idempotent on already-PascalCase keys.
        pascal_payload = _normalize_keys_to_pascal(self._validated_payload)
        self._serialized_document = json.dumps(pascal_payload, ensure_ascii=False)

    @property
    def resolved_schema_version(self) -> str:
        """The resolved schema version (available after _resolve_and_validate)."""
        if self._resolved_schema_version is None:
            raise DataPreparerConfigurationError(
                "Schema not yet resolved. Call DataPreparer.generate/evaluate first."
            )
        return self._resolved_schema_version

    @property
    def resolved_schema(self) -> Dict[str, Any]:
        """The resolved schema dict (available after _resolve_and_validate)."""
        if self._resolved_schema is None:
            raise DataPreparerConfigurationError(
                "Schema not yet resolved. Call DataPreparer.generate/evaluate first."
            )
        return self._resolved_schema

    def to_job_config_document(self) -> str:
        """Return the serialized JSON string for JobConfigDocument."""
        if self._serialized_document is None:
            raise DataPreparerConfigurationError(
                "Config not yet resolved. Call DataPreparer.generate/evaluate first."
            )
        return self._serialized_document

    def job_config_dict(self) -> Dict[str, Any]:
        """Return the resolved JobConfigDocument as a dict.

        Decoded from the same serialized form ``to_job_config_document`` returns,
        so the pipeline capture path composes exactly the document the direct
        path submits.
        """
        return json.loads(self.to_job_config_document())

    def dynamic_model_fields(self) -> Dict[str, Any]:
        """Return field info for the dynamic model's top-level fields (copy)."""
        if self._dynamic_model is None:
            raise DataPreparerConfigurationError(
                "Config not yet resolved. Call DataPreparer.generate/evaluate first."
            )
        import copy

        return copy.copy(dict(self._dynamic_model.model_fields))

    def dynamic_model_json_schema(self) -> Dict[str, Any]:
        """Return the JSON schema of the dynamic model tree."""
        if self._dynamic_model is None:
            raise DataPreparerConfigurationError(
                "Config not yet resolved. Call DataPreparer.generate/evaluate first."
            )
        return self._dynamic_model.model_json_schema()

    @classmethod
    def from_template(
        cls,
        category: JobCategory,
        schema_version: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> "DatasetJobConfig":
        """Return a DatasetJobConfig pre-filled with placeholders for required fields."""
        session = DataPreparerDefaults.get_sagemaker_session(sagemaker_session)
        version, schema = _SchemaResolver.get_schema(category, schema_version, session)
        dynamic_model = _SchemaResolver.build_dynamic_model(category, version, schema)

        template_kwargs: Dict[str, Any] = {"job_category": category, "schema_version": version}
        for field_name, field_info in dynamic_model.model_fields.items():
            if field_info.is_required():
                template_kwargs[field_name] = "<SET_ME>"
            elif field_info.default is not None:
                template_kwargs[field_name] = field_info.default

        return cls(**template_kwargs)

    @classmethod
    def schema(
        cls,
        category: JobCategory,
        schema_version: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Return the resolved JSON schema dict for a category. Pure function; no printing.

        For a human-readable tree, call :meth:`print_schema` instead.
        """
        session = DataPreparerDefaults.get_sagemaker_session(sagemaker_session)
        _version, schema_dict = _SchemaResolver.get_schema(category, schema_version, session)
        return schema_dict

    @classmethod
    def print_schema(
        cls,
        category: JobCategory,
        schema_version: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> None:
        """Pretty-print the resolved schema as a tree to stdout. No return value.

        For the raw dict, call :meth:`schema`.
        """
        from sagemaker.ai_registry.data_preparer._schema_renderer import render_schema_tree

        session = DataPreparerDefaults.get_sagemaker_session(sagemaker_session)
        version, schema_dict = _SchemaResolver.get_schema(category, schema_version, session)
        title = f"{category.value} (JobConfigSchema v{version})"
        output = render_schema_tree(schema_dict, title=title)
        print(output)

    @classmethod
    def required_fields(
        cls,
        category: JobCategory,
        schema_version: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> List[str]:
        """Return the list of required field names (snake_case) for a category's schema.

        See also :meth:`schema` for the full schema dict and
        :meth:`print_schema` for a human-readable tree.

        Args:
            category: JobCategory to query.
            schema_version: Specific version, or None for latest.
            sagemaker_session: Optional SageMaker session.

        Returns:
            List of required field names in snake_case, ordered as they appear in
            the schema. Returns empty list if the schema has no required fields.

        Raises:
            DataPreparerSchemaNotFoundError: If no schema is published for the
                category, or if the requested version is not found.
        """
        session = DataPreparerDefaults.get_sagemaker_session(sagemaker_session)
        _version, schema_dict = _SchemaResolver.get_schema(category, schema_version, session)
        return [_pascal_to_snake(f) for f in schema_dict.get("required", [])]


def _rewrite_validation_error(
    error: ValidationError,
    category: JobCategory,
    version: str,
    dynamic_model: Type[BaseModel],
) -> DataPreparerSchemaValidationError:
    """Wrap pydantic ValidationError with schema-aware suggestions."""
    pydantic_errors = error.errors()
    messages = []

    valid_fields = list(dynamic_model.model_fields.keys())
    # Also include aliases
    alias_map = {
        info.alias: name
        for name, info in dynamic_model.model_fields.items()
        if info.alias
    }
    all_valid = valid_fields + list(alias_map.keys())

    for err in pydantic_errors:
        err_type = err.get("type", "")
        if "extra" in err_type or "unexpected" in err_type:
            loc = err.get("loc", ())
            field_name = loc[-1] if loc else "unknown"
            suggestions = difflib.get_close_matches(str(field_name), all_valid, n=1)
            msg = f"  {'.'.join(str(l) for l in loc)}: unexpected field '{field_name}'"
            msg += f"\n    Valid fields at this level: {', '.join(valid_fields)}"
            if suggestions:
                msg += f"\n    Did you mean '{suggestions[0]}'?"
            messages.append(msg)
        else:
            loc = err.get("loc", ())
            messages.append(f"  {'.'.join(str(l) for l in loc)}: {err.get('msg', '')}")

    messages.append(
        f"\n  See: DatasetJobConfig.print_schema(JobCategory.{category.name})"
    )

    exc = DataPreparerSchemaValidationError(category, version, pydantic_errors)
    # Enhance the message
    detail = "\n".join(messages)
    exc.args = (f"{exc.args[0]}\n{detail}",)
    return exc
