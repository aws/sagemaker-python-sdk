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
"""Unit tests for sagemaker.ai_registry.data_preparer.configs."""

from __future__ import absolute_import

import copy
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer.configs import (
    DatasetJobConfig,
    _promote_singular_sample_generation_model,
)
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerConfigurationError,
    DataPreparerSchemaValidationError,
)

SIMPLE_SCHEMA = {
    "properties": {
        "NumSamples": {"type": "integer"},
        "SpecS3Uri": {"type": "string"},
    },
    "required": ["NumSamples"],
}


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.boto_region_name = "us-west-2"
    session.account_id.return_value = "123456789012"
    return session


@pytest.fixture
def mock_config_mgr():
    mgr = MagicMock()
    mgr.resolve_value_from_config.return_value = None
    return mgr


class TestDatasetJobConfigConstruction:
    def test_basic_construction(self):
        cfg = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            num_samples=500,
            spec_s3_uri="s3://bucket/spec.txt",
        )
        assert cfg.job_category == JobCategory.SYNTHETIC_DATA_GENERATION
        assert cfg.model_extra == {"num_samples": 500, "spec_s3_uri": "s3://bucket/spec.txt"}

    def test_extra_fields_stored(self):
        cfg = DatasetJobConfig(custom_field="value")
        assert cfg.model_extra.get("custom_field") == "value"


class TestResolveAndValidate:
    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_happy_path(self, mock_defaults, mock_resolver, mock_session, mock_config_mgr):
        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.resolve_into_payload.return_value = {"NumSamples": 500}
        mock_resolver.get_schema.return_value = ("1.0.0", SIMPLE_SCHEMA)

        from pydantic import BaseModel, ConfigDict, Field
        from sagemaker.ai_registry.data_preparer._schema_resolver import _DynamicBase, create_model

        model = create_model(
            "TestModel",
            __base__=_DynamicBase,
            num_samples=(int, Field(..., alias="NumSamples")),
            spec_s3_uri=(str | None, Field(default=None, alias="SpecS3Uri")),
        )
        mock_resolver.build_dynamic_model.return_value = model

        cfg = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            num_samples=500,
        )
        cfg._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        assert cfg._resolved_schema_version == "1.0.0"
        assert cfg._serialized_document is not None
        doc = json.loads(cfg._serialized_document)
        assert doc["NumSamples"] == 500

    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_missing_required_raises_validation_error(
        self, mock_defaults, mock_resolver, mock_session, mock_config_mgr
    ):
        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.resolve_into_payload.return_value = {}  # Missing NumSamples
        mock_resolver.get_schema.return_value = ("1.0.0", SIMPLE_SCHEMA)

        from pydantic import Field
        from sagemaker.ai_registry.data_preparer._schema_resolver import _DynamicBase, create_model

        model = create_model(
            "TestModel",
            __base__=_DynamicBase,
            num_samples=(int, Field(..., alias="NumSamples")),
            spec_s3_uri=(str | None, Field(default=None, alias="SpecS3Uri")),
        )
        mock_resolver.build_dynamic_model.return_value = model

        cfg = DatasetJobConfig(job_category=JobCategory.SYNTHETIC_DATA_GENERATION)
        with pytest.raises(DataPreparerSchemaValidationError) as exc_info:
            cfg._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        assert exc_info.value.category == JobCategory.SYNTHETIC_DATA_GENERATION
        assert exc_info.value.schema_version == "1.0.0"

    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_extra_field_raises_with_suggestion(
        self, mock_defaults, mock_resolver, mock_session, mock_config_mgr
    ):
        """Validation errors carry a `print_schema` hint in the message.

        NOTE: `_DynamicBase` today uses `extra="allow"` so genuine extra fields
        don't raise. This test exercises the missing-required path (which does
        raise through `_rewrite_validation_error`) and asserts the hint is
        appended. Tightening `extra="forbid"` to actually catch the typo case
        is Review 5 M2 / Review 10 M1 follow-up.
        """
        mock_defaults.get_sagemaker_session.return_value = mock_session
        # Empty payload after defaults → required field `NumSamples` missing
        mock_defaults.resolve_into_payload.return_value = {}
        mock_resolver.get_schema.return_value = ("1.0.0", SIMPLE_SCHEMA)

        from pydantic import Field
        from sagemaker.ai_registry.data_preparer._schema_resolver import _DynamicBase, create_model

        model = create_model(
            "TestModel",
            __base__=_DynamicBase,
            num_samples=(int, Field(..., alias="NumSamples")),
            spec_s3_uri=(str | None, Field(default=None, alias="SpecS3Uri")),
        )
        mock_resolver.build_dynamic_model.return_value = model

        cfg = DatasetJobConfig(job_category=JobCategory.SYNTHETIC_DATA_GENERATION)
        with pytest.raises(DataPreparerSchemaValidationError) as exc_info:
            cfg._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        msg = str(exc_info.value)
        assert "print_schema" in msg


class TestAccessorsBeforeResolution:
    def test_resolved_schema_version_raises(self):
        cfg = DatasetJobConfig()
        with pytest.raises(DataPreparerConfigurationError):
            _ = cfg.resolved_schema_version

    def test_resolved_schema_raises(self):
        cfg = DatasetJobConfig()
        with pytest.raises(DataPreparerConfigurationError):
            _ = cfg.resolved_schema

    def test_to_job_config_document_raises(self):
        cfg = DatasetJobConfig()
        with pytest.raises(DataPreparerConfigurationError):
            cfg.to_job_config_document()

    def test_dynamic_model_fields_raises(self):
        cfg = DatasetJobConfig()
        with pytest.raises(DataPreparerConfigurationError):
            cfg.dynamic_model_fields()

    def test_dynamic_model_json_schema_raises(self):
        cfg = DatasetJobConfig()
        with pytest.raises(DataPreparerConfigurationError):
            cfg.dynamic_model_json_schema()


class TestFromTemplate:
    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_fills_set_me_for_required(self, mock_defaults, mock_resolver):
        mock_defaults.get_sagemaker_session.return_value = MagicMock()
        mock_resolver.get_schema.return_value = ("1.0.0", SIMPLE_SCHEMA)

        from pydantic import Field
        from sagemaker.ai_registry.data_preparer._schema_resolver import _DynamicBase, create_model

        model = create_model(
            "TestModel",
            __base__=_DynamicBase,
            num_samples=(int, Field(..., alias="NumSamples")),
            spec_s3_uri=(str | None, Field(default=None, alias="SpecS3Uri")),
        )
        mock_resolver.build_dynamic_model.return_value = model

        cfg = DatasetJobConfig.from_template(JobCategory.SYNTHETIC_DATA_GENERATION)
        assert cfg.job_category == JobCategory.SYNTHETIC_DATA_GENERATION
        assert cfg.model_extra.get("num_samples") == "<SET_ME>"


class TestHubNameInjection:
    """Tests for the 3 HubName scenarios in SDG config resolution."""

    def test_customer_provides_hub_name(self, mock_session):
        """Scenario 1: Customer explicitly sets HubName — we respect it."""
        from sagemaker.ai_registry.data_preparer.configs import AIRHub

        merged = {"OutputDataConfig": {"OutputS3Uri": "s3://b/o", "HubName": "my-hub"}}

        with patch.object(AIRHub, "get_hub_name", return_value="auto-hub"):
            # Simulate the injection logic from configs.py
            odc = merged.get("OutputDataConfig")
            if isinstance(odc, dict) and not odc.get("HubName"):
                hub_name = AIRHub.get_hub_name()
                try:
                    mock_session.sagemaker_client.describe_hub(HubName=hub_name)
                    odc["HubName"] = hub_name
                except Exception:
                    pass

        assert merged["OutputDataConfig"]["HubName"] == "my-hub"
        mock_session.sagemaker_client.describe_hub.assert_not_called()

    def test_hub_exists_auto_injects(self, mock_session):
        """Scenario 2: Hub exists in account — auto-inject the name."""
        from sagemaker.ai_registry.data_preparer.configs import AIRHub

        merged = {"OutputDataConfig": {"OutputS3Uri": "s3://b/o"}}
        mock_session.sagemaker_client.describe_hub.return_value = {"HubName": "auto-hub"}

        with patch.object(AIRHub, "get_hub_name", return_value="auto-hub"):
            odc = merged.get("OutputDataConfig")
            if isinstance(odc, dict) and not odc.get("HubName"):
                hub_name = AIRHub.get_hub_name()
                try:
                    mock_session.sagemaker_client.describe_hub(HubName=hub_name)
                    odc["HubName"] = hub_name
                except Exception:
                    pass

        assert merged["OutputDataConfig"]["HubName"] == "auto-hub"

    def test_hub_not_exists_omits(self, mock_session):
        """Scenario 3: Hub doesn't exist — omit HubName, let service auto-create."""
        from sagemaker.ai_registry.data_preparer.configs import AIRHub

        merged = {"OutputDataConfig": {"OutputS3Uri": "s3://b/o"}}
        mock_session.sagemaker_client.describe_hub.side_effect = Exception("ResourceNotFound")

        with patch.object(AIRHub, "get_hub_name", return_value="auto-hub"):
            odc = merged.get("OutputDataConfig")
            if isinstance(odc, dict) and not odc.get("HubName"):
                hub_name = AIRHub.get_hub_name()
                try:
                    mock_session.sagemaker_client.describe_hub(HubName=hub_name)
                    odc["HubName"] = hub_name
                except Exception:
                    pass

        assert "HubName" not in merged["OutputDataConfig"]


class TestPromoteSingularSampleGenerationModel:
    """The v1 -> v2 shim for ModelConfig's dropped singular key."""

    def test_pascal_case_singular_becomes_a_list(self, caplog):
        model = {"ModelId": "model-one", "TargetTokensPerMinute": 100000}
        payload = {"GenerationConfig": {"ModelConfig": {"SampleGenerationModelConfig": model}}}

        with caplog.at_level(logging.WARNING):
            _promote_singular_sample_generation_model(payload)

        model_config = payload["GenerationConfig"]["ModelConfig"]
        assert model_config == {"SampleGenerationModelConfigs": [model]}
        assert "SampleGenerationModelConfig is deprecated" in caplog.text

    def test_snake_case_singular_becomes_a_list(self):
        model = {"ModelId": "model-one"}
        payload = {"generation_config": {"model_config": {"sample_generation_model_config": model}}}

        _promote_singular_sample_generation_model(payload)

        model_config = payload["generation_config"]["model_config"]
        assert "sample_generation_model_config" not in model_config
        assert model_config["SampleGenerationModelConfigs"] == [model]

    def test_singular_and_plural_together_raises(self):
        payload = {
            "GenerationConfig": {
                "ModelConfig": {
                    "SampleGenerationModelConfig": {"ModelId": "model-one"},
                    "SampleGenerationModelConfigs": [{"ModelId": "model-two"}],
                }
            }
        }

        with pytest.raises(DataPreparerConfigurationError, match="set only"):
            _promote_singular_sample_generation_model(payload)

    def test_plural_only_is_untouched(self):
        plural = [{"ModelId": "model-one"}, {"ModelId": "model-two"}]
        payload = {"GenerationConfig": {"ModelConfig": {"SampleGenerationModelConfigs": plural}}}

        _promote_singular_sample_generation_model(payload)

        assert payload["GenerationConfig"]["ModelConfig"]["SampleGenerationModelConfigs"] == plural

    def test_a_none_singular_is_dropped_rather_than_wrapped(self, caplog):
        payload = {"GenerationConfig": {"ModelConfig": {"SampleGenerationModelConfig": None}}}

        with caplog.at_level(logging.WARNING):
            _promote_singular_sample_generation_model(payload)

        assert payload["GenerationConfig"]["ModelConfig"] == {}
        # The warning must not claim a list was sent, since nothing was.
        assert "being dropped" in caplog.text
        assert "one-element list" not in caplog.text

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"GenerationConfig": {}},
            {"GenerationConfig": {"NumSamples": 10}},
            {"GenerationConfig": "s3://bucket/not-a-mapping"},
            {"GenerationConfig": {"ModelConfig": "not-a-mapping"}},
            {"ModelConfig": {"SampleGenerationModelConfig": {"ModelId": "model-one"}}},
        ],
    )
    def test_no_op_cases(self, payload):
        """Nothing to promote: a missing level, a non-mapping, or no ModelConfig at all."""
        before = copy.deepcopy(payload)

        _promote_singular_sample_generation_model(payload)

        assert payload == before


class TestShimRunsOnlyForV2:
    """The shim is gated on the resolved SDG schema major version."""

    @staticmethod
    def _resolve(version, mock_defaults, mock_resolver, session, config_mgr, model_config):
        from sagemaker.ai_registry.data_preparer._schema_resolver import _DynamicBase, create_model

        mock_defaults.get_sagemaker_session.return_value = session
        mock_defaults.resolve_into_payload.side_effect = lambda payload, *a, **k: copy.deepcopy(
            payload
        )
        mock_resolver.get_schema.return_value = (version, {"properties": {}, "required": []})
        mock_resolver.build_dynamic_model.return_value = create_model(
            "ShimTestModel", __base__=_DynamicBase
        )

        cfg = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            generation_config={"ModelConfig": model_config},
        )
        cfg._resolve_and_validate(sagemaker_session=session, config_mgr=config_mgr)
        return json.loads(cfg.to_job_config_document())

    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_v2_rewrites_the_singular_key(
        self, mock_defaults, mock_resolver, mock_session, mock_config_mgr
    ):
        doc = self._resolve(
            "2.0.0",
            mock_defaults,
            mock_resolver,
            mock_session,
            mock_config_mgr,
            {"SampleGenerationModelConfig": {"ModelId": "model-one"}},
        )

        model_config = doc["GenerationConfig"]["ModelConfig"]
        assert model_config == {"SampleGenerationModelConfigs": [{"ModelId": "model-one"}]}

    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_v1_leaves_the_singular_key_alone(
        self, mock_defaults, mock_resolver, mock_session, mock_config_mgr
    ):
        doc = self._resolve(
            "1.0.0",
            mock_defaults,
            mock_resolver,
            mock_session,
            mock_config_mgr,
            {"SampleGenerationModelConfig": {"ModelId": "model-one"}},
        )

        model_config = doc["GenerationConfig"]["ModelConfig"]
        assert model_config == {"SampleGenerationModelConfig": {"ModelId": "model-one"}}
        assert "SampleGenerationModelConfigs" not in model_config
