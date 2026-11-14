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
"""Unit tests for sagemaker.ai_registry.data_preparer.auto_labeling.shapes."""

from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerConfigurationError
from sagemaker.ai_registry.data_preparer.auto_labeling.shapes import (
    ClassificationLabelingOptions,
    Demos,
    EvaluationData,
    Grader,
    LabelingOptions,
    LlmJudgeLabelingOptions,
    NerLabelingOptions,
    OrdinalLabelingOptions,
    QaLabelingOptions,
    Routing,
    Scoring,
    SummarizationLabelingOptions,
)

SCORED_TASKS = [
    (ClassificationLabelingOptions, "labeling.classification"),
    (NerLabelingOptions, "labeling.ner"),
    (QaLabelingOptions, "labeling.qa"),
    (OrdinalLabelingOptions, "labeling.ordinal"),
    (LlmJudgeLabelingOptions, "labeling.llm_judge"),
]


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


def _echo_model():
    """A dynamic model stand-in that accepts anything the payload carries."""
    from sagemaker.ai_registry.data_preparer._schema_resolver import _DynamicBase, create_model

    return create_model("LabelingTestModel", __base__=_DynamicBase)


class TestSerialization:
    def test_every_option_serializes(self):
        options = ClassificationLabelingOptions(
            text_fields=["subject", "body"],
            demos=Demos(s3_uri="s3://bucket/demos.jsonl"),
            evaluation_data=EvaluationData(s3_uri="s3://bucket/gold.jsonl", label_field="label"),
            scoring=Scoring(target=0.9),
            routing=Routing(review_dataset_name="tickets-review"),
        )

        assert options.to_dataset_options() == {
            "Type": "labeling.classification",
            "Config": {
                "TextFields": ["subject", "body"],
                "Demos": {"S3Uri": "s3://bucket/demos.jsonl"},
                "EvaluationData": {
                    "LabelField": "label",
                    "S3Uri": "s3://bucket/gold.jsonl",
                },
                "Scoring": {"Target": 0.9},
                "Routing": {"ReviewDatasetName": "tickets-review"},
            },
        }

    @pytest.mark.parametrize("cls,expected_type", SCORED_TASKS)
    def test_each_task_declares_its_type(self, cls, expected_type):
        assert cls(text_fields=["text"]).to_dataset_options() == {
            "Type": expected_type,
            "Config": {"TextFields": ["text"]},
        }

    def test_unset_options_are_omitted(self):
        config = QaLabelingOptions(text_fields=["question"]).to_dataset_options()["Config"]
        assert config == {"TextFields": ["question"]}

    def test_summarization_graders_nest_under_scoring(self):
        options = SummarizationLabelingOptions(
            text_fields=["article"],
            graders=[
                Grader(model_id="grader-one", target_tokens_per_minute=100000),
                Grader(model_id="grader-two", reasoning_effort="low"),
            ],
        )

        assert options.to_dataset_options() == {
            "Type": "labeling.summarization",
            "Config": {
                "TextFields": ["article"],
                "Scoring": {
                    "Graders": [
                        {"ModelId": "grader-one", "TargetTokensPerMinute": 100000},
                        {"ModelId": "grader-two", "ReasoningEffort": "low"},
                    ]
                },
            },
        }

    def test_summarization_keeps_shared_options(self):
        config = SummarizationLabelingOptions(
            text_fields=["article"],
            demos=Demos(s3_uri="s3://bucket/demos.jsonl"),
            routing=Routing(review_dataset_name="articles-review"),
            graders=[Grader(model_id="grader-one")],
        ).to_dataset_options()["Config"]

        assert config["Demos"] == {"S3Uri": "s3://bucket/demos.jsonl"}
        assert config["Routing"] == {"ReviewDatasetName": "articles-review"}
        assert "Graders" not in config

    def test_evaluation_data_from_dataset_arn(self):
        options = NerLabelingOptions(
            text_fields=["body"],
            evaluation_data=EvaluationData(
                dataset_arn="arn:aws:sagemaker:us-west-2:123456789012:hub-content/"
                "hub/DataSet/gold/1.0.0",
                label_field="entities",
            ),
        )

        assert options.to_dataset_options()["Config"]["EvaluationData"] == {
            "LabelField": "entities",
            "DatasetArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/"
            "hub/DataSet/gold/1.0.0",
        }

    def test_pascal_case_input_accepted(self):
        options = ClassificationLabelingOptions(
            TextFields=["subject"],
            Scoring={"Target": 0.95},
            Routing={"ReviewDatasetName": "tickets-review"},
        )

        assert options.to_dataset_options()["Config"] == {
            "TextFields": ["subject"],
            "Scoring": {"Target": 0.95},
            "Routing": {"ReviewDatasetName": "tickets-review"},
        }


class TestConstructionErrors:
    def test_misspelled_top_level_field_rejected(self):
        with pytest.raises(ValidationError, match="text_field"):
            ClassificationLabelingOptions(text_fields=["body"], text_field=["body"])

    def test_error_locations_use_the_name_the_caller_wrote(self):
        """loc_by_alias=False: a missing field is reported as text_fields, not TextFields."""
        with pytest.raises(ValidationError) as exc_info:
            ClassificationLabelingOptions()

        assert [err["loc"] for err in exc_info.value.errors()] == [("text_fields",)]

    def test_misspelled_nested_field_rejected(self):
        with pytest.raises(ValidationError, match="Targett"):
            ClassificationLabelingOptions(text_fields=["body"], scoring={"Targett": 0.9})

    def test_graders_rejected_on_classification(self):
        with pytest.raises(ValidationError, match="graders"):
            ClassificationLabelingOptions(
                text_fields=["body"], graders=[Grader(model_id="grader-one")]
            )

    def test_scoring_target_rejected_on_summarization(self):
        with pytest.raises(ValidationError, match="scoring"):
            SummarizationLabelingOptions(
                text_fields=["article"],
                graders=[Grader(model_id="grader-one")],
                scoring=Scoring(target=0.9),
            )

    def test_evaluation_data_rejected_on_summarization(self):
        with pytest.raises(ValidationError, match="evaluation_data"):
            SummarizationLabelingOptions(
                text_fields=["article"],
                graders=[Grader(model_id="grader-one")],
                evaluation_data=EvaluationData(s3_uri="s3://bucket/gold.jsonl", label_field="l"),
            )

    def test_summarization_requires_graders(self):
        with pytest.raises(ValidationError, match="graders"):
            SummarizationLabelingOptions(text_fields=["article"])

    def test_summarization_rejects_empty_graders(self):
        with pytest.raises(ValidationError):
            SummarizationLabelingOptions(text_fields=["article"], graders=[])

    def test_empty_text_fields_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationLabelingOptions(text_fields=[])

    def test_repeated_text_fields_rejected(self):
        with pytest.raises(ValidationError, match="twice"):
            ClassificationLabelingOptions(text_fields=["body", "body"])

    def test_evaluation_data_with_no_location_rejected(self):
        with pytest.raises(ValidationError, match="exactly one"):
            EvaluationData(label_field="label")

    def test_evaluation_data_with_two_locations_rejected(self):
        with pytest.raises(ValidationError, match="exactly one"):
            EvaluationData(
                label_field="label", s3_uri="s3://bucket/gold.jsonl", dataset_arn="arn:aws:x"
            )

    def test_base_class_cannot_be_constructed(self):
        """The base names no Type, so it fails where it is written, not at serialization."""
        with pytest.raises(ValidationError, match="names no labeling Type"):
            LabelingOptions(text_fields=["body"])

    def test_base_class_serialization_still_guarded(self):
        """model_construct skips validation, so to_dataset_options keeps its own guard."""
        unvalidated = LabelingOptions.model_construct(text_fields=["body"])

        with pytest.raises(TypeError, match="no labeling Type"):
            unvalidated.to_dataset_options()

    def test_base_class_is_still_usable_as_a_type(self):
        """It stays exported for annotation and isinstance against the task family."""
        assert isinstance(ClassificationLabelingOptions(text_fields=["body"]), LabelingOptions)

    def test_summarization_without_graders_reports_what_is_missing(self):
        """model_construct skips the required-field check, so the move guards itself."""
        unvalidated = SummarizationLabelingOptions.model_construct(text_fields=["article"])

        with pytest.raises(ValueError, match="has no graders"):
            unvalidated.to_dataset_options()


class TestDatasetJobConfigWiring:
    """dataset_options lands in GenerationConfig.DatasetOptions (configs.py)."""

    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def _resolve(self, config, mock_defaults, mock_resolver, session, config_mgr, version="2.0.0"):
        """Resolve *config* against a stub schema and return the payload that was built."""
        mock_defaults.get_sagemaker_session.return_value = session
        mock_defaults.resolve_into_payload.side_effect = lambda payload, *a, **k: dict(payload)
        mock_resolver.get_schema.return_value = (version, {"properties": {}, "required": []})
        mock_resolver.build_dynamic_model.return_value = _echo_model()

        config._resolve_and_validate(sagemaker_session=session, config_mgr=config_mgr)
        return mock_defaults.resolve_into_payload.call_args[0][0]

    def test_options_land_in_generation_config(self, mock_session, mock_config_mgr):
        generation_config = {"InputDataS3Uri": "s3://bucket/corpus/"}
        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="tickets-labeled",
            generation_config=generation_config,
            dataset_options=ClassificationLabelingOptions(
                text_fields=["subject"], scoring=Scoring(target=0.9)
            ),
        )

        payload = self._resolve(config, session=mock_session, config_mgr=mock_config_mgr)

        assert payload["generation_config"] == {
            "InputDataS3Uri": "s3://bucket/corpus/",
            "DatasetOptions": {
                "Type": "labeling.classification",
                "Config": {"TextFields": ["subject"], "Scoring": {"Target": 0.9}},
            },
        }
        # The caller still holds the dict they passed; it was not written through.
        assert generation_config == {"InputDataS3Uri": "s3://bucket/corpus/"}

    def test_options_without_a_generation_config(self, mock_session, mock_config_mgr):
        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_options=QaLabelingOptions(text_fields=["question"]),
        )

        payload = self._resolve(config, session=mock_session, config_mgr=mock_config_mgr)

        assert payload["GenerationConfig"] == {
            "DatasetOptions": {
                "Type": "labeling.qa",
                "Config": {"TextFields": ["question"]},
            }
        }

    def test_pascal_case_generation_config_key_reused(self, mock_session, mock_config_mgr):
        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            GenerationConfig={"NumSamples": 10},
            dataset_options=OrdinalLabelingOptions(text_fields=["body"]),
        )

        payload = self._resolve(config, session=mock_session, config_mgr=mock_config_mgr)

        assert "generation_config" not in payload
        assert payload["GenerationConfig"]["NumSamples"] == 10
        assert payload["GenerationConfig"]["DatasetOptions"]["Type"] == "labeling.ordinal"

    def test_no_options_leaves_the_payload_alone(self, mock_session, mock_config_mgr):
        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            generation_config={"NumSamples": 10},
        )

        payload = self._resolve(config, session=mock_session, config_mgr=mock_config_mgr)

        assert payload["generation_config"] == {"NumSamples": 10}

    def test_typed_and_raw_dataset_options_is_an_error(self, mock_session, mock_config_mgr):
        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            generation_config={"DatasetOptions": {"Type": "labeling.qa", "Config": {}}},
            dataset_options=QaLabelingOptions(text_fields=["question"]),
        )

        with pytest.raises(DataPreparerConfigurationError, match="not both"):
            self._resolve(config, session=mock_session, config_mgr=mock_config_mgr)

    def test_options_on_a_non_sdg_category_is_an_error(self, mock_session, mock_config_mgr):
        config = DatasetJobConfig(
            job_category=JobCategory.DATA_QUALITY_EVALUATION,
            dataset_options=QaLabelingOptions(text_fields=["question"]),
        )

        with pytest.raises(DataPreparerConfigurationError, match="only valid for"):
            self._resolve(config, session=mock_session, config_mgr=mock_config_mgr)

    def test_non_mapping_generation_config_is_an_error(self, mock_session, mock_config_mgr):
        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            generation_config="s3://bucket/not-a-mapping",
            dataset_options=QaLabelingOptions(text_fields=["question"]),
        )

        with pytest.raises(DataPreparerConfigurationError, match="must be a mapping"):
            self._resolve(config, session=mock_session, config_mgr=mock_config_mgr)

    def test_options_on_a_v1_pinned_job_is_an_error(self, mock_session, mock_config_mgr):
        """DatasetOptions only exists from v2, so a v1 pin fails here, not at CreateJob."""
        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            schema_version="1.0.0",
            dataset_options=QaLabelingOptions(text_fields=["question"]),
        )

        with pytest.raises(DataPreparerConfigurationError, match="requires a SyntheticDataGen"):
            self._resolve(config, session=mock_session, config_mgr=mock_config_mgr, version="1.0.0")

    def test_raw_dict_dataset_options_rejected(self):
        """dataset_options is typed; the raw escape hatch is generation_config."""
        with pytest.raises(ValidationError):
            DatasetJobConfig(
                job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
                dataset_options={"Type": "labeling.qa", "Config": {"TextFields": ["q"]}},
            )


class TestAgainstARealSchemaModel:
    """The typed -> payload -> schema contract, through the real dynamic model.

    The wiring tests above stub the dynamic model, so they prove where
    ``DatasetOptions`` lands but not that a real schema-derived model accepts the
    shape. This builds the model with ``_SchemaResolver.build_dynamic_model`` from
    a v2-shaped schema — ``Config`` declared as an open object, as the front-door
    schema declares it — and checks the serialized document round-trips intact.
    """

    # Mirrors the published SyntheticDataGeneration 2.0.0 shape, plus the
    # DatasetOptions envelope CR-313549794 adds.
    V2_SCHEMA = {
        "type": "object",
        "properties": {
            "DatasetName": {"type": "string"},
            "OutputDataConfig": {
                "type": "object",
                "properties": {"S3OutputPath": {"type": "string"}},
            },
            "GenerationConfig": {
                "type": "object",
                "properties": {
                    "DatasetSpecification": {
                        "type": "object",
                        "properties": {"SpecS3Uri": {"type": "string"}},
                    },
                    "NumSamples": {"type": "integer", "minimum": 1},
                    "InputDataS3Uri": {"type": "string"},
                    "DatasetOptions": {
                        "type": "object",
                        "properties": {
                            "Type": {"type": "string"},
                            # Open: the schema puts no structure inside Config.
                            "Config": {"type": "object"},
                        },
                    },
                },
                "required": ["DatasetSpecification"],
            },
        },
        "required": ["GenerationConfig", "DatasetName"],
    }

    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_typed_options_survive_the_real_dynamic_model(
        self, mock_defaults, mock_session, mock_config_mgr
    ):
        from sagemaker.ai_registry.data_preparer._schema_resolver import _SchemaResolver

        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.resolve_into_payload.side_effect = lambda payload, *a, **k: dict(payload)

        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="tickets-labeled",
            generation_config={
                "DatasetSpecification": {"SpecS3Uri": "s3://bucket/spec.md"},
                "InputDataS3Uri": "s3://bucket/tickets/",
            },
            dataset_options=ClassificationLabelingOptions(
                text_fields=["subject", "body"],
                evaluation_data=EvaluationData(
                    s3_uri="s3://bucket/gold.jsonl", label_field="label"
                ),
                scoring=Scoring(target=0.9),
                routing=Routing(review_dataset_name="tickets-review"),
            ),
        )

        # Only get_schema is stubbed; build_dynamic_model and validation are real.
        # A unique version keeps this out of the resolver's model cache.
        with patch.object(
            _SchemaResolver, "get_schema", return_value=("2.0.0-realmodel-test", self.V2_SCHEMA)
        ):
            config._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        document = json.loads(config.to_job_config_document())
        options = document["GenerationConfig"]["DatasetOptions"]
        assert options["Type"] == "labeling.classification"
        # The open Config object passes through field-for-field, unflattened and
        # with its PascalCase keys intact.
        assert options["Config"] == {
            "TextFields": ["subject", "body"],
            "EvaluationData": {"LabelField": "label", "S3Uri": "s3://bucket/gold.jsonl"},
            "Scoring": {"Target": 0.9},
            "Routing": {"ReviewDatasetName": "tickets-review"},
        }

    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_summarization_graders_survive_the_real_dynamic_model(
        self, mock_defaults, mock_session, mock_config_mgr
    ):
        from sagemaker.ai_registry.data_preparer._schema_resolver import _SchemaResolver

        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.resolve_into_payload.side_effect = lambda payload, *a, **k: dict(payload)

        config = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="articles-labeled",
            generation_config={"DatasetSpecification": {"SpecS3Uri": "s3://bucket/spec.md"}},
            dataset_options=SummarizationLabelingOptions(
                text_fields=["article"],
                graders=[Grader(model_id="grader-one", reasoning_effort="low")],
            ),
        )

        with patch.object(
            _SchemaResolver, "get_schema", return_value=("2.0.0-realmodel-summ", self.V2_SCHEMA)
        ):
            config._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        config_block = json.loads(config.to_job_config_document())["GenerationConfig"][
            "DatasetOptions"
        ]["Config"]
        assert config_block["Scoring"] == {
            "Graders": [{"ModelId": "grader-one", "ReasoningEffort": "low"}]
        }
