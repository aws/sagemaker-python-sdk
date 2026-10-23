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
"""``HubModelCatalogEntry`` reads a hub model's catalog metadata from keywords and document.

Before this class every consumer split ``HubContentSearchKeywords`` by prefix itself:
``jumpstart/search.py``, ``recipe_utils.py``, ``training_session/_utils.py``, and four
scripts in the SageMaker agent skills, each with its own partial field list and none
handling the 50-keyword cap that drops ``@model-size:`` and ``@context-window:`` on models
with many languages.
"""

from __future__ import absolute_import

import datetime
import json

import pytest

from sagemaker.core.jumpstart.hub.catalog import (
    HubModelCatalogEntry,
    HubRecipe,
    catalog_entries,
)
from sagemaker.core.resources import HubContent

KEYWORDS = [
    "@task:text generation",
    "@task:summarization",
    "@license:apache-2.0",
    "@provider:Meta",
    "@model-size:1b-10b",
    "@context-window:128k",
    "@framework:huggingface",
    "@language:english",
    "@language:german",
    "@data-type:text",
    "@input-modality:text",
    "@output-modality:text",
    "@model-type:open_weights",
    "@capability:customization",
    "@capability:bedrock_console",
    "@recipe:finetuning_sft_lora",
    "@recipe:finetuning_dpo_lora",
    "@recipe:finetuning_cpt",
    "@recipe:evaluation_llmasjudge",
    "@huggingface-id:meta-llama/Llama-3.2-1B",
    "unknown keyword",
]
ORIGINAL = datetime.datetime(2023, 6, 1, tzinfo=datetime.timezone.utc)
CREATED = datetime.datetime(2024, 3, 1, tzinfo=datetime.timezone.utc)


def _summary(**overrides):
    summary = {
        "HubContentName": "meta-textgeneration-llama-3-2-1b",
        "HubContentVersion": "2.1.0",
        "HubContentDisplayName": "Llama 3.2 1B",
        "HubContentDescription": "A small Llama.",
        "HubContentSearchKeywords": list(KEYWORDS),
        "CreationTime": CREATED,
        "OriginalCreationTime": ORIGINAL,
    }
    summary.update(overrides)
    return summary


def test_entry_reads_every_keyword_prefix_from_a_list_summary():
    entry = HubModelCatalogEntry.from_hub_content(_summary())

    assert entry.hub_content_name == "meta-textgeneration-llama-3-2-1b"
    assert entry.hub_content_version == "2.1.0"
    assert entry.display_name == "Llama 3.2 1B"
    assert entry.tasks == ["text generation", "summarization"]
    assert entry.license == "apache-2.0"
    assert entry.provider == "Meta"
    assert entry.model_size == "1b-10b"
    assert entry.context_window == "128k"
    assert entry.framework == "huggingface"
    assert entry.languages == ["english", "german"]
    assert entry.data_types == ["text"]
    assert entry.input_modalities == ["text"]
    assert entry.output_modalities == ["text"]
    assert entry.model_type == "open_weights"
    assert entry.capabilities == ["customization", "bedrock_console"]
    assert entry.customization_supported is True
    assert entry.bedrock_supported is True
    assert entry.creation_time == CREATED
    assert entry.original_creation_time == ORIGINAL
    assert entry.search_keywords == KEYWORDS


def test_recipes_parse_type_technique_and_optional_strategy():
    entry = HubModelCatalogEntry.from_hub_content(_summary())

    assert HubRecipe("finetuning", "sft", "lora") in entry.recipes
    assert HubRecipe("finetuning", "cpt", None) in entry.recipes
    assert HubRecipe("evaluation", "llmasjudge", None) in entry.recipes
    assert entry.fine_tuning_techniques() == ["cpt", "dpo", "sft"]
    assert entry.supports_technique("SFT") is True
    assert entry.supports_technique("rlvr") is False


def test_recipe_keyword_parser_ignores_other_keywords_and_malformed_recipes():
    assert HubRecipe.from_keyword("@task:text generation") is None
    assert HubRecipe.from_keyword("@recipe:finetuning") is None
    assert HubRecipe.from_keyword("@recipe:FineTuning_SFT_LoRA") == HubRecipe(
        "finetuning", "sft", "lora"
    )


def test_document_fills_fields_the_keyword_cap_dropped():
    document = {
        "Task": "Speech Recognition",
        "License": "MIT",
        "Provider": "OpenAI",
        "ModelSize": "1.5B",
        "ContextWindow": "30s",
        "Languages": ["English", "French"],
        "Capabilities": ["CUSTOMIZATION"],
        "RecipeCollection": [
            {"Type": "FineTuning", "CustomizationTechnique": "SFT", "Strategy": "LORA"},
            {"Type": "Evaluation", "EvaluationType": "ASR"},
        ],
    }
    entry = HubModelCatalogEntry.from_hub_content(
        _summary(
            HubContentSearchKeywords=["@task:speech recognition", "@language:english"],
            HubContentDocument=json.dumps(document),
        )
    )

    # A keyword is authoritative; the document only fills what is missing.
    assert entry.tasks == ["speech recognition"]
    assert entry.languages == ["english", "French"]
    assert entry.license == "MIT"
    assert entry.provider == "OpenAI"
    assert entry.model_size == "1.5B"
    assert entry.context_window == "30s"
    assert entry.capabilities == ["customization"]
    assert entry.recipes == [
        HubRecipe("finetuning", "sft", "lora"),
        HubRecipe("evaluation", "asr", None),
    ]


def test_keyword_wins_over_document_for_scalar_fields():
    entry = HubModelCatalogEntry.from_hub_content(
        _summary(
            HubContentSearchKeywords=["@license:apache-2.0", "@model-size:7b"],
            HubContentDocument=json.dumps({"License": "Apache 2.0", "ModelSize": "8B"}),
        )
    )

    assert entry.license == "apache-2.0"
    assert entry.model_size == "7b"


def test_malformed_document_and_missing_keywords_give_an_empty_entry():
    entry = HubModelCatalogEntry.from_hub_content(
        {"HubContentName": "bare", "HubContentDocument": "not json"}
    )

    assert entry.tasks == []
    assert entry.license is None
    assert entry.capabilities == []
    assert entry.recipes == []
    assert entry.customization_supported is False
    assert entry.original_creation_time is None


def test_entry_requires_a_name():
    with pytest.raises(ValueError, match="HubContentName"):
        HubModelCatalogEntry.from_hub_content({"HubContentSearchKeywords": []})


def test_entry_reads_a_hub_content_resource():
    resource = HubContent(
        hub_name="SageMakerPublicHub",
        hub_content_type="Model",
        hub_content_name="meta-textgeneration-llama-3-2-1b",
        hub_content_version="2.1.0",
        hub_content_search_keywords=["@task:text generation", "@recipe:finetuning_sft_lora"],
        hub_content_document=json.dumps({"License": "Llama 3.2 Community"}),
        creation_time=CREATED,
    )

    entry = HubModelCatalogEntry.from_hub_content(resource)

    assert entry.hub_content_name == "meta-textgeneration-llama-3-2-1b"
    assert entry.tasks == ["text generation"]
    assert entry.license == "Llama 3.2 Community"
    assert entry.fine_tuning_techniques() == ["sft"]
    assert entry.creation_time == CREATED
    # Unassigned() on the resource reads as None, not as the sentinel object.
    assert entry.display_name is None
    assert entry.original_creation_time is None


def test_catalog_entries_maps_every_item():
    entries = catalog_entries([_summary(), _summary(HubContentName="other")])

    assert [entry.hub_content_name for entry in entries] == [
        "meta-textgeneration-llama-3-2-1b",
        "other",
    ]
