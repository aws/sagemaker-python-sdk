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
"""``HubContent`` reads its own catalog metadata through ``HubContentMixin``.

Before the mixin every consumer split ``HubContentSearchKeywords`` by prefix itself:
``jumpstart/search.py``, ``recipe_utils.py``, ``training_session/_utils.py``, and four
scripts in the SageMaker agent skills, each with its own partial field list and none
handling the 50-keyword cap that drops ``@model-size:`` and ``@context-window:`` on models
with many languages.
"""

from __future__ import absolute_import

import json

from sagemaker.core.hub_content import HubContentMixin, HubContentRecipe
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


def _content(**overrides):
    kwargs = {
        "hub_name": "SageMakerPublicHub",
        "hub_content_type": "Model",
        "hub_content_name": "meta-textgeneration-llama-3-2-1b",
        "hub_content_search_keywords": list(KEYWORDS),
    }
    kwargs.update(overrides)
    return HubContent(**kwargs)


def test_hub_content_is_a_hub_content_mixin():
    assert issubclass(HubContent, HubContentMixin)
    # The mixin adds behavior only: no model field, nothing in model_dump().
    assert not {"tasks", "license", "recipes"} & set(HubContent.model_fields)
    assert "tasks" not in _content().model_dump()


def test_every_keyword_prefix_is_readable():
    content = _content()

    assert content.tasks == ["text generation", "summarization"]
    assert content.license == "apache-2.0"
    assert content.provider == "Meta"
    assert content.model_size == "1b-10b"
    assert content.context_window == "128k"
    assert content.framework == "huggingface"
    assert content.languages == ["english", "german"]
    assert content.data_types == ["text"]
    assert content.input_modalities == ["text"]
    assert content.output_modalities == ["text"]
    assert content.model_type == "open_weights"
    assert content.capabilities == ["customization", "bedrock_console"]
    assert content.customization_supported is True
    assert content.bedrock_supported is True


def test_recipes_parse_type_technique_and_optional_strategy():
    content = _content()

    assert HubContentRecipe("finetuning", "sft", "lora") in content.recipes
    assert HubContentRecipe("finetuning", "cpt", None) in content.recipes
    assert HubContentRecipe("evaluation", "llmasjudge", None) in content.recipes
    assert content.fine_tuning_techniques() == ["cpt", "dpo", "sft"]
    assert content.supports_technique("SFT") is True
    assert content.supports_technique("rlvr") is False


def test_recipe_keyword_parser_ignores_other_keywords_and_malformed_recipes():
    assert HubContentRecipe.from_keyword("@task:text generation") is None
    assert HubContentRecipe.from_keyword("@recipe:finetuning") is None
    assert HubContentRecipe.from_keyword("@recipe:FineTuning_SFT_LoRA") == HubContentRecipe(
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
    content = _content(
        hub_content_search_keywords=["@task:speech recognition", "@language:english"],
        hub_content_document=json.dumps(document),
    )

    # A keyword is authoritative; the document only fills what is missing.
    assert content.tasks == ["speech recognition"]
    assert content.languages == ["english", "French"]
    assert content.license == "MIT"
    assert content.provider == "OpenAI"
    assert content.model_size == "1.5B"
    assert content.context_window == "30s"
    assert content.capabilities == ["customization"]
    assert content.recipes == [
        HubContentRecipe("finetuning", "sft", "lora"),
        HubContentRecipe("evaluation", "asr", None),
    ]


def test_keyword_wins_over_document_for_scalar_fields():
    content = _content(
        hub_content_search_keywords=["@license:apache-2.0", "@model-size:7b"],
        hub_content_document=json.dumps({"License": "Apache 2.0", "ModelSize": "8B"}),
    )

    assert content.license == "apache-2.0"
    assert content.model_size == "7b"


def test_unassigned_keywords_and_malformed_document_read_as_empty():
    content = _content(
        hub_content_search_keywords=HubContent.model_fields["hub_content_search_keywords"].default,
        hub_content_document="not json",
    )

    assert content.tasks == []
    assert content.license is None
    assert content.capabilities == []
    assert content.recipes == []
    assert content.customization_supported is False


def test_accessors_follow_a_later_refresh_of_the_document():
    content = _content(hub_content_search_keywords=["@task:text generation"])
    assert content.license is None

    content.hub_content_document = json.dumps({"License": "Llama 3.2 Community"})

    assert content.license == "Llama 3.2 Community"
