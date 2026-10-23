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
"""Typed catalog view of a hub model.

A hub model carries its catalog metadata in two places. ``ListHubContents`` returns up to
50 ``HubContentSearchKeywords`` per model, tagged with a prefix per field (``@task:``,
``@license:``, ``@provider:``, ``@model-size:``, ``@context-window:``, ``@framework:``,
``@language:``, ``@data-type:``, ``@input-modality:``, ``@output-modality:``,
``@model-type:``, ``@capability:``, ``@recipe:``). ``DescribeHubContent`` and
``Search`` return the full ``HubContentDocument`` JSON, whose top-level fields hold the
same values in their original casing without the 50-keyword cap.

``HubModelCatalogEntry`` reads either source into one object, so a caller filters on
``entry.license`` instead of splitting keyword strings by hand. A keyword, when present,
is authoritative. The document fills in only what the keyword cap dropped.
"""

from __future__ import absolute_import

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

KEYWORD_PREFIXES = {
    "tasks": "@task:",
    "license": "@license:",
    "provider": "@provider:",
    "model_size": "@model-size:",
    "context_window": "@context-window:",
    "framework": "@framework:",
    "languages": "@language:",
    "data_types": "@data-type:",
    "input_modalities": "@input-modality:",
    "output_modalities": "@output-modality:",
    "model_type": "@model-type:",
}
CAPABILITY_PREFIX = "@capability:"
RECIPE_PREFIX = "@recipe:"
CUSTOMIZATION_CAPABILITY = "customization"
BEDROCK_CAPABILITY = "bedrock_console"

_DOCUMENT_FIELDS = {
    "tasks": "Task",
    "license": "License",
    "provider": "Provider",
    "model_size": "ModelSize",
    "context_window": "ContextWindow",
    "framework": "Framework",
    "languages": "Languages",
    "data_types": "DataType",
    "input_modalities": "InputModalities",
    "output_modalities": "OutputModalities",
    "model_type": "ModelTypes",
}
_LIST_FIELDS = (
    "tasks",
    "languages",
    "data_types",
    "input_modalities",
    "output_modalities",
)


@dataclass(frozen=True)
class HubRecipe:
    """One ``@recipe:`` keyword or ``RecipeCollection`` entry of a hub model.

    The hub tags a recipe as ``@recipe:{type}_{technique}`` or
    ``@recipe:{type}_{technique}_{strategy}``, all lowercase: ``@recipe:finetuning_sft_lora``
    is a FineTuning recipe for the SFT technique with the LoRA strategy.
    """

    recipe_type: str
    technique: str
    strategy: Optional[str] = None

    @classmethod
    def from_keyword(cls, keyword: str) -> Optional["HubRecipe"]:
        """Parse a ``@recipe:`` keyword, or return ``None`` for any other keyword."""
        if not keyword.lower().startswith(RECIPE_PREFIX):
            return None
        parts = keyword[len(RECIPE_PREFIX) :].lower().split("_", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return None
        strategy = parts[2] if len(parts) == 3 and parts[2] else None
        return cls(recipe_type=parts[0], technique=parts[1], strategy=strategy)

    @classmethod
    def from_document(cls, recipe: Mapping[str, Any]) -> Optional["HubRecipe"]:
        """Parse one ``RecipeCollection`` entry of a hub model document."""
        recipe_type = recipe.get("Type") or recipe.get("RecipeType")
        technique = recipe.get("CustomizationTechnique") or recipe.get("EvaluationType")
        if not recipe_type or not technique:
            return None
        strategy = recipe.get("Strategy") or recipe.get("TrainingType")
        return cls(
            recipe_type=str(recipe_type).lower(),
            technique=str(technique).lower(),
            strategy=str(strategy).lower() if strategy else None,
        )


@dataclass
class HubModelCatalogEntry:
    """The catalog metadata of one hub model, read from keywords and the document.

    Attributes:
        hub_content_name: The model's hub content name (the JumpStart model ID).
        hub_content_version: The version the entry was read from.
        display_name: ``HubContentDisplayName`` when the hub provides one.
        description: ``HubContentDescription`` when the hub provides one.
        tasks: Every ``@task:`` keyword, extended with document ``Task`` values the keyword
            cap dropped. Keyword values keep the hub's spelling.
        license: The ``@license:`` keyword, or the document ``License``.
        provider: The ``@provider:`` keyword, or the document ``Provider``.
        model_size: The raw ``@model-size:`` value (``"7b"``, ``"10b-70b"``, ``"<1b"``), or
            the document ``ModelSize``.
        context_window: The raw ``@context-window:`` value, or the document ``ContextWindow``.
        framework: The ``@framework:`` keyword, or the document ``Framework``.
        languages: Every ``@language:`` keyword, extended with document ``Languages``.
        data_types: Every ``@data-type:`` keyword, extended with document ``DataType``.
        input_modalities: Every ``@input-modality:`` keyword, extended with the document.
        output_modalities: Every ``@output-modality:`` keyword, extended with the document.
        model_type: The ``@model-type:`` keyword, or the first document ``ModelTypes`` value.
        capabilities: Every ``@capability:`` value plus the document ``Capabilities``, lowercase.
        recipes: Every ``@recipe:`` keyword plus the document ``RecipeCollection``.
        creation_time: ``CreationTime`` of the version read.
        original_creation_time: ``OriginalCreationTime`` from ``ListHubContents``, the date the
            model first entered the catalog. ``None`` when read from ``DescribeHubContent``.
        search_keywords: The raw ``HubContentSearchKeywords``.
    """

    hub_content_name: str
    hub_content_version: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    tasks: List[str] = field(default_factory=list)
    license: Optional[str] = None
    provider: Optional[str] = None
    model_size: Optional[str] = None
    context_window: Optional[str] = None
    framework: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    input_modalities: List[str] = field(default_factory=list)
    output_modalities: List[str] = field(default_factory=list)
    model_type: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    recipes: List[HubRecipe] = field(default_factory=list)
    creation_time: Optional[datetime] = None
    original_creation_time: Optional[datetime] = None
    search_keywords: List[str] = field(default_factory=list)

    @property
    def customization_supported(self) -> bool:
        """``True`` when the hub tags the model with the ``customization`` capability."""
        return CUSTOMIZATION_CAPABILITY in self.capabilities

    @property
    def bedrock_supported(self) -> bool:
        """``True`` when the hub tags the model with the ``bedrock_console`` capability."""
        return BEDROCK_CAPABILITY in self.capabilities

    def fine_tuning_techniques(self) -> List[str]:
        """The techniques of the model's FineTuning recipes (``sft``, ``dpo``, ...), sorted."""
        return sorted({r.technique for r in self.recipes if r.recipe_type == "finetuning"})

    def supports_technique(self, technique: str) -> bool:
        """``True`` when a FineTuning recipe for ``technique`` exists (case-insensitive)."""
        return technique.lower() in self.fine_tuning_techniques()

    @classmethod
    def from_hub_content(cls, hub_content: Any) -> "HubModelCatalogEntry":
        """Build an entry from a ``sagemaker.core.resources.HubContent`` or a boto response.

        Accepts the resource object that ``HubContent.get``, ``HubContent.get_all``, and
        ``HubContent.get_all_versions`` return, a ``DescribeHubContent`` response dict, a
        ``ListHubContents`` summary dict, or the ``HubContent`` record of a ``Search`` result.
        """
        if isinstance(hub_content, Mapping):
            return cls._from_boto(hub_content)
        return cls._from_fields(
            name=_attribute(hub_content, "hub_content_name"),
            version=_attribute(hub_content, "hub_content_version"),
            display_name=_attribute(hub_content, "hub_content_display_name"),
            description=_attribute(hub_content, "hub_content_description"),
            keywords=_attribute(hub_content, "hub_content_search_keywords") or [],
            document=_parse_document(_attribute(hub_content, "hub_content_document")),
            creation_time=_attribute(hub_content, "creation_time"),
            original_creation_time=_attribute(hub_content, "original_creation_time"),
        )

    @classmethod
    def _from_boto(cls, record: Mapping[str, Any]) -> "HubModelCatalogEntry":
        return cls._from_fields(
            name=record.get("HubContentName"),
            version=record.get("HubContentVersion"),
            display_name=record.get("HubContentDisplayName"),
            description=record.get("HubContentDescription"),
            keywords=record.get("HubContentSearchKeywords") or [],
            document=_parse_document(record.get("HubContentDocument")),
            creation_time=record.get("CreationTime"),
            original_creation_time=record.get("OriginalCreationTime"),
        )

    @classmethod
    def _from_fields(
        cls,
        name: Optional[str],
        version: Optional[str],
        display_name: Optional[str],
        description: Optional[str],
        keywords: Iterable[str],
        document: Dict[str, Any],
        creation_time: Optional[datetime],
        original_creation_time: Optional[datetime],
    ) -> "HubModelCatalogEntry":
        if not name:
            raise ValueError("A hub content entry must have a HubContentName.")
        keyword_list = [k for k in keywords if isinstance(k, str)]
        values = _keyword_values(keyword_list)
        entry = cls(
            hub_content_name=name,
            hub_content_version=version,
            display_name=display_name,
            description=description,
            creation_time=creation_time,
            original_creation_time=original_creation_time,
            search_keywords=list(keyword_list),
        )
        for attribute in KEYWORD_PREFIXES:
            keyword_hits = values.get(attribute, [])
            document_hits = _document_values(document, _DOCUMENT_FIELDS[attribute])
            if attribute in _LIST_FIELDS:
                merged = list(keyword_hits)
                merged.extend(v for v in document_hits if v.lower() not in _lower_set(merged))
                setattr(entry, attribute, merged)
            else:
                chosen = (
                    keyword_hits[0]
                    if keyword_hits
                    else (document_hits[0] if document_hits else None)
                )
                setattr(entry, attribute, chosen)
        capabilities = [v.lower() for v in values.get("capabilities", [])]
        capabilities.extend(
            v.lower()
            for v in _document_values(document, "Capabilities")
            if v.lower() not in capabilities
        )
        entry.capabilities = capabilities
        recipes = [r for r in (HubRecipe.from_keyword(k) for k in keyword_list) if r]
        for raw in document.get("RecipeCollection") or []:
            if isinstance(raw, Mapping):
                recipe = HubRecipe.from_document(raw)
                if recipe and recipe not in recipes:
                    recipes.append(recipe)
        entry.recipes = recipes
        return entry


def _attribute(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None or type(value).__name__ == "Unassigned":
        return None
    return value


def _parse_document(document: Any) -> Dict[str, Any]:
    if isinstance(document, Mapping):
        return dict(document)
    if isinstance(document, str) and document:
        try:
            parsed = json.loads(document)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _keyword_values(keywords: Iterable[str]) -> Dict[str, List[str]]:
    """Group keyword values by catalog attribute, in hub order."""
    values: Dict[str, List[str]] = {}
    for keyword in keywords:
        lowered = keyword.lower()
        if lowered.startswith(CAPABILITY_PREFIX):
            values.setdefault("capabilities", []).append(keyword[len(CAPABILITY_PREFIX) :])
            continue
        for attribute, prefix in KEYWORD_PREFIXES.items():
            if lowered.startswith(prefix):
                value = keyword[len(prefix) :]
                if value:
                    values.setdefault(attribute, []).append(value)
                break
    return values


def _document_values(document: Mapping[str, Any], field_name: str) -> List[str]:
    value = document.get(field_name)
    items = value if isinstance(value, list) else [value]
    return [item for item in items if isinstance(item, str) and item]


def _lower_set(values: Iterable[str]) -> set:
    return {v.lower() for v in values}


def catalog_entries(hub_contents: Iterable[Any]) -> List[HubModelCatalogEntry]:
    """Build a ``HubModelCatalogEntry`` for every item of ``hub_contents``.

    ``hub_contents`` can be a ``HubContent.get_all(...)`` iterator, a list of
    ``ListHubContents`` summaries, or a list of ``Search`` ``HubContent`` records.
    """
    return [HubModelCatalogEntry.from_hub_content(item) for item in hub_contents]
