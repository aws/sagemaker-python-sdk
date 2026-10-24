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
"""Hand-written behavior of the generated ``sagemaker.core.resources.HubContent`` class.

A hub model describes itself in two places. ``HubContentSearchKeywords`` holds up to 50
prefixed strings (``@task:``, ``@license:``, ``@provider:``, ``@model-size:``,
``@context-window:``, ``@framework:``, ``@language:``, ``@data-type:``,
``@input-modality:``, ``@output-modality:``, ``@model-type:``, ``@capability:``,
``@recipe:``), and ``HubContentDocument`` holds the same values as JSON fields with their
original casing and no cap. ``ListHubContents`` and ``Search`` return the keywords,
``DescribeHubContent`` and ``Search`` return the document.

``HubContentMixin`` gives ``HubContent`` typed accessors over the two: ``tasks``,
``license``, ``provider``, ``model_size``, ``context_window``, ``languages``,
``capabilities``, ``recipes``, and so on. A keyword, when present, is authoritative. The
document fills in only what the keyword cap dropped. The mixin keeps no state, so it
works on a resource from ``HubContent.get`` (document, no ``OriginalCreationTime``), from
``HubContent.get_all`` (keywords and, after the per-item refresh, the document), and on a
``HubContent`` built by hand from a boto response.

``HubContent.search`` runs the service-side ``Search`` API with ``Resource="HubContent"``.
The service indexes the document fields as ``Metadata.<Field>`` (``Metadata.Task``,
``Metadata.License``, ``Metadata.Provider``, ``Metadata.ModelSize``, ...), so a filter such
as ``HubContentFilter.metadata("license", "apache-2.0")`` narrows the public hub without
listing it. Each result record is a full ``HubContent`` with its keywords and document.
"""

from __future__ import absolute_import

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

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
FINE_TUNING_RECIPE_TYPE = "finetuning"

DOCUMENT_FIELDS = {
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
LIST_FIELDS = frozenset(
    ("tasks", "languages", "data_types", "input_modalities", "output_modalities")
)

# Search indexes the hub content document as ``Metadata.<Field>``. The two extra names are
# document fields that have no search keyword.
METADATA_FIELDS = dict(DOCUMENT_FIELDS, capabilities="Capabilities", recipes="RecipeCollection")
SEARCH_OPERATORS = frozenset(
    (
        "Equals",
        "NotEquals",
        "GreaterThan",
        "GreaterThanOrEqualTo",
        "LessThan",
        "LessThanOrEqualTo",
        "Contains",
        "Exists",
        "NotExists",
        "In",
    )
)
# Service limits of a SearchExpression: 20 filters per expression, 1024 characters per value.
MAX_SEARCH_FILTERS = 20
MAX_SEARCH_FILTER_VALUE_LENGTH = 1024
SEARCH_PAGE_SIZE = 100


@dataclass(frozen=True)
class HubContentRecipe:
    """One ``@recipe:`` keyword or ``RecipeCollection`` entry of a hub model.

    The hub tags a recipe as ``@recipe:{type}_{technique}`` or
    ``@recipe:{type}_{technique}_{strategy}``, all lowercase. ``@recipe:finetuning_sft_lora``
    is a FineTuning recipe for the SFT technique with the LoRA strategy, and
    ``@recipe:finetuning_cpt`` has no strategy.
    """

    recipe_type: str
    technique: str
    strategy: Optional[str] = None

    @classmethod
    def from_keyword(cls, keyword: str) -> Optional["HubContentRecipe"]:
        """Parse a ``@recipe:`` keyword, or return ``None`` for any other keyword."""
        if not keyword.lower().startswith(RECIPE_PREFIX):
            return None
        parts = keyword[len(RECIPE_PREFIX) :].lower().split("_", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return None
        strategy = parts[2] if len(parts) == 3 and parts[2] else None
        return cls(recipe_type=parts[0], technique=parts[1], strategy=strategy)

    @classmethod
    def from_document(cls, recipe: Mapping[str, Any]) -> Optional["HubContentRecipe"]:
        """Parse one ``RecipeCollection`` entry of a hub content document."""
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


@dataclass(frozen=True)
class HubContentFilter:
    """One ``Filter`` of a ``Search(Resource="HubContent")`` expression.

    ``name`` is a search property such as ``HubContentName``, ``HubContentSearchKeywords``,
    or ``Metadata.License``. ``operator`` is one of ``SEARCH_OPERATORS``. ``value`` is the
    filter value, or a sequence of values for ``In`` (joined with commas, as the service
    expects). ``Exists`` and ``NotExists`` take no value.
    """

    name: str
    operator: str = "Equals"
    value: Optional[Union[str, Sequence[str]]] = None

    @classmethod
    def metadata(
        cls, attribute: str, value: Union[str, Sequence[str]], operator: str = "Equals"
    ) -> "HubContentFilter":
        """Build a ``Metadata.<Field>`` filter from a ``HubContent`` attribute name.

        ``HubContentFilter.metadata("license", "apache-2.0")`` is
        ``HubContentFilter("Metadata.License", "Equals", "apache-2.0")``. A sequence of
        values selects the ``In`` operator.
        """
        if attribute not in METADATA_FIELDS:
            raise ValueError(
                f"{attribute!r} is not a hub content metadata attribute. "
                f"Choose one of: {', '.join(sorted(METADATA_FIELDS))}."
            )
        if not isinstance(value, str) and operator == "Equals":
            operator = "In"
        return cls(name=f"Metadata.{METADATA_FIELDS[attribute]}", operator=operator, value=value)

    @classmethod
    def keyword(cls, keyword: str) -> "HubContentFilter":
        """Match a hub content whose search keywords hold ``keyword``, such as ``@task:summarization``."""
        return cls(name="HubContentSearchKeywords", operator="Equals", value=keyword)

    def to_request(self) -> Dict[str, str]:
        """Return the filter in the service form, after a check of the operator and value."""
        if self.operator not in SEARCH_OPERATORS:
            raise ValueError(
                f"{self.operator!r} is not a Search operator. "
                f"Choose one of: {', '.join(sorted(SEARCH_OPERATORS))}."
            )
        request = {"Name": self.name, "Operator": self.operator}
        if self.operator in ("Exists", "NotExists"):
            return request
        if self.value is None:
            raise ValueError(f"A {self.operator} filter on {self.name} takes a value.")
        value = self.value if isinstance(self.value, str) else ",".join(self.value)
        if not value or len(value) > MAX_SEARCH_FILTER_VALUE_LENGTH:
            raise ValueError(
                f"A Search filter value holds 1 to {MAX_SEARCH_FILTER_VALUE_LENGTH} characters, "
                f"the value for {self.name} has {len(value)}."
            )
        request["Value"] = value
        return request


class HubContentMixin:
    """Typed accessors over ``hub_content_search_keywords`` and ``hub_content_document``.

    Mixed into the generated ``sagemaker.core.resources.HubContent``. Every accessor reads
    the two attributes at call time, so a ``refresh()`` that fills the document changes the
    answer, and the mixin adds no model field and no state.
    """

    hub_content_search_keywords: Any
    hub_content_document: Any

    @property
    def tasks(self) -> List[str]:
        """Every ``@task:`` keyword, extended with document ``Task`` values the cap dropped."""
        return self._list_field("tasks")

    @property
    def license(self) -> Optional[str]:
        """The ``@license:`` keyword, or the document ``License``."""
        return self._scalar_field("license")

    @property
    def provider(self) -> Optional[str]:
        """The ``@provider:`` keyword, or the document ``Provider``."""
        return self._scalar_field("provider")

    @property
    def model_size(self) -> Optional[str]:
        """The raw ``@model-size:`` value (``"7b"``, ``"1b-10b"``, ``"<1b"``), or the document ``ModelSize``."""
        return self._scalar_field("model_size")

    @property
    def context_window(self) -> Optional[str]:
        """The raw ``@context-window:`` value, or the document ``ContextWindow``."""
        return self._scalar_field("context_window")

    @property
    def framework(self) -> Optional[str]:
        """The ``@framework:`` keyword, or the document ``Framework``."""
        return self._scalar_field("framework")

    @property
    def languages(self) -> List[str]:
        """Every ``@language:`` keyword, extended with document ``Languages``."""
        return self._list_field("languages")

    @property
    def data_types(self) -> List[str]:
        """Every ``@data-type:`` keyword, extended with document ``DataType``."""
        return self._list_field("data_types")

    @property
    def input_modalities(self) -> List[str]:
        """Every ``@input-modality:`` keyword, extended with document ``InputModalities``."""
        return self._list_field("input_modalities")

    @property
    def output_modalities(self) -> List[str]:
        """Every ``@output-modality:`` keyword, extended with document ``OutputModalities``."""
        return self._list_field("output_modalities")

    @property
    def model_type(self) -> Optional[str]:
        """The ``@model-type:`` keyword, or the first document ``ModelTypes`` value."""
        return self._scalar_field("model_type")

    @property
    def capabilities(self) -> List[str]:
        """Every ``@capability:`` value plus the document ``Capabilities``, lowercase."""
        values = [v.lower() for v in self._keyword_values().get("capabilities", [])]
        for value in _document_values(self._document(), "Capabilities"):
            if value.lower() not in values:
                values.append(value.lower())
        return values

    @property
    def customization_supported(self) -> bool:
        """``True`` when the hub tags the model with the ``customization`` capability."""
        return CUSTOMIZATION_CAPABILITY in self.capabilities

    @property
    def bedrock_supported(self) -> bool:
        """``True`` when the hub tags the model with the ``bedrock_console`` capability."""
        return BEDROCK_CAPABILITY in self.capabilities

    @property
    def recipes(self) -> List[HubContentRecipe]:
        """Every ``@recipe:`` keyword plus the document ``RecipeCollection``, de-duplicated."""
        recipes = [
            recipe
            for recipe in (HubContentRecipe.from_keyword(k) for k in self._keywords())
            if recipe
        ]
        for raw in self._document().get("RecipeCollection") or []:
            if isinstance(raw, Mapping):
                recipe = HubContentRecipe.from_document(raw)
                if recipe and recipe not in recipes:
                    recipes.append(recipe)
        return recipes

    @classmethod
    def search(
        cls,
        hub_name: str,
        filters: Optional[Sequence[HubContentFilter]] = None,
        hub_content_type: str = "Model",
        session: Optional[Any] = None,
        region: Optional[str] = None,
    ) -> Iterator["HubContentMixin"]:
        """Search a hub server side and yield each matching ``HubContent``.

        Runs ``Search(Resource="HubContent")`` with an ``And`` expression of ``HubName``,
        ``HubContentType``, and ``filters``, sorted by ``HubContentName`` so the pages are
        stable, and pages until the service returns no ``NextToken``. Each result is a full
        ``HubContent`` bound to ``session`` and ``region``, with its search keywords and
        document, so ``content.license`` and ``content.recipes`` work without a second call.

        Args:
            hub_name (str): The hub to search, for example ``SageMakerPublicHub``.
            filters (Optional[Sequence[HubContentFilter]]): Up to 18 extra filters, for
                example ``[HubContentFilter.metadata("license", "apache-2.0")]``.
            hub_content_type (str): ``Model``, ``Notebook``, ``ModelReference``, or another
                hub content type (default: ``Model``).
            session (Optional[boto3.session.Session]): Boto3 session.
            region (Optional[str]): Region name.

        Returns:
            Iterator[HubContent]: The matching hub content, one page at a time.

        Raises:
            ValueError: If more than ``MAX_SEARCH_FILTERS`` filters result, or a filter has an
                unknown operator or an out-of-range value.
        """
        expression = _search_expression(hub_name, hub_content_type, filters or ())
        client = cls.get_sagemaker_client(session=session, region_name=region)
        next_token = None
        while True:
            request = {
                "Resource": "HubContent",
                "SearchExpression": expression,
                "SortBy": "HubContentName",
                "SortOrder": "Ascending",
                "MaxResults": SEARCH_PAGE_SIZE,
            }
            if next_token:
                request["NextToken"] = next_token
            response = client.search(**request)
            for record in response.get("Results", []):
                content = cls._from_search_record(record.get("HubContent", {}), hub_name)
                if content is not None:
                    yield content._set_client_context(session=session, region=region)
            page_token = response.get("NextToken")
            if not page_token or page_token == next_token:
                return
            next_token = page_token

    @classmethod
    def _from_search_record(
        cls, record: Mapping[str, Any], hub_name: str
    ) -> Optional["HubContentMixin"]:
        """Build a ``HubContent`` from one ``SearchRecord.HubContent`` dict."""
        if not record:
            return None
        from sagemaker.core.utils.code_injection.codec import transform

        transformed = transform(dict(record), "HubContent")
        transformed.setdefault("hub_name", hub_name)
        known = {name: value for name, value in transformed.items() if name in cls.model_fields}
        return cls(**known)

    def fine_tuning_techniques(self) -> List[str]:
        """The techniques of the model's FineTuning recipes (``sft``, ``dpo``, ...), sorted."""
        return sorted(
            {r.technique for r in self.recipes if r.recipe_type == FINE_TUNING_RECIPE_TYPE}
        )

    def supports_technique(self, technique: str) -> bool:
        """``True`` when a FineTuning recipe for ``technique`` exists (case-insensitive)."""
        return technique.lower() in self.fine_tuning_techniques()

    def _keywords(self) -> List[str]:
        keywords = _assigned(self.hub_content_search_keywords) or []
        return [k for k in keywords if isinstance(k, str)]

    def _keyword_values(self) -> Dict[str, List[str]]:
        return _keyword_values(self._keywords())

    def _document(self) -> Dict[str, Any]:
        return _parse_document(_assigned(self.hub_content_document))

    def _scalar_field(self, attribute: str) -> Optional[str]:
        keyword_hits = self._keyword_values().get(attribute, [])
        if keyword_hits:
            return keyword_hits[0]
        document_hits = _document_values(self._document(), DOCUMENT_FIELDS[attribute])
        return document_hits[0] if document_hits else None

    def _list_field(self, attribute: str) -> List[str]:
        merged = list(self._keyword_values().get(attribute, []))
        seen = {v.lower() for v in merged}
        for value in _document_values(self._document(), DOCUMENT_FIELDS[attribute]):
            if value.lower() not in seen:
                merged.append(value)
                seen.add(value.lower())
        return merged


def _assigned(value: Any) -> Any:
    """Return ``value`` unless it is the generated ``Unassigned()`` sentinel or ``None``."""
    if value is None or type(value).__name__ == "Unassigned":
        return None
    return value


def _search_expression(
    hub_name: str, hub_content_type: str, filters: Sequence[HubContentFilter]
) -> Dict[str, Any]:
    """Build the ``And`` expression of a hub content search, inside the service limits."""
    request_filters = [
        HubContentFilter("HubName", "Equals", hub_name).to_request(),
        HubContentFilter("HubContentType", "Equals", hub_content_type).to_request(),
    ]
    request_filters.extend(search_filter.to_request() for search_filter in filters)
    if len(request_filters) > MAX_SEARCH_FILTERS:
        raise ValueError(
            f"A SearchExpression holds at most {MAX_SEARCH_FILTERS} filters, HubName and "
            f"HubContentType included. Got {len(request_filters)}."
        )
    return {"Operator": "And", "Filters": request_filters}


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
    """Group keyword values by attribute, in hub order, keeping the hub's spelling."""
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
