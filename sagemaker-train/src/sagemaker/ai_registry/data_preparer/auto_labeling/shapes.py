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
"""Typed auto-labeling options for ``DataPreparer.generate``.

A SyntheticDataGeneration job runs in auto-labeling mode when its request
carries ``GenerationConfig.DatasetOptions`` with a ``labeling.*`` ``Type``. The
published v2 schema declares that envelope as ``{Type, Config}`` where ``Config``
is an *open* object: the front door puts no structure inside it, so a raw-dict
caller gets no field names from the SDK and a misspelled or misplaced key only
surfaces as a 400 at ``CreateJob``. These classes are the typed surface over
that bag — one class per labeling ``Type``, with ``extra="forbid"`` so a wrong
field name fails where it was typed.

Pass one as ``DatasetJobConfig(dataset_options=...)`` and it serializes into
``GenerationConfig.DatasetOptions``. The raw dict stays supported as the escape
hatch (``generation_config={"DatasetOptions": {...}}``), but not both at once.

Value ranges are deliberately *not* duplicated here: the service validates
``Scoring.Target`` bounds, the grader count, the ``ReasoningEffort`` enum, and
DataSet ownership at ``CreateJob``, and it stays the single source of truth for
them. What this module adds is field names, nesting, and the per-task rules that
change the shape of the request (summarization grades with ``Graders`` and takes
no ``EvaluationData``; every other task scores against gold with a ``Target``).

Example::

    from sagemaker.ai_registry import (
        ClassificationLabelingOptions,
        DataPreparer,
        DatasetJobConfig,
        EvaluationData,
        Routing,
        Scoring,
    )

    config = DatasetJobConfig(
        dataset_name="support-tickets-labeled",
        generation_config={
            "InputDataS3Uri": "s3://my-bucket/tickets/",
            "DatasetSpecification": {"SpecS3Uri": "s3://my-bucket/label-spec.md"},
            # Routing this task votes a jury, which needs two models — see below.
            "ModelConfig": {
                "SampleGenerationModelConfigs": [
                    {"ModelId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
                    {"ModelId": "openai.gpt-oss-120b-1:0"},
                ]
            },
        },
        dataset_options=ClassificationLabelingOptions(
            text_fields=["subject", "body"],
            evaluation_data=EvaluationData(
                s3_uri="s3://my-bucket/gold.jsonl", label_field="label"
            ),
            scoring=Scoring(target=0.90),
            routing=Routing(review_dataset_name="support-tickets-review"),
        ),
    )

    dataset = DataPreparer.generate(config)

How many generation models a labeling job needs depends on the task and on
whether it routes, and the count lives in ``GenerationConfig.ModelConfig``
rather than in these classes. With ``Routing`` set, a vote task
(``classification``, ``ner``, ``qa``) decides each label by juror agreement and
needs at least two ``SampleGenerationModelConfigs``, while ``ordinal`` and
``llm_judge`` resample a single model and take at most one. Without ``Routing``,
and for ``summarization``, one model labels alone and the count is
unconstrained. The service enforces this at ``CreateJob``.

Summarization is the one task shaped differently — graders instead of gold::

    from sagemaker.ai_registry import (
        Grader,
        SummarizationLabelingOptions,
    )

    options = SummarizationLabelingOptions(
        text_fields=["article"],
        graders=[Grader(model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0")],
    )
"""

from __future__ import absolute_import

from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sagemaker.ai_registry.data_preparer._case_conversion import _snake_to_pascal


class _LabelingModel(BaseModel):
    """Shared pydantic configuration for every labeling options class.

    ``alias_generator=_snake_to_pascal`` derives the service's PascalCase key from each
    snake_case field name, so ``s3_uri`` serializes as ``S3Uri`` and no field
    needs a hand-written alias. ``populate_by_name=True`` means a caller may use
    either form on input. ``extra="forbid"`` is the point of the typed layer: a
    stray key is an error here rather than a field the service never sees.

    ``loc_by_alias=False`` keeps validation errors pointing at the attribute the
    caller wrote — without it, a missing ``text_fields`` is reported against
    ``TextFields``, which is not a name the caller used.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        alias_generator=_snake_to_pascal,
        loc_by_alias=False,
        # ``Grader.model_id`` would otherwise trip pydantic's ``model_`` guard.
        protected_namespaces=(),
    )


class Demos(_LabelingModel):
    """Labeled examples shown to the labeling model (``Config.Demos``)."""

    s3_uri: str


class EvaluationData(_LabelingModel):
    """Held-out gold rows the accuracy is measured against (``Config.EvaluationData``).

    ``label_field`` names the gold column. The rows come from exactly one
    location — an S3 URI or a Hub DataSet ARN — and the service rejects both or
    neither, so that rule is checked here too. Not valid for summarization,
    which has no gold to compare against.
    """

    label_field: str
    s3_uri: Optional[str] = None
    dataset_arn: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one_location(self) -> "EvaluationData":
        """Require exactly one of ``s3_uri`` / ``dataset_arn``."""
        if bool(self.s3_uri) == bool(self.dataset_arn):
            raise ValueError(
                "EvaluationData takes exactly one of s3_uri and dataset_arn, not both or neither"
            )
        return self


class Scoring(_LabelingModel):
    """The accuracy to certify on accepted rows (``Config.Scoring``).

    Applies to the gold-scored tasks. The service enforces the accepted range;
    the SDK only carries the value.
    """

    target: float


class Routing(_LabelingModel):
    """Where rows that miss the bar are sent for human review (``Config.Routing``).

    ``review_dataset_name`` is registered as a second DataSet beside the labeled
    one, so it must differ from the labeled DataSet's name.
    """

    review_dataset_name: str


class Grader(_LabelingModel):
    """One summarization grader model (``Config.Scoring.Graders[]``).

    A model id plus optional throughput and reasoning-effort hints. The service
    enforces the grader count, the ``ReasoningEffort`` values, and that no grader
    is the model writing the summaries.
    """

    model_id: str
    target_tokens_per_minute: Optional[int] = None
    reasoning_effort: Optional[str] = None


class LabelingOptions(_LabelingModel):
    """Base class for the per-task labeling options.

    Carries what every labeling task takes: the text columns to label, optional
    demonstrations, and optional review routing. Subclasses set ``TYPE`` to their
    ``labeling.*`` string and add the fields that task accepts.

    Exported so callers can annotate or type-check against the family —
    ``DatasetJobConfig.dataset_options`` is declared as this type — but it is not
    constructible itself: a task has to say which ``labeling.*`` it is, so
    instantiating the base raises rather than deferring the failure to
    serialization. Instantiate a task subclass, such as
    :class:`ClassificationLabelingOptions`.
    """

    TYPE: ClassVar[str] = ""

    text_fields: List[str] = Field(min_length=1)
    demos: Optional[Demos] = None
    routing: Optional[Routing] = None

    @model_validator(mode="after")
    def _reject_the_base_class(self) -> "LabelingOptions":
        """Refuse to build a class that names no labeling ``TYPE``."""
        if not self.TYPE:
            raise ValueError(
                f"{type(self).__name__} names no labeling Type and cannot be used on its own. "
                "Use a task class such as ClassificationLabelingOptions, NerLabelingOptions, "
                "QaLabelingOptions, OrdinalLabelingOptions, LlmJudgeLabelingOptions or "
                "SummarizationLabelingOptions."
            )
        return self

    @model_validator(mode="after")
    def _distinct_text_fields(self) -> "LabelingOptions":
        """Reject a repeated column name in ``text_fields``."""
        if len(set(self.text_fields)) != len(self.text_fields):
            raise ValueError("text_fields must not name the same field twice")
        return self

    def to_dataset_options(self) -> Dict[str, Any]:
        """Serialize to the ``GenerationConfig.DatasetOptions`` value: ``{Type, Config}``.

        ``_reject_the_base_class`` already blocks this at construction; the guard
        stays for an instance built through ``model_construct``, which skips
        validation.
        """
        if not self.TYPE:
            raise TypeError(
                f"{type(self).__name__} has no labeling Type. Instantiate a task subclass "
                "(for example ClassificationLabelingOptions), not the base class."
            )
        return {"Type": self.TYPE, "Config": self._to_config()}

    def _to_config(self) -> Dict[str, Any]:
        """Build the ``Config`` object: PascalCase keys, unset fields omitted."""
        return self.model_dump(by_alias=True, exclude_none=True)


class _ScoredLabelingOptions(LabelingOptions):
    """Base for the tasks scored against gold labels.

    Every task except summarization produces a label that can be compared to a
    known answer, so these accept held-out ``evaluation_data`` and a ``scoring``
    target.
    """

    evaluation_data: Optional[EvaluationData] = None
    scoring: Optional[Scoring] = None


class ClassificationLabelingOptions(_ScoredLabelingOptions):
    """Assign each row one label from the spec's set (``labeling.classification``)."""

    TYPE: ClassVar[str] = "labeling.classification"


class NerLabelingOptions(_ScoredLabelingOptions):
    """Extract named entity spans from each row (``labeling.ner``)."""

    TYPE: ClassVar[str] = "labeling.ner"


class QaLabelingOptions(_ScoredLabelingOptions):
    """Answer the spec's question about each row (``labeling.qa``)."""

    TYPE: ClassVar[str] = "labeling.qa"


class OrdinalLabelingOptions(_ScoredLabelingOptions):
    """Score each row on an ordered scale (``labeling.ordinal``)."""

    TYPE: ClassVar[str] = "labeling.ordinal"


class LlmJudgeLabelingOptions(_ScoredLabelingOptions):
    """Judge each row against the spec's rubric (``labeling.llm_judge``)."""

    TYPE: ClassVar[str] = "labeling.llm_judge"


class SummarizationLabelingOptions(LabelingOptions):
    """Summarize each row, graded rather than compared to gold (``labeling.summarization``).

    A summary has no single right answer, so this task takes neither
    ``evaluation_data`` nor a ``Target``: ``graders`` judge each summary against
    its own source text. The graders serialize under ``Scoring``, where the
    container reads them.
    """

    TYPE: ClassVar[str] = "labeling.summarization"

    graders: List[Grader] = Field(min_length=1)

    def _to_config(self) -> Dict[str, Any]:
        """Move ``Graders`` under ``Scoring``, the shape the container reads.

        ``graders`` is required, so a validated instance always carries it. The
        check covers an instance built through ``model_construct``, which skips
        validation, so that path reports what is missing instead of a ``KeyError``
        from the move.
        """
        config = super()._to_config()
        graders = config.pop("Graders", None)
        if not graders:
            raise ValueError(
                f"{type(self).__name__} has no graders, so there is nothing to grade "
                "summaries against. Pass at least one Grader."
            )
        config["Scoring"] = {"Graders": graders}
        return config


__all__ = [
    "LabelingOptions",
    "ClassificationLabelingOptions",
    "NerLabelingOptions",
    "QaLabelingOptions",
    "OrdinalLabelingOptions",
    "LlmJudgeLabelingOptions",
    "SummarizationLabelingOptions",
    "Demos",
    "EvaluationData",
    "Grader",
    "Routing",
    "Scoring",
]
