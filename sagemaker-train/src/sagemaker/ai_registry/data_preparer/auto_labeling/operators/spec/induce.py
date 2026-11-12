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
"""Induce the per-label decision rules of a labeling spec from a labeled seed.

This does not author the label set or the one-line definitions; those come from the
person's spec. It takes the label set as input and writes the discriminating rules,
the part the refinement loop also produces, cold-started from the seed instead of from
observed failures. The output is a human-readable, editable spec rather than an opaque
tuned prompt.

Leakage contract (the caller enforces, this module assumes): the seed is split into an
induce pool and a held-out select pool upstream; rules are induced from the induce pool
(the dev split) only; the induced spec is then A/B'd against the provided spec on the
select pool and ships only if it wins by a margin; the test split is never read here.

Numeric judge tasks (``llm_judge``, optionally ``ordinal``) induce an anchored rubric
instead: one ``### Score Levels`` entry per level, each with one or two example
rationales quoted from dev rows. The one model call goes to Bedrock Converse
(:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators._bedrock`); pass
``invoke`` to :func:`induce_spec` to route it elsewhere or to test without a model.
"""

from __future__ import absolute_import

import collections
import json
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.rubric import (
    format_level,
    scale_levels,
    select_rubric_anchors,
)

# Characters that can inject a line break or visually reorder/mask surrounding text.
# ZWNJ (U+200C) and ZWJ (U+200D) are deliberately excluded: they are orthographically
# required in Persian/Farsi, Arabic, Devanagari, and ZWJ emoji sequences, so removing
# them corrupts legitimate text without preventing any structural attack.
# \u2060-\u2065 (WORD JOINER and invisible math operators) are similarly excluded.
# Tags block U+E0000-E007F IS stripped: TAG characters mirror printable ASCII
# invisibly and are the standard ASCII-smuggling vector. Tradeoff: the only live
# legitimate use is subdivision-flag emoji (Scotland/Wales/England), which degrade
# to plain U+1F3F4. That cost is accepted over invisible instruction injection.
_UNSAFE_CHARS = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b\u200e\u200f\u2028-\u202e\u2066-\u206f\ufeff\U000e0000-\U000e007f]"
)


def _safe_prompt_text(text):
    """Untrusted row text, made safe to interpolate into one prompt line.

    Folds newlines to spaces AND drops the invisible characters that could
    restructure the prompt. Both halves are here on purpose: _UNSAFE_CHARS
    deliberately keeps \\n, so stripping without folding still leaves a
    break-out from the surrounding line.
    """
    return _UNSAFE_CHARS.sub("", text.replace("\n", " ")).strip()


def _safe_spec_text(text):
    """The model's returned spec, made safe to write and re-upload.

    Drops the same invisible characters but KEEPS newlines: the spec is
    markdown and its line structure is meaningful.
    """
    return _UNSAFE_CHARS.sub("", text)


DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
Invoke = Callable[[str, str, Optional[str]], str]


class InducedSpec(OperatorResult):
    """What :func:`induce_spec` produced.

    The spec's Markdown (``spec``), the ``prompt`` that asked for it, and a ``summary``
    a caller prints (task, labels or levels, row counts, ``chars``). ``warnings``
    carries the thin-evidence notes.
    """

    spec: str
    prompt: str
    summary: Dict[str, Any]


# Numeric tasks whose induced spec is an anchored rubric (Score Levels + a Scale line).
# Every induced spec states the labeling job's own JSON reply format: the job appends
# that format after the spec and reads only its keys, so a spec asking for another
# format (REASONING:/SCORE: lines) contradicts it.
RUBRIC_TASKS = ("llm_judge", "ordinal")
# Anchors are quoted into the spec, so keep each one long enough to judge from but
# bounded (a 5-level, 2-anchor rubric over 2 fields stays under ~10k characters).
ANCHOR_CHARS = 700
# The opt-in reply-keys line (--reply-keys-line; references/offer-options.md): it asks
# for the keys the job appends to the reply format. It costs output tokens on every row
# and has not been shown to raise coverage, so it is the user's choice.
REPLY_KEYS_LINE = (
    "When the reply format the job appends has more keys (such as " "`confidence`), give them too."
)


def _row_text(row, text_fields):
    """A row's input field(s) on one line, capped at 400 characters.

    Several fields (premise + hypothesis, say) join with spaces.
    """
    parts = [str(row.get(f, "")) for f in text_fields if row.get(f)]
    return _safe_prompt_text(" ".join(parts))[:400]


def build_induction_prompt(  # pylint: disable=too-many-positional-arguments
    induce_rows,
    labels,
    task_hint,
    text_fields,
    max_per_class=12,
    label_key="gt_label",
    reply_keys_line=False,
):
    """The classification rule-writer prompt: balanced examples per class, rules asked for.

    Args:
        induce_rows: The dev rows.
        labels: The label set, in the order the spec lists it.
        task_hint: One line describing the task.
        text_fields: The input field(s).
        max_per_class: Examples shown per class (the header shows the full count).
        label_key: The gold field.
        reply_keys_line: Add the opt-in line asking for the keys the job appends.

    Returns:
        The prompt.
    """
    by = collections.defaultdict(list)
    for r in induce_rows:
        lbl = r.get(label_key)
        if lbl is not None:
            by[lbl].append(r)
    blocks = []
    for lbl in labels:
        ex = by.get(lbl, [])[:max_per_class]
        # Header shows the full class count (not the capped example count).
        lines = [f"## Class: {lbl}  ({len(by.get(lbl, []))} labeled examples)"]
        for r in ex:
            lines.append(f"- {_row_text(r, text_fields)}")
        blocks.append("\n".join(lines))
    examples_block = "\n\n".join(blocks)
    reply_keys = f'\n   then this line: "{REPLY_KEYS_LINE}"' if reply_keys_line else ""

    return f"""You are writing a labeling specification from labeled examples. {task_hint}

The label set is: {labels}

Below are labeled examples grouped by class. Study what distinguishes each class \
from the others, then write a concise labeling spec a different annotator could \
follow to reproduce these labels on NEW unseen items.

{examples_block}

Write a markdown labeling spec with EXACTLY these sections, in this order, using \
these exact headers AND heading levels (the downstream pipeline parses this heading \
hierarchy against references/labelling-spec-template.md; the `##` sections are \
top-level and each `###` section MUST be nested under the `##` section shown \
directly above it - never leave a `###` section between two `##` sections without \
its parent):
1. `## Use Case Context` - one or two lines describing the labeling task from the \
task description above: what data is being labeled and how the labels will be used \
(e.g. fine-tuning, evaluation, analytics). Use a placeholder where a detail is unknown.
2. `## Quality Standards` - one or two lines of high-level quality criteria (e.g. \
label accuracy, consistency across similar inputs, coverage of the confusable pairs \
below). Keep it brief.
3. `## Input Data` - one line describing the corpus being labeled (input field, \
domain) from the task description; use a placeholder where a detail is unknown.
4. `### Labelled Seed` (nested under `## Input Data`) - one line noting the labeled \
seed these rules were induced from (ground-truth field, approximate size); use a \
placeholder where unknown.
5. `## Labelling Configuration` - the parent section holding the label set and rules \
below; no prose of its own is required.
6. `### Label Set` (nested under `## Labelling Configuration`) - restate the label \
list, each label with a one-line definition of when it applies.
7. `### Labelling Rules` (nested under `## Labelling Configuration`) - for EACH \
label, 2-4 concrete discriminating rules drawn from the patterns above (what \
features push an item INTO this class vs the others). Be specific (cite the kind of \
language/attributes you saw), not generic. End with a short tie-breaker line for the \
ambiguous / easily-confused pairs.
8. `## Output Schema` - the labeling job's own JSON reply format, verbatim, and no \
other reply format (no `LABEL:` or `CONFIDENCE:` lines):
   {{"reasoning": "<one short sentence explaining the label choice>", \
"label": "<one of the labels from the Label Set above>"}}{reply_keys}
9. `## Data Verification` - the parent section holding the verification mode and \
checks below; no prose of its own is required.
10. `### Verification Mode` (nested under `## Data Verification`) - the single word \
`deterministic`, then one line stating verification is pure Python only and makes no \
LLM calls.
11. `### Checks` (nested under `## Data Verification`) - the structural checks the \
verifier enforces: the output is a single JSON object with "reasoning" and "label" \
keys, the reasoning is non-empty, and the label value is exactly one of the \
labels in the Label Set above (case-sensitive). Do not add checks beyond these.

Do NOT memorize or quote individual examples verbatim. Write general rules that \
transfer. Output ONLY the markdown spec."""


def _anchor_text(row, text_fields, cap=ANCHOR_CHARS):
    """A dev row for the rubric prompt.

    Rows with several fields (source + candidate) keep their field names so the
    rule-writer can tell the parts apart; the cap is shared across fields.
    """
    present = [f for f in text_fields if row.get(f)]
    if len(present) <= 1:
        return _safe_prompt_text(" ".join(str(row.get(f, "")) for f in present))[:cap]
    per = max(80, cap // len(present))
    return " | ".join(f"{f}: " + _safe_prompt_text(str(row[f]))[:per] for f in present)


def build_rubric_induction_prompt(  # pylint: disable=too-many-positional-arguments
    anchors,
    missing,
    task_hint,
    text_fields,
    score_min,
    score_max,
    task="llm_judge",
    gt_key="gt_score",
):
    """The rule-writer prompt for an anchored rubric (llm_judge, optionally ordinal).

    Unlike the classification prompt, the anchors are quoted into each level's entry,
    since they move the judge toward the gold scale. A level with no dev row is named
    so the writer defines it from the scale alone.

    Args:
        anchors: :func:`~.rubric.select_rubric_anchors`' level -> dev rows map.
        missing: The levels with no anchor.
        task_hint: One line describing the task.
        text_fields: The input field(s).
        score_min: The scale's low end.
        score_max: The scale's high end.
        task: ``llm_judge`` or ``ordinal``.
        gt_key: The gold score field.

    Returns:
        The prompt.
    """
    lo, hi = format_level(float(score_min)), format_level(float(score_max))
    levels, banded = scale_levels(score_min, score_max)
    if banded:
        # A wide scale anchors band midpoints (rubric.scale_levels): one bullet per
        # level shown, not one per integer from hi down to lo.
        shown = ", ".join(format_level(v) for v in reversed(levels))
        level_bullets = (
            f"one top-level bullet per level shown above ({shown}), from the "
            f"highest down, each level standing for its band of the {lo} to "
            f"{hi} scale,"
        )
    else:
        level_bullets = f"one top-level bullet per level from {hi} down to {lo},"
    blocks = []
    for lv, rows in anchors.items():
        lines = [f"## Level {format_level(lv)}  ({len(rows)} anchor row(s))"]
        for r in rows:
            lines.append(f"- gold {r.get(gt_key)}: {_anchor_text(r, text_fields)}")
        blocks.append("\n".join(lines))
    examples_block = "\n\n".join(blocks)
    gap_note = ""
    if missing:
        gap_note = (
            "\n\nNo labeled row sits near level(s) "
            + ", ".join(format_level(v) for v in missing)
            + ". Define those levels from the scale and the neighboring levels, "
            "and write `Example: none in the labeled seed` under them."
        )
    kind = (
        "judge rubric: the score rates the quality of the input as described above"
        if task == "llm_judge"
        else "rating scale for the input"
    )
    # The key the job's program reads the score from (engine/sdg_backend.PREDICTION_FIELDS).
    score_key = "judge_score" if task == "llm_judge" else "score"
    return f"""You are writing an ANCHORED RUBRIC, a {kind}. {task_hint}

The score scale runs from {lo} to {hi}. Below are labeled rows grouped by the score \
level nearest their gold score. Study what separates each level from its neighbors.

{examples_block}{gap_note}

Write a markdown labeling spec with EXACTLY these sections, in this order, using \
these exact headers AND heading levels (each `###` section MUST be nested under the \
`##` section shown directly above it):
1. `## Use Case Context` - one or two lines on what is being scored and how the \
scores will be used. Use a placeholder where a detail is unknown.
2. `## Quality Standards` - one or two lines (rank agreement with expert scores, \
consistent use of the whole scale).
3. `## Input Data` - one line describing the input field(s) and domain.
4. `### Labelled Seed` (nested under `## Input Data`) - one line noting the dev rows \
the anchors were drawn from (gold field `{gt_key}`).
5. `## Labelling Configuration` - the parent section; no prose of its own is required.
6. `### Score Levels` (nested under `## Labelling Configuration`) - {level_bullets} \
written exactly as \
`- **<level> - <short name>:** <what earns this score, in observable terms>`, and \
under each level 1 or 2 indented bullets written exactly as \
`  - Example: "<a short excerpt of one anchor row above, under 40 words>" -> <gold>. \
Why: <one sentence tying the excerpt to the level definition>`. Use only the rows \
shown above as examples, at most two per level.
7. `### Labelling Rules` (nested under `## Labelling Configuration`) - 3-6 rules for \
the hard calls: what moves a score up or down one level, and how to use the ends of \
the scale.
8. `## Output Schema` - the labeling job's own JSON reply format, verbatim, then the \
scale line, and no other reply format (no `REASONING:` or `SCORE:` lines):
   {{"reasoning": "<one short sentence explaining the score>", "{score_key}": <a number from {lo} to {hi}>}}
   Scale: {lo} to {hi}
9. `## Data Verification` - the parent section; no prose of its own is required.
10. `### Verification Mode` (nested under `## Data Verification`) - the single word \
`deterministic`, then one line stating verification is pure Python only and makes no \
LLM calls.
11. `### Checks` (nested under `## Data Verification`) - the output is a single JSON \
object with "reasoning" and "{score_key}" keys, the reasoning is non-empty, and the \
{score_key} value is a number from {lo} to {hi}. Do not add checks beyond these.

Do not write a `### Label Set` section. Output ONLY the markdown spec."""


def _span_context(text, span, width=90):
    """Sentence context around a span, newlines stripped, capped."""
    t = _safe_prompt_text(text or "")
    i = t.lower().find(span.lower())
    if i < 0:
        return t[:width]
    a = max(0, i - width // 2)
    return t[a : a + width]


def build_ner_induction_prompt(  # pylint: disable=too-many-positional-arguments
    induce_rows,
    labels,
    task_hint,
    text_fields,
    max_per_type=15,
    entities_key="gt_entities",
    annotation_scope="exhaustive",
):
    """The NER rule-writer prompt: labeled spans grouped by type, in context.

    Asks for per-type rules and type-vs-type tie-breakers.

    Args:
        induce_rows: The dev rows.
        labels: The entity types.
        task_hint: One line describing the task.
        text_fields: The input field(s); the first is the text spans are shown in.
        max_per_type: Spans shown per type.
        entities_key: The gold entity-list field.
        annotation_scope: ``exhaustive`` (every matching mention, CoNLL-style) or
            ``notable`` (only notable, knowledge-base-linked mentions, which raised
            micro and macro F1 on selectively annotated gold). A small seed cannot tell
            the two apart, so it is an explicit choice.

    Returns:
        The prompt.
    """
    text_field = text_fields[0] if text_fields else "text"
    by = collections.defaultdict(list)
    for r in induce_rows:
        text = r.get(text_field, "")
        for e in r.get(entities_key) or []:
            if isinstance(e, dict) and e.get("text") and e.get("type"):
                by[e["type"]].append((e["text"].strip(), text))
    blocks = []
    type_counts = {}
    for lbl in labels:
        n = len(by.get(lbl, []))
        type_counts[lbl] = n
        ex = by.get(lbl, [])[:max_per_type]
        lines = [f"## Type: {lbl}  ({n} labeled spans)"]
        for span, text in ex:
            clean_span = _safe_prompt_text(span)
            lines.append(f'- "{clean_span}"   in: ...{_span_context(text, clean_span)}...')
        blocks.append("\n".join(lines))
    examples_block = "\n\n".join(blocks)

    # Opt-in notable/KB-linked scope directive (see the docstring).
    convention_block = ""
    if annotation_scope == "notable":
        convention_block = """

IMPORTANT - ANNOTATION CONVENTION: this corpus tags entities SELECTIVELY. Tag a \
span ONLY when it names a specific, notable entity (one that would plausibly have \
its own encyclopedia / knowledge-base entry) - and SKIP generic common-noun \
mentions used descriptively, even when they technically match a type (e.g. a bare \
"coyote", "coconut", or "palm wine" mentioned in passing is NOT tagged). Your spec \
MUST include a "Scope / annotation convention" section stating this. It is the \
single biggest precision lever on selectively-annotated corpora."""

    return f"""You are writing a Named-Entity-Recognition labeling specification from labeled examples. {task_hint}

The entity types are: {labels}

Below are labeled spans grouped by type, each shown in its sentence context. Study \
what surface and contextual cues distinguish each type from the others - especially \
the pairs that are easy to confuse - then write a spec a different annotator could \
follow to reproduce these labels on NEW unseen text.

{examples_block}{convention_block}

Write a markdown labeling spec with:
1. A one-line task description.
2. The type list with a one-line definition each.
3. For EACH type, 2-4 concrete discriminating rules drawn from the contexts above \
(what cues push a span INTO this type vs a confusable other type). Be specific, not generic.
4. A "Type-vs-type tie-breakers" section for the ambiguous cases the examples reveal.
5. An output schema: the labeling job's own JSON reply format, verbatim, and no other \
reply format (no `REASONING:` or `ENTITIES:` lines):
   {{"reasoning": "<one sentence>", "entities": [{{"type": "<one of the entity types>", \
"text": "<exact span from the input>"}}]}}
   with `"entities": []` when the text has none.

Do NOT memorize or quote individual examples verbatim. Write general rules that \
transfer. Output ONLY the markdown spec."""


def ner_type_counts(
    induce_rows: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    entities_key: str = "gt_entities",
) -> Dict[str, int]:
    """Labeled spans per entity type in ``labels``, as the NER prompt counts them."""
    counts = {lbl: 0 for lbl in labels}
    for r in induce_rows:
        for e in r.get(entities_key) or []:
            if isinstance(e, dict) and e.get("text") and e.get("type") and e["type"] in counts:
                counts[e["type"]] += 1
    return counts


def invoke_reflection(prompt: str, model_id: str, region: Optional[str]) -> str:
    """The rule-writer call through the shared Bedrock client.

    The induced spec is long, so the token budget (16,000) and the read timeout (600 s)
    are raised; a ``ReadTimeoutError`` is not retryable.
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators._bedrock import (
        invoke,
        make_client,
    )

    client = make_client(region, read_timeout=600, connect_timeout=30)
    return invoke(client, model_id, prompt, temperature=0.3, max_tokens=16000).strip()


def induce_spec(  # pylint: disable=too-many-locals,too-many-branches,too-many-arguments
    rows: Sequence[Dict[str, Any]],
    *,
    task: str = "classification",
    labels: Optional[Sequence[str]] = None,
    task_hint: str = "",
    text_fields: Sequence[str] = ("text",),
    gt_field: str = "gt_label",
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    anchors_per_level: int = 2,
    max_per_class: int = 12,
    min_type_support: int = 3,
    annotation_scope: str = "exhaustive",
    reply_keys_line: bool = False,
    seed_num: int = 0,
    model_id: str = DEFAULT_MODEL,
    region: Optional[str] = None,
    invoke: Optional[Invoke] = None,
) -> InducedSpec:
    """Write a labeling spec's rules (or an anchored rubric) from the dev split.

    Use it on the induce pool only, once the task and the label set are settled, when
    the person has a labeled seed but no rules yet. Do not run it on val or test rows:
    the rules would be scored against their own evidence. Its output is a candidate
    spec; the gold gate on val decides whether it ships.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec import induce

        induced = induce.induce_spec(dev_rows, task="classification", labels=["TP", "FP"],
                                     task_hint="Does the ticket describe a defeat device?", region="us-west-2")
        open("proj/work/labeling_spec_induced.md", "w").write(induced.spec)

    Args:
        rows: The dev rows (the caller guards the read).
        task: ``classification``, ``ner``, ``llm_judge`` or ``ordinal``.
        labels: The label set (classification) or entity types (ner); not used for the
            rubric tasks, whose levels come from the scale.
        task_hint: One line describing the task, to steer the writer.
        text_fields: The input field(s); several for a relational task.
        gt_field: The gold field. ``gt_label`` defaults to ``gt_score`` for the rubric
            tasks.
        score_min: Rubric tasks: the low end of the score scale (required there).
        score_max: Rubric tasks: the high end.
        anchors_per_level: Rubric tasks: dev rows quoted per level, 1 or 2.
        max_per_class: Examples (or spans) shown per class (or type) in the prompt.
        min_type_support: NER: a type with fewer spans gets a warning.
        annotation_scope: NER: ``exhaustive`` tags every matching mention; ``notable``
            tags only notable, knowledge-base-linked entities (a selectively annotated
            corpus), which raised micro and macro F1 on such gold.
        reply_keys_line: Classification: end the Output Schema with the line asking for
            the keys the job appends (off by default; it costs output tokens per row).
        seed_num: The RNG seed for the example order, so a seed yields the same prompt.
        model_id: The rule-writer model (a Bedrock model id).
        region: The Bedrock region.
        invoke: ``(prompt, model_id, region) -> text``; :func:`invoke_reflection` by
            default.

    Returns:
        An :class:`InducedSpec`: write ``spec`` to the spec file; ``summary`` is the
        count report; ``warnings`` the thin-evidence notes (a level with no anchor, an
        entity type under ``min_type_support``).

    Raises:
        DataPreparerOperatorError: A rubric task without a scale, a bad scale,
            ``labels`` missing for classification or NER, no row with a numeric gold,
            or no labeled row.
    """
    labels = list(labels or [])
    text_fields = [f.strip() for f in text_fields if f and f.strip()]
    call = invoke or invoke_reflection
    notes: List[str] = []
    if task in RUBRIC_TASKS:
        if score_min is None or score_max is None:
            raise DataPreparerOperatorError(
                f"induce_spec: --task {task} needs --score-min and --score-max "
                "(the anchored rubric has one level per score)."
            )
        if gt_field == "gt_label":
            gt_field = "gt_score"
    elif not labels:
        raise DataPreparerOperatorError(f"induce_spec: --label-set is required for --task {task}")
    if task in RUBRIC_TASKS:
        try:
            anchors, missing = select_rubric_anchors(
                rows, score_min, score_max, per_level=anchors_per_level, gt_key=gt_field
            )
        except DataPreparerOperatorError as e:
            raise DataPreparerOperatorError(f"induce_spec: {e}") from e
        n_anchor = sum(len(v) for v in anchors.values())
        if not n_anchor:
            raise DataPreparerOperatorError(
                f"induce_spec: no row in --seed has a numeric {gt_field}"
            )
        if missing:
            notes.append(
                "induce_spec WARNING: no dev row near score level(s) "
                + ", ".join(format_level(v) for v in missing)
                + "; those levels are defined from the scale alone. Add seed rows at "
                "those scores for a fully anchored rubric."
            )
        prompt = build_rubric_induction_prompt(
            anchors,
            missing,
            task_hint,
            text_fields,
            score_min,
            score_max,
            task=task,
            gt_key=gt_field,
        )
        spec = _safe_spec_text(call(prompt, model_id, region))
        summary = {
            "task": task,
            "levels": [format_level(v) for v in anchors],
            "anchor_rows": n_anchor,
            "levels_without_anchor": [format_level(v) for v in missing],
            "chars": len(spec),
        }
        return InducedSpec(spec=spec, prompt=prompt, summary=summary, warnings=notes)
    if task == "ner":
        rows = [r for r in rows if r.get(gt_field)]
    else:
        # keep any label that is present, including falsy-but-valid ones (0, "")
        rows = [r for r in rows if r.get(gt_field) is not None]
    if not rows:
        raise DataPreparerOperatorError(
            f"induce_spec: no labeled rows in --seed (need {gt_field} present)"
        )
    rows = list(rows)
    random.Random(seed_num).shuffle(rows)
    if task == "ner":
        prompt = build_ner_induction_prompt(
            rows,
            labels,
            task_hint,
            text_fields,
            max_per_type=max_per_class,
            entities_key=gt_field,
            annotation_scope=annotation_scope,
        )
        counts = ner_type_counts(rows, labels, gt_field)
        starved = {t: n for t, n in counts.items() if n < min_type_support}
        if starved:
            notes.append(
                "induce_spec WARNING: entity types below --min-type-support "
                f"({min_type_support}) \u2014 rules for these will be unreliable; "
                f"add more seed examples: {json.dumps(starved, sort_keys=True)}"
            )
    else:
        prompt = build_induction_prompt(
            rows,
            labels,
            task_hint,
            text_fields,
            max_per_class=max_per_class,
            label_key=gt_field,
            reply_keys_line=reply_keys_line,
        )
    spec = _safe_spec_text(call(prompt, model_id, region))
    summary = {"labels": labels, "induce_rows": len(rows), "chars": len(spec)}
    return InducedSpec(spec=spec, prompt=prompt, summary=summary, warnings=notes)


# ---------------------------------------------------------------------------
# The project-level step (full-loop.md Step 1b): induce the spec's rules from the
# project's dev split and write the draft where the labeling rounds pick it up.
# ---------------------------------------------------------------------------


class SpecDraft(OperatorResult):
    """What :func:`draft_labeling_spec` wrote: ``path`` is what a labeling round takes as its spec."""

    path: str
    iteration: int
    spec: str
    summary: Dict[str, Any]


def draft_labeling_spec(
    project_dir: str,
    data,
    *,
    iteration: int = 0,
    task_hint: str = "",
    model_id: str = DEFAULT_MODEL,
    out: Optional[str] = None,
    **options: Any,
) -> SpecDraft:
    """Write a spec's rules from the project's dev split: ``work/labeling_spec_induced.md``.

    Step 1b of the loop, optional. The dev split is read under the split guard (a val or
    test row would make the rules score against their own evidence), the rules are
    induced (:func:`induce_spec`) with the task, label set, fields and score scale
    ``data`` carries, and the draft is written. It is a candidate: the refinement rounds
    score it and :func:`~..select.select_labeling_spec` decides whether it ships.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import draft_labeling_spec

        draft = draft_labeling_spec("tickets", data, task_hint="Which intent does the ticket express?")
        draft.path   # tickets/work/labeling_spec_induced.md, for tune_labeling_spec(spec=...)

    Args:
        project_dir: The project; its ``splits/dev.jsonl`` is the induce pool.
        data: :func:`~..inspect.inputs.inspect_labeling_data`'s report.
        iteration: The refinement iteration the draft belongs to (its run-log entry).
        task_hint: One line describing the task, to steer the writer.
        model_id: The reflection model.
        out: The file to write instead of the default.
        **options: :func:`induce_spec`'s other keyword arguments.

    Returns:
        A :class:`SpecDraft`.
    """
    import os

    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows, _run_log
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import run_log_event
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split

    dev_path = os.path.join(project_dir, "splits", "dev.jsonl")
    rows = _rows.read_jsonl(dev_path)
    guard = split.guard_read(dev_path, "induce", {"dev"}, project_dir=project_dir, rows=rows)
    _run_log.append_entries(project_dir, guard.events)
    _rows.report_client_read("draft_labeling_spec", len(rows))
    scale = data.score_scale or (None, None)
    induced = induce_spec(
        rows,
        task=data.task,
        labels=data.labels or None,
        task_hint=task_hint,
        text_fields=tuple(data.text_fields),
        gt_field=data.label_field,
        score_min=scale[0],
        score_max=scale[1],
        model_id=model_id,
        region=data.region,
        **options,
    )
    path = out or os.path.join(project_dir, "work", "labeling_spec_induced.md")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(induced.spec + "\n")
    _run_log.append_entry(
        project_dir,
        run_log_event(
            "artifact",
            "induce",
            iteration,
            "induced",
            spec=path,
            model_id=model_id,
            **{k: v for k, v in induced.summary.items() if k != "labels"},
        ),
    )
    return SpecDraft(
        path=path,
        iteration=iteration,
        spec=induced.spec,
        summary=induced.summary,
        warnings=list(induced.warnings),
    )
