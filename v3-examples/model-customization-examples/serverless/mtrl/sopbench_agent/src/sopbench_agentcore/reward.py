"""SOP-Bench reward scoring.

The rollout's answer is the LAST assistant turn that contains a tagged answer: the prompt's
`<task_result><answer_json>{...}</answer_json></task_result>`, or SOP-Bench's own
`<final_output>{...}</final_output>` / `<final_response>...</final_response>` formats. It is
parsed with SOP-Bench's OutputParser and scored field by field against the ground truth:

* dense  - fraction of ground-truth fields answered correctly (0.0-1.0)
* binary - 1.0 only if every field is correct (SOP-Bench's task success rate, TSR)
* reward - dense * FINAL_REWARD_SCALE, or NO_ANSWER_REWARD if the rollout never answered

Fields are compared with SOP-Bench's own value comparison so scores line up with the published
benchmark: case-insensitive, null-like values ("NA", null, NaN) are equivalent, and a value that
contains the expected one (or is contained in it) also matches - so, as in SOP-Bench,
"failed (low battery)" counts as "failed". One guard is added: a value that matches MORE THAN ONE
of the field's allowed values (for example "success or fail", "true false", or "e", which is in
both "true" and "false") never matches. Without it, one answer listing every option would score
full credit on every task.
"""

import ast
import functools
import json
import math
import re

from amazon_sop_bench.evaluation.parser import OutputParser

FINAL_REWARD_SCALE = 5.0
NO_ANSWER_REWARD = -1.0

_parser = OutputParser()

# The prompt asks for <task_result>/<answer_json> tags, renamed from SOP-Bench's
# <final_response>/<final_output> because gpt-oss confuses a tag containing "final" with its
# Harmony `final` channel (see prompts.py). The SOP-Bench parser only knows the original names,
# so tags are rewritten back at scoring time only.
_ANSWER_TAG_ALIASES = (
    (re.compile(r"<(/?)answer_json>", re.I), r"<\1final_output>"),
    (re.compile(r"<(/?)task_result>", re.I), r"<\1final_response>"),
)
_ANSWER_MARKER = re.compile(r"<(final_output|final_response)>", re.I)
_TAGGED_DICT = re.compile(r"<(final_output|final_response)>\s*(\{.*?\})\s*</\1>", re.DOTALL | re.I)
# Null-like values, as in SOP-Bench's comparison.
_NULL_LIKE = {"none", "null", "na", "n/a", "nan", "unknown", ""}


def normalize_answer_tags(text: str) -> str:
    """Rewrite the prompt's answer tags to the SOP-Bench parser's tag names."""
    text = text or ""
    for pattern, repl in _ANSWER_TAG_ALIASES:
        text = pattern.sub(repl, text)
    return text


@functools.lru_cache(maxsize=None)
def field_vocabulary(benchmark_name: str = "aircraft_inspection") -> dict:
    """{field: frozenset of allowed values (lowercase)} across all of the benchmark's tasks."""
    from amazon_sop_bench.benchmarks import load_benchmark

    vocab = {}
    for task in load_benchmark(benchmark_name).tasks:
        for field, value in (task.expected_output or {}).items():
            if isinstance(value, float) and math.isnan(value):
                continue
            if str(value).strip().lower() not in _NULL_LIKE:
                vocab.setdefault(field.lower(), set()).add(str(value).strip().lower())
    return {field: frozenset(values) for field, values in vocab.items()}


def _clean(value) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _matches_several(predicted, allowed) -> bool:
    """True if `predicted` matches more than one allowed value under containment matching."""
    pred = _clean(predicted)
    hits = {v for v in map(_clean, allowed) if v and (v in pred or pred in v)}
    # "fail" inside "failed" is one answer, not two: keep only the longest overlapping values.
    hits = {v for v in hits if not any(v != w and v in w for w in hits)}
    return len(hits) > 1


def _tagged_dict(answer: str) -> dict:
    """The last tagged {...} dict in the answer (JSON or a Python literal), or {}."""
    for _, body in reversed(_TAGGED_DICT.findall(answer)):
        for loads in (json.loads, ast.literal_eval):
            try:
                obj = loads(body)
            except (ValueError, SyntaxError):
                continue
            if isinstance(obj, dict):
                return obj
    return {}


def _decisions(answer: str, ground_truth: dict) -> dict:
    """{field (lowercase): predicted value} for the ground-truth fields present in the answer."""
    cols = list(ground_truth)
    parsed = _parser.parse_decision(answer, cols)
    decisions = parsed.get("decisions")
    if not isinstance(decisions, dict):
        # The parser collapses an answer with only ONE expected field to a bare "decision"
        # without saying which field; read the dict directly so partial answers keep credit.
        decisions = _tagged_dict(answer)
    return {str(k).lower(): v for k, v in decisions.items()}


def _field_matches(field: str, predicted, expected, vocabulary: dict) -> bool:
    pred = str(predicted).strip().lower()
    if pred not in _NULL_LIKE and _matches_several(pred, vocabulary.get(field.lower(), ())):
        return False  # hedged: matches several allowed values
    return _parser._compare_single_value(predicted, expected)


def score_answer(answer: str, ground_truth: dict, vocabulary: dict = None) -> tuple:
    """(dense, binary) for one tagged answer."""
    if not ground_truth:
        return 0.0, 0.0
    vocabulary = field_vocabulary() if vocabulary is None else vocabulary
    decisions = _decisions(answer, ground_truth)
    hits = sum(
        1
        for field, expected in ground_truth.items()
        if str(field).lower() in decisions
        and _field_matches(field, decisions[str(field).lower()], expected, vocabulary)
    )
    return hits / len(ground_truth), 1.0 if hits == len(ground_truth) else 0.0


def last_answer(messages: list):
    """Normalized content of the last assistant turn that contains a tagged answer, or None.

    Turns with tool calls are skipped: the model is still working, and their content may be
    reasoning that quotes the prompt's example answer.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("tool_calls"):
            content = normalize_answer_tags(msg.get("content") or "")
            if _ANSWER_MARKER.search(content):
                return content
    return None


def compute_reward(messages: list, ground_truth: dict, vocabulary: dict = None) -> dict:
    """Score a rollout transcript. The top-level `reward` key is what @sagemaker_rft_handler
    reports. `vocabulary` defaults to the aircraft_inspection field vocabulary."""
    if not ground_truth:
        raise ValueError("ground_truth is required to score a rollout")
    answer = last_answer(messages)
    if answer is None:
        return {"reward": NO_ANSWER_REWARD, "dense": 0.0, "binary": 0.0, "answered": False}
    dense, binary = score_answer(answer, ground_truth, vocabulary)
    return {
        "reward": dense * FINAL_REWARD_SCALE,
        "dense": dense,
        "binary": binary,
        "answered": True,
    }
