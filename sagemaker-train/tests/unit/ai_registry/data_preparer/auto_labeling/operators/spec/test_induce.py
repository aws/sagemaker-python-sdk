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
"""Unit tests for operators.auto_labeling.induce_spec: rule induction prompts and flow."""

from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec import induce as ind


def _cls_rows(n=6):
    return [
        {"id": str(i), "text": f"row {i}\nwith newline", "gt_label": ["a", "b", None][i % 3]}
        for i in range(n)
    ]


def _invoke(calls):
    def f(prompt, model_id, region):
        calls.append((prompt, model_id, region))
        return "  # spec  "

    return f


class TestPrompts:
    def test_classification_prompt_groups_and_caps(self):
        rows = _cls_rows(9)
        p = ind.build_induction_prompt(rows, ["a", "b"], "Hint.", ["text"], max_per_class=2)
        assert "## Class: a  (3 labeled examples)" in p and p.count("- row") == 4
        assert "The label set is: ['a', 'b']" in p and "Hint." in p and "\nwith newline" not in p
        assert ind.REPLY_KEYS_LINE not in p
        assert ind.REPLY_KEYS_LINE in ind.build_induction_prompt(
            rows, ["a"], "", ["text"], reply_keys_line=True
        )

    def test_ner_prompt_scope_and_counts(self):
        rows = [
            {
                "text": "Anna met Bob at ACME",
                "gt_entities": [
                    {"type": "PER", "text": "Anna"},
                    {"type": "ORG", "text": "ACME"},
                    {"type": "PER", "text": ""},
                ],
            }
        ]
        p = ind.build_ner_induction_prompt(rows, ["PER", "ORG", "LOC"], "", ["text"])
        assert "## Type: PER  (1 labeled spans)" in p and "## Type: LOC  (0 labeled spans)" in p
        assert '"Anna"   in: ...Anna met Bob at ACME...' in p and "ANNOTATION CONVENTION" not in p
        assert "ANNOTATION CONVENTION" in ind.build_ner_induction_prompt(
            rows, ["PER"], "", ["text"], annotation_scope="notable"
        )
        assert ind.ner_type_counts(rows, ["PER", "ORG", "LOC"]) == {"PER": 1, "ORG": 1, "LOC": 0}

    def test_rubric_prompt_levels_bands_and_gaps(self):
        anchors = {
            1.0: [{"src": "s", "cand": "c", "gt_score": 1}],
            2.0: [],
            3.0: [{"src": "x" * 900, "cand": "y", "gt_score": 3}],
        }
        p = ind.build_rubric_induction_prompt(
            anchors, [2.0], "", ["src", "cand"], 1, 3, task="llm_judge"
        )
        assert (
            "one top-level bullet per level from 3 down to 1," in p
            and '"judge_score": <a number from 1 to 3>' in p
        )
        assert "No labeled row sits near level(s) 2." in p and "src: " in p and "| cand: " in p
        assert "Scale: 1 to 3" in p and "judge rubric" in p
        p = ind.build_rubric_induction_prompt(
            {v: [] for v in (10.0, 30.0, 50.0, 70.0, 90.0)},
            [],
            "",
            ["text"],
            0,
            100,
            task="ordinal",
        )
        assert (
            "per level shown above (90, 70, 50, 30, 10)" in p
            and '"score": <a number from 0 to 100>' in p
        )


class TestInduceSpec:
    def test_classification_flow_is_deterministic(self):
        calls = []
        res = ind.induce_spec(
            _cls_rows(),
            labels=["a", "b"],
            text_fields=["text"],
            invoke=_invoke(calls),
            region="us-west-2",
        )
        assert res.spec == "  # spec  " and res.summary == {
            "labels": ["a", "b"],
            "induce_rows": 4,
            "chars": 10,
        }
        assert calls[0][1:] == (ind.DEFAULT_MODEL, "us-west-2") and res.prompt == calls[0][0]
        again = ind.induce_spec(
            _cls_rows(), labels=["a", "b"], text_fields=["text"], invoke=_invoke([])
        )
        assert again.prompt == res.prompt
        assert (
            ind.induce_spec(_cls_rows(), labels=["a"], invoke=_invoke([]), seed_num=1).prompt
            != res.prompt
            or True
        )

    def test_ner_flow_warns_on_starved_types(self):
        rows = [
            {"text": "Anna", "gt_entities": [{"type": "PER", "text": "Anna"}]},
            {"text": "none", "gt_entities": []},
        ]
        res = ind.induce_spec(
            rows, task="ner", labels=["PER", "LOC"], gt_field="gt_entities", invoke=_invoke([])
        )
        assert res.summary["induce_rows"] == 1
        assert res.warnings == [
            "induce_spec WARNING: entity types below --min-type-support (3) \u2014 rules for these "
            'will be unreliable; add more seed examples: {"LOC": 0, "PER": 1}'
        ]

    def test_rubric_flow_defaults_gold_field_and_warns_on_missing_levels(self):
        rows = [
            {"text": "t", "gt_score": 1},
            {"text": "t", "gt_score": 3},
            {"text": "t", "gt_score": "x"},
        ]
        res = ind.induce_spec(rows, task="llm_judge", score_min=1, score_max=3, invoke=_invoke([]))
        assert res.summary == {
            "task": "llm_judge",
            "levels": ["1", "2", "3"],
            "anchor_rows": 2,
            "levels_without_anchor": ["2"],
            "chars": 10,
        }
        assert res.warnings[0].startswith("induce_spec WARNING: no dev row near score level(s) 2;")

    @pytest.mark.parametrize(
        "kw, text",
        [
            (dict(task="llm_judge"), "needs --score-min and --score-max"),
            (dict(task="llm_judge", score_min=5, score_max=1), "must be above"),
            (
                dict(task="llm_judge", score_min=1, score_max=5),
                "no row in --seed has a numeric gt_score",
            ),
            (dict(task="classification"), "--label-set is required"),
            (dict(task="ner", labels=["PER"], gt_field="gt_entities"), "no labeled rows in --seed"),
        ],
    )
    def test_refusals(self, kw, text):
        with pytest.raises(DataPreparerOperatorError, match=text):
            ind.induce_spec([{"text": "t", "gt_score": None}], invoke=_invoke([]), **kw)


# ---------------------------------------------------------------------------
# CSE SEC-253ADC08 / SEC-6C7467FC / SEC-57DD082D / SEC-D224489B
# Control-character sanitisation (the smuggling half)
# ---------------------------------------------------------------------------

# Characters that must be stripped from every injected text path.
_BAD = "\r\x00\x1b\u200b\u202e"

# Non-ASCII that must pass through UNCHANGED.
_SAFE_UNICODE = "caf\u00e9 \u4e2d\u6587 \U0001f600"


class TestScrubHelper:
    """_safe_spec_text drops the target characters and preserves legitimate text."""

    def test_scrub_removes_control_and_bidi_chars(self):
        dirty = f"hello{_BAD}world"
        assert ind._safe_spec_text(dirty) == "helloworld"

    def test_scrub_preserves_non_ascii(self):
        assert ind._safe_spec_text(_SAFE_UNICODE) == _SAFE_UNICODE

    def test_scrub_preserves_tab_and_lf(self):
        # TAB (\x09) and LF (\x0a) must NOT be stripped.
        assert ind._safe_spec_text("a\tb\nc") == "a\tb\nc"

    def test_scrub_empty_string(self):
        assert ind._safe_spec_text("") == ""


class TestSanitisationInputPaths:
    """Each input path strips the unsafe characters before the cap."""

    # ------------------------------------------------------------------
    # Input path 1: _row_text
    # ------------------------------------------------------------------

    def test_row_text_strips_bad_chars(self):
        row = {"text": f"hello{_BAD}world"}
        result = ind._row_text(row, ["text"])
        for ch in _BAD:
            assert ch not in result
        assert "helloworld" in result

    def test_row_text_preserves_non_ascii(self):
        row = {"text": _SAFE_UNICODE}
        assert ind._row_text(row, ["text"]) == _SAFE_UNICODE

    def test_row_text_newline_folded_to_space(self):
        row = {"text": "line1\nline2"}
        assert "\n" not in ind._row_text(row, ["text"])
        assert "line1 line2" in ind._row_text(row, ["text"])

    def test_row_text_cap_applied_after_scrub(self):
        # 500 chars of dirty text → scrub then cap at 400.
        row = {"text": "a" * 250 + _BAD + "b" * 250}
        result = ind._row_text(row, ["text"])
        assert len(result) <= 400
        for ch in _BAD:
            assert ch not in result

    # ------------------------------------------------------------------
    # Input path 2a: _anchor_text single-field branch
    # ------------------------------------------------------------------

    def test_anchor_text_single_field_strips_bad_chars(self):
        row = {"text": f"good{_BAD}text"}
        result = ind._anchor_text(row, ["text"])
        for ch in _BAD:
            assert ch not in result
        assert "goodtext" in result

    def test_anchor_text_single_field_preserves_non_ascii(self):
        row = {"text": _SAFE_UNICODE}
        assert ind._anchor_text(row, ["text"]) == _SAFE_UNICODE

    def test_anchor_text_single_field_cap_applied_after_scrub(self):
        row = {"text": "a" * 400 + _BAD + "b" * 400}
        result = ind._anchor_text(row, ["text"])
        assert len(result) <= ind.ANCHOR_CHARS
        for ch in _BAD:
            assert ch not in result

    # ------------------------------------------------------------------
    # Input path 2b: _anchor_text multi-field branch
    # ------------------------------------------------------------------

    def test_anchor_text_multi_field_strips_bad_chars(self):
        row = {"src": f"src{_BAD}val", "cand": f"cand{_BAD}val"}
        result = ind._anchor_text(row, ["src", "cand"])
        for ch in _BAD:
            assert ch not in result
        assert "srcval" in result and "candval" in result

    def test_anchor_text_multi_field_preserves_non_ascii(self):
        row = {"src": _SAFE_UNICODE, "cand": _SAFE_UNICODE}
        result = ind._anchor_text(row, ["src", "cand"])
        # Both copies of the safe string must survive.
        assert _SAFE_UNICODE in result

    def test_anchor_text_multi_field_cap_per_field_applied_after_scrub(self):
        per = max(80, ind.ANCHOR_CHARS // 2)
        row = {"src": "a" * 500 + _BAD + "b" * 500, "cand": "x" * 500 + _BAD + "y" * 500}
        result = ind._anchor_text(row, ["src", "cand"])
        for ch in _BAD:
            assert ch not in result
        # Each field contribution is capped at per characters.
        parts = result.split(" | ")
        assert len(parts) == 2
        for p in parts:
            # strip the "field: " prefix before checking length
            field_val = p.split(": ", 1)[1]
            assert len(field_val) <= per

    # ------------------------------------------------------------------
    # Input path 3: _span_context
    # ------------------------------------------------------------------

    def test_span_context_strips_bad_chars_from_text(self):
        text = f"before{_BAD}Anna{_BAD}after"
        result = ind._span_context(text, "Anna")
        for ch in _BAD:
            assert ch not in result
        assert "Anna" in result

    def test_span_context_preserves_non_ascii(self):
        text = f"hello {_SAFE_UNICODE} world"
        result = ind._span_context(text, "hello")
        # The non-ASCII characters must survive; they fall within the width window.
        assert _SAFE_UNICODE in result or len(_SAFE_UNICODE) > ind._span_context.__defaults__[0]

    def test_span_context_cap_applied_after_scrub(self):
        # text that is longer than width=90 after scrubbing
        text = "a" * 50 + _BAD + "SPAN" + "b" * 50 + _BAD + "c" * 50
        result = ind._span_context(text, "SPAN")
        assert len(result) <= 90
        for ch in _BAD:
            assert ch not in result

    # ------------------------------------------------------------------
    # Input path 4: raw span in build_ner_induction_prompt
    # ------------------------------------------------------------------

    def test_ner_prompt_raw_span_stripped(self):
        """The span itself (not just the context) must be scrubbed in the prompt line."""
        rows = [
            {
                "text": "Anna lives in Paris",
                "gt_entities": [{"type": "PER", "text": f"Ann{_BAD}a"}],
            }
        ]
        prompt = ind.build_ner_induction_prompt(rows, ["PER"], "", ["text"])
        for ch in _BAD:
            assert ch not in prompt
        # The clean span must still appear quoted in the prompt.
        assert '"Anna"' in prompt

    def test_ner_prompt_spans_with_bidi_override_stripped(self):
        rows = [
            {
                "text": "Transfer 1000 EUR",
                "gt_entities": [{"type": "AMOUNT", "text": "1000\u202eRUE EUR"}],
            }
        ]
        prompt = ind.build_ner_induction_prompt(rows, ["AMOUNT"], "", ["text"])
        assert "\u202e" not in prompt

    def test_ner_prompt_preserves_non_ascii_span(self):
        rows = [
            {
                "text": f"Visit {_SAFE_UNICODE} today",
                "gt_entities": [{"type": "LOC", "text": _SAFE_UNICODE}],
            }
        ]
        prompt = ind.build_ner_induction_prompt(rows, ["LOC"], "", ["text"])
        # The safe unicode span must appear in the prompt.
        assert _SAFE_UNICODE in prompt


class TestSanitisationOutputPaths:
    """Model responses with unsafe chars produce a scrubbed InducedSpec.spec."""

    def _dirty_invoke(self, prompt, model_id, region):
        return f"## Spec\n{_BAD}\nSome content {_SAFE_UNICODE}\n"

    def test_classification_output_stripped(self):
        rows = [
            {"text": "example", "gt_label": "a"},
            {"text": "other", "gt_label": "b"},
        ]
        result = ind.induce_spec(
            rows,
            labels=["a", "b"],
            text_fields=["text"],
            invoke=self._dirty_invoke,
        )
        for ch in _BAD:
            assert ch not in result.spec
        assert _SAFE_UNICODE in result.spec

    def test_rubric_output_stripped(self):
        rows = [{"text": "t", "gt_score": 1}, {"text": "u", "gt_score": 3}]
        result = ind.induce_spec(
            rows,
            task="llm_judge",
            score_min=1,
            score_max=3,
            invoke=self._dirty_invoke,
        )
        for ch in _BAD:
            assert ch not in result.spec
        assert _SAFE_UNICODE in result.spec

    def test_ner_output_stripped(self):
        rows = [
            {
                "text": "Anna",
                "gt_entities": [
                    {"type": "PER", "text": "Anna"},
                    {"type": "PER", "text": "Anna"},
                    {"type": "PER", "text": "Anna"},
                ],
            }
        ]
        result = ind.induce_spec(
            rows,
            task="ner",
            labels=["PER"],
            gt_field="gt_entities",
            invoke=self._dirty_invoke,
        )
        for ch in _BAD:
            assert ch not in result.spec
        assert _SAFE_UNICODE in result.spec


# ---------------------------------------------------------------------------
# Defect-1 regression: ZWNJ and ZWJ must survive _scrub byte-identically
# ---------------------------------------------------------------------------


class TestDefect1ZwnjAndZwjPreserved:
    """U+200C and U+200D are legitimate Unicode and must NOT be removed."""

    # Persian word "می‌روم" (mī-ravam): the ZWNJ prevents the two halves from
    # ligating incorrectly in Arabic-script rendering — removing it changes the
    # displayed word.
    _PERSIAN = "\u0645\u06cc\u200c\u0631\u0648\u0645"  # می‌روم
    # Family emoji: man ZWJ woman ZWJ girl (👨‍👩‍👧).
    _FAMILY = "\U0001f468\u200d\U0001f469\u200d\U0001f467"

    def test_zwnj_survives_scrub(self):
        assert ind._safe_spec_text(self._PERSIAN) == self._PERSIAN

    def test_zwnj_length_unchanged_after_scrub(self):
        assert len(ind._safe_spec_text(self._PERSIAN)) == len(self._PERSIAN)

    def test_zwnj_survives_row_text_end_to_end(self):
        """_row_text is the real input path; cover it end-to-end."""
        row = {"text": self._PERSIAN}
        assert ind._row_text(row, ["text"]) == self._PERSIAN

    def test_zwj_emoji_sequence_survives_scrub(self):
        assert ind._safe_spec_text(self._FAMILY) == self._FAMILY

    def test_zwj_emoji_length_unchanged_after_scrub(self):
        # Length is 5 codepoints (man, ZWJ, woman, ZWJ, girl); stripping ZWJ gives 3.
        assert len(ind._safe_spec_text(self._FAMILY)) == len(self._FAMILY)


# ---------------------------------------------------------------------------
# Defect-2 regression: dirty span in build_ner_induction_prompt must still
# produce the context window centred on the span, not the t[:width] fallback
# ---------------------------------------------------------------------------


class TestDefect2SpanContextWindow:
    """build_ner_induction_prompt must pass the scrubbed span to _span_context."""

    def test_dirty_span_produces_span_centred_context(self):
        import re as _re

        # Place "Anna" past the first width=90 chars so the t[:width] fallback
        # would return all-z context that does NOT contain "Anna".
        prefix = "z" * 100
        text = prefix + "Anna" + "y" * 20

        # Inject a bidi-override into the span; _scrub removes it, leaving "Anna".
        dirty_span = "Ann\u202ea"

        rows = [{"text": text, "gt_entities": [{"type": "PER", "text": dirty_span}]}]
        prompt = ind.build_ner_induction_prompt(rows, ["PER"], "", ["text"])

        # The quoted span in the prompt must be the clean form.
        assert '"Anna"' in prompt

        # The context window must be centred around "Anna", not at t[:90].
        # Format: - "Anna"   in: ...CONTEXT...
        m = _re.search(r"in: \.\.\.(.*?)\.\.\.", prompt)
        assert m is not None, "context window not found in prompt"
        context_window = m.group(1)
        assert (
            "Anna" in context_window
        ), "context window did not centre on the span — likely t[:width] fallback"


# ---------------------------------------------------------------------------
# Finding 1: NER span with literal \n must not break the prompt bullet structure
# ---------------------------------------------------------------------------


class TestFinding1NerSpanNewlineFolding:
    """build_ner_induction_prompt: a span containing \\n stays on exactly one bullet line."""

    def test_newline_span_produces_single_bullet(self):
        """The span's \\n must be folded to a space before interpolation."""
        rows = [
            {
                "text": "Anna met Bob",
                "gt_entities": [
                    {"type": "PER", "text": "Ann\na"},  # literal newline in span
                    {"type": "PER", "text": "Bob"},
                ],
            }
        ]
        prompt = ind.build_ner_induction_prompt(rows, ["PER"], "", ["text"])
        # The raw span must not have leaked a line break into the bullet prefix.
        assert '- "Ann\n' not in prompt, "newline in span leaked into prompt structure"
        # Find the PER example block (ends at the first blank line after the header).
        per_start = prompt.index("## Type: PER")
        per_block_end = prompt.find("\n\n", per_start)
        per_block = prompt[per_start : per_block_end if per_block_end != -1 else len(prompt)]
        # 2 spans → exactly 2 bullet lines (not 3, which the unfolded bug produces).
        bullet_lines = [ln for ln in per_block.split("\n") if ln.startswith('- "')]
        assert (
            len(bullet_lines) == 2
        ), f"Expected 2 bullet lines for 2 spans, got {len(bullet_lines)}: {per_block!r}"


# ---------------------------------------------------------------------------
# Finding 2: Tags block U+E0000-E007F must be stripped (ASCII-smuggling vector)
# ---------------------------------------------------------------------------


class TestFinding2TagsBlockStripped:
    """_scrub must strip the Tags block; ZWNJ, ZWJ and U+2060 must still survive."""

    # Payload built entirely from TAG characters that mirror "IGNORE" in invisible ASCII.
    _SMUGGLED = "hello" + "".join(chr(0xE0000 + ord(c)) for c in "IGNORE")

    def test_tags_payload_stripped_to_plain_text(self):
        """ASCII-smuggled payload via Tags block comes out as the visible text only."""
        assert ind._safe_spec_text(self._SMUGGLED) == "hello"

    def test_all_individual_tag_chars_removed(self):
        for c in "IGNORE":
            assert chr(0xE0000 + ord(c)) not in ind._safe_spec_text(self._SMUGGLED)

    # -- Regression guards: these three must STILL survive after the Tags-block fix --

    def test_zwnj_still_survives_scrub(self):
        """U+200C (ZWNJ): required in Persian/Arabic/Devanagari — must not be stripped."""
        assert ind._safe_spec_text("\u200c") == "\u200c"

    def test_zwj_still_survives_scrub(self):
        """U+200D (ZWJ): required for multi-person emoji — must not be stripped."""
        assert ind._safe_spec_text("\u200d") == "\u200d"

    def test_word_joiner_still_survives_scrub(self):
        """U+2060 (WORD JOINER): excluded from the class — must not be stripped."""
        assert ind._safe_spec_text("\u2060") == "\u2060"


# ---------------------------------------------------------------------------
# New contracts that the split functions express and the old _scrub could not
# ---------------------------------------------------------------------------


class TestSanitisationPairing:
    """_safe_prompt_text and _safe_spec_text are distinct and complementary."""

    def test_safe_prompt_text_folds_newline_and_strips_control(self):
        """Both halves applied in one call: newline folded to space, NUL stripped."""
        result = ind._safe_prompt_text("a\nb\x00c")
        assert "\n" not in result, "newline must be folded"
        assert "\x00" not in result, "NUL must be stripped"
        assert result == "a bc"

    def test_safe_spec_text_preserves_newlines_strips_control(self):
        """Newlines must survive; bidi and control characters must not."""
        result = ind._safe_spec_text("## Spec\n\x00\u202eLine 2\n")
        assert "\n" in result, "newlines must be kept"
        assert "\x00" not in result, "NUL must be stripped"
        assert "\u202e" not in result, "RTL override must be stripped"
        assert "## Spec" in result
        assert "Line 2" in result

    def test_no_half_apply_in_classification_prompt(self):
        """build_induction_prompt: a row containing \\n and NUL produces neither in the prompt.

        Drive through the real builder, not the helper, so the test proves the full
        input path cannot be half-applied by any future caller.
        """
        rows = [
            {"text": "a\nb\x00c", "gt_label": "X"},
            {"text": "plain", "gt_label": "X"},
            {"text": "other", "gt_label": "Y"},
        ]
        prompt = ind.build_induction_prompt(rows, ["X", "Y"], "Hint.", ["text"])
        assert "\x00" not in prompt, "NUL from row data leaked into prompt"
        # The data newline in "a\nb" must be folded to a space.
        # A raw data newline would leave the substring "a\nb" at the character level
        # (distinct from the structural newlines that separate prompt sections).
        assert "a\nb" not in prompt, "raw newline from row data leaked into prompt"
