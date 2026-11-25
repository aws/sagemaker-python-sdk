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
"""Unit tests for operators.auto_labeling.rubric: anchored-rubric levels, anchors and gaps."""

from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec import rubric

ANCHORED = """# Spec
### Score Levels
- **1 - Poor:** nothing
  - Example: it ignores the question
- **2 - Weak:** little
  - Example: partial
- **3 - Fair:** some
  - Example: ok
- **4 - Good:** most
  - Example: good
- **5 - Excellent:** all
  - Example: perfect
## Next
- **9:** stray
"""


class TestLevels:
    def test_integer_scales_and_bands(self):
        assert rubric.scale_levels(1, 5) == ([1.0, 2.0, 3.0, 4.0, 5.0], False)
        assert rubric.scale_levels(0, 10)[0] == [float(v) for v in range(11)]
        assert rubric.scale_levels(0, 100) == ([10.0, 30.0, 50.0, 70.0, 90.0], True)
        assert rubric.is_banded(0, 100) and not rubric.is_banded(1, 5)
        assert rubric.is_banded(0.5, 4.5)  # fractional ends: banded
        assert rubric.is_banded(1, 12)  # over MAX_INTEGER_LEVELS integers: banded

    def test_refusals(self):
        with pytest.raises(DataPreparerOperatorError, match="needs the score scale"):
            rubric.score_levels(None, 5)
        with pytest.raises(DataPreparerOperatorError, match="must be above"):
            rubric.score_levels(5, 5)

    def test_format_level(self):
        assert rubric.format_level(1.0) == "1" and rubric.format_level(2.5) == "2.5"


class TestAnchors:
    def test_nearest_level_closest_first_then_file_order(self):
        rows = [
            {"gt_score": 1.4},
            {"gt_score": 1.0},
            {"gt_score": 1.2},
            {"gt_score": 2.5},
            {"gt_score": "x"},
            {"gt_score": True},
            "junk",
        ]
        anchors, missing = rubric.select_rubric_anchors(rows, 1, 3, per_level=2)
        assert [r["gt_score"] for r in anchors[1.0]] == [1.0, 1.2]
        assert anchors[2.0] == [{"gt_score": 2.5}]  # tie 2.5 goes to the lower level
        assert anchors[3.0] == [] and missing == [3.0]
        with pytest.raises(DataPreparerOperatorError, match="per_level must be >= 1"):
            rubric.select_rubric_anchors(rows, 1, 3, per_level=0)


class TestGaps:
    def test_anchored_spec_has_no_gaps(self):
        gaps = rubric.rubric_gaps(ANCHORED, 1, 5)
        assert gaps == {
            "has_section": True,
            "levels_missing": [],
            "levels_without_example": [],
            "n_examples": 5,
        }
        assert rubric.rubric_warning("llm_judge", ANCHORED, 1, 5) is None

    def test_missing_section_levels_and_examples(self):
        assert rubric.rubric_gaps("# Spec\nno levels", 1, 5)["has_section"] is False
        spec = "### Score Levels\n- **1:** a\n  - Example: e\n- **2:** b\n- **Score 4:** d\n  - Example: e\n"
        gaps = rubric.rubric_gaps(spec, 1, 5)
        assert gaps["levels_missing"] == [3.0, 5.0] and gaps["levels_without_example"] == [2.0]
        assert gaps["n_examples"] == 2
        banded = rubric.rubric_gaps(spec, 0, 100)
        assert banded["levels_missing"] == [] and banded["n_examples"] == 2

    def test_warning_text_and_scope(self):
        assert "no `### Score Levels` section" in rubric.rubric_warning("llm_judge", "# x", 1, 5)
        w = rubric.rubric_warning(
            "llm_judge", "### Score Levels\n- **1:** a\n- **2:** b\n  - Example: e\n", 1, 5
        )
        assert "no entry for level(s) 3, 4, 5" in w and "no example under level(s) 1" in w
        assert "fewer than 2 examples" in rubric.rubric_warning(
            "llm_judge", "### Score Levels\n- Example: one\n", 0, 100
        )
        assert rubric.rubric_warning("ordinal", "# x", 1, 5) is None
        assert rubric.rubric_warning("llm_judge", "# x", None, None).startswith(
            "llm_judge spec is not"
        )
