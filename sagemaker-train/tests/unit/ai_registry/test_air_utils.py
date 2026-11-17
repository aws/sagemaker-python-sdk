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
"""Unit tests for sagemaker.ai_registry.air_utils."""

from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.air_utils import _parse_keywords
from sagemaker.ai_registry.dataset import DataSet


class TestParseKeywords:
    """The shared Hub-keyword parser, promoted here from DataSet."""

    def test_splits_on_the_first_colon_only(self):
        """An ARN value contains colons, so a greedy split would truncate it."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/h/DataSet/d/1.0.0"

        assert _parse_keywords([f"review-dataset:{arn}"]) == {"review-dataset": arn}

    def test_parses_several_keywords(self):
        assert _parse_keywords(
            ["method:sdg", "spec-sha256:ab12", "customization_technique:SFT"]
        ) == {"method": "sdg", "spec-sha256": "ab12", "customization_technique": "SFT"}

    def test_keywords_without_a_colon_are_skipped(self):
        assert _parse_keywords(["auto-labeled", "method:sdg"]) == {"method": "sdg"}

    def test_an_empty_value_is_kept(self):
        assert _parse_keywords(["method:"]) == {"method": ""}

    def test_a_later_duplicate_key_wins(self):
        assert _parse_keywords(["method:a", "method:b"]) == {"method": "b"}

    @pytest.mark.parametrize("keywords", [None, [], [None], [42], [None, "method:sdg"]])
    def test_absent_or_non_string_entries_do_not_raise(self, keywords):
        """DataSet passed these straight through before; the shared helper tolerates them."""
        result = _parse_keywords(keywords)

        assert result == ({"method": "sdg"} if keywords == [None, "method:sdg"] else {})

    def test_dataset_delegates_to_the_shared_helper(self):
        """DataSet._parse_keywords is kept as a thin delegate for its own callers."""
        keywords = ["method:sdg", "review-dataset:arn:aws:x"]

        assert DataSet._parse_keywords(keywords) == _parse_keywords(keywords)
