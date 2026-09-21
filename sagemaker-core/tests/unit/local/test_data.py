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

"""Unit tests for sagemaker.core.local.data batch strategies"""

from __future__ import absolute_import

from sagemaker.core.local.data import MultiRecordStrategy


class _ListSplitter:
    """A minimal Splitter stand-in that yields pre-set records."""

    def __init__(self, records):
        self._records = records

    def split(self, file):  # pylint: disable=unused-argument
        for record in self._records:
            yield record


def test_multi_record_strategy_pad_bytes():
    """Binary records must not raise TypeError from str + bytes concatenation."""
    records = [b"abc", b"def", b"ghi"]
    strategy = MultiRecordStrategy(_ListSplitter(records))

    result = list(strategy.pad("dummy", size=6))

    assert result == [b"abcdefghi"]
    assert all(isinstance(chunk, bytes) for chunk in result)


def test_multi_record_strategy_pad_str():
    """Text records continue to be grouped as strings."""
    records = ["abc", "def", "ghi"]
    strategy = MultiRecordStrategy(_ListSplitter(records))

    result = list(strategy.pad("dummy", size=6))

    assert result == ["abcdefghi"]
    assert all(isinstance(chunk, str) for chunk in result)


def test_multi_record_strategy_pad_empty():
    """An empty split yields nothing instead of raising."""
    strategy = MultiRecordStrategy(_ListSplitter([]))

    result = list(strategy.pad("dummy", size=6))

    assert result == []
