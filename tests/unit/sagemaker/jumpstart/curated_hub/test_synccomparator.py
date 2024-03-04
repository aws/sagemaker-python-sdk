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
from __future__ import absolute_import
import unittest
from datetime import datetime
from sagemaker.jumpstart.curated_hub.accessors.fileinfo import FileInfo

from sagemaker.jumpstart.curated_hub.accessors.synccomparator import SizeAndLastUpdatedComparator


class SizeAndLastUpdateComparatorTest(unittest.TestCase):
    comparator = SizeAndLastUpdatedComparator()

    def test_identical_files_returns_false(self):
        file_one = FileInfo("my-file-one", 123456789, datetime.today())
        file_two = FileInfo("my-file-two", 123456789, datetime.today())

        assert self.comparator.determine_should_sync(file_one, file_two) is False

    def test_different_file_sizes_returns_true(self):
        file_one = FileInfo("my-file-one", 123456789, datetime.today())
        file_two = FileInfo("my-file-two", 10101010, datetime.today())

        assert self.comparator.determine_should_sync(file_one, file_two) is True

    def test_different_file_dates_returns_true(self):
        # change ordering of datetime.today() calls to trigger update
        file_two = FileInfo("my-file-two", 123456789, datetime.today())
        file_one = FileInfo("my-file-one", 123456789, datetime.today())

        assert self.comparator.determine_should_sync(file_one, file_two) is True
