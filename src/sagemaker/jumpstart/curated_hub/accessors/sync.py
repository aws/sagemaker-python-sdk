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
"""This module provides a class to help copy HubContent dependencies."""
from __future__ import absolute_import
from typing import Generator, List

from botocore.compat import six
from sagemaker.jumpstart.curated_hub.accessors.synccomparator import SizeAndLastUpdatedComparator

from sagemaker.jumpstart.curated_hub.accessors.fileinfo import FileInfo

advance_iterator = six.advance_iterator


class FileSync:
    """Something."""

    def __init__(self, src_files: List[FileInfo], dest_files: List[FileInfo], dest_bucket: str):
        """Instantiates a ``FileSync`` class.
        Sorts src and dest files by name for easier
        comparisons.

        Args:
            src_files (List[FileInfo]): List of files to sync with destination
            dest_files (List[FileInfo]): List of files already in destination bucket
            dest_bucket (str): Destination bucket name for copied data
        """
        self.comparator = SizeAndLastUpdatedComparator()
        self.src_files: List[FileInfo] = sorted(src_files, lambda x: x.name)
        self.dest_files: List[FileInfo] = sorted(dest_files, lambda x: x.name)
        self.dest_bucket = dest_bucket

    def call(self) -> Generator[FileInfo, FileInfo, FileInfo]:
        """This function performs the actual comparisons. Returns a list of FileInfo
        to copy.

        Algorithm:
            Loop over sorted files in the src directory. If at the end of dest directory,
            add the src file to be copied and continue. Else, take the first file in the sorted
            dest directory. If there is no dest file, signal we're at the end of dest and continue.
            If there is a dest file, compare file names. If the file names are equivalent,
            use the comparator to see if the dest file should be updated. If the file
            names are not equivalent, increment the dest pointer.
        """
        # :var dest_done: True if there are no files from the dest left.
        dest_done = False
        for src_file in self.src_files:
            if dest_done:
                yield src_file
                continue

            while not dest_done:
                try:
                    dest_file: FileInfo = advance_iterator(self.dest_files)
                except StopIteration:
                    dest_done = True
                    break

                if src_file.name == dest_file.name:
                    should_sync = self.comparator.determine_should_sync(src_file, dest_file)

                    if should_sync:
                        yield src_file
                        break
