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
"""This module provides a class that perfrms functionalities similar to ``S3:Copy``."""
from __future__ import absolute_import
from dataclasses import dataclass
from typing import Generator, List

from botocore.compat import six
from sagemaker.jumpstart.curated_hub.accessors.objectlocation import S3ObjectLocation
from sagemaker.jumpstart.curated_hub.accessors.synccomparator import BaseComparator

from sagemaker.jumpstart.curated_hub.accessors.fileinfo import FileInfo

advance_iterator = six.advance_iterator


@dataclass
class SyncTaskInfo:
    """Sync Task Info class"""

    files: List[FileInfo]
    destination: S3ObjectLocation

    def __init__(
        self, files_to_copy: Generator[FileInfo, FileInfo, FileInfo], destination: S3ObjectLocation
    ):
        """Contains information required to sync data.

        Returns:
            :var: files (List[FileInfo]): Files that shoudl be synced.
            :var: destination (S3ObjectLocation): Location to which to sync the files.
        """
        self.files = list(files_to_copy)
        self.destination = destination


class SyncTaskHandler:
    """Generates a ``SyncTaskInfo`` which contains information required to sync data."""

    def __init__(
        self,
        src_files: List[FileInfo],
        dest_files: List[FileInfo],
        destination: S3ObjectLocation,
        comparator: BaseComparator,
    ):
        """Instantiates a ``SyncTaskGenerator`` class.

        Args:
            src_files (List[FileInfo]): List of files to sync to destination bucket
            dest_files (List[FileInfo]): List of files already in destination bucket
            destination (S3ObjectLocation): S3 destination for copied data

        Returns:
            ``SyncTaskInfo`` class containing:
                :var: files (List[FileInfo]): Files that shoudl be synced.
                :var: destination (S3ObjectLocation): Location to which to sync the files.
        """
        self.comparator = comparator
        self.src_files: List[FileInfo] = sorted(src_files, key=lambda x: x.location.key)
        self.dest_files: List[FileInfo] = sorted(dest_files, key=lambda x: x.location.key)
        self.destination = destination

    def create(self) -> SyncTaskInfo:
        """Creates a ``SyncTaskInfo`` object, which contains `files` to copy and the `destination`

        Based on the `s3:sync` algorithm.
        """
        files_to_copy = self._determine_files_to_copy()
        return SyncTaskInfo(files_to_copy, self.destination)

    def _determine_files_to_copy(self) -> Generator[FileInfo, FileInfo, FileInfo]:
        """This function performs the actual comparisons. Returns a list of FileInfo to copy.

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
        iterator = iter(self.dest_files)
        # Begin by advancing the iterator to the first file
        try:
            dest_file: FileInfo = advance_iterator(iterator)
        except StopIteration:
            dest_done = True

        for src_file in self.src_files:
            # End of dest, yield remaining src_files
            if dest_done:
                yield src_file
                continue

            # We've identified two files that have the same name, further compare
            if self._is_same_file_name(src_file.location.key, dest_file.location.key):
                should_sync = self.comparator.determine_should_sync(src_file, dest_file)

                if should_sync:
                    yield src_file

                # Increment dest_files and src_files
                try:
                    dest_file: FileInfo = advance_iterator(iterator)
                except StopIteration:
                    dest_done = True
                continue

            # Past the src file alphabetically in dest file list. Take the src file and continue
            if self._is_alphabetically_larger_file_name(
                src_file.location.key, dest_file.location.key
            ):
                yield src_file
                continue

    def _is_same_file_name(self, src_filename: str, dest_filename: str) -> bool:
        """Compares two file names and determiens if they are the same.

        Destination files might add a prefix.
        """
        return dest_filename.endswith(src_filename)

    def _is_alphabetically_larger_file_name(self, src_filename: str, dest_filename: str) -> bool:
        """Determines if one filename is alphabetically earlier than another."""
        return src_filename > dest_filename
