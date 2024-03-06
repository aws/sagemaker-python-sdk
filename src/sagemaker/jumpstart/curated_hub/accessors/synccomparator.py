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
"""This module provides comparators for syncing s3 files."""
from __future__ import absolute_import
from datetime import timedelta
from sagemaker.jumpstart.constants import JUMPSTART_LOGGER

from sagemaker.jumpstart.curated_hub.accessors.fileinfo import FileInfo


class BaseComparator:
    """BaseComparator object to be extended."""

    def determine_should_sync(self, src_file: FileInfo, dest_file: FileInfo) -> bool:
        """Custom comparator to determine if src file and dest file are in sync."""
        raise NotImplementedError


class SizeAndLastUpdatedComparator(BaseComparator):
    """Comparator that uses file size and last modified time.

    Uses file size (bytes) and last_modified_time (timestamp) to determine sync.
    """

    def determine_should_sync(self, src_file: FileInfo, dest_file: FileInfo) -> bool:
        """Determines if src file should be moved to dest folder."""
        same_size = self.compare_size(src_file, dest_file)
        is_newer_dest_file = self.compare_file_updates(src_file, dest_file)
        should_sync = (not same_size) or (not is_newer_dest_file)
        if should_sync:
            JUMPSTART_LOGGER.warning(
                "syncing: %s -> %s, size: %s -> %s, modified time: %s -> %s",
                src_file.location.key,
                src_file.location.key,
                src_file.size,
                dest_file.size,
                src_file.last_updated,
                dest_file.last_updated,
            )
        return should_sync

    def compare_size(self, src_file: FileInfo, dest_file: FileInfo):
        """Compares sizes of src and dest files.

        :returns: True if the sizes are the same.
            False otherwise.
        """
        return src_file.size == dest_file.size

    def compare_file_updates(self, src_file: FileInfo, dest_file: FileInfo):
        """Compares time delta between src and dest files.

        :returns: True if the file does not need updating based on time of
            last modification and type of operation.
            False if the file does need updating based on the time of
            last modification and type of operation.
        """
        src_time = src_file.last_updated
        dest_time = dest_file.last_updated
        delta = dest_time - src_time
        # pylint: disable=R1703,R1705
        if timedelta.total_seconds(delta) >= 0:
            return True
        else:
            # Destination is older than source, so
            # we have a more recently updated file
            # at the source location.
            return False
