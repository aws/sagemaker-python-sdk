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

import boto3
import botocore
import tqdm

from sagemaker.jumpstart.constants import JUMPSTART_LOGGER
from sagemaker.jumpstart.curated_hub.types import FileInfo
from sagemaker.jumpstart.curated_hub.sync.request import HubSyncRequest

s3transfer = boto3.s3.transfer


# pylint: disable=R1705,R1710
def human_readable_size(value: int) -> str:
    """Convert a size in bytes into a human readable format.

    For example::

        >>> human_readable_size(1)
        '1 Byte'
        >>> human_readable_size(10)
        '10 Bytes'
        >>> human_readable_size(1024)
        '1.0 KiB'
        >>> human_readable_size(1024 * 1024)
        '1.0 MiB'

    :param value: The size in bytes.
    :return: The size in a human readable format based on base-2 units.

    """
    base = 1024
    bytes_int = float(value)

    if bytes_int == 1:
        return "1 Byte"
    elif bytes_int < base:
        return "%d Bytes" % bytes_int

    for i, suffix in enumerate(("KiB", "MiB", "GiB", "TiB", "PiB", "EiB")):
        unit = base ** (i + 2)
        if round((bytes_int / unit) * base) < base:
            return "%.1f %s" % ((base * bytes_int / unit), suffix)


class MultiPartCopyHandler(object):
    """Multi Part Copy Handler class."""

    WORKERS = 20
    MULTIPART_CONFIG = 8 * (1024**2)

    def __init__(
        self,
        region: str,
        sync_request: HubSyncRequest,
    ):
        """Something."""
        self.region = region
        self.files = sync_request.files
        self.dest_location = sync_request.dest_location

        config = botocore.config.Config(max_pool_connections=self.WORKERS)
        self.s3_client = boto3.client("s3", region_name=self.region, config=config)
        transfer_config = s3transfer.TransferConfig(
            multipart_threshold=self.MULTIPART_CONFIG,
            multipart_chunksize=self.MULTIPART_CONFIG,
            max_bandwidth=True,
            use_threads=True,
            max_concurrency=self.WORKERS,
        )
        self.transfer_manager = s3transfer.create_transfer_manager(
            client=self.s3_client, config=transfer_config
        )

    def _copy_file(self, file: FileInfo, progress_cb):
        """Something."""
        copy_source = {"Bucket": file.location.bucket, "Key": file.location.key}
        result = self.transfer_manager.copy(
            bucket=self.dest_location.bucket,
            key=f"{self.dest_location.key}/{file.location.key}",
            copy_source=copy_source,
            subscribers=[
                s3transfer.ProgressCallbackInvoker(progress_cb),
            ],
        )
        # Attempt to access result to throw error if exists. Silently calls if successful.
        result.result()

    def execute(self):
        """Something."""
        total_size = sum([file.size for file in self.files])
        JUMPSTART_LOGGER.warning(
            "Copying %s files (%s) into %s/%s",
            len(self.files),
            human_readable_size(total_size),
            self.dest_location.bucket,
            self.dest_location.key,
        )

        progress = tqdm.tqdm(
            desc="JumpStart Sync",
            total=total_size,
            unit="B",
            unit_scale=1,
            position=0,
            bar_format="{desc:<10}{percentage:3.0f}%|{bar:10}{r_bar}",
        )

        for file in self.files:
            self._copy_file(file, progress.update)

        # Call `shutdown` to wait for copy results
        self.transfer_manager.shutdown()
        progress.close()
