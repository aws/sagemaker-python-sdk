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
"""This module is used to define the CheckpointLocation to remote function."""
from __future__ import absolute_import

from os import PathLike
import re

# Regex is taken from https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CheckpointConfig.html
S3_URI_REGEX_PATTERN = r"^(https|s3)://([^/]+)/?(.*)$"

_JOB_CHECKPOINT_LOCATION = "/opt/ml/checkpoints/"


def _validate_s3_uri_for_checkpoint(s3_uri: str):
    """Validate if checkpoint location is specified with a valid s3 URI."""
    return re.match(S3_URI_REGEX_PATTERN, s3_uri)


class CheckpointLocation(PathLike):
    """Class to represent the location where checkpoints are accessed in a remote function.

    To save or load checkpoints in a remote function, pass an CheckpointLocation object as a
    function parameter and use it as a os.PathLike object. This CheckpointLocation object
    represents the local directory (/opt/ml/checkpoints/) of checkpoints in side the job.
    """

    _local_path = _JOB_CHECKPOINT_LOCATION

    def __init__(self, s3_uri):
        if not _validate_s3_uri_for_checkpoint(s3_uri):
            raise ValueError("CheckpointLocation should be specified with valid s3 URI.")
        self._s3_uri = s3_uri

    def __fspath__(self):
        """Return job local path where checkpoints are stored."""
        return self._local_path
