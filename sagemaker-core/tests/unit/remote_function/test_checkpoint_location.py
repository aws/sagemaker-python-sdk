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
"""Tests for checkpoint_location module."""
from __future__ import absolute_import

import pytest
from sagemaker.core.remote_function.checkpoint_location import (
    CheckpointLocation,
    _validate_s3_uri_for_checkpoint,
    _JOB_CHECKPOINT_LOCATION,
)


class TestValidateS3Uri:
    """Test _validate_s3_uri_for_checkpoint function."""

    def test_valid_s3_uri(self):
        """Test valid s3:// URI."""
        assert _validate_s3_uri_for_checkpoint("s3://my-bucket/path/to/checkpoints")

    def test_valid_https_uri(self):
        """Test valid https:// URI."""
        assert _validate_s3_uri_for_checkpoint("https://my-bucket.s3.amazonaws.com/path")

    def test_valid_s3_uri_no_path(self):
        """Test valid s3:// URI without path."""
        assert _validate_s3_uri_for_checkpoint("s3://my-bucket")

    def test_invalid_uri_no_protocol(self):
        """Test invalid URI without protocol."""
        assert not _validate_s3_uri_for_checkpoint("my-bucket/path")

    def test_invalid_uri_wrong_protocol(self):
        """Test invalid URI with wrong protocol."""
        assert not _validate_s3_uri_for_checkpoint("http://my-bucket/path")

    def test_invalid_uri_empty(self):
        """Test invalid empty URI."""
        assert not _validate_s3_uri_for_checkpoint("")


class TestCheckpointLocation:
    """Test CheckpointLocation class."""

    def test_init_with_valid_s3_uri(self):
        """Test initialization with valid s3 URI."""
        s3_uri = "s3://my-bucket/checkpoints"
        checkpoint_loc = CheckpointLocation(s3_uri)
        assert checkpoint_loc._s3_uri == s3_uri

    def test_init_with_valid_https_uri(self):
        """Test initialization with valid https URI."""
        s3_uri = "https://my-bucket.s3.amazonaws.com/checkpoints"
        checkpoint_loc = CheckpointLocation(s3_uri)
        assert checkpoint_loc._s3_uri == s3_uri

    def test_init_with_invalid_uri_raises_error(self):
        """Test initialization with invalid URI raises ValueError."""
        with pytest.raises(ValueError, match="CheckpointLocation should be specified with valid s3 URI"):
            CheckpointLocation("invalid-uri")

    def test_fspath_returns_local_path(self):
        """Test __fspath__ returns the job local path."""
        checkpoint_loc = CheckpointLocation("s3://my-bucket/checkpoints")
        assert checkpoint_loc.__fspath__() == _JOB_CHECKPOINT_LOCATION

    def test_can_be_used_as_pathlike(self):
        """Test CheckpointLocation can be used as os.PathLike."""
        import os
        checkpoint_loc = CheckpointLocation("s3://my-bucket/checkpoints")
        path = os.fspath(checkpoint_loc)
        assert path == _JOB_CHECKPOINT_LOCATION
