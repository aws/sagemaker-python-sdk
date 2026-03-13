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
from unittest.mock import patch, call
import pytest

from sagemaker.train.local.local_container import _rmtree


class TestRmtree:
    """Test cases for _rmtree function."""

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    def test_rmtree_success(self, mock_rmtree):
        """Normal case — shutil.rmtree succeeds."""
        _rmtree("/tmp/test")
        mock_rmtree.assert_called_once_with("/tmp/test")

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    @patch("sagemaker.train.local.local_container.subprocess.run")
    @patch("sagemaker.train.local.local_container.os.path.exists", return_value=False)
    def test_rmtree_permission_error_docker_fallback(self, mock_exists, mock_run, mock_rmtree):
        """PermissionError triggers docker fallback to remove root-owned files."""
        mock_rmtree.side_effect = PermissionError("Permission denied")

        _rmtree("/tmp/test")

        mock_run.assert_called_once_with(
            ["docker", "run", "--rm", "-v", "/tmp/test:/delete", "alpine", "rm", "-rf", "/delete"],
            check=True,
            capture_output=True,
        )

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    @patch("sagemaker.train.local.local_container.subprocess.run")
    @patch("sagemaker.train.local.local_container.os.path.exists", return_value=True)
    def test_rmtree_cleans_up_mount_point(self, mock_exists, mock_run, mock_rmtree):
        """After docker cleanup, remaining mount point directory is removed."""
        mock_rmtree.side_effect = [PermissionError("Permission denied"), None]

        _rmtree("/tmp/test")

        assert mock_rmtree.call_count == 2
        mock_rmtree.assert_has_calls([
            call("/tmp/test"),
            call("/tmp/test", ignore_errors=True),
        ])

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    @patch("sagemaker.train.local.local_container.subprocess.run")
    def test_rmtree_docker_fallback_fails_raises(self, mock_run, mock_rmtree):
        """If docker fallback also fails, the exception propagates."""
        mock_rmtree.side_effect = PermissionError("Permission denied")
        mock_run.side_effect = Exception("docker not available")

        with pytest.raises(Exception, match="docker not available"):
            _rmtree("/tmp/test")
