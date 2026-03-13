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

IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1-cpu-py310"


class TestRmtree:
    """Test cases for _rmtree function."""

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    def test_rmtree_success(self, mock_rmtree):
        """Normal case — shutil.rmtree succeeds."""
        _rmtree("/tmp/test", IMAGE)
        mock_rmtree.assert_called_once_with("/tmp/test")

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    @patch("sagemaker.train.local.local_container.subprocess.run")
    @patch("sagemaker.train.local.local_container.os.path.exists", return_value=False)
    def test_rmtree_permission_error_docker_fallback(self, mock_exists, mock_run, mock_rmtree):
        """PermissionError triggers docker fallback using the training image."""
        mock_rmtree.side_effect = PermissionError("Permission denied")

        _rmtree("/tmp/test", IMAGE)

        mock_run.assert_called_once_with(
            ["docker", "run", "--rm", "-v", "/tmp/test:/delete", IMAGE, "rm", "-rf", "/delete"],
            check=True,
            capture_output=True,
        )

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    @patch("sagemaker.train.local.local_container.subprocess.run")
    @patch("sagemaker.train.local.local_container.os.path.exists", return_value=False)
    def test_rmtree_studio_adds_network(self, mock_exists, mock_run, mock_rmtree):
        """In Studio, docker run includes --network sagemaker."""
        mock_rmtree.side_effect = PermissionError("Permission denied")

        _rmtree("/tmp/test", IMAGE, is_studio=True)

        mock_run.assert_called_once_with(
            [
                "docker", "run", "--rm",
                "--network", "sagemaker",
                "-v", "/tmp/test:/delete", IMAGE, "rm", "-rf", "/delete",
            ],
            check=True,
            capture_output=True,
        )

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    @patch("sagemaker.train.local.local_container.subprocess.run")
    @patch("sagemaker.train.local.local_container.os.path.exists", return_value=True)
    def test_rmtree_cleans_up_mount_point(self, mock_exists, mock_run, mock_rmtree):
        """After docker cleanup, remaining mount point directory is removed."""
        mock_rmtree.side_effect = [PermissionError("Permission denied"), None]

        _rmtree("/tmp/test", IMAGE)

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
        mock_run.side_effect = Exception("docker failed")

        with pytest.raises(Exception, match="docker failed"):
            _rmtree("/tmp/test", IMAGE)

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    def test_rmtree_no_image_raises(self, mock_rmtree):
        """PermissionError without image raises immediately."""
        mock_rmtree.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError):
            _rmtree("/tmp/test")
