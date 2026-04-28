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
from unittest.mock import patch, call, Mock
import pytest
import subprocess

from sagemaker.train.local.local_container import _rmtree, _LocalContainer
from sagemaker.core.shapes import DataSource, S3DataSource
from sagemaker.core.shapes import Channel

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
    def test_rmtree_permission_error_docker_chmod_fallback(self, mock_run, mock_rmtree):
        """PermissionError triggers docker chmod then retry."""
        mock_rmtree.side_effect = [PermissionError("Permission denied"), None]

        _rmtree("/tmp/test", IMAGE)

        mock_run.assert_called_once_with(
            ["docker", "run", "--rm", "-v", "/tmp/test:/delete", IMAGE, "chmod", "-R", "777", "/delete"],
            check=True,
            capture_output=True,
        )
        assert mock_rmtree.call_count == 2

    @patch("sagemaker.train.local.local_container.shutil.rmtree")
    @patch("sagemaker.train.local.local_container.subprocess.run")
    def test_rmtree_studio_adds_network(self, mock_run, mock_rmtree):
        """In Studio, docker run includes --network sagemaker."""
        mock_rmtree.side_effect = [PermissionError("Permission denied"), None]

        _rmtree("/tmp/test", IMAGE, is_studio=True)

        mock_run.assert_called_once_with(
            [
                "docker", "run", "--rm",
                "--network", "sagemaker",
                "-v", "/tmp/test:/delete", IMAGE,
                "chmod", "-R", "777", "/delete",
            ],
            check=True,
            capture_output=True,
        )

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


@pytest.fixture
def _basic_channel():
    """Create a basic channel for testing."""
    data_source = DataSource(
        s3_data_source=S3DataSource(
            s3_uri="s3://bucket/data",
            s3_data_type="S3Prefix",
            s3_data_distribution_type="FullyReplicated",
        )
    )
    return Channel(channel_name="training", data_source=data_source)


def _make_container(_basic_channel):
    """Helper to create a _LocalContainer for compose prefix tests.

    sagemaker_session is set to None because _get_compose_cmd_prefix does not
    use it, and Pydantic validation rejects Mock objects for the Session type.
    """
    return _LocalContainer(
        training_job_name="test-job",
        instance_type="local",
        instance_count=1,
        image="test-image:latest",
        container_root="/tmp/test",
        input_data_config=[_basic_channel],
        environment={},
        hyper_parameters={},
        container_entrypoint=[],
        container_arguments=[],
        sagemaker_session=None,
    )


class TestGetComposeCmdPrefix:
    """Test cases for _get_compose_cmd_prefix version detection."""

    @patch("sagemaker.train.local.local_container.subprocess.check_output")
    def test_get_compose_cmd_prefix_with_docker_compose_v2(self, mock_check_output, _basic_channel):
        """Docker Compose v2 should be accepted."""
        container = _make_container(_basic_channel)
        mock_check_output.return_value = "Docker Compose version v2.20.0"
        result = container._get_compose_cmd_prefix()
        assert result == ["docker", "compose"]

    @patch("sagemaker.train.local.local_container.subprocess.check_output")
    def test_get_compose_cmd_prefix_with_docker_compose_v5(self, mock_check_output, _basic_channel):
        """Docker Compose v5 should be accepted."""
        container = _make_container(_basic_channel)
        mock_check_output.return_value = "Docker Compose version v5.1.1"
        result = container._get_compose_cmd_prefix()
        assert result == ["docker", "compose"]

    @patch("sagemaker.train.local.local_container.subprocess.check_output")
    def test_get_compose_cmd_prefix_with_docker_compose_v3(self, mock_check_output, _basic_channel):
        """Docker Compose v3 should be accepted."""
        container = _make_container(_basic_channel)
        mock_check_output.return_value = "Docker Compose version v3.0.0"
        result = container._get_compose_cmd_prefix()
        assert result == ["docker", "compose"]

    @patch("sagemaker.train.local.local_container.shutil.which")
    @patch("sagemaker.train.local.local_container.subprocess.check_output")
    def test_get_compose_cmd_prefix_with_docker_compose_v1_falls_through(
        self, mock_check_output, mock_which, _basic_channel
    ):
        """Docker Compose v1 should not be accepted; falls through to docker-compose standalone."""
        container = _make_container(_basic_channel)
        mock_check_output.return_value = "docker-compose version 1.29.2"
        mock_which.return_value = "/usr/bin/docker-compose"
        result = container._get_compose_cmd_prefix()
        assert result == ["docker-compose"]

    @patch("sagemaker.train.local.local_container.shutil.which")
    @patch("sagemaker.train.local.local_container.subprocess.check_output")
    def test_get_compose_cmd_prefix_with_docker_compose_v1_no_standalone_raises(
        self, mock_check_output, mock_which, _basic_channel
    ):
        """Docker Compose v1 with no standalone fallback should raise ImportError."""
        container = _make_container(_basic_channel)
        mock_check_output.return_value = "docker-compose version v1.29.2"
        mock_which.return_value = None
        with pytest.raises(ImportError, match="Docker Compose is not installed"):
            container._get_compose_cmd_prefix()

    @patch("sagemaker.train.local.local_container.shutil.which")
    @patch("sagemaker.train.local.local_container.subprocess.check_output")
    def test_get_compose_cmd_prefix_not_installed_raises(
        self, mock_check_output, mock_which, _basic_channel
    ):
        """When docker compose is not installed at all, should raise ImportError."""
        container = _make_container(_basic_channel)
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_which.return_value = None
        with pytest.raises(ImportError, match="Docker Compose is not installed"):
            container._get_compose_cmd_prefix()

    @patch("sagemaker.train.local.local_container.shutil.which")
    @patch("sagemaker.train.local.local_container.subprocess.check_output")
    def test_get_compose_cmd_prefix_standalone_fallback(
        self, mock_check_output, mock_which, _basic_channel
    ):
        """When docker compose plugin fails, falls back to docker-compose standalone."""
        container = _make_container(_basic_channel)
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_which.return_value = "/usr/local/bin/docker-compose"
        result = container._get_compose_cmd_prefix()
        assert result == ["docker-compose"]
