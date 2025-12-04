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

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import subprocess

from sagemaker.core.modules.local_core.local_container import (
    _LocalContainer,
    DOCKER_COMPOSE_FILENAME,
    DOCKER_COMPOSE_HTTP_TIMEOUT_ENV,
    DOCKER_COMPOSE_HTTP_TIMEOUT,
)
from sagemaker.core.modules import Session
from sagemaker.core.modules.configs import Channel
from sagemaker.core.shapes import DataSource, S3DataSource, FileSystemDataSource
from sagemaker.core.utils.utils import Unassigned


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session"""
    session = Mock(spec=Session)
    session.boto_region_name = "us-west-2"
    session.boto_session = Mock()
    session.s3_resource = Mock()
    session.s3_resource.meta.client._endpoint.host = "https://s3.us-west-2.amazonaws.com"
    return session


@pytest.fixture
def basic_channel():
    """Create a basic channel for testing"""
    data_source = DataSource(
        s3_data_source=S3DataSource(
            s3_uri="s3://bucket/data",
            s3_data_type="S3Prefix",
            s3_data_distribution_type="FullyReplicated",
        )
    )
    return Channel(channel_name="training", data_source=data_source)


class TestLocalContainer:
    """Test cases for _LocalContainer class"""

    def test_init_with_s3_input(self, mock_session, basic_channel):
        """Test initialization with S3 input"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        assert container.training_job_name == "test-job"
        assert container.instance_type == "local"
        assert container.instance_count == 1
        assert container.input_from_s3 is True
        assert len(container.hosts) == 1
        assert container.hosts[0] == "algo-1"

    def test_init_with_local_input(self):
        """Test initialization with local file system input"""
        data_source = DataSource(
            file_system_data_source=FileSystemDataSource(
                file_system_id="fs-123",
                file_system_type="EFS",
                file_system_access_mode="ro",
                directory_path="/mnt/data",
            )
        )
        channel = Channel(channel_name="training", data_source=data_source)

        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
        )

        assert container.input_from_s3 is False

    def test_init_with_multiple_instances(self, mock_session, basic_channel):
        """Test initialization with multiple instances"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=3,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        assert len(container.hosts) == 3
        assert container.hosts == ["algo-1", "algo-2", "algo-3"]

    def test_init_with_invalid_distribution_type(self, mock_session):
        """Test initialization with invalid data distribution type"""
        data_source = DataSource(
            s3_data_source=S3DataSource(
                s3_uri="s3://bucket/data",
                s3_data_type="S3Prefix",
                s3_data_distribution_type="ShardedByS3Key",
            )
        )
        channel = Channel(channel_name="training", data_source=data_source)

        with pytest.raises(RuntimeError, match="Invalid Data Distribution"):
            _LocalContainer(
                training_job_name="test-job",
                instance_type="local",
                instance_count=1,
                image="test-image:latest",
                container_root="/tmp/test",
                input_data_config=[channel],
                environment={},
                hyper_parameters={},
                container_entrypoint=[],
                container_arguments=[],
                sagemaker_session=mock_session,
            )

    def test_init_without_data_source(self):
        """Test initialization without proper data source"""
        channel = Channel(channel_name="training", data_source=DataSource())

        with pytest.raises(ValueError, match="Need channel.data_source"):
            _LocalContainer(
                training_job_name="test-job",
                instance_type="local",
                instance_count=1,
                image="test-image:latest",
                container_root="/tmp/test",
                input_data_config=[channel],
                environment={},
                hyper_parameters={},
                container_entrypoint=[],
                container_arguments=[],
            )

    @patch("sagemaker.core.modules.local_core.local_container.os.makedirs")
    @patch("sagemaker.core.modules.local_core.local_container.subprocess.Popen")
    @patch("sagemaker.core.modules.local_core.local_container._stream_output")
    @patch("sagemaker.core.modules.local_core.local_container.shutil.rmtree")
    def test_train_success(
        self, mock_rmtree, mock_stream, mock_popen, mock_makedirs, mock_session, basic_channel
    ):
        """Test successful training execution"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        mock_process = Mock()
        mock_popen.return_value = mock_process

        with patch.object(_LocalContainer, "_prepare_training_volumes", return_value=[]):
            with patch.object(_LocalContainer, "_create_config_file_directories"):
                with patch.object(_LocalContainer, "_write_config_files"):
                    with patch.object(_LocalContainer, "_ecr_login_if_needed", return_value=False):
                        with patch.object(
                            _LocalContainer,
                            "_generate_compose_file",
                            return_value={"services": {"algo-1": {"volumes": []}}},
                        ):
                            with patch.object(
                                _LocalContainer,
                                "_generate_compose_command",
                                return_value=["docker-compose", "up"],
                            ):
                                with patch.object(
                                    _LocalContainer,
                                    "retrieve_artifacts",
                                    return_value="/tmp/model.tar.gz",
                                ):
                                    result = container.train(wait=True)

        assert result == "/tmp/model.tar.gz"

    @patch("sagemaker.core.modules.local_core.local_container.check_for_studio")
    @patch("sagemaker.core.modules.local_core.local_container.os.path.exists")
    @patch("sagemaker.core.modules.local_core.local_container.os.listdir")
    @patch("sagemaker.core.modules.local_core.local_container.recursive_copy")
    @patch("sagemaker.core.modules.local_core.local_container.create_tar_file")
    @patch("sagemaker.core.modules.local_core.local_container.os.makedirs")
    def test_retrieve_artifacts(
        self,
        mock_makedirs,
        mock_tar,
        mock_copy,
        mock_listdir,
        mock_exists,
        mock_check_studio,
        mock_session,
        basic_channel,
    ):
        """Test retrieve_artifacts method"""
        mock_check_studio.return_value = False
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        mock_exists.return_value = True
        mock_listdir.return_value = ["file1.txt", "file2.txt"]

        compose_data = {
            "services": {
                "algo-1": {
                    "volumes": [
                        "/tmp/test/algo-1/model:/opt/ml/model",
                        "/tmp/test/algo-1/output:/opt/ml/output",
                    ]
                }
            }
        }

        result = container.retrieve_artifacts(compose_data)

        assert "model.tar.gz" in result

    def test_create_config_file_directories(self, mock_session, basic_channel):
        """Test _create_config_file_directories method"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch(
            "sagemaker.core.modules.local_core.local_container.os.makedirs"
        ) as mock_makedirs:
            container._create_config_file_directories("algo-1")

            assert mock_makedirs.call_count >= 4

    @patch("sagemaker.core.modules.local_core.local_container._write_json_file")
    def test_write_config_files(self, mock_write_json, mock_session, basic_channel):
        """Test _write_config_files method"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={"epochs": "10"},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        container._write_config_files(
            host="algo-1", input_data_config=[basic_channel], hyper_parameters={"epochs": "10"}
        )

        assert mock_write_json.call_count == 3

    @patch("builtins.open", new_callable=mock_open)
    def test_generate_compose_file(self, mock_file, mock_session, basic_channel):
        """Test _generate_compose_file method"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        result = container._generate_compose_file(environment={"KEY": "VALUE"}, volumes=[])

        assert "services" in result
        assert "algo-1" in result["services"]

    def test_create_docker_host(self, mock_session, basic_channel):
        """Test _create_docker_host method"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={"KEY": "VALUE"},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch(
            "sagemaker.core.modules.local_core.local_container._aws_credentials",
            return_value=["AWS_KEY=value"],
        ):
            result = container._create_docker_host(
                host="algo-1", environment={"KEY": "VALUE"}, volumes=[]
            )

        assert "image" in result
        assert result["image"] == "test-image:latest"
        assert "environment" in result

    def test_create_docker_host_with_gpu(self, mock_session, basic_channel):
        """Test _create_docker_host with GPU instance"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local_gpu",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch(
            "sagemaker.core.modules.local_core.local_container._aws_credentials", return_value=[]
        ):
            result = container._create_docker_host(host="algo-1", environment={}, volumes=[])

        assert "deploy" in result
        assert "resources" in result["deploy"]

    def test_generate_compose_command(self, mock_session, basic_channel):
        """Test _generate_compose_command method"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch.object(
            _LocalContainer, "_get_compose_cmd_prefix", return_value=["docker", "compose"]
        ):
            command = container._generate_compose_command(wait=True)

        assert "docker" in command
        assert "compose" in command
        assert "up" in command
        assert "--abort-on-container-exit" in command

    @patch("sagemaker.core.modules.local_core.local_container._check_output")
    @patch("sagemaker.core.modules.local_core.local_container.subprocess.Popen")
    def test_ecr_login_if_needed_with_ecr_image(
        self, mock_popen, mock_check_output, mock_session, basic_channel
    ):
        """Test _ecr_login_if_needed with ECR image"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        mock_check_output.return_value = ""  # Image not found locally
        mock_process = Mock()
        mock_popen.return_value = mock_process

        ecr_client = Mock()
        ecr_client.get_authorization_token.return_value = {
            "authorizationData": [
                {
                    "authorizationToken": "QVdTOnRva2Vu",  # base64 encoded "AWS:token"
                    "proxyEndpoint": "https://123456789012.dkr.ecr.us-west-2.amazonaws.com",
                }
            ]
        }
        mock_session.boto_session.client.return_value = ecr_client

        result = container._ecr_login_if_needed()

        assert result is True

    @patch("sagemaker.core.modules.local_core.local_container._check_output")
    def test_ecr_login_if_needed_with_local_image(
        self, mock_check_output, mock_session, basic_channel
    ):
        """Test _ecr_login_if_needed with locally available image"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        mock_check_output.return_value = "image-id-123"  # Image found locally

        result = container._ecr_login_if_needed()

        assert result is False

    def test_ecr_login_if_needed_with_non_ecr_image(self, mock_session, basic_channel):
        """Test _ecr_login_if_needed with non-ECR image"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="docker.io/my-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        result = container._ecr_login_if_needed()

        assert result is False

    @patch("sagemaker.core.modules.local_core.local_container.download_folder")
    @patch("sagemaker.core.modules.local_core.local_container.TemporaryDirectory")
    def test_get_data_source_local_path_s3(self, mock_temp_dir, mock_download, mock_session):
        """Test _get_data_source_local_path with S3 data source"""
        data_source = DataSource(
            s3_data_source=S3DataSource(
                s3_uri="s3://bucket/data",
                s3_data_type="S3Prefix",
                s3_data_distribution_type="FullyReplicated",
            )
        )
        channel = Channel(channel_name="training", data_source=data_source)

        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        mock_temp_dir.return_value.name = "/tmp/temp123"

        result = container._get_data_source_local_path(data_source)

        assert result == "/tmp/temp123"
        mock_download.assert_called_once()

    def test_get_data_source_local_path_local(self, mock_session):
        """Test _get_data_source_local_path with local file system"""
        data_source = DataSource(
            file_system_data_source=FileSystemDataSource(
                file_system_id="fs-123",
                file_system_type="EFS",
                file_system_access_mode="ro",
                directory_path="/mnt/data",
            )
        )
        channel = Channel(channel_name="training", data_source=data_source)

        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
        )

        result = container._get_data_source_local_path(data_source)

        assert "/mnt/data" in result

    @patch("sagemaker.core.modules.local_core.local_container.subprocess.check_output")
    def test_get_compose_cmd_prefix_docker_compose_v2(
        self, mock_check_output, mock_session, basic_channel
    ):
        """Test _get_compose_cmd_prefix with Docker Compose v2"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        mock_check_output.return_value = "Docker Compose version v2.0.0"

        result = container._get_compose_cmd_prefix()

        assert result == ["docker", "compose"]

    @patch("sagemaker.core.modules.local_core.local_container.subprocess.check_output")
    @patch("sagemaker.core.modules.local_core.local_container.shutil.which")
    def test_get_compose_cmd_prefix_docker_compose_standalone(
        self, mock_which, mock_check_output, mock_session, basic_channel
    ):
        """Test _get_compose_cmd_prefix with standalone docker-compose"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_which.return_value = "/usr/local/bin/docker-compose"

        result = container._get_compose_cmd_prefix()

        assert result == ["docker-compose"]

    @patch("sagemaker.core.modules.local_core.local_container.subprocess.check_output")
    @patch("sagemaker.core.modules.local_core.local_container.shutil.which")
    def test_get_compose_cmd_prefix_not_found(
        self, mock_which, mock_check_output, mock_session, basic_channel
    ):
        """Test _get_compose_cmd_prefix when Docker Compose is not found"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_which.return_value = None

        with pytest.raises(ImportError, match="Docker Compose is not installed"):
            container._get_compose_cmd_prefix()

    def test_init_with_container_entrypoint(self, mock_session, basic_channel):
        """Test initialization with container entrypoint"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            sagemaker_session=mock_session,
            container_entrypoint=["/bin/bash", "-c"],
            container_arguments=["echo hello"],
        )

        assert container.container_entrypoint == ["/bin/bash", "-c"]
        assert container.container_arguments == ["echo hello"]

    def test_create_docker_host_with_entrypoint_and_arguments(self, mock_session, basic_channel):
        """Test _create_docker_host with entrypoint and arguments"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            sagemaker_session=mock_session,
            container_entrypoint=["/bin/bash"],
            container_arguments=["-c", "echo test"],
        )

        with patch(
            "sagemaker.core.modules.local_core.local_container._aws_credentials", return_value=[]
        ):
            result = container._create_docker_host(host="algo-1", environment={}, volumes=[])

        assert "entrypoint" in result
        assert result["entrypoint"] == ["/bin/bash", "-c", "echo test"]

    @patch("sagemaker.core.modules.local_core.local_container.check_for_studio")
    def test_init_with_studio_mode(self, mock_check_studio, basic_channel):
        """Test initialization in SageMaker Studio mode"""
        mock_check_studio.return_value = True

        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
        )

        assert container.is_studio is True

    @patch("sagemaker.core.modules.local_core.local_container.check_for_studio")
    def test_create_docker_host_studio_mode(self, mock_check_studio, mock_session, basic_channel):
        """Test _create_docker_host in Studio mode"""
        mock_check_studio.return_value = True
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch(
            "sagemaker.core.modules.local_core.local_container._aws_credentials", return_value=[]
        ):
            result = container._create_docker_host(host="algo-1", environment={}, volumes=[])

        assert result["network_mode"] == "sagemaker"
        assert "SM_STUDIO_LOCAL_MODE=True" in result["environment"]

    @patch("sagemaker.core.modules.local_core.local_container.os.makedirs")
    def test_prepare_training_volumes_with_metadata_dir(
        self, mock_makedirs, mock_session, basic_channel
    ):
        """Test _prepare_training_volumes with metadata directory"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch(
            "sagemaker.core.modules.local_core.local_container.os.path.isdir", return_value=True
        ):
            with patch.object(
                _LocalContainer, "_get_data_source_local_path", return_value="/tmp/data"
            ):
                volumes = container._prepare_training_volumes(
                    data_dir="/tmp/test/input/data",
                    input_data_config=[basic_channel],
                    hyper_parameters={},
                )

        # Should include metadata directory volume
        metadata_volumes = [v for v in volumes if "/opt/ml/metadata" in v]
        assert len(metadata_volumes) > 0

    def test_prepare_training_volumes_with_local_training_script(self, mock_session, basic_channel):
        """Test _prepare_training_volumes with local training script"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={"sagemaker_submit_directory": "file:///tmp/code"},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch("sagemaker.core.modules.local_core.local_container.os.makedirs"):
            with patch(
                "sagemaker.core.modules.local_core.local_container.os.path.isdir",
                return_value=False,
            ):
                with patch.object(
                    _LocalContainer, "_get_data_source_local_path", return_value="/tmp/data"
                ):
                    volumes = container._prepare_training_volumes(
                        data_dir="/tmp/test/input/data",
                        input_data_config=[basic_channel],
                        hyper_parameters={"sagemaker_submit_directory": "file:///tmp/code"},
                    )

        # Should include code and shared directory volumes
        code_volumes = [v for v in volumes if "/opt/ml/code" in v]
        shared_volumes = [v for v in volumes if "/opt/ml/shared" in v]
        assert len(code_volumes) > 0
        assert len(shared_volumes) > 0

    def test_retrieve_artifacts_with_windows_paths(self, mock_session, basic_channel):
        """Test retrieve_artifacts with Windows-style paths"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        compose_data = {
            "services": {
                "algo-1": {
                    "volumes": [
                        "C:/tmp/test/algo-1/model:/opt/ml/model",
                        "C:/tmp/test/algo-1/output:/opt/ml/output",
                    ]
                }
            }
        }

        with patch("sagemaker.core.modules.local_core.local_container.os.makedirs"):
            with patch(
                "sagemaker.core.modules.local_core.local_container.os.listdir",
                return_value=["file.txt"],
            ):
                with patch("sagemaker.core.modules.local_core.local_container.recursive_copy"):
                    with patch("sagemaker.core.modules.local_core.local_container.create_tar_file"):
                        result = container.retrieve_artifacts(compose_data)

        assert "model.tar.gz" in result

    def test_retrieve_artifacts_with_z_suffix_volumes(self, mock_session, basic_channel):
        """Test retrieve_artifacts with :z suffix in volumes"""
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        compose_data = {
            "services": {
                "algo-1": {
                    "volumes": [
                        "/tmp/test/algo-1/model:/opt/ml/model:z",
                        "/tmp/test/algo-1/output:/opt/ml/output:z",
                    ]
                }
            }
        }

        with patch("sagemaker.core.modules.local_core.local_container.os.makedirs"):
            with patch(
                "sagemaker.core.modules.local_core.local_container.os.listdir",
                return_value=["file.txt"],
            ):
                with patch("sagemaker.core.modules.local_core.local_container.recursive_copy"):
                    with patch("sagemaker.core.modules.local_core.local_container.create_tar_file"):
                        result = container.retrieve_artifacts(compose_data)

        assert "model.tar.gz" in result

    @patch("sagemaker.core.modules.local_core.local_container.check_for_studio")
    def test_generate_compose_file_sets_timeout_env(
        self, mock_check_studio, mock_session, basic_channel
    ):
        """Test that _generate_compose_file sets Docker Compose timeout"""
        mock_check_studio.return_value = False
        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[basic_channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch.dict("os.environ", {}, clear=True):
            with patch("builtins.open", mock_open()):
                import yaml

                with patch.object(yaml, "dump"):
                    container._generate_compose_file(environment={}, volumes=[])

        assert os.environ.get(DOCKER_COMPOSE_HTTP_TIMEOUT_ENV) == DOCKER_COMPOSE_HTTP_TIMEOUT

    def test_write_config_files_with_content_type(self, mock_session):
        """Test _write_config_files with content type in channel"""
        data_source = DataSource(
            s3_data_source=S3DataSource(
                s3_uri="s3://bucket/data",
                s3_data_type="S3Prefix",
                s3_data_distribution_type="FullyReplicated",
            )
        )
        channel = Channel(
            channel_name="training", data_source=data_source, content_type="application/json"
        )

        container = _LocalContainer(
            training_job_name="test-job",
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            container_root="/tmp/test",
            input_data_config=[channel],
            environment={},
            hyper_parameters={},
            container_entrypoint=[],
            container_arguments=[],
            sagemaker_session=mock_session,
        )

        with patch(
            "sagemaker.core.modules.local_core.local_container._write_json_file"
        ) as mock_write:
            container._write_config_files(
                host="algo-1", input_data_config=[channel], hyper_parameters={}
            )

        # Verify inputdataconfig.json was written with content type
        calls = mock_write.call_args_list
        input_config_call = [c for c in calls if "inputdataconfig.json" in str(c)]
        assert len(input_config_call) > 0
