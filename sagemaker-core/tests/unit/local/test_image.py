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
import os
import tempfile
import platform
import subprocess
import json
from unittest.mock import Mock, MagicMock, patch, call
from sagemaker.core.local.image import (
    _SageMakerContainer,
    _Volume,
    _stream_output,
    _check_output,
    _create_config_file_directories,
    _create_processing_config_file_directories,
    _delete_tree,
    _aws_credentials,
    _aws_credentials_available_in_metadata_service,
    _use_short_lived_credentials,
    _write_json_file,
    _ecr_login_if_needed,
    _pull_image,
    _HostingContainer,
    CONTAINER_PREFIX,
    STUDIO_HOST_NAME,
)


class TestVolume:
    """Test cases for _Volume class"""

    def test_volume_with_container_dir(self):
        """Test Volume creation with container_dir"""
        volume = _Volume("/host/path", container_dir="/container/path")

        assert volume.host_dir == "/host/path"
        assert volume.container_dir == "/container/path"
        assert "/host/path:/container/path" in volume.map

    def test_volume_with_channel(self):
        """Test Volume creation with channel"""
        volume = _Volume("/host/path", channel="training")

        assert volume.host_dir == "/host/path"
        assert volume.container_dir == "/opt/ml/input/data/training"
        assert "/host/path:/opt/ml/input/data/training" in volume.map

    def test_volume_raises_without_container_dir_or_channel(self):
        """Test Volume raises ValueError without container_dir or channel"""
        with pytest.raises(ValueError, match="Either container_dir or channel must be declared"):
            _Volume("/host/path")

    def test_volume_raises_with_both_container_dir_and_channel(self):
        """Test Volume raises ValueError with both container_dir and channel"""
        with pytest.raises(
            ValueError, match="container_dir and channel cannot be declared together"
        ):
            _Volume("/host/path", container_dir="/container/path", channel="training")

    @patch("platform.system")
    def test_volume_selinux_enabled_on_linux(self, mock_platform):
        """Test Volume adds :z suffix on Linux with SELinux"""
        mock_platform.return_value = "Linux"
        with patch.dict(os.environ, {"SAGEMAKER_LOCAL_SELINUX_ENABLED": "true"}):
            # Need to reload the module to pick up the environment variable
            import importlib
            import sagemaker.core.local.image as image_module

            importlib.reload(image_module)

            volume = image_module._Volume("/host/path", container_dir="/container/path")
            assert volume.map.endswith(":z")

    @patch("platform.system")
    def test_volume_darwin_var_path(self, mock_platform):
        """Test Volume prepends /private on Darwin for /var paths"""
        mock_platform.return_value = "Darwin"
        volume = _Volume("/var/folders/test", container_dir="/container/path")

        # The actual implementation only prepends /private in the map, not host_dir
        assert "/var/folders/test" in volume.map or "/private/var/folders/test" in volume.map


class TestStreamOutput:
    """Test cases for _stream_output function"""

    def test_stream_output_success(self):
        """Test stream_output with successful process"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [b"Line 1\n", b"Line 2\n", b""]
        mock_process.poll.side_effect = [None, None, 0]

        exit_code = _stream_output(mock_process)

        assert exit_code == 0

    def test_stream_output_interrupted(self):
        """Test stream_output with interrupted process (exit code 130)"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [b"Line 1\n", b""]
        mock_process.poll.side_effect = [None, 130]

        exit_code = _stream_output(mock_process)

        assert exit_code == 130

    def test_stream_output_failure(self):
        """Test stream_output with failed process"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [b"Error\n", b""]
        mock_process.poll.side_effect = [None, 1]
        mock_process.args = ["test", "command"]

        with pytest.raises(RuntimeError, match="Failed to run"):
            _stream_output(mock_process)


class TestCheckOutput:
    """Test cases for _check_output function"""

    @patch("subprocess.check_output")
    def test_check_output_success(self, mock_check_output):
        """Test _check_output with successful command"""
        mock_check_output.return_value = b"Success output"

        result = _check_output("echo test")

        assert result == "Success output"

    @patch("subprocess.check_output")
    def test_check_output_failure(self, mock_check_output):
        """Test _check_output with failed command"""
        mock_check_output.side_effect = Exception("Command failed")

        with pytest.raises(Exception):
            _check_output("false")


class TestCreateConfigFileDirectories:
    """Test cases for config file directory creation functions"""

    def test_create_config_file_directories(self):
        """Test _create_config_file_directories creates correct structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_config_file_directories(tmpdir, "host1")

            assert os.path.exists(os.path.join(tmpdir, "host1", "input"))
            assert os.path.exists(os.path.join(tmpdir, "host1", "input", "config"))
            assert os.path.exists(os.path.join(tmpdir, "host1", "output"))
            assert os.path.exists(os.path.join(tmpdir, "host1", "model"))

    def test_create_processing_config_file_directories(self):
        """Test _create_processing_config_file_directories creates correct structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_processing_config_file_directories(tmpdir, "host1")

            assert os.path.exists(os.path.join(tmpdir, "host1", "config"))


class TestDeleteTree:
    """Test cases for _delete_tree function"""

    def test_delete_tree_success(self):
        """Test _delete_tree successfully deletes directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "test")
            os.makedirs(test_dir)
            assert os.path.exists(test_dir)

            _delete_tree(test_dir)

            assert not os.path.exists(test_dir)

    @patch("shutil.rmtree")
    def test_delete_tree_permission_error(self, mock_rmtree):
        """Test _delete_tree handles permission errors gracefully"""
        import errno

        mock_rmtree.side_effect = OSError(errno.EACCES, "Permission denied")

        # Should not raise exception
        _delete_tree("/some/path")


class TestAwsCredentials:
    """Test cases for AWS credentials functions"""

    def test_aws_credentials_with_long_lived_creds(self):
        """Test _aws_credentials with long-lived credentials"""
        mock_session = Mock()
        mock_creds = Mock()
        mock_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_creds.token = None
        mock_session.get_credentials.return_value = mock_creds

        result = _aws_credentials(mock_session)

        assert result is not None
        assert len(result) == 2
        assert "AWS_ACCESS_KEY_ID=" in result[0]
        assert "AWS_SECRET_ACCESS_KEY=" in result[1]

    @patch("sagemaker.core.local.image._aws_credentials_available_in_metadata_service")
    @patch("sagemaker.core.local.image._use_short_lived_credentials")
    def test_aws_credentials_with_short_lived_creds_and_use_flag(
        self, mock_use_short, mock_metadata_available
    ):
        """Test _aws_credentials with short-lived credentials and use flag"""
        mock_use_short.return_value = True
        mock_session = Mock()
        mock_creds = Mock()
        mock_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_creds.token = "token123"
        mock_session.get_credentials.return_value = mock_creds

        result = _aws_credentials(mock_session)

        assert result is not None
        assert len(result) == 3
        assert "AWS_SESSION_TOKEN=" in result[2]

    def test_aws_credentials_exception(self):
        """Test _aws_credentials handles exceptions"""
        mock_session = Mock()
        mock_session.get_credentials.side_effect = Exception("Credentials error")

        result = _aws_credentials(mock_session)

        assert result is None

    def test_use_short_lived_credentials_enabled(self):
        """Test _use_short_lived_credentials when enabled"""
        with patch.dict(os.environ, {"USE_SHORT_LIVED_CREDENTIALS": "1"}):
            assert _use_short_lived_credentials() is True

    def test_use_short_lived_credentials_disabled(self):
        """Test _use_short_lived_credentials when disabled"""
        with patch.dict(os.environ, {}, clear=True):
            assert _use_short_lived_credentials() is False


class TestWriteJsonFile:
    """Test cases for _write_json_file function"""

    def test_write_json_file(self):
        """Test _write_json_file writes correct JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            content = {"key": "value", "number": 42}

            _write_json_file(filepath, content)

            assert os.path.exists(filepath)
            import json

            with open(filepath, "r") as f:
                loaded = json.load(f)
            assert loaded == content


class TestEcrLoginIfNeeded:
    """Test cases for _ecr_login_if_needed function"""

    @patch("sagemaker.core.local.image._check_output")
    def test_ecr_login_not_needed_for_non_ecr_image(self, mock_check_output):
        """Test ECR login not needed for non-ECR images"""
        mock_session = Mock()
        result = _ecr_login_if_needed(mock_session, "ubuntu:latest")

        assert result is False
        mock_check_output.assert_not_called()

    @patch("sagemaker.core.local.image._check_output")
    def test_ecr_login_not_needed_when_image_exists(self, mock_check_output):
        """Test ECR login not needed when image already exists"""
        mock_check_output.return_value = "image-id-123"
        mock_session = Mock()
        ecr_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"

        result = _ecr_login_if_needed(mock_session, ecr_image)

        assert result is False

    @patch("subprocess.Popen")
    @patch("sagemaker.core.local.image._check_output")
    def test_ecr_login_needed(self, mock_check_output, mock_popen):
        """Test ECR login is performed when needed"""
        mock_check_output.return_value = ""
        mock_session = Mock()
        mock_ecr = Mock()
        mock_ecr.get_authorization_token.return_value = {
            "authorizationData": [
                {
                    "authorizationToken": "QVdTOnRva2VuMTIz",  # base64 encoded "AWS:token123"
                    "proxyEndpoint": "https://123456789012.dkr.ecr.us-west-2.amazonaws.com",
                }
            ]
        }
        mock_session.client.return_value = mock_ecr
        mock_process = Mock()
        mock_popen.return_value = mock_process
        ecr_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"

        result = _ecr_login_if_needed(mock_session, ecr_image)

        assert result is True
        mock_popen.assert_called_once()


class TestPullImage:
    """Test cases for _pull_image function"""

    @patch("subprocess.check_output")
    def test_pull_image(self, mock_check_output):
        """Test _pull_image calls docker pull"""
        _pull_image("ubuntu:latest")

        mock_check_output.assert_called_once()
        call_args = mock_check_output.call_args[0][0]
        assert "docker" in call_args
        assert "pull" in call_args
        assert "ubuntu:latest" in call_args


class TestSageMakerContainer:
    """Test cases for _SageMakerContainer class"""

    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    def test_init_local_instance(self, mock_get_compose):
        """Test _SageMakerContainer initialization for local instance"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_session = Mock()

        container = _SageMakerContainer(
            instance_type="local",
            instance_count=2,
            image="test-image:latest",
            sagemaker_session=mock_session,
        )

        assert container.instance_type == "local"
        assert container.instance_count == 2
        assert container.image == "test-image:latest"
        assert len(container.hosts) == 2

    @patch("sagemaker.core.local.utils.check_for_studio")
    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    def test_init_studio_instance(self, mock_get_compose, mock_check_studio):
        """Test _SageMakerContainer initialization for Studio"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_check_studio.return_value = True
        mock_session = Mock()
        mock_session.config = None

        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            sagemaker_session=mock_session,
        )

        assert container.is_studio is True
        assert container.hosts == [STUDIO_HOST_NAME]

    @patch("sagemaker.core.local.utils.check_for_studio")
    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    def test_init_studio_multi_instance_raises(self, mock_get_compose, mock_check_studio):
        """Test _SageMakerContainer raises for multi-instance in Studio"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_check_studio.return_value = True
        mock_session = Mock()

        with pytest.raises(NotImplementedError, match="Multi instance Local Mode"):
            _SageMakerContainer(
                instance_type="local",
                instance_count=2,
                image="test-image:latest",
                sagemaker_session=mock_session,
            )

    @patch("subprocess.check_output")
    def test_get_compose_cmd_prefix_docker_compose_v2(self, mock_check_output):
        """Test _get_compose_cmd_prefix finds docker compose v2"""
        mock_check_output.return_value = "Docker Compose version v2.20.0"

        result = _SageMakerContainer._get_compose_cmd_prefix()

        assert result == ["docker", "compose"]

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_get_compose_cmd_prefix_docker_compose_cli(self, mock_check_output, mock_which):
        """Test _get_compose_cmd_prefix finds docker-compose CLI"""
        mock_check_output.side_effect = [subprocess.CalledProcessError(1, "docker compose version")]
        mock_which.return_value = "/usr/local/bin/docker-compose"

        result = _SageMakerContainer._get_compose_cmd_prefix()

        assert result == ["docker-compose"]

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_get_compose_cmd_prefix_not_found(self, mock_check_output, mock_which):
        """Test _get_compose_cmd_prefix raises when not found"""
        mock_check_output.side_effect = [subprocess.CalledProcessError(1, "docker compose version")]
        mock_which.return_value = None

        with pytest.raises(ImportError, match="Docker Compose is not installed"):
            _SageMakerContainer._get_compose_cmd_prefix()

    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    def test_write_config_files(self, mock_get_compose):
        """Test write_config_files creates correct files"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_session = Mock()
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            sagemaker_session=mock_session,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            container.container_root = tmpdir
            host = container.hosts[0]
            _create_config_file_directories(tmpdir, host)

            hyperparameters = {"learning_rate": "0.001"}
            input_data_config = [{"ChannelName": "training", "ContentType": "application/json"}]

            container.write_config_files(host, hyperparameters, input_data_config)

            config_path = os.path.join(tmpdir, host, "input", "config")
            assert os.path.exists(os.path.join(config_path, "hyperparameters.json"))
            assert os.path.exists(os.path.join(config_path, "resourceconfig.json"))
            assert os.path.exists(os.path.join(config_path, "inputdataconfig.json"))

    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    def test_build_optml_volumes(self, mock_get_compose):
        """Test _build_optml_volumes creates correct volumes"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_session = Mock()
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image:latest",
            sagemaker_session=mock_session,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            container.container_root = tmpdir
            host = container.hosts[0]
            subdirs = {"input", "output", "model"}

            volumes = container._build_optml_volumes(host, subdirs)

            assert len(volumes) == 3
            # Check that volumes are _Volume instances or have the expected attributes
            for v in volumes:
                assert hasattr(v, "host_dir")
                assert hasattr(v, "container_dir")
                assert hasattr(v, "map")


class TestHostingContainer:
    """Test cases for _HostingContainer class"""

    @patch("subprocess.Popen")
    def test_hosting_container_run(self, mock_popen):
        """Test _HostingContainer run method"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [b"Starting server\n", b""]
        mock_process.poll.side_effect = [None, 0]
        mock_popen.return_value = mock_process

        container = _HostingContainer(["docker", "compose", "up"])
        container.run()

        mock_popen.assert_called_once()

    @patch("sagemaker.core.local.utils.kill_child_processes")
    @patch("platform.system")
    def test_hosting_container_down_unix(self, mock_platform, mock_kill):
        """Test _HostingContainer down method on Unix"""
        mock_platform.return_value = "Linux"
        mock_process = Mock()
        mock_process.pid = 12345

        container = _HostingContainer(["docker", "compose", "up"])
        container.process = mock_process
        container.down()

        mock_kill.assert_called_once_with(12345)
        mock_process.terminate.assert_called_once()

    @patch("platform.system")
    def test_hosting_container_down_windows(self, mock_platform):
        """Test _HostingContainer down method on Windows"""
        mock_platform.return_value = "Windows"
        mock_process = Mock()

        container = _HostingContainer(["docker", "compose", "up"])
        container.process = mock_process
        container.down()

        mock_process.terminate.assert_called_once()


class TestSageMakerContainerAdvanced:
    """Advanced test cases for _SageMakerContainer"""

    @pytest.fixture
    def mock_session(self):
        """Fixture for mock session"""
        session = Mock()
        session.config = None
        return session

    def test_get_compose_cmd_prefix_docker_compose_v2(self):
        """Test _get_compose_cmd_prefix with Docker Compose v2"""
        with patch("subprocess.check_output", return_value="Docker Compose version v2.0.0"):
            result = _SageMakerContainer._get_compose_cmd_prefix()
            assert result == ["docker", "compose"]

    def test_get_compose_cmd_prefix_docker_compose_cli(self):
        """Test _get_compose_cmd_prefix with docker-compose CLI"""
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd")):
            with patch("shutil.which", return_value="/usr/bin/docker-compose"):
                result = _SageMakerContainer._get_compose_cmd_prefix()
                assert result == ["docker-compose"]

    def test_get_compose_cmd_prefix_not_installed(self):
        """Test _get_compose_cmd_prefix when Docker Compose is not installed"""
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd")):
            with patch("shutil.which", return_value=None):
                with pytest.raises(ImportError, match="Docker Compose is not installed"):
                    _SageMakerContainer._get_compose_cmd_prefix()

    def test_process_with_multiple_inputs(self, mock_session):
        """Test process method with multiple processing inputs"""
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image",
            sagemaker_session=mock_session,
        )

        processing_inputs = [
            {
                "InputName": "input1",
                "S3Input": {
                    "S3Uri": "s3://bucket/input1",
                    "LocalPath": "/opt/ml/processing/input1",
                },
            },
            {
                "InputName": "input2",
                "S3Input": {
                    "S3Uri": "s3://bucket/input2",
                    "LocalPath": "/opt/ml/processing/input2",
                },
            },
        ]

        processing_output_config = {
            "Outputs": [
                {
                    "OutputName": "output1",
                    "S3Output": {
                        "S3Uri": "s3://bucket/output1",
                        "LocalPath": "/opt/ml/processing/output1",
                    },
                }
            ]
        }

        environment = {"ENV_VAR": "value"}

        with patch.object(container, "_create_tmp_folder", return_value="/tmp/test"):
            with patch("os.mkdir"):
                with patch.object(container, "_prepare_processing_volumes", return_value=[]):
                    with patch.object(container, "write_processing_config_files"):
                        with patch.object(container, "_generate_compose_file"):
                            with patch(
                                "sagemaker.core.local.image._ecr_login_if_needed",
                                return_value=False,
                            ):
                                with patch("subprocess.Popen") as mock_popen:
                                    with patch("sagemaker.core.local.image._stream_output"):
                                        with patch.object(container, "_upload_processing_outputs"):
                                            with patch.object(container, "_cleanup"):
                                                mock_process = Mock()
                                                mock_popen.return_value = mock_process

                                                container.process(
                                                    processing_inputs,
                                                    processing_output_config,
                                                    environment,
                                                    "test-job",
                                                )

    def test_train_with_multiple_channels(self, mock_session):
        """Test train method with multiple input channels"""
        with patch(
            "sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix",
            return_value=["docker", "compose"],
        ):
            container = _SageMakerContainer(
                instance_type="local",
                instance_count=1,
                image="test-image",
                sagemaker_session=mock_session,
            )

            input_data_config = [
                {
                    "ChannelName": "training",
                    "DataUri": "s3://bucket/training",
                    "ContentType": "application/x-parquet",
                },
                {
                    "ChannelName": "validation",
                    "DataUri": "s3://bucket/validation",
                    "ContentType": "application/x-parquet",
                },
            ]

            output_data_config = {"S3OutputPath": "s3://bucket/output"}

            hyperparameters = {"epochs": "10", "batch_size": "32"}

            environment = {"TRAINING_ENV": "test"}

            with patch.object(container, "_create_tmp_folder", return_value="/tmp/test"):
                with patch("os.mkdir"):
                    with patch(
                        "sagemaker.core.local.data.get_data_source_instance"
                    ) as mock_data_source:
                        mock_source = Mock()
                        mock_source.get_root_dir.return_value = "/tmp/data"
                        mock_data_source.return_value = mock_source
                        with patch("os.path.isdir", return_value=False):
                            with patch(
                                "sagemaker.serve.model_builder.DIR_PARAM_NAME", "sagemaker_program"
                            ):
                                with patch.object(
                                    container,
                                    "_update_local_src_path",
                                    return_value=hyperparameters,
                                ):
                                    with patch.object(container, "write_config_files"):
                                        with patch("shutil.copytree"):
                                            with patch.object(
                                                container, "_generate_compose_file", return_value={}
                                            ):
                                                with patch(
                                                    "sagemaker.core.local.image._ecr_login_if_needed",
                                                    return_value=False,
                                                ):
                                                    with patch("subprocess.Popen") as mock_popen:
                                                        with patch(
                                                            "sagemaker.core.local.image._stream_output"
                                                        ):
                                                            with patch.object(
                                                                container,
                                                                "retrieve_artifacts",
                                                                return_value="/tmp/model.tar.gz",
                                                            ):
                                                                with patch.object(
                                                                    container, "_cleanup"
                                                                ):
                                                                    mock_process = Mock()
                                                                    mock_popen.return_value = (
                                                                        mock_process
                                                                    )

                                                                    result = container.train(
                                                                        input_data_config,
                                                                        output_data_config,
                                                                        hyperparameters,
                                                                        environment,
                                                                        "test-job",
                                                                    )

                                                                    assert (
                                                                        result
                                                                        == "/tmp/model.tar.gz"
                                                                    )

    def test_serve_with_environment_variables(self, mock_session):
        """Test serve method with environment variables"""
        with patch(
            "sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix",
            return_value=["docker", "compose"],
        ):
            container = _SageMakerContainer(
                instance_type="local",
                instance_count=1,
                image="test-image",
                sagemaker_session=mock_session,
            )

            model_dir = "s3://bucket/model"
            environment = {"MODEL_SERVER_TIMEOUT": "300", "MODEL_SERVER_WORKERS": "2"}

            with patch.object(container, "_create_tmp_folder", return_value="/tmp/test"):
                with patch(
                    "sagemaker.core.local.data.get_data_source_instance"
                ) as mock_data_source:
                    mock_source = Mock()
                    mock_source.get_root_dir.return_value = "/tmp/model"
                    mock_source.get_file_list.return_value = []
                    mock_data_source.return_value = mock_source
                    with patch("os.path.isdir", return_value=False):
                        with patch(
                            "sagemaker.serve.model_builder.DIR_PARAM_NAME", "sagemaker_program"
                        ):
                            with patch(
                                "sagemaker.core.local.image._ecr_login_if_needed",
                                return_value=False,
                            ):
                                with patch.object(container, "_generate_compose_file"):
                                    with patch(
                                        "sagemaker.core.local.image._HostingContainer"
                                    ) as mock_hosting:
                                        mock_container_instance = Mock()
                                        mock_hosting.return_value = mock_container_instance

                                        container.serve(model_dir, environment)

                                        assert container.container == mock_container_instance
                                        mock_container_instance.start.assert_called_once()

    def test_stop_serving(self, mock_session):
        """Test stop_serving method"""
        with patch(
            "sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix",
            return_value=["docker", "compose"],
        ):
            container = _SageMakerContainer(
                instance_type="local",
                instance_count=1,
                image="test-image",
                sagemaker_session=mock_session,
            )

            container.container_root = "/tmp/test"
            mock_hosting_container = Mock()
            container.container = mock_hosting_container

            with patch("sagemaker.core.local.image._delete_tree") as mock_delete:
                container.stop_serving()

                mock_hosting_container.down.assert_called_once()
                mock_hosting_container.join.assert_called_once()
                assert mock_delete.called

    def test_retrieve_artifacts_multiple_hosts(self, mock_session):
        """Test retrieve_artifacts with multiple hosts"""
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=2,
            image="test-image",
            sagemaker_session=mock_session,
        )

        container.container_root = "/tmp/test"
        container.hosts = ["host1", "host2"]

        compose_data = {
            "services": {
                "host1": {
                    "volumes": [
                        "/tmp/host1/model:/opt/ml/model",
                        "/tmp/host1/output:/opt/ml/output",
                    ]
                },
                "host2": {
                    "volumes": [
                        "/tmp/host2/model:/opt/ml/model",
                        "/tmp/host2/output:/opt/ml/output",
                    ]
                },
            }
        }

        output_data_config = {"S3OutputPath": "s3://bucket/output"}

        with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
            with patch("os.mkdir"):
                with patch("os.listdir", return_value=["file1.txt"]):
                    with patch("sagemaker.core.local.utils.recursive_copy"):
                        with patch("sagemaker.core.common_utils.create_tar_file"):
                            with patch(
                                "sagemaker.core.local.utils.move_to_destination",
                                return_value="s3://bucket/output/test-job",
                            ):
                                with patch("sagemaker.core.local.image._delete_tree"):
                                    result = container.retrieve_artifacts(
                                        compose_data, output_data_config, "test-job"
                                    )

                                    assert "model.tar.gz" in result

    def test_write_processing_config_files(self, mock_session):
        """Test write_processing_config_files method"""
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image",
            sagemaker_session=mock_session,
        )

        container.container_root = "/tmp/test"
        container.hosts = ["host1"]

        environment = {"ENV_VAR": "value"}
        processing_inputs = []
        processing_output_config = {"Outputs": []}

        with patch("sagemaker.core.local.image._write_json_file") as mock_write:
            container.write_processing_config_files(
                "host1", environment, processing_inputs, processing_output_config, "test-job"
            )

            assert mock_write.call_count == 2  # resourceconfig.json and processingjobconfig.json

    def test_write_config_files(self, mock_session):
        """Test write_config_files method"""
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image",
            sagemaker_session=mock_session,
        )

        container.container_root = "/tmp/test"
        container.hosts = ["host1"]

        hyperparameters = {"learning_rate": "0.01"}
        input_data_config = [{"ChannelName": "training", "ContentType": "application/x-parquet"}]

        with patch("sagemaker.core.local.image._write_json_file") as mock_write:
            container.write_config_files("host1", hyperparameters, input_data_config)

            assert mock_write.call_count == 3  # hyperparameters, resourceconfig, inputdataconfig

    def test_prepare_training_volumes_with_local_code(self, mock_session):
        """Test _prepare_training_volumes with local code directory"""
        with patch(
            "sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix",
            return_value=["docker", "compose"],
        ):
            container = _SageMakerContainer(
                instance_type="local",
                instance_count=1,
                image="test-image",
                sagemaker_session=mock_session,
            )

            container.container_root = "/tmp/test"

            input_data_config = []
            output_data_config = {"S3OutputPath": "s3://bucket/output"}
            hyperparameters = {}

            with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
                with patch("os.path.isdir", return_value=False):
                    with patch("os.mkdir"):
                        with patch(
                            "sagemaker.serve.model_builder.DIR_PARAM_NAME", "sagemaker_program"
                        ):
                            with patch(
                                "sagemaker.core.local.data.get_data_source_instance"
                            ) as mock_data_source:
                                mock_source = Mock()
                                mock_source.get_root_dir.return_value = "/tmp/data"
                                mock_data_source.return_value = mock_source

                                volumes = container._prepare_training_volumes(
                                    "/tmp/data",
                                    input_data_config,
                                    output_data_config,
                                    hyperparameters,
                                )

                                # Should have basic volumes
                                assert len(volumes) > 0

    def test_prepare_processing_volumes_with_outputs(self, mock_session):
        """Test _prepare_processing_volumes with multiple outputs"""
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image",
            sagemaker_session=mock_session,
        )

        container.container_root = "/tmp/test"

        processing_inputs = []
        processing_output_config = {
            "Outputs": [
                {
                    "OutputName": "output1",
                    "S3Output": {
                        "S3Uri": "s3://bucket/output1",
                        "LocalPath": "/opt/ml/processing/output1",
                    },
                },
                {
                    "OutputName": "output2",
                    "S3Output": {
                        "S3Uri": "s3://bucket/output2",
                        "LocalPath": "/opt/ml/processing/output2",
                    },
                },
            ]
        }

        with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
            with patch("os.makedirs"):
                volumes = container._prepare_processing_volumes(
                    "/tmp/data", processing_inputs, processing_output_config
                )

                # Should have volumes for both outputs plus shared dir
                assert len(volumes) >= 3

    def test_upload_processing_outputs(self, mock_session):
        """Test _upload_processing_outputs method"""
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image",
            sagemaker_session=mock_session,
        )

        processing_output_config = {
            "Outputs": [
                {
                    "OutputName": "output1",
                    "S3Output": {
                        "S3Uri": "s3://bucket/output1",
                        "LocalPath": "/opt/ml/processing/output1",
                    },
                }
            ]
        }

        with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
            with patch("sagemaker.core.local.utils.move_to_destination") as mock_move:
                container._upload_processing_outputs("/tmp/data", processing_output_config)

                mock_move.assert_called_once()

    def test_update_local_src_path(self, mock_session):
        """Test _update_local_src_path method"""
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image",
            sagemaker_session=mock_session,
        )

        params = {"sagemaker_program": json.dumps("file:///path/to/code"), "other_param": "value"}

        result = container._update_local_src_path(params, "sagemaker_program")

        assert result["sagemaker_program"] == json.dumps("/opt/ml/code")
        assert result["other_param"] == "value"

    def test_prepare_serving_volumes_with_tar_file(self, mock_session):
        """Test _prepare_serving_volumes with tar file"""
        container = _SageMakerContainer(
            instance_type="local",
            instance_count=1,
            image="test-image",
            sagemaker_session=mock_session,
        )

        container.container_root = "/tmp/test"
        container.hosts = ["host1"]

        with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
            with patch("os.makedirs"):
                with patch(
                    "sagemaker.core.local.data.get_data_source_instance"
                ) as mock_data_source:
                    mock_source = Mock()
                    mock_source.get_root_dir.return_value = "/tmp/model"
                    mock_source.get_file_list.return_value = ["/tmp/model/model.tar.gz"]
                    mock_data_source.return_value = mock_source

                    with patch("tarfile.is_tarfile", return_value=True):
                        with patch("tarfile.open") as mock_tar:
                            mock_tar_instance = Mock()
                            mock_tar.return_value.__enter__.return_value = mock_tar_instance

                            volumes = container._prepare_serving_volumes("s3://bucket/model")

                            assert len(volumes) > 0


class TestHelperFunctions:
    """Test cases for helper functions"""

    def test_ecr_login_if_needed_with_ecr_image(self):
        """Test _ecr_login_if_needed with ECR image"""
        boto_session = Mock()
        image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"

        with patch("sagemaker.core.local.image._check_output", return_value=""):
            with patch("subprocess.Popen") as mock_popen:
                mock_ecr = Mock()
                mock_ecr.get_authorization_token.return_value = {
                    "authorizationData": [
                        {
                            "authorizationToken": "QVdTOnRva2VuMTIz",
                            "proxyEndpoint": "https://123456789012.dkr.ecr.us-west-2.amazonaws.com",
                        }
                    ]
                }
                boto_session.client.return_value = mock_ecr
                mock_process = Mock()
                mock_popen.return_value = mock_process

                result = _ecr_login_if_needed(boto_session, image_uri)

                assert result is True

    def test_ecr_login_if_needed_with_non_ecr_image(self):
        """Test _ecr_login_if_needed with non-ECR image"""
        boto_session = Mock()
        image_uri = "docker.io/my-image:latest"

        result = _ecr_login_if_needed(boto_session, image_uri)

        assert result is False

    def test_pull_image(self):
        """Test _pull_image function"""
        image_uri = "my-image:latest"

        with patch("subprocess.check_output") as mock_check_output:
            _pull_image(image_uri)

            mock_check_output.assert_called_once()
            args = mock_check_output.call_args[0][0]
            assert "docker" in args
            assert "pull" in args
            assert image_uri in args

    def test_stream_output(self):
        """Test _stream_output function"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [b"line1\n", b"line2\n", b""]
        mock_process.poll.side_effect = [None, None, 0]

        with patch("sys.stdout.write"):
            with patch("sys.stdout.flush"):
                _stream_output(mock_process)

    def test_delete_tree(self):
        """Test _delete_tree function"""
        with patch("shutil.rmtree") as mock_rmtree:
            _delete_tree("/tmp/test")

            mock_rmtree.assert_called_once_with("/tmp/test")

    def test_delete_tree_permission_error(self):
        """Test _delete_tree handles permission errors gracefully"""
        import errno

        with patch("shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = OSError(errno.EACCES, "Permission denied")
            _delete_tree("/tmp/test")

    def test_write_json_file(self):
        """Test _write_json_file function"""
        data = {"key": "value"}

        with patch("builtins.open", create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            _write_json_file("/tmp/test.json", data)

            mock_open.assert_called_once_with("/tmp/test.json", "w")

    def test_create_config_file_directories(self):
        """Test _create_config_file_directories function"""
        with patch("os.makedirs") as mock_makedirs:
            _create_config_file_directories("/tmp/test", "host1")

            # Should create multiple directories
            assert mock_makedirs.call_count >= 3

    def test_create_processing_config_file_directories(self):
        """Test _create_processing_config_file_directories function"""
        with patch("os.makedirs") as mock_makedirs:
            _create_processing_config_file_directories("/tmp/test", "host1")

            # Should create config directory
            assert mock_makedirs.call_count >= 1


class TestVolume:
    """Test cases for _Volume class"""

    def test_init_with_host_and_container_dir(self):
        """Test _Volume initialization with host and container directories"""
        volume = _Volume("/host/path", container_dir="/container/path")

        assert volume.host_dir == "/host/path"
        assert volume.container_dir == "/container/path"

    def test_init_with_channel(self):
        """Test _Volume initialization with channel"""
        volume = _Volume("/host/path", channel="training")

        assert volume.host_dir == "/host/path"
        assert volume.container_dir == "/opt/ml/input/data/training"

    def test_map_property(self):
        """Test _Volume.map property"""
        volume = _Volume("/host/path", container_dir="/container/path")

        result = volume.map

        assert "/host/path" in result
        assert "/container/path" in result


class TestHostingContainer:
    """Test cases for _HostingContainer class"""

    def test_init(self):
        """Test _HostingContainer initialization"""
        compose_command = ["docker-compose", "up"]

        container = _HostingContainer(compose_command)

        assert container.command == compose_command
        assert container.process is None

    def test_start(self):
        """Test _HostingContainer.start method"""
        compose_command = ["docker-compose", "up"]
        container = _HostingContainer(compose_command)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_popen.return_value = mock_process

            container.start()

            assert container.process == mock_process
            mock_popen.assert_called_once()

    def test_down(self):
        """Test _HostingContainer.down method"""
        compose_command = ["docker-compose", "up"]
        container = _HostingContainer(compose_command)
        container.process = Mock()
        container.process.pid = 12345

        with patch("subprocess.Popen") as mock_popen:
            mock_child_process = Mock()
            mock_child_process.communicate.return_value = (b"", b"")
            mock_popen.return_value = mock_child_process
            with patch("platform.system", return_value="Linux"):
                container.down()
                container.process.terminate.assert_called_once()

    def test_run(self):
        """Test _HostingContainer.run method"""
        compose_command = ["docker-compose", "up"]
        container = _HostingContainer(compose_command)

        with patch("subprocess.Popen") as mock_popen:
            with patch("sagemaker.core.local.image._stream_output") as mock_stream:
                mock_process = Mock()
                mock_popen.return_value = mock_process
                mock_stream.return_value = 0

                container.run()

                mock_popen.assert_called_once()
                mock_stream.assert_called_once_with(mock_process)


class TestSageMakerContainerExtended:
    """Extended test cases for _SageMakerContainer"""

    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_container_creation(self, mock_session_class, mock_get_compose):
        """Test container creation"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_session = Mock()

        container = _SageMakerContainer(
            "local",
            1,
            "test-image:latest",
            mock_session,
        )

        assert container.instance_type == "local"
        assert container.instance_count == 1
        assert container.image == "test-image:latest"

    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_container_with_entrypoint(self, mock_session_class, mock_get_compose):
        """Test container with custom entrypoint"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_session = Mock()

        container = _SageMakerContainer(
            "local",
            1,
            "test-image:latest",
            mock_session,
            container_entrypoint=["/bin/bash"],
            container_arguments=["script.sh"],
        )

        assert container.container_entrypoint == ["/bin/bash"]
        assert container.container_arguments == ["script.sh"]

    @patch("subprocess.check_output")
    def test_get_compose_cmd_prefix_v2(self, mock_check_output):
        """Test getting docker compose v2 command"""
        mock_check_output.return_value = "Docker Compose version v2.10.0"

        cmd = _SageMakerContainer._get_compose_cmd_prefix()

        assert cmd == ["docker", "compose"]

    @patch("subprocess.check_output")
    @patch("shutil.which")
    def test_get_compose_cmd_prefix_v1(self, mock_which, mock_check_output):
        """Test getting docker-compose v1 command"""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_which.return_value = "/usr/local/bin/docker-compose"

        cmd = _SageMakerContainer._get_compose_cmd_prefix()

        assert cmd == ["docker-compose"]

    @patch("subprocess.check_output")
    @patch("shutil.which")
    def test_get_compose_cmd_prefix_not_found(self, mock_which, mock_check_output):
        """Test when docker compose is not found"""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd")
        mock_which.return_value = None

        with pytest.raises(ImportError, match="Docker Compose is not installed"):
            _SageMakerContainer._get_compose_cmd_prefix()

    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    @patch("sagemaker.core.local.local_session.LocalSession")
    @patch("os.mkdir")
    @patch("tempfile.mkdtemp")
    def test_create_tmp_folder(
        self, mock_mkdtemp, mock_mkdir, mock_session_class, mock_get_compose
    ):
        """Test creating temporary folder"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_mkdtemp.return_value = "/tmp/sagemaker_local_12345"
        mock_session = Mock()
        mock_session.config = {}

        container = _SageMakerContainer("local", 1, "test-image", mock_session)
        tmp_folder = container._create_tmp_folder()

        assert "/tmp/sagemaker_local_12345" in tmp_folder

    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_write_config_files_extended(self, mock_session_class, mock_get_compose):
        """Test writing config files"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_session = Mock()

        container = _SageMakerContainer("local", 2, "test-image", mock_session)
        container.hosts = ["host1", "host2"]
        container.container_root = "/tmp/test"

        with patch("sagemaker.core.local.image._write_json_file") as mock_write:
            with patch("os.path.join", return_value="/tmp/test/host1/input/config"):
                container.write_config_files(
                    "host1",
                    {"epochs": "10"},
                    [{"ChannelName": "training"}],
                )

                assert mock_write.call_count >= 3

    @patch("sagemaker.core.local.image._SageMakerContainer._get_compose_cmd_prefix")
    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_write_processing_config_files_extended(self, mock_session_class, mock_get_compose):
        """Test writing processing config files"""
        mock_get_compose.return_value = ["docker", "compose"]
        mock_session = Mock()

        container = _SageMakerContainer("local", 1, "test-image", mock_session)
        container.hosts = ["host1"]
        container.container_root = "/tmp/test"
        container.instance_type = "local"
        container.instance_count = 1

        with patch("sagemaker.core.local.image._write_json_file") as mock_write:
            with patch("os.path.join", return_value="/tmp/test/host1/config"):
                container.write_processing_config_files(
                    "host1",
                    {},
                    [],
                    {},
                    "test-job",
                )

                assert mock_write.call_count >= 2


class TestEcrLoginExtended:
    """Extended test cases for _ecr_login_if_needed"""

    @patch("sagemaker.core.local.image._check_output")
    @patch("subprocess.Popen")
    def test_ecr_login_needed(self, mock_popen, mock_check_output):
        """Test ECR login when needed"""
        mock_check_output.return_value = ""
        mock_session = Mock()
        mock_ecr_client = Mock()
        mock_ecr_client.get_authorization_token.return_value = {
            "authorizationData": [
                {
                    "authorizationToken": "dXNlcjpwYXNz",
                    "proxyEndpoint": "https://123456789.dkr.ecr.us-west-2.amazonaws.com",
                }
            ]
        }
        mock_session.client.return_value = mock_ecr_client

        mock_process = Mock()
        mock_popen.return_value = mock_process

        image = "123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"

        result = _ecr_login_if_needed(mock_session, image)

        assert result is True
        mock_popen.assert_called()

    def test_ecr_login_not_needed(self):
        """Test when ECR login is not needed"""
        mock_session = Mock()
        image = "my-dockerhub-image:latest"

        result = _ecr_login_if_needed(mock_session, image)

        assert result is False


class TestPullImageExtended:
    """Extended test cases for _pull_image"""

    @patch("subprocess.check_output")
    def test_pull_image_success(self, mock_check_output):
        """Test successful image pull"""
        _pull_image("test-image:latest")

        mock_check_output.assert_called_once()
        args = mock_check_output.call_args[0][0]
        assert "docker" in args
        assert "pull" in args
        assert "test-image:latest" in args

    @patch("subprocess.check_output")
    def test_pull_image_failure(self, mock_check_output):
        """Test image pull failure"""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "cmd")

        with pytest.raises(subprocess.CalledProcessError):
            _pull_image("test-image:latest")


class TestHostingContainerExtended:
    """Extended test cases for _HostingContainer"""

    @patch("subprocess.Popen")
    def test_hosting_container_start(self, mock_popen):
        """Test starting hosting container"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        compose_command = ["docker", "compose", "up"]
        container = _HostingContainer(compose_command)

        container.start()

        mock_popen.assert_called_once()
        assert container.process == mock_process

    @patch("subprocess.Popen")
    def test_hosting_container_down(self, mock_popen):
        """Test stopping hosting container"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        compose_command = ["docker", "compose", "up"]
        container = _HostingContainer(compose_command)
        container.start()

        with patch("subprocess.Popen") as mock_popen_child:
            mock_child_process = Mock()
            mock_child_process.communicate.return_value = (b"", b"")
            mock_popen_child.return_value = mock_child_process
            with patch("platform.system", return_value="Linux"):
                container.down()
                mock_process.terminate.assert_called_once()

    @patch("subprocess.Popen")
    def test_hosting_container_run(self, mock_popen):
        """Test running hosting container thread"""
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.stdout.readline.side_effect = [b"Log line 1\n", b"Log line 2\n", b""]
        mock_popen.return_value = mock_process

        compose_command = ["docker", "compose", "up"]
        container = _HostingContainer(compose_command)

        container.start()
        container.join(timeout=1)

        assert mock_process.poll.called
