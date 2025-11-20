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
"""LocalContainer class module."""
from __future__ import absolute_import

import base64
import logging
import os
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict

# Constant defined here to avoid importing from sagemaker.serve.model
# which would unnecessarily load deployment-related dependencies
DIR_PARAM_NAME = "sagemaker_submit_directory"
logger = logging.getLogger(__name__)

from sagemaker.core.local.image import (
    _stream_output,
    _pull_image,
    _write_json_file,
    _aws_credentials,
    _Volume,
    _check_output,
)

from sagemaker.core.local.utils import check_for_studio, recursive_copy
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes import Channel


from sagemaker.core.common_utils import (
    ECR_URI_PATTERN,
    create_tar_file,
    _module_import_error,
    download_folder,
)
from sagemaker.core.utils.utils import Unassigned
from sagemaker.core.shapes import DataSource

from six.moves.urllib.parse import urlparse

STUDIO_HOST_NAME = "sagemaker-local"
DOCKER_COMPOSE_FILENAME = "docker-compose.yaml"
DOCKER_COMPOSE_HTTP_TIMEOUT_ENV = "COMPOSE_HTTP_TIMEOUT"
DOCKER_COMPOSE_HTTP_TIMEOUT = "120"

REGION_ENV_NAME = "AWS_REGION"
TRAINING_JOB_NAME_ENV_NAME = "TRAINING_JOB_NAME"
S3_ENDPOINT_URL_ENV_NAME = "S3_ENDPOINT_URL"
S3_ENDPOINT_URL_ENV_NAME = "S3_ENDPOINT_URL"
SM_STUDIO_LOCAL_MODE = "SM_STUDIO_LOCAL_MODE"


class _LocalContainer(BaseModel):
    """A local training job class for local mode model trainer.

    Attributes:
        training_job_name (str):
            The name of the training job.
        instance_type (str):
            The instance type.
        instance_count (int):
            The number of instances.
        image (str):
            The image name for training.
        container_root (str):
            The directory path for the local container root.
        input_from_s3 (bool):
            If the input is from s3.
        is_studio (bool):
            If the container is running on SageMaker studio instance.
        hosts (Optional[List[str]]):
            The list of host names.
        input_data_config: Optional[List[Channel]]
            The input data channels for the training job.
            Takes a list of Channel objects or a dictionary of channel names to DataSourceType.
            DataSourceType can be an S3 URI string, local file path string,
            S3DataSource object, or FileSystemDataSource object.
        environment (Optional[Dict[str, str]]):
            The environment variables for the training job.
        hyper_parameters (Optional[Dict[str, Any]]):
            The hyperparameters for the training job.
        sagemaker_session (Optional[Session]):
            The SageMaker session.
            For local mode training, SageMaker session will only be used when input is from S3 or
            image needs to be pulled from ECR.
        container_entrypoint (Optional[List[str]]):
            The command to be executed in the container.
        container_arguments (Optional[List[str]]):
            The arguments of the container commands.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    training_job_name: str
    instance_type: str
    instance_count: int
    image: str
    container_root: str
    input_from_s3: Optional[bool] = False
    is_studio: Optional[bool] = False
    hosts: Optional[List[str]] = []
    input_data_config: Optional[List[Channel]]
    environment: Optional[Dict[str, str]]
    hyper_parameters: Optional[Dict[str, str]]
    sagemaker_session: Optional[Session] = None
    container_entrypoint: Optional[List[str]]
    container_arguments: Optional[List[str]]

    _temporary_folders: List[str] = []

    def model_post_init(self, __context: Any):
        """Post init method to perform custom validation and set default values."""
        self.hosts = [f"algo-{i}" for i in range(1, self.instance_count + 1)]
        if self.environment is None:
            self.environment = {}
        if self.hyper_parameters is None:
            self.hyper_parameters = {}

        for channel in self.input_data_config:
            if channel.data_source and channel.data_source.s3_data_source != Unassigned():
                self.input_from_s3 = True
                data_distribution = channel.data_source.s3_data_source.s3_data_distribution_type
                if self.sagemaker_session is None:
                    # In local mode only initiate session when neccessary
                    self.sagemaker_session = Session()
            elif (
                channel.data_source and channel.data_source.file_system_data_source != Unassigned()
            ):
                self.input_from_s3 = False
                data_distribution = channel.data_source.file_system_data_source.file_system_type
            else:
                raise ValueError(
                    "Need channel.data_source to have s3_data_source or file_system_data_source"
                )

            supported_distributions = ["FullyReplicated", "EFS"]
            if data_distribution and data_distribution not in supported_distributions:
                raise RuntimeError(
                    "Invalid Data Distribution: '{}'. Local mode currently supports FullyReplicated "
                    "Distribution for S3 data source and EFS Distribution for local data source.".format(
                        data_distribution,
                    )
                )
        self.is_studio = check_for_studio()

    def train(
        self,
        wait: bool,
    ) -> str:
        """Run a training job locally using docker-compose.

        Args:
            wait (bool):
                Whether to wait the training output before exiting.
        """
        # create output/data folder since sagemaker-containers 2.0 expects it
        os.makedirs(os.path.join(self.container_root, "output", "data"), exist_ok=True)
        # A shared directory for all the containers. It is only mounted if the training script is
        # Local.
        os.makedirs(os.path.join(self.container_root, "shared"), exist_ok=True)

        data_dir = os.path.join(self.container_root, "input", "data")
        os.makedirs(data_dir, exist_ok=True)
        volumes = self._prepare_training_volumes(
            data_dir, self.input_data_config, self.hyper_parameters
        )
        # If local, source directory needs to be updated to mounted /opt/ml/code path
        if DIR_PARAM_NAME in self.hyper_parameters:
            src_dir = self.hyper_parameters[DIR_PARAM_NAME]
            parsed_uri = urlparse(src_dir)
            if parsed_uri.scheme == "file":
                self.hyper_parameters[DIR_PARAM_NAME] = "/opt/ml/code"

        for host in self.hosts:
            # Create the configuration files
            self._create_config_file_directories(host)
            self._write_config_files(host, self.input_data_config, self.hyper_parameters)

        self.environment[TRAINING_JOB_NAME_ENV_NAME] = self.training_job_name
        if self.input_from_s3:
            self.environment[S3_ENDPOINT_URL_ENV_NAME] = (
                self.sagemaker_session.s3_resource.meta.client._endpoint.host
            )

        if self._ecr_login_if_needed():
            _pull_image(self.image)

        if self.sagemaker_session:
            self.environment[REGION_ENV_NAME] = self.sagemaker_session.boto_region_name

        compose_data = self._generate_compose_file(self.environment, volumes)
        compose_command = self._generate_compose_command(wait)
        process = subprocess.Popen(
            compose_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        try:
            _stream_output(process)
        finally:
            artifacts = self.retrieve_artifacts(compose_data)

        # Print our Job Complete line
        logger.info("Local training job completed, output artifacts saved to %s", artifacts)

        shutil.rmtree(os.path.join(self.container_root, "input"))
        shutil.rmtree(os.path.join(self.container_root, "shared"))
        for host in self.hosts:
            shutil.rmtree(os.path.join(self.container_root, host))
        for folder in self._temporary_folders:
            shutil.rmtree(os.path.join(self.container_root, folder))
        return artifacts

    def retrieve_artifacts(
        self,
        compose_data: dict,
    ):
        """Get the model artifacts from all the container nodes.

        Used after training completes to gather the data from all the
        individual containers. As the official SageMaker Training Service, it
        will override duplicate files if multiple containers have the same file
        names.

        Args:
            compose_data (dict): Docker-Compose configuration in dictionary
                format.

        Returns: Local path to the collected model artifacts.
        """
        # We need a directory to store the artfiacts from all the nodes
        # and another one to contained the compressed final artifacts
        artifacts = os.path.join(self.container_root, "artifacts")
        compressed_artifacts = os.path.join(self.container_root, "compressed_artifacts")
        os.makedirs(artifacts, exist_ok=True)

        model_artifacts = os.path.join(artifacts, "model")
        output_artifacts = os.path.join(artifacts, "output")

        artifact_dirs = [model_artifacts, output_artifacts, compressed_artifacts]
        for d in artifact_dirs:
            os.makedirs(d, exist_ok=True)

        # Gather the artifacts from all nodes into artifacts/model and artifacts/output
        for host in self.hosts:
            volumes = compose_data["services"][str(host)]["volumes"]
            volumes = [v[:-2] if v.endswith(":z") else v for v in volumes]
            for volume in volumes:
                if re.search(r"^[A-Za-z]:", volume):
                    unit, host_dir, container_dir = volume.split(":")
                    host_dir = unit + ":" + host_dir
                else:
                    host_dir, container_dir = volume.split(":")
                if container_dir == "/opt/ml/model":
                    recursive_copy(host_dir, model_artifacts)
                elif container_dir == "/opt/ml/output":
                    recursive_copy(host_dir, output_artifacts)

        # Tar Artifacts -> model.tar.gz and output.tar.gz
        model_files = [os.path.join(model_artifacts, name) for name in os.listdir(model_artifacts)]
        output_files = [
            os.path.join(output_artifacts, name) for name in os.listdir(output_artifacts)
        ]
        create_tar_file(model_files, os.path.join(compressed_artifacts, "model.tar.gz"))
        create_tar_file(output_files, os.path.join(compressed_artifacts, "output.tar.gz"))

        output_data = "file://%s" % compressed_artifacts

        return os.path.join(output_data, "model.tar.gz")

    def _create_config_file_directories(self, host: str):
        """Creates the directories for the config files.

        Args:
            host (str): The name of the current host.
        """
        for d in ["input", "input/config", "output", "model"]:
            os.makedirs(os.path.join(self.container_root, host, d), exist_ok=True)

    def _write_config_files(
        self,
        host: str,
        input_data_config: Optional[List[Channel]],
        hyper_parameters: Optional[Dict[str, str]],
    ):
        """Write the config files for the training containers.

        This method writes the hyper_parameters, resources and input data
        configuration files.

        Returns: None

        Args:
            host (str): The name of the current host.
            input_data_config (List[Channel]): Training input channels to be used for
                training.
            hyper_parameters (Dict[str, str]): Hyperparameters for training.
        """
        config_path = os.path.join(self.container_root, host, "input", "config")
        # Only support single container now
        resource_config = {
            "current_host": host,
            "hosts": self.hosts,
            "network_interface_name": "ethwe",
            "current_instance_type": self.instance_type,
        }

        json_input_data_config = {}
        for channel in input_data_config:
            channel_name = channel.channel_name
            json_input_data_config[channel_name] = {"TrainingInputMode": "File"}
            if channel.content_type != Unassigned():
                json_input_data_config[channel_name]["ContentType"] = channel.content_type

        _write_json_file(os.path.join(config_path, "hyperparameters.json"), hyper_parameters)
        _write_json_file(os.path.join(config_path, "resourceconfig.json"), resource_config)
        _write_json_file(os.path.join(config_path, "inputdataconfig.json"), json_input_data_config)

    def _generate_compose_file(self, environment: Dict[str, str], volumes: List[str]) -> dict:
        """Writes a config file describing a training/hosting environment.

        This method generates a docker compose configuration file, it has an
        entry for each container that will be created (based on self.hosts). it
        calls
        :meth:~sagemaker.local_session.SageMakerContainer._create_docker_host to
        generate the config for each individual container.

        Args:
            environment (Dict[str, str]): a dictionary with environment variables to be
                passed on to the containers.
            volumes (List[str]): a list of volumes that will be mapped to
                the containers

        Returns: (dict) A dictionary representation of the configuration that was written.
        """

        if os.environ.get(DOCKER_COMPOSE_HTTP_TIMEOUT_ENV) is None:
            os.environ[DOCKER_COMPOSE_HTTP_TIMEOUT_ENV] = DOCKER_COMPOSE_HTTP_TIMEOUT

        services = {
            host: self._create_docker_host(host, environment, volumes) for host in self.hosts
        }

        if self.is_studio:
            content = {
                "services": services,
            }
        else:
            content = {
                "services": services,
                "networks": {"sagemaker-local": {"name": "sagemaker-local"}},
            }

        docker_compose_path = os.path.join(self.container_root, DOCKER_COMPOSE_FILENAME)

        try:
            import yaml
        except ImportError as e:
            logger.error(_module_import_error("yaml", "Local mode", "local"))
            raise e

        yaml_content = yaml.dump(content, default_flow_style=False)
        with open(docker_compose_path, "w") as f:
            f.write(yaml_content)

        return content

    def _create_docker_host(
        self,
        host: str,
        environment: Dict[str, str],
        volumes: List[str],
    ) -> Dict:
        """Creates the docker host configuration.

        Args:
            host (str): The host address
            environment (Dict[str, str]): a dictionary with environment variables to be
                passed on to the containers.
            volumes (List[str]): List of volumes that will be mapped to the containers
        """
        environment = ["{}={}".format(k, v) for k, v in environment.items()]
        aws_creds = None
        if self.sagemaker_session:
            # In local mode only get aws credentials when neccessary
            aws_creds = _aws_credentials(self.sagemaker_session.boto_session)
        if aws_creds is not None:
            environment.extend(aws_creds)

        if self.is_studio:
            environment.extend([f"{SM_STUDIO_LOCAL_MODE}=True"])

        # Add volumes for the input and output of each host
        host_volumes = volumes.copy()
        subdirs = ["output", "output/data", "input"]
        for subdir in subdirs:
            host_dir = os.path.join(self.container_root, host, subdir)
            container_dir = "/opt/ml/{}".format(subdir)
            volume = _Volume(host_dir, container_dir)
            host_volumes.append(volume.map)

        host_config = {
            "image": self.image,
            "volumes": host_volumes,
            "environment": environment,
        }

        if self.container_entrypoint:
            host_config["entrypoint"] = self.container_entrypoint
        if self.container_arguments:
            host_config["entrypoint"] = host_config["entrypoint"] + self.container_arguments

        if self.is_studio:
            host_config["network_mode"] = "sagemaker"
        else:
            host_config["networks"] = {"sagemaker-local": {"aliases": [host]}}

        # for GPU support pass in nvidia as the runtime, this is equivalent
        # to setting --runtime=nvidia in the docker commandline.
        if self.instance_type == "local_gpu":
            host_config["deploy"] = {
                "resources": {
                    "reservations": {"devices": [{"count": "all", "capabilities": ["gpu"]}]}
                }
            }

        return host_config

    def _generate_compose_command(self, wait: bool):
        """Invokes the docker compose command.

        Args:
            wait (bool): Whether to wait for the docker command result.
        """
        _compose_cmd_prefix = self._get_compose_cmd_prefix()

        command = _compose_cmd_prefix + [
            "-f",
            os.path.join(self.container_root, DOCKER_COMPOSE_FILENAME),
            "up",
            "--build",
            "--abort-on-container-exit" if wait else "--detach",
        ]

        logger.info("docker command: %s", " ".join(command))
        return command

    def _ecr_login_if_needed(self):
        """Log into ECR, if needed.

        Only ECR images that not have been pulled locally need login.
        """
        sagemaker_pattern = re.compile(ECR_URI_PATTERN)
        sagemaker_match = sagemaker_pattern.match(self.image)
        if not sagemaker_match:
            return False

        # Do we already have the image locally?
        if _check_output("docker images -q %s" % self.image).strip():
            return False

        if not self.sagemaker_session:
            # In local mode only initiate session when neccessary
            self.sagemaker_session = Session()

        ecr = self.sagemaker_session.boto_session.client("ecr")
        auth = ecr.get_authorization_token(registryIds=[self.image.split(".")[0]])
        authorization_data = auth["authorizationData"][0]

        raw_token = base64.b64decode(authorization_data["authorizationToken"])
        token = raw_token.decode("utf-8").strip("AWS:")
        ecr_url = auth["authorizationData"][0]["proxyEndpoint"]

        # Log in to ecr, but use communicate to not print creds to the console
        cmd = f"docker login {ecr_url} -u AWS --password-stdin".split()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
        )

        proc.communicate(input=token.encode())

        return True

    def _prepare_training_volumes(
        self,
        data_dir: str,
        input_data_config: Optional[List[Channel]],
        hyper_parameters: Optional[Dict[str, str]],
    ) -> List[str]:
        """Prepares the training volumes based on input and output data configs.

        Args:
            data_dir (str): The directory of input data.
            input_data_config (Optional[List[Channel]]): Training input channels to be used for
                training.
            hyper_parameters (Optional[Dict[str, str]]): Hyperparameters for training.
        """
        volumes = []
        model_dir = os.path.join(self.container_root, "model")
        volumes.append(_Volume(model_dir, "/opt/ml/model").map)

        # Mount the metadata directory if present.
        # Only expected to be present on SM notebook instances.
        # This is used by some DeepEngine libraries
        metadata_dir = "/opt/ml/metadata"
        if os.path.isdir(metadata_dir):
            volumes.append(_Volume(metadata_dir, metadata_dir).map)

        # Set up the channels for the containers. For local data we will
        # mount the local directory to the container. For S3 Data we will download the S3 data
        # first.
        for channel in input_data_config:
            channel_name = channel.channel_name
            channel_dir = os.path.join(data_dir, channel_name)
            os.makedirs(channel_dir, exist_ok=True)

            data_source_local_path = self._get_data_source_local_path(channel.data_source)
            volumes.append(_Volume(data_source_local_path, channel=channel_name).map)

        # If there is a training script directory and it is a local directory,
        # mount it to the container.
        if DIR_PARAM_NAME in hyper_parameters:
            training_dir = hyper_parameters[DIR_PARAM_NAME]
            parsed_uri = urlparse(training_dir)
            if parsed_uri.scheme == "file":
                host_dir = os.path.abspath(parsed_uri.netloc + parsed_uri.path)
                volumes.append(_Volume(host_dir, "/opt/ml/code").map)
                shared_dir = os.path.join(self.container_root, "shared")
                volumes.append(_Volume(shared_dir, "/opt/ml/shared").map)

        return volumes

    def _get_data_source_local_path(self, data_source: DataSource):
        """Return a local data path of :class:`sagemaker.local.data.DataSource`.

        If the data source is from S3, the data will be downloaded to a temporary
        local path.
        If the data source is local file, the absolute path will be returned.

        Args:
            data_source (DataSource): a data source of local file or s3

        Returns:
            str: The local path of the data.
        """
        if data_source.s3_data_source != Unassigned():
            uri = data_source.s3_data_source.s3_uri
            parsed_uri = urlparse(uri)
            local_dir = TemporaryDirectory(prefix=os.path.join(self.container_root + "/")).name
            self._temporary_folders.append(local_dir)
            download_folder(parsed_uri.netloc, parsed_uri.path, local_dir, self.sagemaker_session)
            return local_dir
        else:
            return os.path.abspath(data_source.file_system_data_source.directory_path)

    def _get_compose_cmd_prefix(self) -> List[str]:
        """Gets the Docker Compose command.

        The method initially looks for 'docker compose' v2
        executable, if not found looks for 'docker-compose' executable.

        Returns:
            List[str]: Docker Compose executable split into list.

        Raises:
            ImportError: If Docker Compose executable was not found.
        """
        compose_cmd_prefix = []

        output = None
        try:
            output = subprocess.check_output(
                ["docker", "compose", "version"],
                stderr=subprocess.DEVNULL,
                encoding="UTF-8",
            )
        except subprocess.CalledProcessError:
            logger.info(
                "'Docker Compose' is not installed. "
                "Proceeding to check for 'docker-compose' CLI."
            )

        if output and "v2" in output.strip():
            logger.info("'Docker Compose' found using Docker CLI.")
            compose_cmd_prefix.extend(["docker", "compose"])
            return compose_cmd_prefix

        if shutil.which("docker-compose") is not None:
            logger.info("'Docker Compose' found using Docker Compose CLI.")
            compose_cmd_prefix.extend(["docker-compose"])
            return compose_cmd_prefix

        raise ImportError(
            "Docker Compose is not installed. "
            "Local Mode features will not work without docker compose. "
            "For more information on how to install 'docker compose', please, see "
            "https://docs.docker.com/compose/install/"
        )
