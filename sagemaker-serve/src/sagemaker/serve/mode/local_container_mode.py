"""Module that defines the LocalContainerMode class"""

from __future__ import absolute_import
from pathlib import Path
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Type
import base64
import time
import subprocess
import docker

from sagemaker.core.local.utils import check_for_studio

from sagemaker.serve.model_server.tensorflow_serving.server import LocalTensorflowServing
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.logging_agent import pull_logs
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.utils.exceptions import LocalDeepPingException
from sagemaker.serve.model_server.torchserve.server import LocalTorchServe
from sagemaker.serve.model_server.djl_serving.server import LocalDJLServing
from sagemaker.serve.model_server.triton.server import LocalTritonServer
from sagemaker.serve.model_server.tgi.server import LocalTgiServing
from sagemaker.serve.model_server.tei.server import LocalTeiServing
from sagemaker.serve.model_server.multi_model_server.server import LocalMultiModelServer
from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)

_PING_HEALTH_CHECK_INTERVAL_SEC = 5

_PING_HEALTH_CHECK_FAIL_MSG = (
    "Container did not pass the ping health check. "
    + "Please increase container_timeout_seconds or review your inference code."
)

STUDIO_DOCKER_SOCKET_PATHS = [
    "/docker/proxy/docker.sock",
    "/var/run/docker.sock",
]


def _get_docker_client():
    """Get a Docker client, handling SageMaker Studio's non-standard socket path."""
    if os.environ.get("DOCKER_HOST"):
        return docker.from_env()
    try:
        if check_for_studio():
            for socket_path in STUDIO_DOCKER_SOCKET_PATHS:
                if os.path.exists(socket_path):
                    return docker.DockerClient(base_url=f"unix://{socket_path}")
    except (NotImplementedError, Exception):
        pass
    return docker.from_env()


class LocalContainerMode(
    LocalTorchServe,
    LocalDJLServing,
    LocalTritonServer,
    LocalTgiServing,
    LocalMultiModelServer,
    LocalTensorflowServing,
):
    """A class that holds methods to deploy model to a container in local environment"""

    def __init__(
        self,
        model_server: ModelServer,
        inference_spec: Type[InferenceSpec],
        schema_builder: Type[SchemaBuilder],
        session: Session,
        model_path: str = None,
        env_vars: Dict = None,
    ):
        # pylint: disable=bad-super-call
        super().__init__()
        super(LocalDJLServing, self).__init__()
        super(LocalTritonServer, self).__init__()

        self.inference_spec = inference_spec
        self.model_path = model_path
        self.env_vars = env_vars
        self.session = session
        self.schema_builder = schema_builder
        self.ecr = session.boto_session.client("ecr")
        self.model_server = model_server
        self.client = None
        self.container = None
        self.secret_key = None
        self._ping_container = None
        self._invoke_serving = None

    def load(self, model_path: str = None):
        """Placeholder docstring"""
        path = Path(model_path if model_path else self.model_path)
        if not path.exists():
            raise Exception("model_path does not exist")
        if not path.is_dir():
            raise Exception("model_path is not a valid directory")

        return self.inference_spec.load(str(path))

    def prepare(self):
        """Placeholder docstring"""

    def create_server(
        self,
        image: str,
        container_timeout_seconds: int,
        secret_key: str,
        container_config: Dict,
        ping_fn = None,
        env_vars: Dict[str, str] = None,
        model_path: str = None,
        jumpstart: bool = False,
    ):
        """Placeholder docstring"""

        self._pull_image(image=image)

        self.destroy_server()

        logger.info("Waiting for model server %s to start up...", self.model_server)

        self._ping_container = ping_fn or self._ping_container
        if self.model_server == ModelServer.TRITON:
            self._start_triton_server(
                docker_client=self.client,
                model_path=model_path if model_path else self.model_path,
                image_uri=image,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
        elif self.model_server == ModelServer.DJL_SERVING:
            self._start_djl_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
        elif self.model_server == ModelServer.TORCHSERVE:
            self._start_torch_serve(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
        elif self.model_server == ModelServer.TGI:
            self._start_tgi_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
                jumpstart=jumpstart,
            )
        elif self.model_server == ModelServer.MMS:
            self._start_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
        elif self.model_server == ModelServer.TENSORFLOW_SERVING:
            self._start_tensorflow_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
        elif self.model_server == ModelServer.TEI:
            tei_serving = LocalTeiServing()
            tei_serving._start_tei_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
            tei_serving.schema_builder = self.schema_builder
            self.container = tei_serving.container
            self._invoke_serving = tei_serving._invoke_tei_serving

        # allow some time for container to be ready
        time.sleep(10)

        log_generator = self.container.logs(follow=True, stream=True)
        container_timeout_seconds = 1200
        time_limit = datetime.now() + timedelta(seconds=container_timeout_seconds)
        healthy = False
        while True:
            now = datetime.now()
            final_pull = now > time_limit
            pull_logs(
                (x.decode("UTF-8").rstrip() for x in log_generator),
                log_generator.close,
                datetime.now() + timedelta(seconds=_PING_HEALTH_CHECK_INTERVAL_SEC),
                now > time_limit,
            )

            if final_pull:
                break

            # allow some time for container to be ready
            time.sleep(10)
            healthy, response = self._ping_container()
            if healthy:
                logger.debug("Ping health check has passed. Returned %s", str(response))
                break

        if not healthy:
            raise LocalDeepPingException(_PING_HEALTH_CHECK_FAIL_MSG)

    def destroy_server(self):
        """Placeholder docstring"""
        if self.container:
            try:
                logger.debug("Stopping currently running container...")
                self.container.kill()
            except docker.errors.APIError as exc:
                if exc.response.status_code < 400 or exc.response.status_code > 499:
                    raise Exception("Error encountered when cleaning up local container") from exc
            self.container = None

    def _pull_image(self, image: str):
        """Pull image with proper error handling and early failure detection."""
        
        # Check if Docker is available first
        try:
            self.client = _get_docker_client()
            self.client.ping()  # Test Docker connection
        except Exception as e:
            raise RuntimeError(
                f"Docker is not available or not running. Please ensure Docker is installed and running. "
                f"Error: {e}"
            ) from e
        
        # Handle ECR authentication for ECR images
        if self._is_ecr_image(image):
            try:
                encoded_token = (
                    self.ecr.get_authorization_token()
                    .get("authorizationData")[0]
                    .get("authorizationToken")
                )
                decoded_token = base64.b64decode(encoded_token).decode("utf-8")
                username, password = decoded_token.split(":")
                ecr_uri = image.split("/")[0]
                login_command = ["docker", "login", "-u", username, "-p", password, ecr_uri]
                
                result = subprocess.run(login_command, check=True, capture_output=True, text=True)
                logger.info("Successfully authenticated with ECR")
                
            except subprocess.CalledProcessError as e:
                error_msg = f"ECR authentication failed: {e.stderr if e.stderr else str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"ECR authentication error: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        # Pull the image
        try:
            logger.info("Pulling image %s from repository...", image)
            self.client.images.pull(image)
            logger.info("Successfully pulled image %s", image)
        except docker.errors.NotFound as e:
            raise ValueError(f"Could not find image '{image}' in repository") from e
        except docker.errors.APIError as e:
            raise RuntimeError(f"Failed to pull image '{image}': {e}") from e

    def _is_ecr_image(self, image: str) -> bool:
        """Check if image is from ECR."""
        return ".dkr.ecr." in image and ".amazonaws.com" in image

