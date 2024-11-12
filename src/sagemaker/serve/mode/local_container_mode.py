"""Module that defines the LocalContainerMode class"""

from __future__ import absolute_import
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, Type
import base64
import time
import subprocess
import docker

from sagemaker.base_predictor import PredictorBase
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
from sagemaker.session import Session

logger = logging.getLogger(__name__)

_PING_HEALTH_CHECK_INTERVAL_SEC = 5

_PING_HEALTH_CHECK_FAIL_MSG = (
    "Container did not pass the ping health check. "
    + "Please increase container_timeout_seconds or review your inference code."
)


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
        predictor: PredictorBase,
        env_vars: Dict[str, str] = None,
        model_path: str = None,
        jumpstart: bool = False,
    ):
        """Placeholder docstring"""

        self._pull_image(image=image)

        self.destroy_server()

        logger.info("Waiting for model server %s to start up...", self.model_server)

        if self.model_server == ModelServer.TRITON:
            self._start_triton_server(
                docker_client=self.client,
                model_path=model_path if model_path else self.model_path,
                image_uri=image,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
            self._ping_container = self._triton_deep_ping
        elif self.model_server == ModelServer.DJL_SERVING:
            self._start_djl_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
            self._ping_container = self._djl_deep_ping
        elif self.model_server == ModelServer.TORCHSERVE:
            self._start_torch_serve(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
            self._ping_container = self._torchserve_deep_ping
        elif self.model_server == ModelServer.TGI:
            self._start_tgi_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
                jumpstart=jumpstart,
            )
            self._ping_container = self._tgi_deep_ping
        elif self.model_server == ModelServer.MMS:
            self._start_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
            self._ping_container = self._multi_model_server_deep_ping
        elif self.model_server == ModelServer.TENSORFLOW_SERVING:
            self._start_tensorflow_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
            self._ping_container = self._tensorflow_serving_deep_ping
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
            self._ping_container = tei_serving._tei_deep_ping
            self._invoke_serving = tei_serving._invoke_tei_serving

        # allow some time for container to be ready
        time.sleep(10)

        log_generator = self.container.logs(follow=True, stream=True)
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

            healthy, response = self._ping_container(predictor)
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
        """Placeholder docstring"""
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
            subprocess.run(login_command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.warning("Unable to login to ecr: %s", e)

        self.client = docker.from_env()
        try:
            logger.info("Pulling image %s from repository...", image)
            self.client.images.pull(image)
        except docker.errors.NotFound as e:
            raise ValueError("Could not find remote image to pull") from e
