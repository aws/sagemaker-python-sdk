"""Module that defines the InProcessMode class"""

from __future__ import absolute_import
from pathlib import Path
import logging
from typing import Dict, Type
import time

from sagemaker.base_predictor import PredictorBase
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.utils.exceptions import LocalDeepPingException
from sagemaker.serve.model_server.multi_model_server.server import InProcessMultiModelServer
from sagemaker.session import Session

logger = logging.getLogger(__name__)

_PING_HEALTH_CHECK_INTERVAL_SEC = 5

_PING_HEALTH_CHECK_FAIL_MSG = (
    "Container did not pass the ping health check. "
    + "Please increase container_timeout_seconds or review your inference code."
)


class InProcessMode(
    InProcessMultiModelServer,
):
    """A class that holds methods to deploy model to a container in process environment"""

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
        secret_key: str,
        predictor: PredictorBase,
        env_vars: Dict[str, str] = None,
        model_path: str = None,
    ):
        """Placeholder docstring"""

        # self._pull_image(image=image)

        # self.destroy_server()

        logger.info("Waiting for model server %s to start up...", self.model_server)

        if self.model_server == ModelServer.MMS:
            self._start_serving(
                client=self.client,
                image=image,
                model_path=model_path if model_path else self.model_path,
                secret_key=secret_key,
                env_vars=env_vars if env_vars else self.env_vars,
            )
            logger.info("Starting PING")
            self._ping_container = self._multi_model_server_deep_ping

        while True:
            time.sleep(10)

            healthy, response = self._ping_container(predictor)
            if healthy:
                logger.debug("Ping health check has passed. Returned %s", str(response))
                break

        if not healthy:
            raise LocalDeepPingException(_PING_HEALTH_CHECK_FAIL_MSG)
