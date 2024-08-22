"""Module that defines the InProcessMode class"""

from __future__ import absolute_import

from pathlib import Path
import logging
from typing import Dict, Type
import time
from datetime import datetime, timedelta

from sagemaker.base_predictor import PredictorBase
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.utils.exceptions import InProcessDeepPingException
from sagemaker.serve.model_server.multi_model_server.server import InProcessMultiModelServer
from sagemaker.session import Session

logger = logging.getLogger(__name__)

_PING_HEALTH_CHECK_FAIL_MSG = (
    "Ping health check did not pass. "
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
        self.model_server = model_server
        self._ping_local_server = None

    def load(self, model_path: str = None):
        """Loads model path, checks that path exists"""
        path = Path(model_path if model_path else self.model_path)
        if not path.exists():
            raise ValueError("model_path does not exist")
        if not path.is_dir():
            raise ValueError("model_path is not a valid directory")

        return self.inference_spec.load(str(path))

    def prepare(self):
        """Prepares the server"""

    def create_server(
        self,
        predictor: PredictorBase,
    ):
        """Creating the server and checking ping health."""
        logger.info("Waiting for model server %s to start up...", self.model_server)

        if self.model_server == ModelServer.MMS:
            self._ping_local_server = self._multi_model_server_deep_ping
            self._start_serving()

        # allow some time for server to be ready.
        time.sleep(1)

        time_limit = datetime.now() + timedelta(seconds=5)
        healthy = True
        while True:
            final_pull = datetime.now() > time_limit
            if final_pull:
                break

            healthy, response = self._ping_local_server(predictor)
            if healthy:
                logger.debug("Ping health check has passed. Returned %s", str(response))
                break

        time.sleep(1)

        if not healthy:
            raise InProcessDeepPingException(_PING_HEALTH_CHECK_FAIL_MSG)

    def destroy_server(self):
        """Placeholder docstring"""
        self._stop_serving()
