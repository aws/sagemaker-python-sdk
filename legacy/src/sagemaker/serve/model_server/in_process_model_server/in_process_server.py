"""Module for In_process Serving"""

from __future__ import absolute_import

import requests
import logging
from sagemaker.serve.utils.exceptions import LocalModelInvocationException
from sagemaker.base_predictor import PredictorBase

logger = logging.getLogger(__name__)


class InProcessServing:
    """In Process Mode server instance"""

    def _start_serving(self):
        """Initializes the start of the server"""
        from sagemaker.serve.model_server.in_process_model_server.app import InProcessServer

        self.server = InProcessServer(
            inference_spec=self.inference_spec, model=self.model, schema_builder=self.schema_builder
        )
        self.server.start_server()

    def _stop_serving(self):
        """Stops the server"""
        self.server.stop_server()

    def _invoke_serving(self, request: object, content_type: str, accept: str):
        """Placeholder docstring"""
        try:
            response = requests.post(
                f"http://{self.server.host}:{self.server.port}/invoke",
                data=request,
                headers={"Content-Type": content_type, "Accept": accept},
                timeout=600,
            )
            response.raise_for_status()

            return response.content
        except Exception as e:
            if "Connection refused" in str(e):
                raise Exception(
                    "Unable to send request to the local server: Connection refused."
                ) from e
            raise Exception("Unable to send request to the local container server %s", str(e))

    def _deep_ping(self, predictor: PredictorBase):
        """Sends a deep ping to ensure prediction"""
        healthy = False
        response = None
        try:
            response = predictor.predict(self.schema_builder.sample_input)
            healthy = response is not None
            # pylint: disable=broad-except
        except Exception as e:
            if "422 Client Error: Unprocessable Entity for url" in str(e):
                raise LocalModelInvocationException(str(e))

        return healthy, response
