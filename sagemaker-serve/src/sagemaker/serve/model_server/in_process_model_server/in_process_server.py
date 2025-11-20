"""Module for In_process Serving"""

from __future__ import absolute_import

import requests
import logging

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
