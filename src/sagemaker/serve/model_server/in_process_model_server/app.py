"""FastAPI requests"""

from __future__ import absolute_import

import asyncio
import io
import logging
import threading
from typing import Optional

from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder

logger = logging.getLogger(__name__)


try:
    import uvicorn
except ImportError:
    logger.error("Unable to import uvicorn, check if uvicorn is installed.")


try:
    from fastapi import FastAPI, Request, APIRouter
except ImportError:
    logger.error("Unable to import fastapi, check if fastapi is installed.")


class InProcessServer:
    """Generic In-Process Server for Serving Models using InferenceSpec"""

    def __init__(
        self,
        inference_spec: Optional[InferenceSpec] = None,
        schema_builder: Optional[SchemaBuilder] = None,
    ):
        self._thread = None
        self._loop = None
        self._stop_event = asyncio.Event()
        self._router = APIRouter()
        self.server = None
        self.port = None
        self.host = None
        self.inference_spec = inference_spec
        self.schema_builder = schema_builder
        self._load_model = self.inference_spec.load(model_dir=None)

        @self._router.post("/invoke")
        async def invoke(request: Request):
            """Generate text based on the provided prompt"""

            request_header = request.headers
            request_body = await request.body()
            content_type = request_header.get("Content-Type", None)
            input_data = schema_builder.input_deserializer.deserialize(
                io.BytesIO(request_body), content_type[0]
            )
            logger.debug(f"Received request: {input_data}")
            response = self.inference_spec.invoke(input_data, self._load_model)
            return response

        self._create_server()

    def _create_server(self):
        """Placeholder docstring"""
        app = FastAPI()
        app.include_router(self._router)

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=9007,
            log_level="info",
            loop="asyncio",
            reload=True,
            use_colors=True,
        )

        self.server = uvicorn.Server(config)
        self.host = config.host
        self.port = config.port

    def start_server(self):
        """Starts the uvicorn server."""
        if not (self._thread and self._thread.is_alive()):
            logger.info("Waiting for a connection...")
            self._thread = threading.Thread(target=self._start_run_async_in_thread, daemon=True)
            self._thread.start()

    def stop_server(self):
        """Destroys the uvicorn server."""
        # TODO: Implement me.

    def _start_run_async_in_thread(self):
        """Placeholder docstring"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())

    async def _serve(self):
        """Placeholder docstring"""
        await self.server.serve()
