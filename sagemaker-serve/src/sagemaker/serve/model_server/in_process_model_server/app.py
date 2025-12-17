"""FastAPI requests"""

from __future__ import absolute_import

import asyncio
import io
import logging
import threading
import torch
from typing import Optional, Type

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
        model: Optional[str] = None,
        inference_spec: Optional[InferenceSpec] = None,
        schema_builder: Type[SchemaBuilder] = None,
        task: Optional[str] = None,
    ):
        self._thread = None
        self._loop = None
        self._shutdown_event = threading.Event()
        self._router = APIRouter()
        self._task = task
        self.server = None
        self.port = None
        self.host = None
        self.model = model
        self.inference_spec = inference_spec
        self.schema_builder = schema_builder

        if self.inference_spec:
            # Use inference_spec to load the model
            self._load_model = self.inference_spec.load(model_dir=None)
        elif isinstance(self.model, str):
            try:
                # Use transformers pipeline to load the model
                try:
                    from transformers import pipeline, Pipeline
                except ImportError:
                    logger.error(
                        "Unable to import transformers, check if transformers is installed."
                    )

                device = 0 if torch.cuda.is_available() else -1

                self._load_model = pipeline(task, model=self.model, device=device)
            except Exception:
                logger.info("Falling back to SentenceTransformer for model loading.")
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    logger.error(
                        "Unable to import sentence-transformers, check if sentence-transformers is installed."
                    )

                self._load_model = SentenceTransformer(self.model)
        else:
            raise ValueError("Either inference_spec or model must be provided.")

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
            if self.inference_spec:
                response = self.inference_spec.invoke(input_data, self._load_model)
            else:
                input_data = input_data["inputs"] if "inputs" in input_data else input_data
                if isinstance(self._load_model, Pipeline):
                    response = self._load_model(input_data, max_length=30, num_return_sequences=1)
                else:
                    embeddings = self._load_model.encode(input_data, normalize_embeddings=True)
                    response = {"embeddings": embeddings.tolist()}
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
        """Stops the Uvicorn server by setting the shutdown event."""
        if self._thread and self._thread.is_alive():
            logger.info("Shutting down the server...")
            self._shutdown_event.set()
            self.server.handle_exit(sig=0, frame=None)
            self._thread.join()

        logger.info("Server shutdown complete.")

    def _start_run_async_in_thread(self):
        """Placeholder docstring"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())

    async def _serve(self):
        """Placeholder docstring"""
        await self.server.serve()
