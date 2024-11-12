"""FastAPI requests"""

from __future__ import absolute_import

import asyncio
import logging
import threading
from typing import Optional


logger = logging.getLogger(__name__)


try:
    import uvicorn
except ImportError:
    logger.error("Unable to import uvicorn, check if uvicorn is installed.")


try:
    from transformers import pipeline
except ImportError:
    logger.error("Unable to import transformers, check if transformers is installed.")


try:
    from fastapi import FastAPI, Request, APIRouter
except ImportError:
    logger.error("Unable to import fastapi, check if fastapi is installed.")


class InProcessServer:
    """Placeholder docstring"""

    def __init__(self, model_id: Optional[str] = None, task: Optional[str] = None):
        self._thread = None
        self._loop = None
        self._stop_event = asyncio.Event()
        self._router = APIRouter()
        self._model_id = model_id
        self._task = task
        self.server = None
        self.port = None
        self.host = None
        # TODO: Pick up device automatically.
        self._generator = pipeline(task, model=model_id, device="cpu")

        # pylint: disable=unused-variable
        @self._router.post("/generate")
        async def generate_text(prompt: Request):
            """Placeholder docstring"""
            str_prompt = await prompt.json()
            str_prompt = str_prompt["inputs"] if "inputs" in str_prompt else str_prompt

            generated_text = self._generator(
                str_prompt, max_length=30, num_return_sequences=1, truncation=True
            )
            return generated_text

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
