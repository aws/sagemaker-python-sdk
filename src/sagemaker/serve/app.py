"""FastAPI requests"""

from __future__ import absolute_import

import logging
from fastapi import FastAPI, Request


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Transformers In Process Server",
    version="1.0",
    description="A simple server",
)


@app.get("/")
def read_root():
    """Placeholder docstring"""
    return {"Hello": "World"}


@app.get("/generate")
async def generate_text(prompt: Request):
    """Placeholder docstring"""
    logger.info("Generating Text....")

    str_prompt = await prompt.json()

    logger.info(str_prompt)

    generated_text = generator(str_prompt, max_length=30, num_return_sequences=5, truncation=True)
    return generated_text[0]["generated_text"]


try:
    from transformers import pipeline

    generator = pipeline("text-generation", model="gpt2")

except ImportError:
    logger.error(
        "To enable in_process mode for Transformers install transformers from HuggingFace hub"
    )

try:
    import uvicorn

except ImportError:
    logger.error("To enable in_process mode for Transformers install uvicorn from HuggingFace hub")


@app.post("/post")
def post(payload: dict):
    """Placeholder docstring"""
    return payload


async def main():
    """Running server locally with uvicorn"""
    logger.info("Running")
    config = uvicorn.Config(
        "sagemaker.app:app",
        host="127.0.0.1",
        port=9007,
        log_level="info",
        loop="asyncio",
        reload=True,
        workers=3,
        use_colors=True,
    )
    server = uvicorn.Server(config)
    logger.info("Waiting for a connection...")
    await server.serve()
