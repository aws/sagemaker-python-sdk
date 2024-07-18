"""FastAPI requests"""

from __future__ import absolute_import

import logging
from transformers import pipeline
from fastapi import FastAPI
import uvicorn

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


@app.post("/generate")
def generate_text(prompt: str, max_length=500, num_return_sequences=1):
    """Placeholder docstring"""
    logger.info("Generating Text....")

    generated_text = generator(
        prompt, max_length=max_length, num_return_sequences=num_return_sequences
    )
    return generated_text[0]["generated_text"]


generator = pipeline("text-generation", model="gpt2")


@app.post("/post")
def post(prompt: str):
    """Placeholder docstring"""
    return prompt


async def main():
    """Running server locally with uvicorn"""
    logger.info("Running")
    config = uvicorn.Config(
        "sagemaker.app:app", host="0.0.0.0", port=8080, log_level="info", loop="asyncio"
    )
    server = uvicorn.Server(config)
    await server.serve()
