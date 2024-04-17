"""Placeholder docstring"""

from __future__ import absolute_import
import logging

from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.marshalling.custom_payload_translator import CustomPayloadTranslator

__all__ = (
    "ModelBuilder",
    "InferenceSpec",
    "Mode",
    "SchemaBuilder",
    "CustomPayloadTranslator",
    "ModelServer",
)

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logging.Formatter("ModelBuilder: %(levelname)s:     %(message)s"))
logger.addHandler(streamHandler)
