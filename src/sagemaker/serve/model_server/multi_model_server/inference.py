"""This module is for SageMaker inference.py."""

from __future__ import absolute_import
import io
import cloudpickle
import shutil
from pathlib import Path
from functools import partial
from sagemaker.serve.spec.inference_spec import InferenceSpec
import logging

logger = logging.getLogger(__name__)

inference_spec = None
schema_builder = None


def model_fn(model_dir):
    """Placeholder docstring"""
    shared_libs_path = Path(model_dir + "/shared_libs")

    if shared_libs_path.exists():
        # before importing, place dynamic linked libraries in shared lib path
        shutil.copytree(shared_libs_path, "/lib", dirs_exist_ok=True)

    serve_path = Path(__file__).parent.joinpath("serve.pkl")
    with open(str(serve_path), mode="rb") as file:
        global inference_spec, schema_builder
        obj = cloudpickle.load(file)
        if isinstance(obj[0], InferenceSpec): 
            inference_spec, schema_builder = obj
            
    if inference_spec:
        return partial(inference_spec.invoke, model=inference_spec.load(model_dir))


def input_fn(input_data, content_type):
    """Placeholder docstring"""
    try:
        if hasattr(schema_builder, "custom_input_translator"):
            return schema_builder.custom_input_translator.deserialize(
                io.BytesIO(input_data), content_type
            )
        else:
            return schema_builder.input_deserializer.deserialize(
                io.BytesIO(input_data), content_type[0]
            )
    except Exception as e:
        raise Exception("Encountered error in deserialize_request.") from e


def predict_fn(input_data, predict_callable):
    """Placeholder docstring"""
    return predict_callable(input_data)


def output_fn(predictions, accept_type):
    """Placeholder docstring"""
    try:
        if hasattr(schema_builder, "custom_output_translator"):
            return schema_builder.custom_output_translator.serialize(predictions, accept_type)
        else:
            return schema_builder.output_serializer.serialize(predictions)
    except Exception as e:
        logger.error("Encountered error: %s in serialize_response." % e)
        raise Exception("Encountered error in serialize_response.") from e


