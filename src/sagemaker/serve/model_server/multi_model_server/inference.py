"""This module is for SageMaker inference.py."""

from __future__ import absolute_import
import os
import io
import cloudpickle
import shutil
import platform
from pathlib import Path
from functools import partial
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.validations.check_integrity import perform_integrity_check
import logging

logger = logging.getLogger(__name__)

inference_spec = None
schema_builder = None
SHARED_LIBS_DIR = Path(__file__).parent.parent.joinpath("shared_libs")
SERVE_PATH = Path(__file__).parent.joinpath("serve.pkl")
METADATA_PATH = Path(__file__).parent.joinpath("metadata.json")


def model_fn(model_dir):
    """Overrides default method for loading a model"""
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
    """Deserializes the bytes that were received from the model server"""
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
        logger.error("Encountered error: %s in deserialize_response." % e)
        raise Exception("Encountered error in deserialize_request.") from e


def predict_fn(input_data, predict_callable):
    """Invokes the model that is taken in by model server"""
    return predict_callable(input_data)


def output_fn(predictions, accept_type):
    """Prediction is serialized to bytes and sent back to the customer"""
    try:
        if hasattr(schema_builder, "custom_output_translator"):
            return schema_builder.custom_output_translator.serialize(predictions, accept_type)
        else:
            return schema_builder.output_serializer.serialize(predictions)
    except Exception as e:
        logger.error("Encountered error: %s in serialize_response." % e)
        raise Exception("Encountered error in serialize_response.") from e


def _run_preflight_diagnostics():
    _py_vs_parity_check()
    _pickle_file_integrity_check()


def _py_vs_parity_check():
    container_py_vs = platform.python_version()
    local_py_vs = os.getenv("LOCAL_PYTHON")

    if not local_py_vs or container_py_vs.split(".")[1] != local_py_vs.split(".")[1]:
        logger.warning(
            f"The local python version {local_py_vs} differs from the python version "
            f"{container_py_vs} on the container. Please align the two to avoid unexpected behavior"
        )


def _pickle_file_integrity_check():
    with open(SERVE_PATH, "rb") as f:
        buffer = f.read()

    perform_integrity_check(buffer=buffer, metadata_path=METADATA_PATH)


# on import, execute
_run_preflight_diagnostics()
