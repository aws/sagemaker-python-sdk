"""This module is for SageMaker inference.py."""

from __future__ import absolute_import
import os
import io
import json
import cloudpickle
import shutil
import platform
from pathlib import Path
from sagemaker.serve.validations.check_integrity import perform_integrity_check
import logging

logger = logging.getLogger(__name__)

schema_builder = None
SHARED_LIBS_DIR = Path(__file__).parent.parent.joinpath("shared_libs")
SERVE_PATH = Path(__file__).parent.joinpath("serve.pkl")
METADATA_PATH = Path(__file__).parent.joinpath("metadata.json")


def input_handler(data, context):
    """Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    read_data = data.read()
    deserialized_data = None
    try:
        if hasattr(schema_builder, "custom_input_translator"):
            deserialized_data = schema_builder.custom_input_translator.deserialize(
                io.BytesIO(read_data), context.request_content_type
            )
        else:
            deserialized_data = schema_builder.input_deserializer.deserialize(
                io.BytesIO(read_data), context.request_content_type
            )
    except Exception as e:
        logger.error("Encountered error: %s in deserialize_request." % e)
        raise Exception("Encountered error in deserialize_request.") from e

    try:
        return json.dumps({"instances": _convert_for_serialization(deserialized_data)})
    except Exception as e:
        logger.error(
            "Encountered error: %s in deserialize_request. "
            "Deserialized data is not json serializable. " % e
        )
        raise Exception("Encountered error in deserialize_request.") from e


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))

    response_content_type = context.accept_header
    prediction = data.content
    try:
        prediction_dict = json.loads(prediction.decode("utf-8"))
        if hasattr(schema_builder, "custom_output_translator"):
            return (
                schema_builder.custom_output_translator.serialize(
                    prediction_dict["predictions"], response_content_type
                ),
                response_content_type,
            )
        else:
            return schema_builder.output_serializer.serialize(prediction), response_content_type
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


def _set_up_schema_builder():
    """Sets up the schema_builder object."""
    global schema_builder
    with open(SERVE_PATH, "rb") as f:
        schema_builder = cloudpickle.load(f)


def _set_up_shared_libs():
    """Sets up the shared libs path."""
    if SHARED_LIBS_DIR.exists():
        # before importing, place dynamic linked libraries in shared lib path
        shutil.copytree(SHARED_LIBS_DIR, "/lib", dirs_exist_ok=True)


def _convert_for_serialization(deserialized_data):
    """Attempt to convert non-serializable objects to a serializable form.

    Args:
        deserialized_data: The object to convert.

    Returns:
        The converted object if it was not originally serializable, otherwise the original object.
    """
    import numpy as np
    import pandas as pd

    if isinstance(deserialized_data, np.ndarray):
        return deserialized_data.tolist()
    elif isinstance(deserialized_data, pd.DataFrame):
        return deserialized_data.to_dict(orient="list")
    elif isinstance(deserialized_data, pd.Series):
        return deserialized_data.tolist()
    return deserialized_data


# on import, execute
_run_preflight_diagnostics()
_set_up_schema_builder()
_set_up_shared_libs()
