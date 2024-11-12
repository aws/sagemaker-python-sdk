"""This module is for Triton Python backend."""

from __future__ import absolute_import
import os
import logging
import ssl
from pathlib import Path
import platform

import triton_python_backend_utils as pb_utils
import cloudpickle
from sagemaker.serve.validations.check_integrity import perform_integrity_check

logger = logging.getLogger(__name__)

# Otherwise it will complain SSL: CERTIFICATE_VERIFY_FAILED
# When trying to download models from torchvision
ssl._create_default_https_context = ssl._create_unverified_context

TRITON_MODEL_DIR = os.getenv("TRITON_MODEL_DIR")


class TritonPythonModel:
    """A class for Triton Python Backend"""

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Placeholder docstring"""
        return auto_complete_model_config

    def initialize(self, args: dict) -> None:
        """Placeholder docstring"""
        serve_path = Path(TRITON_MODEL_DIR).joinpath("serve.pkl")
        with open(str(serve_path), mode="rb") as f:
            inference_spec, schema_builder = cloudpickle.load(f)

        # TODO: HMAC signing for integrity check

        self.inference_spec = inference_spec
        self.schema_builder = schema_builder
        self.model = inference_spec.load(model_dir=TRITON_MODEL_DIR)

    def execute(self, requests):
        """Placeholder docstring"""
        responses = []

        for request in requests:
            input_ndarray = pb_utils.get_input_tensor_by_name(request, "input_1").as_numpy()
            converted_input = self.schema_builder.input_deserializer.deserialize(input_ndarray)
            output = self.inference_spec.invoke(input_object=converted_input, model=self.model)
            output_ndarray = self.schema_builder.output_serializer.serialize(output)
            response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("output_1", output_ndarray)]
            )
            responses.append(response)

        return responses


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
    serve_path = Path(TRITON_MODEL_DIR).joinpath("serve.pkl")
    metadata_path = Path(TRITON_MODEL_DIR).joinpath("metadata.json")
    with open(str(serve_path), "rb") as f:
        buffer = f.read()
    perform_integrity_check(buffer=buffer, metadata_path=metadata_path)


# on import, execute
_run_preflight_diagnostics()
