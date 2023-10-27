"""This module is for Triton Python backend."""
from __future__ import absolute_import
import os
import logging
import ssl
from pathlib import Path

import triton_python_backend_utils as pb_utils
import cloudpickle

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
