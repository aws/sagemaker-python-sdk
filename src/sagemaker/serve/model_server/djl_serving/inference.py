"""This module is for SageMaker inference.py."""

from __future__ import absolute_import
import io
import yaml
import logging

from pathlib import Path
from djl_python import Input
from djl_python import Output


class DJLPythonInference(object):
    """A class for DJL inference"""

    def __init__(self) -> None:
        self.inference_spec = None
        self.model_dir = None
        self.model = None
        self.schema_builder = None
        self.inferenceSpec = None
        self.metadata = None
        self.default_serializer = None
        self.default_deserializer = None
        self.initialized = False

    def load_yaml(self, path: str):
        """Placeholder docstring"""
        with open(path, mode="r") as file:
            return yaml.full_load(file)

    def load_metadata(self):
        """Placeholder docstring"""
        metadata_path = Path(self.model_dir).joinpath("metadata.yaml")
        return self.load_yaml(metadata_path)

    def load_and_validate_pkl(self, path, hash_tag):
        """Placeholder docstring"""

        import os
        import hmac
        import hashlib
        import cloudpickle

        with open(path, mode="rb") as file:
            buffer = file.read()
            secret_key = os.getenv("SAGEMAKER_SERVE_SECRET_KEY")
            stored_hash_tag = hmac.new(
                secret_key.encode(), msg=buffer, digestmod=hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(stored_hash_tag, hash_tag):
                raise Exception("Object is not valid: " + path)

        with open(path, mode="rb") as file:
            return cloudpickle.load(file)

    def load(self):
        """Detecting for inference spec and loading model"""
        self.metadata = self.load_metadata()
        if "InferenceSpec" in self.metadata:
            inference_spec_path = (
                Path(self.model_dir).joinpath(self.metadata.get("InferenceSpec")).absolute()
            )
            self.inference_spec = self.load_and_validate_pkl(
                inference_spec_path, self.metadata.get("InferenceSpecHMAC")
            )

        # Load model
        if self.inference_spec:
            self.model = self.inference_spec.load(self.model_dir)
        else:
            raise Exception(
                "SageMaker model format does not support model type: "
                + self.metadata.get("ModelType")
            )

    def initialize(self, properties):
        """Initialize SageMaker service, loading model and inferenceSpec"""
        self.model_dir = properties.get("model_dir")
        self.load()
        self.initialized = True
        logging.info("SageMaker saved format entry-point is applied, service is initilized")

    def preprocess_djl(self, inputs: Input):
        """Placeholder docstring"""
        content_type = inputs.get_property("content-type")
        logging.info(f"Content-type is: {content_type}")
        if self.schema_builder:
            logging.info("Customized input deserializer is applied")
            try:
                if hasattr(self.schema_builder, "custom_input_translator"):
                    return self.schema_builder.custom_input_translator.deserialize(
                        io.BytesIO(inputs.get_as_bytes()), content_type
                    )
                else:
                    raise Exception("No custom input translator in cutomized schema builder.")
            except Exception as e:
                raise Exception("Encountered error in deserialize_request.") from e
        elif self.default_deserializer:
            return self.default_deserializer.deserialize(
                io.BytesIO(inputs.get_as_bytes()), content_type
            )

    def postproces_djl(self, output):
        """Placeholder docstring"""
        if self.schema_builder:
            logging.info("Customized output serializer is applied")
            try:
                if hasattr(self.schema_builder, "custom_output_translator"):
                    return self.schema_builder.custom_output_translator.serialize(output)
                else:
                    raise Exception("No custom output translator in cutomized schema builder.")
            except Exception as e:
                raise Exception("Encountered error in serialize_response.") from e
        elif self.default_serializer:
            return self.default_serializer.serialize(output)

    def inference(self, inputs: Input):
        """Detects if inference spec used, returns output accordingly"""
        processed_input = self.preprocess_djl(inputs=inputs)
        if self.inference_spec:
            output = self.inference_spec.invoke(processed_input, self.model)
        else:
            raise Exception(
                "SageMaker model format does not support model type: "
                + self.metadata.get("ModelType")
            )
        processed_output = self.postproces_djl(output=output)
        output_data = Output()
        return output_data.add(processed_output)


_service = DJLPythonInference()


def handle(inputs: Input) -> Output:
    """Placeholder docstring"""
    if not _service.initialized:
        properties = inputs.get_properties()
        _service.initialize(properties)

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
