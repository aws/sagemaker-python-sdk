"""Validates the integrity of pickled file with HMAC signing."""
from __future__ import absolute_import
import secrets
import hmac
import hashlib
import os
import logging
import sys
import platform
from collections import defaultdict
from pathlib import Path
import yaml

from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.serve.builder.schema_builder import SchemaBuilder

logger = logging.getLogger(__name__)


def save_pkl(path: str, obj) -> None:
    """Placeholder docstring"""
    with open(path, mode="wb") as file:
        import cloudpickle

        cloudpickle.dump(obj, file)


def save_yaml(path: str, obj: dict) -> None:
    """Placeholder docstring"""
    with open(path, mode="w") as file:
        yaml.dump(obj, file)


def upload_to_s3(s3_path: str, local_path: str, session: Session) -> bool:
    """Sync local path to S3 path using sagemaker session"""
    if not session or not s3_path or not local_path:
        logger.info("Skipping upload to s3. Missing required parameters")
        return False

    session.upload_data(path=local_path, key_prefix=s3_path)
    return True


def capture_schema(schema_path: str, schema_format: str, schema_builder: SchemaBuilder):
    """Placeholder docstring"""
    logging.info("Capturing schema...")

    marshalling_map = schema_builder.generate_marshalling_map()

    if marshalling_map["custom_input_translator"] or marshalling_map["custom_output_translator"]:
        logger.info("Saving custom input/output translator in pkl format")
        path = Path(f"{schema_path}.{schema_format}")
        save_pkl(path, schema_builder)
    else:
        logger.info("Saving marshalling functions in yaml format")
        schema_format = "yaml"
        path = Path(f"{schema_path}.{schema_format}")
        save_yaml(path, marshalling_map)
    logging.info("Schema captured successfully")

    return schema_path, schema_format


def capture_dependencies(requirements_path: str):
    """Placeholder docstring"""
    logger.info("Capturing dependencies...")

    command = f"pigar gen -f {Path(requirements_path)} {os.getcwd()}"
    logging.info("Running command %s", command)

    os.system(command)
    logger.info("Dependencies captured successfully")


def capture_optimization_metadata(model: Model, framework) -> dict:
    """Placeholder docstring"""
    logging.info("Capturing optimization metadata...")
    # get size of the model
    model_size = sys.getsizeof(model)
    # TODO: We may need to use framework specific way to get the model size
    # example torchsummary for pytorch etc.
    # need more dive deep into this
    optimizer_metadata = defaultdict(str)
    optimizer_metadata["ModelSizeInMemory"] = model_size

    if framework == "pytorch":
        # TODO: get input shape if pytorch
        # self.optimizer_metadata['input-shape'] = self.schema_builder.input_schema.shape
        True  # pylint: disable=W0104

    logging.info("Optimization metadata captured successfully")
    return optimizer_metadata


def generate_secret_key(nbytes: int = 32) -> str:
    """Generates secret key"""
    return secrets.token_hex(nbytes)


def compute_hash(buffer: bytes, secret_key: str) -> str:
    """Compute hash value using HMAC"""
    return hmac.new(secret_key.encode(), msg=buffer, digestmod=hashlib.sha256).hexdigest()


def perform_integrity_check(buffer: bytes):  # pylint: disable=W0613
    """Placeholder docstring"""
    # TODO: performing integrity check is
    # specific to the framework with will be coming soon


def detect_framework_and_its_versions(model: object) -> bool:
    """Placeholder docstring"""
    model_base = model.__class__.__base__

    if object == model_base:
        model_base = model.__class__

    framework_string = str(model_base)
    logger.info("Inferred Framework string: %s", framework_string)

    py_tuple = platform.python_version_tuple()

    logger.info("Inferred Python version tuple: %s", py_tuple)
    fw = ""
    vs = ""
    if "torch" in framework_string:
        fw = "pytorch"
        try:
            import torch

            vs = torch.__version__.split("+")[0]
        except ImportError:
            raise Exception("Unable to import torch, check if pytorch is installed")
    elif "xgb" in framework_string:
        fw = "xgboost"
        try:
            import xgboost

            vs = xgboost.__version__
        except ImportError:
            raise Exception("Unable to import xgboost, check if pytorch is installed")
    else:
        raise Exception("Unable to determine framework for tht base model" % framework_string)

    logger.info("Inferred framework and its version: %s %s", fw, vs)

    return [(fw, vs), py_tuple]
