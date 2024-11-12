"""Validates the integrity of pickled file with HMAC signing."""

from __future__ import absolute_import
import secrets
import hmac
import hashlib
import os
import logging
import subprocess
import sys
import platform
from collections import defaultdict
from pathlib import Path
import tempfile
import yaml

from sagemaker.model import Model
from sagemaker.session import Session
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.s3_utils import parse_s3_url
from sagemaker.utils import create_tar_file

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


def upload_to_s3(s3_path: str, local_path: str, session: Session) -> str:
    """Sync local path to S3 path using sagemaker session"""
    if not session or not s3_path or not local_path:
        logger.info("Skipping upload to s3. Missing required parameters")
        return ""

    bucket, key_prefix = parse_s3_url(url=s3_path)
    files = [os.path.join(local_path, name) for name in os.listdir(local_path)]
    tmp = tempfile.mkdtemp(dir="/tmp")

    tar_file = create_tar_file(files, os.path.join(tmp, "model.tar.gz"))
    s3_model_url = session.upload_data(
        path=os.path.join(tmp, "model.tar.gz"), bucket=bucket, key_prefix=key_prefix
    )
    os.remove(tar_file)
    logger.info("Model file has uploaded to s3")
    return s3_model_url


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

    try:
        import pigar

        pigar.__version__  # pylint: disable=W0104
    except ModuleNotFoundError:
        logger.warning(
            "pigar module is not installed in python environment, "
            "dependency generation may be incomplete"
            "Checkout the instructions on the installation page of its repo: "
            "https://github.com/damnever/pigar "
            "And follow the ones that match your environment."
            "Please note that you may need to restart your runtime after installation."
        )
        import sagemaker

        sagemaker_dependency = f"{sagemaker.__package__}=={sagemaker.__version__}"
        with open(requirements_path, "w") as f:
            f.write(sagemaker_dependency)
        return

    command = ["pigar", "gen", "-f", str(Path(requirements_path)), str(os.getcwd())]
    logging.info("Running command: %s", " ".join(command))

    subprocess.run(command, check=True, capture_output=True)
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


def get_image_url(model_server: str) -> str:
    """Placeholder docstring"""
    supported_image = {
        "DJLServing": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.25.0-cpu-full"
    }
    if model_server in supported_image:
        return supported_image.get(model_server)
    raise Exception("Model server specified is not supported in SageMaker save format")
