"""This module is for SageMaker inference.py."""

from __future__ import absolute_import
import os
import io
import subprocess
import cloudpickle
import shutil
import platform
import importlib
from pathlib import Path
from functools import partial
import logging

logger = logging.getLogger(__name__)

inference_spec = None
native_model = None
schema_builder = None


def model_fn(model_dir):
    """Placeholder docstring"""
    from sagemaker.serve.spec.inference_spec import InferenceSpec
    from sagemaker.serve.detector.image_detector import (
        _detect_framework_and_version,
        _get_model_base,
    )
    from sagemaker.serve.detector.pickler import load_xgboost_from_json

    shared_libs_path = Path(model_dir + "/shared_libs")

    if shared_libs_path.exists():
        # before importing, place dynamic linked libraries in shared lib path
        shutil.copytree(shared_libs_path, "/lib", dirs_exist_ok=True)

    serve_path = Path(__file__).parent.joinpath("serve.pkl")
    mlflow_flavor = _get_mlflow_flavor()
    with open(str(serve_path), mode="rb") as file:
        global inference_spec, native_model, schema_builder
        obj = cloudpickle.load(file)
        if mlflow_flavor is not None:
            schema_builder = obj
            loaded_model = _load_mlflow_model(deployment_flavor=mlflow_flavor, model_dir=model_dir)
            return loaded_model if callable(loaded_model) else loaded_model.predict
        elif isinstance(obj[0], InferenceSpec):
            inference_spec, schema_builder = obj
        elif isinstance(obj[0], str) and obj[0] == "xgboost":
            model_class_name = os.getenv("MODEL_CLASS_NAME")
            model_save_path = Path(__file__).parent.joinpath("model.json")
            native_model = load_xgboost_from_json(
                model_save_path=str(model_save_path), class_name=model_class_name
            )
            schema_builder = obj[1]
        else:
            native_model, schema_builder = obj
    if native_model:
        framework, _ = _detect_framework_and_version(
            model_base=str(_get_model_base(model=native_model))
        )
        if framework == "pytorch":
            native_model.eval()
        return native_model if callable(native_model) else native_model.predict
    elif inference_spec:
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


def _run_preflight_diagnostics():
    install_package("sagemaker")
    install_package("boto3", "1.17.52")
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
    from sagemaker.serve.validations.check_integrity import perform_integrity_check

    with open("/opt/ml/model/code/serve.pkl", "rb") as f:
        buffer = f.read()

    metadeata_path = Path("/opt/ml/model/code/metadata.json")
    perform_integrity_check(buffer=buffer, metadata_path=metadeata_path)


def install_package(package_name, version=None):
    """Placeholder docstring"""
    if version:
        command = f"pip install {package_name}=={version}"
    else:
        command = f"pip install {package_name}"

    try:
        subprocess.check_call(command, shell=True)
        print(f"Successfully installed {package_name} using install_package")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}. Error: {e}")


def _get_mlflow_flavor():
    return os.getenv("MLFLOW_MODEL_FLAVOR")


def _load_mlflow_model(deployment_flavor, model_dir):
    # TODO: move this to constants section
    flavor_loader_map = {
        "keras": ("mlflow.keras", "load_model"),
        "python_function": ("mlflow.pyfunc", "load_model"),
        "pytorch": ("mlflow.pytorch", "load_model"),
        "sklearn": ("mlflow.sklearn", "load_model"),
        "tensorflow": ("mlflow.tensorflow", "load_model"),
        "xgboost": ("mlflow.xgboost", "load_model"),
        "langchain": ("mlflow.pyfunc", "load_model"),
    }

    flavor_module_name, load_function_name = flavor_loader_map.get(
        deployment_flavor, ("mlflow.pyfunc", "load_model")
    )
    logger.info(f"Using {flavor_module_name}.{load_function_name} loading the model.")
    flavor_module = importlib.import_module(flavor_module_name)
    load_model_function = getattr(flavor_module, load_function_name)
    return load_model_function(model_dir)


# on import, execute
_run_preflight_diagnostics()
