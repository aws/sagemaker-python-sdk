# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Prepare DjlModel for Deployment"""

from __future__ import absolute_import
import shutil
import tarfile
import subprocess
import json
import logging
from typing import List
from pathlib import Path

from sagemaker.utils import _tmpdir
from sagemaker.djl_inference import DJLModel
from sagemaker.djl_inference.model import _read_existing_serving_properties
from sagemaker.serve.utils.local_hardware import _check_disk_space, _check_docker_disk_usage

_SERVING_PROPERTIES_FILE = "serving.properties"
_ENTRY_POINT_SCRIPT = "inference.py"
_SETTING_PROPERTY_STMT = "Setting property: %s to %s"

logger = logging.getLogger(__name__)


def _has_serving_properties_file(code_dir: Path) -> bool:
    """Placeholder Docstring"""
    return code_dir.joinpath(_SERVING_PROPERTIES_FILE).is_file()


def _members(resources: object, depth: int):
    """Placeholder Docstring"""
    for member in resources.getmembers():
        member.path = member.path.split("/", depth)[-1]
        yield member


def _copy_jumpstart_artifacts(model_data: str, js_id: str, code_dir: Path):
    """Placeholder Docstring"""
    logger.info("Downloading JumpStart artifacts from S3...")
    with _tmpdir(directory=str(code_dir)) as js_model_dir:
        subprocess.run(["aws", "s3", "cp", model_data, js_model_dir])

        logger.info("Uncompressing JumpStart artifacts for faster loading...")
        tmp_sourcedir = Path(js_model_dir).joinpath(f"infer-prepack-{js_id}.tar.gz")
        with tarfile.open(str(tmp_sourcedir)) as resources:
            resources.extractall(path=code_dir, members=_members(resources, 1))

    existing_properties = _read_existing_serving_properties(code_dir)
    config_json_file = code_dir.joinpath("config.json")

    hf_model_config = None
    if config_json_file.is_file():
        with open(str(config_json_file)) as config_json:
            hf_model_config = json.load(config_json)

    return (existing_properties, hf_model_config, True)


def _generate_properties_file(
    model: DJLModel, code_dir: Path, overwrite_props_from_file: bool, manual_set_props: dict
):
    """Placeholder Docstring"""
    if _has_serving_properties_file(code_dir):
        existing_properties = _read_existing_serving_properties(code_dir)
    else:
        existing_properties = {}

    serving_properties_dict = model.generate_serving_properties()
    serving_properties_file = code_dir.joinpath(_SERVING_PROPERTIES_FILE)

    with open(serving_properties_file, mode="w+") as file:
        covered_keys = set()

        if manual_set_props:
            for key, value in manual_set_props.items():
                logger.info(_SETTING_PROPERTY_STMT, key, value.strip())
                covered_keys.add(key)
                file.write(f"{key}={value}")

        for key, value in serving_properties_dict.items():
            if not overwrite_props_from_file:
                logger.info(_SETTING_PROPERTY_STMT, key, value)
                file.write(f"{key}={value}\n")
            else:
                existing_property = existing_properties.get(key)
                covered_keys.add(key)
                if not existing_property:
                    logger.info(_SETTING_PROPERTY_STMT, key, value)
                    file.write(f"{key}={value}\n")
                else:
                    logger.info(_SETTING_PROPERTY_STMT, key, existing_property.strip())
                    file.write(f"{key}={existing_property}")

        if overwrite_props_from_file:
            # for addition provided properties
            for key, value in existing_properties.items():
                if key not in covered_keys:
                    logger.info(_SETTING_PROPERTY_STMT, key, value.strip())
                    file.write(f"{key}={value}")


def _store_share_libs(model_path: Path, shared_libs):
    """Placeholder Docstring"""
    shared_libs_dir = model_path.joinpath("shared_libs")
    shared_libs_dir.mkdir(exist_ok=True)
    for shared_lib in shared_libs:
        shutil.copy2(Path(shared_lib), shared_libs_dir)


def _copy_inference_script(code_dir):
    """Placeholder Docstring"""
    if code_dir.joinpath("inference.py").is_file():
        return

    inference_file = Path(__file__).parent.joinpath(_ENTRY_POINT_SCRIPT)
    shutil.copy2(inference_file, code_dir)


def _create_dir_structure(model_path: str) -> tuple:
    """Placeholder Docstring"""
    model_path = Path(model_path)
    if not model_path.exists():
        model_path.mkdir(parents=True)
    elif not model_path.is_dir():
        raise ValueError("model_dir is not a valid directory")

    code_dir = model_path.joinpath("code")
    code_dir.mkdir(exist_ok=True, parents=True)

    _check_disk_space(model_path)
    _check_docker_disk_usage()

    return (model_path, code_dir)


def prepare_for_djl_serving(
    model_path: str,
    model: DJLModel,
    shared_libs: List[str] = None,
    dependencies: str = None,
    overwrite_props_from_file: bool = True,
    manual_set_props: dict = None,
):
    """Prepare serving when a HF model id is given

    Args:to
        model_path (str) : Argument
        model (DJLModel) : Argument
        shared_libs (List[]) : Argument
        dependencies (str) : Argument

    Returns:
        ( str ) :

    """
    model_path, code_dir = _create_dir_structure(model_path)

    if shared_libs:
        _store_share_libs(model_path, shared_libs)

    _copy_inference_script(code_dir)

    _generate_properties_file(model, code_dir, overwrite_props_from_file, manual_set_props)


def prepare_djl_js_resources(
    model_path: str,
    js_id: str,
    shared_libs: List[str] = None,
    dependencies: str = None,
    model_data: str = None,
) -> tuple:
    """Prepare serving when a JumpStart model id is given

    Args:
        model_path (str) : Argument
        js_id (str): Argument
        to_uncompressed (bool): Argument
        shared_libs (List[]) : Argument
        dependencies (str) : Argument
        model_data (str) : Argument

    Returns:
        ( str ) :

    """
    model_path, code_dir = _create_dir_structure(model_path)

    return _copy_jumpstart_artifacts(model_data, js_id, code_dir)
