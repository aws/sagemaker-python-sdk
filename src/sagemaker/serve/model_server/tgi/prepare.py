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
"""Prepare TgiModel for Deployment"""

from __future__ import absolute_import

import json
import tarfile
import logging
from typing import List
from pathlib import Path

from sagemaker.serve.utils.local_hardware import _check_disk_space, _check_docker_disk_usage
from sagemaker.utils import _tmpdir, custom_extractall_tarfile
from sagemaker.s3 import S3Downloader

logger = logging.getLogger(__name__)


def _extract_js_resource(js_model_dir: str, code_dir: Path, js_id: str):
    """Uncompress the jumpstart resource"""
    tmp_sourcedir = Path(js_model_dir).joinpath(f"infer-prepack-{js_id}.tar.gz")
    with tarfile.open(str(tmp_sourcedir)) as resources:
        custom_extractall_tarfile(resources, code_dir)


def _copy_jumpstart_artifacts(model_data: str, js_id: str, code_dir: Path) -> tuple:
    """Copy the associated JumpStart Resource into the code directory"""
    logger.info("Downloading JumpStart artifacts from S3...")

    s3_downloader = S3Downloader()
    if isinstance(model_data, str):
        if model_data.endswith(".tar.gz"):
            logger.info("Uncompressing JumpStart artifacts for faster loading...")
            with _tmpdir(directory=str(code_dir)) as js_model_dir:
                s3_downloader.download(model_data, js_model_dir)
                _extract_js_resource(js_model_dir, code_dir, js_id)
        else:
            logger.info("Copying uncompressed JumpStart artifacts...")
            s3_downloader.download(model_data, code_dir)
    elif (
        isinstance(model_data, dict)
        and model_data.get("S3DataSource")
        and model_data.get("S3DataSource").get("S3Uri")
    ):
        logger.info("Copying uncompressed JumpStart artifacts...")
        s3_downloader.download(model_data.get("S3DataSource").get("S3Uri"), code_dir)
    else:
        raise ValueError("JumpStart model data compression format is unsupported: %s", model_data)

    config_json_file = code_dir.joinpath("config.json")
    hf_model_config = None
    if config_json_file.is_file():
        with open(str(config_json_file)) as config_json:
            hf_model_config = json.load(config_json)

    return (hf_model_config, True)


def _create_dir_structure(model_path: str) -> tuple:
    """Create the expected model directory structure for the TGI server"""
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


def prepare_tgi_js_resources(
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
