"""Prepare TgiModel for Deployment"""

from __future__ import absolute_import
import tarfile
import subprocess
import logging
from typing import List
from pathlib import Path

from sagemaker.serve.utils.local_hardware import _check_disk_space, _check_docker_disk_usage
from sagemaker.utils import _tmpdir

logger = logging.getLogger(__name__)


def _copy_jumpstart_artifacts(model_data: str, js_id: str, code_dir: Path) -> bool:
    """Placeholder Docstring"""
    logger.info("Downloading JumpStart artifacts from S3...")
    with _tmpdir(directory=str(code_dir)) as js_model_dir:
        # TODO: remove if block after 10/30 once everything is shifted to uncompressed
        if model_data.endswith("tar.gz"):
            subprocess.run(["aws", "s3", "cp", model_data, js_model_dir])

            logger.info("Uncompressing JumpStart artifacts for faster loading...")
            tmp_sourcedir = Path(js_model_dir).joinpath(f"infer-prepack-{js_id}.tar.gz")
            with tarfile.open(str(tmp_sourcedir)) as resources:
                resources.extractall(path=code_dir)
        else:
            subprocess.run(["aws", "s3", "cp", model_data, js_model_dir, "--recursive"])
    return True


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


def prepare_tgi_js_resources(
    model_path: str,
    js_id: str,
    shared_libs: List[str] = None,
    dependencies: str = None,
    model_data: str = None,
) -> bool:
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
