"""Shared fixtures for remote function integration tests."""

import glob
import os
import subprocess
import sys
import tempfile

import cloudpickle
import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.s3 import S3Uploader


def _get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _build_and_upload_core_wheel(sagemaker_session):
    """Build sagemaker-core wheel and upload to S3. Returns (s3_prefix, wheel_basename)."""
    repo_root = _get_repo_root()
    dist_dir = tempfile.mkdtemp(prefix="sagemaker_core_wheel_")

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "-w", dist_dir, "."],
        cwd=os.path.join(repo_root, "sagemaker-core"),
        check=True,
    )

    matches = glob.glob(os.path.join(dist_dir, "sagemaker_core-*.whl"))
    if not matches:
        raise FileNotFoundError(f"No sagemaker-core wheel found in {dist_dir}")
    wheel_path = matches[0]

    s3_prefix = f"s3://{sagemaker_session.default_bucket()}/remote-function-test/wheels"
    S3Uploader.upload(wheel_path, s3_prefix, sagemaker_session=sagemaker_session)

    return s3_prefix, os.path.basename(wheel_path)


@pytest.fixture(scope="module")
def sagemaker_session():
    import boto3
    return Session(boto3.Session())


@pytest.fixture(scope="module")
def role(sagemaker_session):
    import boto3
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:iam::{account_id}:role/Admin"


@pytest.fixture(scope="module")
def image_uri(sagemaker_session):
    region = sagemaker_session.boto_region_name
    return f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.0.0-cpu-py310"


@pytest.fixture(scope="module")
def dev_sdk_pre_execution_commands(sagemaker_session):
    """Build dev sagemaker-core wheel, upload to S3, and return pre_execution_commands."""
    s3_prefix, wheel_name = _build_and_upload_core_wheel(sagemaker_session)
    cp_version = cloudpickle.__version__
    return [
        f"pip install cloudpickle=={cp_version}",
        f"aws s3 cp {s3_prefix}/{wheel_name} /tmp/{wheel_name}",
        f"pip install /tmp/{wheel_name}",
    ]
