"""Shared fixtures for remote function integration tests."""

import os
import shutil
import tempfile

import cloudpickle
import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.s3 import S3Uploader


def _get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _upload_core_source(sagemaker_session):
    """Tar the sagemaker-core source and upload to S3. Returns (s3_prefix, tar_basename)."""
    repo_root = _get_repo_root()
    core_dir = os.path.join(repo_root, "sagemaker-core")
    dist_dir = tempfile.mkdtemp(prefix="sagemaker_core_src_")

    archive_path = shutil.make_archive(
        os.path.join(dist_dir, "sagemaker-core-src"), "gztar", root_dir=core_dir, base_dir="."
    )

    s3_prefix = f"s3://{sagemaker_session.default_bucket()}/remote-function-test/src"
    S3Uploader.upload(archive_path, s3_prefix, sagemaker_session=sagemaker_session)

    return s3_prefix, os.path.basename(archive_path)


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
    """Upload dev sagemaker-core source to S3 and return pre_execution_commands."""
    s3_prefix, tar_name = _upload_core_source(sagemaker_session)
    cp_version = cloudpickle.__version__
    return [
        f"pip install cloudpickle=={cp_version}",
        f"aws s3 cp {s3_prefix}/{tar_name} /tmp/{tar_name}",
        "mkdir -p /tmp/sagemaker-core-src && tar xzf /tmp/{tar_name} -C /tmp/sagemaker-core-src".format(tar_name=tar_name),
        "pip install --no-deps /tmp/sagemaker-core-src",
    ]
