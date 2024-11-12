"""Upload model artifacts to S3"""

from __future__ import absolute_import
import logging
import os
import tempfile
import botocore
import boto3
import tqdm
from sagemaker.session import Session
from sagemaker.s3_utils import s3_path_join
from sagemaker.s3 import S3Uploader
from sagemaker.utils import create_tar_file

logger = logging.getLogger(__name__)

# Minimum size required for multi-part uploads
BUF_SIZE = 5 * 1024 * 1024


def _get_dir_size(path):
    """Calculate the size of a directory"""
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += _get_dir_size(entry.path)
    return total


class Uploader(object):
    """Uploader class that handles uploading data to S3 and display progress bar"""

    def __init__(self) -> None:
        self.total_left = None
        self.pbar = None

    def observe(self, bytes_amount):
        """Placeholder docstring"""
        self.total_left -= bytes_amount
        self.pbar.update(bytes_amount)

    def upload(
        self,
        model_dir,
        total_size: int,
        credentials: botocore.credentials.Credentials,
        region_name: str,
        bucket: str,
        key: str,
    ):
        """Compress and upload the model tar object to S3"""
        self.total_left = total_size
        with tqdm.tqdm(
            total=total_size, desc="Uploading model artifacts", unit="bytes", ncols=100
        ) as self.pbar:
            files = [os.path.join(model_dir, name) for name in os.listdir(model_dir)]
            temp = tempfile.mkdtemp()
            tar_file = create_tar_file(files, os.path.join(temp, "model.tar.gz"))

            s3 = boto3.session.Session(
                region_name=region_name,
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
            ).client("s3")
            s3.upload_file(os.path.join(temp, "model.tar.gz"), bucket, key, Callback=self.observe)
            os.remove(tar_file)
            self.pbar.update(self.total_left)
            self.pbar.close()
            self.pbar = None

    def upload_uncompressed(
        self,
        model_dir: str,
        sagemaker_session: Session,
        bucket: str,
        key_prefix: str,
        total_size: int,
    ):
        """Upload uncompressed model artifacts to S3"""
        self.total_left = total_size
        with tqdm.tqdm(
            total=total_size, desc="Uploading model artifacts", unit="bytes", ncols=100
        ) as self.pbar:
            S3Uploader.upload(
                local_path=model_dir,
                desired_s3_uri=s3_path_join("s3://", bucket, key_prefix),
                sagemaker_session=sagemaker_session,
                callback=self.observe,
            )
            self.pbar.update(self.total_left)


def upload(sagemaker_session: Session, model_dir: str, bucket: str, key_prefix: str) -> str:
    """Wrapper function of method upload"""
    key = key_prefix + "/serve.tar.gz"
    uploader = Uploader()
    uploader.upload(
        model_dir,
        _get_dir_size(model_dir),
        sagemaker_session.boto_session.get_credentials(),
        sagemaker_session.boto_session.region_name,
        bucket,
        key,
    )
    return s3_path_join("s3://", bucket, key)


def upload_uncompressed(
    sagemaker_session: Session, model_dir: str, bucket: str, key_prefix: str
) -> str:
    """Wrapper function of method upload_uncompressed"""
    uploader = Uploader()
    uploader.upload_uncompressed(
        model_dir, sagemaker_session, bucket, key_prefix, _get_dir_size(model_dir)
    )
    return s3_path_join("s3://", bucket, key_prefix, with_end_slash=True)
