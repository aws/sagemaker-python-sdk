# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This module contains Enums and helper methods related to S3."""
from __future__ import print_function, absolute_import

import pathlib
import logging

from six.moves.urllib.parse import urlparse
from sagemaker.session import Session

logger = logging.getLogger("sagemaker")


def parse_s3_url(url):
    """Returns an (s3 bucket, key name/prefix) tuple from a url with an s3
    scheme.

    Args:
        url (str):

    Returns:
        tuple: A tuple containing:

            - str: S3 bucket name
            - str: S3 key
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: {} in {}.".format(parsed_url.scheme, url))
    return parsed_url.netloc, parsed_url.path.lstrip("/")


def s3_path_join(*args):
    """Returns the arguments joined by a slash ("/"), similarly to ``os.path.join()`` (on Unix).

    If the first argument is "s3://", then that is preserved.

    Args:
        *args: The strings to join with a slash.

    Returns:
        str: The joined string.
    """
    if args[0].startswith("s3://"):
        path = str(pathlib.PurePosixPath(*args[1:])).lstrip("/")
        return str(pathlib.PurePosixPath(args[0], path)).replace("s3:/", "s3://")

    return str(pathlib.PurePosixPath(*args)).lstrip("/")


class S3Uploader(object):
    """Contains static methods for uploading directories or files to S3."""

    @staticmethod
    def upload(local_path, desired_s3_uri, kms_key=None, sagemaker_session=None):
        """Static method that uploads a given file or directory to S3.

        Args:
            local_path (str): Path (absolute or relative) of local file or directory to upload.
            desired_s3_uri (str): The desired S3 location to upload to. It is the prefix to
                which the local filename will be added.
            kms_key (str): The KMS key to use to encrypt the files.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.

        Returns:
            The S3 uri of the uploaded file(s).

        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=desired_s3_uri)
        if kms_key is not None:
            extra_args = {"SSEKMSKeyId": kms_key, "ServerSideEncryption": "aws:kms"}

        else:
            extra_args = None

        return sagemaker_session.upload_data(
            path=local_path, bucket=bucket, key_prefix=key_prefix, extra_args=extra_args
        )

    @staticmethod
    def upload_string_as_file_body(body, desired_s3_uri=None, kms_key=None, sagemaker_session=None):
        """Static method that uploads a given file or directory to S3.

        Args:
            body (str): String representing the body of the file.
            desired_s3_uri (str): The desired S3 uri to upload to.
            kms_key (str): The KMS key to use to encrypt the files.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.

        Returns:
            str: The S3 uri of the uploaded file(s).

        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key = parse_s3_url(desired_s3_uri)

        sagemaker_session.upload_string_as_file_body(
            body=body, bucket=bucket, key=key, kms_key=kms_key
        )

        return desired_s3_uri


class S3Downloader(object):
    """Contains static methods for downloading directories or files from S3."""

    @staticmethod
    def download(s3_uri, local_path, kms_key=None, sagemaker_session=None):
        """Static method that downloads a given S3 uri to the local machine.

        Args:
            s3_uri (str): An S3 uri to download from.
            local_path (str): A local path to download the file(s) to.
            kms_key (str): The KMS key to use to decrypt the files.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=s3_uri)
        if kms_key is not None:
            extra_args = {"SSECustomerKey": kms_key}
        else:
            extra_args = None

        sagemaker_session.download_data(
            path=local_path, bucket=bucket, key_prefix=key_prefix, extra_args=extra_args
        )

    @staticmethod
    def read_file(s3_uri, sagemaker_session=None):
        """Static method that returns the contents of an s3 uri file body as a string.

        Args:
            s3_uri (str): An S3 uri that refers to a single file.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.

        Returns:
            str: The body of the file.
        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=s3_uri)

        return sagemaker_session.read_s3_file(bucket=bucket, key_prefix=key_prefix)

    @staticmethod
    def list(s3_uri, sagemaker_session=None):
        """Static method that lists the contents of an S3 uri.

        Args:
            s3_uri (str): The S3 base uri to list objects in.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.

        Returns:
            [str]: The list of S3 URIs in the given S3 base uri.
        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=s3_uri)

        file_keys = sagemaker_session.list_s3_files(bucket=bucket, key_prefix=key_prefix)
        return [s3_path_join("s3://", bucket, file_key) for file_key in file_keys]
