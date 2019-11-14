# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from six.moves.urllib.parse import urlparse
from sagemaker.session import Session


def parse_s3_url(url):
    """Returns an (s3 bucket, key name/prefix) tuple from a url with an s3
    scheme.
    Args:
        url (str):
    Returns:
        tuple: A tuple containing:
            str: S3 bucket name str: S3 key
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: {} in {}.".format(parsed_url.scheme, url))
    return parsed_url.netloc, parsed_url.path.lstrip("/")


class S3Uploader(object):
    """Contains static methods for uploading directories or files to S3."""

    @staticmethod
    def upload(local_path, desired_s3_uri, kms_key=None, session=None):
        """Static method that uploads a given file or directory to S3.

        Args:
            local_path (str): A local path to a file or directory.
            desired_s3_uri (str): The desired S3 uri to upload to.
            kms_key (str): A KMS key to be provided as an extra argument.
            session (sagemaker.session.Session):

        Returns: The S3 uri of the uploaded file(s).
        """
        sagemaker_session = session or Session()
        bucket, key_prefix = parse_s3_url(desired_s3_uri)
        if kms_key is not None:
            extra_args = {"SSEKMSKeyId": kms_key}
        else:
            extra_args = None

        return sagemaker_session.upload_data(
            path=local_path, bucket=bucket, key_prefix=key_prefix, extra_args=extra_args
        )


class S3Downloader(object):
    """Contains static methods for downloading directories or files from S3."""

    @staticmethod
    def download(s3_uri, local_path, kms_key=None, session=None):
        """Static method that downloads a given S3 uri to the local machine.

        Args:
            s3_uri (str): An S3 uri to download from.
            local_path (str): A local path to download the file(s) to.
            kms_key (str): A KMS key to be provided as an extra argument.
            session (sagemaker.session.Session):
        """
        sagemaker_session = session or Session()
        bucket, key_prefix = parse_s3_url(s3_uri)
        if kms_key is not None:
            extra_args = {"SSEKMSKeyId": kms_key}
        else:
            extra_args = None

        sagemaker_session.download_data(
            path=local_path, bucket=bucket, key_prefix=key_prefix, extra_args=extra_args
        )
