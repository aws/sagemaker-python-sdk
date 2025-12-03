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
"""This module contains Enums and helper methods related to S3."""
from __future__ import print_function, absolute_import

import logging
import io

from typing import Union
from functools import reduce
from typing import Optional

from six.moves.urllib.parse import urlparse
from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger("sagemaker")


class S3Uploader(object):
    """Contains static methods for uploading directories or files to S3."""

    @staticmethod
    def upload(local_path, desired_s3_uri, kms_key=None, sagemaker_session=None, callback=None):
        """Static method that uploads a given file or directory to S3.

        Args:
            local_path (str): Path (absolute or relative) of local file or directory to upload.
            desired_s3_uri (str): The desired S3 location to upload to. It is the prefix to
                which the local filename will be added.
            kms_key (str): The KMS key to use to encrypt the files.
            sagemaker_session (sagemaker.core.helper.session_helper.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
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
            path=local_path,
            bucket=bucket,
            key_prefix=key_prefix,
            callback=callback,
            extra_args=extra_args,
        )

    @staticmethod
    def upload_string_as_file_body(
        body: str, desired_s3_uri=None, kms_key=None, sagemaker_session=None
    ):
        """Static method that uploads a given file or directory to S3.

        Args:
            body (str): String representing the body of the file.
            desired_s3_uri (str): The desired S3 uri to upload to.
            kms_key (str): The KMS key to use to encrypt the files.
            sagemaker_session (sagemaker.core.helper.session_helper.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed.

        Returns:
            str: The S3 uri of the uploaded file.

        """
        sagemaker_session = sagemaker_session or Session()

        bucket, key = parse_s3_url(desired_s3_uri)

        sagemaker_session.upload_string_as_file_body(
            body=body, bucket=bucket, key=key, kms_key=kms_key
        )

        return desired_s3_uri

    @staticmethod
    def upload_bytes(b: Union[bytes, io.BytesIO], s3_uri, kms_key=None, sagemaker_session=None):
        """Static method that uploads a given file or directory to S3.

        Args:
            b (bytes or io.BytesIO): bytes.
            s3_uri (str): The S3 uri to upload to.
            kms_key (str): The KMS key to use to encrypt the files.
            sagemaker_session (sagemaker.core.helper.session_helper.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            str: The S3 uri of the uploaded file.

        """
        sagemaker_session = sagemaker_session or Session()

        bucket, object_key = parse_s3_url(s3_uri)

        if kms_key is not None:
            extra_args = {"SSEKMSKeyId": kms_key, "ServerSideEncryption": "aws:kms"}
        else:
            extra_args = None

        b = b if isinstance(b, io.BytesIO) else io.BytesIO(b)
        sagemaker_session.s3_resource.Bucket(bucket).upload_fileobj(
            b, object_key, ExtraArgs=extra_args
        )

        return s3_uri


class S3Downloader(object):
    """Contains static methods for downloading directories or files from S3."""

    @staticmethod
    def download(s3_uri, local_path, kms_key=None, sagemaker_session=None):
        """Static method that downloads a given S3 uri to the local machine.

        Args:
            s3_uri (str): An S3 uri to download from.
            local_path (str): A local path to download the file(s) to.
            kms_key (str): The KMS key to use to decrypt the files.
            sagemaker_session (sagemaker.core.helper.session_helper.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            list[str]: List of local paths of downloaded files
        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=s3_uri)
        if kms_key is not None:
            extra_args = {"SSECustomerKey": kms_key}
        else:
            extra_args = None

        return sagemaker_session.download_data(
            path=local_path, bucket=bucket, key_prefix=key_prefix, extra_args=extra_args
        )

    @staticmethod
    def read_file(s3_uri, sagemaker_session=None) -> str:
        """Static method that returns the contents of a s3 uri file body as a string.

        Args:
            s3_uri (str): An S3 uri that refers to a single file.
            sagemaker_session (sagemaker.core.helper.session_helper.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            str: The body of the file.
        """
        sagemaker_session = sagemaker_session or Session()

        bucket, object_key = parse_s3_url(url=s3_uri)

        return sagemaker_session.read_s3_file(bucket=bucket, key_prefix=object_key)

    @staticmethod
    def read_bytes(s3_uri, sagemaker_session=None) -> bytes:
        """Static method that returns the contents of a s3 object as bytes.

        Args:
            s3_uri (str): An S3 uri that refers to a s3 object.
            sagemaker_session (sagemaker.core.helper.session_helper.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            bytes: The body of the file.
        """
        sagemaker_session = sagemaker_session or Session()

        bucket, object_key = parse_s3_url(s3_uri)

        bytes_io = io.BytesIO()
        sagemaker_session.s3_resource.Bucket(bucket).download_fileobj(object_key, bytes_io)
        bytes_io.seek(0)
        return bytes_io.read()

    @staticmethod
    def list(s3_uri, sagemaker_session=None):
        """Static method that lists the contents of an S3 uri.

        Args:
            s3_uri (str): The S3 base uri to list objects in.
            sagemaker_session (sagemaker.core.helper.session_helper.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

        Returns:
            [str]: The list of S3 URIs in the given S3 base uri.
        """
        sagemaker_session = sagemaker_session or Session()
        bucket, key_prefix = parse_s3_url(url=s3_uri)

        file_keys = sagemaker_session.list_s3_files(bucket=bucket, key_prefix=key_prefix)
        return [s3_path_join("s3://", bucket, file_key) for file_key in file_keys]


def parse_s3_url(url):
    """Returns an (s3 bucket, key name/prefix) tuple from a url with an s3 scheme.

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


def is_s3_url(url):
    """Returns True if url is an s3 url, False if not

    Args:
        url (str):

    Returns:
        bool:
    """
    parsed_url = urlparse(url)
    return parsed_url.scheme == "s3"


def s3_path_join(*args, with_end_slash: bool = False):
    """Returns the arguments joined by a slash ("/"), similar to ``os.path.join()`` (on Unix).

    Behavior of this function:
    - If the first argument is "s3://", then that is preserved.
    - The output by default will have no slashes at the beginning or end. There is one exception
        (see `with_end_slash`). For example, `s3_path_join("/foo", "bar/")` will yield
        `"foo/bar"` and `s3_path_join("foo", "bar", with_end_slash=True)` will yield `"foo/bar/"`
    - Any repeat slashes will be removed in the output (except for "s3://" if provided at the
        beginning). For example, `s3_path_join("s3://", "//foo/", "/bar///baz")` will yield
        `"s3://foo/bar/baz"`.
    - Empty or None arguments will be skipped. For example
        `s3_path_join("foo", "", None, "bar")` will yield `"foo/bar"`

    Alternatives to this function that are NOT recommended for S3 paths:
    - `os.path.join(...)` will have different behavior on Unix machines vs non-Unix machines
    - `pathlib.PurePosixPath(...)` will apply potentially unintended simplification of single
        dots (".") and root directories. (for example
        `pathlib.PurePosixPath("foo", "/bar/./", "baz")` would yield `"/bar/baz"`)
    - `"{}/{}/{}".format(...)` and similar may result in unintended repeat slashes

    Args:
        *args: The strings to join with a slash.
        with_end_slash (bool): (default: False) If true and if the path is not empty, appends a "/"
            to the end of the path

    Returns:
        str: The joined string, without a slash at the end unless with_end_slash is True.
    """
    delimiter = "/"

    non_empty_args = list(filter(lambda item: item is not None and item != "", args))

    merged_path = ""
    for index, path in enumerate(non_empty_args):
        if (
            index == 0
            or (merged_path and merged_path[-1] == delimiter)
            or (path and path[0] == delimiter)
        ):
            # dont need to add an extra slash because either this is the beginning of the string,
            # or one (or more) slash already exists
            merged_path += path
        else:
            merged_path += delimiter + path

    if with_end_slash and merged_path and merged_path[-1] != delimiter:
        merged_path += delimiter

    # At this point, merged_path may include slashes at the beginning and/or end. And some of the
    # provided args may have had duplicate slashes inside or at the ends.
    # For backwards compatibility reasons, these need to be filtered out (done below). In the
    # future, if there is a desire to support multiple slashes for S3 paths throughout the SDK,
    # one option is to create a new optional argument (or a new function) that only executes the
    # logic above.
    filtered_path = merged_path

    # remove duplicate slashes
    if filtered_path:

        def duplicate_delimiter_remover(sequence, next_char):
            if sequence[-1] == delimiter and next_char == delimiter:
                return sequence
            return sequence + next_char

        if filtered_path.startswith("s3://"):
            filtered_path = reduce(
                duplicate_delimiter_remover, filtered_path[5:], filtered_path[:5]
            )
        else:
            filtered_path = reduce(duplicate_delimiter_remover, filtered_path)

    # remove beginning slashes
    filtered_path = filtered_path.lstrip(delimiter)

    # remove end slashes
    if not with_end_slash and filtered_path != "s3://":
        filtered_path = filtered_path.rstrip(delimiter)

    return filtered_path


def determine_bucket_and_prefix(
    bucket: Optional[str] = None, key_prefix: Optional[str] = None, sagemaker_session=None
):
    """Helper function that returns the correct S3 bucket and prefix to use depending on the inputs.

    Args:
        bucket (Optional[str]): S3 Bucket to use (if it exists)
        key_prefix (Optional[str]): S3 Object Key Prefix to use or append to (if it exists)
        sagemaker_session (sagemaker.core.helper.session_helper.Session): Session to fetch a default bucket and
            prefix from, if bucket doesn't exist. Expected to exist

    Returns: The correct S3 Bucket and S3 Object Key Prefix that should be used
    """
    if bucket:
        final_bucket = bucket
        final_key_prefix = key_prefix
    else:
        final_bucket = sagemaker_session.default_bucket()

        # default_bucket_prefix (if it exists) should be appended if (and only if) 'bucket' does not
        # exist and we are using the Session's default_bucket.
        final_key_prefix = s3_path_join(sagemaker_session.default_bucket_prefix, key_prefix)

    # We should not append default_bucket_prefix even if the bucket exists but is equal to the
    # default_bucket, because either:
    # (1) the bucket was explicitly passed in by the user and just happens to be the same as the
    # default_bucket (in which case we don't want to change the user's input), or
    # (2) the default_bucket was fetched from Session earlier already (and the default prefix
    # should have been fetched then as well), and then this function was
    # called with it. If we appended the default prefix here, we would be appending it more than
    # once in total.

    return final_bucket, final_key_prefix
