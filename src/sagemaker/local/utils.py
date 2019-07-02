# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import os
import shutil

from distutils.dir_util import copy_tree
from six.moves.urllib.parse import urlparse


def copy_directory_structure(destination_directory, relative_path):
    """Create all the intermediate directories required for relative_path to exist within destination_directory.
    This assumes that relative_path is a directory located within root_dir.

    Examples:
        destination_directory: /tmp/destination
        relative_path: test/unit/

        will create:  /tmp/destination/test/unit

    Args:
        destination_directory (str): root of the destination directory where the directory structure will be created.
        relative_path (str): relative path that will be created within destination_directory
    """
    full_path = os.path.join(destination_directory, relative_path)
    if os.path.exists(full_path):
        return

    os.makedirs(destination_directory, relative_path)


def move_to_destination(source, destination, job_name, sagemaker_session):
    """move source to destination. Can handle uploading to S3

    Args:
        source (str): root directory to move
        destination (str): file:// or s3:// URI that source will be moved to.
        job_name (str): SageMaker job name.
        sagemaker_session (sagemaker.Session): a sagemaker_session to interact with S3 if needed

    Returns:
        (str): destination URI
    """
    parsed_uri = urlparse(destination)
    if parsed_uri.scheme == "file":
        recursive_copy(source, parsed_uri.path)
        final_uri = destination
    elif parsed_uri.scheme == "s3":
        bucket = parsed_uri.netloc
        path = "%s%s" % (parsed_uri.path.lstrip("/"), job_name)
        final_uri = "s3://%s/%s" % (bucket, path)
        sagemaker_session.upload_data(source, bucket, path)
    else:
        raise ValueError("Invalid destination URI, must be s3:// or file://, got: %s" % destination)

    shutil.rmtree(source)
    return final_uri


def recursive_copy(source, destination):
    """A wrapper around distutils.dir_util.copy_tree but won't throw any exception when the source
    directory does not exist.

    Args:
        source (str): source path
        destination (str): destination path
    """
    if os.path.isdir(source):
        copy_tree(source, destination)
