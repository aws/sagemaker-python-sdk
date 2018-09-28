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

import errno
import os
import shutil
from six.moves.urllib.parse import urlparse


def copy_directory_structure(destination_directory, relative_path):
    """
    Creates all the intermediate directories required for relative_path to exist within destination_directory.
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


def move_to_destination(source, destination, sagemaker_session):
    """
    move source to destination. Can handle uploading to S3
    Args:
        source (str): root directory to move
        destination (str): file:// or s3:// URI that source will be moved to.
        sagemaker_session (sagemaker.Session): a sagemaker_session to interact with S3 if needed

    """
    parsed_uri = urlparse(destination)
    if parsed_uri.scheme == 'file':
        recursive_copy(source, parsed_uri.path)
    elif parsed_uri.scheme == 's3':
        bucket = parsed_uri.netloc
        path = parsed_uri.path.strip('/')
        sagemaker_session.upload_data(source, bucket, path)
    else:
        raise ValueError('Invalid destination URI, must be s3:// or file://')

    shutil.rmtree(source)


def recursive_copy(source, destination):
    """
    Similar to shutil.copy but the destination directory can exist. Existing files will be overriden.
    Args:
        source (str):
        destination:

    Returns:

    """
    for root, dirs, files in os.walk(source):
        root = os.path.relpath(root, source)
        current_path = os.path.join(source, root)
        target_path = os.path.join(destination, root)

        for file in files:
            shutil.copy(os.path.join(current_path, file), os.path.join(target_path, file))
        for dir in dirs:
            new_dir = os.path.join(target_path, dir)
            if not os.path.exists(new_dir):
                os.mkdir(os.path.join(target_path, dir))


def download_folder(bucket_name, prefix, target, sagemaker_session):
    boto_session = sagemaker_session.boto_session

    s3 = boto_session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    prefix = prefix.lstrip('/')

    # there is a chance that the prefix points to a file and not a 'directory' if that is the case
    # we should just download it.
    objects = list(bucket.objects.filter(Prefix=prefix))

    if len(objects) > 0 and objects[0].key == prefix and prefix[-1] != '/':
        s3.Object(bucket_name, prefix).download_file(os.path.join(target, os.path.basename(prefix)))
        return

    # the prefix points to an s3 'directory' download the whole thing
    for obj_sum in bucket.objects.filter(Prefix=prefix):
        # if obj_sum is a folder object skip it.
        if obj_sum.key != '' and obj_sum.key[-1] == '/':
            continue
        obj = s3.Object(obj_sum.bucket_name, obj_sum.key)
        s3_relative_path = obj_sum.key[len(prefix):].lstrip('/')
        file_path = os.path.join(target, s3_relative_path)

        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        obj.download_file(file_path)


def download_file(bucket_name, path, target, sagemaker_session):
    path = path.lstrip('/')
    boto_session = sagemaker_session.boto_session

    s3 = boto_session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(path, target)
