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
import os
import re
import tarfile
import tempfile
from collections import namedtuple
from six.moves.urllib.parse import urlparse

"""This module contains utility functions shared across ``Framework`` components."""


UploadedCode = namedtuple('UserCode', ['s3_prefix', 'script_name'])
"""sagemaker.fw_utils.UserCode: An object containing the S3 prefix and script name.

This is for the source code used for the entry point with an ``Estimator``. It can be
instantiated with positional or keyword arguments.
"""


def create_image_uri(region, framework, instance_type, framework_version, py_version, account='520713654638'):
    """Return the ECR URI of an image.

    Args:
        region (str): AWS region where the image is uploaded.
        framework (str): framework used by the image.
        instance_type (str): EC2 instance type. Used to determine whether to use the CPU image or GPU image.
        framework_version (str): The version of the framework.
        py_version (str): Python version. One of 'py2' or 'py3'.
        account (str): AWS account that contains the image. (default: '520713654638')

    Returns:
        str: The appropriate image URI based on the given parameters.
    """
    device_type = 'cpu'
    # Instance types that start with G, P are GPU powered: https://aws.amazon.com/sagemaker/pricing/instance-types/
    if instance_type[3] in ['g', 'p']:
        device_type = 'gpu'

    tag = "{}-{}-{}".format(framework_version, device_type, py_version)
    return "{}.dkr.ecr.{}.amazonaws.com/sagemaker-{}-{}-{}:{}" \
        .format(account, region, framework, py_version, device_type, tag)


def tar_and_upload_dir(session, bucket, s3_key_prefix, script, directory):
    """Pack and upload source files to S3 only if directory is empty or local.

    Note:
        If the directory points to S3 no action is taken.

    Args:
        session (boto3.Session): Boto session used to access S3.
        bucket (str): S3 bucket to which the compressed file is uploaded.
        s3_key_prefix (str): Prefix for the S3 key.
        script (str): Script filename.
        directory (str): Directory containing the source file. If it starts with "s3://", no action is taken.

    Returns:
        sagemaker.fw_utils.UserCode: An object with the S3 bucket and key (S3 prefix) and script name.

    Raises:
        ValueError: If ``directory`` does not exist, is not a directory, or does not contain ``script``.
    """
    if directory:
        if directory.lower().startswith("s3://"):
            return UploadedCode(s3_prefix=directory, script_name=os.path.basename(script))
        if not os.path.exists(directory):
            raise ValueError('"{}" does not exist.'.format(directory))
        if not os.path.isdir(directory):
            raise ValueError('"{}" is not a directory.'.format(directory))
        if script not in os.listdir(directory):
            raise ValueError('No file named "{}" was found in directory "{}".'.format(script, directory))
        script_name = script
        source_files = [os.path.join(directory, name) for name in os.listdir(directory)]
    else:
        # If no directory is specified, the script parameter needs to be a valid relative path.
        os.path.exists(script)
        script_name = os.path.basename(script)
        source_files = [script]

    s3 = session.resource('s3')
    key = '{}/{}'.format(s3_key_prefix, 'sourcedir.tar.gz')

    with tempfile.TemporaryFile() as f:
        with tarfile.open(mode='w:gz', fileobj=f) as t:
            for sf in source_files:
                # Add all files from the directory into the root of the directory structure of the tar
                t.add(sf, arcname=os.path.basename(sf))
        # Need to reset the file descriptor position after writing to prepare for read
        f.seek(0)
        s3.Object(bucket, key).put(Body=f)

    return UploadedCode(s3_prefix='s3://{}/{}'.format(bucket, key), script_name=script_name)


def framework_name_from_image(image_name):
    """Extract the framework and Python version from the image name.

    Args:
        image_name (str): Image URI, which should take the form
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<framework>-<py_ver>-<device>:<tag>'

    Returns:
        tuple: A tuple containing:
            str: The framework name
            str: The Python version
            str: The image tag
    """
    # image name format: <account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<framework>-<py_ver>-<device>:<tag>
    sagemaker_pattern = re.compile('^(\d+)(\.)dkr(\.)ecr(\.)(.+)(\.)amazonaws.com(/)(.*:.*)$')
    sagemaker_match = sagemaker_pattern.match(image_name)
    if sagemaker_match is None:
        return None, None, None
    else:
        # extract framework, python version and image tag
        name_pattern = re.compile('^sagemaker-(tensorflow|mxnet)-(py2|py3)-(cpu|gpu):(.*)$')

        name_match = name_pattern.match(sagemaker_match.group(8))

        if name_match is None:
            return None, None, None
        else:
            return name_match.group(1), name_match.group(2), name_match.group(4)


def framework_version_from_tag(image_tag):
    """Extract the framework version from the image tag.

    Args:
        image_tag (str): Image tag, which should take the form '<framework_version>-<device>-<py_version>'

    Returns:
        str: The framework version.
    """
    tag_pattern = re.compile('^(.*)-(cpu|gpu)-(py2|py3)$')
    tag_match = tag_pattern.match(image_tag)
    return None if tag_match is None else tag_match.group(1)


def parse_s3_url(url):
    """Returns an (s3 bucket, key name/prefix) tuple from a url with an s3 scheme

    Args:
        url (str):

    Returns:
        tuple: A tuple containing:
            str: S3 bucket name
            str: S3 key
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: {} in {}".format(parsed_url.scheme, url))
    return parsed_url.netloc, parsed_url.path.lstrip('/')
