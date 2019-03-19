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

from collections import namedtuple

import os
import re
import sagemaker.utils
import shutil
import tempfile
from six.moves.urllib.parse import urlparse

from sagemaker.utils import get_ecr_image_uri_prefix, ECR_URI_PATTERN

_TAR_SOURCE_FILENAME = 'source.tar.gz'

UploadedCode = namedtuple('UserCode', ['s3_prefix', 'script_name'])
"""sagemaker.fw_utils.UserCode: An object containing the S3 prefix and script name.

This is for the source code used for the entry point with an ``Estimator``. It can be
instantiated with positional or keyword arguments.
"""

EMPTY_FRAMEWORK_VERSION_WARNING = 'No framework_version specified, defaulting to version {}.'
LATER_FRAMEWORK_VERSION_WARNING = 'This is not the latest supported version. ' \
                                  'If you would like to use version {latest}, ' \
                                  'please add framework_version={latest} to your constructor.'

EMPTY_FRAMEWORK_VERSION_ERROR = 'framework_version is required for script mode estimator. ' \
                                'Please add framework_version={} to your constructor to avoid this error.'

VALID_PY_VERSIONS = ['py2', 'py3']
VALID_EIA_FRAMEWORKS = ['tensorflow', 'tensorflow-serving', 'mxnet']
VALID_ACCOUNTS_BY_REGION = {'us-gov-west-1': '246785580436',
                            'us-iso-east-1': '744548109606'}


def create_image_uri(region, framework, instance_type, framework_version, py_version=None,
                     account='520713654638', accelerator_type=None, optimized_families=None):
    """Return the ECR URI of an image.

    Args:
        region (str): AWS region where the image is uploaded.
        framework (str): framework used by the image.
        instance_type (str): SageMaker instance type. Used to determine device type (cpu/gpu/family-specific optimized).
        framework_version (str): The version of the framework.
        py_version (str): Optional. Python version. If specified, should be one of 'py2' or 'py3'.
            If not specified, image uri will not include a python component.
        account (str): AWS account that contains the image. (default: '520713654638')
        accelerator_type (str): SageMaker Elastic Inference accelerator type.
        optimized_families (str): Instance families for which there exist specific optimized images.

    Returns:
        str: The appropriate image URI based on the given parameters.
    """
    optimized_families = optimized_families or []

    if py_version and py_version not in VALID_PY_VERSIONS:
        raise ValueError('invalid py_version argument: {}'.format(py_version))

    # Handle Account Number for Gov Cloud
    account = VALID_ACCOUNTS_BY_REGION.get(region, account)

    # Handle Local Mode
    if instance_type.startswith('local'):
        device_type = 'cpu' if instance_type == 'local' else 'gpu'
    elif not instance_type.startswith('ml.'):
        raise ValueError('{} is not a valid SageMaker instance type. See: '
                         'https://aws.amazon.com/sagemaker/pricing/instance-types/'.format(instance_type))
    else:
        family = instance_type.split('.')[1]

        # For some frameworks, we have optimized images for specific families, e.g c5 or p3. In those cases,
        # we use the family name in the image tag. In other cases, we use 'cpu' or 'gpu'.
        if family in optimized_families:
            device_type = family
        elif family[0] in ['g', 'p']:
            device_type = 'gpu'
        else:
            device_type = 'cpu'

    if py_version:
        tag = "{}-{}-{}".format(framework_version, device_type, py_version)
    else:
        tag = "{}-{}".format(framework_version, device_type)

    if _accelerator_type_valid_for_framework(framework=framework, accelerator_type=accelerator_type,
                                             optimized_families=optimized_families):
        framework += '-eia'

    return "{}/sagemaker-{}:{}" \
        .format(get_ecr_image_uri_prefix(account, region), framework, tag)


def _accelerator_type_valid_for_framework(framework, accelerator_type=None, optimized_families=None):
    if accelerator_type is None:
        return False

    if framework not in VALID_EIA_FRAMEWORKS:
        raise ValueError('{} is not supported with Amazon Elastic Inference. Currently only '
                         'Python-based TensorFlow and MXNet are supported.'.format(framework))

    if optimized_families:
        raise ValueError('Neo does not support Amazon Elastic Inference.')

    if not accelerator_type.startswith('ml.eia') and not accelerator_type == 'local_sagemaker_notebook':
        raise ValueError('{} is not a valid SageMaker Elastic Inference accelerator type. '
                         'See: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html'
                         .format(accelerator_type))

    return True


def validate_source_dir(script, directory):
    """Validate that the source directory exists and it contains the user script

    Args:
        script (str):  Script filename.
        directory (str): Directory containing the source file.

    Raises:
        ValueError: If ``directory`` does not exist, is not a directory, or does not contain ``script``.
    """
    if directory:
        if not os.path.isfile(os.path.join(directory, script)):
            raise ValueError('No file named "{}" was found in directory "{}".'.format(script, directory))

    return True


def tar_and_upload_dir(session, bucket, s3_key_prefix, script,
                       directory=None, dependencies=None, kms_key=None):
    """Package source files and upload a compress tar file to S3. The S3 location will be
    ``s3://<bucket>/s3_key_prefix/sourcedir.tar.gz``.

    If directory is an S3 URI, an UploadedCode object will be returned, but nothing will be
    uploaded to S3 (this allow reuse of code already in S3).

    If directory is None, the script will be added to the archive at ``./<basename of script>``.

    If directory is not None, the (recursive) contents of the directory will be added to
    the archive. directory is treated as the base path of the archive, and the script name is
    assumed to be a filename or relative path inside the directory.

    Args:
        session (boto3.Session): Boto session used to access S3.
        bucket (str): S3 bucket to which the compressed file is uploaded.
        s3_key_prefix (str): Prefix for the S3 key.
        script (str): Script filename or path.
        directory (str): Optional. Directory containing the source file. If it starts with "s3://",
            no action is taken.
        dependencies (List[str]): Optional. A list of paths to directories (absolute or relative)
                                containing additional libraries that will be copied into
                                /opt/ml/lib
        kms_key (str):  Optional. KMS key ID used to upload objects to the bucket (default: None).

    Returns:
        sagemaker.fw_utils.UserCode: An object with the S3 bucket and key (S3 prefix) and
            script name.
    """
    if directory and directory.lower().startswith('s3://'):
        return UploadedCode(s3_prefix=directory, script_name=os.path.basename(script))

    script_name = script if directory else os.path.basename(script)
    dependencies = dependencies or []
    key = '%s/sourcedir.tar.gz' % s3_key_prefix
    tmp = tempfile.mkdtemp()

    try:
        source_files = _list_files_to_compress(script, directory) + dependencies
        tar_file = sagemaker.utils.create_tar_file(source_files,
                                                   os.path.join(tmp, _TAR_SOURCE_FILENAME))

        if kms_key:
            extra_args = {'ServerSideEncryption': 'aws:kms', 'SSEKMSKeyId': kms_key}
        else:
            extra_args = None

        session.resource('s3').Object(bucket, key).upload_file(tar_file, ExtraArgs=extra_args)
    finally:
        shutil.rmtree(tmp)

    return UploadedCode(s3_prefix='s3://%s/%s' % (bucket, key), script_name=script_name)


def _list_files_to_compress(script, directory):
    if directory is None:
        return [script]

    basedir = directory if directory else os.path.dirname(script)
    return [os.path.join(basedir, name) for name in os.listdir(basedir)]


def framework_name_from_image(image_name):
    """Extract the framework and Python version from the image name.

    Args:
        image_name (str): Image URI, which should be one of the following forms:
            legacy:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<container_version>'
            legacy:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<fw_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>:<fw_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-<fw>:<rl_toolkit><rl_version>-<device>-<py_ver>'

    Returns:
        tuple: A tuple containing:
            str: The framework name
            str: The Python version
            str: The image tag
            str: If the image is script mode
    """
    sagemaker_pattern = re.compile(ECR_URI_PATTERN)
    sagemaker_match = sagemaker_pattern.match(image_name)
    if sagemaker_match is None:
        return None, None, None, None
    else:
        # extract framework, python version and image tag
        # We must support both the legacy and current image name format.
        name_pattern = re.compile(
            r'^sagemaker(?:-rl)?-(tensorflow|mxnet|chainer|pytorch|scikit-learn)(?:-)?(scriptmode)?:(.*)-(.*?)-(py2|py3)$') # noqa
        legacy_name_pattern = re.compile(
            r'^sagemaker-(tensorflow|mxnet)-(py2|py3)-(cpu|gpu):(.*)$')

        name_match = name_pattern.match(sagemaker_match.group(9))
        legacy_match = legacy_name_pattern.match(sagemaker_match.group(9))

        if name_match is not None:
            fw, scriptmode, ver, device, py = name_match.group(1), name_match.group(2), name_match.group(3),\
                name_match.group(4), name_match.group(5)
            return fw, py, '{}-{}-{}'.format(ver, device, py), scriptmode
        elif legacy_match is not None:
            return legacy_match.group(1), legacy_match.group(2), legacy_match.group(4), None
        else:
            return None, None, None, None


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


def model_code_key_prefix(code_location_key_prefix, model_name, image):
    """Returns the s3 key prefix for uploading code during model deployment

    The location returned is a potential concatenation of 2 parts
        1. code_location_key_prefix if it exists
        2. model_name or a name derived from the image

    Args:
        code_location_key_prefix (str): the s3 key prefix from code_location
        model_name (str): the name of the model
        image (str): the image from which a default name can be extracted

    Returns:
        str: the key prefix to be used in uploading code
    """
    training_job_name = sagemaker.utils.name_from_image(image)
    return '/'.join(filter(None, [code_location_key_prefix, model_name or training_job_name]))


def empty_framework_version_warning(default_version, latest_version):
    msgs = [EMPTY_FRAMEWORK_VERSION_WARNING.format(default_version)]
    if default_version != latest_version:
        msgs.append(LATER_FRAMEWORK_VERSION_WARNING.format(latest=latest_version))
    return ' '.join(msgs)
