# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Utility methods used by framework classes"""
from __future__ import absolute_import

import logging
import os
import re
import shutil
import tempfile
from collections import namedtuple

import sagemaker.utils
from sagemaker import s3
from sagemaker.utils import get_ecr_image_uri_prefix, ECR_URI_PATTERN

logger = logging.getLogger("sagemaker")

_TAR_SOURCE_FILENAME = "source.tar.gz"

UploadedCode = namedtuple("UserCode", ["s3_prefix", "script_name"])
"""sagemaker.fw_utils.UserCode: An object containing the S3 prefix and script name.
This is for the source code used for the entry point with an ``Estimator``. It can be
instantiated with positional or keyword arguments.
"""

EMPTY_FRAMEWORK_VERSION_WARNING = "No framework_version specified, defaulting to version {}."
LATER_FRAMEWORK_VERSION_WARNING = (
    "This is not the latest supported version. "
    "If you would like to use version {latest}, "
    "please add framework_version={latest} to your constructor."
)
PYTHON_2_DEPRECATION_WARNING = (
    "{latest_supported_version} is the latest version of {framework} that supports "
    "Python 2. Newer versions of {framework} will only be available for Python 3."
    "Please set the argument \"py_version='py3'\" to use the Python 3 {framework} image."
)
PARAMETER_SERVER_MULTI_GPU_WARNING = (
    "You have selected a multi-GPU training instance type. "
    "You have also enabled parameter server for distributed training. "
    "Distributed training with the default parameter server configuration will not "
    "fully leverage all GPU cores; the parameter server will be configured to run "
    "only one worker per host regardless of the number of GPUs."
)


EMPTY_FRAMEWORK_VERSION_ERROR = (
    "framework_version is required for script mode estimator. "
    "Please add framework_version={} to your constructor to avoid this error."
)
UNSUPPORTED_FRAMEWORK_VERSION_ERROR = (
    "{} framework does not support version {}. Please use one of the following: {}."
)

VALID_PY_VERSIONS = ["py2", "py3"]
VALID_EIA_FRAMEWORKS = [
    "tensorflow",
    "tensorflow-serving",
    "mxnet",
    "mxnet-serving",
    "pytorch-serving",
]
PY2_RESTRICTED_EIA_FRAMEWORKS = ["pytorch-serving"]
VALID_ACCOUNTS_BY_REGION = {
    "us-gov-west-1": "246785580436",
    "us-iso-east-1": "744548109606",
    "cn-north-1": "422961961927",
    "cn-northwest-1": "423003514399",
}
ASIMOV_VALID_ACCOUNTS_BY_REGION = {
    "us-gov-west-1": "442386744353",
    "us-iso-east-1": "886529160074",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
}
OPT_IN_ACCOUNTS_BY_REGION = {"ap-east-1": "057415533634", "me-south-1": "724002660598"}
ASIMOV_OPT_IN_ACCOUNTS_BY_REGION = {"ap-east-1": "871362719292", "me-south-1": "217643126080"}
DEFAULT_ACCOUNT = "520713654638"
ASIMOV_PROD_ACCOUNT = "763104351884"
ASIMOV_DEFAULT_ACCOUNT = ASIMOV_PROD_ACCOUNT
SINGLE_GPU_INSTANCE_TYPES = ("ml.p2.xlarge", "ml.p3.2xlarge")

MERGED_FRAMEWORKS_REPO_MAP = {
    "tensorflow-scriptmode": "tensorflow-training",
    "tensorflow-serving": "tensorflow-inference",
    "tensorflow-serving-eia": "tensorflow-inference-eia",
    "mxnet": "mxnet-training",
    "mxnet-serving": "mxnet-inference",
    "mxnet-serving-eia": "mxnet-inference-eia",
    "pytorch": "pytorch-training",
    "pytorch-serving": "pytorch-inference",
    "pytorch-serving-eia": "pytorch-inference-eia",
}

MERGED_FRAMEWORKS_LOWEST_VERSIONS = {
    "tensorflow-scriptmode": {"py3": [1, 13, 1], "py2": [1, 14, 0]},
    "tensorflow-serving": [1, 13, 0],
    "tensorflow-serving-eia": [1, 14, 0],
    "mxnet": {"py3": [1, 4, 1], "py2": [1, 6, 0]},
    "mxnet-serving": {"py3": [1, 4, 1], "py2": [1, 6, 0]},
    "mxnet-serving-eia": [1, 4, 1],
    "pytorch": [1, 2, 0],
    "pytorch-serving": [1, 2, 0],
    "pytorch-serving-eia": [1, 3, 1],
}

INFERENTIA_VERSION_RANGES = {
    "neo-mxnet": [[1, 5, 1], [1, 5, 1]],
    "neo-tensorflow": [[1, 15, 0], [1, 15, 0]],
}

INFERENTIA_SUPPORTED_REGIONS = ["us-east-1", "us-west-2"]

DEBUGGER_UNSUPPORTED_REGIONS = ["us-gov-west-1", "us-iso-east-1"]


def is_version_equal_or_higher(lowest_version, framework_version):
    """Determine whether the ``framework_version`` is equal to or higher than
    ``lowest_version``

    Args:
        lowest_version (List[int]): lowest version represented in an integer
            list
        framework_version (str): framework version string

    Returns:
        bool: Whether or not ``framework_version`` is equal to or higher than
            ``lowest_version``
    """
    version_list = [int(s) for s in framework_version.split(".")]
    return version_list >= lowest_version[0 : len(version_list)]


def is_version_equal_or_lower(highest_version, framework_version):
    """Determine whether the ``framework_version`` is equal to or lower than
    ``highest_version``

    Args:
        highest_version (List[int]): highest version represented in an integer
            list
        framework_version (str): framework version string

    Returns:
        bool: Whether or not ``framework_version`` is equal to or lower than
            ``highest_version``
    """
    version_list = [int(s) for s in framework_version.split(".")]
    return version_list <= highest_version[0 : len(version_list)]


def _is_dlc_version(framework, framework_version, py_version):
    """Return if the framework's version uses the corresponding DLC image.

    Args:
        framework (str): The framework name, e.g. "tensorflow-scriptmode"
        framework_version (str): The framework version
        py_version (str): The Python version, e.g. "py3"

    Returns:
        bool: Whether or not the framework's version uses the DLC image.
    """
    lowest_version_list = MERGED_FRAMEWORKS_LOWEST_VERSIONS.get(framework)
    if isinstance(lowest_version_list, dict):
        lowest_version_list = lowest_version_list[py_version]

    if lowest_version_list:
        return is_version_equal_or_higher(lowest_version_list, framework_version)
    return False


def _is_inferentia_supported(framework, framework_version):
    """Return if Inferentia supports the framework and its version.

    Args:
        framework (str): The framework name, e.g. "tensorflow"
        framework_version (str): The framework version

    Returns:
        bool: Whether or not Inferentia supports the framework and its version.
    """
    lowest_version_list = INFERENTIA_VERSION_RANGES.get(framework)[0]
    highest_version_list = INFERENTIA_VERSION_RANGES.get(framework)[1]
    return is_version_equal_or_higher(
        lowest_version_list, framework_version
    ) and is_version_equal_or_lower(highest_version_list, framework_version)


def _registry_id(region, framework, py_version, account, framework_version):
    """Return the Amazon ECR registry number (or AWS account ID) for
    the given framework, framework version, Python version, and region.

    Args:
        region (str): The AWS region.
        framework (str): The framework name, e.g. "tensorflow-scriptmode".
        py_version (str): The Python version, e.g. "py3".
        account (str): The AWS account ID to use as a default.
        framework_version (str): The framework version.

    Returns:
        str: The appropriate Amazon ECR registry number. If there is no
            specific one for the framework, framework version, Python version,
            and region, then ``account`` is returned.
    """
    if _is_dlc_version(framework, framework_version, py_version):
        if region in ASIMOV_OPT_IN_ACCOUNTS_BY_REGION:
            return ASIMOV_OPT_IN_ACCOUNTS_BY_REGION.get(region)
        if region in ASIMOV_VALID_ACCOUNTS_BY_REGION:
            return ASIMOV_VALID_ACCOUNTS_BY_REGION.get(region)
        return ASIMOV_DEFAULT_ACCOUNT
    if region in OPT_IN_ACCOUNTS_BY_REGION:
        return OPT_IN_ACCOUNTS_BY_REGION.get(region)
    return VALID_ACCOUNTS_BY_REGION.get(region, account)


def create_image_uri(
    region,
    framework,
    instance_type,
    framework_version,
    py_version=None,
    account=None,
    accelerator_type=None,
    optimized_families=None,
):
    """Return the ECR URI of an image.

    Args:
        region (str): AWS region where the image is uploaded.
        framework (str): framework used by the image.
        instance_type (str): SageMaker instance type. Used to determine device
            type (cpu/gpu/family-specific optimized).
        framework_version (str): The version of the framework.
        py_version (str): Optional. Python version. If specified, should be one
            of 'py2' or 'py3'. If not specified, image uri will not include a
            python component.
        account (str): AWS account that contains the image. (default:
            '520713654638')
        accelerator_type (str): SageMaker Elastic Inference accelerator type.
        optimized_families (str): Instance families for which there exist
            specific optimized images.

    Returns:
        str: The appropriate image URI based on the given parameters.
    """
    optimized_families = optimized_families or []

    if py_version and py_version not in VALID_PY_VERSIONS:
        raise ValueError("invalid py_version argument: {}".format(py_version))

    if _accelerator_type_valid_for_framework(
        framework=framework,
        py_version=py_version,
        accelerator_type=accelerator_type,
        optimized_families=optimized_families,
    ):
        framework += "-eia"

    # Handle account number for specific cases (e.g. GovCloud, opt-in regions, DLC images etc.)
    if account is None:
        account = _registry_id(
            region=region,
            framework=framework,
            py_version=py_version,
            account=DEFAULT_ACCOUNT,
            framework_version=framework_version,
        )

    # Handle Local Mode
    if instance_type.startswith("local"):
        device_type = "cpu" if instance_type == "local" else "gpu"
    elif not instance_type.startswith("ml."):
        raise ValueError(
            "{} is not a valid SageMaker instance type. See: "
            "https://aws.amazon.com/sagemaker/pricing/instance-types/".format(instance_type)
        )
    else:
        family = instance_type.split(".")[1]

        # For some frameworks, we have optimized images for specific families, e.g c5 or p3.
        # In those cases, we use the family name in the image tag. In other cases, we use
        # 'cpu' or 'gpu'.
        if family in optimized_families:
            device_type = family
        elif family.startswith("inf"):
            device_type = "inf"
        elif family[0] in ["g", "p"]:
            device_type = "gpu"
        else:
            device_type = "cpu"

    if device_type == "inf":
        if region not in INFERENTIA_SUPPORTED_REGIONS:
            raise ValueError(
                "Inferentia is not supported in region {}. Supported regions are {}".format(
                    region, ", ".join(INFERENTIA_SUPPORTED_REGIONS)
                )
            )
        if framework not in INFERENTIA_VERSION_RANGES:
            raise ValueError(
                "Inferentia does not support {}. Currently it supports "
                "MXNet and TensorFlow with more frameworks coming soon.".format(
                    framework.split("-")[-1]
                )
            )
        if not _is_inferentia_supported(framework, framework_version):
            raise ValueError(
                "Inferentia is not supported with {} version {}.".format(
                    framework.split("-")[-1], framework_version
                )
            )

    use_dlc_image = _is_dlc_version(framework, framework_version, py_version)

    if not py_version or (use_dlc_image and framework == "tensorflow-serving-eia"):
        tag = "{}-{}".format(framework_version, device_type)
    else:
        tag = "{}-{}-{}".format(framework_version, device_type, py_version)

    if use_dlc_image:
        ecr_repo = MERGED_FRAMEWORKS_REPO_MAP[framework]
    else:
        ecr_repo = "sagemaker-{}".format(framework)

    return "{}/{}:{}".format(get_ecr_image_uri_prefix(account, region), ecr_repo, tag)


def _accelerator_type_valid_for_framework(
    framework, py_version, accelerator_type=None, optimized_families=None
):
    """
    Args:
        framework:
        py_version:
        accelerator_type:
        optimized_families:
    """
    if accelerator_type is None:
        return False

    if py_version == "py2" and framework in PY2_RESTRICTED_EIA_FRAMEWORKS:
        raise ValueError(
            "{} is not supported with Amazon Elastic Inference in Python 2.".format(framework)
        )

    if framework not in VALID_EIA_FRAMEWORKS:
        raise ValueError(
            "{} is not supported with Amazon Elastic Inference. Currently only "
            "Python-based TensorFlow, MXNet, PyTorch are supported.".format(framework)
        )

    if optimized_families:
        raise ValueError("Neo does not support Amazon Elastic Inference.")

    if (
        not accelerator_type.startswith("ml.eia")
        and not accelerator_type == "local_sagemaker_notebook"
    ):
        raise ValueError(
            "{} is not a valid SageMaker Elastic Inference accelerator type. "
            "See: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html".format(accelerator_type)
        )

    return True


def validate_source_dir(script, directory):
    """Validate that the source directory exists and it contains the user script
    Args:
        script (str): Script filename.
        directory (str): Directory containing the source file.
    Raises:
        ValueError: If ``directory`` does not exist, is not a directory, or does
            not contain ``script``.
    """
    if directory:
        if not os.path.isfile(os.path.join(directory, script)):
            raise ValueError(
                'No file named "{}" was found in directory "{}".'.format(script, directory)
            )

    return True


def tar_and_upload_dir(
    session,
    bucket,
    s3_key_prefix,
    script,
    directory=None,
    dependencies=None,
    kms_key=None,
    s3_resource=None,
):
    """Package source files and upload a compress tar file to S3. The S3
    location will be ``s3://<bucket>/s3_key_prefix/sourcedir.tar.gz``.
    If directory is an S3 URI, an UploadedCode object will be returned, but
    nothing will be uploaded to S3 (this allow reuse of code already in S3).
    If directory is None, the script will be added to the archive at
    ``./<basename of script>``.
    If directory is not None, the (recursive) contents of the directory will
    be added to the archive. directory is treated as the base path of the
    archive, and the script name is assumed to be a filename or relative path
    inside the directory.
    Args:
        session (boto3.Session): Boto session used to access S3.
        bucket (str): S3 bucket to which the compressed file is uploaded.
        s3_key_prefix (str): Prefix for the S3 key.
        script (str): Script filename or path.
        directory (str): Optional. Directory containing the source file. If it
            starts with "s3://", no action is taken.
        dependencies (List[str]): Optional. A list of paths to directories
            (absolute or relative) containing additional libraries that will be
            copied into /opt/ml/lib
        kms_key (str): Optional. KMS key ID used to upload objects to the bucket
            (default: None).
        s3_resource (boto3.resource("s3")): Optional. Pre-instantiated Boto3 Resource
            for S3 connections, can be used to customize the configuration,
            e.g. set the endpoint URL (default: None).
    Returns:
        sagemaker.fw_utils.UserCode: An object with the S3 bucket and key (S3 prefix) and
            script name.
    """
    if directory and directory.lower().startswith("s3://"):
        return UploadedCode(s3_prefix=directory, script_name=os.path.basename(script))

    script_name = script if directory else os.path.basename(script)
    dependencies = dependencies or []
    key = "%s/sourcedir.tar.gz" % s3_key_prefix
    tmp = tempfile.mkdtemp()

    try:
        source_files = _list_files_to_compress(script, directory) + dependencies
        tar_file = sagemaker.utils.create_tar_file(
            source_files, os.path.join(tmp, _TAR_SOURCE_FILENAME)
        )

        if kms_key:
            extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key}
        else:
            extra_args = None

        if s3_resource is None:
            s3_resource = session.resource("s3", region_name=session.region_name)
        else:
            print("Using provided s3_resource")

        s3_resource.Object(bucket, key).upload_file(tar_file, ExtraArgs=extra_args)
    finally:
        shutil.rmtree(tmp)

    return UploadedCode(s3_prefix="s3://%s/%s" % (bucket, key), script_name=script_name)


def _list_files_to_compress(script, directory):
    """
    Args:
        script:
        directory:
    """
    if directory is None:
        return [script]

    basedir = directory if directory else os.path.dirname(script)
    return [os.path.join(basedir, name) for name in os.listdir(basedir)]


def framework_name_from_image(image_name):
    # noinspection LongLine
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
            str: The framework name str: The Python version str: The image tag
            str: If the image is script mode
        """
    sagemaker_pattern = re.compile(ECR_URI_PATTERN)
    sagemaker_match = sagemaker_pattern.match(image_name)
    if sagemaker_match is None:
        return None, None, None, None
    # extract framework, python version and image tag
    # We must support both the legacy and current image name format.
    name_pattern = re.compile(
        r"^(?:sagemaker(?:-rl)?-)?(tensorflow|mxnet|chainer|pytorch|scikit-learn|xgboost)(?:-)?(scriptmode|training)?:(.*)-(.*?)-(py2|py3)$"  # noqa: E501 # pylint: disable=line-too-long
    )
    legacy_name_pattern = re.compile(r"^sagemaker-(tensorflow|mxnet)-(py2|py3)-(cpu|gpu):(.*)$")

    name_match = name_pattern.match(sagemaker_match.group(9))
    legacy_match = legacy_name_pattern.match(sagemaker_match.group(9))

    if name_match is not None:
        fw, scriptmode, ver, device, py = (
            name_match.group(1),
            name_match.group(2),
            name_match.group(3),
            name_match.group(4),
            name_match.group(5),
        )
        return fw, py, "{}-{}-{}".format(ver, device, py), scriptmode
    if legacy_match is not None:
        return (legacy_match.group(1), legacy_match.group(2), legacy_match.group(4), None)
    return None, None, None, None


def framework_version_from_tag(image_tag):
    """Extract the framework version from the image tag.
    Args:
        image_tag (str): Image tag, which should take the form
            '<framework_version>-<device>-<py_version>'
    Returns:
        str: The framework version.
    """
    tag_pattern = re.compile("^(.*)-(cpu|gpu)-(py2|py3)$")
    tag_match = tag_pattern.match(image_tag)
    return None if tag_match is None else tag_match.group(1)


def parse_s3_url(url):
    """Calls the method with the same name in the s3 module.

    :func:~sagemaker.s3.parse_s3_url

    Args:
        url: A URL, expected with an s3 scheme.

    Returns: The return value of s3.parse_s3_url, which is a tuple containing:
        str: S3 bucket name str: S3 key
    """
    return s3.parse_s3_url(url)


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
    return "/".join(filter(None, [code_location_key_prefix, model_name or training_job_name]))


def empty_framework_version_warning(default_version, latest_version):
    """
    Args:
        default_version:
        latest_version:
    """
    msgs = [EMPTY_FRAMEWORK_VERSION_WARNING.format(default_version)]
    if default_version != latest_version:
        msgs.append(LATER_FRAMEWORK_VERSION_WARNING.format(latest=latest_version))
    return " ".join(msgs)


def warn_if_parameter_server_with_multi_gpu(training_instance_type, distributions):
    """Warn the user that training will not fully leverage all the GPU
    cores if parameter server is enabled and a multi-GPU instance is selected.
    Distributed training with the default parameter server setup doesn't
    support multi-GPU instances.

    Args:
        training_instance_type (str): A string representing the type of training instance selected.
        distributions (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) For example:

            .. code:: python

                {
                    'parameter_server':
                    {
                        'enabled': True
                    }
                }


    """
    if training_instance_type == "local" or distributions is None:
        return

    is_multi_gpu_instance = (
        training_instance_type.split(".")[1].startswith("p")
        and training_instance_type not in SINGLE_GPU_INSTANCE_TYPES
    )

    ps_enabled = "parameter_server" in distributions and distributions["parameter_server"].get(
        "enabled", False
    )

    if is_multi_gpu_instance and ps_enabled:
        logger.warning(PARAMETER_SERVER_MULTI_GPU_WARNING)


def get_unsupported_framework_version_error(
    framework_name, unsupported_version, supported_versions
):
    """Return error message for unsupported framework version.

    This should also return the supported versions for customers.

    :param framework_name:
    :param unsupported_version:
    :param supported_versions:
    :return:
    """
    return UNSUPPORTED_FRAMEWORK_VERSION_ERROR.format(
        framework_name,
        unsupported_version,
        ", ".join('"{}"'.format(version) for version in supported_versions),
    )


def python_deprecation_warning(framework, latest_supported_version):
    """
    Args:
        framework:
        latest_supported_version:
    """
    return PYTHON_2_DEPRECATION_WARNING.format(
        framework=framework, latest_supported_version=latest_supported_version
    )


def _region_supports_debugger(region_name):
    """Returns boolean indicating whether the region supports Amazon SageMaker Debugger.

    Args:
        region_name (str): Name of the region to check against.

    Returns:
        bool: Whether or not the region supports Amazon SageMaker Debugger.

    """
    return region_name.lower() not in DEBUGGER_UNSUPPORTED_REGIONS
