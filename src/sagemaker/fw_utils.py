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
"""Utility methods used by framework classes"""
from __future__ import absolute_import

import logging
import os
import re
import shutil
import tempfile
from collections import namedtuple

import sagemaker.image_uris
import sagemaker.utils

from sagemaker.deprecations import renamed_warning

logger = logging.getLogger(__name__)

_TAR_SOURCE_FILENAME = "source.tar.gz"

UploadedCode = namedtuple("UserCode", ["s3_prefix", "script_name"])
"""sagemaker.fw_utils.UserCode: An object containing the S3 prefix and script name.
This is for the source code used for the entry point with an ``Estimator``. It can be
instantiated with positional or keyword arguments.
"""

PYTHON_2_DEPRECATION_WARNING = (
    "{latest_supported_version} is the latest version of {framework} that supports "
    "Python 2. Newer versions of {framework} will only be available for Python 3."
    "Please set the argument \"py_version='py3'\" to use the Python 3 {framework} image."
)
PARAMETER_SERVER_MULTI_GPU_WARNING = (
    "If you have selected a multi-GPU training instance type "
    "and also enabled parameter server for distributed training, "
    "distributed training with the default parameter server configuration will not "
    "fully leverage all GPU cores; the parameter server will be configured to run "
    "only one worker per host regardless of the number of GPUs."
)

DEBUGGER_UNSUPPORTED_REGIONS = ("us-iso-east-1",)
PROFILER_UNSUPPORTED_REGIONS = ("us-iso-east-1",)

SINGLE_GPU_INSTANCE_TYPES = ("ml.p2.xlarge", "ml.p3.2xlarge")
SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES = (
    "ml.p3.16xlarge",
    "ml.p3dn.24xlarge",
    "ml.p4d.24xlarge",
    "local_gpu",
)
SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSIONS = {
    "tensorflow": [
        "2.3",
        "2.3.1",
        "2.3.2",
        "2.4",
        "2.4.1",
        "2.4.3",
        "2.5",
        "2.5.0",
        "2.5.1",
        "2.6",
        "2.6.0",
        "2.6.2",
    ],
    "pytorch": ["1.6", "1.6.0", "1.7", "1.7.1", "1.8", "1.8.0", "1.8.1", "1.9", "1.9.0", "1.9.1"],
}
SMDISTRIBUTED_SUPPORTED_STRATEGIES = ["dataparallel", "modelparallel"]


def validate_source_dir(script, directory):
    """Validate that the source directory exists and it contains the user script.

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


def get_mp_parameters(distribution):
    """Get the model parallelism parameters provided by the user.

    Args:
        distribution: distribution dictionary defined by the user.

    Returns:
        params: dictionary containing model parallelism parameters
        used for training.
    """
    try:
        mp_dict = distribution["smdistributed"]["modelparallel"]
    except KeyError:
        mp_dict = {}
    if mp_dict.get("enabled", False) is True:
        params = mp_dict.get("parameters", {})
        validate_mp_config(params)
        return params
    return None


def validate_mp_config(config):
    """Validate the configuration dictionary for model parallelism.

    Args:
       config (dict): Dictionary holding configuration keys and values.

    Raises:
        ValueError: If any of the keys have incorrect values.
    """

    if "partitions" not in config:
        raise ValueError("'partitions' is a required parameter.")

    def validate_positive(key):
        try:
            if not isinstance(config[key], int) or config[key] < 1:
                raise ValueError(f"The number of {key} must be a positive integer.")
        except KeyError:
            pass

    def validate_in(key, vals):
        try:
            if config[key] not in vals:
                raise ValueError(f"{key} must be a value in: {vals}.")
        except KeyError:
            pass

    def validate_bool(keys):
        validate_in(keys, [True, False])

    validate_in("pipeline", ["simple", "interleaved", "_only_forward"])
    validate_in("placement_strategy", ["spread", "cluster"])
    validate_in("optimize", ["speed", "memory"])

    for key in ["microbatches", "partitions", "active_microbatches"]:
        validate_positive(key)

    for key in [
        "auto_partition",
        "contiguous",
        "load_partition",
        "horovod",
        "ddp",
        "deterministic_server",
    ]:
        validate_bool(key)

    if "partition_file" in config and not isinstance(config.get("partition_file"), str):
        raise ValueError("'partition_file' must be a str.")

    if config.get("auto_partition") is False and "default_partition" not in config:
        raise ValueError("default_partition must be supplied if auto_partition is set to False!")

    if "default_partition" in config and config["default_partition"] >= config["partitions"]:
        raise ValueError("default_partition must be less than the number of partitions!")

    if "memory_weight" in config and (
        config["memory_weight"] > 1.0 or config["memory_weight"] < 0.0
    ):
        raise ValueError("memory_weight must be between 0.0 and 1.0!")

    if "ddp_port" in config and "ddp" not in config:
        raise ValueError("`ddp_port` needs `ddp` to be set as well")

    if "ddp_dist_backend" in config and "ddp" not in config:
        raise ValueError("`ddp_dist_backend` needs `ddp` to be set as well")

    if "ddp_port" in config:
        if not isinstance(config["ddp_port"], int) or config["ddp_port"] < 0:
            value = config["ddp_port"]
            raise ValueError(f"Invalid port number {value}.")

    if config.get("horovod", False) and config.get("ddp", False):
        raise ValueError("'ddp' and 'horovod' cannot be simultaneously enabled.")


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
    """Package source files and upload a compress tar file to S3.

    The S3 location will be ``s3://<bucket>/s3_key_prefix/sourcedir.tar.gz``.
    If directory is an S3 URI, an UploadedCode object will be returned, but
    nothing will be uploaded to S3 (this allow reuse of code already in S3).
    If directory is None, the script will be added to the archive at
    ``./<basename of script>``. If directory is not None, the (recursive) contents
    of the directory will be added to the archive. directory is treated as the base
    path of the archive, and the script name is assumed to be a filename or relative path
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
        return UploadedCode(s3_prefix=directory, script_name=script)

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
    """Placeholder docstring"""
    if directory is None:
        return [script]

    basedir = directory if directory else os.path.dirname(script)
    return [os.path.join(basedir, name) for name in os.listdir(basedir)]


def framework_name_from_image(image_uri):
    # noinspection LongLine
    """Extract the framework and Python version from the image name.

    Args:
        image_uri (str): Image URI, which should be one of the following forms:
            legacy:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<container_version>'
            legacy:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>-<py_ver>-<device>:<fw_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-<fw>:<fw_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/sagemaker-rl-<fw>:<rl_toolkit><rl_version>-<device>-<py_ver>'
            current:
            '<account>.dkr.ecr.<region>.amazonaws.com/<fw>-<image_scope>:<fw_version>-<device>-<py_ver>'

    Returns:
        tuple: A tuple containing:

            - str: The framework name
            - str: The Python version
            - str: The image tag
            - str: If the TensorFlow image is script mode
    """
    sagemaker_pattern = re.compile(sagemaker.utils.ECR_URI_PATTERN)
    sagemaker_match = sagemaker_pattern.match(image_uri)
    if sagemaker_match is None:
        return None, None, None, None

    # extract framework, python version and image tag
    # We must support both the legacy and current image name format.
    name_pattern = re.compile(
        r"""^(?:sagemaker(?:-rl)?-)?
        (tensorflow|mxnet|chainer|pytorch|scikit-learn|xgboost
        |huggingface-tensorflow|huggingface-pytorch
        |huggingface-tensorflow-trcomp|huggingface-pytorch-trcomp)(?:-)?
        (scriptmode|training)?
        :(.*)-(.*?)-(py2|py3\d*)(?:.*)$""",
        re.VERBOSE,
    )
    name_match = name_pattern.match(sagemaker_match.group(9))
    if name_match is not None:
        fw, scriptmode, ver, device, py = (
            name_match.group(1),
            name_match.group(2),
            name_match.group(3),
            name_match.group(4),
            name_match.group(5),
        )
        return fw, py, "{}-{}-{}".format(ver, device, py), scriptmode

    legacy_name_pattern = re.compile(r"^sagemaker-(tensorflow|mxnet)-(py2|py3)-(cpu|gpu):(.*)$")
    legacy_match = legacy_name_pattern.match(sagemaker_match.group(9))
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
    tag_pattern = re.compile(r"^(.*)-(cpu|gpu)-(py2|py3\d*)$")
    tag_match = tag_pattern.match(image_tag)
    return None if tag_match is None else tag_match.group(1)


def model_code_key_prefix(code_location_key_prefix, model_name, image):
    """Returns the s3 key prefix for uploading code during model deployment.

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


def warn_if_parameter_server_with_multi_gpu(training_instance_type, distribution):
    """Warn the user about training when it doesn't leverage all the GPU cores.

    Warn the user that training will not fully leverage all the GPU
    cores if parameter server is enabled and a multi-GPU instance is selected.
    Distributed training with the default parameter server setup doesn't
    support multi-GPU instances.

    Args:
        training_instance_type (str): A string representing the type of training instance selected.
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) For example:

            .. code:: python

                {
                    "parameter_server": {
                        "enabled": True
                    }
                }


    """
    if training_instance_type == "local" or distribution is None:
        return

    is_multi_gpu_instance = (
        training_instance_type == "local_gpu"
        or training_instance_type.split(".")[1].startswith("p")
    ) and training_instance_type not in SINGLE_GPU_INSTANCE_TYPES

    ps_enabled = "parameter_server" in distribution and distribution["parameter_server"].get(
        "enabled", False
    )

    if is_multi_gpu_instance and ps_enabled:
        logger.warning(PARAMETER_SERVER_MULTI_GPU_WARNING)


def validate_smdistributed(
    instance_type, framework_name, framework_version, py_version, distribution, image_uri=None
):
    """Check if smdistributed strategy is correctly invoked by the user.

    Currently, two strategies are supported: `dataparallel` or `modelparallel`.
    Validate if the user requested strategy is supported.

    Currently, only one strategy can be specified at a time. Validate if the user has requested
    more than one strategy simultaneously.

    Validate if the smdistributed dict arg is syntactically correct.

    Additionally, perform strategy-specific validations.

    Args:
        instance_type (str): A string representing the type of training instance selected.
        framework_name (str): A string representing the name of framework selected.
        framework_version (str): A string representing the framework version selected.
        py_version (str): A string representing the python version selected.
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) For example:

            .. code:: python

                {
                    "smdistributed": {
                        "dataparallel": {
                            "enabled": True
                        }
                    }
                }
        image_uri (str): A string representing a Docker image URI.

    Raises:
        ValueError: if distribution dictionary isn't correctly formatted or
            multiple strategies are requested simultaneously or
            an unsupported strategy is requested or
            strategy-specific inputs are incorrect/unsupported
    """
    if "smdistributed" not in distribution:
        # Distribution strategy other than smdistributed is selected
        return

    # distribution contains smdistributed
    smdistributed = distribution["smdistributed"]
    if not isinstance(smdistributed, dict):
        raise ValueError("smdistributed strategy requires a dictionary")

    if len(smdistributed) > 1:
        # more than 1 smdistributed strategy requested by the user
        err_msg = (
            "Cannot use more than 1 smdistributed strategy. \n"
            "Choose one of the following supported strategies:"
            f"{SMDISTRIBUTED_SUPPORTED_STRATEGIES}"
        )
        raise ValueError(err_msg)

    # validate if smdistributed strategy is supported
    # currently this for loop essentially checks for only 1 key
    for strategy in smdistributed:
        if strategy not in SMDISTRIBUTED_SUPPORTED_STRATEGIES:
            err_msg = (
                f"Invalid smdistributed strategy provided: {strategy} \n"
                f"Supported strategies: {SMDISTRIBUTED_SUPPORTED_STRATEGIES}"
            )
            raise ValueError(err_msg)

    # smdataparallel-specific input validation
    if "dataparallel" in smdistributed:
        _validate_smdataparallel_args(
            instance_type, framework_name, framework_version, py_version, distribution, image_uri
        )


def _validate_smdataparallel_args(
    instance_type, framework_name, framework_version, py_version, distribution, image_uri=None
):
    """Check if request is using unsupported arguments.

    Validate if user specifies a supported instance type, framework version, and python
    version.

    Args:
        instance_type (str): A string representing the type of training instance selected. Ex: `ml.p3.16xlarge`
        framework_name (str): A string representing the name of framework selected. Ex: `tensorflow`
        framework_version (str): A string representing the framework version selected. Ex: `2.3.1`
        py_version (str): A string representing the python version selected. Ex: `py3`
        distribution (dict): A dictionary with information to enable distributed training.
            (Defaults to None if distributed training is not enabled.) Ex:

            .. code:: python

                {
                    "smdistributed": {
                        "dataparallel": {
                            "enabled": True
                        }
                    }
                }
        image_uri (str): A string representing a Docker image URI.

    Raises:
        ValueError: if
            (`instance_type` is not in SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES or
            `py_version` is not python3 or
            `framework_version` is not in SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSION
    """
    smdataparallel_enabled = (
        distribution.get("smdistributed").get("dataparallel").get("enabled", False)
    )

    if not smdataparallel_enabled:
        return

    is_instance_type_supported = instance_type in SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES

    err_msg = ""

    if not is_instance_type_supported:
        # instance_type is required
        err_msg += (
            f"Provided instance_type {instance_type} is not supported by smdataparallel.\n"
            "Please specify one of the supported instance types:"
            f"{SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES}\n"
        )

    if not image_uri:
        # ignore framework_version & py_version if image_uri is set
        # in case image_uri is not set, then both are mandatory
        supported = SM_DATAPARALLEL_SUPPORTED_FRAMEWORK_VERSIONS[framework_name]
        if framework_version not in supported:
            err_msg += (
                f"Provided framework_version {framework_version} is not supported by"
                " smdataparallel.\n"
                f"Please specify one of the supported framework versions: {supported} \n"
            )

        if "py3" not in py_version:
            err_msg += (
                f"Provided py_version {py_version} is not supported by smdataparallel.\n"
                "Please specify py_version>=py3"
            )

    if err_msg:
        raise ValueError(err_msg)


def python_deprecation_warning(framework, latest_supported_version):
    """Placeholder docstring"""
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


def _region_supports_profiler(region_name):
    """Returns bool indicating whether region supports Amazon SageMaker Debugger profiling feature.

    Args:
        region_name (str): Name of the region to check against.

    Returns:
        bool: Whether or not the region supports Amazon SageMaker Debugger profiling feature.

    """
    return region_name.lower() not in PROFILER_UNSUPPORTED_REGIONS


def validate_version_or_image_args(framework_version, py_version, image_uri):
    """Checks if version or image arguments are specified.

    Validates framework and model arguments to enforce version or image specification.

    Args:
        framework_version (str): The version of the framework.
        py_version (str): The version of Python.
        image_uri (str): The URI of the image.

    Raises:
        ValueError: if `image_uri` is None and either `framework_version` or `py_version` is
            None.
    """
    if (framework_version is None or py_version is None) and image_uri is None:
        raise ValueError(
            "framework_version or py_version was None, yet image_uri was also None. "
            "Either specify both framework_version and py_version, or specify image_uri."
        )


def create_image_uri(
    region,
    framework,
    instance_type,
    framework_version,
    py_version=None,
    account=None,  # pylint: disable=W0613
    accelerator_type=None,
    optimized_families=None,  # pylint: disable=W0613
):
    """Deprecated method. Please use sagemaker.image_uris.retrieve().

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
        optimized_families (str): Deprecated. A no-op argument.

    Returns:
        the image uri
    """
    renamed_warning("The method create_image_uri")
    return sagemaker.image_uris.retrieve(
        framework=framework,
        region=region,
        version=framework_version,
        py_version=py_version,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
    )
