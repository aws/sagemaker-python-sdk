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
"""Placeholder docstring"""
from __future__ import absolute_import

import contextlib
import errno
import logging
import os
import random
import re
import shutil
import tarfile
import tempfile
import time
from datetime import datetime

import botocore
from six.moves.urllib import parse

from sagemaker import deprecations


ECR_URI_PATTERN = r"^(\d+)(\.)dkr(\.)ecr(\.)(.+)(\.)(.*)(/)(.*:.*)$"
MAX_BUCKET_PATHS_COUNT = 5
S3_PREFIX = "s3://"
HTTP_PREFIX = "http://"
HTTPS_PREFIX = "https://"
DEFAULT_SLEEP_TIME_SECONDS = 10

logger = logging.getLogger(__name__)


# Use the base name of the image as the job name if the user doesn't give us one
def name_from_image(image, max_length=63):
    """Create a training job name based on the image name and a timestamp.

    Args:
        image (str): Image name.

    Returns:
        str: Training job name using the algorithm from the image name and a
            timestamp.
        max_length (int): Maximum length for the resulting string (default: 63).
    """
    return name_from_base(base_name_from_image(image), max_length=max_length)


def name_from_base(base, max_length=63, short=False):
    """Append a timestamp to the provided string.

    This function assures that the total length of the resulting string is
    not longer than the specified max length, trimming the input parameter if
    necessary.

    Args:
        base (str): String used as prefix to generate the unique name.
        max_length (int): Maximum length for the resulting string (default: 63).
        short (bool): Whether or not to use a truncated timestamp (default: False).

    Returns:
        str: Input parameter with appended timestamp.
    """
    timestamp = sagemaker_short_timestamp() if short else sagemaker_timestamp()
    trimmed_base = base[: max_length - len(timestamp) - 1]
    return "{}-{}".format(trimmed_base, timestamp)


def unique_name_from_base(base, max_length=63):
    """Placeholder Docstring"""
    unique = "%04x" % random.randrange(16 ** 4)  # 4-digit hex
    ts = str(int(time.time()))
    available_length = max_length - 2 - len(ts) - len(unique)
    trimmed = base[:available_length]
    return "{}-{}-{}".format(trimmed, ts, unique)


def base_name_from_image(image):
    """Extract the base name of the image to use as the 'algorithm name' for the job.

    Args:
        image (str): Image name.

    Returns:
        str: Algorithm name, as extracted from the image name.
    """
    m = re.match("^(.+/)?([^:/]+)(:[^:]+)?$", image)
    algo_name = m.group(2) if m else image
    return algo_name


def base_from_name(name):
    """Extract the base name of the resource name (for use with future resource name generation).

    This function looks for timestamps that match the ones produced by
    :func:`~sagemaker.utils.name_from_base`.

    Args:
        name (str): The resource name.

    Returns:
        str: The base name, as extracted from the resource name.
    """
    m = re.match(r"^(.+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}|\d{6}-\d{4})", name)
    return m.group(1) if m else name


def sagemaker_timestamp():
    """Return a timestamp with millisecond precision."""
    moment = time.time()
    moment_ms = repr(moment).split(".")[1][:3]
    return time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_ms), time.gmtime(moment))


def sagemaker_short_timestamp():
    """Return a timestamp that is relatively short in length"""
    return time.strftime("%y%m%d-%H%M")


def build_dict(key, value):
    """Return a dict of key and value pair if value is not None, otherwise return an empty dict.

    Args:
        key (str): input key
        value (str): input value

    Returns:
        dict: dict of key and value or an empty dict.
    """
    if value:
        return {key: value}
    return {}


def get_config_value(key_path, config):
    """Placeholder Docstring"""
    if config is None:
        return None

    current_section = config
    for key in key_path.split("."):
        if key in current_section:
            current_section = current_section[key]
        else:
            return None

    return current_section


def get_short_version(framework_version):
    """Return short version in the format of x.x

    Args:
        framework_version: The version string to be shortened.

    Returns:
        str: The short version string
    """
    return ".".join(framework_version.split(".")[:2])


def secondary_training_status_changed(current_job_description, prev_job_description):
    """Returns true if training job's secondary status message has changed.

    Args:
        current_job_description: Current job description, returned from DescribeTrainingJob call.
        prev_job_description: Previous job description, returned from DescribeTrainingJob call.

    Returns:
        boolean: Whether the secondary status message of a training job changed
        or not.
    """
    current_secondary_status_transitions = current_job_description.get("SecondaryStatusTransitions")
    if (
        current_secondary_status_transitions is None
        or len(current_secondary_status_transitions) == 0
    ):
        return False

    prev_job_secondary_status_transitions = (
        prev_job_description.get("SecondaryStatusTransitions")
        if prev_job_description is not None
        else None
    )

    last_message = (
        prev_job_secondary_status_transitions[-1]["StatusMessage"]
        if prev_job_secondary_status_transitions is not None
        and len(prev_job_secondary_status_transitions) > 0
        else ""
    )

    message = current_job_description["SecondaryStatusTransitions"][-1]["StatusMessage"]

    return message != last_message


def secondary_training_status_message(job_description, prev_description):
    """Returns a string contains last modified time and the secondary training job status message.

    Args:
        job_description: Returned response from DescribeTrainingJob call
        prev_description: Previous job description from DescribeTrainingJob call

    Returns:
        str: Job status string to be printed.
    """

    if (
        job_description is None
        or job_description.get("SecondaryStatusTransitions") is None
        or len(job_description.get("SecondaryStatusTransitions")) == 0
    ):
        return ""

    prev_description_secondary_transitions = (
        prev_description.get("SecondaryStatusTransitions") if prev_description is not None else None
    )
    prev_transitions_num = (
        len(prev_description["SecondaryStatusTransitions"])
        if prev_description_secondary_transitions is not None
        else 0
    )
    current_transitions = job_description["SecondaryStatusTransitions"]

    if len(current_transitions) == prev_transitions_num:
        # Secondary status is not changed but the message changed.
        transitions_to_print = current_transitions[-1:]
    else:
        # Secondary status is changed we need to print all the entries.
        transitions_to_print = current_transitions[
            prev_transitions_num - len(current_transitions) :
        ]

    status_strs = []
    for transition in transitions_to_print:
        message = transition["StatusMessage"]
        time_str = datetime.utcfromtimestamp(
            time.mktime(job_description["LastModifiedTime"].timetuple())
        ).strftime("%Y-%m-%d %H:%M:%S")
        status_strs.append("{} {} - {}".format(time_str, transition["Status"], message))

    return "\n".join(status_strs)


def download_folder(bucket_name, prefix, target, sagemaker_session):
    """Download a folder from S3 to a local path

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 prefix within the bucket that will be downloaded. Can
            be a single file.
        target (str): destination path where the downloaded items will be placed
        sagemaker_session (sagemaker.session.Session): a sagemaker session to
            interact with S3.
    """
    boto_session = sagemaker_session.boto_session
    s3 = boto_session.resource("s3", region_name=boto_session.region_name)

    prefix = prefix.lstrip("/")

    # Try to download the prefix as an object first, in case it is a file and not a 'directory'.
    # Do this first, in case the object has broader permissions than the bucket.
    if not prefix.endswith("/"):
        try:
            file_destination = os.path.join(target, os.path.basename(prefix))
            s3.Object(bucket_name, prefix).download_file(file_destination)
            return
        except botocore.exceptions.ClientError as e:
            err_info = e.response["Error"]
            if err_info["Code"] == "404" and err_info["Message"] == "Not Found":
                # S3 also throws this error if the object is a folder,
                # so assume that is the case here, and then raise for an actual 404 later.
                pass
            else:
                raise

    _download_files_under_prefix(bucket_name, prefix, target, s3)


def _download_files_under_prefix(bucket_name, prefix, target, s3):
    """Download all S3 files which match the given prefix

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 prefix within the bucket that will be downloaded
        target (str): destination path where the downloaded items will be placed
        s3 (boto3.resources.base.ServiceResource): S3 resource
    """
    bucket = s3.Bucket(bucket_name)
    for obj_sum in bucket.objects.filter(Prefix=prefix):
        # if obj_sum is a folder object skip it.
        if obj_sum.key.endswith("/"):
            continue
        obj = s3.Object(obj_sum.bucket_name, obj_sum.key)
        s3_relative_path = obj_sum.key[len(prefix) :].lstrip("/")
        file_path = os.path.join(target, s3_relative_path)

        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:
            # EEXIST means the folder already exists, this is safe to skip
            # anything else will be raised.
            if exc.errno != errno.EEXIST:
                raise
        obj.download_file(file_path)


def create_tar_file(source_files, target=None):
    """Create a tar file containing all the source_files

    Args:
        source_files: (List[str]): List of file paths that will be contained in the tar file
        target:

    Returns:
        (str): path to created tar file
    """
    if target:
        filename = target
    else:
        _, filename = tempfile.mkstemp()

    with tarfile.open(filename, mode="w:gz") as t:
        for sf in source_files:
            # Add all files from the directory into the root of the directory structure of the tar
            t.add(sf, arcname=os.path.basename(sf))
    return filename


@contextlib.contextmanager
def _tmpdir(suffix="", prefix="tmp"):
    """Create a temporary directory with a context manager.

    The file is deleted when the context exits.
    The prefix, suffix, and dir arguments are the same as for mkstemp().

    Args:
        suffix (str): If suffix is specified, the file name will end with that
            suffix, otherwise there will be no suffix.
        prefix (str): If prefix is specified, the file name will begin with that
            prefix; otherwise, a default prefix is used.

    Returns:
        str: path to the directory
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=None)
    yield tmp
    shutil.rmtree(tmp)


def repack_model(
    inference_script,
    source_directory,
    dependencies,
    model_uri,
    repacked_model_uri,
    sagemaker_session,
    kms_key=None,
):
    """Unpack model tarball and creates a new model tarball with the provided code script.

    This function does the following: - uncompresses model tarball from S3 or
    local system into a temp folder - replaces the inference code from the model
    with the new code provided - compresses the new model tarball and saves it
    in S3 or local file system

    Args:
        inference_script (str): path or basename of the inference script that
            will be packed into the model
        source_directory (str): path including all the files that will be packed
            into the model
        dependencies (list[str]): A list of paths to directories (absolute or
            relative) with any additional libraries that will be exported to the
            container (default: []). The library folders will be copied to
            SageMaker in the same folder where the entrypoint is copied.
            Example

                The following call >>> Estimator(entry_point='train.py',
                dependencies=['my/libs/common', 'virtual-env']) results in the
                following inside the container:

                >>> $ ls

                >>> opt/ml/code
                >>>     |------ train.py
                >>>     |------ common
                >>>     |------ virtual-env
        model_uri (str): S3 or file system location of the original model tar
        repacked_model_uri (str): path or file system location where the new
            model will be saved
        sagemaker_session (sagemaker.session.Session): a sagemaker session to
            interact with S3.
        kms_key (str): KMS key ARN for encrypting the repacked model file

    Returns:
        str: path to the new packed model
    """
    dependencies = dependencies or []

    with _tmpdir() as tmp:
        model_dir = _extract_model(model_uri, sagemaker_session, tmp)

        _create_or_update_code_dir(
            model_dir, inference_script, source_directory, dependencies, sagemaker_session, tmp
        )

        tmp_model_path = os.path.join(tmp, "temp-model.tar.gz")
        with tarfile.open(tmp_model_path, mode="w:gz") as t:
            t.add(model_dir, arcname=os.path.sep)

        _save_model(repacked_model_uri, tmp_model_path, sagemaker_session, kms_key=kms_key)


def _save_model(repacked_model_uri, tmp_model_path, sagemaker_session, kms_key):
    """Placeholder docstring"""
    if repacked_model_uri.lower().startswith("s3://"):
        url = parse.urlparse(repacked_model_uri)
        bucket, key = url.netloc, url.path.lstrip("/")
        new_key = key.replace(os.path.basename(key), os.path.basename(repacked_model_uri))

        if kms_key:
            extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key}
        else:
            extra_args = None
        sagemaker_session.boto_session.resource(
            "s3", region_name=sagemaker_session.boto_region_name
        ).Object(bucket, new_key).upload_file(tmp_model_path, ExtraArgs=extra_args)
    else:
        shutil.move(tmp_model_path, repacked_model_uri.replace("file://", ""))


def _create_or_update_code_dir(
    model_dir, inference_script, source_directory, dependencies, sagemaker_session, tmp
):
    """Placeholder docstring"""
    code_dir = os.path.join(model_dir, "code")
    if source_directory and source_directory.lower().startswith("s3://"):
        local_code_path = os.path.join(tmp, "local_code.tar.gz")
        download_file_from_url(source_directory, local_code_path, sagemaker_session)

        with tarfile.open(name=local_code_path, mode="r:gz") as t:
            t.extractall(path=code_dir)

    elif source_directory:
        if os.path.exists(code_dir):
            shutil.rmtree(code_dir)
        shutil.copytree(source_directory, code_dir)
    else:
        if not os.path.exists(code_dir):
            os.mkdir(code_dir)
        try:
            shutil.copy2(inference_script, code_dir)
        except FileNotFoundError:
            if os.path.exists(os.path.join(code_dir, inference_script)):
                pass
            else:
                raise

    for dependency in dependencies:
        lib_dir = os.path.join(code_dir, "lib")
        if os.path.isdir(dependency):
            shutil.copytree(dependency, os.path.join(lib_dir, os.path.basename(dependency)))
        else:
            if not os.path.exists(lib_dir):
                os.mkdir(lib_dir)
            shutil.copy2(dependency, lib_dir)


def _extract_model(model_uri, sagemaker_session, tmp):
    """Placeholder docstring"""
    tmp_model_dir = os.path.join(tmp, "model")
    os.mkdir(tmp_model_dir)
    if model_uri.lower().startswith("s3://"):
        local_model_path = os.path.join(tmp, "tar_file")
        download_file_from_url(model_uri, local_model_path, sagemaker_session)
    else:
        local_model_path = model_uri.replace("file://", "")
    with tarfile.open(name=local_model_path, mode="r:gz") as t:
        t.extractall(path=tmp_model_dir)
    return tmp_model_dir


def download_file_from_url(url, dst, sagemaker_session):
    """Placeholder docstring"""
    url = parse.urlparse(url)
    bucket, key = url.netloc, url.path.lstrip("/")

    download_file(bucket, key, dst, sagemaker_session)


def download_file(bucket_name, path, target, sagemaker_session):
    """Download a Single File from S3 into a local path

    Args:
        bucket_name (str): S3 bucket name
        path (str): file path within the bucket
        target (str): destination directory for the downloaded file.
        sagemaker_session (sagemaker.session.Session): a sagemaker session to
            interact with S3.
    """
    path = path.lstrip("/")
    boto_session = sagemaker_session.boto_session

    s3 = boto_session.resource("s3", region_name=sagemaker_session.boto_region_name)
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(path, target)


def sts_regional_endpoint(region):
    """Get the AWS STS endpoint specific for the given region.

    We need this function because the AWS SDK does not yet honor
    the ``region_name`` parameter when creating an AWS STS client.

    For the list of regional endpoints, see
    https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_enable-regions.html#id_credentials_region-endpoints.

    Args:
        region (str): AWS region name

    Returns:
        str: AWS STS regional endpoint
    """
    endpoint_data = _botocore_resolver().construct_endpoint("sts", region)
    return "https://{}".format(endpoint_data["hostname"])


def retries(max_retry_count, exception_message_prefix, seconds_to_sleep=DEFAULT_SLEEP_TIME_SECONDS):
    """Retries until max retry count is reached.

    Args:
        max_retry_count (int): The retry count.
        exception_message_prefix (str): The message to include in the exception on failure.
        seconds_to_sleep (int): The number of seconds to sleep between executions.

    """
    for i in range(max_retry_count):
        yield i
        time.sleep(seconds_to_sleep)

    raise Exception(
        "'{}' has reached the maximum retry count of {}".format(
            exception_message_prefix, max_retry_count
        )
    )


def _botocore_resolver():
    """Get the DNS suffix for the given region.

    Args:
        region (str): AWS region name

    Returns:
        str: the DNS suffix
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data("endpoints"))


def _aws_partition(region):
    """Given a region name (ex: "cn-north-1"), return the corresponding aws partition ("aws-cn").

    Args:
        region (str): The region name for which to return the corresponding partition.
        Ex: "cn-north-1"

    Returns:
        str: partition corresponding to the region name passed in. Ex: "aws-cn"
    """
    endpoint_data = _botocore_resolver().construct_endpoint("sts", region)
    return endpoint_data["partition"]


class DeferredError(object):
    """Stores an exception and raises it at a later time if this object is accessed in any way.

    Useful to allow soft-dependencies on imports, so that the ImportError can be raised again
    later if code actually relies on the missing library.

    Example::

        try:
            import obscurelib
        except ImportError as e:
            logger.warning("Failed to import obscurelib. Obscure features will not work.")
            obscurelib = DeferredError(e)
    """

    def __init__(self, exception):
        """Placeholder docstring"""
        self.exc = exception

    def __getattr__(self, name):
        """Called by Python interpreter before using any method or property on the object.

        So this will short-circuit essentially any access to this object.

        Args:
            name:
        """
        raise self.exc


def _module_import_error(py_module, feature, extras):
    """Return error message for module import errors, provide installation details.

    Args:
        py_module (str): Module that failed to be imported
        feature (str): Affected SageMaker feature
        extras (str): Name of the `extras_require` to install the relevant dependencies

    Returns:
        str: Error message with installation instructions.
    """
    error_msg = (
        "Failed to import {}. {} features will be impaired or broken. "
        "Please run \"pip install 'sagemaker[{}]'\" "
        "to install all required dependencies."
    )
    return error_msg.format(py_module, feature, extras)


get_ecr_image_uri_prefix = deprecations.removed_function("get_ecr_image_uri_prefix")
