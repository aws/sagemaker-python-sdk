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
import random
import re
import sys
import tarfile
import tempfile
import time

from datetime import datetime
from functools import wraps

import six


ECR_URI_PATTERN = r'^(\d+)(\.)dkr(\.)ecr(\.)(.+)(\.)(amazonaws.com|c2s.ic.gov)(/)(.*:.*)$'


# Use the base name of the image as the job name if the user doesn't give us one
def name_from_image(image):
    """Create a training job name based on the image name and a timestamp.

    Args:
        image (str): Image name.

    Returns:
        str: Training job name using the algorithm from the image name and a timestamp.
    """
    return name_from_base(base_name_from_image(image))


def name_from_base(base, max_length=63, short=False):
    """Append a timestamp to the provided string.

    This function assures that the total length of the resulting string is not
    longer than the specified max length, trimming the input parameter if necessary.

    Args:
        base (str): String used as prefix to generate the unique name.
        max_length (int): Maximum length for the resulting string.
        short (bool): Whether or not to use a truncated timestamp.

    Returns:
        str: Input parameter with appended timestamp.
    """
    timestamp = sagemaker_short_timestamp() if short else sagemaker_timestamp()
    trimmed_base = base[:max_length - len(timestamp) - 1]
    return '{}-{}'.format(trimmed_base, timestamp)


def unique_name_from_base(base, max_length=63):
    unique = '%04x' % random.randrange(16**4)  # 4-digit hex
    ts = str(int(time.time()))
    available_length = max_length - 2 - len(ts) - len(unique)
    trimmed = base[:available_length]
    return '{}-{}-{}'.format(trimmed, ts, unique)


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


def sagemaker_timestamp():
    """Return a timestamp with millisecond precision."""
    moment = time.time()
    moment_ms = repr(moment).split('.')[1][:3]
    return time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_ms), time.gmtime(moment))


def sagemaker_short_timestamp():
    """Return a timestamp that is relatively short in length"""
    return time.strftime('%y%m%d-%H%M')


def debug(func):
    """Print the function name and arguments for debugging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("{} args: {} kwargs: {}".format(func.__name__, args, kwargs))
        return func(*args, **kwargs)

    return wrapper


def get_config_value(key_path, config):
    if config is None:
        return None

    current_section = config
    for key in key_path.split('.'):
        if key in current_section:
            current_section = current_section[key]
        else:
            return None

    return current_section


def to_str(value):
    """Convert the input to a string, unless it is a unicode string in Python 2.

    Unicode strings are supported as native strings in Python 3, but ``str()`` cannot be
    invoked on unicode strings in Python 2, so we need to check for that case when
    converting user-specified values to strings.

    Args:
        value: The value to convert to a string.

    Returns:
        str or unicode: The string representation of the value or the unicode string itself.
    """
    if sys.version_info.major < 3 and isinstance(value, six.string_types):
        return value
    return str(value)


def extract_name_from_job_arn(arn):
    """Returns the name used in the API given a full ARN for a training job
    or hyperparameter tuning job.
    """
    slash_pos = arn.find('/')
    if slash_pos == -1:
        raise ValueError("Cannot parse invalid ARN: %s" % arn)
    return arn[(slash_pos + 1):]


def secondary_training_status_changed(current_job_description, prev_job_description):
    """Returns true if training job's secondary status message has changed.

    Args:
        current_job_desc: Current job description, returned from DescribeTrainingJob call.
        prev_job_desc: Previous job description, returned from DescribeTrainingJob call.

    Returns:
        boolean: Whether the secondary status message of a training job changed or not.

    """
    current_secondary_status_transitions = current_job_description.get('SecondaryStatusTransitions')
    if current_secondary_status_transitions is None or len(current_secondary_status_transitions) == 0:
        return False

    prev_job_secondary_status_transitions = prev_job_description.get('SecondaryStatusTransitions') \
        if prev_job_description is not None else None

    last_message = prev_job_secondary_status_transitions[-1]['StatusMessage'] \
        if prev_job_secondary_status_transitions is not None and len(prev_job_secondary_status_transitions) > 0 else ''

    message = current_job_description['SecondaryStatusTransitions'][-1]['StatusMessage']

    return message != last_message


def secondary_training_status_message(job_description, prev_description):
    """Returns a string contains last modified time and the secondary training job status message.

    Args:
        job_description: Returned response from DescribeTrainingJob call
        prev_description: Previous job description from DescribeTrainingJob call

    Returns:
        str: Job status string to be printed.

    """

    if job_description is None or job_description.get('SecondaryStatusTransitions') is None\
            or len(job_description.get('SecondaryStatusTransitions')) == 0:
        return ''

    prev_description_secondary_transitions = prev_description.get('SecondaryStatusTransitions')\
        if prev_description is not None else None
    prev_transitions_num = len(prev_description['SecondaryStatusTransitions'])\
        if prev_description_secondary_transitions is not None else 0
    current_transitions = job_description['SecondaryStatusTransitions']

    if len(current_transitions) == prev_transitions_num:
        # Secondary status is not changed but the message changed.
        transitions_to_print = current_transitions[-1:]
    else:
        # Secondary status is changed we need to print all the entries.
        transitions_to_print = current_transitions[prev_transitions_num - len(current_transitions):]

    status_strs = []
    for transition in transitions_to_print:
        message = transition['StatusMessage']
        time_str = datetime.utcfromtimestamp(
            time.mktime(job_description['LastModifiedTime'].timetuple())).strftime('%Y-%m-%d %H:%M:%S')
        status_strs.append('{} {} - {}'.format(time_str, transition['Status'], message))

    return '\n'.join(status_strs)


def download_folder(bucket_name, prefix, target, sagemaker_session):
    """Download a folder from S3 to a local path

    Args:
        bucket_name (str): S3 bucket name
        prefix (str): S3 prefix within the bucket that will be downloaded. Can be a single file.
        target (str): destination path where the downloaded items will be placed
        sagemaker_session (:class:`sagemaker.session.Session`): a sagemaker session to interact with S3.
    """
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
            # EEXIST means the folder already exists, this is safe to skip
            # anything else will be raised.
            if exc.errno != errno.EEXIST:
                raise
        obj.download_file(file_path)


def create_tar_file(source_files, target=None):
    """Create a tar file containing all the source_files

    Args:
        source_files (List[str]): List of file paths that will be contained in the tar file

    Returns:
         (str): path to created tar file

    """
    if target:
        filename = target
    else:
        _, filename = tempfile.mkstemp()

    with tarfile.open(filename, mode='w:gz') as t:
        for sf in source_files:
            # Add all files from the directory into the root of the directory structure of the tar
            t.add(sf, arcname=os.path.basename(sf))
    return filename


def download_file(bucket_name, path, target, sagemaker_session):
    """Download a Single File from S3 into a local path

    Args:
        bucket_name (str): S3 bucket name
        path (str): file path within the bucket
        target (str): destination directory for the downloaded file.
        sagemaker_session (:class:`sagemaker.session.Session`): a sagemaker session to interact with S3.
    """
    path = path.lstrip('/')
    boto_session = sagemaker_session.boto_session

    s3 = boto_session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(path, target)


def get_ecr_image_uri_prefix(account, region):
    """get prefix of ECR image URI

    Args:
        account (str): AWS account number
        region (str): AWS region name

    Returns:
        (str): URI prefix of ECR image
    """
    domain = 'c2s.ic.gov' if region == 'us-iso-east-1' else 'amazonaws.com'
    return '{}.dkr.ecr.{}.{}'.format(account, region, domain)


class DeferredError(object):
    """Stores an exception and raises it at a later time if this
    object is accessed in any way.  Useful to allow soft-dependencies on imports,
    so that the ImportError can be raised again later if code actually
    relies on the missing library.

    Example::

        try:
            import obscurelib
        except ImportError as e:
            logging.warning("Failed to import obscurelib. Obscure features will not work.")
            obscurelib = DeferredError(e)
    """

    def __init__(self, exception):
        self.exc = exception

    def __getattr__(self, name):
        """Called by Python interpreter before using any method or property
        on the object.  So this will short-circuit essentially any access to this
        object.
        """
        raise self.exc
