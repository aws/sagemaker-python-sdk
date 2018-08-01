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

import sys
import time

import re
from datetime import datetime
from functools import wraps


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
    if sys.version_info.major < 3 and isinstance(value, unicode):  # noqa: F821
        return value
    else:
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
    """Returns a string contains start time and the secondary training job status message.

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
        return current_transitions[-1]['StatusMessage']
    else:
        transitions_to_print = current_transitions[prev_transitions_num - len(current_transitions):]
        status_strs = []
        for transition in transitions_to_print:
            message = transition['StatusMessage']
            time_str = datetime.utcfromtimestamp(
                time.mktime(transition['StartTime'].timetuple())).strftime('%Y-%m-%d %H:%M:%S')
            status_strs.append('{} {} - {}'.format(time_str, transition['Status'], message))

        return '\n'.join(status_strs)


class DeferredError(object):
    """Stores an exception and raises it at a later time anytime this
    object is accessed in any way.  Useful to allow soft-dependencies on imports,
    so that the ImportError can be raised again later if code actually
    relies on the missing library.
    """

    def __init__(self, exception):
        self.exc = exception

    def __getattr__(self, name):
        """Called by Python interpreter before using any method or property
        on the object.  So this will short-circuit essentially any access to this
        object.
        """
        raise self.exc
