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
