# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


def name_from_base(base):
    """Append a timestamp to the provided string.

    The appended timestamp is precise to the millisecond. This function assures that the total length of the resulting
    string is not longer that 63, trimming the input parameter if necessary.

    Args:
        base (str): String used as prefix to generate the unique name.

    Returns:
        str: Input parameter with appended timestamp (no longer than 63 characters).
    """
    max_length = 63
    timestamp = sagemaker_timestamp()
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


def debug(func):
    """Print the function name and arguments for debugging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("{} args: {} kwargs: {}".format(func.__name__, args, kwargs))
        return func(*args, **kwargs)

    return wrapper
