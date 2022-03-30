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
from __future__ import absolute_import
import logging
from functools import wraps

from tests.conftest import NO_P3_REGIONS, NO_M4_REGIONS
from sagemaker.exceptions import CapacityError


def gpu_list(region):
    if region in NO_P3_REGIONS:
        return ["ml.p2.xlarge"]
    else:
        return ["ml.p3.2xlarge", "ml.p2.xlarge"]


def cpu_list(region):
    if region in NO_M4_REGIONS:
        return ["ml.m5.xlarge"]
    else:
        return ["ml.m4.xlarge", "ml.m5.xlarge"]


def retry_with_instance_list(instance_list):
    """Decorator for running an Integ test with an instance_list and
    break on first success

    Args:
        instance_list (list): List of Compute instances for integ test.
    Usage:
        @retry_with_instance_list(instance_list=["ml.g3.2", "ml.g2"])
        def sample_function():
            print("xxxx....")
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not (instance_list and isinstance(instance_list, list)):
                error_string = f"Parameter instance_list = {instance_list} \
                is expected to be a non-empty list of instance types."
                raise Exception(error_string)
            for i_type in instance_list:
                logging.info(f"Using the instance type: {i_type} for {func.__name__}")
                try:
                    kwargs.update({"instance_type": i_type})
                    func(*args, **kwargs)
                except CapacityError as e:
                    if i_type != instance_list[-1]:
                        logging.warning(
                            "Failure using instance type: {}. {}".format(i_type, str(e))
                        )
                        continue
                    else:
                        raise
                break

        return wrapper

    return decorator
