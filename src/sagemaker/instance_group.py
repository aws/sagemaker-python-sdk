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
"""This file defines instance group for heterogeneous cluster."""
from __future__ import absolute_import


class InstanceGroup(object):
    """Accepts instance group parameters for conversion to request dict.

    The `_to_request_dict` provides a method to turn the parameters into a dict.
    """

    def __init__(
        self,
        instance_group_name=None,
        instance_type=None,
        instance_count=None,
    ):
        """Initialize a ``InstanceGroup`` instance.

        InstanceGroup accepts instance group parameters and provides a method to turn
        these parameters into a dictionary.

        Args:
            instance_group_name (str): Name of the instance group.
            instance_type (str): Type of EC2 instance to use in the instance group,
                for example, 'ml.c4.xlarge'.
            instance_count (int): Number of EC2 instances to use in the instance group.
        """
        self.instance_group_name = instance_group_name
        self.instance_type = instance_type
        self.instance_count = instance_count

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        return {
            "InstanceGroupName": self.instance_group_name,
            "InstanceType": self.instance_type,
            "InstanceCount": self.instance_count,
        }
