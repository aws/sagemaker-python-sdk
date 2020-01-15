# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This file contains code related to network configuration, including
encryption, network isolation, and VPC configurations.
"""
from __future__ import absolute_import


class NetworkConfig(object):
    """Accepts network configuration parameters and provides a method to turn these parameters
    into a dictionary."""

    def __init__(self, enable_network_isolation=False, security_group_ids=None, subnets=None):
        """Initialize a ``NetworkConfig`` instance. NetworkConfig accepts network configuration
        parameters and provides a method to turn these parameters into a dictionary.

        Args:
            enable_network_isolation (bool): Boolean that determines whether to enable
                network isolation.
            security_group_ids ([str]): A list of strings representing security group IDs.
            subnets ([str]): A list of strings representing subnets.
        """
        self.enable_network_isolation = enable_network_isolation
        self.security_group_ids = security_group_ids
        self.subnets = subnets

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        network_config_request = {"EnableNetworkIsolation": self.enable_network_isolation}

        if self.security_group_ids is not None or self.subnets is not None:
            network_config_request["VpcConfig"] = {}

        if self.security_group_ids is not None:
            network_config_request["VpcConfig"]["SecurityGroupIds"] = self.security_group_ids

        if self.subnets is not None:
            network_config_request["VpcConfig"]["Subnets"] = self.subnets

        return network_config_request
