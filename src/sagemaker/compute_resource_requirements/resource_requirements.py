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
"""Defines the ResourceRequirements class that configures the compute resources for the Model."""

from __future__ import absolute_import

import logging
from typing import Optional, Dict

from sagemaker.utils import stringify_object

LOGGER = logging.getLogger("sagemaker")


class ResourceRequirements(object):
    """Configures the compute resources for a Model."""

    def __init__(
        self,
        requests: Optional[Dict[str, int]] = None,
        limits: Optional[Dict[str, int]] = None,
    ):
        """Initializes a ResourceRequirements object for an Amazon SageMaker inference component.

        Args:
            requests (dict): Basic resource to be requested, including num_cpus, memory (in MB),
                accelerator_memory (in MB), copies.
            limits (dict): Max resource limit to be requested, including memory (in MB).

            Example:
                requests = {
                     num_cpus: 1,
                     memory: 1024,
                     copies: 5
                },
                limits = {
                    memory: 2048
                }
        """
        self.requests = requests
        self.limits = limits
        self.num_accelerators: Optional[int] = None
        self.num_cpus: Optional[int] = None
        self.min_memory: Optional[int] = None
        self.max_memory: Optional[int] = None
        self.copy_count = 1

        if requests:
            if "num_accelerators" in requests:
                self.num_accelerators = requests["num_accelerators"]
            if "num_cpus" in requests:
                self.num_cpus = requests["num_cpus"]
            if "memory" in requests:
                self.min_memory = requests["memory"]
            if "copies" in requests:
                self.copy_count = requests["copies"]
        if limits:
            if "memory" in limits:
                self.max_memory = limits["memory"]

    def __str__(self) -> str:
        """Overriding str(*) method to make more human-readable."""
        return stringify_object(self)

    def __eq__(self, other) -> bool:
        """Implement equal method for easy unit testing."""
        return self.requests == other.requests and self.limits == other.limits

    def get_compute_resource_requirements(self) -> dict:
        """Returns a dict of resource requirements."""
        resource_requirements = {
            "MinMemoryRequiredInMb": self.min_memory,
        }

        if self.max_memory:
            resource_requirements.update({"MaxMemoryRequiredInMb": self.max_memory})
        if self.num_cpus:
            resource_requirements.update({"NumberOfCpuCoresRequired": self.num_cpus})
        if self.num_accelerators:
            resource_requirements.update(
                {"NumberOfAcceleratorDevicesRequired": self.num_accelerators}
            )

        return resource_requirements
