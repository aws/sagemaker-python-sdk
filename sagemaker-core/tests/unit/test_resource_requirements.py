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
"""Unit tests for sagemaker.core.compute_resource_requirements.resource_requirements module."""
from __future__ import absolute_import

import pytest
from sagemaker.core.compute_resource_requirements.resource_requirements import ResourceRequirements


class TestResourceRequirements:
    """Test ResourceRequirements class."""

    def test_init_with_requests_only(self):
        """Test initialization with requests only."""
        requests = {
            "num_cpus": 2,
            "memory": 1024,
            "copies": 3
        }
        rr = ResourceRequirements(requests=requests)
        
        assert rr.requests == requests
        assert rr.limits is None
        assert rr.num_cpus == 2
        assert rr.min_memory == 1024
        assert rr.copy_count == 3
        assert rr.max_memory is None

    def test_init_with_limits_only(self):
        """Test initialization with limits only."""
        limits = {"memory": 2048}
        rr = ResourceRequirements(limits=limits)
        
        assert rr.requests is None
        assert rr.limits == limits
        assert rr.max_memory == 2048
        assert rr.min_memory is None

    def test_init_with_requests_and_limits(self):
        """Test initialization with both requests and limits."""
        requests = {
            "num_cpus": 1,
            "memory": 1024,
            "num_accelerators": 1,
            "copies": 5
        }
        limits = {"memory": 2048}
        rr = ResourceRequirements(requests=requests, limits=limits)
        
        assert rr.num_cpus == 1
        assert rr.min_memory == 1024
        assert rr.max_memory == 2048
        assert rr.num_accelerators == 1
        assert rr.copy_count == 5

    def test_init_empty(self):
        """Test initialization with no arguments."""
        rr = ResourceRequirements()
        
        assert rr.requests is None
        assert rr.limits is None
        assert rr.num_cpus is None
        assert rr.min_memory is None
        assert rr.max_memory is None
        assert rr.copy_count == 1

    def test_str_method(self):
        """Test string representation."""
        requests = {"num_cpus": 2, "memory": 1024}
        rr = ResourceRequirements(requests=requests)
        
        result = str(rr)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_eq_method_equal(self):
        """Test equality comparison for equal objects."""
        requests = {"num_cpus": 2, "memory": 1024}
        limits = {"memory": 2048}
        
        rr1 = ResourceRequirements(requests=requests, limits=limits)
        rr2 = ResourceRequirements(requests=requests, limits=limits)
        
        assert rr1 == rr2

    def test_eq_method_not_equal(self):
        """Test equality comparison for non-equal objects."""
        rr1 = ResourceRequirements(requests={"num_cpus": 2})
        rr2 = ResourceRequirements(requests={"num_cpus": 4})
        
        assert not (rr1 == rr2)

    def test_get_compute_resource_requirements_minimal(self):
        """Test get_compute_resource_requirements with minimal config."""
        requests = {"memory": 1024}
        rr = ResourceRequirements(requests=requests)
        
        result = rr.get_compute_resource_requirements()
        
        assert result == {"MinMemoryRequiredInMb": 1024}

    def test_get_compute_resource_requirements_full(self):
        """Test get_compute_resource_requirements with all fields."""
        requests = {
            "num_cpus": 2,
            "memory": 1024,
            "num_accelerators": 1
        }
        limits = {"memory": 2048}
        rr = ResourceRequirements(requests=requests, limits=limits)
        
        result = rr.get_compute_resource_requirements()
        
        assert result["MinMemoryRequiredInMb"] == 1024
        assert result["MaxMemoryRequiredInMb"] == 2048
        assert result["NumberOfCpuCoresRequired"] == 2
        assert result["NumberOfAcceleratorDevicesRequired"] == 1

    def test_get_compute_resource_requirements_no_memory(self):
        """Test get_compute_resource_requirements with no memory specified."""
        rr = ResourceRequirements()
        
        result = rr.get_compute_resource_requirements()
        
        assert result == {"MinMemoryRequiredInMb": None}

    def test_copy_count_default(self):
        """Test that copy_count defaults to 1."""
        rr = ResourceRequirements()
        assert rr.copy_count == 1

    def test_copy_count_from_requests(self):
        """Test that copy_count is set from requests."""
        requests = {"copies": 10}
        rr = ResourceRequirements(requests=requests)
        assert rr.copy_count == 10
