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

from sagemaker.core.instance_group import InstanceGroup


def test_instance_group_initialization():
    """Test InstanceGroup initialization with all parameters."""
    instance_group = InstanceGroup(
        instance_group_name="worker-group",
        instance_type="ml.p3.2xlarge",
        instance_count=4
    )
    
    assert instance_group.instance_group_name == "worker-group"
    assert instance_group.instance_type == "ml.p3.2xlarge"
    assert instance_group.instance_count == 4


def test_instance_group_initialization_with_none():
    """Test InstanceGroup initialization with None values."""
    instance_group = InstanceGroup()
    
    assert instance_group.instance_group_name is None
    assert instance_group.instance_type is None
    assert instance_group.instance_count is None


def test_instance_group_to_request_dict():
    """Test _to_request_dict generates correct dictionary."""
    instance_group = InstanceGroup(
        instance_group_name="training-group",
        instance_type="ml.g4dn.xlarge",
        instance_count=2
    )
    
    request_dict = instance_group._to_request_dict()
    
    assert request_dict == {
        "InstanceGroupName": "training-group",
        "InstanceType": "ml.g4dn.xlarge",
        "InstanceCount": 2
    }


def test_instance_group_to_request_dict_with_none():
    """Test _to_request_dict with None values."""
    instance_group = InstanceGroup()
    
    request_dict = instance_group._to_request_dict()
    
    assert request_dict == {
        "InstanceGroupName": None,
        "InstanceType": None,
        "InstanceCount": None
    }


def test_instance_group_single_instance():
    """Test InstanceGroup with single instance."""
    instance_group = InstanceGroup(
        instance_group_name="single-node",
        instance_type="ml.m5.xlarge",
        instance_count=1
    )
    
    assert instance_group.instance_count == 1
    request_dict = instance_group._to_request_dict()
    assert request_dict["InstanceCount"] == 1


def test_instance_group_large_cluster():
    """Test InstanceGroup with large instance count."""
    instance_group = InstanceGroup(
        instance_group_name="large-cluster",
        instance_type="ml.c5.18xlarge",
        instance_count=100
    )
    
    assert instance_group.instance_count == 100
    request_dict = instance_group._to_request_dict()
    assert request_dict["InstanceCount"] == 100


def test_instance_group_gpu_instance():
    """Test InstanceGroup with GPU instance type."""
    instance_group = InstanceGroup(
        instance_group_name="gpu-workers",
        instance_type="ml.p4d.24xlarge",
        instance_count=8
    )
    
    assert instance_group.instance_type == "ml.p4d.24xlarge"
    request_dict = instance_group._to_request_dict()
    assert request_dict["InstanceType"] == "ml.p4d.24xlarge"


def test_instance_group_cpu_instance():
    """Test InstanceGroup with CPU instance type."""
    instance_group = InstanceGroup(
        instance_group_name="cpu-workers",
        instance_type="ml.c5.2xlarge",
        instance_count=5
    )
    
    assert instance_group.instance_type == "ml.c5.2xlarge"
    request_dict = instance_group._to_request_dict()
    assert request_dict["InstanceType"] == "ml.c5.2xlarge"


def test_instance_group_name_with_special_chars():
    """Test InstanceGroup with special characters in name."""
    instance_group = InstanceGroup(
        instance_group_name="worker-group-1",
        instance_type="ml.m5.large",
        instance_count=3
    )
    
    assert instance_group.instance_group_name == "worker-group-1"
    request_dict = instance_group._to_request_dict()
    assert request_dict["InstanceGroupName"] == "worker-group-1"


def test_instance_group_modification():
    """Test modifying InstanceGroup attributes after initialization."""
    instance_group = InstanceGroup(
        instance_group_name="initial-group",
        instance_type="ml.m5.xlarge",
        instance_count=2
    )
    
    # Modify attributes
    instance_group.instance_group_name = "modified-group"
    instance_group.instance_type = "ml.m5.2xlarge"
    instance_group.instance_count = 4
    
    assert instance_group.instance_group_name == "modified-group"
    assert instance_group.instance_type == "ml.m5.2xlarge"
    assert instance_group.instance_count == 4
    
    request_dict = instance_group._to_request_dict()
    assert request_dict["InstanceGroupName"] == "modified-group"
    assert request_dict["InstanceType"] == "ml.m5.2xlarge"
    assert request_dict["InstanceCount"] == 4
