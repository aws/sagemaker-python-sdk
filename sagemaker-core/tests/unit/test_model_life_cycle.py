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

from unittest.mock import Mock

from sagemaker.core.model_life_cycle import ModelLifeCycle


def test_model_life_cycle_initialization_empty():
    """Test ModelLifeCycle initialization with no parameters."""
    life_cycle = ModelLifeCycle()
    
    assert life_cycle.stage is None
    assert life_cycle.stage_status is None
    assert life_cycle.stage_description is None


def test_model_life_cycle_to_request_dict_empty():
    """Test _to_request_dict with no parameters returns empty dict."""
    life_cycle = ModelLifeCycle()
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict == {}


def test_model_life_cycle_with_stage():
    """Test ModelLifeCycle with stage."""
    life_cycle = ModelLifeCycle(stage="Production")
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict == {"Stage": "Production"}


def test_model_life_cycle_with_stage_status():
    """Test ModelLifeCycle with stage_status."""
    life_cycle = ModelLifeCycle(stage_status="Approved")
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict == {"StageStatus": "Approved"}


def test_model_life_cycle_with_stage_description():
    """Test ModelLifeCycle with stage_description."""
    life_cycle = ModelLifeCycle(stage_description="Model ready for production deployment")
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict == {"StageDescription": "Model ready for production deployment"}


def test_model_life_cycle_all_parameters():
    """Test ModelLifeCycle with all parameters."""
    life_cycle = ModelLifeCycle(
        stage="Staging",
        stage_status="PendingApproval",
        stage_description="Model in staging for testing"
    )
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict == {
        "Stage": "Staging",
        "StageStatus": "PendingApproval",
        "StageDescription": "Model in staging for testing"
    }


def test_model_life_cycle_with_pipeline_variable():
    """Test ModelLifeCycle with PipelineVariable."""
    mock_pipeline_var = Mock()
    mock_pipeline_var.__str__ = Mock(return_value="pipeline_var")
    
    life_cycle = ModelLifeCycle(stage=mock_pipeline_var)
    
    assert life_cycle.stage == mock_pipeline_var
    request_dict = life_cycle._to_request_dict()
    assert "Stage" in request_dict


def test_model_life_cycle_partial_parameters():
    """Test ModelLifeCycle with partial parameters."""
    life_cycle = ModelLifeCycle(
        stage="Development",
        stage_description="Initial development phase"
    )
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict == {
        "Stage": "Development",
        "StageDescription": "Initial development phase"
    }
    assert "StageStatus" not in request_dict


def test_model_life_cycle_empty_string_values():
    """Test ModelLifeCycle with empty string values are excluded."""
    life_cycle = ModelLifeCycle(
        stage="",
        stage_status="Active",
        stage_description=""
    )
    
    request_dict = life_cycle._to_request_dict()
    
    # Empty strings are falsy, so they should not be included
    assert "Stage" not in request_dict
    assert "StageDescription" not in request_dict
    assert request_dict == {"StageStatus": "Active"}


def test_model_life_cycle_modification():
    """Test modifying ModelLifeCycle attributes after initialization."""
    life_cycle = ModelLifeCycle(stage="Development")
    
    life_cycle.stage = "Production"
    life_cycle.stage_status = "Active"
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict["Stage"] == "Production"
    assert request_dict["StageStatus"] == "Active"


def test_model_life_cycle_production_stage():
    """Test ModelLifeCycle with production stage."""
    life_cycle = ModelLifeCycle(
        stage="Production",
        stage_status="Active",
        stage_description="Model deployed to production"
    )
    
    assert life_cycle.stage == "Production"
    request_dict = life_cycle._to_request_dict()
    assert request_dict["Stage"] == "Production"


def test_model_life_cycle_archived_stage():
    """Test ModelLifeCycle with archived stage."""
    life_cycle = ModelLifeCycle(
        stage="Archived",
        stage_status="Inactive",
        stage_description="Model archived and no longer in use"
    )
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict["Stage"] == "Archived"
    assert request_dict["StageStatus"] == "Inactive"


def test_model_life_cycle_long_description():
    """Test ModelLifeCycle with long description."""
    long_description = "This is a very long description " * 10
    life_cycle = ModelLifeCycle(
        stage="Testing",
        stage_description=long_description
    )
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict["StageDescription"] == long_description


def test_model_life_cycle_special_characters():
    """Test ModelLifeCycle with special characters in values."""
    life_cycle = ModelLifeCycle(
        stage="Pre-Production",
        stage_status="Pending-Approval",
        stage_description="Model ready for pre-production (v2.0)"
    )
    
    request_dict = life_cycle._to_request_dict()
    
    assert request_dict["Stage"] == "Pre-Production"
    assert request_dict["StageStatus"] == "Pending-Approval"
    assert request_dict["StageDescription"] == "Model ready for pre-production (v2.0)"


def test_model_life_cycle_common_stages():
    """Test ModelLifeCycle with common stage values."""
    stages = ["Development", "Testing", "Staging", "Production", "Archived"]
    
    for stage in stages:
        life_cycle = ModelLifeCycle(stage=stage)
        request_dict = life_cycle._to_request_dict()
        assert request_dict["Stage"] == stage
