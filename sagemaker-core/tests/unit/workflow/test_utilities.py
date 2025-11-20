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

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from sagemaker.core.workflow.utilities import (
    list_to_request,
    hash_file,
    hash_files_or_dirs,
    hash_object,
    get_processing_dependencies,
    get_processing_code_hash,
    get_training_code_hash,
    validate_step_args_input,
    override_pipeline_parameter_var,
    trim_request_dict,
    _collect_parameters,
)
from sagemaker.core.workflow.entities import Entity
from sagemaker.core.workflow.parameters import Parameter
from sagemaker.core.workflow.pipeline_context import _StepArguments


class MockEntity(Entity):
    """Mock entity for testing"""
    def to_request(self):
        return {"Type": "MockEntity"}


class TestWorkflowUtilities:
    """Test cases for workflow utility functions"""

    def test_list_to_request_with_entities(self):
        """Test list_to_request with Entity objects"""
        entities = [MockEntity(), MockEntity()]
        
        result = list_to_request(entities)
        
        assert len(result) == 2
        assert all(item["Type"] == "MockEntity" for item in result)

    def test_list_to_request_with_step_collection(self):
        """Test list_to_request with StepCollection"""
        from sagemaker.mlops.workflow.step_collections import StepCollection
        
        mock_collection = Mock(spec=StepCollection)
        mock_collection.request_dicts.return_value = [{"Type": "Step1"}, {"Type": "Step2"}]
        
        result = list_to_request([mock_collection])
        
        assert len(result) == 2

    def test_list_to_request_mixed(self):
        """Test list_to_request with mixed entities and collections"""
        from sagemaker.mlops.workflow.step_collections import StepCollection
        
        mock_collection = Mock(spec=StepCollection)
        mock_collection.request_dicts.return_value = [{"Type": "Step1"}]
        
        entities = [MockEntity(), mock_collection]
        
        result = list_to_request(entities)
        
        assert len(result) == 2

    def test_hash_object(self):
        """Test hash_object produces consistent hash"""
        obj = {"key": "value", "number": 123}
        
        hash1 = hash_object(obj)
        hash2 = hash_object(obj)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 character hex string

    def test_hash_object_different_objects(self):
        """Test hash_object produces different hashes for different objects"""
        obj1 = {"key": "value1"}
        obj2 = {"key": "value2"}
        
        hash1 = hash_object(obj1)
        hash2 = hash_object(obj2)
        
        assert hash1 != hash2

    def test_hash_file(self):
        """Test hash_file produces consistent hash"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            hash1 = hash_file(temp_file)
            hash2 = hash_file(temp_file)
            
            assert hash1 == hash2
            assert len(hash1) == 64
        finally:
            os.unlink(temp_file)

    def test_hash_file_different_content(self):
        """Test hash_file produces different hashes for different content"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            f1.write("content1")
            temp_file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            f2.write("content2")
            temp_file2 = f2.name
        
        try:
            hash1 = hash_file(temp_file1)
            hash2 = hash_file(temp_file2)
            
            assert hash1 != hash2
        finally:
            os.unlink(temp_file1)
            os.unlink(temp_file2)

    def test_hash_files_or_dirs_single_file(self):
        """Test hash_files_or_dirs with single file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            result = hash_files_or_dirs([temp_file])
            
            assert len(result) == 64
        finally:
            os.unlink(temp_file)

    def test_hash_files_or_dirs_multiple_files(self):
        """Test hash_files_or_dirs with multiple files"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            f1.write("content1")
            temp_file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            f2.write("content2")
            temp_file2 = f2.name
        
        try:
            result = hash_files_or_dirs([temp_file1, temp_file2])
            
            assert len(result) == 64
        finally:
            os.unlink(temp_file1)
            os.unlink(temp_file2)

    def test_hash_files_or_dirs_directory(self):
        """Test hash_files_or_dirs with directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some files in the directory
            Path(temp_dir, "file1.txt").write_text("content1")
            Path(temp_dir, "file2.txt").write_text("content2")
            
            result = hash_files_or_dirs([temp_dir])
            
            assert len(result) == 64

    def test_hash_files_or_dirs_order_matters(self):
        """Test hash_files_or_dirs produces same hash regardless of input order"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            f1.write("content1")
            temp_file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
            f2.write("content2")
            temp_file2 = f2.name
        
        try:
            # Hash should be same regardless of order due to sorting
            hash1 = hash_files_or_dirs([temp_file1, temp_file2])
            hash2 = hash_files_or_dirs([temp_file2, temp_file1])
            
            assert hash1 == hash2
        finally:
            os.unlink(temp_file1)
            os.unlink(temp_file2)

    def test_get_processing_dependencies_empty(self):
        """Test get_processing_dependencies with empty lists"""
        result = get_processing_dependencies([None, None, None])
        
        assert result == []

    def test_get_processing_dependencies_single_list(self):
        """Test get_processing_dependencies with single list"""
        result = get_processing_dependencies([["dep1", "dep2"], None, None])
        
        assert result == ["dep1", "dep2"]

    def test_get_processing_dependencies_multiple_lists(self):
        """Test get_processing_dependencies with multiple lists"""
        result = get_processing_dependencies([
            ["dep1", "dep2"],
            ["dep3"],
            ["dep4", "dep5"]
        ])
        
        assert result == ["dep1", "dep2", "dep3", "dep4", "dep5"]

    def test_get_processing_code_hash_with_source_dir(self):
        """Test get_processing_code_hash with source_dir"""
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir, "script.py")
            code_file.write_text("print('hello')")
            
            result = get_processing_code_hash(
                code=str(code_file),
                source_dir=temp_dir,
                dependencies=[]
            )
            
            assert result is not None
            assert len(result) == 64

    def test_get_processing_code_hash_code_only(self):
        """Test get_processing_code_hash with code only"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('hello')")
            temp_file = f.name
        
        try:
            result = get_processing_code_hash(
                code=temp_file,
                source_dir=None,
                dependencies=[]
            )
            
            assert result is not None
            assert len(result) == 64
        finally:
            os.unlink(temp_file)

    def test_get_processing_code_hash_s3_uri(self):
        """Test get_processing_code_hash with S3 URI returns None"""
        result = get_processing_code_hash(
            code="s3://bucket/script.py",
            source_dir=None,
            dependencies=[]
        )
        
        assert result is None

    def test_get_processing_code_hash_with_dependencies(self):
        """Test get_processing_code_hash with dependencies"""
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = Path(temp_dir, "script.py")
            code_file.write_text("print('hello')")
            
            dep_file = Path(temp_dir, "utils.py")
            dep_file.write_text("def helper(): pass")
            
            result = get_processing_code_hash(
                code=str(code_file),
                source_dir=temp_dir,
                dependencies=[str(dep_file)]
            )
            
            assert result is not None

    def test_get_training_code_hash_with_source_dir(self):
        """Test get_training_code_hash with source_dir"""
        with tempfile.TemporaryDirectory() as temp_dir:
            entry_file = Path(temp_dir, "train.py")
            entry_file.write_text("print('training')")
            
            result = get_training_code_hash(
                entry_point=str(entry_file),
                source_dir=temp_dir,
                dependencies=[]
            )
            
            assert result is not None
            assert len(result) == 64

    def test_get_training_code_hash_entry_point_only(self):
        """Test get_training_code_hash with entry_point only"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('training')")
            temp_file = f.name
        
        try:
            result = get_training_code_hash(
                entry_point=temp_file,
                source_dir=None,
                dependencies=[]
            )
            
            assert result is not None
            assert len(result) == 64
        finally:
            os.unlink(temp_file)

    def test_get_training_code_hash_s3_uri(self):
        """Test get_training_code_hash with S3 URI returns None"""
        result = get_training_code_hash(
            entry_point="s3://bucket/train.py",
            source_dir=None,
            dependencies=[]
        )
        
        assert result is None

    def test_get_training_code_hash_pipeline_variable(self):
        """Test get_training_code_hash with pipeline variable returns None"""
        with patch("sagemaker.core.workflow.is_pipeline_variable", return_value=True):
            result = get_training_code_hash(
                entry_point="train.py",
                source_dir="source",
                dependencies=[]
            )
            
            assert result is None

    def test_validate_step_args_input_valid(self):
        """Test validate_step_args_input with valid input"""
        step_args = _StepArguments(
            caller_name="test_function",
            func=Mock(),
            func_args=[],
            func_kwargs={}
        )
        
        # Should not raise an error
        validate_step_args_input(
            step_args,
            expected_caller={"test_function"},
            error_message="Invalid input"
        )

    def test_validate_step_args_input_invalid_type(self):
        """Test validate_step_args_input with invalid type"""
        with pytest.raises(TypeError):
            validate_step_args_input(
                "not_step_args",
                expected_caller={"test_function"},
                error_message="Invalid input"
            )

    def test_validate_step_args_input_wrong_caller(self):
        """Test validate_step_args_input with wrong caller"""
        step_args = _StepArguments(
            caller_name="wrong_function",
            func=Mock(),
            func_args=[],
            func_kwargs={}
        )
        
        with pytest.raises(ValueError):
            validate_step_args_input(
                step_args,
                expected_caller={"test_function"},
                error_message="Invalid input"
            )

    def test_override_pipeline_parameter_var_decorator(self):
        """Test override_pipeline_parameter_var decorator"""
        from sagemaker.core.workflow.parameters import ParameterInteger
        
        @override_pipeline_parameter_var
        def test_func(param1, param2=None):
            return param1, param2
        
        param = ParameterInteger(name="test", default_value=10)
        
        result = test_func(param, param2=20)
        
        assert result[0] == 10  # Should use default_value
        assert result[1] == 20

    def test_override_pipeline_parameter_var_decorator_kwargs(self):
        """Test override_pipeline_parameter_var decorator with kwargs"""
        from sagemaker.core.workflow.parameters import ParameterInteger
        
        @override_pipeline_parameter_var
        def test_func(param1, param2=None):
            return param1, param2
        
        param = ParameterInteger(name="test", default_value=5)
        
        result = test_func(1, param2=param)
        
        assert result[0] == 1
        assert result[1] == 5  # Should use default_value

    def test_trim_request_dict_without_config(self):
        """Test trim_request_dict without config removes job_name"""
        request_dict = {"job_name": "test-job-123", "other": "value"}
        
        result = trim_request_dict(request_dict, "job_name", None)
        
        assert "job_name" not in result
        assert result["other"] == "value"

    def test_trim_request_dict_with_config_use_custom_prefix(self):
        """Test trim_request_dict with config and use_custom_job_prefix"""
        from sagemaker.core.workflow.pipeline_definition_config import PipelineDefinitionConfig
        
        config = Mock()
        config.pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
        
        request_dict = {"job_name": "test-job-123", "other": "value"}
        
        with patch("sagemaker.core.workflow.utilities.base_from_name", return_value="test-job"):
            result = trim_request_dict(request_dict, "job_name", config)
        
        assert result["job_name"] == "test-job"

    def test_trim_request_dict_with_config_none_job_name(self):
        """Test trim_request_dict raises error when job_name is None with use_custom_job_prefix"""
        from sagemaker.core.workflow.pipeline_definition_config import PipelineDefinitionConfig
        
        config = Mock()
        config.pipeline_definition_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
        
        request_dict = {"job_name": None, "other": "value"}
        
        with pytest.raises(ValueError, match="name field .* has not been specified"):
            trim_request_dict(request_dict, "job_name", config)

    def test_collect_parameters_decorator(self):
        """Test _collect_parameters decorator"""
        class TestClass:
            @_collect_parameters
            def __init__(self, param1, param2, param3=None):
                pass
        
        obj = TestClass("value1", "value2", param3="value3")
        
        assert obj.param1 == "value1"
        assert obj.param2 == "value2"
        assert obj.param3 == "value3"

    def test_collect_parameters_decorator_excludes_self(self):
        """Test _collect_parameters decorator excludes self"""
        class TestClass:
            @_collect_parameters
            def __init__(self, param1):
                pass
        
        obj = TestClass("value1")
        
        assert not hasattr(obj, "self")
        assert obj.param1 == "value1"

    def test_collect_parameters_decorator_excludes_depends_on(self):
        """Test _collect_parameters decorator excludes depends_on"""
        class TestClass:
            @_collect_parameters
            def __init__(self, param1, depends_on=None):
                pass
        
        obj = TestClass("value1", depends_on=["step1"])
        
        assert not hasattr(obj, "depends_on")
        assert obj.param1 == "value1"

    def test_collect_parameters_decorator_with_defaults(self):
        """Test _collect_parameters decorator with default values"""
        class TestClass:
            @_collect_parameters
            def __init__(self, param1, param2="default"):
                pass
        
        obj = TestClass("value1")
        
        assert obj.param1 == "value1"
        assert obj.param2 == "default"

    def test_collect_parameters_decorator_overrides_existing(self):
        """Test _collect_parameters decorator overrides existing attributes"""
        class TestClass:
            def __init__(self, param1):
                self.param1 = "old_value"
            
            @_collect_parameters
            def reinit(self, param1):
                pass
        
        obj = TestClass("initial")
        obj.reinit("new_value")
        
        assert obj.param1 == "new_value"
