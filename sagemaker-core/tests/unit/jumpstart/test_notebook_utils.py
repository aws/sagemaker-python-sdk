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

"""Unit tests for sagemaker.core.jumpstart.notebook_utils module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from packaging.version import Version

from sagemaker.core.jumpstart import notebook_utils
from sagemaker.core.jumpstart.enums import JumpStartScriptScope, JumpStartModelType
from sagemaker.core.jumpstart.filters import And, BooleanValues, Constant, ModelFilter, Operator
from sagemaker.core.jumpstart.types import JumpStartModelHeader


class TestCompareModelVersionTuples:
    """Test cases for _compare_model_version_tuples function"""

    def test_both_none(self):
        """Test with both tuples None"""
        result = notebook_utils._compare_model_version_tuples(None, None)
        assert result == 0

    def test_first_none(self):
        """Test with first tuple None"""
        result = notebook_utils._compare_model_version_tuples(None, ("model", "1.0"))
        assert result == -1

    def test_second_none(self):
        """Test with second tuple None"""
        result = notebook_utils._compare_model_version_tuples(("model", "1.0"), None)
        assert result == 1

    def test_different_model_ids_first_less(self):
        """Test with different model IDs, first less than second"""
        result = notebook_utils._compare_model_version_tuples(
            ("model-a", "1.0"), ("model-b", "1.0")
        )
        assert result == -1

    def test_different_model_ids_first_greater(self):
        """Test with different model IDs, first greater than second"""
        result = notebook_utils._compare_model_version_tuples(
            ("model-b", "1.0"), ("model-a", "1.0")
        )
        assert result == 1

    def test_same_model_different_versions_first_newer(self):
        """Test with same model, first version newer"""
        result = notebook_utils._compare_model_version_tuples(("model", "2.0"), ("model", "1.0"))
        assert result == -1

    def test_same_model_different_versions_second_newer(self):
        """Test with same model, second version newer"""
        result = notebook_utils._compare_model_version_tuples(("model", "1.0"), ("model", "2.0"))
        assert result == 1

    def test_same_model_same_version(self):
        """Test with same model and version"""
        result = notebook_utils._compare_model_version_tuples(("model", "1.0"), ("model", "1.0"))
        assert result == 0


class TestModelFilterInOperatorGenerator:
    """Test cases for _model_filter_in_operator_generator function"""

    def test_with_model_filters(self):
        """Test with model filters in operator"""
        operator = And("task == ic", "framework == pytorch")

        result = list(notebook_utils._model_filter_in_operator_generator(operator))
        assert len(result) == 2

    def test_without_model_filters(self):
        """Test without model filters"""
        operator = Operator([Constant(BooleanValues.TRUE)])

        result = list(notebook_utils._model_filter_in_operator_generator(operator))
        assert len(result) == 0


class TestPutResolvedBooleansIntoFilter:
    """Test cases for _put_resolved_booleans_into_filter function"""

    def test_resolve_filters(self):
        """Test resolving filters"""
        filter1 = ModelFilter("task", "ic", "==")
        operator = Operator([filter1])

        model_filters_to_resolved_values = {filter1: BooleanValues.TRUE}

        notebook_utils._put_resolved_booleans_into_filter(
            operator, model_filters_to_resolved_values
        )

        # Check that resolved value was set
        for op in notebook_utils._model_filter_in_operator_generator(operator):
            assert op.resolved_value == BooleanValues.TRUE

    def test_unknown_filter(self):
        """Test with unknown filter"""
        filter1 = ModelFilter("task", "ic", "==")
        operator = Operator([filter1])

        model_filters_to_resolved_values = {}

        notebook_utils._put_resolved_booleans_into_filter(
            operator, model_filters_to_resolved_values
        )

        # Should default to UNKNOWN
        for op in notebook_utils._model_filter_in_operator_generator(operator):
            assert op.resolved_value == BooleanValues.UNKNOWN


class TestPopulateModelFiltersToResolvedValues:
    """Test cases for _populate_model_filters_to_resolved_values function"""

    def test_populate_with_cached_values(self):
        """Test populating with cached values"""
        filter1 = ModelFilter("task", "ic", "==")
        manifest_specs_cached_values = {"task": "ic"}
        model_filters_to_resolved_values = {}
        model_filters = [filter1]

        notebook_utils._populate_model_filters_to_resolved_values(
            manifest_specs_cached_values, model_filters_to_resolved_values, model_filters
        )

        assert filter1 in model_filters_to_resolved_values
        assert model_filters_to_resolved_values[filter1] == BooleanValues.TRUE

    def test_populate_without_cached_values(self):
        """Test populating without cached values"""
        filter1 = ModelFilter("task", "ic", "==")
        manifest_specs_cached_values = {}
        model_filters_to_resolved_values = {}
        model_filters = [filter1]

        notebook_utils._populate_model_filters_to_resolved_values(
            manifest_specs_cached_values, model_filters_to_resolved_values, model_filters
        )

        # Should not add to resolved values
        assert filter1 not in model_filters_to_resolved_values


class TestExtractFrameworkTaskModel:
    """Test cases for extract_framework_task_model function"""

    def test_valid_model_id(self):
        """Test with valid model ID"""
        framework, task, name = notebook_utils.extract_framework_task_model("pytorch-ic-mobilenet")
        assert framework == "pytorch"
        assert task == "ic"
        assert name == "mobilenet"

    def test_model_id_with_hyphens(self):
        """Test with model ID containing hyphens"""
        framework, task, name = notebook_utils.extract_framework_task_model(
            "tensorflow-od-ssd-resnet-50"
        )
        assert framework == "tensorflow"
        assert task == "od"
        assert name == "ssd-resnet-50"

    def test_short_model_id(self):
        """Test with short model ID"""
        framework, task, name = notebook_utils.extract_framework_task_model("ab")
        assert framework == ""
        assert task == ""
        assert name == ""


class TestExtractModelTypeFilterRepresentation:
    """Test cases for extract_model_type_filter_representation function"""

    def test_proprietary_model(self):
        """Test with proprietary model spec key"""
        result = notebook_utils.extract_model_type_filter_representation(
            "proprietary-models/model-id/specs.json"
        )
        assert result == JumpStartModelType.PROPRIETARY.value

    def test_open_weights_model(self):
        """Test with open weights model spec key"""
        result = notebook_utils.extract_model_type_filter_representation(
            "models/model-id/specs.json"
        )
        assert result == JumpStartModelType.OPEN_WEIGHTS.value


class TestListJumpStartTasks:
    """Test cases for list_jumpstart_tasks function"""

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    def test_list_tasks(self, mock_region, mock_generate):
        """Test listing tasks"""
        mock_region.return_value = "us-west-2"
        mock_generate.return_value = iter(
            [
                ("pytorch-ic-mobilenet", "1.0"),
                ("tensorflow-od-ssd", "1.0"),
                ("pytorch-ic-resnet", "1.0"),
            ]
        )

        result = notebook_utils.list_jumpstart_tasks()

        assert "ic" in result
        assert "od" in result
        assert len(result) == 2

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    def test_list_tasks_with_filter(self, mock_region, mock_generate):
        """Test listing tasks with filter"""
        mock_region.return_value = "us-west-2"
        mock_generate.return_value = iter(
            [
                ("pytorch-ic-mobilenet", "1.0"),
            ]
        )

        result = notebook_utils.list_jumpstart_tasks(filter="framework == pytorch")

        assert isinstance(result, list)


class TestListJumpStartFrameworks:
    """Test cases for list_jumpstart_frameworks function"""

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    def test_list_frameworks(self, mock_region, mock_generate):
        """Test listing frameworks"""
        mock_region.return_value = "us-west-2"
        mock_generate.return_value = iter(
            [
                ("pytorch-ic-mobilenet", "1.0"),
                ("tensorflow-od-ssd", "1.0"),
                ("pytorch-ic-resnet", "1.0"),
            ]
        )

        result = notebook_utils.list_jumpstart_frameworks()

        assert "pytorch" in result
        assert "tensorflow" in result
        assert len(result) == 2


class TestListJumpStartScripts:
    """Test cases for list_jumpstart_scripts function"""

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    @patch("sagemaker.core.jumpstart.notebook_utils.verify_model_region_and_return_specs")
    def test_list_scripts_with_training(self, mock_verify, mock_region, mock_generate):
        """Test listing scripts with training support"""
        mock_region.return_value = "us-west-2"
        mock_generate.return_value = iter(
            [
                ("pytorch-ic-mobilenet", "1.0"),
            ]
        )

        mock_specs = Mock()
        mock_specs.training_supported = True
        mock_verify.return_value = mock_specs

        result = notebook_utils.list_jumpstart_scripts()

        assert JumpStartScriptScope.INFERENCE in result
        assert JumpStartScriptScope.TRAINING in result

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    def test_list_scripts_with_true_filter(self, mock_region, mock_generate):
        """Test listing scripts with TRUE filter"""
        mock_region.return_value = "us-west-2"

        result = notebook_utils.list_jumpstart_scripts(filter=Constant(BooleanValues.TRUE))

        # Should return all script scopes
        assert len(result) == len([e.value for e in JumpStartScriptScope])


class TestIsValidVersion:
    """Test cases for _is_valid_version function"""

    def test_valid_semantic_version(self):
        """Test with valid semantic version"""
        assert notebook_utils._is_valid_version("1.0.0") is True

    def test_valid_short_version(self):
        """Test with valid short version"""
        assert notebook_utils._is_valid_version("1.0") is True

    def test_invalid_version(self):
        """Test with invalid version"""
        assert notebook_utils._is_valid_version("invalid") is False

    def test_wildcard_version(self):
        """Test with wildcard version"""
        assert notebook_utils._is_valid_version("*") is False


class TestListJumpStartModels:
    """Test cases for list_jumpstart_models function"""

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    def test_list_models_without_versions(self, mock_region, mock_generate):
        """Test listing models without versions"""
        mock_region.return_value = "us-west-2"
        mock_generate.return_value = iter(
            [
                ("model-a", "1.0"),
                ("model-a", "2.0"),
                ("model-b", "1.0"),
            ]
        )

        result = notebook_utils.list_jumpstart_models()

        assert "model-a" in result
        assert "model-b" in result
        assert len(result) == 2

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    def test_list_models_with_versions(self, mock_region, mock_generate):
        """Test listing models with versions"""
        mock_region.return_value = "us-west-2"
        mock_generate.return_value = iter(
            [
                ("model-a", "1.0"),
                ("model-a", "2.0"),
                ("model-b", "1.0"),
            ]
        )

        result = notebook_utils.list_jumpstart_models(list_versions=True)

        assert ("model-a", "1.0") in result or ("model-a", "2.0") in result
        assert ("model-b", "1.0") in result

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    def test_list_models_with_old_versions(self, mock_region, mock_generate):
        """Test listing models with old versions"""
        mock_region.return_value = "us-west-2"
        mock_generate.return_value = iter(
            [
                ("model-a", "1.0"),
                ("model-a", "2.0"),
            ]
        )

        result = notebook_utils.list_jumpstart_models(list_versions=True, list_old_models=True)

        assert ("model-a", "1.0") in result
        assert ("model-a", "2.0") in result

    @patch("sagemaker.core.jumpstart.notebook_utils._generate_jumpstart_model_versions")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    def test_list_models_with_non_semantic_versions(self, mock_region, mock_generate):
        """Test listing models with non-semantic versions"""
        mock_region.return_value = "us-west-2"
        mock_generate.return_value = iter(
            [
                ("model-a", "v1"),
                ("model-a", "v2"),
            ]
        )

        result = notebook_utils.list_jumpstart_models(list_versions=True, list_old_models=True)

        # Should handle non-semantic versions
        assert len(result) > 0


class TestGetModelUrl:
    """Test cases for get_model_url function"""

    @patch("sagemaker.core.jumpstart.notebook_utils.validate_model_id_and_get_type")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    @patch("sagemaker.core.jumpstart.notebook_utils.verify_model_region_and_return_specs")
    def test_get_model_url(self, mock_verify, mock_region, mock_validate):
        """Test getting model URL"""
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        mock_region.return_value = "us-west-2"

        mock_specs = Mock()
        mock_specs.url = "https://example.com/model"
        mock_verify.return_value = mock_specs

        result = notebook_utils.get_model_url("test-model", "1.0")

        assert result == "https://example.com/model"

    @patch("sagemaker.core.jumpstart.notebook_utils.validate_model_id_and_get_type")
    @patch("sagemaker.core.jumpstart.notebook_utils.get_region_fallback")
    @patch("sagemaker.core.jumpstart.notebook_utils.verify_model_region_and_return_specs")
    def test_get_model_url_with_config(self, mock_verify, mock_region, mock_validate):
        """Test getting model URL with config name"""
        mock_validate.return_value = JumpStartModelType.OPEN_WEIGHTS
        mock_region.return_value = "us-west-2"

        mock_specs = Mock()
        mock_specs.url = "https://example.com/model"
        mock_verify.return_value = mock_specs

        result = notebook_utils.get_model_url("test-model", "1.0", config_name="default")

        assert result == "https://example.com/model"
