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
"""Unit tests for workflow steps."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.mlops.workflow.steps import CacheConfig


def test_cache_config_default():
    config = CacheConfig()
    assert config.enable_caching is False
    assert config.expire_after is None


def test_cache_config_enabled():
    config = CacheConfig(enable_caching=True, expire_after="P30D")
    assert config.enable_caching is True
    assert config.expire_after == "P30D"
    assert config.config == {"CacheConfig": {"Enabled": True, "ExpireAfter": "P30D"}}


def test_training_step_requires_step_args():
    from sagemaker.mlops.workflow.steps import TrainingStep
    
    step = TrainingStep(name="training-step", step_args=None)
    with pytest.raises(ValueError, match="step_args input is required"):
        _ = step.arguments


def test_processing_step_requires_step_args():
    from sagemaker.mlops.workflow.steps import ProcessingStep
    
    with pytest.raises(ValueError, match="step_args is required"):
        ProcessingStep(name="processing-step", step_args=None)


def test_transform_step_requires_step_args():
    from sagemaker.mlops.workflow.steps import TransformStep
    
    with pytest.raises(ValueError, match="step_args is required"):
        TransformStep(name="transform-step", step_args=None)


def test_step_add_depends_on():
    from sagemaker.mlops.workflow.steps import Step
    
    step = Mock(spec=Step)
    step.name = "test-step"
    step._depends_on = None
    
    Step.add_depends_on(step, ["other-step"])
    assert step._depends_on == ["other-step"]



def test_step_depends_on_setter_with_none():
    from sagemaker.mlops.workflow.steps import Step
    
    step = Mock(spec=Step)
    step.name = "test-step"
    step._depends_on = ["existing"]
    
    Step.depends_on.fset(step, None)
    assert step._depends_on is None


def test_step_add_depends_on_empty_list():
    from sagemaker.mlops.workflow.steps import Step
    
    step = Mock(spec=Step)
    step.name = "test-step"
    step._depends_on = None
    
    Step.add_depends_on(step, [])
    assert step._depends_on is None


def test_step_add_depends_on_extends_existing():
    from sagemaker.mlops.workflow.steps import Step
    
    step = Mock(spec=Step)
    step.name = "test-step"
    step._depends_on = ["step1"]
    
    Step.add_depends_on(step, ["step2", "step3"])
    assert step._depends_on == ["step1", "step2", "step3"]


def test_step_get_step_name_from_str_undefined():
    from sagemaker.mlops.workflow.steps import Step
    
    with pytest.raises(ValueError, match="Step undefined-step is undefined"):
        Step._get_step_name_from_str("undefined-step", {})


def test_step_get_step_name_from_str_step_collection():
    from sagemaker.mlops.workflow.steps import Step
    from sagemaker.mlops.workflow.step_collections import StepCollection
    
    mock_step = Mock()
    mock_step.name = "last-step"
    mock_collection = Mock(spec=StepCollection)
    mock_collection.steps = [Mock(), mock_step]
    
    result = Step._get_step_name_from_str("collection", {"collection": mock_collection})
    assert result == "last-step"


def test_step_get_step_name_from_str_regular_step():
    from sagemaker.mlops.workflow.steps import Step
    
    result = Step._get_step_name_from_str("step-name", {"step-name": Mock()})
    assert result == "step-name"


def test_step_trim_experiment_config_with_display_name():
    from sagemaker.mlops.workflow.steps import Step
    
    request_dict = {
        "ExperimentConfig": {
            "TrialComponentDisplayName": "my-trial",
            "ExperimentName": "my-experiment",
            "TrialName": "my-trial-name"
        }
    }
    Step._trim_experiment_config(request_dict)
    assert request_dict["ExperimentConfig"] == {"TrialComponentDisplayName": "my-trial"}


def test_step_trim_experiment_config_without_display_name():
    from sagemaker.mlops.workflow.steps import Step
    
    request_dict = {
        "ExperimentConfig": {
            "ExperimentName": "my-experiment"
        }
    }
    Step._trim_experiment_config(request_dict)
    assert "ExperimentConfig" not in request_dict


def test_step_trim_experiment_config_no_config():
    from sagemaker.mlops.workflow.steps import Step
    
    request_dict = {"SomeOtherKey": "value"}
    Step._trim_experiment_config(request_dict)
    assert "ExperimentConfig" not in request_dict


def test_cache_config_without_expire_after():
    config = CacheConfig(enable_caching=True)
    assert config.config == {"CacheConfig": {"Enabled": True}}


def test_configurable_retry_step_add_retry_policy_empty():
    from sagemaker.mlops.workflow.steps import TrainingStep
    from sagemaker.mlops.workflow.retry import RetryPolicy
    
    step = TrainingStep(name="test", step_args=None)
    step.retry_policies = []
    step.add_retry_policy(None)
    assert step.retry_policies == []


def test_configurable_retry_step_add_retry_policy():
    from sagemaker.mlops.workflow.steps import TrainingStep
    from sagemaker.mlops.workflow.retry import RetryPolicy
    
    step = TrainingStep(name="test", step_args=None)
    step.retry_policies = None
    policy = Mock(spec=RetryPolicy)
    step.add_retry_policy(policy)
    assert len(step.retry_policies) == 1


def test_configurable_retry_step_to_request_with_retry_policies():
    from sagemaker.mlops.workflow.steps import TrainingStep
    from sagemaker.mlops.workflow.retry import RetryPolicy
    
    step = TrainingStep(name="test", step_args=None)
    step._properties = Mock()
    
    policy = Mock(spec=RetryPolicy)
    policy.to_request.return_value = {"ExceptionType": ["ThrottlingException"]}
    step.retry_policies = [policy]
    
    with pytest.raises(ValueError):
        request = step.to_request()


def test_step_find_dependencies_in_depends_on_list_with_step():
    from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
    
    step1 = Mock(spec=Step)
    step1.name = "step1"
    
    step2 = Mock(spec=Step)
    step2.name = "step2"
    step2.step_type = StepTypeEnum.TRAINING
    step2.depends_on = [step1]
    
    dependencies = Step._find_dependencies_in_depends_on_list(step2, {})
    assert "step1" in dependencies


def test_step_find_dependencies_in_depends_on_list_with_step_collection():
    from sagemaker.mlops.workflow.steps import Step
    from sagemaker.mlops.workflow.step_collections import StepCollection
    
    last_step = Mock()
    last_step.name = "last-step"
    
    collection = Mock(spec=StepCollection)
    collection.steps = [Mock(), last_step]
    
    step = Mock(spec=Step)
    step.name = "test-step"
    step.depends_on = [collection]
    
    dependencies = Step._find_dependencies_in_depends_on_list(step, {})
    assert "last-step" in dependencies


def test_step_find_dependencies_in_depends_on_list_with_string():
    from sagemaker.mlops.workflow.steps import Step
    
    mock_other_step = Mock()
    mock_other_step.name = "other-step"
    
    step = Mock(spec=Step)
    step.name = "test-step"
    step.depends_on = ["other-step"]
    step._get_step_name_from_str = Step._get_step_name_from_str
    
    step_map = {"other-step": mock_other_step}
    dependencies = Step._find_dependencies_in_depends_on_list(step, step_map)
    assert "other-step" in dependencies


def test_step_validate_json_get_property_file_reference_invalid_step_type():
    from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
    from sagemaker.core.workflow.functions import JsonGet
    
    step = Mock(spec=Step)
    step.name = "current-step"
    
    processing_step = Mock()
    processing_step.name = "training-step"
    processing_step.step_type = StepTypeEnum.TRAINING
    
    json_get = Mock(spec=JsonGet)
    json_get.step_name = "training-step"
    json_get.property_file = "property-file"
    json_get.expr = "$.test"
    
    step_map = {"training-step": processing_step}
    
    with pytest.raises(ValueError, match="can only be evaluated on processing step outputs"):
        Step._validate_json_get_property_file_reference(step, json_get, step_map)


def test_step_validate_json_get_property_file_reference_undefined_property_file():
    from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
    from sagemaker.core.workflow.functions import JsonGet
    from sagemaker.core.workflow.properties import PropertyFile
    
    step = Mock(spec=Step)
    step.name = "current-step"
    
    processing_step = Mock()
    processing_step.name = "processing-step"
    processing_step.step_type = StepTypeEnum.PROCESSING
    processing_step.property_files = []
    
    json_get = Mock(spec=JsonGet)
    json_get.step_name = "processing-step"
    json_get.property_file = "undefined-file"
    json_get.expr = "$.test"
    
    step_map = {"processing-step": processing_step}
    
    with pytest.raises(ValueError, match="is undefined in step"):
        Step._validate_json_get_property_file_reference(step, json_get, step_map)


def test_step_validate_json_get_property_file_reference_missing_output():
    from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
    from sagemaker.core.workflow.functions import JsonGet
    from sagemaker.core.workflow.properties import PropertyFile
    
    step = Mock(spec=Step)
    step.name = "current-step"
    
    prop_file = PropertyFile(name="prop-file", output_name="output1", path="path.json")
    
    processing_step = Mock()
    processing_step.name = "processing-step"
    processing_step.step_type = StepTypeEnum.PROCESSING
    processing_step.property_files = [prop_file]
    processing_step.arguments = {
        "ProcessingOutputConfig": {
            "Outputs": [{"OutputName": "different-output"}]
        }
    }
    
    json_get = Mock(spec=JsonGet)
    json_get.step_name = "processing-step"
    json_get.property_file = "prop-file"
    json_get.expr = "$.test"
    
    step_map = {"processing-step": processing_step}
    
    with pytest.raises(ValueError, match="not found in processing step"):
        Step._validate_json_get_property_file_reference(step, json_get, step_map)


def test_step_validate_json_get_property_file_reference_with_property_file_object():
    from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
    from sagemaker.core.workflow.functions import JsonGet
    from sagemaker.core.workflow.properties import PropertyFile
    
    step = Mock(spec=Step)
    step.name = "current-step"
    
    prop_file = PropertyFile(name="prop-file", output_name="output1", path="path.json")
    
    processing_step = Mock()
    processing_step.name = "processing-step"
    processing_step.step_type = StepTypeEnum.PROCESSING
    processing_step.property_files = [prop_file]
    processing_step.arguments = {
        "ProcessingOutputConfig": {
            "Outputs": [{"OutputName": "output1"}]
        }
    }
    
    json_get = Mock(spec=JsonGet)
    json_get.step_name = "processing-step"
    json_get.property_file = prop_file
    json_get.expr = "$.test"
    
    step_map = {"processing-step": processing_step}
    
    Step._validate_json_get_property_file_reference(step, json_get, step_map)


def test_step_validate_json_get_function_with_property_file():
    from sagemaker.mlops.workflow.steps import Step
    from sagemaker.core.workflow.functions import JsonGet
    from sagemaker.core.workflow.properties import PropertyFile
    
    step = Mock(spec=Step)
    step.name = "current-step"
    
    prop_file = PropertyFile(name="prop-file", output_name="output1", path="path.json")
    
    processing_step = Mock()
    processing_step.name = "processing-step"
    processing_step.step_type = Mock()
    processing_step.property_files = [prop_file]
    processing_step.arguments = {
        "ProcessingOutputConfig": {
            "Outputs": [{"OutputName": "output1"}]
        }
    }
    
    json_get = Mock(spec=JsonGet)
    json_get.step_name = "processing-step"
    json_get.property_file = prop_file
    
    step_map = {"processing-step": processing_step}
    
    Step._validate_json_get_function(step, json_get, step_map)


def test_step_find_dependencies_in_step_arguments_with_json_get():
    from unittest.mock import patch
    from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
    from sagemaker.core.workflow.functions import JsonGet
    
    step1 = Mock(spec=Step)
    step1.name = "step1"
    step1.step_type = StepTypeEnum.PROCESSING
    step1.property_files = []
    step1.arguments = {}
    
    json_get = Mock(spec=JsonGet)
    json_get._referenced_steps = [step1]
    json_get.property_file = None
    
    step2 = Mock(spec=Step)
    step2.name = "step2"
    step2._validate_json_get_function = Mock()
    step2._get_step_name_from_str = Step._get_step_name_from_str
    
    obj = {"key": json_get}
    
    with patch('sagemaker.mlops.workflow.steps.TYPE_CHECKING', False):
        with patch.dict('sys.modules', {'sagemaker.core.workflow.function_step': Mock()}):
            dependencies = Step._find_dependencies_in_step_arguments(step2, obj, {"step1": step1})
            assert "step1" in dependencies


def test_step_find_dependencies_in_step_arguments_with_delayed_return():
    from unittest.mock import patch
    from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
    from sagemaker.core.workflow.functions import JsonGet
    from sagemaker.core.helper.pipeline_variable import PipelineVariable
    
    step1 = Mock(spec=Step)
    step1.name = "step1"
    step1.step_type = StepTypeEnum.PROCESSING
    step1.property_files = []
    step1.arguments = {}
    
    json_get = Mock(spec=JsonGet)
    json_get.property_file = None
    
    delayed_return_class = type('DelayedReturn', (PipelineVariable,), {})
    delayed_return = Mock(spec=delayed_return_class)
    delayed_return._referenced_steps = [step1]
    delayed_return._to_json_get = Mock(return_value=json_get)
    delayed_return.__class__ = delayed_return_class
    
    step2 = Mock(spec=Step)
    step2.name = "step2"
    step2._validate_json_get_function = Mock()
    step2._get_step_name_from_str = Step._get_step_name_from_str
    
    obj = {"key": delayed_return}
    
    mock_module = Mock()
    mock_module.DelayedReturn = delayed_return_class
    
    with patch.dict('sys.modules', {'sagemaker.core.workflow.function_step': mock_module}):
        dependencies = Step._find_dependencies_in_step_arguments(step2, obj, {"step1": step1})
        assert "step1" in dependencies


def test_step_find_dependencies_in_step_arguments_with_string_reference():
    from unittest.mock import patch
    from sagemaker.mlops.workflow.steps import Step
    from sagemaker.core.helper.pipeline_variable import PipelineVariable
    
    step1 = Mock(spec=Step)
    step1.name = "step1"
    
    pipeline_var = Mock(spec=PipelineVariable)
    pipeline_var._referenced_steps = ["step1"]
    
    step2 = Mock(spec=Step)
    step2.name = "step2"
    step2._get_step_name_from_str = Step._get_step_name_from_str
    
    obj = {"key": pipeline_var}
    
    step_map = {"step1": step1}
    
    delayed_return_class = type('DelayedReturn', (PipelineVariable,), {})
    mock_module = Mock()
    mock_module.DelayedReturn = delayed_return_class
    
    with patch.dict('sys.modules', {'sagemaker.core.workflow.function_step': mock_module}):
        dependencies = Step._find_dependencies_in_step_arguments(step2, obj, step_map)
        assert "step1" in dependencies


def test_tuning_step_requires_step_args():
    from sagemaker.mlops.workflow.steps import TuningStep
    
    with pytest.raises(ValueError, match="step_args is required"):
        TuningStep(name="tuning-step", step_args=None)


def test_tuning_step_get_top_model_s3_uri_with_prefix():
    from sagemaker.mlops.workflow.steps import TuningStep
    from sagemaker.core.workflow.pipeline_context import _JobStepArguments
    
    step_args = Mock(spec=_JobStepArguments)
    step_args.caller_name = "tune"
    step = TuningStep(name="tuning-step", step_args=step_args)
    
    result = step.get_top_model_s3_uri(top_k=0, s3_bucket="my-bucket", prefix="my-prefix")
    
    from sagemaker.core.workflow.functions import Join
    assert isinstance(result, Join)


def test_tuning_step_get_top_model_s3_uri_without_prefix():
    from sagemaker.mlops.workflow.steps import TuningStep
    from sagemaker.core.workflow.pipeline_context import _JobStepArguments
    
    step_args = Mock(spec=_JobStepArguments)
    step_args.caller_name = "tune"
    step = TuningStep(name="tuning-step", step_args=step_args)
    
    result = step.get_top_model_s3_uri(top_k=0, s3_bucket="my-bucket", prefix="")
    
    from sagemaker.core.workflow.functions import Join
    assert isinstance(result, Join)


def test_tuning_step_get_top_model_s3_uri_with_none_prefix():
    from sagemaker.mlops.workflow.steps import TuningStep
    from sagemaker.core.workflow.pipeline_context import _JobStepArguments
    
    step_args = Mock(spec=_JobStepArguments)
    step_args.caller_name = "tune"
    step = TuningStep(name="tuning-step", step_args=step_args)
    
    result = step.get_top_model_s3_uri(top_k=0, s3_bucket="my-bucket", prefix=None)
    
    from sagemaker.core.workflow.functions import Join
    assert isinstance(result, Join)
