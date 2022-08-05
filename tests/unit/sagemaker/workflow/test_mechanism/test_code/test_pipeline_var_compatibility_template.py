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

import json

from random import getrandbits
from typing import Optional, List
from typing_extensions import get_origin

from sagemaker import Model, PipelineModel, AlgorithmEstimator
from sagemaker.estimator import EstimatorBase
from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase
from sagemaker.mxnet import MXNet
from sagemaker.processing import Processor
from sagemaker.clarify import SageMakerClarifyProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.transformer import Transformer
from sagemaker.tuner import HyperparameterTuner
from sagemaker.workflow.step_collections import StepCollection
from tests.unit.sagemaker.workflow.test_mechanism.test_code import (
    STEP_CLASS,
    FIXED_ARGUMENTS,
    STR_VAL,
    CLAZZ_ARGS,
    FUNC_ARGS,
    REQUIRED,
    OPTIONAL,
    INIT,
    TYPE,
    DEFAULT_VALUE,
    COMMON,
    PipelineVariableEncoder,
)
from tests.unit.sagemaker.workflow.test_mechanism.test_code.utilities import (
    support_pipeline_variable,
    get_param_dict,
    generate_pipeline_vars_per_type,
    clean_up_types,
)
from tests.unit.sagemaker.workflow.test_mechanism.test_code.parameter_skip_checker import (
    ParameterSkipChecker,
)


class PipelineVarCompatiTestTemplate:
    """Check the compatibility between Pipeline variables and the given class, target method"""

    def __init__(self, clazz: type, default_args: dict):
        """Initialize a `PipelineVarCompatiTestTemplate` instance.

        Args:
            clazz (type): The class to test the compatibility.
            default_args (dict): The given default arguments for the class and its target method.
        """
        self.clazz = clazz
        self.clazz_type = self._get_clazz_type()
        self._target_funcs = self._get_target_functions()
        self._clz_params = get_param_dict(clazz.__init__, clazz)
        self._func_params = dict()
        for func in self._target_funcs:
            self._func_params[func.__name__] = get_param_dict(func)
        self._set_and_restructure_default_args(default_args)
        self._skip_param_checker = ParameterSkipChecker(self)

    def _set_and_restructure_default_args(self, default_args: dict):
        """Set and restructure the default_args

        Restructure the default_args[FUNC_ARGS] if it's missing the layer of target function name

        Args:
            default_args (dict): The given default arguments for the class and its target method.
        """
        self.default_args = default_args
        # restructure the default_args[FUNC_ARGS] if it's missing the layer of target function name
        if len(self._target_funcs) == 1:
            target_func_name = self._target_funcs[0].__name__
            if target_func_name not in default_args[FUNC_ARGS]:
                args = self.default_args.pop(FUNC_ARGS)
                self.default_args[FUNC_ARGS] = dict()
                self.default_args[FUNC_ARGS][target_func_name] = args

        self._check_or_fill_in_args(
            params={**self._clz_params[REQUIRED], **self._clz_params[OPTIONAL]},
            default_args=self.default_args[CLAZZ_ARGS],
        )
        for func in self._target_funcs:
            func_name = func.__name__
            self._check_or_fill_in_args(
                params={
                    **self._func_params[func_name][REQUIRED],
                    **self._func_params[func_name][OPTIONAL],
                },
                default_args=self.default_args[FUNC_ARGS][func_name],
            )

    def _get_clazz_type(self) -> str:
        """Get the type (in str) of the downstream class"""
        if issubclass(self.clazz, Processor):
            return "processor"
        if issubclass(self.clazz, EstimatorBase):
            return "estimator"
        if issubclass(self.clazz, Transformer):
            return "transformer"
        if issubclass(self.clazz, HyperparameterTuner):
            return "tuner"
        if issubclass(self.clazz, Model):
            return "model"
        if issubclass(self.clazz, PipelineModel):
            return "pipelinemodel"
        raise TypeError(f"Unsupported downstream class: {self.clazz}")

    def check_compatibility(self):
        """The entry to check the compatibility"""
        print(
            "Starting to check Pipeline variable compatibility for class (%s) and target methods (%s)\n"
            % (self.clazz.__name__, [func.__name__ for func in self._target_funcs])
        )

        # Check the case when all args are assigned not-None values
        print("## Starting to check the compatibility when all optional args are not None ##")
        self._iterate_params_to_check_compatibility()

        # Check the case when one of the optional arg is None
        print(
            "## Starting to check the compatibility when one of the optional arg is None in each round ##"
        )
        self._iterate_optional_params_to_check_compatibility()

    def _iterate_params_to_check_compatibility(
        self,
        param_with_none: Optional[str] = None,
        test_func_for_none: Optional[str] = None,
    ):
        """Iterate each parameter and assign a pipeline var to it to test compatibility

        Args:
            param_with_none (str): The name of the parameter with None value.
            test_func_for_none (str): The name of the function which is being tested by
                replacing optional parameters to None.
        """
        self._iterate_clz_params_to_check_compatibility(param_with_none, test_func_for_none)
        self._iterate_func_params_to_check_compatibility(param_with_none, test_func_for_none)

    def _iterate_optional_params_to_check_compatibility(self):
        """Iterate each optional parameter and set it to none to test compatibility"""
        self._iterate_class_optional_params()
        self._iterate_func_optional_params()

    def _iterate_class_optional_params(self):
        """Iterate each optional parameter in class __init__ and check compatibility"""
        print("### Starting to iterate optional parameters in class __init__")
        self._iterate_optional_params(
            optional_params=self._clz_params[OPTIONAL],
            default_args=self.default_args[CLAZZ_ARGS],
        )

    def _iterate_func_optional_params(self):
        """Iterate each function parameter and check compatibility"""
        for func in self._target_funcs:
            print(f"### Starting to iterate optional parameters in function {func.__name__}")
            self._iterate_optional_params(
                optional_params=self._func_params[func.__name__][OPTIONAL],
                default_args=self.default_args[FUNC_ARGS][func.__name__],
                test_func_for_none=func.__name__,
            )

    def _iterate_optional_params(
        self,
        optional_params: dict,
        default_args: dict,
        test_func_for_none: Optional[str] = None,
    ):
        """Iterate each optional parameter and check compatibility
        Args:
            optional_params (dict): The dict containing the optional parameters of a class or method.
            default_args (dict): The dict containing the default arguments of a class or method.
            test_func_for_none (str): The name of the function which is being tested by
                replacing optional parameters to None.
        """
        for param_name in optional_params.keys():
            if self._skip_param_checker.skip_setting_param_to_none(param_name, INIT):
                continue
            origin_val = default_args[param_name]
            default_args[param_name] = None
            print("=== Parameter (%s) is None in this round ===" % param_name)
            self._iterate_params_to_check_compatibility(param_name, test_func_for_none)
            default_args[param_name] = origin_val

    def _iterate_clz_params_to_check_compatibility(
        self,
        param_with_none: Optional[str] = None,
        test_func_for_none: Optional[str] = None,
    ):
        """Iterate each class parameter and assign a pipeline var to it to test compatibility

        Args:
            param_with_none (str): The name of the parameter with None value.
            test_func_for_none (str): The name of the function which is being tested by
                replacing optional parameters to None.
        """
        print(
            f"#### Iterating parameters (with PipelineVariable annotation) "
            f"in class {self.clazz.__name__} __init__ function"
        )
        clz_params = {**self._clz_params[REQUIRED], **self._clz_params[OPTIONAL]}
        # Iterate through each default arg
        for clz_param_name, clz_default_arg in self.default_args[CLAZZ_ARGS].items():
            if clz_param_name == param_with_none:
                continue
            clz_param_type = clz_params[clz_param_name][TYPE]
            if not support_pipeline_variable(clz_param_type):
                continue
            if self._skip_param_checker.skip_setting_clz_param_to_ppl_var(
                clz_param=clz_param_name, target_func=INIT
            ):
                continue

            # For each arg which supports pipeline variables,
            # Replace it with each one of generated pipeline variables
            ppl_vars = generate_pipeline_vars_per_type(clz_param_name, clz_param_type)
            for clz_ppl_var, expected_clz_expr in ppl_vars:
                self.default_args[CLAZZ_ARGS][clz_param_name] = clz_ppl_var
                obj = self.clazz(**self.default_args[CLAZZ_ARGS])
                for func in self._target_funcs:
                    func_name = func.__name__
                    if test_func_for_none and test_func_for_none != func_name:
                        # Iterating optional parameters of a specific target function
                        # (test_func_for_none), which does not impact other target functions,
                        # so we can skip them
                        continue
                    if self._skip_param_checker._need_set_param_bonded_with_none(
                        param_with_none, func_name
                    ):  # TODO: add to a public method
                        continue
                    if self._skip_param_checker.skip_setting_clz_param_to_ppl_var(
                        clz_param=clz_param_name,
                        target_func=func_name,
                        param_with_none=param_with_none,
                    ):
                        continue
                    # print(
                    #     "Replacing class init arg (%s) with pipeline variable which is expected "
                    #     "to be (%s). Testing with target function (%s)"
                    #     % (clz_param_name, expected_clz_expr, func_name)
                    # )
                    self._generate_and_verify_step_definition(
                        target_func=getattr(obj, func_name),
                        expected_expr=expected_clz_expr,
                        param_with_none=param_with_none,
                    )

            # print("============================\n")
            self.default_args[CLAZZ_ARGS][clz_param_name] = clz_default_arg

    def _iterate_func_params_to_check_compatibility(
        self,
        param_with_none: Optional[str] = None,
        test_func_for_none: Optional[str] = None,
    ):
        """Iterate each target func parameter and assign a pipeline var to it

        Args:
            param_with_none (str): The name of the parameter with None value.
            test_func_for_none (str): The name of the function which is being tested by
                replacing optional parameters to None.
        """
        obj = self.clazz(**self.default_args[CLAZZ_ARGS])
        for func in self._target_funcs:
            func_name = func.__name__
            if test_func_for_none and test_func_for_none != func_name:
                # Iterating optional parameters of a specific target function (test_func_for_none),
                # which does not impact other target functions, so we can skip them
                continue
            if self._skip_param_checker._need_set_param_bonded_with_none(
                param_with_none, func_name
            ):  # TODO: add to a public method
                continue
            print(
                f"#### Iterating parameters (with PipelineVariable annotation) in target function: {func_name}"
            )
            func_params = {
                **self._func_params[func_name][REQUIRED],
                **self._func_params[func_name][OPTIONAL],
            }
            for func_param_name, func_default_arg in self.default_args[FUNC_ARGS][
                func_name
            ].items():
                if func_param_name == param_with_none:
                    continue
                if not support_pipeline_variable(func_params[func_param_name][TYPE]):
                    continue
                if self._skip_param_checker.skip_setting_func_param_to_ppl_var(
                    func_param=func_param_name,
                    target_func=func_name,
                    param_with_none=param_with_none,
                ):
                    continue
                # For each arg which supports pipeline variables,
                # Replace it with each one of generated pipeline variables
                ppl_vars = generate_pipeline_vars_per_type(
                    func_param_name, func_params[func_param_name][TYPE]
                )
                for func_ppl_var, expected_func_expr in ppl_vars:
                    # print(
                    #     "Replacing func arg (%s) with pipeline variable which is expected to be (%s)"
                    #     % (func_param_name, expected_func_expr)
                    # )
                    self.default_args[FUNC_ARGS][func_name][func_param_name] = func_ppl_var
                    self._generate_and_verify_step_definition(
                        target_func=getattr(obj, func_name),
                        expected_expr=expected_func_expr,
                        param_with_none=param_with_none,
                    )

                self.default_args[FUNC_ARGS][func_name][func_param_name] = func_default_arg
                # print("-------------------------\n")

    def _generate_and_verify_step_definition(
        self,
        target_func: callable,
        expected_expr: dict,
        param_with_none: str,
    ):
        """Generate a pipeline and verify the pipeline definition

        Args:
            target_func (callable): The function to generate step_args.
            expected_expr (dict): The expected json expression of a class or method argument.
            param_with_none (str): The name of the parameter with None value.
        """
        args = dict(
            name="MyStep",
            step_args=target_func(**self.default_args[FUNC_ARGS][target_func.__name__]),
        )
        step = STEP_CLASS[self.clazz_type](**args)
        if isinstance(step, StepCollection):
            request_dicts = step.request_dicts()
        else:
            request_dicts = [step.to_request()]

        step_dsl = json.dumps(request_dicts, cls=PipelineVariableEncoder)
        step_dsl_obj = json.loads(step_dsl)
        exp_origin = json.dumps(expected_expr["origin"])
        exp_to_str = json.dumps(expected_expr["to_string"])
        # if the testing arg is a dict, we may need to remove the outer {} of its expected expr
        # to compare, since for HyperParameters, some other arguments are auto inserted to the dict
        assert (
            exp_origin in step_dsl
            or exp_to_str in step_dsl
            or exp_origin[1:-1] in step_dsl
            or exp_to_str[1:-1] in step_dsl
        )
        self._verify_composite_object_against_pipeline_var(param_with_none, step_dsl, step_dsl_obj)

    def _verify_composite_object_against_pipeline_var(
        self,
        param_with_none: str,
        step_dsl: str,
        step_dsl_obj: List[dict],
    ):
        """verify pipeline definition regarding composite objects against pipeline variables

        Args:
            param_with_none (str): The name of the parameter with None value.
            step_dsl (str): The step definition retrieved from the pipeline definition DSL.
            step_dsl_obj (List[dict]): The json load object of the step definition.
        """
        # TODO: remove the following hard code assertion once recursive assignment is added
        if issubclass(self.clazz, Processor):
            if param_with_none != "network_config":
                assert '{"Get": "Parameters.nw_cfg_subnets"}' in step_dsl
                assert '{"Get": "Parameters.nw_cfg_security_group_ids"}' in step_dsl
                assert '{"Get": "Parameters.nw_cfg_enable_nw_isolation"}' in step_dsl
            if issubclass(self.clazz, SageMakerClarifyProcessor):
                if param_with_none != "data_config":
                    assert '{"Get": "Parameters.clarify_processor_input"}' in step_dsl
                    assert '{"Get": "Parameters.clarify_processor_output"}' in step_dsl
            else:
                if param_with_none != "outputs":
                    assert '{"Get": "Parameters.proc_output_source"}' in step_dsl
                    assert '{"Get": "Parameters.proc_output_dest"}' in step_dsl
                    assert '{"Get": "Parameters.proc_output_app_managed"}' in step_dsl
                if param_with_none != "inputs":
                    assert '{"Get": "Parameters.proc_input_source"}' in step_dsl
                    assert '{"Get": "Parameters.proc_input_dest"}' in step_dsl
                    assert '{"Get": "Parameters.proc_input_s3_data_type"}' in step_dsl
                    assert '{"Get": "Parameters.proc_input_app_managed"}' in step_dsl
        elif issubclass(self.clazz, EstimatorBase):
            if (
                param_with_none != "instance_groups"
                and self.default_args[CLAZZ_ARGS]["instance_groups"]
            ):
                assert '{"Get": "Parameters.instance_group_name"}' in step_dsl
                assert '{"Get": "Parameters.instance_group_instance_count"}' in step_dsl
            if issubclass(self.clazz, AmazonAlgorithmEstimatorBase):
                # AmazonAlgorithmEstimatorBase's input is records
                if param_with_none != "records":
                    assert '{"Get": "Parameters.records_s3_data"}' in step_dsl
                    assert '{"Get": "Parameters.records_s3_data_type"}' in step_dsl
                    assert '{"Get": "Parameters.records_channel"}' in step_dsl
            else:
                if param_with_none != "inputs":
                    assert '{"Get": "Parameters.train_inputs_s3_data"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_distribution"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_compression"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_content_type"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_record_wrapping"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_s3_data_type"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_input_mode"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_attribute_name"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_target_attr_name"}' in step_dsl
                    assert '{"Get": "Parameters.train_inputs_instance_groups"}' in step_dsl
            if not issubclass(self.clazz, (TensorFlow, MXNet, PyTorch, AlgorithmEstimator)):
                # debugger_hook_config may be disabled for these first 3 frameworks
                # AlgorithmEstimator ignores the kwargs
                if param_with_none != "debugger_hook_config":
                    assert '{"Get": "Parameters.debugger_hook_s3_output"}' in step_dsl
                    assert '{"Get": "Parameters.debugger_container_output"}' in step_dsl
                    assert '{"Get": "Parameters.debugger_hook_param"}' in step_dsl
                    assert '{"Get": "Parameters.debugger_collections_name"}' in step_dsl
                    assert '{"Get": "Parameters.debugger_collections_param"}' in step_dsl
            if not issubclass(self.clazz, AlgorithmEstimator):
                # AlgorithmEstimator ignores the kwargs
                if param_with_none != "profiler_config":
                    assert '{"Get": "Parameters.profile_config_s3_output_path"}' in step_dsl
                    assert '{"Get": "Parameters.profile_config_system_monitor"}' in step_dsl
                if param_with_none != "tensorboard_output_config":
                    assert '{"Get": "Parameters.tensorboard_s3_output"}' in step_dsl
                    assert '{"Get": "Parameters.tensorboard_container_output"}' in step_dsl
                if param_with_none != "rules":
                    assert '{"Get": "Parameters.rules_image_uri"}' in step_dsl
                    assert '{"Get": "Parameters.rules_instance_type"}' in step_dsl
                    assert '{"Get": "Parameters.rules_volume_size"}' in step_dsl
                    assert '{"Get": "Parameters.rules_to_invoke"}' in step_dsl
                    assert '{"Get": "Parameters.rules_local_output"}' in step_dsl
                    assert '{"Get": "Parameters.rules_to_s3_output_path"}' in step_dsl
                    assert '{"Get": "Parameters.rules_other_s3_input"}' in step_dsl
                    assert '{"Get": "Parameters.rules_param"}' in step_dsl
                    if not issubclass(self.clazz, (TensorFlow, MXNet, PyTorch)):
                        # The collections_to_save is added to debugger rules,
                        # which may be disabled for some frameworks
                        assert '{"Get": "Parameters.rules_collections_name"}' in step_dsl
                        assert '{"Get": "Parameters.rules_collections_param"}' in step_dsl
        elif issubclass(self.clazz, HyperparameterTuner):
            if param_with_none != "inputs":
                assert '{"Get": "Parameters.inputs_estimator_1"}' in step_dsl
            if param_with_none != "warm_start_config":
                assert '{"Get": "Parameters.warm_start_cfg_parent"}' in step_dsl
            if param_with_none != "hyperparameter_ranges":
                assert (
                    json.dumps(
                        {
                            "Std:Join": {
                                "On": "",
                                "Values": [{"Get": "Parameters.hyper_range_min_value"}],
                            }
                        }
                    )
                    in step_dsl
                )
                assert (
                    json.dumps(
                        {
                            "Std:Join": {
                                "On": "",
                                "Values": [{"Get": "Parameters.hyper_range_max_value"}],
                            }
                        }
                    )
                    in step_dsl
                )
                assert '{"Get": "Parameters.hyper_range_scaling_type"}' in step_dsl
        elif issubclass(self.clazz, (Model, PipelineModel)):
            if step_dsl_obj[-1]["Type"] == "Model":
                return
            if param_with_none != "model_metrics":
                assert '{"Get": "Parameters.model_statistics_content_type"}' in step_dsl
                assert '{"Get": "Parameters.model_statistics_s3_uri"}' in step_dsl
                assert '{"Get": "Parameters.model_statistics_content_digest"}' in step_dsl
            if param_with_none != "metadata_properties":
                assert '{"Get": "Parameters.meta_properties_commit_id"}' in step_dsl
                assert '{"Get": "Parameters.meta_properties_repository"}' in step_dsl
                assert '{"Get": "Parameters.meta_properties_generated_by"}' in step_dsl
                assert '{"Get": "Parameters.meta_properties_project_id"}' in step_dsl
            if param_with_none != "drift_check_baselines":
                assert '{"Get": "Parameters.drift_constraints_content_type"}' in step_dsl
                assert '{"Get": "Parameters.drift_constraints_s3_uri"}' in step_dsl
                assert '{"Get": "Parameters.drift_constraints_content_digest"}' in step_dsl
                assert '{"Get": "Parameters.drift_bias_content_type"}' in step_dsl
                assert '{"Get": "Parameters.drift_bias_s3_uri"}' in step_dsl
                assert '{"Get": "Parameters.drift_bias_content_digest"}' in step_dsl

    def _get_non_pipeline_val(self, n: str, t: type) -> object:
        """Get the value (not a Pipeline variable) based on parameter type and name

        Args:
            n (str): The parameter name. If a parameter has a pre-defined value,
                it will be returned directly.
            t (type): The parameter type. If a parameter does not have a pre-defined value,
                an arg will be auto-generated based on the type.

        Return:
            object: A Python primitive value is returned.
        """
        if n in FIXED_ARGUMENTS[COMMON]:
            return FIXED_ARGUMENTS[COMMON][n]
        if n in FIXED_ARGUMENTS[self.clazz_type]:
            return FIXED_ARGUMENTS[self.clazz_type][n]
        if t is str:
            return STR_VAL
        if t is int:
            return 1
        if t is float:
            return 1e-4
        if t is bool:
            return bool(getrandbits(1))
        if t in [list, tuple, dict, set]:
            return t()

        raise TypeError(f"Unable to parse type: {t}.")

    def _check_or_fill_in_args(self, params: dict, default_args: dict):
        """Check if every args are provided and not None

        Otherwise fill in with some default values

        Args:
            params (dict): The dict indicating the type of each parameter.
            default_args (dict): The dict of args to be checked or filled in.
        """
        for param_name, value in params.items():
            if param_name in default_args:
                # User specified the default value
                continue
            if value[DEFAULT_VALUE]:
                # The parameter has default value in method definition
                default_args[param_name] = value[DEFAULT_VALUE]
                continue
            clean_type = clean_up_types(value[TYPE])
            origin_type = get_origin(clean_type)
            if origin_type is None:
                default_args[param_name] = self._get_non_pipeline_val(param_name, clean_type)
            else:
                default_args[param_name] = self._get_non_pipeline_val(param_name, origin_type)

        self._check_or_update_default_args(default_args)

    def _check_or_update_default_args(self, default_args: dict):
        """To check if the default args are valid and update them if not

        Args:
            default_args (dict): The dict of args to be checked or updated.
        """
        if issubclass(self.clazz, EstimatorBase):
            if "disable_profiler" in default_args and default_args["disable_profiler"] is True:
                default_args["profiler_config"] = None

    def _get_target_functions(self) -> list:
        """Fetch the target functions based on class

        Return:
            list: The list of target functions is returned.
        """
        if issubclass(self.clazz, Processor):
            if issubclass(self.clazz, SageMakerClarifyProcessor):
                return [
                    self.clazz.run_pre_training_bias,
                    self.clazz.run_post_training_bias,
                    self.clazz.run_bias,
                    self.clazz.run_explainability,
                ]
            return [self.clazz.run]
        if issubclass(self.clazz, EstimatorBase):
            return [self.clazz.fit]
        if issubclass(self.clazz, Transformer):
            return [self.clazz.transform]
        if issubclass(self.clazz, HyperparameterTuner):
            return [self.clazz.fit]
        if issubclass(self.clazz, (Model, PipelineModel)):
            return [self.clazz.register, self.clazz.create]
        raise TypeError(f"Unable to get target function for class {self.clazz}")
