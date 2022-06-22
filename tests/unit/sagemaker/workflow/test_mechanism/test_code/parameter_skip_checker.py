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

from typing import TYPE_CHECKING

from tests.unit.sagemaker.workflow.test_mechanism.test_code import (
    UNSET_PARAM_BONDED_WITH_NONE,
    UNSET_PARAM_BONDED_WITH_NOT_NONE,
    SET_PARAM_BONDED_WITH_NOT_NONE,
    SET_PARAM_BONDED_WITH_NONE,
    BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS,
    CLASS_PARAMS_EXCLUDED_IN_TARGET_FUNC,
    PARAMS_SHOULD_NOT_BE_NONE,
    CLAZZ_ARGS,
    FUNC_ARGS,
    INIT,
    COMMON,
)

if TYPE_CHECKING:
    from tests.unit.sagemaker.workflow.test_mechanism.test_code.test_pipeline_var_compatibility_template import (
        PipelineVarCompatiTestTemplate,
    )


class ParameterSkipChecker:
    """Check if the current parameter can skipped for test"""

    def __init__(self, template: "PipelineVarCompatiTestTemplate"):
        """Initialize a `ParameterSkipChecker` instance.

        Args:
            template (PipelineVarCompatiTestTemplate): The template object to check
            the compatibility between Pipeline variables and the given class and target method.
        """
        self._template = template

    def skip_setting_param_to_none(
        self,
        param_name: str,
        target_func: str,
    ) -> bool:
        """Check if to skip setting the parameter to none.

        Args:
            param_name (str): The name of the parameter, which is to be verified that
                if we can skip it in the loop.
            target_func (str): The target function impacted by the check.
                If target_func is init, it means the parameter affects initiating
                the class object.
        """
        return (
            self._is_param_should_not_be_none(param_name, target_func)
            or self._need_set_param_bonded_with_none(param_name, target_func)
            or self._need_set_param_bonded_with_not_none(param_name, target_func)
        )

    def skip_setting_clz_param_to_ppl_var(
        self,
        clz_param: str,
        target_func: str,
        param_with_none: str = None,
    ) -> bool:
        """Check if to skip setting the class parameter to pipeline variable

        Args:
            clz_param (str): The name of the class __init__ parameter, which is to
                be verified that if we can skip it in the loop.
            target_func (str): The target function impacted by the check.
                If target_func is init, it means the parameter affects initiating
                the class object.
            param_with_none (str): The name of the parameter with None value.
        """
        if target_func == INIT:
            return (
                self._need_unset_param_bonded_with_none(clz_param, INIT)
                or self._is_base_clz_param_excluded_in_subclz(clz_param)
                or self._need_unset_param_bonded_with_not_none(clz_param, INIT)
            )
        else:
            return (
                self._need_unset_param_bonded_with_none(clz_param, target_func)
                or self._need_unset_param_bonded_with_not_none(clz_param, target_func)
                or self._is_param_should_not_be_none(param_with_none, target_func)
                or self._is_overridden_class_param(clz_param, target_func)
                or self._is_clz_param_excluded_in_func(clz_param, target_func)
            )

    def skip_setting_func_param_to_ppl_var(
        self,
        func_param: str,
        target_func: str,
        param_with_none: str = None,
    ) -> bool:
        """Check if to skip setting the function parameter to pipeline variable

        Args:
            func_param (str): The name of the function parameter, which is to
                be verified that if we can skip it in the loop.
            target_func (str): The target function impacted by the check.
            param_with_none (str): The name of the parameter with None value.
        """
        return (
            self._is_param_should_not_be_none(param_with_none, target_func)
            or self._need_unset_param_bonded_with_none(func_param, target_func)
            or self._need_unset_param_bonded_with_not_none(func_param, target_func)
        )

    def _need_unset_param_bonded_with_none(
        self,
        param_name: str,
        target_func: str,
    ) -> bool:
        """Check if to skip testing with pipeline variables due to the bond relationship.

        I.e. the parameter (param_name) does not present in the definition json
        or it is not allowed to be a pipeline variable if its boned parameter is None.
        Then we can skip replacing the param_name with pipeline variables

        Args:
            param_name (str): The name of the parameter, which is to be verified that
                if we can skip setting it to be pipeline variables.
            target_func (str): The target function impacted by the check.
                If target_func is init, it means the bonded parameters affect initiating
                the class object
        """
        return self._is_param_bonded_with_none(
            param_map=UNSET_PARAM_BONDED_WITH_NONE,
            param_name=param_name,
            target_func=target_func,
        )

    def _need_unset_param_bonded_with_not_none(self, param_name: str, target_func: str) -> bool:
        """Check if to skip testing with pipeline variables due to the bond relationship.

        I.e. the parameter (param_name) does not present in the definition json or it should
        not be presented or it is not allowed to be a pipeline variable if its boned parameter
        is not None. Then we can skip replacing the param_name with pipeline variables

        Args:
            param_name (str): The name of the parameter, which is to be verified that
                if we can skip replacing it with pipeline variables.
            target_func (str): The target function impacted by the check.
        """
        return self._is_param_bonded_with_not_none(
            param_map=UNSET_PARAM_BONDED_WITH_NOT_NONE,
            param_name=param_name,
            target_func=target_func,
        )

    def _need_set_param_bonded_with_none(self, param_name: str, target_func: str) -> bool:
        """Check if to skip testing with None value due to the bond relationship.

        I.e. if a parameter (another_param) is None, its substitute parameter (param_name)
        should not be None. Thus we can skip the test round which sets the param_name
        to None under the target function (target_func).

        Args:
            param_name (str): The name of the parameter, which is to be verified regarding
                None value.
            target_func (str): The target function impacted by this check.
        """
        return self._is_param_bonded_with_none(
            param_map=SET_PARAM_BONDED_WITH_NONE,
            param_name=param_name,
            target_func=target_func,
        )

    def _need_set_param_bonded_with_not_none(self, param_name: str, target_func: str) -> bool:
        """Check if to skip testing with None value due to the bond relationship.

        I.e. if the parameter (another_param) is not None, its bonded parameter (param_name)
        should not be None. Thus we can skip the test round which sets the param_name
        to None under the target function (target_func).

        Args:
            param_name (str): The name of the parameter, which is to be verified
                regarding None value.
            target_func (str): The target function impacted by this check.
        """
        return self._is_param_bonded_with_not_none(
            param_map=SET_PARAM_BONDED_WITH_NOT_NONE,
            param_name=param_name,
            target_func=target_func,
        )

    def _is_param_bonded_with_not_none(
        self,
        param_map: dict,
        param_name: str,
        target_func: str,
    ) -> bool:
        """Check if the parameter is bonded one with not None value.

        Args:
            param_map (dict): The parameter map storing the bond relationship.
            param_name (str): The name of the parameter to be verified.
            target_func (str): The target function impacted by this check.
        """
        template = self._template

        def _not_none_checker(func: str, params_dict: dict):
            for another_param in params_dict:
                if template.default_args[CLAZZ_ARGS].get(another_param, None):
                    return True
                if func == INIT:
                    continue
                if template.default_args[FUNC_ARGS][func].get(another_param, None):
                    return True
            return False

        return self._is_param_bonded(
            param_map=param_map,
            param_name=param_name,
            target_func=target_func,
            checker_func=_not_none_checker,
        )

    def _is_param_bonded_with_none(
        self,
        param_map: dict,
        param_name: str,
        target_func: str,
    ) -> bool:
        """Check if the parameter is bonded with another one with None value.

        Args:
            param_map (dict): The parameter map storing the bond relationship.
            param_name (str): The name of the parameter to be verified.
            target_func (str): The target function impacted by this check.
        """
        template = self._template

        def _none_checker(func: str, params_dict: dict):
            for another_param in params_dict:
                if template.default_args[CLAZZ_ARGS].get(another_param, "N/A") is None:
                    return True
                if func == INIT:
                    continue
                if template.default_args[FUNC_ARGS][func].get(another_param, "N/A") is None:
                    return True
            return False

        return self._is_param_bonded(
            param_map=param_map,
            param_name=param_name,
            target_func=target_func,
            checker_func=_none_checker,
        )

    def _is_param_bonded(
        self,
        param_map: dict,
        param_name: str,
        target_func: str,
        checker_func: callable,
    ) -> bool:
        """Check if the parameter has a specific bond relationship.

        Args:
            param_map (dict): The parameter map storing the bond relationship.
            param_name (str): The name of the parameter to be verified.
            target_func (str): The target function impacted by this check.
            checker_func (callable): The checker function to check the specific bond relationship.
        """
        template = self._template
        if template.clazz_type not in param_map:
            return False
        if target_func not in param_map[template.clazz_type]:
            return False
        params_dict = param_map[template.clazz_type][target_func][COMMON].get(param_name, {})
        if not params_dict:
            if template.clazz.__name__ not in param_map[template.clazz_type][target_func]:
                return False
            params_dict = param_map[template.clazz_type][target_func][template.clazz.__name__].get(
                param_name, {}
            )
        return checker_func(target_func, params_dict)

    def _is_base_clz_param_excluded_in_subclz(self, clz_param_name: str) -> bool:
        """Check if to skip testing with pipeline variables on class parameter due to exclusion.

        I.e. the base class parameter (clz_param_name) should not be replaced with pipeline variables,
        as it's not used in the subclass.

        Args:
            clz_param_name (str): The name of the class parameter, which is to be verified.
        """
        template = self._template
        if template.clazz_type not in BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS:
            return False
        if (
            template.clazz.__name__
            not in BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS[template.clazz_type]
        ):
            return False
        return (
            clz_param_name
            in BASE_CLASS_PARAMS_EXCLUDED_IN_SUB_CLASS[template.clazz_type][template.clazz.__name__]
        )

    def _is_overridden_class_param(self, clz_param_name: str, target_func: str) -> bool:
        """Check if to skip testing with pipeline variables on class parameter due to override.

        I.e. the class parameter (clz_param_name) should not be replaced with pipeline variables
        and tested on the target function (target_func) because it's overridden by a
        function parameter with the same name.
        e.g. image_uri in model.create can override that in model constructor.

        Args:
            clz_param_name (str): The name of the class parameter, which is to be verified.
            target_func (str): The target function impacted by the check.
        """
        template = self._template
        return template.default_args[FUNC_ARGS][target_func].get(clz_param_name, None) is not None

    def _is_clz_param_excluded_in_func(self, clz_param_name: str, target_func: str) -> bool:
        """Check if to skip testing with pipeline variables on class parameter due to exclusion.

        I.e. the class parameter (clz_param_name) should not be replaced with pipeline variables
        and tested on the target function (target_func), as it's not used there.

        Args:
            clz_param_name (str): The name of the class parameter, which is to be verified.
            target_func (str): The target function impacted by the check.
        """
        return self._is_param_included(
            param_map=CLASS_PARAMS_EXCLUDED_IN_TARGET_FUNC,
            param_name=clz_param_name,
            target_func=target_func,
        )

    def _is_param_should_not_be_none(self, param_name: str, target_func: str) -> bool:
        """Check if to skip testing due to the parameter should not be None.

        I.e. the parameter (param_name) is set to None in this round but it is not allowed
        according to the logic. Thus we can skip this round of test.

        Args:
            param_name (str): The name of the parameter, which is to be verified regarding None value.
            target_func (str): The target function impacted by this check.
        """
        return self._is_param_included(
            param_map=PARAMS_SHOULD_NOT_BE_NONE,
            param_name=param_name,
            target_func=target_func,
        )

    def _is_param_included(
        self,
        param_map: dict,
        param_name: str,
        target_func: str,
    ) -> bool:
        """Check if the parameter is included in a specific relationship.

        Args:
            param_map (dict): The parameter map storing the specific relationship.
            param_name (str): The name of the parameter to be verified.
            target_func (str): The target function impacted by this check.
        """
        template = self._template
        if template.clazz_type not in param_map:
            return False
        if target_func not in param_map[template.clazz_type]:
            return False
        if param_name in param_map[template.clazz_type][target_func][COMMON]:
            return True
        if template.clazz.__name__ not in param_map[template.clazz_type][target_func]:
            return False
        return param_name in param_map[template.clazz_type][target_func][template.clazz.__name__]
