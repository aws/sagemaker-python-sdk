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

import os
from inspect import signature

from sagemaker import Model
from sagemaker.estimator import EstimatorBase
from sagemaker.fw_utils import UploadedCode
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import PipelineVariable
from typing import Union
from typing_extensions import get_args, get_origin
from tests.unit.sagemaker.workflow.test_mechanism.test_code import (
    PIPELINE_VARIABLES,
    REQUIRED,
    OPTIONAL,
    DEFAULT_VALUE,
    IMAGE_URI,
)


def support_pipeline_variable(t: type) -> bool:
    """Check if pipeline variable is supported by a parameter according to its type

    Args:
        t (type): The type to be checked

    Return:
        bool: True if it supports. False otherwise.
    """
    return "PipelineVariable" in str(t)


def get_param_dict(func, clazz=None) -> dict:
    """Get a parameter dict of a given function

    The parameter dict indicates if a parameter is required or not, as well as its type.

    Arg:
        func (function): A class constructor method or other class methods.
        clazz (type): The corresponding class whose method is passed in.

    Return:
        dict: A parameter dict is returned.
    """
    params = list()
    params.append(signature(func))
    if func.__name__ == "__init__" and issubclass(clazz, (EstimatorBase, Model)):
        # Go through all parent classes constructor function to get the entire parameters since
        # estimator and model classes use **kwargs for parameters defined in parent classes
        # The leaf class's parameters should be on top of the params list and have high priority
        _get_params_from_parent_class_constructors(clazz, params)

    params_dict = dict(
        required=dict(),
        optional=dict(),
    )
    for param in params:
        for param_val in param.parameters.values():
            if param_val.annotation is param_val.empty:
                continue
            val = dict(type=param_val.annotation, default_value=None)
            if param_val.name == "sagemaker_session":
                # Treat sagemaker_session as required as it must be a PipelineSession obj
                if not _is_in_params_dict(param_val.name, params_dict):
                    params_dict[REQUIRED][param_val.name] = val
            elif param_val.default is param_val.empty or param_val.default is not None:
                if not _is_in_params_dict(param_val.name, params_dict):
                    # Some parameters e.g. entry_point in TensorFlow appears as both required (in Framework)
                    # and optional (in EstimatorBase) parameter. The annotation defined in the
                    # class node (i.e. Framework) which is closer to the leaf class (TensorFlow) should win.
                    if param_val.default is not param_val.empty:
                        val[DEFAULT_VALUE] = param_val.default
                    params_dict[REQUIRED][param_val.name] = val
            else:
                if not _is_in_params_dict(param_val.name, params_dict):
                    params_dict[OPTIONAL][param_val.name] = val
    return params_dict


def _is_in_params_dict(param_name: str, params_dict: dict):
    """To check if the parameter is in the parameter dict

    Args:
        param_name (str): The name of the parameter to be checked
        params_dict (dict): The parameter dict among which to check if the param_name exists
    """
    return param_name in params_dict[REQUIRED] or param_name in params_dict[OPTIONAL]


def _get_params_from_parent_class_constructors(clazz: type, params: list):
    """Get constructor parameters from parent class

    Args:
        clazz (type): The downstream class to collect parameters from all its parent constructors
        params (list): The list to collect all parameters
    """
    while clazz.__name__ not in {"EstimatorBase", "Model"}:
        parent_class = clazz.__base__
        params.append(signature(parent_class.__init__))
        clazz = parent_class


def generate_pipeline_vars_per_type(
    param_name: str,
    param_type: type,
) -> list:
    """Provide a list of possible PipelineVariable objects.

    For example, if type_hint is Union[str, PipelineVariable],
    return [ParameterString, Properties, JsonGet, Join, ExecutionVariable]

    Args:
        param_name (str): The name of the parameter to generate the pipeline variable list.
        param_type (type): The type of the parameter to generate the pipeline variable list.

    Return:
        list: A list of possible PipelineVariable objects are returned.
    """
    # verify if params allow pipeline variables
    if "PipelineVariable" not in str(param_type):
        raise TypeError(("The type: %s does not support PipelineVariable.", param_type))

    types = get_args(param_type)
    # e.g. Union[str, PipelineVariable] or Union[str, PipelineVariable, NoneType]
    if PipelineVariable in types:
        # PipelineVariable corresponds to Python Primitive types
        # i.e. str, int, float, bool
        ppl_var = _get_pipeline_var(types=types)
        return ppl_var

    # e.g. Union[List[...], NoneType] or Union[Dict[...], NoneType] etc.
    clean_type = clean_up_types(param_type)
    origin_type = get_origin(clean_type)
    if origin_type not in [list, dict, set, tuple]:
        raise TypeError(f"Unsupported type: {param_type} for param: {param_name}")
    sub_types = get_args(clean_type)

    # e.g. List[...], Tuple[...], Set[...]
    if origin_type in [list, tuple, set]:
        ppl_var_list = generate_pipeline_vars_per_type(param_name, sub_types[0])
        return [
            (
                origin_type([var]),
                dict(
                    origin=origin_type([expected["origin"]]),
                    to_string=origin_type([expected["to_string"]]),
                ),
            )
            for var, expected in ppl_var_list
        ]

    # e.g. Dict[...]
    if origin_type is dict:
        key_type = sub_types[0]
        if key_type is not str:
            raise TypeError(
                f"Unsupported type: {key_type} for dict key in {param_name} of {param_type} type"
            )
        ppl_var_list = generate_pipeline_vars_per_type(param_name, sub_types[1])
        return [
            (
                dict(MyKey=var),
                dict(
                    origin=dict(MyKey=expected["origin"]),
                    to_string=dict(MyKey=expected["to_string"]),
                ),
            )
            for var, expected in ppl_var_list
        ]
    return list()


def clean_up_types(t: type) -> type:
    """Clean up the Union type and return the first subtype (not a NoneType) of it

    For example for Union[str, int, NoneType], it will return str

    Args:
        t (type): The type of a parameter to be cleaned up.

    Return:
        type: The cleaned up type is returned.
    """
    if get_origin(t) == Union:
        types = get_args(t)
        return list(filter(lambda t: "NoneType" not in str(t), types))[0]
    return t


def _get_pipeline_var(types: tuple) -> list:
    """Get a Pipeline variable based on one kind of the parameter types.

    Args:
        types (tuple): The possible types of a parameter.

    Return:
        list: a list of possible PipelineVariable objects are returned
    """
    if str in types:
        return PIPELINE_VARIABLES["str"]
    if int in types:
        return PIPELINE_VARIABLES["int"]
    if float in types:
        return PIPELINE_VARIABLES["float"]
    if bool in types:
        return PIPELINE_VARIABLES["bool"]
    raise TypeError(f"Unable to parse types: {types}.")


def mock_tar_and_upload_dir(
    session,
    bucket,
    s3_key_prefix,
    script,
    directory=None,
    dependencies=None,
    kms_key=None,
    s3_resource=None,
    settings=None,
):
    """Briefly mock the behavior of tar_and_upload_dir"""
    if directory and (is_pipeline_variable(directory) or directory.lower().startswith("s3://")):
        return UploadedCode(s3_prefix=directory, script_name=script)
    script_name = script if directory else os.path.basename(script)
    key = "%s/sourcedir.tar.gz" % s3_key_prefix
    return UploadedCode(s3_prefix="s3://%s/%s" % (bucket, key), script_name=script_name)


def mock_image_uris_retrieve(
    framework,
    region,
    version=None,
    py_version=None,
    instance_type=None,
    accelerator_type=None,
    image_scope=None,
    container_version=None,
    distribution=None,
    base_framework_version=None,
    training_compiler_config=None,
    model_id=None,
    model_version=None,
    tolerate_vulnerable_model=False,
    tolerate_deprecated_model=False,
    sdk_version=None,
    inference_tool=None,
    serverless_inference_config=None,
) -> str:
    """Briefly mock the behavior of image_uris.retrieve"""
    args = dict(locals())
    for name, val in args.items():
        if is_pipeline_variable(val):
            raise ValueError("%s should not be a pipeline variable (%s)" % (name, type(val)))
    return IMAGE_URI
