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
import logging

from dataclasses import asdict
import re

from sagemaker.core.utils.code_injection.shape_dag import SHAPE_DAG
from sagemaker.core.utils.code_injection.constants import (
    BASIC_TYPES,
    STRUCTURE_TYPE,
    LIST_TYPE,
    MAP_TYPE,
)
from io import BytesIO


def pascal_to_snake(pascal_str):
    """
    Converts a PascalCase string to snake_case.

    Args:
        pascal_str (str): The PascalCase string to be converted.

    Returns:
        str: The converted snake_case string.
    """
    snake_case = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", pascal_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_case).lower()


def deserialize(data, cls) -> object:
    """
    Deserialize the given data into an instance of the specified class.

    Args:
        data (dict): The data to be deserialized.
        cls (str or type): The class or class name to deserialize into.

    Returns:
        object: An instance of the specified class with the deserialized data.
    """
    # Convert the keys to snake_case
    logging.debug(f"Deserialize: pascal cased data: {data}")
    data = {pascal_to_snake(k): v for k, v in data.items()}
    logging.debug(f"Deserialize: snake cased data: {data}")

    # Get the class from the cls_name string
    if type(cls) == str:
        cls = globals()[cls]

    # Create a new instance of the class
    instance = cls(**data)

    return instance


def snake_to_pascal(snake_str):
    """
    Convert a snake_case string to PascalCase.

    Args:
        snake_str (str): The snake_case string to be converted.

    Returns:
        str: The PascalCase string.

    """
    components = snake_str.split("_")
    return "".join(x.title() for x in components[0:])


def serialize(data) -> object:
    """
    Serializes the given data object into a dictionary.

    Args:
        data: The data object to be serialized.

    Returns:
        A dictionary containing the serialized data.

    """
    data_dict = asdict(data)

    # Convert the keys to pascalCase
    data_dict = {snake_to_pascal(k): v for k, v in data_dict.items() if v is not None}

    return data_dict


def _evaluate_list_type(raw_list, shape) -> list:
    """
    Evaluates a list type based on the given shape.

    Args:
        raw_list (list): The raw list to be evaluated.
        shape (dict): The shape of the list.

    Returns:
        list: The evaluated list based on the shape.

    Raises:
        ValueError: If an unhandled list member type is encountered.

    """
    _shape_member_type = shape["member_type"]
    _shape_member_shape = shape["member_shape"]
    _evaluated_list = []
    if _shape_member_type in BASIC_TYPES:
        # if basic types directly assign list value.
        _evaluated_list = raw_list
    elif _shape_member_type == STRUCTURE_TYPE:
        # if structure type, transform each list item and assign value.
        # traverse through response list and evaluate item
        for item in raw_list:
            _evaluated_item = transform(item, _shape_member_shape)
            _evaluated_list.append(_evaluated_item)
    elif _shape_member_type == LIST_TYPE:
        # if list type, transform each list item and assign value.
        # traverse through response list and evaluate item
        for item in raw_list:
            _list_type_shape = SHAPE_DAG[_shape_member_shape]
            _evaluated_item = _evaluate_list_type(item, _list_type_shape)
            _evaluated_list.append(_evaluated_item)
    elif _shape_member_type == MAP_TYPE:
        # if structure type, transform each list item and assign value.
        # traverse through response list and evaluate item
        for item in raw_list:
            _map_type_shape = SHAPE_DAG[_shape_member_shape]
            _evaluated_item = _evaluate_map_type(item, _map_type_shape)
            _evaluated_list.append(_evaluated_item)
    else:
        raise ValueError(
            f"Unhandled List member type "
            f"[{_shape_member_type}] encountered. "
            "Needs additional logic for support"
        )
    return _evaluated_list


def _evaluate_map_type(raw_map, shape) -> dict:
    """
    Evaluates a map type based on the given shape.

    Args:
        raw_map (dict): The raw map to be evaluated.
        shape (dict): The shape of the map.

    Returns:
        dict: The evaluated map.

    Raises:
        ValueError: If an unhandled map key type or list member type is encountered.
    """
    _shape_key_type = shape["key_type"]
    _shape_value_type = shape["value_type"]
    _shape_value_shape = shape["value_shape"]
    if _shape_key_type != "string":
        raise ValueError(
            f"Unhandled Map key type "
            f"[{_shape_key_type}] encountered. "
            "Needs additional logic for support"
        )

    _evaluated_map = {}
    if _shape_value_type in BASIC_TYPES:
        # if basic types directly assign value.
        # Ex. response["map_member"] = {"key":"value"}
        _evaluated_map = raw_map
    elif _shape_value_type == STRUCTURE_TYPE:
        # if structure type loop through and evaluate values
        for k, v in raw_map.items():
            _evaluated_value = transform(v, _shape_value_shape)
            _evaluated_map[k] = _evaluated_value
    elif _shape_value_type == LIST_TYPE:
        for k, v in raw_map.items():
            _list_type_shape = SHAPE_DAG[_shape_value_shape]
            evaluated_values = _evaluate_list_type(v, _list_type_shape)
            _evaluated_map[k] = evaluated_values
    elif _shape_value_type == MAP_TYPE:
        for k, v in raw_map.items():
            _map_type_shape = SHAPE_DAG[_shape_value_shape]
            evaluated_values = _evaluate_map_type(v, _map_type_shape)
            _evaluated_map[k] = evaluated_values
    else:
        raise ValueError(
            f"Unhandled List member type "
            f"[{_shape_value_type}] encountered. "
            "Needs additional logic for support"
        )

    return _evaluated_map


def transform(data, shape, object_instance=None) -> dict:
    """
    Transforms the given data based on the given shape.

    Args:
        data (dict): The data to be transformed.
        shape (str): The shape of the data.
        object_instance (object): The object to be transformed. (Optional)

    Returns:
        dict: The transformed data.

    Raises:
        ValueError: If an unhandled shape type is encountered.
    """
    result = {}
    _shape = SHAPE_DAG[shape]

    if _shape["type"] in BASIC_TYPES:
        raise ValueError("Unexpected low-level operation model shape")

    for member in _shape["members"]:
        _member_name = member["name"]
        _member_shape = member["shape"]
        _member_type = member["type"]
        if data.get(_member_name) is None:
            # skip members that are not in the response
            continue
        # 1. set snake case attribute name
        attribute_name = pascal_to_snake(_member_name)
        # 2. assign response value
        if _member_type in BASIC_TYPES:
            evaluated_value = data[_member_name]
        elif _member_type == STRUCTURE_TYPE:
            evaluated_value = transform(data[_member_name], _member_shape)
        elif _member_type == LIST_TYPE:
            _list_type_shape = SHAPE_DAG[_member_shape]
            # 2. assign response value
            evaluated_value = _evaluate_list_type(data[_member_name], _list_type_shape)
        elif _member_type == MAP_TYPE:
            _map_type_shape = SHAPE_DAG[_member_shape]
            evaluated_value = _evaluate_map_type(data[_member_name], _map_type_shape)
        elif _member_type == "blob":
            blob_data = data[_member_name]
            if isinstance(blob_data, bytes):
                evaluated_value = BytesIO(blob_data)
            elif hasattr(blob_data, "read"):
                # If it's already a file-like object, use it as is
                evaluated_value = blob_data
            else:
                raise ValueError(f"Unexpected blob data type: {type(blob_data)}")
        else:
            raise ValueError(f"Unexpected member type encountered: {_member_type}")

        result[attribute_name] = evaluated_value
        if object_instance:
            # 3. set attribute value
            setattr(object_instance, attribute_name, evaluated_value)

    return result
