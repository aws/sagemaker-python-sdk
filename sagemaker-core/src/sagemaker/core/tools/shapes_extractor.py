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
"""Extracts the shapes to DAG structure."""
import textwrap
import pprint
from functools import lru_cache
from typing import Optional, Any

from sagemaker.core.tools.constants import BASIC_JSON_TYPES_TO_PYTHON_TYPES, SHAPE_DAG_FILE_PATH
from sagemaker.core.utils.utils import (
    reformat_file_with_black,
    convert_to_snake_case,
    snake_to_pascal,
)
from sagemaker.core.tools.data_extractor import load_combined_shapes_data


class ShapesExtractor:
    """Extracts the shapes to DAG structure."""

    def __init__(self, combined_shapes: Optional[dict] = None):
        """
        Initializes a new instance of the ShapesExtractor class.

        :param combined_shapes: All the shapes put together from all Sagemaker Service JSONs
        """
        self.combined_shapes = combined_shapes or load_combined_shapes_data()

        self.shape_dag = self.get_shapes_dag()
        with open(SHAPE_DAG_FILE_PATH, "w") as f:
            f.write("SHAPE_DAG=")
            f.write(textwrap.indent(pprint.pformat(self.shape_dag, width=1), "") + "\n")
        reformat_file_with_black(SHAPE_DAG_FILE_PATH)

    # @property
    def get_shapes_dag(self):
        """
        Parses the Service Json and generates the Shape DAG.

        DAG is stored in a Dictionary data structure, and each key denotes a DAG Node.
        Nodes can be of composite types: structure, list, map. Basic types (Ex. str, int, etc)
        are omitted from compactness and can be inferred from composite type nodes.

        The connections of Nodes are can be followed by using the shape.

        Possible scenarios of nested associations:

        1. StructA → StructB → basic_type_member.
        2. StructA → list → basic_type_member
        3. StructA → list → StructB → basic_type_member
        4. StructA → map → basic_type_member
        5. StructA → map → StructBMapValue → basic_type_member
        6. StructA → map → map → basic_type_member
        7. StructA → map → list → basic_type_member

        Example:

           "ContainerDefinition": { # type: structure
               "type":"structure",
               "members":[
                    {"name": "ModelName", "shape": "ModelName", "type": "string"},
                    {"name": "ContainerDefinition", "shape": "ContainerDefinition", "type": "list"},
                    {"name": "CustomerMetadata", "shape": "CustomerMetadataMap", "type": "map"},
               ],
           },
           "ContainerDefinitionList": {  # type: list
                "type":"list",
                "member_shape":"ContainerDefinition",
                "member_type":"ContainerDefinition", # potential types: string, structure
           },
           "CustomerMetadataMap": { # type: map
                "type":"map",
                "key_shape":"CustomerMetadataKey",
                "key_type":"string",     # allowed types: string
                "value_shape":"CustomerMetadataValue",
                "value_type":"string", # potential types: string, structure, list, map
            },

        :return: The generated Shape DAG.
        :rtype: dict
        """
        _dag = {}
        _all_shapes = self.combined_shapes
        for shape, shape_attrs in _all_shapes.items():
            shape_data = _all_shapes[shape]
            if "type" not in shape_data:
                continue
            if shape_data["type"] == "structure":
                _dag[shape] = {"type": "structure", "members": []}
                for member, member_attrs in shape_data["members"].items():
                    shape_node_member = {"name": member, "shape": member_attrs["shape"]}
                    member_shape_dict = _all_shapes[member_attrs["shape"]]
                    shape_node_member["type"] = member_shape_dict["type"]
                    _dag[shape]["members"].append(shape_node_member)
            elif shape_data["type"] == "list":
                _dag[shape] = {"type": "list"}
                _list_member_shape = shape_data["member"]["shape"]
                _dag[shape]["member_shape"] = _list_member_shape
                _dag[shape]["member_type"] = _all_shapes[_list_member_shape]["type"]
            elif shape_data["type"] == "map":
                _dag[shape] = {"type": "map"}
                _map_key_shape = shape_data["key"]["shape"]
                _dag[shape]["key_shape"] = _map_key_shape
                _map_value_shape = shape_data["value"]["shape"]
                _dag[shape]["value_shape"] = _map_value_shape
                _dag[shape]["key_type"] = _all_shapes[_map_key_shape]["type"]
                _dag[shape]["value_type"] = _all_shapes[_map_value_shape]["type"]
        return _dag

    def _evaluate_list_type(self, member_shape):
        list_shape_name = member_shape["member"]["shape"]
        list_shape_member = self.combined_shapes[list_shape_name]
        list_shape_type = list_shape_member["type"]
        if list_shape_type == "list":
            member_type = f"List[{self._evaluate_list_type(list_shape_member)}]"
        elif list_shape_type == "map":
            member_type = f"List[{self._evaluate_map_type(list_shape_member)}]"
        elif list_shape_type == "structure":
            # handling an edge case of nested structure
            if list_shape_name == "SearchExpression":
                member_type = f"List['{list_shape_name}']"
            else:
                member_type = f"List[{list_shape_name}]"
        elif list_shape_type in BASIC_JSON_TYPES_TO_PYTHON_TYPES.keys():
            member_type = f"List[{BASIC_JSON_TYPES_TO_PYTHON_TYPES[list_shape_type]}]"
        else:
            raise Exception(
                f"Unhandled list shape key type {list_shape_type} for Shape: {list_shape_name} encountered, needs extra logic to handle this"
            )
        return member_type

    def _evaluate_map_type(self, member_shape):
        map_key_shape_name = member_shape["key"]["shape"]
        map_value_shape_name = member_shape["value"]["shape"]
        map_key_shape = self.combined_shapes[map_key_shape_name]
        map_value_shape = self.combined_shapes[map_value_shape_name]
        map_key_shape_type = map_key_shape["type"]
        map_value_shape_type = map_value_shape["type"]
        # Map keys are always expected to be "string" type
        if map_key_shape_type != "string":
            raise Exception(
                "Unhandled map shape key type encountered, needs extra logic to handle this"
            )
        if map_value_shape_type == "structure":
            member_type = (
                f"Dict[{BASIC_JSON_TYPES_TO_PYTHON_TYPES[map_key_shape_type]}, "
                f"{map_value_shape_name}]"
            )
        elif map_value_shape_type == "list":
            member_type = (
                f"Dict[{BASIC_JSON_TYPES_TO_PYTHON_TYPES[map_key_shape_type]}, "
                f"{self._evaluate_list_type(map_value_shape)}]"
            )
        elif map_value_shape_type == "map":
            member_type = (
                f"Dict[{BASIC_JSON_TYPES_TO_PYTHON_TYPES[map_key_shape_type]}, "
                f"{self._evaluate_map_type(map_value_shape)}]"
            )
        else:
            member_type = (
                f"Dict[{BASIC_JSON_TYPES_TO_PYTHON_TYPES[map_key_shape_type]}, "
                f"{BASIC_JSON_TYPES_TO_PYTHON_TYPES[map_value_shape_type]}]"
            )
        return member_type

    def generate_data_shape_members_and_string_body(
        self, shape, resource_plan: Optional[Any] = None, required_override=()
    ):
        shape_members = self.generate_shape_members(shape, required_override)
        resource_names = None
        if resource_plan is not None:
            resource_names = [row["resource_name"] for _, row in resource_plan.iterrows()]
        init_data_body = ""
        for attr, value in shape_members.items():
            if (
                resource_names
                and attr.endswith("name")
                and attr[: -len("_name")] != shape
                and attr != "name"
                and snake_to_pascal(attr[: -len("_name")]) in resource_names
            ):
                if value.startswith("Optional"):
                    init_data_body += (
                        f"{attr}: Optional[Union[StrPipeVar, object]] = Unassigned()\n"
                    )
                else:
                    init_data_body += f"{attr}: Union[StrPipeVar, object]\n"
            elif attr == "lambda":
                init_data_body += f"# {attr}: {value}\n"
            else:
                init_data_body += f"{attr}: {value}\n"
        return shape_members, init_data_body

    def generate_data_shape_string_body(self, shape, resource_plan, required_override=()):
        return self.generate_data_shape_members_and_string_body(
            shape, resource_plan, required_override
        )[1]

    def generate_data_shape_members(self, shape, resource_plan, required_override=()):
        return self.generate_data_shape_members_and_string_body(
            shape, resource_plan, required_override
        )[0]

    @lru_cache
    def generate_shape_members(self, shape, required_override=()):
        shape_dict = self.combined_shapes[shape]
        members = shape_dict["members"]
        required_args = list(required_override) or shape_dict.get("required", [])
        init_data_body = {}
        # bring the required members in front
        ordered_members = {key: members[key] for key in required_args if key in members}
        ordered_members.update(members)
        for member_name, member_attrs in ordered_members.items():
            member_shape_name = member_attrs["shape"]
            if self.combined_shapes[member_shape_name]:
                member_shape = self.combined_shapes[member_shape_name]
                member_shape_type = member_shape["type"]
                if member_shape_type == "structure":
                    member_type = member_shape_name
                elif member_shape_type == "list":
                    member_type = self._evaluate_list_type(member_shape)
                elif member_shape_type == "map":
                    member_type = self._evaluate_map_type(member_shape)
                else:
                    # Shape is a simple type like string
                    member_type = BASIC_JSON_TYPES_TO_PYTHON_TYPES[member_shape_type]
            else:
                raise Exception("The Shape definition mush exist. The Json Data might be corrupt")
            member_name_snake_case = convert_to_snake_case(member_name)
            if member_name in required_args:
                init_data_body[f"{member_name_snake_case}"] = f"{member_type}"
            else:
                init_data_body[f"{member_name_snake_case}"] = (
                    f"Optional[{member_type}] = Unassigned()"
                )
        return init_data_body

    @lru_cache
    def fetch_shape_members_and_doc_strings(self, shape, required_override=()):
        shape_dict = self.combined_shapes[shape]
        members = shape_dict["members"]
        required_args = list(required_override) or shape_dict.get("required", [])
        # bring the required members in front
        ordered_members = {key: members[key] for key in required_args if key in members}
        ordered_members.update(members)
        shape_members_and_docstrings = {}
        for member_name, member_attrs in ordered_members.items():
            member_shape_documentation = member_attrs.get("documentation")
            shape_members_and_docstrings[member_name] = member_shape_documentation
        return shape_members_and_docstrings

    def get_required_members(self, shape):
        shape_dict = self.combined_shapes[shape]
        required_args = shape_dict.get("required", [])

        return [convert_to_snake_case(arg) for arg in required_args]
