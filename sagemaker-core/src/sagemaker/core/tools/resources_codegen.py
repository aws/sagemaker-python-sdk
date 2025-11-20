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
"""Generates the resource classes for the service model."""
from functools import lru_cache

import os
import json
from sagemaker.core.utils.code_injection.codec import pascal_to_snake
from sagemaker.core.config_schema import SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA
from sagemaker.core.utils.exceptions import IntelligentDefaultsError
from sagemaker.core.utils.utils import get_textual_rich_logger
from sagemaker.core.tools.constants import (
    BASIC_RETURN_TYPES,
    GENERATED_CLASSES_LOCATION,
    RESOURCES_CODEGEN_FILE_NAME,
    LICENCES_STRING,
    TERMINAL_STATES,
    LOGGER_STRING,
    CONFIG_SCHEMA_FILE_NAME,
    PYTHON_TYPES_TO_BASIC_JSON_TYPES,
    CONFIGURABLE_ATTRIBUTE_SUBSTRINGS,
    RESOURCE_WITH_LOGS,
)
from sagemaker.core.tools.method import Method, MethodType
from sagemaker.core.utils.utils import (
    add_indent,
    convert_to_snake_case,
    snake_to_pascal,
    remove_html_tags,
    escape_special_rst_characters,
)
from sagemaker.core.tools.resources_extractor import ResourcesExtractor
from sagemaker.core.tools.shapes_extractor import ShapesExtractor
from sagemaker.core.tools.templates import (
    CALL_OPERATION_API_NO_ARG_TEMPLATE,
    CALL_OPERATION_API_TEMPLATE,
    CREATE_METHOD_TEMPLATE,
    DELETE_FAILED_STATUS_CHECK,
    DELETED_STATUS_CHECK,
    DESERIALIZE_INPUT_AND_RESPONSE_TO_CLS_TEMPLATE,
    DESERIALIZE_RESPONSE_TEMPLATE,
    DESERIALIZE_RESPONSE_TO_BASIC_TYPE_TEMPLATE,
    GENERIC_METHOD_TEMPLATE,
    GET_METHOD_TEMPLATE,
    INITIALIZE_CLIENT_TEMPLATE,
    REFRESH_METHOD_TEMPLATE,
    RESOURCE_BASE_CLASS_TEMPLATE,
    RETURN_ITERATOR_TEMPLATE,
    SERIALIZE_INPUT_TEMPLATE,
    STOP_METHOD_TEMPLATE,
    DELETE_METHOD_TEMPLATE,
    WAIT_FOR_DELETE_METHOD_TEMPLATE,
    WAIT_METHOD_TEMPLATE,
    WAIT_FOR_STATUS_METHOD_TEMPLATE,
    UPDATE_METHOD_TEMPLATE,
    POPULATE_DEFAULTS_DECORATOR_TEMPLATE,
    CREATE_METHOD_TEMPLATE_WITHOUT_DEFAULTS,
    IMPORT_METHOD_TEMPLATE,
    FAILED_STATUS_ERROR_TEMPLATE,
    GET_NAME_METHOD_TEMPLATE,
    GET_ALL_METHOD_NO_ARGS_TEMPLATE,
    GET_ALL_METHOD_WITH_ARGS_TEMPLATE,
    UPDATE_METHOD_TEMPLATE_WITHOUT_DECORATOR,
    RESOURCE_METHOD_EXCEPTION_DOCSTRING,
    INIT_WAIT_LOGS_TEMPLATE,
    PRINT_WAIT_LOGS,
    SERIALIZE_INPUT_ENDPOINT_TEMPLATE,
    DESERIALIZE_RESPONSE_ENDPOINT_TEMPLATE,
)
from sagemaker.core.tools.data_extractor import (
    load_combined_shapes_data,
    load_combined_operations_data,
)

log = get_textual_rich_logger(__name__)

TYPE = "type"
OBJECT = "object"
PROPERTIES = "properties"
SAGEMAKER = "SageMaker"
PYTHON_SDK = "PythonSDK"
SCHEMA_VERSION = "SchemaVersion"
RESOURCES = "Resources"
REQUIRED = "required"
GLOBAL_DEFAULTS = "GlobalDefaults"


class ResourcesCodeGen:
    """
    A class for generating resources based on a service JSON file.

    Args:
        service_json (dict): The Botocore service.json containing the shape definitions.

    Attributes:
        service_json (dict): The Botocore service.json containing the shape definitions.
        version (str): The API version of the service.
        protocol (str): The protocol used by the service.
        service (str): The full name of the service.
        service_id (str): The ID of the service.
        uid (str): The unique identifier of the service.
        operations (dict): The operations supported by the service.
        shapes (dict): The shapes used by the service.
        resources_extractor (ResourcesExtractor): An instance of the ResourcesExtractor class.
        resources_plan (DataFrame): The resource plan in dataframe format.
        shapes_extractor (ShapesExtractor): An instance of the ShapesExtractor class.

    Raises:
        Exception: If the service ID is not supported or the protocol is not supported.

    """

    def __init__(self, service_json: dict):
        # Initialize the service_json dict
        self.service_json = service_json

        # Extract the metadata
        metadata = self.service_json["metadata"]
        self.version = metadata["apiVersion"]
        self.protocol = metadata["protocol"]
        self.service = metadata["serviceFullName"]
        self.service_id = metadata["serviceId"]
        self.uid = metadata["uid"]

        # Check if the service ID and protocol are supported
        if self.service_id != "SageMaker":
            raise Exception(f"ServiceId {self.service_id} not supported in this resource generator")
        if self.protocol != "json":
            raise Exception(f"Protocol {self.protocol} not supported in this resource generator")

        # Extract the operations and shapes
        self.operations = load_combined_operations_data()
        self.shapes = load_combined_shapes_data()

        # Initialize the resources and shapes extractors
        self.resources_extractor = ResourcesExtractor()
        self.shapes_extractor = ShapesExtractor()

        # Extract the resources plan and shapes DAG
        self.resources_plan = self.resources_extractor.get_resource_plan()
        self.resource_methods = self.resources_extractor.get_resource_methods()
        self.shape_dag = self.shapes_extractor.get_shapes_dag()

        # Create the Config Schema
        self.generate_config_schema()
        # Generate the resources
        self.generate_resources()

    def generate_license(self) -> str:
        """
        Generate the license for the generated resources file.

        Returns:
            str: The license.

        """
        return LICENCES_STRING

    def generate_imports(self) -> str:
        """
        Generate the import statements for the generated resources file.

        Returns:
            str: The import statements.
        """
        # List of import statements
        imports = [
            "import botocore",
            "import datetime",
            "import time",
            "import functools",
            "from pydantic import validate_call",
            "from typing import Dict, List, Literal, Optional, Union, Any\n"
            "from boto3.session import Session",
            "from rich.console import Group",
            "from rich.live import Live",
            "from rich.panel import Panel",
            "from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn",
            "from rich.status import Status",
            "from rich.style import Style",
            "from sagemaker.core.shapes import *",
            "from sagemaker.core.helper.pipeline_variable import StrPipeVar",
            "from sagemaker.core.utils.code_injection.codec import transform",
            "from sagemaker.core.utils.code_injection.constants import Color",
            "from sagemaker.core.utils.utils import SageMakerClient, ResourceIterator, Unassigned, get_textual_rich_logger, "
            "snake_to_pascal, pascal_to_snake, is_not_primitive, is_not_str_dict, is_primitive_list, serialize",
            "from sagemaker.core.config.config_manager import SageMakerConfig",
            "from sagemaker.core.utils.logs import MultiLogStreamHandler",
            "from sagemaker.core.utils.exceptions import *",
            "from typing import ClassVar",
            "from sagemaker.core.serializers.base import BaseSerializer",
            "from sagemaker.core.deserializers.base import BaseDeserializer",
        ]

        formated_imports = "\n".join(imports)
        formated_imports += "\n\n"

        # Join the import statements with a newline character and return
        return formated_imports

    def generate_base_class(self) -> str:
        """
        Generate the base class for the resources.

        Returns:
            str: The base class.

        """
        return RESOURCE_BASE_CLASS_TEMPLATE

    def generate_logging(self) -> str:
        """
        Generate the logging statements for the generated resources file.

        Returns:
            str: The logging statements.

        """
        return LOGGER_STRING

    @staticmethod
    def generate_defaults_decorator(
        config_schema_for_resource: dict, resource_name: str, class_attributes: dict
    ) -> str:
        return POPULATE_DEFAULTS_DECORATOR_TEMPLATE.format(
            config_schema_for_resource=add_indent(
                json.dumps(config_schema_for_resource.get(PROPERTIES), indent=2), 4
            ),
            resource_name=resource_name,
            configurable_attributes=CONFIGURABLE_ATTRIBUTE_SUBSTRINGS,
            class_attributes=class_attributes,
        )

    def generate_resources(
        self,
        output_folder: str = GENERATED_CLASSES_LOCATION,
        file_name: str = RESOURCES_CODEGEN_FILE_NAME,
    ) -> None:
        """
        Generate the resources file.

        Args:
            output_folder (str, optional): The output folder path. Defaults to "GENERATED_CLASSES_LOCATION".
            file_name (str, optional): The output file name. Defaults to "RESOURCES_CODEGEN_FILE_NAME".
        """
        # Check if the output folder exists, if not, create it
        os.makedirs(output_folder, exist_ok=True)

        # Create the full path for the output file
        output_file = os.path.join(output_folder, file_name)

        # Open the output file
        with open(output_file, "w") as file:
            # Generate and write the license to the file
            file.write(self.generate_license())

            # Generate and write the imports to the file
            file.write(self.generate_imports())

            # Generate and write the logging statements to the file
            file.write(self.generate_logging())

            # Generate and write the base class to the file
            file.write(self.generate_base_class())

            self.resource_names = [
                row["resource_name"] for _, row in self.resources_plan.iterrows()
            ]
            # Iterate over the rows in the resources plan
            for _, row in self.resources_plan.iterrows():
                # Extract the necessary data from the row
                resource_name = row["resource_name"]
                class_methods = row["class_methods"]
                object_methods = row["object_methods"]
                additional_methods = row["additional_methods"]
                raw_actions = row["raw_actions"]
                resource_status_chain = row["resource_status_chain"]
                resource_states = row["resource_states"]

                # Generate the resource class
                resource_class = self.generate_resource_class(
                    resource_name,
                    class_methods,
                    object_methods,
                    additional_methods,
                    raw_actions,
                    resource_status_chain,
                    resource_states,
                )

                # If the resource class was successfully generated, write it to the file
                if resource_class:
                    file.write(f"{resource_class}\n\n")

    def _evaluate_method(
        self, resource_name: str, method_name: str, methods: list, **kwargs
    ) -> str:
        """Evaluate the specified method for a resource.

        Args:
            resource_name (str): The name of the resource.
            method_name (str): The name of the method to evaluate.
            methods (list): The list of methods for the resource.

        Returns:
            str: Formatted method if needed for a resource, else returns an empty string.
        """
        if method_name in methods:
            return getattr(self, f"generate_{method_name}_method")(resource_name, **kwargs)
        else:
            # log.warning(f"Resource {resource_name} does not have a {method_name.upper()} method")
            return ""

    def generate_resource_class(
        self,
        resource_name: str,
        class_methods: list,
        object_methods: list,
        additional_methods: list,
        raw_actions: list,
        resource_status_chain: list,
        resource_states: list,
    ) -> str:
        """
        Generate the resource class for a resource.

        Args:
            resource_name (str): The name of the resource.
            class_methods (list): The class methods.
            object_methods (list): The object methods.
            additional_methods (list): The additional methods.
            raw_actions (list): The raw actions.

        Returns:
            str: The formatted resource class.

        """
        # Initialize an empty string for the resource class
        resource_class = ""

        # _get_class_attributes will return value only if the resource has get or get_all method
        if class_attribute_info := self._get_class_attributes(resource_name, class_methods):
            class_attributes, class_attributes_string, attributes_and_documentation = (
                class_attribute_info
            )

            # Start defining the class
            resource_class = f"class {resource_name}(Base):\n"

            class_documentation_string = f"Class representing resource {resource_name}\n\n"
            class_documentation_string += f"Attributes:\n"
            class_documentation_string += self._get_shape_attr_documentation_string(
                attributes_and_documentation
            )
            resource_attributes = list(class_attributes.keys())

            defaults_decorator_method = ""
            # Check if 'create' is in the class methods
            if "create" in class_methods or "update" in class_methods:
                if config_schema_for_resource := self._get_config_schema_for_resources().get(
                    resource_name
                ):
                    defaults_decorator_method = self.generate_defaults_decorator(
                        resource_name=resource_name,
                        class_attributes=class_attributes,
                        config_schema_for_resource=config_schema_for_resource,
                    )
            needs_defaults_decorator = defaults_decorator_method != ""

            # Add the class attributes and methods to the class definition
            resource_class += add_indent(f'"""\n{class_documentation_string}\n"""\n', 4)

            # Add the class attributes and methods to the class definition
            resource_class += add_indent(class_attributes_string, 4)

            resource_lower = convert_to_snake_case(resource_name)
            get_name_method = self.generate_get_name_method(resource_lower=resource_lower)
            resource_class += add_indent(get_name_method, 4)

            if defaults_decorator_method:
                resource_class += "\n"
                resource_class += add_indent(defaults_decorator_method, 4)

            if create_method := self._evaluate_method(
                resource_name,
                "create",
                class_methods,
                needs_defaults_decorator=needs_defaults_decorator,
            ):
                resource_class += add_indent(create_method, 4)

            if get_method := self._evaluate_method(
                resource_name,
                "get",
                class_methods,
            ):
                resource_class += add_indent(get_method, 4)

            if refresh_method := self._evaluate_method(
                resource_name, "refresh", object_methods, resource_attributes=resource_attributes
            ):
                resource_class += add_indent(refresh_method, 4)

            if update_method := self._evaluate_method(
                resource_name,
                "update",
                object_methods,
                resource_attributes=resource_attributes,
                needs_defaults_decorator=needs_defaults_decorator,
            ):
                resource_class += add_indent(update_method, 4)

            if delete_method := self._evaluate_method(
                resource_name, "delete", object_methods, resource_attributes=resource_attributes
            ):
                resource_class += add_indent(delete_method, 4)

            if start_method := self._evaluate_method(
                resource_name, "start", object_methods, resource_attributes=resource_attributes
            ):
                resource_class += add_indent(start_method, 4)

            if stop_method := self._evaluate_method(resource_name, "stop", object_methods):
                resource_class += add_indent(stop_method, 4)

            if wait_method := self._evaluate_method(resource_name, "wait", object_methods):
                resource_class += add_indent(wait_method, 4)

            if wait_for_status_method := self._evaluate_method(
                resource_name, "wait_for_status", object_methods
            ):
                resource_class += add_indent(wait_for_status_method, 4)

            if wait_for_delete_method := self._evaluate_method(
                resource_name, "wait_for_delete", object_methods
            ):
                resource_class += add_indent(wait_for_delete_method, 4)

            if import_method := self._evaluate_method(resource_name, "import", class_methods):
                resource_class += add_indent(import_method, 4)

            if list_method := self._evaluate_method(resource_name, "get_all", class_methods):
                resource_class += add_indent(list_method, 4)

        else:
            # If there's no 'get' or 'list' or 'create' method, generate a class with no attributes
            resource_attributes = []
            resource_class = f"class {resource_name}(Base):\n"
            class_documentation_string = f"Class representing resource {resource_name}\n"
            resource_class += add_indent(f'"""\n{class_documentation_string}\n"""\n', 4)

        if resource_name in self.resource_methods:
            # TODO: use resource_methods for all methods
            for method in self.resource_methods[resource_name].values():
                formatted_method = self.generate_method(method, resource_attributes)
                resource_class += add_indent(formatted_method, 4)

        # Return the class definition
        return resource_class

    def _get_class_attributes(self, resource_name: str, class_methods: list) -> tuple:
        """Get the class attributes for a resource.

        Args:
            resource_name (str): The name of the resource.
            class_methods (list): The class methods of the resource. Now it can only get the class
                attributes if the resource has get or get_all method.

        Returns:
            tuple:
                class_attributes: The class attributes and the formatted class attributes string.
                class_attributes_string: The code string of the class attributes
                attributes_and_documentation: A dict of doc strings of the class attributes
        """
        if "get" in class_methods:
            # Get the operation and shape for the 'get' method
            get_operation = self.operations["Describe" + resource_name]
            get_operation_shape = get_operation["output"]["shape"]

            # Use 'get' operation input as the required class attributes.
            # These are the mimumum identifing attributes for a resource object (ie, required for refresh())
            get_operation_input_shape = get_operation["input"]["shape"]
            required_attributes = self.shapes[get_operation_input_shape].get("required", [])

            # Generate the class attributes based on the shape
            class_attributes, class_attributes_string = (
                self.shapes_extractor.generate_data_shape_members_and_string_body(
                    shape=get_operation_shape, required_override=tuple(required_attributes)
                )
            )
            attributes_and_documentation = (
                self.shapes_extractor.fetch_shape_members_and_doc_strings(get_operation_shape)
            )
            # Some resources are configured in the service.json inconsistently.
            # These resources take in the main identifier in the create and get methods , but is not present in the describe response output
            # Hence for consistent behaviour of functions such as refresh and delete, the identifiers are hardcoded
            if resource_name == "ImageVersion":
                class_attributes["image_name"] = "StrPipeVar"
                class_attributes_string = "image_name: StrPipeVar\n" + class_attributes_string
            if resource_name == "Workteam":
                class_attributes["workteam_name"] = "StrPipeVar"
                class_attributes_string = "workteam_name: StrPipeVar\n" + class_attributes_string
            if resource_name == "Workforce":
                class_attributes["workforce_name"] = "StrPipeVar"
                class_attributes_string = "workforce_name: StrPipeVar\n" + class_attributes_string
            if resource_name == "SubscribedWorkteam":
                class_attributes["workteam_arn"] = "StrPipeVar"
                class_attributes_string = "workteam_arn: StrPipeVar\n" + class_attributes_string

            if resource_name == "HubContent":
                class_attributes["hub_name"] = "Optional[str] = Unassigned()"
                class_attributes_string = class_attributes_string.replace("hub_name: str", "")
                class_attributes_string = (
                    class_attributes_string + "hub_name: Optional[str] = Unassigned()"
                )
            if resource_name == "Endpoint":
                class_attributes["serializer"] = "Optional[BaseSerializer] = None"
                class_attributes_string = class_attributes_string.replace(
                    "serializer: BaseSerializer", ""
                )
                class_attributes_string = (
                    class_attributes_string + "serializer: Optional[BaseSerializer] = None\n"
                )
                class_attributes["deserializer"] = "Optional[BaseDeserializer] = None"
                class_attributes_string = class_attributes_string.replace(
                    "deserializer: BaseDeserializer", ""
                )
                class_attributes_string = (
                    class_attributes_string + "deserializer: Optional[BaseDeserializer] = None\n"
                )

            return class_attributes, class_attributes_string, attributes_and_documentation
        elif "get_all" in class_methods:
            # Get the operation and shape for the 'get_all' method
            list_operation = self.operations["List" + resource_name + "s"]
            list_operation_output_shape = list_operation["output"]["shape"]
            list_operation_output_members = self.shapes[list_operation_output_shape]["members"]

            # Use the object shape of 'get_all' operation output as the required class attributes.
            filtered_list_operation_output_members = next(
                {key: value}
                for key, value in list_operation_output_members.items()
                if key != "NextToken"
            )

            summaries_key = next(iter(filtered_list_operation_output_members))
            summaries_shape_name = filtered_list_operation_output_members[summaries_key]["shape"]
            summary_name = self.shapes[summaries_shape_name]["member"]["shape"]
            required_attributes = self.shapes[summary_name].get("required", [])
            # Generate the class attributes based on the shape
            class_attributes, class_attributes_string = (
                self.shapes_extractor.generate_data_shape_members_and_string_body(
                    shape=summary_name, required_override=tuple(required_attributes)
                )
            )
            attributes_and_documentation = (
                self.shapes_extractor.fetch_shape_members_and_doc_strings(summary_name)
            )
            return class_attributes, class_attributes_string, attributes_and_documentation
        elif "create" in class_methods:
            # Get the operation and shape for the 'create' method
            create_operation = self.operations["Create" + resource_name]
            create_operation_input_shape = create_operation["input"]["shape"]
            create_operation_output_shape = create_operation["output"]["shape"]
            # Generate the class attributes based on the input and output shape
            class_attributes, class_attributes_string = self._get_resource_members_and_string_body(
                resource_name, create_operation_input_shape, create_operation_output_shape
            )
            attributes_and_documentation = self._get_resouce_attributes_and_documentation(
                create_operation_input_shape, create_operation_output_shape
            )
            return class_attributes, class_attributes_string, attributes_and_documentation
        else:
            return None

    def _get_resource_members_and_string_body(self, resource_name: str, input_shape, output_shape):
        input_members = self.shapes_extractor.generate_shape_members(input_shape)
        output_members = self.shapes_extractor.generate_shape_members(output_shape)
        resource_members = {**input_members, **output_members}
        # bring the required members in front
        ordered_members = {
            attr: value
            for attr, value in resource_members.items()
            if not value.startswith("Optional")
        }
        ordered_members.update(resource_members)

        resource_name_snake_case = pascal_to_snake(resource_name)
        resource_names = [row["resource_name"] for _, row in self.resources_plan.iterrows()]
        init_data_body = ""
        for attr, value in ordered_members.items():
            if (
                resource_names
                and attr.endswith("name")
                and attr[: -len("_name")] != resource_name_snake_case
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
        return ordered_members, init_data_body

    def _get_resouce_attributes_and_documentation(self, input_shape, output_shape):
        input_members = self.shapes[input_shape]["members"]
        required_args = set(self.shapes[input_shape].get("required", []))
        output_members = self.shapes[output_shape]["members"]
        members = {**input_members, **output_members}
        required_args.update(self.shapes[output_shape].get("required", []))
        # bring the required members in front
        ordered_members = {key: members[key] for key in members if key in required_args}
        ordered_members.update(members)
        shape_members_and_docstrings = {}
        for member_name, member_attrs in ordered_members.items():
            member_shape_documentation = member_attrs.get("documentation")
            shape_members_and_docstrings[member_name] = member_shape_documentation
        return shape_members_and_docstrings

    def _get_shape_attr_documentation_string(
        self, attributes_and_documentation, exclude_resource_attrs=None
    ) -> str:
        documentation_string = ""
        for attribute, documentation in attributes_and_documentation.items():
            attribute_snake = pascal_to_snake(attribute)
            if exclude_resource_attrs and attribute_snake in exclude_resource_attrs:
                #  exclude resource attributes from documentation
                continue
            else:
                if documentation == None:
                    documentation_string += f"{attribute_snake}: \n"
                else:
                    documentation_string += f"{attribute_snake}: {documentation}\n"
        documentation_string = add_indent(documentation_string)
        documentation_string = remove_html_tags(documentation_string)
        return escape_special_rst_characters(documentation_string)

    def _generate_create_method_args(
        self, operation_input_shape_name: str, resource_name: str
    ) -> str:
        """Generates the arguments for a method.
        Args:
            operation_input_shape_name (str): The name of the input shape for the operation.
        Returns:
            str: The generated arguments string.
        """
        typed_shape_members = self.shapes_extractor.generate_shape_members(
            operation_input_shape_name
        )
        resource_name_in_snake_case = pascal_to_snake(resource_name)
        method_args = ""
        last_key = list(typed_shape_members.keys())[-1]
        for attr, attr_type in typed_shape_members.items():
            method_parameter_type = attr_type
            if (
                attr.endswith("name")
                and attr[: -len("_name")] != resource_name_in_snake_case
                and attr != "name"
                and snake_to_pascal(attr[: -len("_name")]) in self.resource_names
            ):
                if attr_type.startswith("Optional"):
                    method_args += f"{attr}: Optional[Union[StrPipeVar, object]] = Unassigned(),"
                else:
                    method_args += f"{attr}: Union[StrPipeVar, object],"
            else:
                method_args += f"{attr}: {method_parameter_type},"
            if attr != last_key:
                method_args += "\n"
        method_args = add_indent(method_args)
        return method_args

    # TODO: use this method to replace _generate_operation_input_args
    def _generate_operation_input_args_updated(
        self,
        resource_operation: dict,
        is_class_method: bool,
        resource_attributes: list = [],
        exclude_list: list = [],
    ) -> str:
        """Generate the operation input arguments string.

        Args:
            resource_operation (dict): The resource operation dictionary.
            is_class_method (bool): Indicates method is class method, else object method.

        Returns:
            str: The formatted operation input arguments string.
        """
        input_shape_name = resource_operation["input"]["shape"]
        input_shape_members = list(self.shapes[input_shape_name]["members"].keys())

        if is_class_method:
            args = (
                f"'{member}': {convert_to_snake_case(member)}"
                for member in input_shape_members
                if convert_to_snake_case(member) not in exclude_list
            )
        else:
            args = []
            for member in input_shape_members:
                if convert_to_snake_case(member) not in exclude_list:
                    if convert_to_snake_case(member) in resource_attributes:
                        args.append(f"'{member}': self.{convert_to_snake_case(member)}")
                    else:
                        args.append(f"'{member}': {convert_to_snake_case(member)}")

        operation_input_args = ",\n".join(args)
        operation_input_args += ","
        operation_input_args = add_indent(operation_input_args, 8)

        return operation_input_args

    def _generate_operation_input_args(
        self, resource_operation: dict, is_class_method: bool, exclude_list: list = []
    ) -> str:
        """Generate the operation input arguments string.

        Args:
            resource_operation (dict): The resource operation dictionary.
            is_class_method (bool): Indicates method is class method, else object method.

        Returns:
            str: The formatted operation input arguments string.
        """
        input_shape_name = resource_operation["input"]["shape"]
        input_shape_members = list(self.shapes[input_shape_name]["members"].keys())

        if is_class_method:
            args = (
                f"'{member}': {convert_to_snake_case(member)}"
                for member in input_shape_members
                if convert_to_snake_case(member) not in exclude_list
            )
        else:
            args = (
                f"'{member}': self.{convert_to_snake_case(member)}"
                for member in input_shape_members
                if convert_to_snake_case(member) not in exclude_list
            )

        operation_input_args = ",\n".join(args)
        operation_input_args += ","
        operation_input_args = add_indent(operation_input_args, 8)

        return operation_input_args

    def _generate_operation_input_necessary_args(
        self, resource_operation: dict, resource_attributes: list
    ) -> str:
        """
        Generate the operation input arguments string.
        This will try to re-use args from the object attributes if present and it not presebt will use te ones provided in the parameter.
        Args:
            resource_operation (dict): The resource operation dictionary.
            is_class_method (bool): Indicates method is class method, else object method.

        Returns:
            str: The formatted operation input arguments string.
        """
        input_shape_name = resource_operation["input"]["shape"]
        input_shape_members = list(self.shapes[input_shape_name]["members"].keys())

        args = list()
        for member in input_shape_members:
            if convert_to_snake_case(member) in resource_attributes:
                args.append(f"'{member}': self.{convert_to_snake_case(member)}")
            else:
                args.append(f"'{member}': {convert_to_snake_case(member)}")

        operation_input_args = ",\n".join(args)
        operation_input_args += ","
        operation_input_args = add_indent(operation_input_args, 8)

        return operation_input_args

    def _generate_method_args(
        self, operation_input_shape_name: str, exclude_list: list = []
    ) -> str:
        """Generates the arguments for a method.
        This will exclude attributes in the exclude_list from the arguments. For example, This is used for update() method
         which does not require the resource identifier attributes to be passed as arguments.

        Args:
            operation_input_shape_name (str): The name of the input shape for the operation.
            exclude_list (list): The list of attributes to exclude from the arguments.

        Returns:
            str: The generated arguments string.
        """
        typed_shape_members = self.shapes_extractor.generate_shape_members(
            operation_input_shape_name
        )

        args = (
            f"{attr}: {attr_type}"
            for attr, attr_type in typed_shape_members.items()
            if attr not in exclude_list
        )
        method_args = ",\n".join(args)
        if not method_args:
            return ""
        method_args += ","
        method_args = add_indent(method_args)
        return method_args

    def _generate_get_args(self, resource_name: str, operation_input_shape_name: str) -> str:
        """
        Generates a resource identifier based on the required members for the Describe and Create operations.

        Args:
            resource_name (str): The name of the resource.
            operation_input_shape_name (str): The name of the input shape for the operation.

        Returns:
            str: The generated resource identifier.
        """
        describe_operation = self.operations["Describe" + resource_name]
        describe_operation_input_shape_name = describe_operation["input"]["shape"]

        required_members = self.shapes_extractor.get_required_members(
            describe_operation_input_shape_name
        )

        operation_required_members = self.shapes_extractor.get_required_members(
            operation_input_shape_name
        )

        identifiers = []
        for member in required_members:
            if member not in operation_required_members:
                identifiers.append(f"{member}=response['{snake_to_pascal(member)}']")
            else:
                identifiers.append(f"{member}={member}")

        get_args = ", ".join(identifiers)
        return get_args

    def generate_create_method(self, resource_name: str, **kwargs) -> str:
        """
        Auto-generate the CREATE method for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted Create Method template.

        """
        # Get the operation and shape for the 'create' method
        operation_name = "Create" + resource_name
        operation_metadata = self.operations[operation_name]
        operation_input_shape_name = operation_metadata["input"]["shape"]

        # Generate the arguments for the 'create' method
        create_args = self._generate_create_method_args(operation_input_shape_name, resource_name)

        operation_input_args = self._generate_operation_input_args(
            operation_metadata, is_class_method=True
        )

        # Convert the resource name to snake case
        resource_lower = convert_to_snake_case(resource_name)

        # Convert the operation name to snake case
        operation = convert_to_snake_case(operation_name)

        docstring = self._generate_docstring(
            title=f"Create a {resource_name} resource",
            operation_name=operation_name,
            resource_name=resource_name,
            operation_input_shape_name=operation_input_shape_name,
            include_session_region=True,
            include_return_resource_docstring=True,
            include_intelligent_defaults_errors=True,
        )

        if "Describe" + resource_name in self.operations:
            # If the resource has Describe method, call Describe API and return its value
            get_args = self._generate_get_args(resource_name, operation_input_shape_name)

            # Format the method using the CREATE_METHOD_TEMPLATE
            if kwargs["needs_defaults_decorator"]:
                formatted_method = CREATE_METHOD_TEMPLATE.format(
                    docstring=docstring,
                    resource_name=resource_name,
                    create_args=create_args,
                    resource_lower=resource_lower,
                    # TODO: change service name based on the service - runtime, sagemaker, etc.
                    service_name="sagemaker",
                    operation_input_args=operation_input_args,
                    operation=operation,
                    get_args=get_args,
                )
            else:
                formatted_method = CREATE_METHOD_TEMPLATE_WITHOUT_DEFAULTS.format(
                    docstring=docstring,
                    resource_name=resource_name,
                    create_args=create_args,
                    resource_lower=resource_lower,
                    # TODO: change service name based on the service - runtime, sagemaker, etc.
                    service_name="sagemaker",
                    operation_input_args=operation_input_args,
                    operation=operation,
                    get_args=get_args,
                )
            # Return the formatted method
            return formatted_method
        else:
            # If the resource does not have Describe method, return a instance with
            # the input and output of Create method
            decorator = "@classmethod"
            serialize_operation_input = SERIALIZE_INPUT_TEMPLATE.format(
                operation_input_args=operation_input_args
            )
            initialize_client = INITIALIZE_CLIENT_TEMPLATE.format(service_name="sagemaker")
            call_operation_api = CALL_OPERATION_API_TEMPLATE.format(
                operation=convert_to_snake_case(operation_name)
            )
            operation_output_shape_name = operation_metadata["output"]["shape"]
            deserialize_response = DESERIALIZE_INPUT_AND_RESPONSE_TO_CLS_TEMPLATE.format(
                operation_output_shape=operation_output_shape_name
            )
            method_args = (
                add_indent("cls,\n", 4)
                + create_args
                + "\n"
                + add_indent("session: Optional[Session] = None,\n", 4)
                + add_indent("region: Optional[str] = None,", 4)
            )
            formatted_method = GENERIC_METHOD_TEMPLATE.format(
                docstring=docstring,
                decorator=decorator,
                method_name="create",
                method_args=method_args,
                return_type=f'Optional["{resource_name}"]',
                serialize_operation_input=serialize_operation_input,
                initialize_client=initialize_client,
                call_operation_api=call_operation_api,
                deserialize_response=deserialize_response,
            )
            # Return the formatted method
            return formatted_method

    @lru_cache
    def _fetch_shape_errors_and_doc_strings(self, operation):
        operation_dict = self.operations[operation]
        errors = operation_dict.get("errors", [])
        shape_errors_and_docstrings = {}
        if errors:
            for e in errors:
                error_shape = e["shape"]
                error_shape_dict = self.shapes[error_shape]
                error_shape_documentation = error_shape_dict.get("documentation")
                if error_shape_documentation:
                    error_shape_documentation.strip()
                shape_errors_and_docstrings[error_shape] = error_shape_documentation
        sorted_keys = sorted(shape_errors_and_docstrings.keys())
        return {key: shape_errors_and_docstrings[key] for key in sorted_keys}

    def _exception_docstring(self, operation: str) -> str:
        _docstring = RESOURCE_METHOD_EXCEPTION_DOCSTRING
        for error, documentaion in self._fetch_shape_errors_and_doc_strings(operation).items():
            if documentaion:
                _docstring += f"\n    {error}: {remove_html_tags(documentaion).strip()}"
            else:
                _docstring += f"\n    {error}"
        return _docstring

    def _generate_docstring(
        self,
        title: str,
        operation_name: str,
        resource_name: str,
        operation_input_shape_name: str = None,
        include_session_region: bool = False,
        include_return_resource_docstring: bool = False,
        return_string: str = None,
        include_intelligent_defaults_errors: bool = False,
        exclude_resource_attrs: list = None,
    ) -> str:
        """
        Generate the docstring for a method of a resource.

        Args:
            title (str): The title of the docstring.
            operation_name (str): The name of the operation.
            resource_name (str): The name of the resource.
            operation_input_shape_name (str): The name of the operation input shape.
            include_session_region (bool): Whether to include session and region documentation.
            include_return_resource_docstring (bool): Whether to include resource-specific documentation.
            return_string (str): The return string.
            include_intelligent_defaults_errors (bool): Whether to include intelligent defaults errors.
            exclude_resource_attrs (list): A list of attributes to exclude from the docstring.

        Returns:
            str: The generated docstring for the IMPORT method.
        """
        docstring = f"{title}\n"
        _shape_attr_documentation_string = ""
        if operation_input_shape_name:
            _shape_attr_documentation_string = self._get_shape_attr_documentation_string(
                self.shapes_extractor.fetch_shape_members_and_doc_strings(
                    operation_input_shape_name
                ),
                exclude_resource_attrs=exclude_resource_attrs,
            )
            if _shape_attr_documentation_string:
                docstring += f"\nParameters:\n"
                docstring += _shape_attr_documentation_string

        if include_session_region:
            if not _shape_attr_documentation_string:
                docstring += f"\nParameters:\n"
            docstring += add_indent(f"session: Boto3 session.\nregion: Region name.\n")

        if include_return_resource_docstring:
            docstring += f"\nReturns:\n" f"    The {resource_name} resource.\n"
        elif return_string:
            docstring += "\n" + return_string

        docstring += self._exception_docstring(operation_name)

        if include_intelligent_defaults_errors:
            subclasses = set(IntelligentDefaultsError.__subclasses__())
            _id_exception_docstrings = [
                f"\n    {subclass.__name__}: {subclass.__doc__}" for subclass in subclasses
            ]
            sorted_id_exception_docstrings = sorted(_id_exception_docstrings)
            docstring += "".join(sorted_id_exception_docstrings)
        docstring = add_indent(f'"""\n{docstring}\n"""\n', 4)

        return docstring

    def generate_import_method(self, resource_name: str) -> str:
        """
        Auto-generate the IMPORT method for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted Import Method template.

        """
        # Get the operation and shape for the 'import' method
        operation_name = "Import" + resource_name
        operation_metadata = self.operations[operation_name]
        operation_input_shape_name = operation_metadata["input"]["shape"]

        # Generate the arguments for the 'import' method
        import_args = self._generate_method_args(operation_input_shape_name)

        operation_input_args = self._generate_operation_input_args(
            operation_metadata, is_class_method=True
        )

        # Convert the resource name to snake case
        resource_lower = convert_to_snake_case(resource_name)

        # Convert the operation name to snake case
        operation = convert_to_snake_case(operation_name)

        get_args = self._generate_get_args(resource_name, operation_input_shape_name)

        docstring = self._generate_docstring(
            title=f"Import a {resource_name} resource",
            operation_name=operation_name,
            resource_name=resource_name,
            operation_input_shape_name=operation_input_shape_name,
            include_session_region=True,
            include_return_resource_docstring=True,
        )

        # Format the method using the IMPORT_METHOD_TEMPLATE
        formatted_method = IMPORT_METHOD_TEMPLATE.format(
            docstring=docstring,
            resource_name=resource_name,
            import_args=import_args,
            resource_lower=resource_lower,
            # TODO: change service name based on the service - runtime, sagemaker, etc.
            service_name="sagemaker",
            operation_input_args=operation_input_args,
            operation=operation,
            get_args=get_args,
        )

        # Return the formatted method
        return formatted_method

    def generate_get_name_method(self, resource_lower: str) -> str:
        """
        Autogenerate the method that would return the identifier of the object
        Args:
            resource_name: Name of Resource
        Returns:
            str: Formatted Get Name Method
        """
        return GET_NAME_METHOD_TEMPLATE.format(resource_lower=resource_lower)

    def generate_update_method(self, resource_name: str, **kwargs) -> str:
        """
        Auto-generate the UPDATE method for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted Update Method template.

        """
        # Get the operation and shape for the 'update' method
        operation_name = "Update" + resource_name
        operation_metadata = self.operations[operation_name]
        operation_input_shape_name = operation_metadata["input"]["shape"]

        required_members = self.shapes[operation_input_shape_name]["required"]

        # Exclude any required attributes that are already present as resource attributes and are also identifiers
        exclude_required_attributes = []
        for member in required_members:
            snake_member = convert_to_snake_case(member)
            if snake_member in kwargs["resource_attributes"] and any(
                id in snake_member for id in ["name", "arn", "id"]
            ):
                exclude_required_attributes.append(snake_member)

        # Generate the arguments for the 'update' method
        update_args = self._generate_method_args(
            operation_input_shape_name, exclude_required_attributes
        )

        operation_input_args = self._generate_operation_input_necessary_args(
            operation_metadata, exclude_required_attributes
        )

        # Convert the resource name to snake case
        resource_lower = convert_to_snake_case(resource_name)

        # Convert the operation name to snake case
        operation = convert_to_snake_case(operation_name)

        docstring = self._generate_docstring(
            title=f"Update a {resource_name} resource",
            operation_name=operation_name,
            resource_name=resource_name,
            operation_input_shape_name=operation_input_shape_name,
            include_session_region=False,
            include_return_resource_docstring=True,
            exclude_resource_attrs=kwargs["resource_attributes"],
        )

        # Format the method using the CREATE_METHOD_TEMPLATE
        if kwargs["needs_defaults_decorator"]:
            formatted_method = UPDATE_METHOD_TEMPLATE.format(
                docstring=docstring,
                service_name="sagemaker",
                resource_name=resource_name,
                resource_lower=resource_lower,
                update_args=update_args,
                operation_input_args=operation_input_args,
                operation=operation,
            )
        else:
            formatted_method = UPDATE_METHOD_TEMPLATE_WITHOUT_DECORATOR.format(
                docstring=docstring,
                service_name="sagemaker",
                resource_name=resource_name,
                resource_lower=resource_lower,
                update_args=update_args,
                operation_input_args=operation_input_args,
                operation=operation,
            )

        # Return the formatted method
        return formatted_method

    def generate_get_method(self, resource_name: str) -> str:
        """
        Auto-generate the GET method (describe API) for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted Get Method template.

        """
        operation_name = "Describe" + resource_name
        operation_metadata = self.operations[operation_name]
        resource_operation_input_shape_name = operation_metadata["input"]["shape"]
        resource_operation_output_shape_name = operation_metadata["output"]["shape"]

        operation_input_args = self._generate_operation_input_args(
            operation_metadata, is_class_method=True
        )

        # Generate the arguments for the 'update' method
        describe_args = self._generate_method_args(resource_operation_input_shape_name)

        resource_lower = convert_to_snake_case(resource_name)

        operation = convert_to_snake_case(operation_name)

        # generate docstring
        docstring = self._generate_docstring(
            title=f"Get a {resource_name} resource",
            operation_name=operation_name,
            resource_name=resource_name,
            operation_input_shape_name=resource_operation_input_shape_name,
            include_session_region=True,
            include_return_resource_docstring=True,
        )

        formatted_method = GET_METHOD_TEMPLATE.format(
            docstring=docstring,
            resource_name=resource_name,
            # TODO: change service name based on the service - runtime, sagemaker, etc.
            service_name="sagemaker",
            describe_args=describe_args,
            resource_lower=resource_lower,
            operation_input_args=operation_input_args,
            operation=operation,
            describe_operation_output_shape=resource_operation_output_shape_name,
        )
        return formatted_method

    def generate_refresh_method(self, resource_name: str, **kwargs) -> str:
        """Auto-Generate 'refresh' object Method [describe API] for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted refresh Method template.
        """
        operation_name = "Describe" + resource_name
        operation_metadata = self.operations[operation_name]
        resource_operation_input_shape_name = operation_metadata["input"]["shape"]
        resource_operation_output_shape_name = operation_metadata["output"]["shape"]

        # Generate the arguments for the 'refresh' method
        refresh_args = self._generate_method_args(
            resource_operation_input_shape_name, kwargs["resource_attributes"]
        )

        operation_input_args = self._generate_operation_input_necessary_args(
            operation_metadata, kwargs["resource_attributes"]
        )

        operation = convert_to_snake_case(operation_name)

        # generate docstring
        docstring = self._generate_docstring(
            title=f"Refresh a {resource_name} resource",
            operation_name=operation_name,
            resource_name=resource_name,
            include_session_region=False,
            include_return_resource_docstring=True,
        )

        formatted_method = REFRESH_METHOD_TEMPLATE.format(
            docstring=docstring,
            resource_name=resource_name,
            operation_input_args=operation_input_args,
            refresh_args=refresh_args,
            operation=operation,
            describe_operation_output_shape=resource_operation_output_shape_name,
        )
        return formatted_method

    def generate_delete_method(self, resource_name: str, **kwargs) -> str:
        """Auto-Generate 'delete' object Method [delete API] for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted delete Method template.
        """
        operation_name = "Delete" + resource_name
        operation_metadata = self.operations[operation_name]
        resource_operation_input_shape_name = operation_metadata["input"]["shape"]

        # Generate the arguments for the 'update' method
        delete_args = self._generate_method_args(
            resource_operation_input_shape_name, kwargs["resource_attributes"]
        )
        operation_input_args = self._generate_operation_input_necessary_args(
            operation_metadata, kwargs["resource_attributes"]
        )

        operation = convert_to_snake_case(operation_name)

        # generate docstring
        docstring = self._generate_docstring(
            title=f"Delete a {resource_name} resource",
            operation_name=operation_name,
            resource_name=resource_name,
            include_session_region=False,
            include_return_resource_docstring=False,
        )

        formatted_method = DELETE_METHOD_TEMPLATE.format(
            docstring=docstring,
            resource_name=resource_name,
            delete_args=delete_args,
            operation_input_args=operation_input_args,
            operation=operation,
        )
        return formatted_method

    def generate_start_method(self, resource_name: str, **kwargs) -> str:
        """Auto-Generate 'start' object Method [delete API] for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted stop Method template.
        """
        operation_name = "Start" + resource_name
        operation_metadata = self.operations[operation_name]
        operation_input_shape_name = operation_metadata["input"]["shape"]
        resource_attributes = kwargs["resource_attributes"]

        method_args = add_indent("self,\n", 4)
        method_args += (
            self._generate_method_args(operation_input_shape_name, resource_attributes) + "\n"
        )
        operation_input_args = self._generate_operation_input_args_updated(
            operation_metadata, False, resource_attributes
        )
        exclude_resource_attrs = resource_attributes
        method_args += add_indent("session: Optional[Session] = None,\n", 4)
        method_args += add_indent("region: Optional[str] = None,", 4)

        serialize_operation_input = SERIALIZE_INPUT_TEMPLATE.format(
            operation_input_args=operation_input_args
        )
        call_operation_api = CALL_OPERATION_API_TEMPLATE.format(
            operation=convert_to_snake_case(operation_name)
        )

        # generate docstring
        docstring = self._generate_docstring(
            title=f"Start a {resource_name} resource",
            operation_name=operation_name,
            resource_name=resource_name,
            include_session_region=True,
            include_return_resource_docstring=False,
            exclude_resource_attrs=exclude_resource_attrs,
        )

        initialize_client = INITIALIZE_CLIENT_TEMPLATE.format(service_name="sagemaker")

        formatted_method = GENERIC_METHOD_TEMPLATE.format(
            docstring=docstring,
            decorator="",
            method_name="start",
            method_args=method_args,
            return_type="None",
            serialize_operation_input=serialize_operation_input,
            initialize_client=initialize_client,
            call_operation_api=call_operation_api,
            deserialize_response="",
        )
        return formatted_method

    def generate_stop_method(self, resource_name: str) -> str:
        """Auto-Generate 'stop' object Method [delete API] for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted stop Method template.
        """
        operation_name = "Stop" + resource_name
        operation_metadata = self.operations[operation_name]

        operation_input_args = self._generate_operation_input_args(
            operation_metadata, is_class_method=False
        )

        operation = convert_to_snake_case(operation_name)

        # generate docstring
        docstring = self._generate_docstring(
            title=f"Stop a {resource_name} resource",
            operation_name=operation_name,
            resource_name=resource_name,
            include_session_region=False,
            include_return_resource_docstring=False,
        )

        formatted_method = STOP_METHOD_TEMPLATE.format(
            docstring=docstring,
            resource_name=resource_name,
            operation_input_args=operation_input_args,
            operation=operation,
        )
        return formatted_method

    def generate_method(self, method: Method, resource_attributes: list):
        # TODO: Use special templates for some methods with different formats like list and wait
        if method.method_name.startswith("get_all"):
            return self.generate_additional_get_all_method(method, resource_attributes)
        operation_metadata = self.operations[method.operation_name]
        operation_input_shape_name = operation_metadata["input"]["shape"]
        if method.method_type == MethodType.CLASS.value:
            decorator = "@classmethod"
            method_args = add_indent("cls,\n", 4)
            method_args += self._generate_method_args(operation_input_shape_name)
            operation_input_args = self._generate_operation_input_args_updated(
                operation_metadata, True, resource_attributes
            )
            exclude_resource_attrs = None
        elif method.method_type == MethodType.STATIC.value:
            decorator = "@staticmethod"
            method_args = self._generate_method_args(operation_input_shape_name)
            operation_input_args = self._generate_operation_input_args_updated(
                operation_metadata, True
            )
            exclude_resource_attrs = None
        else:
            decorator = ""
            method_args = add_indent("self,\n", 4)
            method_args += (
                self._generate_method_args(operation_input_shape_name, resource_attributes) + "\n"
            )
            operation_input_args = self._generate_operation_input_args_updated(
                operation_metadata, False, resource_attributes
            )
            exclude_resource_attrs = resource_attributes
        method_args += add_indent("session: Optional[Session] = None,\n", 4)
        method_args += add_indent("region: Optional[str] = None,", 4)

        initialize_client = INITIALIZE_CLIENT_TEMPLATE.format(service_name=method.service_name)
        if len(self.shapes[operation_input_shape_name]["members"]) != 0:
            # the method has input arguments
            serialize_operation_input = SERIALIZE_INPUT_TEMPLATE.format(
                operation_input_args=operation_input_args
            )
            call_operation_api = CALL_OPERATION_API_TEMPLATE.format(
                operation=convert_to_snake_case(method.operation_name)
            )
        else:
            # the method has no input arguments
            serialize_operation_input = ""
            call_operation_api = CALL_OPERATION_API_NO_ARG_TEMPLATE.format(
                operation=convert_to_snake_case(method.operation_name)
            )

        if method.return_type == "None":
            return_type = "None"
            deserialize_response = ""
            return_string = None
        elif method.return_type in BASIC_RETURN_TYPES:
            return_type = f"Optional[{method.return_type}]"
            deserialize_response = DESERIALIZE_RESPONSE_TO_BASIC_TYPE_TEMPLATE
            return_string = f"Returns:\n" f"    {method.return_type}\n"
        else:
            if method.return_type == "cls":
                return_type = f'Optional["{method.resource_name}"]'
                return_type_conversion = "cls"
                return_string = f"Returns:\n" f"    {method.resource_name}\n"
            else:
                return_type = f"Optional[{method.return_type}]"
                return_type_conversion = method.return_type
                return_string = f"Returns:\n" f"    {method.return_type}\n"
            operation_output_shape = operation_metadata["output"]["shape"]
            deserialize_response = DESERIALIZE_RESPONSE_TEMPLATE.format(
                operation_output_shape=operation_output_shape,
                return_type_conversion=return_type_conversion,
            )

        initialize_client = INITIALIZE_CLIENT_TEMPLATE.format(service_name=method.service_name)
        if len(self.shapes[operation_input_shape_name]["members"]) != 0:
            # the method has input arguments
            if method.resource_name == "Endpoint" and method.method_name == "invoke":
                serialize_operation_input = SERIALIZE_INPUT_ENDPOINT_TEMPLATE.format(
                    operation_input_args=operation_input_args
                )
                return_type_conversion = method.return_type
                operation_output_shape = operation_metadata["output"]["shape"]
                deserialize_response = DESERIALIZE_RESPONSE_ENDPOINT_TEMPLATE.format(
                    operation_output_shape=operation_output_shape,
                    return_type_conversion=return_type_conversion,
                )

            else:
                serialize_operation_input = SERIALIZE_INPUT_TEMPLATE.format(
                    operation_input_args=operation_input_args
                )
            call_operation_api = CALL_OPERATION_API_TEMPLATE.format(
                operation=convert_to_snake_case(method.operation_name)
            )
        else:
            # the method has no input arguments
            serialize_operation_input = ""
            call_operation_api = CALL_OPERATION_API_NO_ARG_TEMPLATE.format(
                operation=convert_to_snake_case(method.operation_name)
            )

        # generate docstring
        docstring = self._generate_docstring(
            title=method.docstring_title,
            operation_name=method.operation_name,
            resource_name=method.resource_name,
            operation_input_shape_name=operation_input_shape_name,
            include_session_region=True,
            return_string=return_string,
            exclude_resource_attrs=exclude_resource_attrs,
        )

        formatted_method = GENERIC_METHOD_TEMPLATE.format(
            docstring=docstring,
            decorator=decorator,
            method_name=method.method_name,
            method_args=method_args,
            return_type=return_type,
            serialize_operation_input=serialize_operation_input,
            initialize_client=initialize_client,
            call_operation_api=call_operation_api,
            deserialize_response=deserialize_response,
        )
        return formatted_method

    def generate_additional_get_all_method(self, method: Method, resource_attributes: list):
        """Auto-Generate methods that return a list of objects.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted method code.
        """
        # TODO: merge this with generate_get_all_method
        operation_metadata = self.operations[method.operation_name]
        operation_input_shape_name = operation_metadata["input"]["shape"]
        exclude_list = ["next_token", "max_results"]
        if method.method_type == MethodType.CLASS.value:
            decorator = "@classmethod"
            method_args = add_indent("cls,\n", 4)
            method_args += self._generate_method_args(operation_input_shape_name, exclude_list)
            operation_input_args = self._generate_operation_input_args_updated(
                operation_metadata, True, resource_attributes, exclude_list
            )
            exclude_resource_attrs = None
        else:
            decorator = ""
            method_args = add_indent("self,\n", 4)
            method_args += self._generate_method_args(
                operation_input_shape_name, exclude_list + resource_attributes
            )
            operation_input_args = self._generate_operation_input_args_updated(
                operation_metadata, False, resource_attributes, exclude_list
            )
            exclude_resource_attrs = resource_attributes
        method_args += add_indent("session: Optional[Session] = None,\n", 4)
        method_args += add_indent("region: Optional[str] = None,", 4)

        if method.return_type == method.resource_name:
            return_type = f'ResourceIterator["{method.resource_name}"]'
        else:
            return_type = f"ResourceIterator[{method.return_type}]"
        return_string = f"Returns:\n" f"    Iterator for listed {method.return_type}.\n"

        get_list_operation_output_shape = operation_metadata["output"]["shape"]
        list_operation_output_members = self.shapes[get_list_operation_output_shape]["members"]

        filtered_list_operation_output_members = next(
            {key: value}
            for key, value in list_operation_output_members.items()
            if key != "NextToken"
        )
        summaries_key = next(iter(filtered_list_operation_output_members))
        summaries_shape_name = filtered_list_operation_output_members[summaries_key]["shape"]
        summary_name = self.shapes[summaries_shape_name]["member"]["shape"]

        list_method = convert_to_snake_case(method.operation_name)

        # TODO: add rules for custom key mapping and list methods with no args
        resource_iterator_args_list = [
            "client=client",
            f"list_method='{list_method}'",
            f"summaries_key='{summaries_key}'",
            f"summary_name='{summary_name}'",
            f"resource_cls={method.return_type}",
            "list_method_kwargs=operation_input_args",
        ]

        resource_iterator_args = ",\n".join(resource_iterator_args_list)
        resource_iterator_args = add_indent(resource_iterator_args, 8)
        serialize_operation_input = SERIALIZE_INPUT_TEMPLATE.format(
            operation_input_args=operation_input_args
        )
        initialize_client = INITIALIZE_CLIENT_TEMPLATE.format(service_name=method.service_name)
        deserialize_response = RETURN_ITERATOR_TEMPLATE.format(
            resource_iterator_args=resource_iterator_args
        )

        # generate docstring
        docstring = self._generate_docstring(
            title=method.docstring_title,
            operation_name=method.operation_name,
            resource_name=method.resource_name,
            operation_input_shape_name=operation_input_shape_name,
            include_session_region=True,
            return_string=return_string,
            exclude_resource_attrs=exclude_resource_attrs,
        )

        return GENERIC_METHOD_TEMPLATE.format(
            docstring=docstring,
            decorator=decorator,
            method_name=method.method_name,
            method_args=method_args,
            return_type=return_type,
            serialize_operation_input=serialize_operation_input,
            initialize_client=initialize_client,
            call_operation_api="",
            deserialize_response=deserialize_response,
        )

    def _get_failure_reason_ref(self, resource_name: str) -> str:
        """Get the failure reason reference for a resource object.
        Args:
            resource_name (str): The resource name.
        Returns:
            str: The failure reason reference for resource object
        """
        describe_output = self.operations["Describe" + resource_name]["output"]["shape"]
        shape_members = self.shapes[describe_output]

        for member in shape_members["members"]:
            if "FailureReason" in member or "StatusMessage" in member:
                return f"self.{convert_to_snake_case(member)}"

        return "'(Unknown)'"

    def _get_instance_count_ref(self, resource_name: str) -> str:
        """Get the instance count reference for a resource object.
        Args:
            resource_name (str): The resource name.
        Returns:
            str: The instance count reference for resource object
        """

        if resource_name == "TrainingJob":
            return """(
                sum(instance_group.instance_count for instance_group in self.resource_config.instance_groups)
                if self.resource_config.instance_groups and not isinstance(self.resource_config.instance_groups, Unassigned)
                else self.resource_config.instance_count
            )
            """
        elif resource_name == "TransformJob":
            return "self.transform_resources.instance_count"
        elif resource_name == "ProcessingJob":
            return "self.processing_resources.cluster_config.instance_count"

        raise ValueError(f"Instance count reference not found for resource {resource_name}")

    def generate_wait_method(self, resource_name: str) -> str:
        """Auto-Generate WAIT Method for a waitable resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted Wait Method template.
        """
        resource_status_chain, resource_states = (
            self.resources_extractor.get_status_chain_and_states(resource_name)
        )

        # Get terminal states for resource
        terminal_resource_states = []
        for state in resource_states:
            # Handles when a resource has terminal states like UpdateCompleted, CreateFailed, etc.
            # Checking lower because case is not consistent accross resources (ie, COMPLETED vs Completed)
            if any(terminal_state.lower() in state.lower() for terminal_state in TERMINAL_STATES):
                terminal_resource_states.append(state)

        # Get resource status key path
        status_key_path = ""
        for member in resource_status_chain:
            status_key_path += f'.{convert_to_snake_case(member["name"])}'

        failure_reason = self._get_failure_reason_ref(resource_name)
        formatted_failed_block = FAILED_STATUS_ERROR_TEMPLATE.format(
            resource_name=resource_name, reason=failure_reason
        )
        formatted_failed_block = add_indent(formatted_failed_block, 16)

        logs_arg = ""
        logs_arg_doc = ""
        init_wait_logs = ""
        print_wait_logs = ""
        if resource_name in RESOURCE_WITH_LOGS:
            logs_arg = "logs: Optional[bool] = False,"
            logs_arg_doc = "logs: Whether to print logs while waiting.\n"

            instance_count = self._get_instance_count_ref(resource_name)
            init_wait_logs = add_indent(
                INIT_WAIT_LOGS_TEMPLATE.format(
                    get_instance_count=instance_count,
                    job_type=resource_name,
                )
            )
            print_wait_logs = add_indent(PRINT_WAIT_LOGS, 12)

        formatted_method = WAIT_METHOD_TEMPLATE.format(
            terminal_resource_states=terminal_resource_states,
            status_key_path=status_key_path,
            failed_error_block=formatted_failed_block,
            resource_name=resource_name,
            logs_arg=logs_arg,
            logs_arg_doc=logs_arg_doc,
            init_wait_logs=init_wait_logs,
            print_wait_logs=print_wait_logs,
        )
        return formatted_method

    def generate_wait_for_status_method(self, resource_name: str) -> str:
        """Auto-Generate WAIT_FOR_STATUS Method for a waitable resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted wait_for_status Method template.
        """
        resource_status_chain, resource_states = (
            self.resources_extractor.get_status_chain_and_states(resource_name)
        )

        # Get resource status key path
        status_key_path = ""
        for member in resource_status_chain:
            status_key_path += f'.{convert_to_snake_case(member["name"])}'

        formatted_failed_block = ""
        if any("failed" in state.lower() for state in resource_states):
            failure_reason = self._get_failure_reason_ref(resource_name)
            formatted_failed_block = FAILED_STATUS_ERROR_TEMPLATE.format(
                resource_name=resource_name, reason=failure_reason
            )
            formatted_failed_block = add_indent(formatted_failed_block, 12)

        formatted_method = WAIT_FOR_STATUS_METHOD_TEMPLATE.format(
            resource_states=resource_states,
            status_key_path=status_key_path,
            failed_error_block=formatted_failed_block,
            resource_name=resource_name,
        )
        return formatted_method

    def generate_wait_for_delete_method(self, resource_name: str) -> str:
        """Auto-Generate WAIT_FOR_DELETE Method for a resource with deleting status.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted wait_for_delete Method template.
        """
        resource_status_chain, resource_states = (
            self.resources_extractor.get_status_chain_and_states(resource_name)
        )

        # Get resource status key path
        status_key_path = ""
        for member in resource_status_chain:
            status_key_path += f'.{convert_to_snake_case(member["name"])}'

        formatted_failed_block = ""
        if any("delete_failed" in state.lower() for state in resource_states):
            failure_reason = self._get_failure_reason_ref(resource_name)
            formatted_failed_block = DELETE_FAILED_STATUS_CHECK.format(
                resource_name=resource_name, reason=failure_reason
            )
            formatted_failed_block = add_indent(formatted_failed_block, 16)

        if any(state.lower() == "deleted" for state in resource_states):
            deleted_status_check = add_indent(DELETED_STATUS_CHECK, 16)
        else:
            deleted_status_check = ""

        formatted_method = WAIT_FOR_DELETE_METHOD_TEMPLATE.format(
            resource_states=resource_states,
            status_key_path=status_key_path,
            delete_failed_error_block=formatted_failed_block,
            deleted_status_check=deleted_status_check,
            resource_name=resource_name,
        )
        return formatted_method

    def generate_get_all_method(self, resource_name: str) -> str:
        """Auto-Generate 'get_all' class Method [list API] for a resource.

        Args:
            resource_name (str): The resource name.

        Returns:
            str: The formatted get_all Method template.
        """
        operation_name = "List" + resource_name + "s"
        operation_metadata = self.operations[operation_name]
        operation_input_shape_name = operation_metadata["input"]["shape"]

        operation = convert_to_snake_case(operation_name)

        get_list_operation_output_shape = self.operations[operation_name]["output"]["shape"]
        list_operation_output_members = self.shapes[get_list_operation_output_shape]["members"]

        filtered_list_operation_output_members = next(
            {key: value}
            for key, value in list_operation_output_members.items()
            if key != "NextToken"
        )

        summaries_key = next(iter(filtered_list_operation_output_members))
        summaries_shape_name = filtered_list_operation_output_members[summaries_key]["shape"]

        summary_name = self.shapes[summaries_shape_name]["member"]["shape"]
        summary_members = self.shapes[summary_name]["members"].keys()

        if "Describe" + resource_name in self.operations:
            get_operation = self.operations["Describe" + resource_name]
            get_operation_input_shape = get_operation["input"]["shape"]
            get_operation_required_input = self.shapes[get_operation_input_shape].get(
                "required", []
            )
        else:
            get_operation_required_input = []

        custom_key_mapping_str = ""
        if any(member not in summary_members for member in get_operation_required_input):
            if "MonitoringJobDefinitionSummary" == summary_name:
                custom_key_mapping = {
                    "monitoring_job_definition_name": "job_definition_name",
                    "monitoring_job_definition_arn": "job_definition_arn",
                }
                custom_key_mapping_str = f"custom_key_mapping = {json.dumps(custom_key_mapping)}"
                custom_key_mapping_str = add_indent(custom_key_mapping_str, 4)
            else:
                log.warning(
                    f"Resource {resource_name} summaries do not have required members to create object instance. Resource may require custom key mapping for get_all().\n"
                    f"List {summary_name} Members: {summary_members}, Object Required Members: {get_operation_required_input}"
                )
                return ""

        resource_iterator_args_list = [
            "client=client",
            f"list_method='{operation}'",
            f"summaries_key='{summaries_key}'",
            f"summary_name='{summary_name}'",
            f"resource_cls={resource_name}",
        ]

        if custom_key_mapping_str:
            resource_iterator_args_list.append(f"custom_key_mapping=custom_key_mapping")

        exclude_list = ["next_token", "max_results"]
        get_all_args = self._generate_method_args(operation_input_shape_name, exclude_list)

        if not get_all_args.strip().strip(","):
            resource_iterator_args = ",\n".join(resource_iterator_args_list)
            resource_iterator_args = add_indent(resource_iterator_args, 8)

            formatted_method = GET_ALL_METHOD_NO_ARGS_TEMPLATE.format(
                service_name="sagemaker",
                resource=resource_name,
                operation=operation,
                custom_key_mapping=custom_key_mapping_str,
                resource_iterator_args=resource_iterator_args,
            )
            return formatted_method

        operation_input_args = self._generate_operation_input_args(
            operation_metadata, is_class_method=True, exclude_list=exclude_list
        )

        resource_iterator_args_list.append("list_method_kwargs=operation_input_args")
        resource_iterator_args = ",\n".join(resource_iterator_args_list)
        resource_iterator_args = add_indent(resource_iterator_args, 8)

        # generate docstring
        docstring = self._generate_docstring(
            title=f"Get all {resource_name} resources",
            operation_name=operation_name,
            resource_name=resource_name,
            operation_input_shape_name=operation_input_shape_name,
            include_session_region=True,
            include_return_resource_docstring=False,
            return_string=f"Returns:\n" f"    Iterator for listed {resource_name} resources.\n",
        )

        formatted_method = GET_ALL_METHOD_WITH_ARGS_TEMPLATE.format(
            docstring=docstring,
            service_name="sagemaker",
            resource=resource_name,
            get_all_args=get_all_args,
            operation_input_args=operation_input_args,
            custom_key_mapping=custom_key_mapping_str,
            resource_iterator_args=resource_iterator_args,
        )
        return formatted_method

    def generate_config_schema(self):
        """
        Generates the Config Schema that is used by json Schema to validate config jsons .
        This function creates a python file with a variable that is consumed in the scripts to further fetch configs.

        Input for generating the Schema is the service JSON that is already loaded in the class

        """
        resource_properties = {}

        for _, row in self.resources_plan.iterrows():
            resource_name = row["resource_name"]
            # Get the operation and shape for the 'get' method
            if self._is_get_in_class_methods(row["class_methods"]):
                get_operation = self.operations["Describe" + resource_name]
                get_operation_shape = get_operation["output"]["shape"]

                # Generate the class attributes based on the shape
                class_attributes = self.shapes_extractor.generate_shape_members(get_operation_shape)
                cleaned_class_attributes = self._cleanup_class_attributes_types(class_attributes)
                resource_name = row["resource_name"]

                if default_attributes := self._get_dict_with_default_configurable_attributes(
                    cleaned_class_attributes
                ):
                    resource_properties[resource_name] = {
                        TYPE: OBJECT,
                        PROPERTIES: default_attributes,
                    }

        combined_config_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            TYPE: OBJECT,
            PROPERTIES: {
                SCHEMA_VERSION: {
                    TYPE: "string",
                    "enum": ["1.0"],
                    "description": "The schema version of the document.",
                },
                SAGEMAKER: {
                    TYPE: OBJECT,
                    PROPERTIES: {
                        PYTHON_SDK: {
                            TYPE: OBJECT,
                            PROPERTIES: {
                                RESOURCES: {
                                    TYPE: OBJECT,
                                    PROPERTIES: resource_properties,
                                }
                            },
                            "required": [RESOURCES],
                        }
                    },
                    "required": [PYTHON_SDK],
                },
            },
            "required": [SAGEMAKER],
        }

        output = f"{GENERATED_CLASSES_LOCATION}/{CONFIG_SCHEMA_FILE_NAME}"
        # Open the output file
        with open(output, "w") as file:
            # Generate and write the license to the file
            file.write(
                f"SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA = {json.dumps(combined_config_schema, indent=4)}"
            )

    def _cleanup_class_attributes_types(self, class_attributes: dict) -> dict:
        """
        Helper function that creates a direct mapping of attribute to type without default parameters assigned and without Optionals
        Args:
            class_attributes: attributes of the class in raw form

        Returns:
            class attributes that have a direct mapping and can be used for processing

        """
        cleaned_class_attributes = {}
        for key, value in class_attributes.items():
            new_val = value.split("=")[0].strip()
            if new_val.startswith("Optional"):
                new_val = new_val.replace("Optional[", "")[:-1]
            cleaned_class_attributes[key] = new_val
        return cleaned_class_attributes

    def _get_dict_with_default_configurable_attributes(self, class_attributes: dict) -> dict:
        """
        Creates default attributes dict for a particular resource.
        Iterates through all class attributes and filters by attributes that have particular substrings in their name
        Args:
            class_attributes: Dict that has all the attributes of a class

        Returns:
            Dict with attributes that can be configurable

        """
        PYTHON_TYPES = ["StrPipeVar", "datetime.datetime", "bool", "int", "float"]
        default_attributes = {}
        for key, value in class_attributes.items():
            if value in PYTHON_TYPES or value.startswith("List"):
                for config_attribute_substring in CONFIGURABLE_ATTRIBUTE_SUBSTRINGS:
                    if config_attribute_substring in key:
                        if value.startswith("List"):
                            element = value.replace("List[", "")[:-1]
                            if element in PYTHON_TYPES:
                                default_attributes[key] = {
                                    TYPE: "array",
                                    "items": {
                                        TYPE: self._get_json_schema_type_from_python_type(element)
                                    },
                                }
                        else:
                            default_attributes[key] = {
                                TYPE: self._get_json_schema_type_from_python_type(value) or value
                            }
            elif value.startswith("List") or value.startswith("Dict"):
                log.debug("Script does not currently support list of objects as configurable")
                continue
            else:
                class_attributes = self.shapes_extractor.generate_shape_members(value)
                cleaned_class_attributes = self._cleanup_class_attributes_types(class_attributes)
                if nested_default_attributes := self._get_dict_with_default_configurable_attributes(
                    cleaned_class_attributes
                ):
                    default_attributes[key] = nested_default_attributes

        return default_attributes

    def _get_json_schema_type_from_python_type(self, python_type) -> str:
        """
        Helper for generating Schema
        Converts Python Types to JSON Schema compliant string
        Args:
            python_type: Type as a string

        Returns:
            JSON Schema compliant type
        """
        if python_type.startswith("List"):
            return "array"
        return PYTHON_TYPES_TO_BASIC_JSON_TYPES.get(python_type, None)

    @staticmethod
    def _is_get_in_class_methods(class_methods) -> bool:
        """
        Helper to check if class methods contain Get
        Args:
            class_methods: list of methods

        Returns:
            True if 'get' in list , else False
        """
        return "get" in class_methods

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_config_schema_for_resources():
        """
        Fetches Schema JSON for all resources from generated file
        """
        return SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA[PROPERTIES][SAGEMAKER][PROPERTIES][PYTHON_SDK][
            PROPERTIES
        ][RESOURCES][PROPERTIES]
