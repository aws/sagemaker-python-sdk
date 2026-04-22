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
"""A class for generating class structure from Service Model JSON.

To run the script be sure to set the PYTHONPATH
export PYTHONPATH=<sagemaker-code-gen repo directory>:$PYTHONPATH
"""
import os
from functools import lru_cache

from sagemaker.core.utils.code_injection.codec import pascal_to_snake
from sagemaker.core.tools.constants import (
    LICENCES_STRING,
    GENERATED_CLASSES_LOCATION,
    SHAPES_CODEGEN_FILE_NAME,
)
from sagemaker.core.tools.shapes_extractor import ShapesExtractor
from sagemaker.core.utils.utils import (
    add_indent,
    convert_to_snake_case,
    remove_html_tags,
    escape_special_rst_characters,
)
from sagemaker.core.tools.templates import SHAPE_CLASS_TEMPLATE, SHAPE_BASE_CLASS_TEMPLATE
from sagemaker.core.tools.data_extractor import (
    load_combined_shapes_data,
    load_combined_operations_data,
)
from .resources_extractor import ResourcesExtractor


class ShapesCodeGen:
    """
    Generates shape classes based on an input Botocore service.json.

    Args:
        service_json (dict): The Botocore service.json containing the shape definitions.

    Attributes:
        service_json (dict): The Botocore service.json containing the shape definitions.
        shapes_extractor (ShapesExtractor): An instance of the ShapesExtractor class.
        shape_dag (dict): Shape DAG generated from service.json

    Methods:
        build_graph(): Builds a directed acyclic graph (DAG) representing the dependencies between shapes.
        topological_sort(): Performs a topological sort on the DAG to determine the order in which shapes should be generated.
        generate_data_class_for_shape(shape): Generates a data class for a given shape.
        _generate_doc_string_for_shape(shape): Generates the docstring for a given shape.
        generate_imports(): Generates the import statements for the generated shape classes.
        generate_base_class(): Generates the base class for the shape classes.
        _filter_input_output_shapes(shape): Filters out shapes that are used as input or output for operations.
        generate_shapes(output_folder): Generates the shape classes and writes them to the specified output folder.
    """

    def __init__(self):
        self.combined_shapes = load_combined_shapes_data()
        self.combined_operations = load_combined_operations_data()
        self.shapes_extractor = ShapesExtractor()
        self.shape_dag = self.shapes_extractor.get_shapes_dag()
        self.resources_extractor = ResourcesExtractor()
        self.resources_plan = self.resources_extractor.get_resource_plan()
        self.resource_methods = self.resources_extractor.get_resource_methods()

    def build_graph(self):
        """
        Builds a directed acyclic graph (DAG) representing the dependencies between shapes.

        Steps:
        1. Loop over the Service Json shapes.
            1.1. If dependency(members) found, add association of node -> dependency.
                1.1.1. Sometimes members are not shape themselves, but have associated links to actual shapes.
                    In that case add link to node -> dependency (actual)
                        CreateExperimentRequest -> [ExperimentEntityName, ExperimentDescription, TagList]
            1.2. else leaf node found (no dependent members), add association of node -> None.

        :return: A dict which defines the structure of the DAG in the format:
            {key : [dependencies]}
            Example input:
                {'CreateExperimentRequest': ['ExperimentEntityName', 'ExperimentEntityName',
                    'ExperimentDescription', 'TagList'],
                'CreateExperimentResponse': ['ExperimentArn'],
                'DeleteExperimentRequest': ['ExperimentEntityName'],
                'DeleteExperimentResponse': ['ExperimentArn']}
        """
        graph = {}

        for node, attributes in self.combined_shapes.items():
            if "members" in attributes:
                for member, member_attributes in attributes["members"].items():
                    # add shapes and not shape attribute
                    # i.e. ExperimentEntityName taken over ExperimentName
                    if member_attributes["shape"] in self.combined_shapes.keys():
                        node_deps = graph.get(node, [])
                        # evaluate the member shape and then append to node deps
                        member_shape = self.combined_shapes[member_attributes["shape"]]
                        if member_shape["type"] == "list":
                            node_deps.append(member_shape["member"]["shape"])
                        elif member_shape["type"] == "map":
                            node_deps.append(member_shape["key"]["shape"])
                            node_deps.append(member_shape["value"]["shape"])
                        else:
                            node_deps.append(member_attributes["shape"])
                        graph[node] = node_deps
            else:
                graph[node] = None
        return graph

    def topological_sort(self):
        """
        Performs a topological sort on the DAG to determine the order in which shapes should be generated.

        :return: A list of shape names in the order of topological sort.
        """
        graph = self.build_graph()
        visited = set()
        stack = []

        def dfs(node):
            visited.add(node)
            # unless leaf node is reached do dfs
            if graph.get(node) is not None:
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return stack

    def generate_data_class_for_shape(self, shape):
        """
        Generates a data class for a given shape.

        :param shape: The name of the shape.
        :return: The generated data class as a string.
        """
        class_name = shape
        init_data = self.shapes_extractor.generate_data_shape_string_body(
            shape, self.resources_plan
        )

        try:
            data_class_members = add_indent(init_data, 4)
        except Exception:
            print("DEBUG HELP\n", init_data)
            raise
        return SHAPE_CLASS_TEMPLATE.format(
            class_name=class_name + "(Base)",
            data_class_members=data_class_members,
            docstring=self._generate_doc_string_for_shape(shape),
            class_name_snake=pascal_to_snake(class_name),
        )

    def _generate_doc_string_for_shape(self, shape):
        """
        Generates the docstring for a given shape.

        :param shape: The name of the shape.
        :return: The generated docstring as a string.
        """
        shape_dict = self.combined_shapes[shape]

        docstring = f"{shape}"
        if "documentation" in shape_dict:
            docstring += f"\n  {shape_dict['documentation']}"

        docstring += "\n\nAttributes"
        docstring += "\n----------------------"

        if "members" in shape_dict:
            for member, member_attributes in shape_dict["members"].items():
                docstring += f"\n{convert_to_snake_case(member)}"
                if "documentation" in member_attributes:
                    docstring += f": {member_attributes['documentation']}"

        docstring = remove_html_tags(docstring)
        return escape_special_rst_characters(docstring)

    def generate_license(self):
        """
        Generates the license string.

        Returns:
            str: The license string.
        """
        return LICENCES_STRING

    def generate_imports(self):
        """
        Generates the import statements for the generated shape classes.

        :return: The generated import statements as a string.
        """
        imports = "import datetime\n"
        imports += "import warnings\n"
        imports += "\n"
        imports += "from pydantic import BaseModel, ConfigDict\n"
        imports += "from typing import List, Dict, Optional, Any, Union\n"
        imports += "from sagemaker.core.utils.utils import Unassigned\n"
        imports += "from sagemaker.core.helper.pipeline_variable import StrPipeVar\n"
        imports += "\n"
        imports += "# Suppress Pydantic warnings about field names shadowing parent attributes\n"
        imports += "warnings.filterwarnings('ignore', message='.*shadows an attribute.*')\n"
        imports += "\n"
        return imports

    def generate_base_class(self):
        """
        Generates the base class for the shape classes.

        :return: The generated base class as a string.
        """
        # more customizations would be added later
        return SHAPE_BASE_CLASS_TEMPLATE.format(
            class_name="Base(BaseModel)",
        )

    def _filter_input_output_shapes(self, shape):
        """
        Filters out shapes that are used as input or output for operations.

        :param shape: The name of the shape.
        :return: True if the shape should be generated, False otherwise.
        """
        operation_input_output_shapes = set()
        for operation, attrs in self.combined_operations.items():
            if attrs.get("input"):
                operation_input_output_shapes.add(attrs["input"]["shape"])
            if attrs.get("output"):
                operation_input_output_shapes.add(attrs["output"]["shape"])

        required_output_shapes = set()
        for resource_name in self.resource_methods:
            for method in self.resource_methods[resource_name].values():
                required_output_shapes.add(method.return_type)

        if shape in operation_input_output_shapes and shape not in required_output_shapes:
            # Before filtering out, check if this shape is transitively referenced
            # by any operation input/output shape that is itself used as a resource
            # class attribute. For example, CreateEndpointConfigInput is an operation
            # input shape but is also a member type inside
            # CreateEndpointConfigInputInternal, which feeds the
            # EndpointConfigInternal resource class.
            if shape in self._get_shapes_required_by_resources():
                return True
            return False
        return True

    @lru_cache(maxsize=1)
    def _get_shapes_required_by_resources(self):
        """Collect all shapes transitively referenced by resource class attributes.

        Resource classes derive their attributes from operation input/output shapes.
        Some of those attributes reference shapes that are also operation input/output
        shapes (and would normally be filtered out). This method finds those shapes
        so they can be kept.
        """
        required = set()
        resource_plan = self.resources_extractor.get_resource_plan()

        for _, row in resource_plan.iterrows():
            resource_name = row["resource_name"]
            class_methods = row["class_methods"]

            # Determine which shapes feed this resource's class attributes
            attr_shapes = []
            if "get" in class_methods:
                op = self.combined_operations.get("Describe" + resource_name)
                if op:
                    attr_shapes.append(op["output"]["shape"])
            elif "create" in class_methods:
                op = self.combined_operations.get("Create" + resource_name)
                if op:
                    attr_shapes.append(op["input"]["shape"])
                    attr_shapes.append(op["output"]["shape"])

            # Walk one level of members to find referenced structure shapes
            for attr_shape in attr_shapes:
                shape_def = self.combined_shapes.get(attr_shape, {})
                for member_attrs in shape_def.get("members", {}).values():
                    member_shape_name = member_attrs.get("shape")
                    if member_shape_name and member_shape_name in self.combined_shapes:
                        member_shape_def = self.combined_shapes[member_shape_name]
                        if member_shape_def.get("type") == "structure":
                            required.add(member_shape_name)

        return required

    def generate_shapes(
        self,
        output_folder=GENERATED_CLASSES_LOCATION + "/shapes",
        file_name=SHAPES_CODEGEN_FILE_NAME,
    ) -> None:
        """
        Generates the shape classes and writes them to the specified output folder.

        :param output_folder: The path to the output folder.
        """
        # Check if the output folder exists, if not, create it
        os.makedirs(output_folder, exist_ok=True)

        # Create the full path for the output file
        output_file = os.path.join(output_folder, file_name)

        # Open the output file
        with open(output_file, "w") as file:
            # Generate and write the license to the file
            license = self.generate_license()
            file.write(license)

            # Generate and write the imports to the file
            imports = self.generate_imports()
            file.write(imports)

            # Generate and write Base Class
            base_class = self.generate_base_class()
            file.write(base_class)
            file.write("\n\n")

            # Iterate through shapes in topological order and generate classes
            topological_order = self.topological_sort()
            for shape in topological_order:

                # Extract the necessary data for the shape
                if self._filter_input_output_shapes(shape):
                    shape_dict = self.combined_shapes[shape]
                    shape_type = shape_dict["type"]
                    if shape_type == "structure":

                        # Generate and write data class for shape
                        shape_class = self.generate_data_class_for_shape(shape)
                        file.write(shape_class)
