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
"""A class for extracting resource information from a service JSON."""
from typing import Optional

import pandas as pd

from sagemaker.core.utils.utils import get_textual_rich_logger
from sagemaker.core.tools.constants import CLASS_METHODS, OBJECT_METHODS
from sagemaker.core.tools.data_extractor import (
    load_additional_operations_data,
    load_combined_operations_data,
    load_combined_shapes_data,
)
from sagemaker.core.tools.method import Method

log = get_textual_rich_logger(__name__)
"""
This class is used to extract the resources and its actions from the service-2.json file.
"""


class ResourcesExtractor:
    """
    A class for extracting resource information from a service JSON.

    Args:
        service_json (dict): The Botocore service.json containing the shape definitions.

    Attributes:
        service_json (dict): The service JSON containing operations and shapes.
        operations (dict): The operations defined in the service JSON.
        shapes (dict): The shapes defined in the service JSON.
        resource_actions (dict): A dictionary mapping resources to their associated actions.
        actions_under_resource (set): A set of actions that are performed on resources.
        create_resources (set): A set of resources that can be created.
        add_resources (set): A set of resources that can be added.
        start_resources (set): A set of resources that can be started.
        register_resources (set): A set of resources that can be registered.
        import_resources (set): A set of resources that can be imported.
        resources (set): A set of all resources.
        df (DataFrame): A DataFrame containing resource information.

    Methods:
        _filter_actions_for_resources(resources): Filters actions based on the given resources.
        _extract_resources_plan(): Extracts the resource plan from the service JSON.
        _get_status_chain_and_states(shape_name, status_chain): Recursively extracts the status chain and states for a given shape.
        _extract_resource_plan_as_dataframe(): Builds a DataFrame containing resource information.
        get_resource_plan(): Returns the resource plan DataFrame.
    """

    RESOURCE_TO_ADDITIONAL_METHODS = {
        "Cluster": ["DescribeClusterNode", "ListClusterNodes"],
    }

    def __init__(
        self,
        combined_shapes: Optional[dict] = None,
        combined_operations: Optional[dict] = None,
    ):
        """
        Initializes a ResourceExtractor object.

        Args:
            service_json (dict): The service JSON containing operations and shapes.
        """
        self.operations = combined_operations or load_combined_operations_data()
        self.shapes = combined_shapes or load_combined_shapes_data()
        self.additional_operations = load_additional_operations_data()
        # contains information about additional methods only now.
        # TODO: replace resource_actions with resource_methods to include all methods
        self.resource_methods = {}
        self.resource_actions = {}
        self.actions_under_resource = set()

        self._extract_resources_plan()

    def _filter_additional_operations(self):
        """
        Extracts information from additional operations defined in additional_operations.json

        Returns:
            None
        """
        for resource_name, resource_operations in self.additional_operations.items():
            self.resources.add(resource_name)
            if resource_name not in self.resource_methods:
                self.resource_methods[resource_name] = dict()
            for operation_name, operation in resource_operations.items():
                self.actions_under_resource.add(operation_name)
                method = Method(**operation)
                method.get_docstring_title(self.operations[operation_name])
                self.resource_methods[resource_name][operation["method_name"]] = method
                self.actions.remove(operation_name)

    def _filter_actions_for_resources(self, resources):
        """
        Filters actions based on the given resources.

        Args:
            resources (set): A set of resources.

        Returns:
            None
        """
        for resource in sorted(resources, key=len, reverse=True):
            filtered_actions = set(
                [
                    a
                    for a in self.actions
                    if a.endswith(resource)
                    or (a.startswith("List") and a.endswith(resource + "s"))
                    or a.startswith("Invoke" + resource)
                ]
            )
            self.actions_under_resource.update(filtered_actions)
            self.resource_actions[resource] = filtered_actions

            self.actions = self.actions - filtered_actions

    def _extract_resources_plan(self):
        """
        Extracts the resource plan from the service JSON.

        Returns:
            None
        """
        self.actions = set(self.operations.keys())
        log.info(f"Total actions - {len(self.actions)}")

        # Filter out additional operations and resources first
        self.resources = set()
        self._filter_additional_operations()

        self.create_resources = set(
            [key[len("Create") :] for key in self.actions if key.startswith("Create")]
        )

        self.add_resources = set(
            [key[len("Add") :] for key in self.actions if key.startswith("Add")]
        )

        self.start_resources = set(
            [key[len("Start") :] for key in self.actions if key.startswith("Start")]
        )

        self.register_resources = set(
            [key[len("Register") :] for key in self.actions if key.startswith("Register")]
        )

        self.import_resources = set(
            [key[len("Import") :] for key in self.actions if key.startswith("Import")]
        )

        self.resources.update(
            self.create_resources
            | self.add_resources
            | self.start_resources
            | self.register_resources
            | self.import_resources
        )

        self._filter_actions_for_resources(self.resources)

        log.info(f"Total resource - {len(self.resources)}")

        log.info(f"Supported actions - {len(self.actions_under_resource)}")

        log.info(f"Unsupported actions - {len(self.actions)}")

        self._extract_resource_plan_as_dataframe()

    def get_status_chain_and_states(self, resource_name):
        """
        Extract the status chain and states for a given resource.

        Args:
            resource_name (str): The name of the resource

        Returns:
            status_chain (list): The status chain for the resource.
            resource_states (list): The states associated with the resource.
        """
        resource_operation = self.operations["Describe" + resource_name]
        resource_operation_output_shape_name = resource_operation["output"]["shape"]
        output_members_data = self.shapes[resource_operation_output_shape_name]["members"]
        if len(output_members_data) == 1:
            single_member_name = next(iter(output_members_data))
            single_member_shape_name = output_members_data[single_member_name]["shape"]
            status_chain = []
            status_chain.append(
                {"name": single_member_name, "shape_name": single_member_shape_name}
            )
            resource_status_chain, resource_states = self._get_status_chain_and_states(
                single_member_shape_name, status_chain
            )
        else:
            resource_status_chain, resource_states = self._get_status_chain_and_states(
                resource_operation_output_shape_name
            )

        return resource_status_chain, resource_states

    def _get_status_chain_and_states(self, shape_name, status_chain: list = None):
        """
        Recursively extracts the status chain and states for a given shape.

        Args:
            shape_name (str): The name of the shape.
            status_chain (list): The current status chain.

        Returns:
            status_chain (list): The status chain for the shape.
            resource_states (list): The states associated with the shape.
        """
        if status_chain is None:
            status_chain = []

        member_data = self.shapes[shape_name]["members"]
        status_name = next((member for member in member_data if "status" in member.lower()), None)
        if status_name is None:
            return [], []

        status_shape_name = member_data[status_name]["shape"]

        status_chain.append({"name": status_name, "shape_name": status_shape_name})

        if "enum" in self.shapes[status_shape_name]:
            resource_states = self.shapes[status_shape_name]["enum"]
            return status_chain, resource_states
        else:
            status_chain, resource_states = self._get_status_chain_and_states(
                status_shape_name, status_chain
            )
            return status_chain, resource_states

    def _extract_resource_plan_as_dataframe(self):
        """
        Builds a DataFrame containing resource information.

        Returns:
            None
        """
        self.df = pd.DataFrame(
            columns=[
                "resource_name",
                "type",
                "class_methods",
                "object_methods",
                "chain_resource_name",
                "additional_methods",
                "raw_actions",
                "resource_status_chain",
                "resource_states",
            ]
        )

        for resource, actions in sorted(self.resource_actions.items()):
            class_methods = set()
            object_methods = set()
            additional_methods = set()
            chain_resource_names = set()
            resource_status_chain = set()
            resource_states = set()

            for action in actions:
                action_low = action.lower()
                resource_low = resource.lower()

                if action_low.split(resource_low)[0] == "describe":
                    class_methods.add("get")
                    object_methods.add("refresh")

                    output_shape_name = self.operations[action]["output"]["shape"]
                    output_members_data = self.shapes[output_shape_name]["members"]

                    resource_status_chain, resource_states = self.get_status_chain_and_states(
                        resource
                    )

                    if resource_low.endswith("job") or resource_low.endswith("jobv2"):
                        object_methods.add("wait")
                    elif resource_states and resource_low != "action":
                        object_methods.add("wait_for_status")

                    if "Deleting" in resource_states or "DELETING" in resource_states:
                        object_methods.add("wait_for_delete")

                    continue

                if action_low.split(resource_low)[0] == "create":
                    shape_name = self.operations[action]["input"]["shape"]
                    input = self.shapes[shape_name]
                    for member in input["members"]:
                        if member.endswith("Name") or member.endswith("Names"):
                            chain_resource_name = member[: -len("Name")]

                            if (
                                chain_resource_name != resource
                                and chain_resource_name in self.resources
                            ):
                                chain_resource_names.add(chain_resource_name)
                action_split = action_low.split(resource_low)
                if action_split[0] in CLASS_METHODS:
                    if action_low.split(resource_low)[0] == "list":
                        class_methods.add("get_all")
                    else:
                        class_methods.add(action_low.split(resource_low)[0])
                elif action_split[0] in OBJECT_METHODS:
                    object_methods.add(action_split[0])
                else:
                    additional_methods.add(action)

            if resource in self.RESOURCE_TO_ADDITIONAL_METHODS:
                additional_methods.update(self.RESOURCE_TO_ADDITIONAL_METHODS[resource])

            new_row = pd.DataFrame(
                {
                    "resource_name": [resource],
                    "type": ["resource"],
                    "class_methods": [list(sorted(class_methods))],
                    "object_methods": [list(sorted(object_methods))],
                    "chain_resource_name": [list(sorted(chain_resource_names))],
                    "additional_methods": [list(sorted(additional_methods))],
                    "raw_actions": [list(sorted(actions))],
                    "resource_status_chain": [list(resource_status_chain)],
                    "resource_states": [list(resource_states)],
                }
            )

            self.df = pd.concat([self.df, new_row], ignore_index=True)

        self.df.to_csv("resource_plan.csv", index=False)

    def get_resource_plan(self):
        """
        Returns the resource plan DataFrame.

        Returns:
            df (DataFrame): The resource plan DataFrame.
        """
        return self.df

    def get_resource_methods(self):
        """
        Returns the resource methods dict.

        Returns:
            resource_methods (dict): The resource methods dict.
        """
        return self.resource_methods
