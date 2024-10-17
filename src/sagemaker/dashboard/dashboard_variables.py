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
"""This module contains code containing wrapper classes for dashboard variables in CloudWatch.

These classes assist with creating dashboards in Python3 and then using boto3 CloudWatch client
to publish the generated dashboards. To be used to aid dashboard creation in ClarifyModelMonitor
and ModelMonitor.
"""
from __future__ import absolute_import
import json


class DashboardVariable:
    """Represents a dashboard variable used for dynamic configuration in CloudWatch Dashboards.

    Attributes:
        variable_type (str): Type of dashboard variable ('property' or 'pattern').
        variable_property (str): Property affected by the variable, such as metric dimension.
        inputType (str): Type of input field ('input', 'select', or 'radio') for user interaction.
        id (str): Identifier for the variable, up to 32 characters.
        label (str): Label displayed for the input field (optional, defaults based on context).
        search (str): Metric search expression to populate fields (required for 'select').
        populateFrom (str): Dimension name used to populate fields from search results.
    """

    def __init__(
        self, variable_type, variable_property, inputType, variable_id, label, search, populateFrom
    ):
        """Initializes a DashboardVariable instance.

        Args:
            variable_type (str): Type of dashboard variable ('property' or 'pattern').
            variable_property (str): Property affected by the variable, such as metric dimension.
            inputType (str): Type of input field ('input', 'select', or 'radio').
            variable_id (str): Identifier for the variable, up to 32 characters.
            label (str, optional): Label displayed for the input field (default is None).
            search (str, optional): Metric search expression to populate input options.
            populateFrom (str, optional): Dimension name used to populate field.
        """
        self.variable_type = variable_type
        self.variable_property = variable_property
        self.inputType = inputType
        self.id = variable_id
        self.label = label
        self.search = search
        self.populateFrom = populateFrom

    def to_dict(self):
        """Converts DashboardVariable instance to a dictionary representation.

        Returns:
            dict: Dictionary containing variable properties suitable for JSON serialization.
        """
        variable_properties_dict = {}
        if self.variable_type is not None:
            variable_properties_dict["type"] = self.variable_type
        if self.variable_property is not None:
            variable_properties_dict["property"] = self.variable_property
        if self.inputType is not None:
            variable_properties_dict["inputType"] = self.inputType
        if self.id is not None:
            variable_properties_dict["id"] = self.id
        if self.label is not None:
            variable_properties_dict["label"] = self.label
        if self.search is not None:
            variable_properties_dict["search"] = self.search
        if self.populateFrom is not None:
            variable_properties_dict["populateFrom"] = self.populateFrom
        return variable_properties_dict

    def to_json(self):
        """Converts DashboardVariable instance to a JSON string.

        Returns:
            str: JSON string representation of the variable properties.
        """
        json.dumps(self.to_dict(), indent=4)
