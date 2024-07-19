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

import json

class DashboardVariable:
    def __init__(
        self, variable_type, variable_property, inputType, variable_id, label, search, populateFrom
    ):
        self.variable_type = variable_type
        self.variable_property = variable_property
        self.inputType = inputType
        self.id = variable_id
        self.label = label
        self.search = search
        self.populateFrom = populateFrom

    def to_dict(self):
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
        json.dumps(self.to_dict(), indent=4)

