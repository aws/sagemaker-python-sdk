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
"""This module contains code containing wrapper classes for dashboard widgets in CloudWatch.

These classes assist with creating dashboards in Python3 and then using boto3 CloudWatch client
to publish the generated dashboards. To be used to aid dashboard creation in ClarifyModelMonitor
and ModelMonitor. 
"""

import json 

class DashboardWidgetProperties:
    def __init__(
        self,
        view=None,
        stacked=None,
        metrics=None,
        region=None,
        period=None,
        title=None,
        markdown=None,
    ):
        self.view = view
        self.stacked = stacked
        self.metrics = metrics
        self.region = region
        self.period = period
        self.title = title
        self.markdown = markdown

    def to_dict(self):
        widget_properties_dict = {}
        if self.view is not None:
            widget_properties_dict["view"] = self.view
        if self.period is not None:
            widget_properties_dict["period"] = self.period
        if self.markdown is not None:
            widget_properties_dict["markdown"] = self.markdown
        if self.stacked is not None:
            widget_properties_dict["stacked"] = self.stacked
        if self.region is not None:
            widget_properties_dict["region"] = self.region
        if self.metrics is not None:
            widget_properties_dict["metrics"] = self.metrics
        if self.title is not None:
            widget_properties_dict["title"] = self.title
        return widget_properties_dict

    def to_json(self):
        json.dumps(self.to_dict(), indent=4)


class DashboardWidget:
    def __init__(self, height, width, widget_type, properties=None):
        self.height = height
        self.width = width
        self.type = widget_type
        self.properties = (
            properties
            if properties
            else DashboardWidgetProperties(None, False, [], None, None, None)
        )

    def to_dict(self):
        return {
            "height": self.height,
            "width": self.width,
            "type": self.type,
            "properties": self.properties.to_dict(),
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
