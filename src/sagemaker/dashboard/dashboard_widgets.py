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
from __future__ import absolute_import
import json


class DashboardWidgetProperties:
    """Represents properties of a dashboard widget used for metrics in CloudWatch.

    Attributes:
        view (str): Type of visualization ('timeSeries', 'bar', 'pie', 'table').
        stacked (bool): Whether to display graph as stacked lines (applies to 'timeSeries' view).
        metrics (list): Array of metrics configurations for the widget.
        region (str): Region associated with the metrics.
        period (int): Period in seconds for data points on the graph.
        title (str): Title displayed for the graph or number (optional).
        markdown (str): Markdown content to display within the widget (optional).
    """

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
        """Initializes DashboardWidgetProperties instance.

        Args:
            view (str, optional): Type of visualization ('timeSeries', 'bar', 'pie', 'table').
            stacked (bool, optional): Whether to display the graph as stacked lines.
            metrics (list, optional): Array of metrics configurations for the widget.
            region (str, optional): Region associated with the metrics.
            period (int, optional): Period in seconds for data points on the graph.
            title (str, optional): Title displayed for the graph or number.
            markdown (str, optional): Markdown content to display within the widget.
        """
        self.view = view
        self.stacked = stacked
        self.metrics = metrics
        self.region = region
        self.period = period
        self.title = title
        self.markdown = markdown

    def to_dict(self):
        """Converts DashboardWidgetProperties instance to a dictionary representation.

        Returns:
            dict: Dictionary containing widget properties suitable for JSON serialization.
        """
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
        """Converts DashboardWidgetProperties instance to a JSON string.

        Returns:
            str: JSON string representation of the widget properties.
        """
        json.dumps(self.to_dict(), indent=4)


class DashboardWidget:
    """Represents a widget in a CloudWatch dashboard.

    Attributes:
        height (int): Height of the widget.
        width (int): Width of the widget.
        type (str): Type of the widget.
        properties (DashboardWidgetProperties): Properties specific to the widget type.
    """

    def __init__(self, height, width, widget_type, properties=None):
        """Initializes DashboardWidget instance.

        Args:
            height (int): Height of the widget.
            width (int): Width of the widget.
            widget_type (str): Type of the widget.
            properties (DashboardWidgetProperties, optional): Properties of the widget type.
        """
        self.height = height
        self.width = width
        self.type = widget_type
        self.properties = (
            properties
            if properties
            else DashboardWidgetProperties(None, False, [], None, None, None)
        )

    def to_dict(self):
        """Converts DashboardWidget instance to a dictionary representation.

        Returns:
            dict: Dictionary containing widget attributes suitable for JSON serialization.
        """
        return {
            "height": self.height,
            "width": self.width,
            "type": self.type,
            "properties": self.properties.to_dict(),
        }

    def to_json(self):
        """Converts DashboardWidget instance to a JSON string.

        Returns:
            str: JSON string representation of the widget attributes.
        """
        return json.dumps(self.to_dict(), indent=4)
