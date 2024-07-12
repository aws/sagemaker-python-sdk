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
"""This module contains code containing wrapper classes for dashboard structures in CloudWatch.

These classes assist with creating dashboards in Python3 and then using boto3 CloudWatch client 
to publish the generated dashboards. To be used to aid dashboard creation in the create_monitoring_schedule 
and update_monitoring_schedule methods in model_monitoring.py 
"""

import json


class Variable:
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


class WidgetProperties:
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


class Widget:
    def __init__(self, height, width, widget_type, properties=None):
        self.height = height
        self.width = width
        self.type = widget_type
        self.properties = (
            properties if properties else WidgetProperties(None, False, [], None, None, None)
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


class AutomaticDataQualityDashboard:
    DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE = (
        "{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule}"
    )
    DATA_QUALITY_METRICS_BATCH_NAMESPACE = (
        "{aws/sagemaker/ModelMonitoring/data-metrics,Feature,MonitoringSchedule}"
    )

    def __init__(self, endpoint_name, monitoring_schedule_name, batch_transform_input, region_name):
        self.endpoint = endpoint_name
        self.monitoring_schedule = monitoring_schedule_name
        self.batch_transform = batch_transform_input
        self.region = region_name

        variables = self._generate_variables()
        type_counts_widget = self._generate_type_counts_widget()
        null_counts_widget = self._generate_null_counts_widget()
        estimated_unique_values_widget = self._generate_estimated_unique_values_widget()
        completeness_widget = self._generate_completeness_widget()
        baseline_drift_widget = self._generate_baseline_drift_widget()

        self.dashboard = {
            "variables": variables,
            "widgets": [
                type_counts_widget,
                null_counts_widget,
                estimated_unique_values_widget,
                completeness_widget,
                baseline_drift_widget,
            ],
        }

    def _generate_variables(self):
        if self.batch_transform:
            return [
                Variable(
                    variable_type="property",
                    variable_property="Feature",
                    inputType="select",
                    variable_id="Feature",
                    label="Feature",
                    search=AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE,
                    populateFrom="Feature",
                )
            ]

        return [
            Variable(
                variable_type="property",
                variable_property="Feature",
                inputType="select",
                variable_id="Feature",
                label="Feature",
                search=AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE,
                populateFrom="Feature",
            )
        ]

    def _generate_type_counts_widget(self):
        if self.batch_transform:
            type_counts_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} %^(feature_fractional_counts_|feature_string_counts_|feature_integral_counts_|feature_boolean_counts_|feature_unknown_counts_).*% Feature=\"_\" MonitoringSchedule=\"{self.monitoring_schedule}\" ', 'Average')"
                        }
                    ]
                ],
                region=self.region,
                title="Type Counts",
            )
        else:
            type_counts_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^(feature_fractional_counts_|feature_string_counts_|feature_integral_counts_|feature_boolean_counts_|feature_unknown_counts_).*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
                        }
                    ]
                ],
                region=self.region,
                title="Type Counts",
            )
        return Widget(
            height=8, width=12, widget_type="metric", properties=type_counts_widget_properties
        )

    def _generate_null_counts_widget(self):
        if self.batch_transform:
            null_counts_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} %^(feature_null_|feature_non_null_).*% Feature=\"_\" MonitoringSchedule=\"{self.monitoring_schedule}\" ', 'Average')"
                        }
                    ]
                ],
                region=self.region,
                title="Missing Data Counts",
            )
        else:
            null_counts_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^(feature_null_|feature_non_null_).*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
                        }
                    ]
                ],
                region=self.region,
                title="Missing Data Counts",
            )
        return Widget(
            height=8, width=12, widget_type="metric", properties=null_counts_widget_properties
        )

    def _generate_estimated_unique_values_widget(self):
        if self.batch_transform:
            estimated_unique_vals_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} %^feature_estimated_unique_values_.*% Feature=\"_\" MonitoringSchedule=\"{self.monitoring_schedule}\" ', 'Average')"
                        }
                    ]
                ],
                region=self.region,
                title="Estimated Unique Values",
            )
        else:
            estimated_unique_vals_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^feature_estimated_unique_values_.*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
                        }
                    ]
                ],
                region=self.region,
                title="Estimated Unique Values",
            )

        return Widget(
            height=8,
            width=12,
            widget_type="metric",
            properties=estimated_unique_vals_widget_properties,
        )

    def _generate_completeness_widget(self):
        if self.batch_transform:
            completeness_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} %^feature_completeness_.*% Feature=\"_\" MonitoringSchedule=\"{self.monitoring_schedule}\" ', 'Average')"
                        }
                    ]
                ],
                region=self.region,
                title="Completeness",
            )
        else:
            completeness_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^feature_completeness_.*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
                        }
                    ]
                ],
                region=self.region,
                title="Completeness",
            )
        return Widget(
            height=8, width=12, widget_type="metric", properties=completeness_widget_properties
        )

    def _generate_baseline_drift_widget(self):
        if self.batch_transform:
            baseline_drift_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} %^feature_baseline_drift_.*% Feature=\"_\" MonitoringSchedule=\"{self.monitoring_schedule}\" ', 'Average')"
                        }
                    ]
                ],
                region=self.region,
                title="Baseline Drift",
            )
        else:
            baseline_drift_widget_properties = WidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^feature_baseline_drift_.*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
                        }
                    ]
                ],
                region=self.region,
                title="Baseline Drift",
            )
        return Widget(
            height=8, width=12, widget_type="metric", properties=baseline_drift_widget_properties
        )

    def to_dict(self):
        return {
            "variables": [var.to_dict() for var in self.dashboard["variables"]],
            "widgets": [widget.to_dict() for widget in self.dashboard["widgets"]],
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
