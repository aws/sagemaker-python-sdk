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
"""This module the wrapper class for data quality dashboard.

To be used to aid dashboard creation in ModelMonitor.
"""
from __future__ import absolute_import

import json
from sagemaker.dashboard.dashboard_variables import DashboardVariable
from sagemaker.dashboard.dashboard_widgets import DashboardWidget, DashboardWidgetProperties


class AutomaticDataQualityDashboard:
    """A wrapper class for creating a data quality dashboard to aid ModelMonitor dashboard creation.

    This class generates dashboard variables and widgets based on the endpoint and monitoring
    schedule provided.

    Attributes:
        DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE (str): Namespace for endpoint.
        DATA_QUALITY_METRICS_BATCH_NAMESPACE (str): Namespace for batch transform.

    Methods:
        __init__(self, endpoint_name, monitoring_schedule_name, batch_transform_input, region_name):
            Initializes the AutomaticDataQualityDashboard instance.

        _generate_variables(self):
            Generates variables for the dashboard based on whether batch transform is used or not.

        _generate_type_counts_widget(self):
            Generates a widget for displaying type counts.

        _generate_null_counts_widget(self):
            Generates a widget for displaying null and non-null counts.

        _generate_estimated_unique_values_widget(self):
            Generates a widget for displaying estimated unique values.

        _generate_completeness_widget(self):
            Generates a widget for displaying completeness.

        _generate_baseline_drift_widget(self):
            Generates a widget for displaying baseline drift.

        to_dict(self):
            Converts the dashboard configuration to a dictionary representation.

        to_json(self):
            Converts the dashboard configuration to a JSON formatted string.

    """

    DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE = (
        "{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule}"
    )
    DATA_QUALITY_METRICS_BATCH_NAMESPACE = (
        "{aws/sagemaker/ModelMonitoring/data-metrics,Feature,MonitoringSchedule}"
    )

    def __init__(self, endpoint_name, monitoring_schedule_name, batch_transform_input, region_name):
        """Initializes an instance of AutomaticDataQualityDashboard.

        Args:
            endpoint_name (str or EndpointInput): Name of the endpoint or EndpointInput object.
            monitoring_schedule_name (str): Name of the monitoring schedule.
            batch_transform_input (str): Name of the batch transform input.
            region_name (str): AWS region name.

        If endpoint_name is of type EndpointInput, it extracts endpoint_name from it.

        """

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
        """Generates dashboard variables based on the presence of batch transform.

        Returns:
            list: List of DashboardVariable objects.

        """
        if self.batch_transform is not None:
            return [
                DashboardVariable(
                    variable_type="property",
                    variable_property="Feature",
                    inputType="select",
                    variable_id="Feature",
                    label="Feature",
                    search=self.DATA_QUALITY_METRICS_BATCH_NAMESPACE
                    + f' MonitoringSchedule="{self.monitoring_schedule}" ',
                    populateFrom="Feature",
                )
            ]

        return [
            DashboardVariable(
                variable_type="property",
                variable_property="Feature",
                inputType="select",
                variable_id="Feature",
                label="Feature",
                search=self.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE
                + f' Endpoint="{self.endpoint}"'
                + f' MonitoringSchedule="{self.monitoring_schedule}" ',
                populateFrom="Feature",
            )
        ]

    def _generate_type_counts_widget(self):
        """Generates a widget for displaying type counts based on endpoint or batch transform.

        Returns:
            DashboardWidget: A DashboardWidget object configured for type counts.

        """
        if self.batch_transform is not None:
            type_counts_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_fractional_counts_.*% OR "
                                f"%^feature_integral_counts_.*% OR "
                                f"%^feature_string_counts_.*% OR "
                                f"%^feature_boolean_counts_.*% OR "
                                f"%^feature_unknown_counts_.*% "
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Type Counts",
            )

        else:
            type_counts_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                f"%^feature_fractional_counts_.*% OR "
                                f"%^feature_integral_counts_.*% OR "
                                f"%^feature_string_counts_.*% OR "
                                f"%^feature_boolean_counts_.*% OR "
                                f"%^feature_unknown_counts_.*% "
                                f'Endpoint="{self.endpoint}" '
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Type Counts",
            )

        return DashboardWidget(
            height=8, width=12, widget_type="metric", properties=type_counts_widget_properties
        )

    def _generate_null_counts_widget(self):
        """Generates a widget for displaying null and non-null counts.

        Returns:
            DashboardWidget: A DashboardWidget object configured for null counts.

        """
        if self.batch_transform is not None:
            null_counts_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_null_.*% OR %^feature_non_null_.*% "
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Missing Data Counts",
            )

        else:
            null_counts_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                f"%^feature_null_.*% OR %^feature_non_null_.*% "
                                f'Endpoint="{self.endpoint}" '
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Missing Data Counts",
            )
        return DashboardWidget(
            height=8, width=12, widget_type="metric", properties=null_counts_widget_properties
        )

    def _generate_estimated_unique_values_widget(self):
        """Generates a widget for displaying estimated unique values.

        Returns:
            DashboardWidget: A DashboardWidget object configured for estimated unique values.

        """
        if self.batch_transform is not None:
            estimated_unique_vals_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_estimated_unique_values_.*% "
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Estimated Unique Values",
            )

        else:
            estimated_unique_vals_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                f"%^feature_estimated_unique_values_.*% "
                                f'Endpoint="{self.endpoint}" '
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Estimated Unique Values",
            )

        return DashboardWidget(
            height=8,
            width=12,
            widget_type="metric",
            properties=estimated_unique_vals_widget_properties,
        )

    def _generate_completeness_widget(self):
        """Generates a widget for displaying completeness based on endpoint or batch transform.

        Returns:
            DashboardWidget: A DashboardWidget object configured for completeness.

        """
        if self.batch_transform is not None:
            completeness_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_completeness_.*% "
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Completeness",
            )

        else:
            completeness_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                f"%^feature_completeness_.*% "
                                f'Endpoint="{self.endpoint}" '
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Completeness",
            )

        return DashboardWidget(
            height=8, width=12, widget_type="metric", properties=completeness_widget_properties
        )

    def _generate_baseline_drift_widget(self):
        """Generates a widget for displaying baseline drift based on endpoint or batch transform.

        Returns:
            DashboardWidget: A DashboardWidget object configured for baseline drift.

        """
        if self.batch_transform is not None:
            baseline_drift_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_baseline_drift_.*% "
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Baseline Drift",
            )

        else:
            baseline_drift_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{self.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                f"%^feature_baseline_drift_.*% "
                                f'Endpoint="{self.endpoint}" '
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Baseline Drift",
            )
        return DashboardWidget(
            height=8, width=12, widget_type="metric", properties=baseline_drift_widget_properties
        )

    def to_dict(self):
        """Converts the AutomaticDataQualityDashboard configuration to a dictionary representation.

        Returns:
            dict: A dictionary containing variables and widgets configurations.

        """
        return {
            "variables": [var.to_dict() for var in self.dashboard["variables"]],
            "widgets": [widget.to_dict() for widget in self.dashboard["widgets"]],
        }

    def to_json(self):
        """Converts the AutomaticDataQualityDashboard configuration to a JSON formatted string.

        Returns:
            str: A JSON formatted string representation of the dashboard configuration.

        """
        return json.dumps(self.to_dict(), indent=4)
