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
"""This module the wrapper class for data quality dashboard. To be used to aid dashboard 
creation in ModelMonitor. 
"""

import json 
from sagemaker.dashboard.dashboard_variables import DashboardVariable
from sagemaker.dashboard.dashboard_widgets import DashboardWidget, DashboardWidgetProperties

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
        if self.batch_transform is not None:
            return [
                DashboardVariable(
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
            DashboardVariable(
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
        if self.batch_transform is not None:
            type_counts_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_fractional_counts_.*% OR "
                                f"%^feature_integral_counts_.*% OR "
                                f"%^feature_string_counts_.*% OR "
                                f"%^feature_boolean_counts_.*% OR "
                                f"%^feature_unknown_counts_.*% "
                                f"Feature=\"_\" "
                                f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Type Counts"
            )

        else:
            type_counts_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                f"%^feature_fractional_counts_.*% OR "
                                f"%^feature_integral_counts_.*% OR "
                                f"%^feature_string_counts_.*% OR "
                                f"%^feature_boolean_counts_.*% OR "
                                f"%^feature_unknown_counts_.*% "
                                f"Endpoint=\"{self.endpoint}\" "
                                f"Feature=\"_\" "
                                f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Type Counts"
            )

        return DashboardWidget(
            height=8, width=12, widget_type="metric", properties=type_counts_widget_properties
        )

    def _generate_null_counts_widget(self):
        if self.batch_transform is not None:
            null_counts_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_null_.*% OR %^feature_non_null_.*% "
                                f"Feature=\"_\" "
                                f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Missing Data Counts"
            )

        else:
            null_counts_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} '
                                f'%^feature_null_.*% OR %^feature_non_null_.*% '
                                f'Endpoint="{self.endpoint}" '
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f'\'Average\')'
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
        if self.batch_transform is not None:
            estimated_unique_vals_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_estimated_unique_values_.*% "
                                f"Feature=\"_\" "
                                f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Estimated Unique Values"
            )

        else:
            estimated_unique_vals_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                f"%^feature_estimated_unique_values_.*% "
                                f"Endpoint=\"{self.endpoint}\" "
                                f"Feature=\"_\" "
                                f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Estimated Unique Values"
            )

        return DashboardWidget(
            height=8,
            width=12,
            widget_type="metric",
            properties=estimated_unique_vals_widget_properties,
        )

    def _generate_completeness_widget(self):
        if self.batch_transform is not None:
            completeness_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_completeness_.*% "
                                f"Feature=\"_\" "
                                f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
                                f"'Average')"
                            )
                        }
                    ]
                ],
                region=self.region,
                title="Completeness"
            )

        else:
            completeness_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                f"%^feature_completeness_.*% "
                                f"Endpoint=\"{self.endpoint}\" "
                                f"Feature=\"_\" "
                                f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
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
        if self.batch_transform is not None:
            baseline_drift_widget_properties = DashboardWidgetProperties(
                view="timeSeries",
                stacked=False,
                metrics=[
                    [
                        {
                            "expression": (
                                f"SEARCH( '{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_BATCH_NAMESPACE} "
                                f"%^feature_baseline_drift_.*% "
                                f"Feature=\"_\" "
                                f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
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
                                f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} '
                                f'%^feature_baseline_drift_.*% '
                                f'Endpoint="{self.endpoint}" '
                                f'Feature="_" '
                                f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                f'\'Average\')'
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
        return {
            "variables": [var.to_dict() for var in self.dashboard["variables"]],
            "widgets": [widget.to_dict() for widget in self.dashboard["widgets"]],
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
