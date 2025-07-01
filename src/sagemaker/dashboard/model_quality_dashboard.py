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
"""This module the wrapper class for model quality dashboard.

To be used to aid dashboard creation in ModelMonitor.
"""

from __future__ import absolute_import
import json
from sagemaker.dashboard.dashboard_widgets import DashboardWidget, DashboardWidgetProperties


class AutomaticModelQualityDashboard:
    """Represents a dashboard for automatic model quality metrics in Amazon SageMaker.

    Methods:
        __init__(self, endpoint_name, monitoring_schedule_name,
                batch_transform_input, problem_type, region_name):
            Initializes an AutomaticModelQualityDashboard instance.

        _generate_widgets(self):
            Generates widgets based on the specified problem type and metrics.

        to_dict(self):
            Converts the dashboard instance to a dictionary representation.

        to_json(self):
            Converts the dashboard instance to a JSON string.
    """

    MODEL_QUALITY_METRICS_ENDPOINT_NAMESPACE = (
        "{aws/sagemaker/Endpoints/model-metrics,Endpoint,MonitoringSchedule}"
    )

    MODEL_QUALITY_METRICS_BATCH_NAMESPACE = (
        "{aws/sagemaker/ModelMonitoring/model-metrics,MonitoringSchedule}"
    )

    REGRESSION_MODEL_QUALITY_METRICS = [
        # The outer list represents the graphs per line in cloudwatch
        [
            # each tuple here contains the title and the metrics that are being graphed
            ("Mean Squared Error", ["mse"]),
            ("Root Mean Squared Error", ["rmse"]),
        ],
        [
            ("R-squared", ["r2"]),
            ("Mean Absolute Error", ["mae"]),
        ],
    ]

    BINARY_CLASSIFICATION_MODEL_QUALITY_METRICS = [
        [
            ("Accuracy", ["accuracy", "accuracy_best_constant_classifier"]),
            ("Precision", ["precision", "precision_best_constant_classifier"]),
            ("Recall", ["recall", "recall_best_constant_classifier"]),
        ],
        [
            ("F0.5", ["f0_5", "f0_5_best_constant_classifier"]),
            ("F1", ["f1", "f1_best_constant_classifier"]),
            ("F2", ["f2", "f2_best_constant_classifier"]),
        ],
        [
            ("True Positive Rate", ["true_positive_rate"]),
            ("True Negative Rate", ["true_negative_rate"]),
            ("False Positive Rate", ["false_positive_rate"]),
            ("False Negative Rate", ["false_negative_rate"]),
        ],
        [
            ("Area Under Precision-Recall Curve", ["au_prc"]),
            ("Area Under ROC curve", ["auc"]),
        ],
    ]

    MULTICLASS_CLASSIFICATION_MODEL_QUALITY_METRICS = [
        [
            ("Accuracy", ["accuracy", "accuracy_best_constant_classifier"]),
            (
                "Weighted Precision",
                ["weighted_precision", "weighted_precision_best_constant_classifier"],
            ),
            ("Weighted Recall", ["weighted_recall", "weighted_recall_best_constant_classifier"]),
        ],
        [
            ("Weighted F0.5", ["weighted_f0_5", "weighted_f0_5_best_constant_classifier"]),
            ("Weighted F1", ["weighted_f1", "weighted_f1_best_constant_classifier"]),
            ("Weighted F2", ["weighted_f2", "weighted_f2_best_constant_classifier"]),
        ],
    ]

    def __init__(
        self,
        endpoint_name,
        monitoring_schedule_name,
        batch_transform_input,
        problem_type,
        region_name,
    ):
        """Initializes an AutomaticModelQualityDashboard instance.

        Args:
            endpoint_name (str): Name of the SageMaker endpoint.
            monitoring_schedule_name (str): Name of the monitoring schedule.
            batch_transform_input (str): Batch transform input (can be None).
            problem_type (str): Type of problem
                                ('Regression', 'BinaryClassification', 'MulticlassClassification').
            region_name (str): AWS region name.
        """
        self.endpoint = endpoint_name
        self.monitoring_schedule = monitoring_schedule_name
        self.batch_transform = batch_transform_input
        self.region = region_name
        self.problem_type = problem_type

        self.dashboard = {
            "widgets": self._generate_widgets(),
        }

    def _generate_widgets(self):
        """Generates widgets based on the specified problem type and metrics.

        Returns:
            list: List of DashboardWidget instances representing each metric graph.
        """
        list_of_widgets = []
        metrics_to_graph = None
        if self.problem_type == "Regression":
            metrics_to_graph = self.REGRESSION_MODEL_QUALITY_METRICS
        elif self.problem_type == "BinaryClassification":
            metrics_to_graph = self.BINARY_CLASSIFICATION_MODEL_QUALITY_METRICS
        elif self.problem_type == "MulticlassClassification":
            metrics_to_graph = self.MULTICLASS_CLASSIFICATION_MODEL_QUALITY_METRICS
        else:
            raise ValueError(
                "Parameter problem_type is invalid. Valid options are "
                "Regression, BinaryClassification, or MulticlassClassification."
            )

        for graphs_per_line in metrics_to_graph:
            for graph in graphs_per_line:
                graph_title = graph[0]
                graph_metrics = ["%^" + str(metric) + "$%" for metric in graph[1]]
                metrics_string = " OR ".join(graph_metrics)
                if self.batch_transform is not None:
                    graph_properties = DashboardWidgetProperties(
                        view="timeSeries",
                        stacked=False,
                        metrics=[
                            [
                                {
                                    "expression": (
                                        f"SEARCH( '{self.MODEL_QUALITY_METRICS_BATCH_NAMESPACE} "
                                        f"{metrics_string} "
                                        f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                        "'Average')"
                                    )
                                }
                            ]
                        ],
                        region=self.region,
                        title=graph_title,
                    )
                else:
                    graph_properties = DashboardWidgetProperties(
                        view="timeSeries",
                        stacked=False,
                        metrics=[
                            [
                                {
                                    "expression": (
                                        f"SEARCH( '{self.MODEL_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                        f"{metrics_string} "
                                        f'Endpoint="{self.endpoint}" '
                                        f'MonitoringSchedule="{self.monitoring_schedule}" \', '
                                        f"'Average')"
                                    )
                                }
                            ]
                        ],
                        region=self.region,
                        title=graph_title,
                    )
                list_of_widgets.append(
                    DashboardWidget(
                        height=8,
                        width=24 // len(graphs_per_line),
                        widget_type="metric",
                        properties=graph_properties,
                    )
                )

        return list_of_widgets

    def to_dict(self):
        """Converts the AutomaticModelQualityDashboard instance to a dictionary representation.

        Returns:
            dict: Dictionary containing the dashboard widgets.
        """
        return {
            "widgets": [widget.to_dict() for widget in self.dashboard["widgets"]],
        }

    def to_json(self):
        """Converts the AutomaticModelQualityDashboard instance to a JSON string.

        Returns:
            str: JSON string representation of the dashboard widgets.
        """
        return json.dumps(self.to_dict(), indent=4)
