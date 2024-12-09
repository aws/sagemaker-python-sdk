from __future__ import absolute_import

from sagemaker.dashboard.data_quality_dashboard import AutomaticDataQualityDashboard
from sagemaker.dashboard.dashboard_variables import DashboardVariable
from sagemaker.dashboard.dashboard_widgets import DashboardWidget, DashboardWidgetProperties


def test_variable_to_dict():
    var = DashboardVariable(
        variable_type="property",
        variable_property="Feature",
        inputType="select",
        variable_id="Feature",
        label="Feature",
        search="{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule}",
        populateFrom="Feature",
    )
    expected_dict = {
        "type": "property",
        "property": "Feature",
        "inputType": "select",
        "id": "Feature",
        "label": "Feature",
        "search": "{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule}",
        "populateFrom": "Feature",
    }
    assert var.to_dict() == expected_dict


def test_widget_properties_to_dict():
    widget_properties = DashboardWidgetProperties(
        view="timeSeries",
        stacked=False,
        metrics=[
            [
                {
                    "expression": (
                        "SEARCH("
                        " 'aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule "
                        "%^(feature_null_|feature_non_null_).*% ', "
                        "'Average')"
                    )
                }
            ]
        ],
        region="us-east-1",
        title="Missing Data Counts",
    )
    expected_dict = {
        "view": "timeSeries",
        "stacked": False,
        "metrics": [
            [
                {
                    "expression": (
                        "SEARCH( "
                        "'aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule "
                        "%^(feature_null_|feature_non_null_).*% ', "
                        "'Average')"
                    )
                }
            ]
        ],
        "region": "us-east-1",
        "title": "Missing Data Counts",
    }
    assert widget_properties.to_dict() == expected_dict


def test_widget_to_dict():
    widget_properties = DashboardWidgetProperties(
        view="timeSeries",
        stacked=False,
        metrics=[
            [
                {
                    "expression": (
                        "SEARCH( 'aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule "
                        "%^(feature_null_|feature_non_null_).*% ', 'Average')"
                    )
                }
            ]
        ],
        region="us-east-1",
        title="Missing Data Counts",
    )
    widget = DashboardWidget(height=8, width=12, widget_type="metric", properties=widget_properties)
    expected_dict = {
        "height": 8,
        "width": 12,
        "type": "metric",
        "properties": {
            "view": "timeSeries",
            "stacked": False,
            "metrics": [
                [
                    {
                        "expression": (
                            "SEARCH( 'aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule "
                            "%^(feature_null_|feature_non_null_).*% ', "
                            "'Average')"
                        )
                    }
                ]
            ],
            "region": "us-east-1",
            "title": "Missing Data Counts",
        },
    }
    assert widget.to_dict() == expected_dict


def test_automatic_data_quality_dashboard_endpoint():
    mock_generate_variables = [
        DashboardVariable(
            variable_type="property",
            variable_property="Feature",
            inputType="select",
            variable_id="Feature",
            label="Feature",
            search="{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule}"
            + ' Endpoint="endpoint"'
            + ' MonitoringSchedule="monitoring_schedule" ',
            populateFrom="Feature",
        )
    ]
    mock_generate_type_counts_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            region="us-west-2",
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule} "
                            "%^feature_fractional_counts_.*% OR "
                            "%^feature_integral_counts_.*% OR "
                            "%^feature_string_counts_.*% OR "
                            "%^feature_boolean_counts_.*% OR "
                            "%^feature_unknown_counts_.*% "
                            'Endpoint="endpoint" Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            title="Type Counts",
        ),
    )
    mock_generate_null_counts_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            region="us-west-2",
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule} "
                            "%^feature_null_.*% OR "
                            "%^feature_non_null_.*% "
                            'Endpoint="endpoint" Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            title="Missing Data Counts",
        ),
    )
    mock_generate_estimated_unique_values_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            region="us-west-2",
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule} "
                            "%^feature_estimated_unique_values_.*% "
                            'Endpoint="endpoint" Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            title="Estimated Unique Values",
        ),
    )
    mock_generate_completeness_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            region="us-west-2",
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule} "
                            "%^feature_completeness_.*% "
                            'Endpoint="endpoint" Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            title="Completeness",
        ),
    )

    mock_generate_baseline_drift_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            region="us-west-2",
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule} "
                            "%^feature_baseline_drift_.*% "
                            'Endpoint="endpoint" Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            title="Baseline Drift",
        ),
    )

    dashboard = AutomaticDataQualityDashboard("endpoint", "monitoring_schedule", None, "us-west-2")

    expected_dashboard = {
        "variables": [var.to_dict() for var in mock_generate_variables],
        "widgets": [
            widget.to_dict()
            for widget in [
                mock_generate_type_counts_widget,
                mock_generate_null_counts_widget,
                mock_generate_estimated_unique_values_widget,
                mock_generate_completeness_widget,
                mock_generate_baseline_drift_widget,
            ]
        ],
    }
    assert dashboard.to_dict() == expected_dashboard


def test_automatic_data_quality_dashboard_batch_transform():
    mock_generate_variables = [
        DashboardVariable(
            variable_type="property",
            variable_property="Feature",
            inputType="select",
            variable_id="Feature",
            label="Feature",
            search="{aws/sagemaker/ModelMonitoring/data-metrics,Feature,MonitoringSchedule}"
            + ' MonitoringSchedule="monitoring_schedule" ',
            populateFrom="Feature",
        )
    ]
    mock_generate_type_counts_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/ModelMonitoring/data-metrics,Feature,MonitoringSchedule} "
                            "%^feature_fractional_counts_.*% OR "
                            "%^feature_integral_counts_.*% OR "
                            "%^feature_string_counts_.*% OR "
                            "%^feature_boolean_counts_.*% OR "
                            "%^feature_unknown_counts_.*% "
                            'Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            region="us-west-2",
            title="Type Counts",
        ),
    )
    mock_generate_null_counts_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/ModelMonitoring/data-metrics,Feature,MonitoringSchedule} "
                            "%^feature_null_.*% OR "
                            "%^feature_non_null_.*% "
                            'Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            region="us-west-2",
            title="Missing Data Counts",
        ),
    )
    mock_generate_estimated_unique_values_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/ModelMonitoring/data-metrics,Feature,MonitoringSchedule} "
                            "%^feature_estimated_unique_values_.*% "
                            'Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            region="us-west-2",
            title="Estimated Unique Values",
        ),
    )
    mock_generate_completeness_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/ModelMonitoring/data-metrics,Feature,MonitoringSchedule} "
                            "%^feature_completeness_.*% "
                            'Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            region="us-west-2",
            title="Completeness",
        ),
    )
    mock_generate_baseline_drift_widget = DashboardWidget(
        height=8,
        width=12,
        widget_type="metric",
        properties=DashboardWidgetProperties(
            view="timeSeries",
            stacked=False,
            metrics=[
                [
                    {
                        "expression": (
                            "SEARCH( '{aws/sagemaker/ModelMonitoring/data-metrics,Feature,MonitoringSchedule} "
                            "%^feature_baseline_drift_.*% "
                            'Feature="_" MonitoringSchedule="monitoring_schedule" \', '
                            "'Average')"
                        )
                    }
                ]
            ],
            region="us-west-2",
            title="Baseline Drift",
        ),
    )

    # Pass any non None value for batch transform input to check if the dashboard correctly uses the other namespace
    dashboard = AutomaticDataQualityDashboard(None, "monitoring_schedule", True, "us-west-2")

    expected_dashboard = {
        "variables": [var.to_dict() for var in mock_generate_variables],
        "widgets": [
            widget.to_dict()
            for widget in [
                mock_generate_type_counts_widget,
                mock_generate_null_counts_widget,
                mock_generate_estimated_unique_values_widget,
                mock_generate_completeness_widget,
                mock_generate_baseline_drift_widget,
            ]
        ],
    }

    assert dashboard.to_dict() == expected_dashboard
