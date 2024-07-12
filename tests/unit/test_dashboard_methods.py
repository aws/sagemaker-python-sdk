import pytest
from unittest.mock import patch
from sagemaker.model_monitor.dashboards import (
    Variable,
    WidgetProperties,
    Widget,
    AutomaticDataQualityDashboard,
)


def test_variable_to_dict():
    var = Variable(
        variable_type="property",
        variable_property="Feature",
        inputType="select",
        id="Feature",
        label="Feature",
        search=AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE,
        populateFrom="Feature",
    )
    expected_dict = {
        "type": "property",
        "property": "Feature",
        "inputType": "select",
        "id": "Feature",
        "label": "Feature",
        "search": AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE,
        "populateFrom": "Feature",
    }
    assert var.to_dict() == expected_dict


def test_widget_properties_to_dict():
    widget_properties = WidgetProperties(
        view="timeSeries",
        stacked=False,
        metrics=[
            [
                {
                    "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^(feature_null_|feature_non_null_).*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
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
                    "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^(feature_null_|feature_non_null_).*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
                }
            ]
        ],
        "region": "us-east-1",
        "title": "Missing Data Counts",
    }
    assert widget_properties.to_dict() == expected_dict


def test_widget_to_dict():
    widget_properties = WidgetProperties(
        view="timeSeries",
        stacked=False,
        metrics=[
            [
                {
                    "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^(feature_null_|feature_non_null_).*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
                }
            ]
        ],
        region="us-east-1",
        title="Missing Data Counts",
    )
    widget = Widget(height=8, width=12, widget_type="metric", properties=widget_properties)
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
                        "expression": f'SEARCH( \'{AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE} %^(feature_null_|feature_non_null_).*% Endpoint="{self.endpoint}" Feature="_" MonitoringSchedule="{self.monitoring_schedule}" \', \'Average\')'
                    }
                ]
            ],
            "region": "us-east-1",
            "title": "Missing Data Counts",
        },
    }
    assert widget.to_dict() == expected_dict


@patch("sagemaker.model_monitor.dashboards.AutomaticDataQualityDashboard._generate_variables")
@patch(
    "sagemaker.model_monitor.dashboards.AutomaticDataQualityDashboard._generate_type_counts_widget"
)
@patch(
    "sagemaker.model_monitor.dashboards.AutomaticDataQualityDashboard._generate_null_counts_widget"
)
@patch(
    "sagemaker.model_monitor.dashboards.AutomaticDataQualityDashboard._generate_estimated_unique_values_widget"
)
@patch(
    "sagemaker.model_monitor.dashboards.AutomaticDataQualityDashboard._generate_completeness_widget"
)
@patch(
    "sagemaker.model_monitor.dashboards.AutomaticDataQualityDashboard._generate_baseline_drift_widget"
)
def test_automatic_data_quality_dashboard(
    mock_generate_variables,
    mock_generate_type_counts_widget,
    mock_generate_null_counts_widget,
    mock_generate_estimated_unique_values_widget,
    mock_generate_completeness_widget,
    mock_generate_baseline_drift_widget,
):
    mock_generate_variables.return_value = [
        Variable(
            variable_type="property",
            variable_property="Feature",
            inputType="select",
            id="Feature",
            label="Feature",
            search=AutomaticDataQualityDashboard.DATA_QUALITY_METRICS_ENDPOINT_NAMESPACE,
            populateFrom="Feature",
        )
    ]
    mock_generate_type_counts_widget.return_value = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )
    mock_generate_null_counts_widget.return_value = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )
    mock_generate_estimated_unique_values_widget.return_value = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )
    mock_generate_completeness_widget.return_value = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )
    mock_generate_baseline_drift_widget.return_value = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )

    dashboard = AutomaticDataQualityDashboard(
        endpoint_name="my-endpoint",
        monitoring_schedule_name="my-schedule",
        batch_transform_input=False,
        region_name="us-east-1",
    )

    expected_dashboard = {
        "variables": [var.to_dict() for var in mock_generate_variables.return_value],
        "widgets": [
            widget.to_dict()
            for widget in [
                mock_generate_type_counts_widget.return_value,
                mock_generate_null_counts_widget.return_value,
                mock_generate_estimated_unique_values_widget.return_value,
                mock_generate_completeness_widget.return_value,
                mock_generate_baseline_drift_widget.return_value,
            ]
        ],
    }
    assert dashboard.to_dict() == expected_dashboard
