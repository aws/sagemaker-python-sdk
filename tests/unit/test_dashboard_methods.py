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
    widget_properties = WidgetProperties(
        view="timeSeries",
        stacked=False,
        metrics=[
            [
                {
                    "expression": f'SEARCH( \'aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule %^(feature_null_|feature_non_null_).*% \', \'Average\')'
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
                    "expression": f'SEARCH( \'aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule %^(feature_null_|feature_non_null_).*% \', \'Average\')'
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
                    "expression": f'SEARCH( \'aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule %^(feature_null_|feature_non_null_).*% \', \'Average\')'
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
                        "expression": f'SEARCH( \'aws/sagemaker/Endpoints/data-metrics,Endpoint,Feature,MonitoringSchedule %^(feature_null_|feature_non_null_).*% \', \'Average\')'
                    }
                ]
            ],
            "region": "us-east-1",
            "title": "Missing Data Counts",
        },
    }
    assert widget.to_dict() == expected_dict


def test_automatic_data_quality_dashboard():
    mock_generate_variables = [
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
    mock_generate_type_counts_widget = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )
    mock_generate_null_counts_widget = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )
    mock_generate_estimated_unique_values_widget = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )
    mock_generate_completeness_widget = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )
    mock_generate_baseline_drift_widget = Widget(
        height=8, width=12, widget_type="metric", properties=WidgetProperties()
    )

    dashboard = AutomaticDataQualityDashboard("endpoint", "monitoring_schedule", None, "us-west-2")

    expected_dashboard = {
        "variables": [var.to_dict() for var in mock_generate_variables],
        "widgets": [
            widget.to_dict() for widget in [
                mock_generate_type_counts_widget,
                mock_generate_null_counts_widget,
                mock_generate_estimated_unique_values_widget,
                mock_generate_completeness_widget,
                mock_generate_baseline_drift_widget,
            ]
        ],
    }
        
    assert dashboard.to_dict() == expected_dashboard
