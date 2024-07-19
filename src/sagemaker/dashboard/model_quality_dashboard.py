import json
from sagemaker.dashboard.dashboard_widgets import DashboardWidget, DashboardWidgetProperties

class AutomaticModelQualityDashboard:   
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
            ("Weighted Precision", ["weighted_precision", "weighted_precision_best_constant_classifier"]), 
            ("Weighted Recall", ["weighted_recall", "weighted_recall_best_constant_classifier"]),
        ],
        [
            ("Weighted F0.5", ["weighted_f0_5", "weighted_f0_5_best_constant_classifier"]),
            ("Weighted F1", ["weighted_f1", "weighted_f1_best_constant_classifier"]), 
            ("Weighted F2", ["weighted_f2", "weighted_f2_best_constant_classifier"]),  
        ],        
    ]
    
    def __init__(self, endpoint_name, monitoring_schedule_name, batch_transform_input, problem_type, region_name):
        self.endpoint = endpoint_name
        self.monitoring_schedule = monitoring_schedule_name
        self.batch_transform = batch_transform_input
        self.region = region_name
        self.problem_type = problem_type
        
        self.dashboard = {
            "widgets" : self._generate_widgets(),
        }
        
    
    def _generate_widgets(self):
        list_of_widgets = []
        metrics_to_graph = None
        if (self.problem_type == "Regression"):
            metrics_to_graph = AutomaticModelQualityDashboard.REGRESSION_MODEL_QUALITY_METRICS
        elif (self.problem_type == "BinaryClassification"):
            metrics_to_graph = AutomaticModelQualityDashboard.BINARY_CLASSIFICATION_MODEL_QUALITY_METRICS
        elif (self.problem_type == "MulticlassClassification"):
            metrics_to_graph = AutomaticModelQualityDashboard.MULTICLASS_CLASSIFICATION_MODEL_QUALITY_METRICS
        else:
            raise ValueError("Parameter problem_type is invalid. Valid options are Regression, BinaryClassification, or MulticlassClassification.")
            
        for graphs_per_line in metrics_to_graph:
            for graph in graphs_per_line:
                graph_title = graph[0]
                graph_metrics = graph[1] 
                if self.batch_transform is not None:
                    graph_properties = DashboardWidgetProperties(
                        view="timeSeries",
                        stacked=False,
                        metrics=[
                            [
                                {
                                    "expression": (
                                        f"SEARCH( '{AutomaticModelQualityDashboard.MODEL_QUALITY_METRICS_BATCH_NAMESPACE} "
                                        f"{" OR ".join(graph_metrics)}"
                                        f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
                                        f"'Average')"
                                    )
                                }
                            ]
                        ],
                        region=self.region,
                        title=graph_title
                    )
                else:
                    graph_properties = DashboardWidgetProperties(
                        view="timeSeries",
                        stacked=False,
                        metrics=[
                            [
                                {
                                    "expression": (
                                        f"SEARCH( '{AutomaticModelQualityDashboard.MODEL_QUALITY_METRICS_ENDPOINT_NAMESPACE} "
                                        f"{" OR ".join(graph_metrics)}"
                                        f"Endpoint=\"{self.endpoint}\" "
                                        f"MonitoringSchedule=\"{self.monitoring_schedule}\" ', "
                                        f"'Average')"
                                    )
                                }
                            ]
                        ],
                        region=self.region,
                        title=graph_title
                    )
                list_of_widgets.append(
                    DashboardWidget(
                        height=8, width=24//len(graph_metrics), widget_type="metric", properties=graph_properties
                    )
                )
                    
        return list_of_widgets

    def to_dict(self):
        return {
            "widgets": [widget.to_dict() for widget in self.dashboard["widgets"]],
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)