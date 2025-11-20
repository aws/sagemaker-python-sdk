import pytest
from pydantic import ValidationError
from sagemaker.core.shapes import (
    RiskRating,
    Function,
    LinearGraphMetric,
    SimpleMetric,
    MatrixMetric,
    BarChartMetric,
    MetricGroupsItem,
    EvaluationDetailsItem,
    ObjectiveFunction,
    IntendedUses,
    TrainingHyperParameter,
    TrainingMetric,
    TrainingEnvironment,
    TrainingJobDetails,
    TrainingDetails,
    BusinessDetails,
    ModelOverview,
    AdditionalInformation,
    ModelCardContent,
    ContainersItem,
    InferenceSpecification,
)


class TestRiskRating:
    def test_valid_values(self):
        assert RiskRating.HIGH == "High"
        assert RiskRating.MEDIUM == "Medium"
        assert RiskRating.LOW == "Low"
        assert RiskRating.UNKNOWN == "Unknown"


class TestFunction:
    def test_valid_values(self):
        assert Function.MAXIMIZE == "Maximize"
        assert Function.MINIMIZE == "Minimize"


class TestLinearGraphMetric:
    def test_required_fields(self):
        metric = LinearGraphMetric(name="test", value=[1, 2, 3])
        assert metric.name == "test"
        assert metric.value == [1, 2, 3]

    def test_optional_fields(self):
        metric = LinearGraphMetric(
            name="test", value=[1, 2, 3], notes="test notes", x_axis_name="x", y_axis_name="y"
        )
        assert metric.notes == "test notes"
        assert metric.x_axis_name == "x"
        assert metric.y_axis_name == "y"

    def test_notes_max_length(self):
        with pytest.raises(ValidationError):
            LinearGraphMetric(name="test", value=[1], notes="x" * 1025)


class TestSimpleMetric:
    def test_number_value(self):
        metric = SimpleMetric(name="test", value=42.5)
        assert metric.value == 42.5

    def test_string_value(self):
        metric = SimpleMetric(name="test", value="test_string")
        assert metric.value == "test_string"

    def test_boolean_value(self):
        metric = SimpleMetric(name="test", value=True)
        assert metric.value is True


class TestMatrixMetric:
    def test_required_fields(self):
        metric = MatrixMetric(name="test", value=[[1, 2], [3, 4]])
        assert metric.name == "test"
        assert metric.value == [[1, 2], [3, 4]]

    def test_axis_names(self):
        metric = MatrixMetric(
            name="test", value=[[1, 2]], x_axis_name=["x1", "x2"], y_axis_name=["y1", "y2"]
        )
        assert metric.x_axis_name == ["x1", "x2"]
        assert metric.y_axis_name == ["y1", "y2"]


class TestBarChartMetric:
    def test_required_fields(self):
        metric = BarChartMetric(name="test", value=[1, 2, 3])
        assert metric.name == "test"
        assert metric.value == [1, 2, 3]

    def test_axis_names(self):
        metric = BarChartMetric(
            name="test", value=[1, 2], x_axis_name=["x1", "x2"], y_axis_name="y"
        )
        assert metric.x_axis_name == ["x1", "x2"]
        assert metric.y_axis_name == "y"


class TestMetricGroupsItem:
    def test_required_fields(self):
        simple_metric = SimpleMetric(name="metric1", value=1.0)
        group = MetricGroupsItem(name="group1", metric_data=[simple_metric])
        assert group.name == "group1"
        assert len(group.metric_data) == 1


class TestEvaluationDetailsItem:
    def test_required_fields(self):
        item = EvaluationDetailsItem(name="eval1")
        assert item.name == "eval1"
        assert item.metric_groups == []

    def test_all_fields(self):
        item = EvaluationDetailsItem(
            name="eval1",
            evaluation_observation="test observation",
            evaluation_job_arn="arn:aws:sagemaker:us-east-1:123456789012:processing-job/test",
            datasets=["s3://bucket/data"],
            metadata={"key": "value"},
        )
        assert item.evaluation_observation == "test observation"
        assert item.datasets == ["s3://bucket/data"]
        assert item.metadata == {"key": "value"}

    def test_datasets_max_items(self):
        datasets = [f"dataset{i}" for i in range(11)]
        with pytest.raises(ValidationError):
            EvaluationDetailsItem(name="eval1", datasets=datasets)


class TestObjectiveFunction:
    def test_all_fields(self):
        obj_func = ObjectiveFunction(
            function=Function.MAXIMIZE, facet="accuracy", condition="validation"
        )
        assert obj_func.function == Function.MAXIMIZE
        assert obj_func.facet == "accuracy"
        assert obj_func.condition == "validation"


class TestIntendedUses:
    def test_all_fields(self):
        uses = IntendedUses(
            purpose_of_model="Classification",
            intended_uses="Image classification",
            factors_affecting_model_efficiency="Data quality",
            risk_rating=RiskRating.LOW,
            explanations_for_risk_rating="Low risk model",
        )
        assert uses.purpose_of_model == "Classification"
        assert uses.risk_rating == RiskRating.LOW

    def test_max_lengths(self):
        with pytest.raises(ValidationError):
            IntendedUses(purpose_of_model="x" * 2049)


class TestTrainingHyperParameter:
    def test_required_fields(self):
        param = TrainingHyperParameter(name="learning_rate", value="0.01")
        assert param.name == "learning_rate"
        assert param.value == "0.01"


class TestTrainingMetric:
    def test_required_fields(self):
        metric = TrainingMetric(name="accuracy", value=0.95)
        assert metric.name == "accuracy"
        assert metric.value == 0.95

    def test_with_notes(self):
        metric = TrainingMetric(name="loss", value=0.1, notes="Final loss")
        assert metric.notes == "Final loss"


class TestTrainingEnvironment:
    def test_container_image(self):
        env = TrainingEnvironment(container_image=["image1", "image2"])
        assert env.container_image == ["image1", "image2"]


class TestTrainingJobDetails:
    def test_all_fields(self):
        details = TrainingJobDetails(
            training_arn="arn:aws:sagemaker:us-east-1:123456789012:training-job/test",
            training_datasets=["s3://bucket/train"],
            training_environment=TrainingEnvironment(container_image=["image1"]),
            training_metrics=[TrainingMetric(name="accuracy", value=0.9)],
            hyper_parameters=[TrainingHyperParameter(name="lr", value="0.01")],
        )
        assert details.training_arn.startswith("arn:aws:sagemaker")
        assert len(details.training_datasets) == 1
        assert len(details.training_metrics) == 1


class TestTrainingDetails:
    def test_all_fields(self):
        details = TrainingDetails(
            objective_function=ObjectiveFunction(function=Function.MAXIMIZE),
            training_observations="Good performance",
            training_job_details=TrainingJobDetails(),
        )
        assert details.objective_function.function == Function.MAXIMIZE
        assert details.training_observations == "Good performance"


class TestBusinessDetails:
    def test_all_fields(self):
        details = BusinessDetails(
            business_problem="Customer churn prediction",
            business_stakeholders="Marketing team",
            line_of_business="Retail",
        )
        assert details.business_problem == "Customer churn prediction"
        assert details.business_stakeholders == "Marketing team"
        assert details.line_of_business == "Retail"

    def test_max_lengths(self):
        with pytest.raises(ValidationError):
            BusinessDetails(business_problem="x" * 2049)


class TestModelOverview:
    def test_all_fields(self):
        overview = ModelOverview(
            model_description="A classification model",
            model_creator="Data Science Team",
            model_artifact=["s3://bucket/model.tar.gz"],
            algorithm_type="XGBoost",
            problem_type="Classification",
            model_owner="ML Team",
        )
        assert overview.model_description == "A classification model"
        assert overview.algorithm_type == "XGBoost"
        assert overview.problem_type == "Classification"

    def test_max_lengths(self):
        with pytest.raises(ValidationError):
            ModelOverview(model_description="x" * 1025)


class TestAdditionalInformation:
    def test_all_fields(self):
        info = AdditionalInformation(
            ethical_considerations="No bias detected",
            caveats_and_recommendations="Use with caution",
            custom_details={"version": "1.0", "author": "team"},
        )
        assert info.ethical_considerations == "No bias detected"
        assert info.custom_details["version"] == "1.0"

    def test_max_lengths(self):
        with pytest.raises(ValidationError):
            AdditionalInformation(ethical_considerations="x" * 2049)


class TestModelCardContent:
    def test_all_fields(self):
        content = ModelCardContent(
            model_overview=ModelOverview(model_description="Test model"),
            intended_uses=IntendedUses(purpose_of_model="Testing"),
            business_details=BusinessDetails(business_problem="Test problem"),
            training_details=TrainingDetails(training_observations="Test obs"),
            evaluation_details=[EvaluationDetailsItem(name="eval1")],
            additional_information=AdditionalInformation(ethical_considerations="None"),
        )
        assert content.model_overview.model_description == "Test model"
        assert len(content.evaluation_details) == 1


class TestContainersItem:
    def test_required_fields(self):
        container = ContainersItem(
            image="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-model:latest"
        )
        assert container.image == "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-model:latest"

    def test_all_fields(self):
        container = ContainersItem(
            model_data_url="s3://bucket/model.tar.gz",
            image="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-model:latest",
            nearest_model_name="base-model",
        )
        assert container.model_data_url == "s3://bucket/model.tar.gz"
        assert container.nearest_model_name == "base-model"


class TestInferenceSpecification:
    def test_required_fields(self):
        container = ContainersItem(image="test-image")
        spec = InferenceSpecification(containers=[container])
        assert len(spec.containers) == 1
        assert spec.containers[0].image == "test-image"
