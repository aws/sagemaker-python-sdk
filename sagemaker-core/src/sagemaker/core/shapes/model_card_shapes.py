"""Pydantic shape definitions for SageMaker model card content."""

from typing import List, Optional, Dict, Union, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

from sagemaker.core import shapes

if TYPE_CHECKING:
    pass


class RiskRating(str, Enum):
    """Risk rating levels for a model card."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class Function(str, Enum):
    """A named function used in a model card."""

    MAXIMIZE = "Maximize"
    MINIMIZE = "Minimize"


class ContainersItem(BaseModel):
    """A container entry in an inference specification."""

    model_data_url: Optional[str] = Field(None, max_length=1024)
    image: Optional[str] = Field(None, max_length=255)
    nearest_model_name: Optional[str] = None
    model_data_source: Optional[shapes.ModelDataSource] = None
    is_checkpoint: Optional[bool] = None
    base_model: Optional[shapes.BaseModel] = None


class InferenceSpecification(BaseModel):
    """Inference specification content for a model card."""

    containers: List[ContainersItem]


class ObjectiveFunction(BaseModel):
    """Objective function details for a model card."""

    function: Optional[Function] = None
    facet: Optional[str] = Field(None, max_length=63)
    condition: Optional[str] = Field(None, max_length=63)


class TrainingMetric(BaseModel):
    """A single training metric for a model card."""

    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    value: float


class TrainingEnvironment(BaseModel):
    """Training environment details for a model card."""

    container_image: Optional[List[str]] = None


class TrainingHyperParameter(BaseModel):
    """A single training hyperparameter for a model card."""

    name: str = Field(pattern=".{1,255}")
    value: Optional[str] = Field(None, pattern=".{0,255}")


class TrainingJobDetails(BaseModel):
    """Details of a training job for a model card."""

    training_arn: Optional[str] = Field(None, max_length=1024)
    training_datasets: Optional[List[str]] = None
    training_environment: Optional[TrainingEnvironment] = None
    training_metrics: Optional[List[TrainingMetric]] = None
    user_provided_training_metrics: Optional[List[TrainingMetric]] = None
    hyper_parameters: Optional[List[TrainingHyperParameter]] = None
    user_provided_hyper_parameters: Optional[List[TrainingHyperParameter]] = None


class TrainingDetails(BaseModel):
    """Training details content for a model card."""

    objective_function: Optional[ObjectiveFunction] = None
    training_observations: Optional[str] = Field(None, max_length=1024)
    training_job_details: Optional[TrainingJobDetails] = None


class ModelOverview(BaseModel):
    """Model overview content for a model card."""

    model_description: Optional[str] = Field(None, max_length=1024)
    model_creator: Optional[str] = Field(None, max_length=1024)
    model_artifact: Optional[List[str]] = None
    algorithm_type: Optional[str] = Field(None, max_length=1024)
    problem_type: Optional[str] = None
    model_owner: Optional[str] = Field(None, max_length=1024)


class AdditionalInformation(BaseModel):
    """Additional information content for a model card."""

    ethical_considerations: Optional[str] = Field(None, max_length=2048)
    caveats_and_recommendations: Optional[str] = Field(None, max_length=2048)
    custom_details: Optional[Dict[str, str]] = None


class SimpleMetric(BaseModel):
    """A simple scalar metric for a model card."""

    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    type: Literal["number", "string", "boolean"] = None
    value: Union[float, str, bool]
    x_axis_name: Optional[str] = None
    y_axis_name: Optional[str] = None


class BarChartMetric(BaseModel):
    """A bar chart metric for a model card."""

    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    type: Literal["bar_chart"] = None
    value: List
    x_axis_name: Optional[List[str]] = None
    y_axis_name: Optional[str] = None


class LinearGraphMetric(BaseModel):
    """A linear graph metric for a model card."""

    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    type: Literal["linear_graph"] = None
    value: List
    x_axis_name: Optional[str] = None
    y_axis_name: Optional[str] = None


class MatrixMetric(BaseModel):
    """A matrix metric for a model card."""

    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    type: Literal["matrix"] = None
    value: List
    x_axis_name: Optional[List[str]] = None
    y_axis_name: Optional[List[str]] = None


class MetricGroupsItem(BaseModel):
    """A group of metrics for a model card."""

    name: str = Field(pattern=".{1,63}")
    metric_data: List[Union[SimpleMetric, LinearGraphMetric, BarChartMetric, MatrixMetric]]


class EvaluationDetailsItem(BaseModel):
    """An evaluation details entry for a model card."""

    name: str = Field(pattern=".{1,63}")
    evaluation_observation: Optional[str] = Field(None, max_length=2096)
    evaluation_job_arn: Optional[str] = Field(None, max_length=256)
    datasets: Optional[List[str]] = Field(None, max_length=10)
    metadata: Optional[Dict[str, str]] = None
    metric_groups: Optional[List[MetricGroupsItem]] = Field(default_factory=list)


class IntendedUses(BaseModel):
    """Intended uses content for a model card."""

    purpose_of_model: Optional[str] = Field(None, max_length=2048)
    intended_uses: Optional[str] = Field(None, max_length=2048)
    factors_affecting_model_efficiency: Optional[str] = Field(None, max_length=2048)
    risk_rating: Optional[RiskRating] = None
    explanations_for_risk_rating: Optional[str] = Field(None, max_length=2048)


class BusinessDetails(BaseModel):
    """Business details content for a model card."""

    business_problem: Optional[str] = Field(None, max_length=2048)
    business_stakeholders: Optional[str] = Field(None, max_length=2048)
    line_of_business: Optional[str] = Field(None, max_length=2048)


class ModelCardContent(BaseModel):
    """Top-level content of a model card."""

    model_overview: Optional[ModelOverview] = None
    intended_uses: Optional[IntendedUses] = None
    business_details: Optional[BusinessDetails] = None
    training_details: Optional[TrainingDetails] = None
    evaluation_details: Optional[List[EvaluationDetailsItem]] = Field(default_factory=list)
    additional_information: Optional[AdditionalInformation] = None
