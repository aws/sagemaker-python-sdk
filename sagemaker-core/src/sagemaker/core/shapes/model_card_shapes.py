from typing import List, Optional, Dict, Union, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

from sagemaker.core import shapes
from sagemaker.core.shapes import ModelDataSource

if TYPE_CHECKING:
    from sagemaker.core.shapes.shapes import BaseModel as CoreBaseModel


class RiskRating(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class Function(str, Enum):
    MAXIMIZE = "Maximize"
    MINIMIZE = "Minimize"


class ContainersItem(BaseModel):
    model_data_url: Optional[str] = Field(None, max_length=1024)
    image: Optional[str] = Field(None, max_length=255)
    nearest_model_name: Optional[str] = None
    model_data_source: Optional[shapes.ModelDataSource] = None
    is_checkpoint: Optional[bool] = None
    base_model: Optional[shapes.BaseModel] = None


class InferenceSpecification(BaseModel):
    containers: List[ContainersItem]


class ObjectiveFunction(BaseModel):
    function: Optional[Function] = None
    facet: Optional[str] = Field(None, max_length=63)
    condition: Optional[str] = Field(None, max_length=63)


class TrainingMetric(BaseModel):
    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    value: float


class TrainingEnvironment(BaseModel):
    container_image: Optional[List[str]] = None


class TrainingHyperParameter(BaseModel):
    name: str = Field(pattern=".{1,255}")
    value: Optional[str] = Field(None, pattern=".{0,255}")


class TrainingJobDetails(BaseModel):
    training_arn: Optional[str] = Field(None, max_length=1024)
    training_datasets: Optional[List[str]] = None
    training_environment: Optional[TrainingEnvironment] = None
    training_metrics: Optional[List[TrainingMetric]] = None
    user_provided_training_metrics: Optional[List[TrainingMetric]] = None
    hyper_parameters: Optional[List[TrainingHyperParameter]] = None
    user_provided_hyper_parameters: Optional[List[TrainingHyperParameter]] = None


class TrainingDetails(BaseModel):
    objective_function: Optional[ObjectiveFunction] = None
    training_observations: Optional[str] = Field(None, max_length=1024)
    training_job_details: Optional[TrainingJobDetails] = None


class ModelOverview(BaseModel):
    model_description: Optional[str] = Field(None, max_length=1024)
    model_creator: Optional[str] = Field(None, max_length=1024)
    model_artifact: Optional[List[str]] = None
    algorithm_type: Optional[str] = Field(None, max_length=1024)
    problem_type: Optional[str] = None
    model_owner: Optional[str] = Field(None, max_length=1024)


class AdditionalInformation(BaseModel):
    ethical_considerations: Optional[str] = Field(None, max_length=2048)
    caveats_and_recommendations: Optional[str] = Field(None, max_length=2048)
    custom_details: Optional[Dict[str, str]] = None


class SimpleMetric(BaseModel):
    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    type: Literal["number", "string", "boolean"] = None
    value: Union[float, str, bool]
    x_axis_name: Optional[str] = None
    y_axis_name: Optional[str] = None


class BarChartMetric(BaseModel):
    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    type: Literal["bar_chart"] = None
    value: List
    x_axis_name: Optional[List[str]] = None
    y_axis_name: Optional[str] = None


class LinearGraphMetric(BaseModel):
    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    type: Literal["linear_graph"] = None
    value: List
    x_axis_name: Optional[str] = None
    y_axis_name: Optional[str] = None


class MatrixMetric(BaseModel):
    name: str = Field(pattern=".{1,255}")
    notes: Optional[str] = Field(None, max_length=1024)
    type: Literal["matrix"] = None
    value: List
    x_axis_name: Optional[List[str]] = None
    y_axis_name: Optional[List[str]] = None


class MetricGroupsItem(BaseModel):
    name: str = Field(pattern=".{1,63}")
    metric_data: List[Union[SimpleMetric, LinearGraphMetric, BarChartMetric, MatrixMetric]]


class EvaluationDetailsItem(BaseModel):
    name: str = Field(pattern=".{1,63}")
    evaluation_observation: Optional[str] = Field(None, max_length=2096)
    evaluation_job_arn: Optional[str] = Field(None, max_length=256)
    datasets: Optional[List[str]] = Field(None, max_length=10)
    metadata: Optional[Dict[str, str]] = None
    metric_groups: Optional[List[MetricGroupsItem]] = Field(default_factory=list)


class IntendedUses(BaseModel):
    purpose_of_model: Optional[str] = Field(None, max_length=2048)
    intended_uses: Optional[str] = Field(None, max_length=2048)
    factors_affecting_model_efficiency: Optional[str] = Field(None, max_length=2048)
    risk_rating: Optional[RiskRating] = None
    explanations_for_risk_rating: Optional[str] = Field(None, max_length=2048)


class BusinessDetails(BaseModel):
    business_problem: Optional[str] = Field(None, max_length=2048)
    business_stakeholders: Optional[str] = Field(None, max_length=2048)
    line_of_business: Optional[str] = Field(None, max_length=2048)


class ModelCardContent(BaseModel):
    model_overview: Optional[ModelOverview] = None
    intended_uses: Optional[IntendedUses] = None
    business_details: Optional[BusinessDetails] = None
    training_details: Optional[TrainingDetails] = None
    evaluation_details: Optional[List[EvaluationDetailsItem]] = Field(default_factory=list)
    additional_information: Optional[AdditionalInformation] = None
