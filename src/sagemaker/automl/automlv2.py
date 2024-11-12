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
"""A class for SageMaker AutoML V2 Jobs."""

from __future__ import absolute_import, annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from sagemaker import Model, PipelineModel, s3
from sagemaker.automl.automl import AutoML
from sagemaker.automl.candidate_estimator import CandidateEstimator
from sagemaker.config import (
    AUTO_ML_INTER_CONTAINER_ENCRYPTION_PATH,
    AUTO_ML_KMS_KEY_ID_PATH,
    AUTO_ML_ROLE_ARN_PATH,
    AUTO_ML_VOLUME_KMS_KEY_ID_PATH,
    AUTO_ML_VPC_CONFIG_PATH,
)
from sagemaker.job import _Job
from sagemaker.session import Session
from sagemaker.utils import Tags, format_tags, name_from_base, resolve_value_from_config

logger = logging.getLogger("sagemaker")


@dataclass
class AutoMLTabularConfig(object):
    """Configuration of a tabular problem.

    Args:
        target_attribute_name (str): The name of the column in the tabular dataset
            that contains the values to be predicted.
        algorithms_config (list(str)): The selection of algorithms run on a dataset to train
            the model candidates of an Autopilot job.
        feature_specification_s3_uri (str): A URL to the Amazon S3 data source containing
            selected features and specified data types from the input data source of an AutoML job.
        generate_candidate_definitions_only (bool): Whether to generates
            possible candidates without training the models.
        mode (str): The method that AutoML job uses to train the model.
            Valid values: AUTO or ENSEMBLING or HYPERPARAMETER_TUNING.
        problem_type (str): Defines the type of supervised learning
            available for the candidates. Available problem types are:
            `BinaryClassification`, `MulticlassClassification` and `Regression`.
        sample_weight_attribute_name (str): The name of dataset column representing
            sample weights.
        max_candidates (int): The maximum number of training jobs allowed to run.
        max_runtime_per_training_job_in_seconds (int): The maximum time, in seconds,
            that each training job executed inside hyperparameter tuning
            is allowed to run as part of a hyperparameter tuning job.
        max_total_job_runtime_in_seconds (int): The total wait time of an AutoML job.
    """

    target_attribute_name: str
    algorithms_config: Optional[List[str]] = None
    feature_specification_s3_uri: Optional[str] = None
    generate_candidate_definitions_only: Optional[bool] = None
    mode: Optional[str] = None
    problem_type: Optional[str] = None
    sample_weight_attribute_name: Optional[str] = None
    max_candidates: Optional[int] = None
    max_runtime_per_training_job_in_seconds: Optional[int] = None
    max_total_job_runtime_in_seconds: Optional[int] = None

    @classmethod
    def from_response_dict(cls, api_problem_type_config: dict):
        """Convert the API response to the native object."""
        completion_criteria = api_problem_type_config.get("CompletionCriteria", {})
        return cls(
            max_candidates=completion_criteria.get("MaxCandidates"),
            max_runtime_per_training_job_in_seconds=completion_criteria.get(
                "MaxRuntimePerTrainingJobInSeconds"
            ),
            max_total_job_runtime_in_seconds=completion_criteria.get(
                "MaxAutoMLJobRuntimeInSeconds"
            ),
            algorithms_config=api_problem_type_config.get("CandidateGenerationConfig", {})
            .get("AlgorithmsConfig", [{}])[0]
            .get("AutoMLAlgorithms", None),
            feature_specification_s3_uri=api_problem_type_config.get("FeatureSpecificationS3Uri"),
            mode=api_problem_type_config.get("Mode"),
            generate_candidate_definitions_only=api_problem_type_config.get(
                "GenerateCandidateDefinitionsOnly", None
            ),
            problem_type=api_problem_type_config.get("ProblemType"),
            target_attribute_name=api_problem_type_config.get("TargetAttributeName"),
            sample_weight_attribute_name=api_problem_type_config.get("SampleWeightAttributeName"),
        )

    def to_request_dict(self):
        """Convert the native object to the API request format."""
        config = {}
        if _is_completion_criteria_exists_in_config(
            max_candidates=self.max_candidates,
            max_runtime_per_training_job_in_seconds=self.max_runtime_per_training_job_in_seconds,
            max_total_job_runtime_in_seconds=self.max_total_job_runtime_in_seconds,
        ):
            config["CompletionCriteria"] = _completion_criteria_to_request_dict(
                self.max_candidates,
                self.max_runtime_per_training_job_in_seconds,
                self.max_total_job_runtime_in_seconds,
            )
        config["TargetAttributeName"] = self.target_attribute_name
        if self.problem_type is not None:
            config["ProblemType"] = self.problem_type
        if self.sample_weight_attribute_name is not None:
            config["SampleWeightAttributeName"] = self.sample_weight_attribute_name
        if self.mode is not None:
            config["Mode"] = self.mode
        if self.generate_candidate_definitions_only is not None:
            config["GenerateCandidateDefinitionsOnly"] = self.generate_candidate_definitions_only
        if self.feature_specification_s3_uri is not None:
            config["FeatureSpecificationS3Uri"] = self.feature_specification_s3_uri

        if self.algorithms_config is not None:
            config["CandidateGenerationConfig"] = {
                "AlgorithmsConfig": [{"AutoMLAlgorithms": self.algorithms_config}]
            }
        return {"TabularJobConfig": config}


@dataclass
class AutoMLImageClassificationConfig(object):
    """Configuration of an image classification problem.

    Args:
        max_candidates (int): The maximum number of training jobs allowed to run.
        max_runtime_per_training_job_in_seconds (int): The maximum time, in seconds,
            that each training job executed inside hyperparameter tuning
            is allowed to run as part of a hyperparameter tuning job.
        max_total_job_runtime_in_seconds (int): The total wait time of an AutoML job.
    """

    max_candidates: Optional[int] = None
    max_runtime_per_training_job_in_seconds: Optional[int] = None
    max_total_job_runtime_in_seconds: Optional[int] = None

    @classmethod
    def from_response_dict(cls, api_problem_type_config: dict):
        """Convert the API response to the native object."""
        completion_criteria = api_problem_type_config.get("CompletionCriteria", {})
        return cls(
            max_candidates=completion_criteria.get("MaxCandidates"),
            max_runtime_per_training_job_in_seconds=completion_criteria.get(
                "MaxRuntimePerTrainingJobInSeconds"
            ),
            max_total_job_runtime_in_seconds=completion_criteria.get(
                "MaxAutoMLJobRuntimeInSeconds"
            ),
        )

    def to_request_dict(self):
        """Convert the native object to the API request format."""
        config = {}
        if _is_completion_criteria_exists_in_config(
            max_candidates=self.max_candidates,
            max_runtime_per_training_job_in_seconds=self.max_runtime_per_training_job_in_seconds,
            max_total_job_runtime_in_seconds=self.max_total_job_runtime_in_seconds,
        ):
            config["CompletionCriteria"] = _completion_criteria_to_request_dict(
                self.max_candidates,
                self.max_runtime_per_training_job_in_seconds,
                self.max_total_job_runtime_in_seconds,
            )
        return {"ImageClassificationJobConfig": config}


@dataclass
class AutoMLTextClassificationConfig(object):
    """Configuration of a text classification problem.

    Args:
        content_column (str): The name of the column used to provide the text to be classified.
            It should not be the same as the target label column.
        target_label_column (str): The name of the column used to provide the class labels.
            It should not be same as the content column.
        max_candidates (int): The maximum number of training jobs allowed to run.
        max_runtime_per_training_job_in_seconds (int): The maximum time, in seconds,
            that each training job executed inside hyperparameter tuning
            is allowed to run as part of a hyperparameter tuning job.
        max_total_job_runtime_in_seconds (int): The total wait time of an AutoML job.
    """

    content_column: str
    target_label_column: str
    max_candidates: Optional[int] = None
    max_runtime_per_training_job_in_seconds: Optional[int] = None
    max_total_job_runtime_in_seconds: Optional[int] = None

    @classmethod
    def from_response_dict(cls, api_problem_type_config: dict):
        """Convert the API response to the native object."""
        completion_criteria = api_problem_type_config.get("CompletionCriteria", {})
        return cls(
            max_candidates=completion_criteria.get("MaxCandidates"),
            max_runtime_per_training_job_in_seconds=completion_criteria.get(
                "MaxRuntimePerTrainingJobInSeconds"
            ),
            max_total_job_runtime_in_seconds=completion_criteria.get(
                "MaxAutoMLJobRuntimeInSeconds"
            ),
            content_column=api_problem_type_config["ContentColumn"],
            target_label_column=api_problem_type_config["TargetLabelColumn"],
        )

    def to_request_dict(self):
        """Convert the native object to the API request format."""
        config = {}
        if _is_completion_criteria_exists_in_config(
            max_candidates=self.max_candidates,
            max_runtime_per_training_job_in_seconds=self.max_runtime_per_training_job_in_seconds,
            max_total_job_runtime_in_seconds=self.max_total_job_runtime_in_seconds,
        ):
            config["CompletionCriteria"] = _completion_criteria_to_request_dict(
                self.max_candidates,
                self.max_runtime_per_training_job_in_seconds,
                self.max_total_job_runtime_in_seconds,
            )

        config["ContentColumn"] = self.content_column
        config["TargetLabelColumn"] = self.target_label_column

        return {"TextClassificationJobConfig": config}


@dataclass
class AutoMLTextGenerationConfig(object):
    """Configuration of a text generation problem.

    Args:
        base_model_name (str): The name of the base model to fine-tune.
            Autopilot supports fine-tuning a variety of large language models.
            For information on the list of supported models, see Text generation models supporting
            fine-tuning in Autopilot:
            https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-llms-finetuning-models.html#autopilot-llms-finetuning-supported-llms.
            If no BaseModelName is provided, the default model used is Falcon7BInstruct.
        accept_eula (bool): Specifies agreement to the model end-user license agreement (EULA).
            The AcceptEula value must be explicitly defined as True
            in order to accept the EULA that this model requires.
            For example, LLAMA2 requires to accept EULA. You are responsible for reviewing
            and complying with any applicable license terms and making sure they are acceptable
            for your use case before downloading or using a model.
        text_generation_hyper_params (dict): The hyperparameters used to configure and optimize
            the learning process of the base model. You can set any combination of the following
            hyperparameters for all base models. Supported parameters are:

            - epochCount: The number of times the model goes through the entire training dataset.
            - batchSize: The number of data samples used in each iteration of training.
            - learningRate: The step size at which a model's parameters are updated during training.
            - learningRateWarmupSteps: The number of training steps during which the learning rate
                gradually increases before reaching its target or maximum value.

        max_candidates (int): The maximum number of training jobs allowed to run.
        max_runtime_per_training_job_in_seconds (int): The maximum time, in seconds,
            that each training job executed inside hyperparameter tuning
            is allowed to run as part of a hyperparameter tuning job.
        max_total_job_runtime_in_seconds (int): The total wait time of an AutoML job.
    """

    base_model_name: Optional[str] = None
    accept_eula: Optional[bool] = None
    text_generation_hyper_params: Optional[Dict[str, str]] = None
    max_candidates: Optional[int] = None
    max_runtime_per_training_job_in_seconds: Optional[int] = None
    max_total_job_runtime_in_seconds: Optional[int] = None

    @classmethod
    def from_response_dict(cls, api_problem_type_config: dict):
        """Convert the API response to the native object."""
        completion_criteria = api_problem_type_config.get("CompletionCriteria", {})
        return cls(
            max_candidates=completion_criteria.get("MaxCandidates"),
            max_runtime_per_training_job_in_seconds=completion_criteria.get(
                "MaxRuntimePerTrainingJobInSeconds"
            ),
            max_total_job_runtime_in_seconds=completion_criteria.get(
                "MaxAutoMLJobRuntimeInSeconds"
            ),
            base_model_name=api_problem_type_config.get("BaseModelName"),
            text_generation_hyper_params=api_problem_type_config.get(
                "TextGenerationHyperParameters"
            ),
            accept_eula=api_problem_type_config.get("ModelAccessConfig", {}).get(
                "AcceptEula", None
            ),
        )

    def to_request_dict(self):
        """Convert the native object to the API request format."""
        config = {}
        if _is_completion_criteria_exists_in_config(
            max_candidates=self.max_candidates,
            max_runtime_per_training_job_in_seconds=self.max_runtime_per_training_job_in_seconds,
            max_total_job_runtime_in_seconds=self.max_total_job_runtime_in_seconds,
        ):
            config["CompletionCriteria"] = {}
            if self.max_candidates is not None:
                config["CompletionCriteria"]["MaxCandidates"] = self.max_candidates
            if self.max_runtime_per_training_job_in_seconds is not None:
                config["CompletionCriteria"][
                    "MaxRuntimePerTrainingJobInSeconds"
                ] = self.max_runtime_per_training_job_in_seconds
            if self.max_total_job_runtime_in_seconds is not None:
                config["CompletionCriteria"][
                    "MaxAutoMLJobRuntimeInSeconds"
                ] = self.max_total_job_runtime_in_seconds

        if self.base_model_name is not None:
            config["BaseModelName"] = self.base_model_name
        if self.accept_eula is not None:
            config["ModelAccessConfig"] = {"AcceptEula": self.accept_eula}
        if self.text_generation_hyper_params is not None:
            config["TextGenerationHyperParameters"] = self.text_generation_hyper_params

        return {"TextGenerationJobConfig": config}


@dataclass
class AutoMLTimeSeriesForecastingConfig(object):
    """Configuration of a time series forecasting problem.

    Args:
        forecast_frequency (str): The frequency of predictions in a forecast.
            Valid intervals are an integer followed by Y (Year),
            M (Month), W (Week), D (Day), H (Hour), and min (Minute).
            For example, 1D indicates every day and 15min indicates every 15 minutes.
            The value of a frequency must not overlap with the next larger frequency.
            For example, you must use a frequency of 1H instead of 60min.
        forecast_horizon (int): The number of time-steps that the model predicts. The forecast
            horizon is also called the prediction length. The maximum forecast horizon
            is the lesser of 500 time-steps or 1/4 of the time-steps in the dataset.
        item_identifier_attribute_name (str): The name of the column that represents
            the set of item identifiers for which you want to predict the target value.
        target_attribute_name (str): The name of the column representing the target variable
            that you want to predict for each item in your dataset.
            The data type of the target variable must be numerical.
        timestamp_attribute_name (str): The name of the column indicating a point in time at which
            the target value of a given item is recorded.
        grouping_attribute_names (list(str)): A set of columns names that can be grouped with the
            item identifier column to create a composite key for which a target value is predicted.
        feature_specification_s3_uri (str): A URL to the Amazon S3 data source containing
            selected features and specified data types from the input data source of an AutoML job.
        forecast_quantiles (list(str)): The quantiles used to train the model for forecasts
            at a specified quantile. You can specify quantiles from 0.01 (p1) to 0.99 (p99),
            by increments of 0.01 or higher. Up to five forecast quantiles can be specified.
            When ForecastQuantiles is not provided, the AutoML job uses the quantiles p10, p50,
            and p90 as default.
        holiday_config (list(str)): The country code for the holiday calendar.
            For the list of public holiday calendars supported by AutoML job V2, see Country Codes:
            https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-timeseries-forecasting-holiday-calendars.html#holiday-country-codes.
            Use the country code corresponding to the country of your choice.
        aggregation (dict): A key value pair defining the aggregation method for a column,
            where the key is the column name and the value is the aggregation method.
            Aggregation is only supported for the target column. The supported aggregation methods
            are sum (default), avg, first, min, max.
        filling (dict): A key value pair defining the filling method for a column,
            where the key is the column name and the value is an object which defines
            the filling logic. You can specify multiple filling methods for a single column.
            The supported filling methods and their corresponding options are:

            - frontfill: none (Supported only for target column)
            - middlefill: zero, value, median, mean, min, max
            - backfill: zero, value, median, mean, min, max
            - futurefill: zero, value, median, mean, min, max

            To set a filling method to a specific value, set the fill parameter to
            the chosen filling method value (for example "backfill" : "value"),
            and define the filling value in an additional parameter prefixed with "_value".
            For example, to set backfill to a value of 2, you must include two parameters:
            "backfill": "value" and "backfill_value":"2".
        max_candidates (int): The maximum number of training jobs allowed to run.
        max_runtime_per_training_job_in_seconds (int): The maximum time, in seconds,
            that each training job executed inside hyperparameter tuning
            is allowed to run as part of a hyperparameter tuning job.
        max_total_job_runtime_in_seconds (int): The total wait time of an AutoML job.
    """

    forecast_frequency: str
    forecast_horizon: int
    item_identifier_attribute_name: str
    target_attribute_name: str
    timestamp_attribute_name: str
    grouping_attribute_names: Optional[List[str]] = None
    feature_specification_s3_uri: Optional[str] = None
    forecast_quantiles: Optional[List[str]] = None
    holiday_config: Optional[List[str]] = None
    aggregation: Optional[Dict[str, str]] = None
    filling: Optional[Dict[str, str]] = None
    max_candidates: Optional[int] = None
    max_runtime_per_training_job_in_seconds: Optional[int] = None
    max_total_job_runtime_in_seconds: Optional[int] = None

    @classmethod
    def from_response_dict(cls, api_problem_type_config: dict):
        """Convert the API response to the native object."""
        completion_criteria = api_problem_type_config.get("CompletionCriteria", {})
        return cls(
            max_candidates=completion_criteria.get("MaxCandidates"),
            max_runtime_per_training_job_in_seconds=completion_criteria.get(
                "MaxRuntimePerTrainingJobInSeconds"
            ),
            max_total_job_runtime_in_seconds=completion_criteria.get(
                "MaxAutoMLJobRuntimeInSeconds"
            ),
            feature_specification_s3_uri=api_problem_type_config.get("FeatureSpecificationS3Uri"),
            forecast_frequency=api_problem_type_config["ForecastFrequency"],
            forecast_horizon=api_problem_type_config["ForecastHorizon"],
            item_identifier_attribute_name=api_problem_type_config["TimeSeriesConfig"][
                "ItemIdentifierAttributeName"
            ],
            target_attribute_name=api_problem_type_config["TimeSeriesConfig"][
                "TargetAttributeName"
            ],
            timestamp_attribute_name=api_problem_type_config["TimeSeriesConfig"][
                "TimestampAttributeName"
            ],
            forecast_quantiles=api_problem_type_config.get("ForecastQuantiles"),
            aggregation=api_problem_type_config.get("Transformations", {}).get("Aggregation"),
            filling=api_problem_type_config.get("Transformations", {}).get("Filling"),
            grouping_attribute_names=api_problem_type_config.get("TimeSeriesConfig", {}).get(
                "GroupingAttributeNames"
            ),
            holiday_config=api_problem_type_config.get("HolidayConfig", [{}])[0].get("CountryCode"),
        )

    def to_request_dict(self):
        """Convert the native object to the API request format."""
        config = {}
        if _is_completion_criteria_exists_in_config(
            max_candidates=self.max_candidates,
            max_runtime_per_training_job_in_seconds=self.max_runtime_per_training_job_in_seconds,
            max_total_job_runtime_in_seconds=self.max_total_job_runtime_in_seconds,
        ):
            config["CompletionCriteria"] = _completion_criteria_to_request_dict(
                self.max_candidates,
                self.max_runtime_per_training_job_in_seconds,
                self.max_total_job_runtime_in_seconds,
            )

        if self.feature_specification_s3_uri is not None:
            config["FeatureSpecificationS3Uri"] = self.feature_specification_s3_uri

        config["ForecastHorizon"] = self.forecast_horizon
        config["ForecastFrequency"] = self.forecast_frequency
        config["TimeSeriesConfig"] = {
            "TargetAttributeName": self.target_attribute_name,
            "TimestampAttributeName": self.timestamp_attribute_name,
            "ItemIdentifierAttributeName": self.item_identifier_attribute_name,
        }
        if self.grouping_attribute_names:
            config["TimeSeriesConfig"]["GroupingAttributeNames"] = self.grouping_attribute_names

        if self.forecast_quantiles:
            config["ForecastQuantiles"] = self.forecast_quantiles

        if self.holiday_config:
            config["HolidayConfig"] = []
            config["HolidayConfig"].append({"CountryCode": self.holiday_config})

        if self.aggregation or self.filling:
            config["Transformations"] = {}
            if self.aggregation:
                config["Transformations"]["Aggregation"] = self.aggregation
            if self.filling:
                config["Transformations"]["Filling"] = self.filling

        return {"TimeSeriesForecastingJobConfig": config}


@dataclass
class AutoMLDataChannel(object):
    """Class to represnt the datasource which will be used for mode training.

    Args:
        s3_data_type (str): The data type for S3 data source. Valid values: ManifestFile,
            AugmentedManifestFile or S3Prefix.
        s3_uri (str): The URL to the Amazon S3 data source. The Uri refers to the Amazon S3 prefix
            or ManifestFile depending on the data type.
        channel_type (str): The type of channel. Valid values: `training` or `validation`.
            Defines whether the data are used for training or validation.
            The default value is training.
            Channels for training and validation must share the same content_type.
        compression_type (str): The compression type for input data. Gzip or None.
        content_type (str): The content type of the data from the input source.
    """

    s3_data_type: str
    s3_uri: str
    channel_type: Optional[str] = None
    compression_type: Optional[str] = None
    content_type: Optional[str] = None

    @classmethod
    def from_response_dict(cls, data_channel: dict):
        """Convert the API response to the native object."""
        return cls(
            s3_data_type=data_channel["DataSource"]["S3DataSource"]["S3DataType"],
            s3_uri=data_channel["DataSource"]["S3DataSource"]["S3Uri"],
            channel_type=data_channel.get("ChannelType"),
            compression_type=data_channel.get("CompressionType"),
            content_type=data_channel.get("ContentType"),
        )

    def to_request_dict(self):
        """Convert the native object to the API request format."""
        request_dict = {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": self.s3_data_type,
                    "S3Uri": self.s3_uri,
                }
            },
        }
        if self.channel_type:
            request_dict["ChannelType"] = self.channel_type
        if self.compression_type:
            request_dict["CompressionType"] = self.compression_type
        if self.content_type:
            request_dict["ContentType"] = self.content_type
        return request_dict


@dataclass
class LocalAutoMLDataChannel(object):
    """Class to represnt a local datasource which will be uploaded to S3.

    Args:
        data_type (str): The data type for S3 data source. Valid values: ManifestFile,
            AugmentedManifestFile or S3Prefix.
        path (str): The path to the local data which will be uploaded to S3.
        channel_type (str): The type of channel. Valid values: `training` or `validation`.
            Defines whether the data are used for training or validation.
            The default value is training.
            Channels for training and validation must share the same content_type.
        compression_type (str): The compression type for input data. Gzip or None.
        content_type (str): The content type of the data from the input source.
    """

    data_type: str
    path: str
    channel_type: Optional[str] = None
    compression_type: Optional[str] = None
    content_type: Optional[str] = None


def _upload_local_dataset(
    local_dataset: LocalAutoMLDataChannel, sagemaker_session: Session
) -> AutoMLDataChannel:
    """Method to upload a local dataset to the S3 and convert it to an AutoMLDataChannel object."""
    s3_path = sagemaker_session.upload_data(local_dataset.path, key_prefix="auto-ml-v2-input-data")
    return AutoMLDataChannel(
        s3_uri=s3_path,
        s3_data_type=local_dataset.data_type,
        channel_type=local_dataset.channel_type,
        compression_type=local_dataset.compression_type,
        content_type=local_dataset.content_type,
    )


def _is_completion_criteria_exists_in_config(
    max_candidates: int = None,
    max_runtime_per_training_job_in_seconds: int = None,
    max_total_job_runtime_in_seconds: int = None,
) -> bool:
    """Check is the completion criteria was provided as part of the problem config or not."""
    return (
        max_candidates is not None
        or max_runtime_per_training_job_in_seconds is not None
        or max_total_job_runtime_in_seconds is not None
    )


def _completion_criteria_to_request_dict(
    max_candidates: int = None,
    max_runtime_per_training_job_in_seconds: int = None,
    max_total_job_runtime_in_seconds: int = None,
):
    """Convert a completion criteria object to an API request format."""
    config = {}
    if max_candidates is not None:
        config["MaxCandidates"] = max_candidates
    if max_runtime_per_training_job_in_seconds is not None:
        config["MaxRuntimePerTrainingJobInSeconds"] = max_runtime_per_training_job_in_seconds
    if max_total_job_runtime_in_seconds is not None:
        config["MaxAutoMLJobRuntimeInSeconds"] = max_total_job_runtime_in_seconds
    return config


class AutoMLV2(object):
    """A class for creating and interacting with SageMaker AutoMLV2 jobs."""

    def __init__(
        self,
        problem_config: Union[
            AutoMLTabularConfig,
            AutoMLImageClassificationConfig,
            AutoMLTextClassificationConfig,
            AutoMLTextGenerationConfig,
            AutoMLTimeSeriesForecastingConfig,
        ],
        base_job_name: Optional[str] = None,
        output_path: Optional[str] = None,
        job_objective: Optional[Dict[str, str]] = None,
        validation_fraction: Optional[float] = None,
        auto_generate_endpoint_name: Optional[bool] = None,
        endpoint_name: Optional[str] = None,
        output_kms_key: Optional[str] = None,
        role: Optional[str] = None,
        volume_kms_key: Optional[str] = None,
        encrypt_inter_container_traffic: Optional[bool] = None,
        vpc_config: Optional[Dict[str, List]] = None,
        tags: Optional[Tags] = None,
        sagemaker_session: Optional[Session] = None,
    ):
        """Initialize an AutoMLV2 object.

        Args:
            problem_config (object): A collection of settings specific
                to the problem type used to configure an AutoML job V2.
                There must be one and only one config of the following type.
                Supported problem types are:

                - Image Classification (sagemaker.automl.automlv2.ImageClassificationJobConfig),
                - Tabular (sagemaker.automl.automlv2.TabularJobConfig),
                - Text Classification (sagemaker.automl.automlv2.TextClassificationJobConfig),
                - Text Generation (TextGenerationJobConfig),
                - Time Series Forecasting (
                    sagemaker.automl.automlv2.TimeSeriesForecastingJobConfig).

            base_job_name (str): The name of AutoML job.
                The name must be unique to within the AWS account and is case-insensitive.
            output_path (str): The Amazon S3 output path. Must be 128 characters or less.
            job_objective (dict[str, str]): Defines the objective metric
                used to measure the predictive quality of an AutoML job.
                In the format of: {"MetricName": str}. Available metrics are listed here:
                https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-metrics-validation.html
            validation_fraction (float): A float that specifies the portion of
                the input dataset to be used for validation. The value should be in (0, 1) range.
            auto_generate_endpoint_name (bool): Whether to automatically generate
                an endpoint name for a one-click Autopilot model deployment.
                If set auto_generate_endpoint_name to True, do not specify the endpoint_name.
            endpoint_name (str): Specifies the endpoint name to use for a one-click AutoML
                model deployment if the endpoint name is not generated automatically.
                Specify the endpoint_name if and only if
                auto_generate_endpoint_name is set to False
            output_kms_key (str): The AWS KMS encryption key ID for output data configuration
            role (str): The ARN of the role that is used to create the job and access the data.
            volume_kms_key (str): The key used to encrypt stored data.
            encrypt_inter_container_traffic (bool): whether to use traffic encryption
                between the container layers.
            vpc_config (dict): Specifies a VPC that your training jobs and hosted models have
                access to. Contents include "SecurityGroupIds" and "Subnets".
            tags (Optional[Tags]): Tags to attach to this specific endpoint.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions.

        Returns:
            AutoMLV2 object.
        """
        self.base_job_name = base_job_name
        self.problem_config = problem_config
        self.job_objective = job_objective
        self.validation_fraction = validation_fraction
        self.auto_generate_endpoint_name = auto_generate_endpoint_name
        self.endpoint_name = endpoint_name
        self.output_path = output_path
        self.sagemaker_session = sagemaker_session or Session()

        self.vpc_config = resolve_value_from_config(
            vpc_config,
            AUTO_ML_VPC_CONFIG_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.volume_kms_key = resolve_value_from_config(
            volume_kms_key,
            AUTO_ML_VOLUME_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.output_kms_key = resolve_value_from_config(
            output_kms_key,
            AUTO_ML_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.role = resolve_value_from_config(
            role, AUTO_ML_ROLE_ARN_PATH, sagemaker_session=self.sagemaker_session
        )
        if not self.role:
            # Originally IAM role was a required parameter.
            # Now we marked that as Optional because we can fetch it from SageMakerConfig
            # Because of marking that parameter as optional, we should validate if it is None, even
            # after fetching the config.
            raise ValueError("An AWS IAM role is required to create an AutoML job.")

        if isinstance(problem_config, AutoMLTabularConfig):
            self._check_problem_type_and_job_objective(problem_config.problem_type, job_objective)

        self.encrypt_inter_container_traffic = resolve_value_from_config(
            direct_input=encrypt_inter_container_traffic,
            config_path=AUTO_ML_INTER_CONTAINER_ENCRYPTION_PATH,
            default_value=False,
            sagemaker_session=self.sagemaker_session,
        )

        if self.output_path is None:
            self.output_path = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                with_end_slash=True,
            )

        self.tags = format_tags(tags)
        self.sagemaker_session = sagemaker_session or Session()

        self.current_job_name = None
        self.inputs = None
        self.latest_auto_ml_job = None
        self._auto_ml_job_desc = None
        self._best_candidate = None

    @classmethod
    def from_auto_ml(cls, auto_ml: AutoML) -> AutoMLV2:
        """Create an AutoMLV2 object from an AutoML object.

        This method maps AutoML properties into an AutoMLV2 object,
        so you can create AutoMLV2 jobs from the existing AutoML objects.

        Args:
            auto_ml (sagemaker.automl.automl.AutoML): An AutoML object from which
                an AutoMLV2 object will be created.
        """
        auto_ml_v2 = AutoMLV2(
            problem_config=AutoMLTabularConfig(
                target_attribute_name=auto_ml.target_attribute_name,
                feature_specification_s3_uri=auto_ml.feature_specification_s3_uri,
                generate_candidate_definitions_only=auto_ml.generate_candidate_definitions_only,
                mode=auto_ml.mode,
                problem_type=auto_ml.problem_type,
                sample_weight_attribute_name=auto_ml.sample_weight_attribute_name,
                max_candidates=auto_ml.max_candidate,
                max_runtime_per_training_job_in_seconds=auto_ml.max_runtime_per_training_job_in_seconds,  # noqa E501  # pylint: disable=c0301
                max_total_job_runtime_in_seconds=auto_ml.total_job_runtime_in_seconds,
            ),
            base_job_name=auto_ml.base_job_name,
            output_path=auto_ml.output_path,
            output_kms_key=auto_ml.output_kms_key,
            job_objective=auto_ml.job_objective,
            validation_fraction=auto_ml.validation_fraction,
            auto_generate_endpoint_name=auto_ml.auto_generate_endpoint_name,
            endpoint_name=auto_ml.endpoint_name,
            role=auto_ml.role,
            volume_kms_key=auto_ml.volume_kms_key,
            encrypt_inter_container_traffic=auto_ml.encrypt_inter_container_traffic,
            vpc_config=auto_ml.vpc_config,
            tags=auto_ml.tags,
            sagemaker_session=auto_ml.sagemaker_session,
        )
        auto_ml_v2._best_candidate = auto_ml._best_candidate
        return auto_ml_v2

    def fit(
        self,
        inputs: Optional[
            Union[
                LocalAutoMLDataChannel,
                AutoMLDataChannel,
                List[LocalAutoMLDataChannel],
                List[AutoMLDataChannel],
            ]
        ],
        wait: bool = True,
        logs: bool = True,
        job_name: str = None,
    ):
        """Create an AutoML Job with the input dataset.

        Args:
            inputs (LocalAutoMLDataChannel or list(LocalAutoMLDataChannel) or AutoMLDataChannel
                or list(AutoMLDataChannel)): Local path or S3 Uri where the training data is stored.
                Or an AutoMLDataChannel object. Or a list of AutoMLDataChannel objects.
                If a local path in LocalAutoMLDataChannel is provided,
                the dataset will be uploaded to an S3 location.
                The list of AutoMLDataChannel objects is to specify the training or the validation
                input source. Input source for training and validation
                must share the same content type and target attribute name.
                Minimum number of 1 item. Maximum number of 2 items for list[AutoMLDataChannel].
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job. Only meaningful when wait
                is True (default: True). if ``wait`` is False, ``logs`` will be set to False as
                well.
            job_name (str): The job name. If not specified, the estimator generates
                a default job name, based on the training image name and current timestamp.
        """
        if not wait and logs:
            logs = False
            logger.warning("Setting logs to False. logs is only meaningful when wait is True.")

        # upload data for users if provided local path with LocalAutoMLDataChannel
        if isinstance(inputs, LocalAutoMLDataChannel):
            inputs = _upload_local_dataset(inputs, self.sagemaker_session)
        elif isinstance(inputs, list) and all(
            isinstance(channel, LocalAutoMLDataChannel) for channel in inputs
        ):
            inputs = [_upload_local_dataset(channel, self.sagemaker_session) for channel in inputs]

        self._prepare_for_auto_ml_job(job_name=job_name)
        self.inputs = inputs
        self.latest_auto_ml_job = AutoMLJobV2.start_new(self, inputs)  # pylint: disable=W0201
        if wait:
            self.latest_auto_ml_job.wait(logs=logs)

    @classmethod
    def attach(cls, auto_ml_job_name, sagemaker_session=None):
        """Attach to an existing AutoML job.

        Creates and returns a AutoML bound to an existing automl job.

        Args:
            auto_ml_job_name (str): AutoML job name
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, the one originally associated with the ``AutoML`` instance is used.

        Returns:
            sagemaker.automl.AutoML: A ``AutoMLV2`` instance with the attached automl job.

        """
        sagemaker_session = sagemaker_session or Session()

        auto_ml_job_desc = sagemaker_session.describe_auto_ml_job_v2(auto_ml_job_name)
        automl_job_tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=auto_ml_job_desc["AutoMLJobArn"]
        )["Tags"]
        inputs = [
            AutoMLDataChannel.from_response_dict(channel)
            for channel in auto_ml_job_desc["AutoMLJobInputDataConfig"]
        ]

        problem_type = auto_ml_job_desc["AutoMLProblemTypeConfigName"]
        problem_config = None
        if problem_type == "ImageClassification":
            problem_config = AutoMLImageClassificationConfig.from_response_dict(
                auto_ml_job_desc["AutoMLProblemTypeConfig"]["ImageClassificationJobConfig"]
            )
        elif problem_type == "TextClassification":
            problem_config = AutoMLTextClassificationConfig.from_response_dict(
                auto_ml_job_desc["AutoMLProblemTypeConfig"]["TextClassificationJobConfig"]
            )
        elif problem_type == "TimeSeriesForecasting":
            problem_config = AutoMLTimeSeriesForecastingConfig.from_response_dict(
                auto_ml_job_desc["AutoMLProblemTypeConfig"]["TimeSeriesForecastingJobConfig"]
            )
        elif problem_type == "Tabular":
            problem_config = AutoMLTabularConfig.from_response_dict(
                auto_ml_job_desc["AutoMLProblemTypeConfig"]["TabularJobConfig"]
            )
        elif problem_type == "TextGeneration":
            problem_config = AutoMLTextGenerationConfig.from_response_dict(
                auto_ml_job_desc["AutoMLProblemTypeConfig"]["TextGenerationJobConfig"]
            )

        amlj = AutoMLV2(
            role=auto_ml_job_desc["RoleArn"],
            problem_config=problem_config,
            output_path=auto_ml_job_desc["OutputDataConfig"]["S3OutputPath"],
            output_kms_key=auto_ml_job_desc["OutputDataConfig"].get("KmsKeyId"),
            base_job_name=auto_ml_job_name,
            sagemaker_session=sagemaker_session,
            volume_kms_key=auto_ml_job_desc.get("SecurityConfig", {}).get("VolumeKmsKeyId"),
            # Do not override encrypt_inter_container_traffic from config because this info
            # is pulled from an existing automl job
            encrypt_inter_container_traffic=auto_ml_job_desc.get("SecurityConfig", {}).get(
                "EnableInterContainerTrafficEncryption"
            ),
            vpc_config=auto_ml_job_desc.get("SecurityConfig", {}).get("VpcConfig"),
            job_objective=auto_ml_job_desc.get("AutoMLJobObjective", {}),
            auto_generate_endpoint_name=auto_ml_job_desc.get("ModelDeployConfig", {}).get(
                "AutoGenerateEndpointName", False
            ),
            endpoint_name=auto_ml_job_desc.get("ModelDeployConfig", {}).get("EndpointName"),
            validation_fraction=auto_ml_job_desc.get("DataSplitConfig", {}).get(
                "ValidationFraction"
            ),
            tags=automl_job_tags,
        )
        amlj.current_job_name = auto_ml_job_name
        amlj.latest_auto_ml_job = auto_ml_job_name  # pylint: disable=W0201
        amlj._auto_ml_job_desc = auto_ml_job_desc
        amlj.inputs = inputs
        return amlj

    def describe_auto_ml_job(self, job_name=None):
        """Returns the job description of an AutoML job for the given job name.

        Args:
            job_name (str): The name of the AutoML job to describe.
                If None, will use object's latest_auto_ml_job name.

        Returns:
            dict: A dictionary response with the AutoML Job description.
        """
        if job_name is None:
            job_name = self.current_job_name
        self._auto_ml_job_desc = self.sagemaker_session.describe_auto_ml_job_v2(job_name=job_name)
        return self._auto_ml_job_desc

    def best_candidate(self, job_name=None):
        """Returns the best candidate of an AutoML job for a given name.

        Args:
            job_name (str): The name of the AutoML job. If None, object's
                _current_auto_ml_job_name will be used.

        Returns:
            dict: A dictionary with information of the best candidate.
        """
        if self._best_candidate:
            return self._best_candidate

        if job_name is None:
            job_name = self.current_job_name
        if self._auto_ml_job_desc is None:
            self._auto_ml_job_desc = self.sagemaker_session.describe_auto_ml_job_v2(
                job_name=job_name
            )
        elif self._auto_ml_job_desc["AutoMLJobName"] != job_name:
            self._auto_ml_job_desc = self.sagemaker_session.describe_auto_ml_job_v2(
                job_name=job_name
            )

        self._best_candidate = self._auto_ml_job_desc["BestCandidate"]
        return self._best_candidate

    def list_candidates(
        self,
        job_name=None,
        status_equals=None,
        candidate_name=None,
        candidate_arn=None,
        sort_order=None,
        sort_by=None,
        max_results=None,
    ):
        """Returns the list of candidates of an AutoML job for a given name.

        Args:
            job_name (str): The name of the AutoML job. If None, will use object's
                _current_job name.
            status_equals (str): Filter the result with candidate status, values could be
                "Completed", "InProgress", "Failed", "Stopped", "Stopping"
            candidate_name (str): The name of a specified candidate to list.
                Default to None.
            candidate_arn (str): The Arn of a specified candidate to list.
                Default to None.
            sort_order (str): The order that the candidates will be listed in result.
                Default to None.
            sort_by (str): The value that the candidates will be sorted by.
                Default to None.
            max_results (int): The number of candidates will be listed in results,
                between 1 to 100. Default to None. If None, will return all the candidates.

        Returns:
            list: A list of dictionaries with candidates information.
        """
        if job_name is None:
            job_name = self.current_job_name

        list_candidates_args = {"job_name": job_name}

        if status_equals:
            list_candidates_args["status_equals"] = status_equals
        if candidate_name:
            list_candidates_args["candidate_name"] = candidate_name
        if candidate_arn:
            list_candidates_args["candidate_arn"] = candidate_arn
        if sort_order:
            list_candidates_args["sort_order"] = sort_order
        if sort_by:
            list_candidates_args["sort_by"] = sort_by
        if max_results:
            list_candidates_args["max_results"] = max_results

        return self.sagemaker_session.list_candidates(**list_candidates_args)["Candidates"]

    def create_model(
        self,
        name,
        sagemaker_session=None,
        candidate=None,
        vpc_config=None,
        enable_network_isolation=False,
        model_kms_key=None,
        predictor_cls=None,
        inference_response_keys=None,
    ):
        """Creates a model from a given candidate or the best candidate from the job.

        Args:
            name (str): The pipeline model name.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, the one originally associated with the ``AutoML`` instance is used.:
            candidate (CandidateEstimator or dict): a CandidateEstimator used for deploying
                to a SageMaker Inference Pipeline. If None, the best candidate will
                be used. If the candidate input is a dict, a CandidateEstimator will be
                created from it.
            vpc_config (dict): Specifies a VPC that your training jobs and hosted models have
                access to. Contents include "SecurityGroupIds" and "Subnets".
            enable_network_isolation (bool): Isolates the training container. No inbound or
                outbound network calls can be made, except for calls between peers within a
                training cluster for distributed training. Default: False
            model_kms_key (str): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If
                specified, ``deploy()``  returns the result of invoking this
                function on the created endpoint name.
            inference_response_keys (list): List of keys for response content. The order of the
                keys will dictate the content order in the response.

        Returns:
            PipelineModel object.
        """
        sagemaker_session = sagemaker_session or self.sagemaker_session

        if candidate is None:
            candidate_dict = self.best_candidate()
            candidate = CandidateEstimator(candidate_dict, sagemaker_session=sagemaker_session)
        elif isinstance(candidate, dict):
            candidate = CandidateEstimator(candidate, sagemaker_session=sagemaker_session)

        inference_containers = candidate.containers

        self.validate_and_update_inference_response(inference_containers, inference_response_keys)

        models = []

        for container in inference_containers:
            model = Model(
                image_uri=container["Image"],
                model_data=container["ModelDataUrl"],
                role=self.role,
                env=container["Environment"],
                vpc_config=vpc_config,
                sagemaker_session=sagemaker_session or self.sagemaker_session,
                enable_network_isolation=enable_network_isolation,
                model_kms_key=model_kms_key,
            )
            models.append(model)

        pipeline = PipelineModel(
            models=models,
            role=self.role,
            predictor_cls=predictor_cls,
            name=name,
            vpc_config=vpc_config,
            enable_network_isolation=enable_network_isolation,
            sagemaker_session=sagemaker_session or self.sagemaker_session,
        )
        return pipeline

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        serializer=None,
        deserializer=None,
        candidate=None,
        sagemaker_session=None,
        name=None,
        endpoint_name=None,
        tags=None,
        wait=True,
        vpc_config=None,
        enable_network_isolation=False,
        model_kms_key=None,
        predictor_cls=None,
        inference_response_keys=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
    ):
        """Deploy a candidate to a SageMaker Inference Pipeline.

        Args:
            initial_instance_count (int): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``.
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
                serializer object, used to encode data for an inference endpoint
                (default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``.
            deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
                deserializer object, used to decode data from an inference
                endpoint (default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``.
            candidate (CandidateEstimator or dict): a CandidateEstimator used for deploying
                to a SageMaker Inference Pipeline. If None, the best candidate will
                be used. If the candidate input is a dict, a CandidateEstimator will be
                created from it.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, the one originally associated with the ``AutoML`` instance is used.
            name (str): The pipeline model name. If None, a default model name will
                be selected on each ``deploy``.
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
            tags (Optional[Tags]): The list of tags to attach to this
                specific endpoint.
            wait (bool): Whether the call should wait until the deployment of
                model completes (default: True).
            vpc_config (dict): Specifies a VPC that your training jobs and hosted models have
                access to. Contents include "SecurityGroupIds" and "Subnets".
            enable_network_isolation (bool): Isolates the training container. No inbound or
                outbound network calls can be made, except for calls between peers within a
                training cluster for distributed training. Default: False
            model_kms_key (str): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If
                specified, ``deploy()``  returns the result of invoking this
                function on the created endpoint name.
            inference_response_keys (list): List of keys for response content. The order of the
                keys will dictate the content order in the response.
            volume_size (int): The size, in GB, of the ML storage volume attached to individual
                inference instance associated with the production variant. Currenly only Amazon EBS
                gp2 storage volumes are supported.
            model_data_download_timeout (int): The timeout value, in seconds, to download and
                extract model data from Amazon S3 to the individual inference instance associated
                with this production variant.
            container_startup_health_check_timeout (int): The timeout value, in seconds, for your
                inference container to pass health check by SageMaker Hosting. For more information
                about health check see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-algo-ping-requests

        Returns:
            callable[string, sagemaker.session.Session] or ``None``:
                If ``predictor_cls`` is specified, the invocation of ``self.predictor_cls`` on
                the created endpoint name. Otherwise, ``None``.
        """
        sagemaker_session = sagemaker_session or self.sagemaker_session
        model = self.create_model(
            name=name,
            sagemaker_session=sagemaker_session,
            candidate=candidate,
            inference_response_keys=inference_response_keys,
            vpc_config=vpc_config,
            enable_network_isolation=enable_network_isolation,
            model_kms_key=model_kms_key,
            predictor_cls=predictor_cls,
        )

        return model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            serializer=serializer,
            deserializer=deserializer,
            endpoint_name=endpoint_name,
            kms_key=model_kms_key,
            tags=format_tags(tags),
            wait=wait,
            volume_size=volume_size,
            model_data_download_timeout=model_data_download_timeout,
            container_startup_health_check_timeout=container_startup_health_check_timeout,
        )

    def _prepare_for_auto_ml_job(self, job_name=None):
        """Set any values in the AutoMLJob that need to be set before creating request.

        Args:
            job_name (str): The name of the AutoML job. If None, a job name will be
                created from base_job_name or "sagemaker-auto-ml".
        """
        if job_name is not None:
            self.current_job_name = job_name
        else:
            if self.base_job_name:
                base_name = self.base_job_name
            else:
                base_name = "automl"
            # CreateAutoMLJob API validates that member length less than or equal to 32
            self.current_job_name = name_from_base(base_name, max_length=32)

        if self.output_path is None:
            self.output_path = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
                with_end_slash=True,
            )

    def _check_problem_type_and_job_objective(self, problem_type, job_objective):
        """Validate if problem_type and job_objective are both None or are both provided.

        Args:
            problem_type (str): The type of problem of this AutoMLJob. Valid values are
                "Regression", "BinaryClassification", "MultiClassClassification".
            job_objective (dict): AutoMLJob objective, contains "AutoMLJobObjectiveType" (optional),
                "MetricName" and "Value".

        Raises (ValueError): raises ValueError if one of problem_type and job_objective is provided
            while the other is None.
        """
        if not (problem_type and job_objective) and (problem_type or job_objective):
            raise ValueError(
                "One of problem type and objective metric provided. "
                "Either both of them should be provided or none of them should be provided."
            )

    @classmethod
    def _get_supported_inference_keys(cls, container, default=None):
        """Returns the inference keys supported by the container.

        Args:
            container (dict): Dictionary representing container
            default (object): The value to be returned if the container definition
                              has no marker environment variable

        Returns:
            List of inference keys the container support or default

        Raises:
            KeyError if the default is None and the container definition has
            no marker environment variable SAGEMAKER_INFERENCE_SUPPORTED.
        """
        try:
            return [
                x.strip()
                for x in container["Environment"]["SAGEMAKER_INFERENCE_SUPPORTED"].split(",")
            ]
        except KeyError:
            if default is None:
                raise
        return default

    @classmethod
    def _check_inference_keys(cls, inference_response_keys, containers):
        """Checks if the pipeline supports the inference keys for the containers.

        Given inference response keys and list of containers, determines whether
        the keys are supported.

        Args:
            inference_response_keys (list): List of keys for inference response content.
            containers (list): list of inference container.

        Raises:
            ValueError, if one or more keys in inference_response_keys are not supported
            the inference pipeline.
        """
        if not inference_response_keys:
            return
        try:
            supported_inference_keys = cls._get_supported_inference_keys(container=containers[-1])
        except KeyError:
            raise ValueError(
                "The inference model does not support selection of inference content beyond "
                "it's default content. Please retry without setting "
                "inference_response_keys key word argument."
            )
        bad_keys = []
        for key in inference_response_keys:
            if key not in supported_inference_keys:
                bad_keys.append(key)

        if bad_keys:
            raise ValueError(
                "Requested inference output keys [{bad_keys_str}] are unsupported. "
                "The supported inference keys are [{allowed_keys_str}]".format(
                    bad_keys_str=", ".join(bad_keys),
                    allowed_keys_str=", ".join(supported_inference_keys),
                )
            )

    @classmethod
    def validate_and_update_inference_response(cls, inference_containers, inference_response_keys):
        """Validates the requested inference keys and updates response content.

        On validation, also updates the inference containers to emit appropriate response
        content in the inference response.

        Args:
            inference_containers (list): list of inference containers
            inference_response_keys (list): list of inference response keys

        Raises:
            ValueError: if one or more of inference_response_keys are unsupported by the model
        """
        if not inference_response_keys:
            return

        cls._check_inference_keys(inference_response_keys, inference_containers)

        previous_container_output = None

        for container in inference_containers:
            supported_inference_keys_container = cls._get_supported_inference_keys(
                container, default=[]
            )
            if not supported_inference_keys_container:
                previous_container_output = None
                continue
            current_container_output = None
            for key in inference_response_keys:
                if key in supported_inference_keys_container:
                    current_container_output = (
                        current_container_output + "," + key if current_container_output else key
                    )

            if previous_container_output:
                container["Environment"].update(
                    {"SAGEMAKER_INFERENCE_INPUT": previous_container_output}
                )
            if current_container_output:
                container["Environment"].update(
                    {"SAGEMAKER_INFERENCE_OUTPUT": current_container_output}
                )
            previous_container_output = current_container_output


class AutoMLJobV2(_Job):
    """A class for interacting with CreateAutoMLJobV2 API."""

    def __init__(self, sagemaker_session, job_name, inputs):
        self.inputs = inputs
        self.job_name = job_name
        super(AutoMLJobV2, self).__init__(sagemaker_session=sagemaker_session, job_name=job_name)

    @classmethod
    def _get_auto_ml_v2_args(cls, auto_ml, inputs):
        """Constructs a dict of arguments for an Amazon SageMaker AutoMLV2 job.

        Args:
            auto_ml (sagemaker.automl.AutoMLV2): AutoMLV2 object
                created by the user.
            inputs (AutoMLDataChannel or list[AutoMLDataChannel]):
                Parameters used when called
                :meth:`~sagemaker.automl.AutoML.fit`.

        Returns:
            Dict: dict for `sagemaker.session.Session.auto_ml` method
        """
        config = cls._load_config(inputs, auto_ml)
        auto_ml_args = config.copy()
        auto_ml_args["job_name"] = auto_ml.current_job_name
        auto_ml_args["job_objective"] = auto_ml.job_objective
        auto_ml_args["tags"] = auto_ml.tags

        return auto_ml_args

    @classmethod
    def start_new(cls, auto_ml, inputs):
        """Create a new Amazon SageMaker AutoMLV2 job from auto_ml_v2 object.

        Args:
            auto_ml (sagemaker.automl.AutoMLV2): AutoMLV2 object
                created by the user.
            inputs (AutoMLDataChannel or list[AutoMLDataChannel]):
                Parameters used when called
                :meth:`~sagemaker.automl.AutoML.fit`.

        Returns:
            sagemaker.automl.AutoMLJobV2: Constructed object that captures
            all information about the started AutoMLV2 job.
        """
        auto_ml_args = cls._get_auto_ml_v2_args(auto_ml, inputs)

        auto_ml.sagemaker_session.create_auto_ml_v2(**auto_ml_args)
        return cls(auto_ml.sagemaker_session, auto_ml.current_job_name, inputs)

    @classmethod
    def _load_config(cls, inputs, auto_ml, expand_role=True):
        """Load job_config, input_config and output config from auto_ml and inputs.

        Args:
            inputs (AutoMLDataChannel or list[AutoMLDataChannel]): Parameters used when called
                :meth:`~sagemaker.automl.AutoML.fit`.
            auto_ml (AutoMLV2): an AutoMLV2 object that user initiated.
            expand_role (str): The expanded role arn that allows for Sagemaker
                executionts.
            validate_uri (bool): indicate whether to validate the S3 uri.

        Returns (dict): a config dictionary that contains input_config, output_config,
            problem_config and role information.

        """

        if not inputs:
            msg = (
                "Cannot format input {}. Expecting an AutoMLDataChannel or "
                "a list of AutoMLDataChannel or a LocalAutoMLDataChannel or a list of "
                "LocalAutoMLDataChannel."
            )
            raise ValueError(msg.format(inputs))

        if isinstance(inputs, AutoMLDataChannel):
            input_config = [inputs.to_request_dict()]
        elif isinstance(inputs, list) and all(
            isinstance(channel, AutoMLDataChannel) for channel in inputs
        ):
            input_config = [channel.to_request_dict() for channel in inputs]

        output_config = _Job._prepare_output_config(auto_ml.output_path, auto_ml.output_kms_key)
        role = auto_ml.sagemaker_session.expand_role(auto_ml.role) if expand_role else auto_ml.role

        problem_config = auto_ml.problem_config.to_request_dict()

        config = {
            "input_config": input_config,
            "output_config": output_config,
            "problem_config": problem_config,
            "role": role,
            "job_objective": auto_ml.job_objective,
        }

        if (
            auto_ml.volume_kms_key
            or auto_ml.vpc_config
            or auto_ml.encrypt_inter_container_traffic is not None
        ):
            config["security_config"] = {}
            if auto_ml.volume_kms_key:
                config["security_config"]["VolumeKmsKeyId"] = auto_ml.volume_kms_key
            if auto_ml.vpc_config:
                config["security_config"]["VpcConfig"] = auto_ml.vpc_config
            if auto_ml.encrypt_inter_container_traffic is not None:
                config["security_config"][
                    "EnableInterContainerTrafficEncryption"
                ] = auto_ml.encrypt_inter_container_traffic

        # Model deploy config

        auto_ml_model_deploy_config = {}
        if auto_ml.auto_generate_endpoint_name is not None:
            auto_ml_model_deploy_config["AutoGenerateEndpointName"] = (
                auto_ml.auto_generate_endpoint_name
            )
        if not auto_ml.auto_generate_endpoint_name and auto_ml.endpoint_name is not None:
            auto_ml_model_deploy_config["EndpointName"] = auto_ml.endpoint_name

        if auto_ml_model_deploy_config:
            config["model_deploy_config"] = auto_ml_model_deploy_config
        # Data split config
        if auto_ml.validation_fraction is not None:
            config["data_split_config"] = {"ValidationFraction": auto_ml.validation_fraction}
        return config

    def describe(self):
        """Returns a response from the DescribeAutoMLJobV2 API call."""
        return self.sagemaker_session.describe_auto_ml_job_v2(job_name=self.job_name)

    def wait(self, logs=True):
        """Wait for the AutoML job to finish.

        Args:
            logs (bool): indicate whether to output logs.
        """
        if logs:
            self.sagemaker_session.logs_for_auto_ml_job(job_name=self.job_name, wait=True)
        else:
            self.sagemaker_session.wait_for_auto_ml_job(job_name=self.job_name)
