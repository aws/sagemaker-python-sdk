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
from __future__ import absolute_import

from sagemaker import (
    AutoMLTabularConfig,
    AutoMLImageClassificationConfig,
    AutoMLTextGenerationConfig,
    AutoMLTextClassificationConfig,
    AutoMLTimeSeriesForecastingConfig,
)

# Common params
MAX_CANDIDATES = 10
MAX_RUNTIME_PER_TRAINING_JOB = 3600
TOTAL_JOB_RUNTIME = 36000
BUCKET_NAME = "mybucket"
FEATURE_SPECIFICATION_S3_URI = "s3://{}/features.json".format(BUCKET_NAME)

# Tabular params
AUTO_ML_TABULAR_ALGORITHMS = "xgboost"
MODE = "ENSEMBLING"
GENERATE_CANDIDATE_DEFINITIONS_ONLY = True
PROBLEM_TYPE = "BinaryClassification"
TARGET_ATTRIBUTE_NAME = "target"
SAMPLE_WEIGHT_ATTRIBUTE_NAME = "sampleWeight"

TABULAR_PROBLEM_CONFIG = {
    "CompletionCriteria": {
        "MaxCandidates": MAX_CANDIDATES,
        "MaxRuntimePerTrainingJobInSeconds": MAX_RUNTIME_PER_TRAINING_JOB,
        "MaxAutoMLJobRuntimeInSeconds": TOTAL_JOB_RUNTIME,
    },
    "CandidateGenerationConfig": {
        "AlgorithmsConfig": [{"AutoMLAlgorithms": AUTO_ML_TABULAR_ALGORITHMS}],
    },
    "FeatureSpecificationS3Uri": FEATURE_SPECIFICATION_S3_URI,
    "Mode": MODE,
    "GenerateCandidateDefinitionsOnly": GENERATE_CANDIDATE_DEFINITIONS_ONLY,
    "ProblemType": PROBLEM_TYPE,
    "TargetAttributeName": TARGET_ATTRIBUTE_NAME,
    "SampleWeightAttributeName": SAMPLE_WEIGHT_ATTRIBUTE_NAME,
}

# Image classification params

IMAGE_CLASSIFICATION_PROBLEM_CONFIG = {
    "CompletionCriteria": {
        "MaxCandidates": MAX_CANDIDATES,
        "MaxRuntimePerTrainingJobInSeconds": MAX_RUNTIME_PER_TRAINING_JOB,
        "MaxAutoMLJobRuntimeInSeconds": TOTAL_JOB_RUNTIME,
    },
}

# Text classification
CONTEXT_COLUMN = "text"
TARGET_LABEL_COLUMN = "class"

TEXT_CLASSIFICATION_PROBLEM_CONFIG = {
    "CompletionCriteria": {
        "MaxCandidates": MAX_CANDIDATES,
        "MaxRuntimePerTrainingJobInSeconds": MAX_RUNTIME_PER_TRAINING_JOB,
        "MaxAutoMLJobRuntimeInSeconds": TOTAL_JOB_RUNTIME,
    },
    "ContentColumn": CONTEXT_COLUMN,
    "TargetLabelColumn": TARGET_LABEL_COLUMN,
}

# Text generation params
BASE_MODEL_NAME = "base_model"
TEXT_GENERATION_HYPER_PARAMS = {"test": 1}
ACCEPT_EULA = True

TEXT_GENERATION_PROBLEM_CONFIG = {
    "CompletionCriteria": {
        "MaxCandidates": MAX_CANDIDATES,
        "MaxRuntimePerTrainingJobInSeconds": MAX_RUNTIME_PER_TRAINING_JOB,
        "MaxAutoMLJobRuntimeInSeconds": TOTAL_JOB_RUNTIME,
    },
    "BaseModelName": BASE_MODEL_NAME,
    "TextGenerationHyperParameters": TEXT_GENERATION_HYPER_PARAMS,
    "ModelAccessConfig": {
        "AcceptEula": ACCEPT_EULA,
    },
}

# Time series forecasting params
FORECAST_FREQUENCY = "1D"
FORECAST_HORIZON = 5
ITEM_IDENTIFIER_ATTRIBUTE_NAME = "identifier_attribute"
TIMESTAMP_ATTRIBUTE_NAME = "timestamp_attribute"
FORECAST_QUANTILES = ["p1"]
HOLIDAY_CONFIG = "DE"


TIME_SERIES_FORECASTING_PROBLEM_CONFIG = {
    "CompletionCriteria": {
        "MaxCandidates": MAX_CANDIDATES,
        "MaxRuntimePerTrainingJobInSeconds": MAX_RUNTIME_PER_TRAINING_JOB,
        "MaxAutoMLJobRuntimeInSeconds": TOTAL_JOB_RUNTIME,
    },
    "FeatureSpecificationS3Uri": FEATURE_SPECIFICATION_S3_URI,
    "ForecastFrequency": FORECAST_FREQUENCY,
    "ForecastHorizon": FORECAST_HORIZON,
    "TimeSeriesConfig": {
        "ItemIdentifierAttributeName": ITEM_IDENTIFIER_ATTRIBUTE_NAME,
        "TargetAttributeName": TARGET_ATTRIBUTE_NAME,
        "TimestampAttributeName": TIMESTAMP_ATTRIBUTE_NAME,
    },
    "ForecastQuantiles": FORECAST_QUANTILES,
    "HolidayConfig": [
        {
            "CountryCode": HOLIDAY_CONFIG,
        }
    ],
}


def test_tabular_problem_config_from_response():
    problem_config = AutoMLTabularConfig.from_response_dict(TABULAR_PROBLEM_CONFIG)
    assert problem_config.algorithms_config == AUTO_ML_TABULAR_ALGORITHMS
    assert problem_config.feature_specification_s3_uri == FEATURE_SPECIFICATION_S3_URI
    assert problem_config.generate_candidate_definitions_only == GENERATE_CANDIDATE_DEFINITIONS_ONLY
    assert problem_config.max_candidates == MAX_CANDIDATES
    assert problem_config.max_runtime_per_training_job_in_seconds == MAX_RUNTIME_PER_TRAINING_JOB
    assert problem_config.max_total_job_runtime_in_seconds == TOTAL_JOB_RUNTIME
    assert problem_config.mode == MODE
    assert problem_config.problem_type == PROBLEM_TYPE
    assert problem_config.sample_weight_attribute_name == SAMPLE_WEIGHT_ATTRIBUTE_NAME
    assert problem_config.target_attribute_name == TARGET_ATTRIBUTE_NAME


def test_tabular_problem_config_to_request():
    problem_config = AutoMLTabularConfig(
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        algorithms_config=AUTO_ML_TABULAR_ALGORITHMS,
        feature_specification_s3_uri=FEATURE_SPECIFICATION_S3_URI,
        generate_candidate_definitions_only=GENERATE_CANDIDATE_DEFINITIONS_ONLY,
        mode=MODE,
        problem_type=PROBLEM_TYPE,
        sample_weight_attribute_name=SAMPLE_WEIGHT_ATTRIBUTE_NAME,
        max_candidates=MAX_CANDIDATES,
        max_total_job_runtime_in_seconds=TOTAL_JOB_RUNTIME,
        max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_TRAINING_JOB,
    )

    assert problem_config.to_request_dict()["TabularJobConfig"] == TABULAR_PROBLEM_CONFIG


def test_image_classification_problem_config_from_response():
    problem_config = AutoMLImageClassificationConfig.from_response_dict(
        IMAGE_CLASSIFICATION_PROBLEM_CONFIG
    )
    assert problem_config.max_candidates == MAX_CANDIDATES
    assert problem_config.max_runtime_per_training_job_in_seconds == MAX_RUNTIME_PER_TRAINING_JOB
    assert problem_config.max_total_job_runtime_in_seconds == TOTAL_JOB_RUNTIME


def test_image_classification_problem_config_to_request():
    problem_config = AutoMLImageClassificationConfig(
        max_candidates=MAX_CANDIDATES,
        max_total_job_runtime_in_seconds=TOTAL_JOB_RUNTIME,
        max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_TRAINING_JOB,
    )

    assert (
        problem_config.to_request_dict()["ImageClassificationJobConfig"]
        == IMAGE_CLASSIFICATION_PROBLEM_CONFIG
    )


def test_text_classification_problem_config_from_response():
    problem_config = AutoMLTextClassificationConfig.from_response_dict(
        TEXT_CLASSIFICATION_PROBLEM_CONFIG
    )
    assert problem_config.content_column == CONTEXT_COLUMN
    assert problem_config.target_label_column == TARGET_LABEL_COLUMN
    assert problem_config.max_candidates == MAX_CANDIDATES
    assert problem_config.max_runtime_per_training_job_in_seconds == MAX_RUNTIME_PER_TRAINING_JOB
    assert problem_config.max_total_job_runtime_in_seconds == TOTAL_JOB_RUNTIME


def test_text_classification_to_request():
    problem_config = AutoMLTextClassificationConfig(
        content_column=CONTEXT_COLUMN,
        target_label_column=TARGET_LABEL_COLUMN,
        max_candidates=MAX_CANDIDATES,
        max_total_job_runtime_in_seconds=TOTAL_JOB_RUNTIME,
        max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_TRAINING_JOB,
    )

    assert (
        problem_config.to_request_dict()["TextClassificationJobConfig"]
        == TEXT_CLASSIFICATION_PROBLEM_CONFIG
    )


def test_text_generation_problem_config_from_response():
    problem_config = AutoMLTextGenerationConfig.from_response_dict(TEXT_GENERATION_PROBLEM_CONFIG)
    assert problem_config.accept_eula == ACCEPT_EULA
    assert problem_config.base_model_name == BASE_MODEL_NAME
    assert problem_config.max_candidates == MAX_CANDIDATES
    assert problem_config.max_runtime_per_training_job_in_seconds == MAX_RUNTIME_PER_TRAINING_JOB
    assert problem_config.max_total_job_runtime_in_seconds == TOTAL_JOB_RUNTIME
    assert problem_config.text_generation_hyper_params == TEXT_GENERATION_HYPER_PARAMS


def test_text_generation_problem_config_to_request():
    problem_config = AutoMLTextGenerationConfig(
        accept_eula=ACCEPT_EULA,
        base_model_name=BASE_MODEL_NAME,
        text_generation_hyper_params=TEXT_GENERATION_HYPER_PARAMS,
        max_candidates=MAX_CANDIDATES,
        max_total_job_runtime_in_seconds=TOTAL_JOB_RUNTIME,
        max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_TRAINING_JOB,
    )

    assert (
        problem_config.to_request_dict()["TextGenerationJobConfig"]
        == TEXT_GENERATION_PROBLEM_CONFIG
    )


def test_time_series_forecasting_problem_config_from_response():
    problem_config = AutoMLTimeSeriesForecastingConfig.from_response_dict(
        TIME_SERIES_FORECASTING_PROBLEM_CONFIG
    )
    assert problem_config.forecast_frequency == FORECAST_FREQUENCY
    assert problem_config.forecast_horizon == FORECAST_HORIZON
    assert problem_config.item_identifier_attribute_name == ITEM_IDENTIFIER_ATTRIBUTE_NAME
    assert problem_config.target_attribute_name == TARGET_ATTRIBUTE_NAME
    assert problem_config.timestamp_attribute_name == TIMESTAMP_ATTRIBUTE_NAME
    assert problem_config.max_candidates == MAX_CANDIDATES
    assert problem_config.max_runtime_per_training_job_in_seconds == MAX_RUNTIME_PER_TRAINING_JOB
    assert problem_config.max_total_job_runtime_in_seconds == TOTAL_JOB_RUNTIME
    assert problem_config.forecast_quantiles == FORECAST_QUANTILES
    assert problem_config.holiday_config == HOLIDAY_CONFIG
    assert problem_config.feature_specification_s3_uri == FEATURE_SPECIFICATION_S3_URI


def test_time_series_forecasting_problem_config_to_request():
    problem_config = AutoMLTimeSeriesForecastingConfig(
        forecast_frequency=FORECAST_FREQUENCY,
        forecast_horizon=FORECAST_HORIZON,
        item_identifier_attribute_name=ITEM_IDENTIFIER_ATTRIBUTE_NAME,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        timestamp_attribute_name=TIMESTAMP_ATTRIBUTE_NAME,
        forecast_quantiles=FORECAST_QUANTILES,
        holiday_config=HOLIDAY_CONFIG,
        feature_specification_s3_uri=FEATURE_SPECIFICATION_S3_URI,
        max_candidates=MAX_CANDIDATES,
        max_total_job_runtime_in_seconds=TOTAL_JOB_RUNTIME,
        max_runtime_per_training_job_in_seconds=MAX_RUNTIME_PER_TRAINING_JOB,
    )

    assert (
        problem_config.to_request_dict()["TimeSeriesForecastingJobConfig"]
        == TIME_SERIES_FORECASTING_PROBLEM_CONFIG
    )
