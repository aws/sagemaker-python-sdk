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

import os
import time

from sagemaker import (
    AutoMLV2,
    AutoMLTabularConfig,
    AutoMLImageClassificationConfig,
    AutoMLTextClassificationConfig,
    AutoMLTextGenerationConfig,
    AutoMLTimeSeriesForecastingConfig,
    AutoMLDataChannel,
)
from tests.integ import DATA_DIR, AUTO_ML_V2_DEFAULT_WAITING_TIME_MINUTES

ROLE = "SageMakerRole"
DATA_DIR = os.path.join(DATA_DIR, "automl", "data")
PREFIX = "sagemaker/beta-automl-v2"

# Problem types
TABULAR_PROBLEM_TYPE = "Tabular"
IMAGE_CLASSIFICATION_PROBLEM_TYPE = "ImageClassification"
TEXT_CLASSIFICATION_PROBLEM_TYPE = "TextClassification"
TEXT_GENERATION_PROBLEM_TYPE = "TextGeneration"
TIME_SERIES_FORECASTING_PROBLEM_TYPE = "TimeSeriesForecasting"

PROBLEM_TYPES = [
    TABULAR_PROBLEM_TYPE,
    IMAGE_CLASSIFICATION_PROBLEM_TYPE,
    TEXT_CLASSIFICATION_PROBLEM_TYPE,
    TEXT_GENERATION_PROBLEM_TYPE,
    TIME_SERIES_FORECASTING_PROBLEM_TYPE,
]


PROBLEM_CONFIGS = {
    TABULAR_PROBLEM_TYPE: AutoMLTabularConfig(
        target_attribute_name="virginica",
        max_candidates=3,
        generate_candidate_definitions_only=False,
    ),
    IMAGE_CLASSIFICATION_PROBLEM_TYPE: AutoMLImageClassificationConfig(
        max_candidates=1,
    ),
    TEXT_CLASSIFICATION_PROBLEM_TYPE: AutoMLTextClassificationConfig(
        content_column="text",
        target_label_column="label",
        max_candidates=1,
    ),
    TEXT_GENERATION_PROBLEM_TYPE: AutoMLTextGenerationConfig(
        max_candidates=1,
    ),
    TIME_SERIES_FORECASTING_PROBLEM_TYPE: AutoMLTimeSeriesForecastingConfig(
        forecast_frequency="1H",
        forecast_horizon=1,
        item_identifier_attribute_name="item_id",
        target_attribute_name="target",
        timestamp_attribute_name="timestamp",
        forecast_quantiles=["p10", "p50", "p90"],
    ),
}

DATA_CONFIGS = {
    TABULAR_PROBLEM_TYPE: {
        "path": os.path.join(DATA_DIR, "iris_training.csv"),
        "prefix": PREFIX + "/input",
        "content_type": "text/csv;header=present",
    },
    IMAGE_CLASSIFICATION_PROBLEM_TYPE: {
        "path": os.path.join(DATA_DIR, "cifar10_subset"),
        "prefix": PREFIX + "/input/cifar10_subset",
        "content_type": "image/png",
    },
    TEXT_CLASSIFICATION_PROBLEM_TYPE: {
        "path": os.path.join(DATA_DIR, "CoLA.csv"),
        "prefix": PREFIX + "/input",
        "content_type": "text/csv;header=present",
    },
    TEXT_GENERATION_PROBLEM_TYPE: {
        "path": os.path.join(DATA_DIR, "customer_support.csv"),
        "prefix": PREFIX + "/input",
        "content_type": "text/csv;header=present",
    },
    TIME_SERIES_FORECASTING_PROBLEM_TYPE: {
        "path": os.path.join(DATA_DIR, "sample_time_series.csv"),
        "prefix": PREFIX + "/input",
        "content_type": "text/csv;header=present",
    },
}


def create_auto_ml_job_v2_if_not_exist(sagemaker_session, auto_ml_job_name, problem_type):
    try:
        sagemaker_session.describe_auto_ml_job_v2(job_name=auto_ml_job_name)
    except Exception as e:  # noqa: F841
        auto_ml = AutoMLV2(
            base_job_name="automl_v2_test",
            role=ROLE,
            problem_config=PROBLEM_CONFIGS[problem_type],
            sagemaker_session=sagemaker_session,
        )
        s3_uri = sagemaker_session.upload_data(
            path=DATA_CONFIGS[problem_type]["path"], key_prefix=DATA_CONFIGS[problem_type]["prefix"]
        )
        inputs = [
            AutoMLDataChannel(
                s3_data_type="S3Prefix",
                s3_uri=s3_uri
                if DATA_CONFIGS[problem_type]["path"] != os.path.join(DATA_DIR, "cifar10_subset")
                else s3_uri + "/",
                channel_type="training",
                content_type=DATA_CONFIGS[problem_type]["content_type"],
            )
        ]
        auto_ml.fit(inputs, job_name=auto_ml_job_name, wait=False)
        time.sleep(AUTO_ML_V2_DEFAULT_WAITING_TIME_MINUTES * 60)
