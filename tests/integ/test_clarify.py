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

from __future__ import print_function, absolute_import


import json
import math
import numpy as np
import os
import pandas as pd
import pytest
import statistics
import tempfile

from sagemaker import s3
from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
)

from sagemaker.amazon.linear_learner import LinearLearner, LinearLearnerPredictor
from sagemaker import utils
from tests import integ
from tests.integ import timeout


CLARIFY_DEFAULT_TIMEOUT_MINUTES = 15


@pytest.fixture(scope="module")
def training_set():
    label = (np.random.rand(100, 1) > 0.5).astype(np.int32)
    features = np.random.rand(100, 4)
    return features, label


@pytest.yield_fixture(scope="module")
def data_path(training_set):
    features, label = training_set
    data = pd.concat([pd.DataFrame(label), pd.DataFrame(features)], axis=1, sort=False)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "train.csv")
        data.to_csv(filename, index=False, header=False)
        yield filename


@pytest.fixture(scope="module")
def headers():
    return [
        "Label",
        "F1",
        "F2",
        "F3",
        "F4",
    ]


@pytest.yield_fixture(scope="module")
def model_name(sagemaker_session, cpu_instance_type, training_set):
    job_name = utils.unique_name_from_base("clarify-xgb")

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        ll = LinearLearner(
            "SageMakerRole",
            1,
            cpu_instance_type,
            predictor_type="binary_classifier",
            sagemaker_session=sagemaker_session,
            disable_profiler=True,
        )
        ll.binary_classifier_model_selection_criteria = "accuracy"
        ll.early_stopping_tolerance = 0.0001
        ll.early_stopping_patience = 3
        ll.num_models = 1
        ll.epochs = 1
        ll.num_calibration_samples = 1

        features, label = training_set
        ll.fit(
            ll.record_set(features.astype(np.float32), label.reshape(-1).astype(np.float32)),
            job_name=job_name,
        )

    with timeout.timeout_and_delete_endpoint_by_name(job_name, sagemaker_session):
        ll.deploy(1, cpu_instance_type, endpoint_name=job_name, model_name=job_name, wait=True)
        yield job_name


@pytest.fixture(scope="module")
def clarify_processor(sagemaker_session, cpu_instance_type):
    processor = SageMakerClarifyProcessor(
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )
    return processor


@pytest.fixture
def data_config(sagemaker_session, data_path, headers):
    test_run = utils.unique_name_from_base("test_run")
    output_path = "s3://{}/{}/{}".format(
        sagemaker_session.default_bucket(), "linear_learner_analysis_result", test_run
    )
    return DataConfig(
        s3_data_input_path=data_path,
        s3_output_path=output_path,
        label="Label",
        headers=headers,
        dataset_type="text/csv",
    )


@pytest.fixture(scope="module")
def data_bias_config():
    return BiasConfig(
        label_values_or_threshold=[1],
        facet_name="F1",
        facet_values_or_threshold=[0.5],
        group_name="F2",
    )


@pytest.fixture(scope="module")
def model_config(model_name):
    return ModelConfig(
        model_name=model_name,
        instance_type="ml.c5.xlarge",
        instance_count=1,
        accept_type="application/jsonlines",
        endpoint_name_prefix="myprefix",
    )


@pytest.fixture(scope="module")
def model_predicted_label_config(sagemaker_session, model_name, training_set):
    predictor = LinearLearnerPredictor(
        model_name,
        sagemaker_session=sagemaker_session,
    )
    result = predictor.predict(training_set[0].astype(np.float32))
    predictions = [float(record.label["score"].float32_tensor.values[0]) for record in result]
    probability_threshold = statistics.median(predictions)
    return ModelPredictedLabelConfig(label="score", probability_threshold=probability_threshold)


@pytest.fixture(scope="module")
def shap_config():
    return SHAPConfig(
        baseline=[
            [
                0.94672389,
                0.47108862,
                0.63350081,
                0.00604642,
            ]
        ],
        num_samples=2,
        agg_method="mean_sq",
        seed=123,
    )


def test_pre_training_bias(clarify_processor, data_config, data_bias_config, sagemaker_session):
    with timeout.timeout(minutes=CLARIFY_DEFAULT_TIMEOUT_MINUTES):
        clarify_processor.run_pre_training_bias(
            data_config,
            data_bias_config,
            job_name=utils.unique_name_from_base("clarify-pretraining-bias"),
            wait=True,
        )
        analysis_result_json = s3.S3Downloader.read_file(
            data_config.s3_output_path + "/analysis.json",
            sagemaker_session,
        )
        analysis_result = json.loads(analysis_result_json)
        assert (
            math.fabs(
                analysis_result["pre_training_bias_metrics"]["facets"]["F1"][0]["metrics"][0][
                    "value"
                ]
            )
            <= 1.0
        )
        check_analysis_config(data_config, sagemaker_session, "pre_training_bias")


def test_post_training_bias(
    clarify_processor,
    data_config,
    data_bias_config,
    model_config,
    model_predicted_label_config,
    sagemaker_session,
):
    with timeout.timeout(minutes=CLARIFY_DEFAULT_TIMEOUT_MINUTES):
        clarify_processor.run_post_training_bias(
            data_config,
            data_bias_config,
            model_config,
            model_predicted_label_config,
            job_name=utils.unique_name_from_base("clarify-posttraining-bias"),
            wait=True,
        )
        analysis_result_json = s3.S3Downloader.read_file(
            data_config.s3_output_path + "/analysis.json",
            sagemaker_session,
        )
        analysis_result = json.loads(analysis_result_json)
        assert (
            math.fabs(
                analysis_result["post_training_bias_metrics"]["facets"]["F1"][0]["metrics"][0][
                    "value"
                ]
            )
            <= 1.0
        )
        check_analysis_config(data_config, sagemaker_session, "post_training_bias")


def test_shap(clarify_processor, data_config, model_config, shap_config, sagemaker_session):
    with timeout.timeout(minutes=CLARIFY_DEFAULT_TIMEOUT_MINUTES):
        clarify_processor.run_explainability(
            data_config,
            model_config,
            shap_config,
            model_scores="score",
            job_name=utils.unique_name_from_base("clarify-explainability"),
            wait=True,
        )
        analysis_result_json = s3.S3Downloader.read_file(
            data_config.s3_output_path + "/analysis.json",
            sagemaker_session,
        )
        analysis_result = json.loads(analysis_result_json)
        assert (
            math.fabs(
                analysis_result["explanations"]["kernel_shap"]["label0"]["global_shap_values"]["F2"]
            )
            <= 1
        )
        check_analysis_config(data_config, sagemaker_session, "shap")


def check_analysis_config(data_config, sagemaker_session, method):
    analysis_config_json = s3.S3Downloader.read_file(
        data_config.s3_output_path + "/analysis_config.json",
        sagemaker_session,
    )
    analysis_config = json.loads(analysis_config_json)
    assert method in analysis_config["methods"]
