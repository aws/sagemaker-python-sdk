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


@pytest.fixture(scope="module")
def training_set_5cols():
    label = (np.random.rand(100, 1) > 0.5).astype(np.int32)
    features = np.random.rand(100, 5)
    return features, label


@pytest.fixture(scope="module")
def training_set_no_label():
    features = np.random.rand(100, 2)
    return features


@pytest.fixture(scope="module")
def training_set_label_index():
    label = (np.random.rand(100, 1) > 0.5).astype(np.int32)
    features = np.random.rand(100, 2)
    index = np.arange(0, 100)  # to be used as joinsource
    return features, label, index


@pytest.fixture(scope="module")
def facet_dataset_joinsource():
    features = np.random.rand(100, 2)
    index = np.arange(0, 100)  # to be used as joinsource
    return features, index


@pytest.fixture(scope="module")
def facet_dataset():
    features = np.random.rand(100, 1)
    return features


@pytest.fixture(scope="module")
def facet_dataset_joinsource_split_1():
    features = np.random.rand(50, 2)
    index = np.arange(0, 50)  # to be used as joinsource
    return features, index


@pytest.fixture(scope="module")
def facet_dataset_joinsource_split_2():
    features = np.random.rand(50, 2)
    index = np.arange(50, 100)  # to be used as joinsource
    return features, index


@pytest.fixture(scope="module")
def pred_label_dataset():
    pred_label = (np.random.rand(100, 1) > 0.5).astype(np.int32)
    return pred_label


@pytest.yield_fixture(scope="module")
def data_path(training_set):
    features, label = training_set
    data = pd.concat([pd.DataFrame(label), pd.DataFrame(features)], axis=1, sort=False)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "train.csv")
        data.to_csv(filename, index=False, header=False)
        yield filename


@pytest.yield_fixture(scope="module")
def data_path_excl_cols(training_set_5cols):
    features, label = training_set_5cols
    data = pd.concat([pd.DataFrame(label), pd.DataFrame(features)], axis=1, sort=False)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "train.csv")
        data.to_csv(filename, index=False, header=False)
        yield filename


# training data with no label column and joinsource
@pytest.yield_fixture(scope="module")
def data_path_no_label_index(training_set_no_label):
    data = pd.DataFrame(training_set_no_label)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "train_no_label_index.csv")
        data.to_csv(filename, index=False, header=False)
        yield filename


# training data with label column & joinsource (index)
@pytest.yield_fixture(scope="module")
def data_path_label_index(training_set_label_index):
    features, label, index = training_set_label_index
    data = pd.concat(
        [pd.DataFrame(label), pd.DataFrame(features), pd.DataFrame(index)],
        axis=1,
        sort=False,
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "train_label_index.csv")
        data.to_csv(filename, index=False, header=False)
        yield filename


# training data with label column & joinsource (index)
@pytest.yield_fixture(scope="module")
def data_path_label_index_6col(training_set_label_index):
    features, label, index = training_set_label_index
    data = pd.concat(
        [
            pd.DataFrame(label),
            pd.DataFrame(features),
            pd.DataFrame(features),
            pd.DataFrame(index),
        ],
        axis=1,
        sort=False,
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "train_label_index_6col.csv")
        data.to_csv(filename, index=False, header=False)
        yield filename


@pytest.yield_fixture(scope="module")
def facet_data_path(facet_dataset_joinsource):
    features, index = facet_dataset_joinsource
    data = pd.concat([pd.DataFrame(index), pd.DataFrame(features)], axis=1, sort=False)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "facet_with_joinsource.csv")
        data.to_csv(filename, index=False, header=False)
        yield filename


# split facet dataset across 2 files
@pytest.yield_fixture(scope="module")
def facet_data_path_multiple_files(
    facet_dataset_joinsource_split_1, facet_dataset_joinsource_split_2
):
    features_1, index_1 = facet_dataset_joinsource_split_1
    data_1 = pd.concat([pd.DataFrame(index_1), pd.DataFrame(features_1)], axis=1, sort=False)
    features_2, index_2 = facet_dataset_joinsource_split_2
    data_2 = pd.concat([pd.DataFrame(index_2), pd.DataFrame(features_2)], axis=1, sort=False)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename1 = os.path.join(tmpdirname, "facet1.csv")
        data_1.to_csv(filename1, index=False, header=False)
        filename2 = os.path.join(tmpdirname, "facet2.csv")
        data_2.to_csv(filename2, index=False, header=False)
        yield filename1, filename2


@pytest.yield_fixture(scope="module")
def pred_data_path(pred_label_dataset, pred_label_headers):
    data = pd.DataFrame(pred_label_dataset, columns=pred_label_headers)
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = os.path.join(tmpdirname, "predicted_label.csv")
        data.to_csv(filename, index=False, header=pred_label_headers)
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


@pytest.fixture(scope="module")
def headers_excl_cols():
    return [
        "Label",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
    ]


@pytest.fixture(scope="module")
def headers_no_label_joinsource():
    return [
        "F3",
        "F4",
        "Index",
    ]


@pytest.fixture(scope="module")
def headers_label_joinsource():
    return [
        "Label",
        "F3",
        "F4",
        "Index",
    ]


@pytest.fixture(scope="module")
def headers_label_joinsource_6col():
    return [
        "Label",
        "F3",
        "F4",
        "F5",
        "F6",
        "Index",
    ]


@pytest.fixture(scope="module")
def facet_headers():
    return [
        "F1",
        "F2",
    ]


@pytest.fixture(scope="module")
def facet_headers_joinsource():
    return [
        "Index",
        "F1",
        "F2",
    ]


@pytest.fixture(scope="module")
def pred_label_headers():
    return ["PredictedLabel"]


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


# for testing posttraining bias with excluded columns
@pytest.fixture
def data_config_excluded_columns(sagemaker_session, data_path_excl_cols, headers_excl_cols):
    test_run = utils.unique_name_from_base("test_run")
    output_path = "s3://{}/{}/{}".format(
        sagemaker_session.default_bucket(), "linear_learner_analysis_result", test_run
    )
    return DataConfig(
        s3_data_input_path=data_path_excl_cols,
        s3_output_path=output_path,
        label="Label",
        headers=headers_excl_cols,
        dataset_type="text/csv",
        excluded_columns=["F2"],
    )


# dataset config for running analysis with facets not included in input dataset
# (with facets in multiple files), excluded columns, and no predicted_labels (so run inference)
@pytest.fixture
def data_config_facets_not_included_multiple_files(
    sagemaker_session,
    data_path_label_index_6col,
    facet_data_path_multiple_files,
    headers_label_joinsource_6col,
    facet_headers_joinsource,
):
    test_run = utils.unique_name_from_base("test_run")
    output_path = "s3://{}/{}/{}".format(
        sagemaker_session.default_bucket(), "linear_learner_analysis_result", test_run
    )
    # upload facet datasets
    facet_data_folder_s3_uri = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "linear_learner_analysis_resources",
        test_run,
        "facets_folder",
    )
    facet_data1_s3_uri = facet_data_folder_s3_uri + "/facet1.csv"
    facet_data2_s3_uri = facet_data_folder_s3_uri + "/facet2.csv"
    facet1, facet2 = facet_data_path_multiple_files
    _upload_dataset(facet1, facet_data1_s3_uri, sagemaker_session)
    _upload_dataset(facet2, facet_data2_s3_uri, sagemaker_session)

    return DataConfig(
        s3_data_input_path=data_path_label_index_6col,
        s3_output_path=output_path,
        label="Label",
        headers=headers_label_joinsource_6col,
        dataset_type="text/csv",
        joinsource="Index",
        facet_dataset_uri=facet_data_folder_s3_uri,
        facet_headers=facet_headers_joinsource,
        excluded_columns=["F4"],
    )


# for testing pretraining bias with facets not included
@pytest.fixture
def data_config_facets_not_included(
    sagemaker_session,
    data_path_label_index,
    facet_data_path,
    headers_label_joinsource,
    facet_headers_joinsource,
):
    test_run = utils.unique_name_from_base("test_run")
    output_path = "s3://{}/{}/{}".format(
        sagemaker_session.default_bucket(), "linear_learner_analysis_result", test_run
    )
    # upload facet dataset
    facet_data_s3_uri = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "linear_learner_analysis_resources",
        test_run,
        "facet_with_joinsource.csv",
    )
    _upload_dataset(facet_data_path, facet_data_s3_uri, sagemaker_session)
    return DataConfig(
        s3_data_input_path=data_path_label_index,
        s3_output_path=output_path,
        label="Label",
        headers=headers_label_joinsource,
        dataset_type="text/csv",
        joinsource="Index",
        facet_dataset_uri=facet_data_s3_uri,
        facet_headers=facet_headers_joinsource,
    )


# for testing posttraining bias with facets not included
# and separate predicted label dataset
# no excluded_columns (does not make calls to model inference API)
@pytest.fixture
def data_config_facets_not_included_pred_labels(
    sagemaker_session,
    data_path_no_label_index,
    facet_data_path,
    pred_data_path,
    headers_no_label_joinsource,
    facet_headers,
    pred_label_headers,
):
    test_run = utils.unique_name_from_base("test_run")
    output_path = "s3://{}/{}/{}".format(
        sagemaker_session.default_bucket(), "linear_learner_analysis_result", test_run
    )
    # upload facet dataset for testing
    facet_data_s3_uri = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "linear_learner_analysis_resources",
        test_run,
        "facet_with_joinsource.csv",
    )
    _upload_dataset(facet_data_path, facet_data_s3_uri, sagemaker_session)
    # upload predicted_labels dataset for testing
    pred_label_data_s3_uri = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "linear_learner_analysis_resources",
        test_run,
        "predicted_labels_with_joinsource.csv",
    )
    _upload_dataset(pred_data_path, pred_label_data_s3_uri, sagemaker_session)
    return DataConfig(
        s3_data_input_path=data_path_no_label_index,
        s3_output_path=output_path,
        headers=headers_no_label_joinsource,
        dataset_type="text/csv",
        joinsource="Index",
        facet_dataset_uri=facet_data_s3_uri,
        facet_headers=facet_headers,
        predicted_label_dataset_uri=pred_label_data_s3_uri,
        predicted_label_headers=pred_label_headers,
        predicted_label=0,
    )


@pytest.fixture
def data_config_pred_labels(
    sagemaker_session,
    pred_data_path,
    data_path,
    headers,
    pred_label_headers,
):
    test_run = utils.unique_name_from_base("test_run")
    output_path = "s3://{}/{}/{}".format(
        sagemaker_session.default_bucket(), "linear_learner_analysis_result", test_run
    )
    pred_label_data_s3_uri = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "linear_learner_analysis_resources",
        test_run,
        "predicted_labels.csv",
    )
    _upload_dataset(pred_data_path, pred_label_data_s3_uri, sagemaker_session)
    return DataConfig(
        s3_data_input_path=data_path,
        s3_output_path=output_path,
        label="Label",
        headers=headers,
        dataset_type="text/csv",
        predicted_label_dataset_uri=pred_label_data_s3_uri,
        predicted_label_headers=pred_label_headers,
        predicted_label="PredictedLabel",
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
def data_bias_config_excluded_columns():
    return BiasConfig(
        label_values_or_threshold=[1],
        facet_name="F1",
        facet_values_or_threshold=[0.5],
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


def test_pre_training_bias_facets_not_included(
    clarify_processor,
    data_config_facets_not_included,
    data_bias_config,
    sagemaker_session,
):
    with timeout.timeout(minutes=CLARIFY_DEFAULT_TIMEOUT_MINUTES):
        clarify_processor.run_pre_training_bias(
            data_config_facets_not_included,
            data_bias_config,
            job_name=utils.unique_name_from_base("clarify-pretraining-bias-facets-not-included"),
            wait=True,
        )
        analysis_result_json = s3.S3Downloader.read_file(
            data_config_facets_not_included.s3_output_path + "/analysis.json",
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
        check_analysis_config(
            data_config_facets_not_included, sagemaker_session, "pre_training_bias"
        )


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


# run posttraining bias with no predicted labels provided, so make calls to model inference API
def test_post_training_bias_facets_not_included_excluded_columns(
    clarify_processor,
    data_config_facets_not_included_multiple_files,
    data_bias_config,
    model_config,
    model_predicted_label_config,
    sagemaker_session,
):
    with timeout.timeout(minutes=CLARIFY_DEFAULT_TIMEOUT_MINUTES):
        clarify_processor.run_post_training_bias(
            data_config_facets_not_included_multiple_files,
            data_bias_config,
            model_config,
            model_predicted_label_config,
            job_name=utils.unique_name_from_base("clarify-posttraining-bias-excl-cols-facets-sep"),
            wait=True,
        )
        analysis_result_json = s3.S3Downloader.read_file(
            data_config_facets_not_included_multiple_files.s3_output_path + "/analysis.json",
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
        check_analysis_config(
            data_config_facets_not_included_multiple_files,
            sagemaker_session,
            "post_training_bias",
        )


def test_post_training_bias_excluded_columns(
    clarify_processor,
    data_config_excluded_columns,
    data_bias_config_excluded_columns,
    model_config,
    model_predicted_label_config,
    sagemaker_session,
):
    with timeout.timeout(minutes=CLARIFY_DEFAULT_TIMEOUT_MINUTES):
        clarify_processor.run_post_training_bias(
            data_config_excluded_columns,
            data_bias_config_excluded_columns,
            model_config,
            model_predicted_label_config,
            job_name=utils.unique_name_from_base("clarify-posttraining-bias-excl-cols"),
            wait=True,
        )
        analysis_result_json = s3.S3Downloader.read_file(
            data_config_excluded_columns.s3_output_path + "/analysis.json",
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
        check_analysis_config(data_config_excluded_columns, sagemaker_session, "post_training_bias")


def test_post_training_bias_predicted_labels(
    clarify_processor,
    data_config_pred_labels,
    data_bias_config,
    model_predicted_label_config,
    sagemaker_session,
):
    model_config = None
    with timeout.timeout(minutes=CLARIFY_DEFAULT_TIMEOUT_MINUTES):
        clarify_processor.run_post_training_bias(
            data_config_pred_labels,
            data_bias_config,
            model_config,
            model_predicted_label_config,
            job_name=utils.unique_name_from_base("clarify-posttraining-bias-pred-labels"),
            wait=True,
        )
        analysis_result_json = s3.S3Downloader.read_file(
            data_config_pred_labels.s3_output_path + "/analysis.json",
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
        check_analysis_config(data_config_pred_labels, sagemaker_session, "post_training_bias")


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


def test_bias_and_explainability(
    clarify_processor,
    data_config,
    model_config,
    shap_config,
    data_bias_config,
    sagemaker_session,
):
    with timeout.timeout(minutes=CLARIFY_DEFAULT_TIMEOUT_MINUTES):
        clarify_processor.run_bias_and_explainability(
            data_config,
            model_config,
            shap_config,
            data_bias_config,
            pre_training_methods="all",
            post_training_methods="all",
            model_predicted_label_config="score",
            job_name=utils.unique_name_from_base("clarify-bias-and-explainability"),
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

        assert (
            math.fabs(
                analysis_result["post_training_bias_metrics"]["facets"]["F1"][0]["metrics"][0][
                    "value"
                ]
            )
            <= 1.0
        )
        check_analysis_config(data_config, sagemaker_session, "post_training_bias")


def check_analysis_config(data_config, sagemaker_session, method):
    analysis_config_json = s3.S3Downloader.read_file(
        data_config.s3_output_path + "/analysis_config.json",
        sagemaker_session,
    )
    analysis_config = json.loads(analysis_config_json)
    assert method in analysis_config["methods"]


def _upload_dataset(dataset_local_path, s3_dataset_path, sagemaker_session):
    """Upload dataset (intended for facet or predicted labels dataset, not training dataset) to S3

    Args:
        dataset_local_path (str): File path to the local analysis config file.
        s3_dataset_path (str): S3 prefix to store the analysis config file.
        sagemaker_session (:class:`~sagemaker.session.Session`):
            Session object which manages interactions with Amazon SageMaker and
            any other AWS services needed. If not specified, the processor creates
            one using the default AWS configuration chain.

    Returns:
        The S3 uri of the uploaded dataset.
    """
    return s3.S3Uploader.upload(
        local_path=dataset_local_path,
        desired_s3_uri=s3_dataset_path,
        sagemaker_session=sagemaker_session,
    )
