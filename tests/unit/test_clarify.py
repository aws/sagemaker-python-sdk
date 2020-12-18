# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from mock import patch, Mock, MagicMock
import pytest

from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
)
from sagemaker import image_uris


def test_uri():
    uri = image_uris.retrieve("clarify", "us-west-2")
    assert "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0" == uri


def test_data_config():
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    label_name = "Label"
    headers = [
        "Label",
        "F1",
        "F2",
        "F3",
        "F4",
    ]
    dataset_type = "text/csv"
    data_config = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=s3_output_path,
        label=label_name,
        headers=headers,
        dataset_type=dataset_type,
    )

    expected_config = {
        "dataset_type": "text/csv",
        "headers": headers,
        "label": "Label",
    }
    assert expected_config == data_config.get_config()
    assert s3_data_input_path == data_config.s3_data_input_path
    assert s3_output_path == data_config.s3_output_path
    assert "None" == data_config.s3_compression_type
    assert "FullyReplicated" == data_config.s3_data_distribution_type


def test_data_bias_config():
    label_values = [1]
    facet_name = "F1"
    facet_threshold = 0.3
    group_name = "A151"

    data_bias_config = BiasConfig(
        label_values_or_threshold=label_values,
        facet_name=facet_name,
        facet_values_or_threshold=facet_threshold,
        group_name=group_name,
    )

    expected_config = {
        "label_values_or_threshold": label_values,
        "facet": [{"name_or_index": facet_name, "value_or_threshold": facet_threshold}],
        "group_variable": group_name,
    }
    assert expected_config == data_bias_config.get_config()


def test_model_config():
    model_name = "xgboost-model"
    instance_type = "ml.c5.xlarge"
    instance_count = 1
    accept_type = "text/csv"
    content_type = "application/jsonlines"
    model_config = ModelConfig(
        model_name=model_name,
        instance_type=instance_type,
        instance_count=instance_count,
        accept_type=accept_type,
        content_type=content_type,
    )
    expected_config = {
        "model_name": model_name,
        "instance_type": instance_type,
        "initial_instance_count": instance_count,
        "accept_type": accept_type,
        "content_type": content_type,
    }
    assert expected_config == model_config.get_predictor_config()


def test_invalid_model_config():
    with pytest.raises(ValueError) as error:
        ModelConfig(
            model_name="xgboost-model",
            instance_type="ml.c5.xlarge",
            instance_count=1,
            accept_type="invalid_accept_type",
        )
    assert (
        "Invalid accept_type invalid_accept_type. Please choose text/csv or application/jsonlines."
        in str(error.value)
    )


def test_model_predicted_label_config():
    label = "label"
    probability = "pr"
    probability_threshold = 0.2
    label_headers = ["success"]
    model_config = ModelPredictedLabelConfig(
        label=label,
        probability=probability,
        probability_threshold=probability_threshold,
        label_headers=label_headers,
    )
    pr_threshold, config = model_config.get_predictor_config()
    expected_config = {
        "label": label,
        "probability": probability,
        "label_headers": label_headers,
    }
    assert probability_threshold == pr_threshold
    assert expected_config == config


def test_invalid_model_predicted_label_config():
    with pytest.raises(TypeError) as error:
        ModelPredictedLabelConfig(
            probability_threshold="invalid",
        )
    assert (
        "Invalid probability_threshold invalid. Please choose one that can be cast to float."
        in str(error.value)
    )


def test_shap_config():
    baseline = [
        [
            0.26124998927116394,
            0.2824999988079071,
            0.06875000149011612,
        ]
    ]
    num_samples = 100
    agg_method = "mean_sq"
    use_logit = True
    shap_config = SHAPConfig(
        baseline=baseline,
        num_samples=num_samples,
        agg_method=agg_method,
        use_logit=use_logit,
    )
    expected_config = {
        "shap": {
            "baseline": baseline,
            "num_samples": num_samples,
            "agg_method": agg_method,
            "use_logit": use_logit,
            "save_local_shap_values": True,
        }
    }
    assert expected_config == shap_config.get_explainability_config()


def test_invalid_shap_config():
    with pytest.raises(ValueError) as error:
        SHAPConfig(
            baseline=[[1]],
            num_samples=1,
            agg_method="invalid",
        )
    assert "Invalid agg_method invalid. Please choose mean_abs, median, or mean_sq." in str(
        error.value
    )


@pytest.fixture(scope="module")
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name="us-west-2")
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name="us-west-2",
        config=None,
        local_mode=False,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value="mybucket")
    session_mock.upload_data = Mock(
        name="upload_data", return_value="mocked_s3_uri_from_upload_data"
    )
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = "arn:aws:iam::012345678901:role/SageMakerRole"
    return session_mock


@pytest.fixture(scope="module")
def clarify_processor(sagemaker_session):
    return SageMakerClarifyProcessor(
        role="AmazonSageMaker-ExecutionRole",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture(scope="module")
def data_config():
    return DataConfig(
        s3_data_input_path="s3://input/train.csv",
        s3_output_path="s3://output/analysis_test_result",
        label="Label",
        headers=[
            "Label",
            "F1",
            "F2",
            "F3",
        ],
        dataset_type="text/csv",
    )


@pytest.fixture(scope="module")
def data_bias_config():
    return BiasConfig(
        label_values_or_threshold=[1],
        facet_name="F1",
        group_name="F2",
    )


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(
        model_name="xgboost-model",
        instance_type="ml.c5.xlarge",
        instance_count=1,
    )


@pytest.fixture(scope="module")
def model_predicted_label_config():
    return ModelPredictedLabelConfig()


@pytest.fixture(scope="module")
def shap_config():
    return SHAPConfig(
        baseline=[
            [
                0.26124998927116394,
                0.2824999988079071,
                0.06875000149011612,
            ]
        ],
        num_samples=100,
        agg_method="mean_sq",
    )


def test_pre_training_bias(clarify_processor, data_config, data_bias_config):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        clarify_processor.run_pre_training_bias(
            data_config, data_bias_config, wait=True, job_name="test"
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
            ],
            "label": "Label",
            "label_values_or_threshold": [1],
            "facet": [{"name_or_index": "F1"}],
            "group_variable": "F2",
            "methods": {"pre_training_bias": {"methods": "all"}},
        }
        mock_method.assert_called_once_with(
            data_config, expected_analysis_config, True, True, "test", None
        )


def test_post_training_bias(
    clarify_processor, data_config, data_bias_config, model_config, model_predicted_label_config
):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        clarify_processor.run_post_training_bias(
            data_config,
            data_bias_config,
            model_config,
            model_predicted_label_config,
            wait=True,
            job_name="test",
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
            ],
            "label": "Label",
            "label_values_or_threshold": [1],
            "facet": [{"name_or_index": "F1"}],
            "group_variable": "F2",
            "methods": {"post_training_bias": {"methods": "all"}},
            "predictor": {
                "model_name": "xgboost-model",
                "instance_type": "ml.c5.xlarge",
                "initial_instance_count": 1,
            },
        }
        mock_method.assert_called_once_with(
            data_config, expected_analysis_config, True, True, "test", None
        )


def test_shap(clarify_processor, data_config, model_config, shap_config):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        clarify_processor.run_explainability(
            data_config, model_config, shap_config, model_scores=None, wait=True, job_name="test"
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
            ],
            "label": "Label",
            "methods": {
                "shap": {
                    "baseline": [
                        [
                            0.26124998927116394,
                            0.2824999988079071,
                            0.06875000149011612,
                        ]
                    ],
                    "num_samples": 100,
                    "agg_method": "mean_sq",
                    "use_logit": False,
                    "save_local_shap_values": True,
                }
            },
            "predictor": {
                "model_name": "xgboost-model",
                "instance_type": "ml.c5.xlarge",
                "initial_instance_count": 1,
            },
        }
        mock_method.assert_called_once_with(
            data_config, expected_analysis_config, True, True, "test", None
        )
