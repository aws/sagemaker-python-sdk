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

import copy

from mock import patch, Mock, MagicMock
import pytest

from sagemaker.clarify import (
    SageMakerClarifyProcessor,
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
    PDPConfig,
)
from sagemaker import image_uris, Processor

JOB_NAME_PREFIX = "my-prefix"
TIMESTAMP = "2021-06-17-22-29-54-685"
JOB_NAME = "{}-{}".format(JOB_NAME_PREFIX, TIMESTAMP)


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


def test_invalid_data_config():
    with pytest.raises(ValueError, match=r"^Invalid dataset_type"):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="whatnot_type",
        )


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


def test_data_bias_config_multi_facet():
    label_values = [1]
    facet_name = ["Facet1", "Facet2"]
    facet_threshold = [[0], [1, 2]]
    group_name = "A151"

    data_bias_config = BiasConfig(
        label_values_or_threshold=label_values,
        facet_name=facet_name,
        facet_values_or_threshold=facet_threshold,
        group_name=group_name,
    )

    expected_config = {
        "label_values_or_threshold": label_values,
        "facet": [
            {"name_or_index": facet_name[0], "value_or_threshold": facet_threshold[0]},
            {"name_or_index": facet_name[1], "value_or_threshold": facet_threshold[1]},
        ],
        "group_variable": group_name,
    }
    assert expected_config == data_bias_config.get_config()


def test_data_bias_config_multi_facet_not_all_with_value():
    label_values = [1]
    facet_name = ["Facet1", "Facet2"]
    facet_threshold = [[0], None]
    group_name = "A151"

    data_bias_config = BiasConfig(
        label_values_or_threshold=label_values,
        facet_name=facet_name,
        facet_values_or_threshold=facet_threshold,
        group_name=group_name,
    )

    expected_config = {
        "label_values_or_threshold": label_values,
        "facet": [
            {"name_or_index": facet_name[0], "value_or_threshold": facet_threshold[0]},
            {"name_or_index": facet_name[1]},
        ],
        "group_variable": group_name,
    }
    assert expected_config == data_bias_config.get_config()


def test_model_config():
    model_name = "xgboost-model"
    instance_type = "ml.c5.xlarge"
    instance_count = 1
    accept_type = "text/csv"
    content_type = "application/jsonlines"
    custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"
    accelerator_type = "ml.eia1.medium"
    model_config = ModelConfig(
        model_name=model_name,
        instance_type=instance_type,
        instance_count=instance_count,
        accept_type=accept_type,
        content_type=content_type,
        custom_attributes=custom_attributes,
        accelerator_type=accelerator_type,
    )
    expected_config = {
        "model_name": model_name,
        "instance_type": instance_type,
        "initial_instance_count": instance_count,
        "accept_type": accept_type,
        "content_type": content_type,
        "custom_attributes": custom_attributes,
        "accelerator_type": accelerator_type,
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


def test_invalid_model_config_with_bad_endpoint_name_prefix():
    with pytest.raises(ValueError) as error:
        ModelConfig(
            model_name="xgboost-model",
            instance_type="ml.c5.xlarge",
            instance_count=1,
            accept_type="invalid_accept_type",
            endpoint_name_prefix="~invalid_endpoint_prefix",
        )
    assert (
        "Invalid endpoint_name_prefix. Please follow pattern ^[a-zA-Z0-9](-*[a-zA-Z0-9])."
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
    seed = 123
    shap_config = SHAPConfig(
        baseline=baseline,
        num_samples=num_samples,
        agg_method=agg_method,
        use_logit=use_logit,
        seed=seed,
    )
    expected_config = {
        "shap": {
            "baseline": baseline,
            "num_samples": num_samples,
            "agg_method": agg_method,
            "use_logit": use_logit,
            "save_local_shap_values": True,
            "seed": seed,
        }
    }
    assert expected_config == shap_config.get_explainability_config()


def test_shap_config_no_baseline():
    num_samples = 100
    agg_method = "mean_sq"
    use_logit = True
    seed = 123
    shap_config = SHAPConfig(
        num_samples=num_samples,
        agg_method=agg_method,
        num_clusters=2,
        use_logit=use_logit,
        seed=seed,
    )
    expected_config = {
        "shap": {
            "num_samples": num_samples,
            "agg_method": agg_method,
            "num_clusters": 2,
            "use_logit": use_logit,
            "save_local_shap_values": True,
            "seed": seed,
        }
    }
    assert expected_config == shap_config.get_explainability_config()


def test_shap_config_no_parameters():
    shap_config = SHAPConfig()
    expected_config = {
        "shap": {
            "use_logit": False,
            "save_local_shap_values": True,
        }
    }
    assert expected_config == shap_config.get_explainability_config()


def test_pdp_config():
    pdp_config = PDPConfig(features=["f1", "f2"], grid_resolution=20)
    expected_config = {
        "pdp": {"features": ["f1", "f2"], "grid_resolution": 20, "top_k_features": 10}
    }
    assert expected_config == pdp_config.get_explainability_config()


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
    with pytest.raises(ValueError) as error:
        SHAPConfig(baseline=[[1]], num_samples=1, agg_method="mean_abs", num_clusters=2)
    assert (
        "Baseline and num_clusters cannot be provided together. Please specify one of the two."
        in str(error.value)
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
def clarify_processor_with_job_name_prefix(sagemaker_session):
    return SageMakerClarifyProcessor(
        role="AmazonSageMaker-ExecutionRole",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
        job_name_prefix=JOB_NAME_PREFIX,
    )


@pytest.fixture(scope="module")
def data_config():
    return DataConfig(
        s3_data_input_path="s3://input/train.csv",
        s3_output_path="s3://output/analysis_test_result",
        label="Label",
        headers=["Label", "F1", "F2", "F3", "F4"],
        dataset_type="text/csv",
        joinsource="F4",
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
            ],
        ],
        num_samples=100,
        agg_method="mean_sq",
    )


@pytest.fixture(scope="module")
def pdp_config():
    return PDPConfig(features=["F1", "F2"], grid_resolution=20)


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_pre_training_bias(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    data_bias_config,
):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        clarify_processor.run_pre_training_bias(
            data_config,
            data_bias_config,
            wait=True,
            job_name="test",
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
                "F4",
            ],
            "joinsource_name_or_index": "F4",
            "label": "Label",
            "label_values_or_threshold": [1],
            "facet": [{"name_or_index": "F1"}],
            "group_variable": "F2",
            "methods": {"pre_training_bias": {"methods": "all"}},
        }
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        clarify_processor_with_job_name_prefix.run_pre_training_bias(
            data_config,
            data_bias_config,
            wait=True,
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        name_from_base.assert_called_with(JOB_NAME_PREFIX)
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            JOB_NAME,
            None,
            {"ExperimentName": "AnExperiment"},
        )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_post_training_bias(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    data_bias_config,
    model_config,
    model_predicted_label_config,
):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        clarify_processor.run_post_training_bias(
            data_config,
            data_bias_config,
            model_config,
            model_predicted_label_config,
            wait=True,
            job_name="test",
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
                "F4",
            ],
            "label": "Label",
            "label_values_or_threshold": [1],
            "joinsource_name_or_index": "F4",
            "facet": [{"name_or_index": "F1"}],
            "group_variable": "F2",
            "methods": {"post_training_bias": {"methods": "all"}},
            "predictor": {
                "model_name": "xgboost-model",
                "instance_type": "ml.c5.xlarge",
                "initial_instance_count": 1,
            },
        }
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        clarify_processor_with_job_name_prefix.run_post_training_bias(
            data_config,
            data_bias_config,
            model_config,
            model_predicted_label_config,
            wait=True,
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        name_from_base.assert_called_with(JOB_NAME_PREFIX)
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            JOB_NAME,
            None,
            {"ExperimentName": "AnExperiment"},
        )


@patch.object(Processor, "run")
def test_run_on_s3_analysis_config_file(
    processor_run, sagemaker_session, clarify_processor, data_config
):
    analysis_config = {
        "methods": {"post_training_bias": {"methods": "all"}},
    }
    with patch("sagemaker.clarify._upload_analysis_config", return_value=None) as mock_method:
        clarify_processor._run(
            data_config,
            analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        analysis_config_file = mock_method.call_args[0][0]
        mock_method.assert_called_with(
            analysis_config_file, data_config.s3_output_path, sagemaker_session, None
        )

        data_config_with_analysis_config_output = DataConfig(
            s3_data_input_path="s3://input/train.csv",
            s3_output_path="s3://output/analysis_test_result",
            s3_analysis_config_output_path="s3://analysis_config_output",
            label="Label",
            headers=[
                "Label",
                "F1",
                "F2",
                "F3",
            ],
            dataset_type="text/csv",
        )
        clarify_processor._run(
            data_config_with_analysis_config_output,
            analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        analysis_config_file = mock_method.call_args[0][0]
        mock_method.assert_called_with(
            analysis_config_file,
            data_config_with_analysis_config_output.s3_analysis_config_output_path,
            sagemaker_session,
            None,
        )


def _run_test_explain(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
    pdp_config,
    model_scores,
    expected_predictor_config,
):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        explanation_configs = None
        if shap_config and pdp_config:
            explanation_configs = [shap_config, pdp_config]
        elif shap_config:
            explanation_configs = shap_config
        elif pdp_config:
            explanation_configs = pdp_config

        clarify_processor.run_explainability(
            data_config,
            model_config,
            explanation_configs,
            model_scores=model_scores,
            wait=True,
            job_name="test",
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
                "F4",
            ],
            "label": "Label",
            "joinsource_name_or_index": "F4",
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
            "predictor": expected_predictor_config,
        }
        expected_explanation_configs = {}
        if shap_config:
            expected_explanation_configs["shap"] = {
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
        if pdp_config:
            expected_explanation_configs["pdp"] = {
                "features": ["F1", "F2"],
                "grid_resolution": 20,
                "top_k_features": 10,
            }
        expected_analysis_config["methods"] = expected_explanation_configs
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        clarify_processor_with_job_name_prefix.run_explainability(
            data_config,
            model_config,
            explanation_configs,
            model_scores=model_scores,
            wait=True,
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        name_from_base.assert_called_with(JOB_NAME_PREFIX)
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            JOB_NAME,
            None,
            {"ExperimentName": "AnExperiment"},
        )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_pdp(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
    pdp_config,
):
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    _run_test_explain(
        name_from_base,
        clarify_processor,
        clarify_processor_with_job_name_prefix,
        data_config,
        model_config,
        None,
        pdp_config,
        None,
        expected_predictor_config,
    )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_shap(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
):
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    _run_test_explain(
        name_from_base,
        clarify_processor,
        clarify_processor_with_job_name_prefix,
        data_config,
        model_config,
        shap_config,
        None,
        None,
        expected_predictor_config,
    )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_explainability_with_invalid_config(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
):
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    with pytest.raises(
        AttributeError, match="'NoneType' object has no attribute 'get_explainability_config'"
    ):
        _run_test_explain(
            name_from_base,
            clarify_processor,
            clarify_processor_with_job_name_prefix,
            data_config,
            model_config,
            None,
            None,
            None,
            expected_predictor_config,
        )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_explainability_with_multiple_shap_config(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
):
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    with pytest.raises(ValueError, match="Duplicate explainability configs are provided"):
        second_shap_config = copy.deepcopy(shap_config)
        second_shap_config.shap_config["num_samples"] = 200
        _run_test_explain(
            name_from_base,
            clarify_processor,
            clarify_processor_with_job_name_prefix,
            data_config,
            model_config,
            [shap_config, second_shap_config],
            None,
            None,
            expected_predictor_config,
        )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_shap_with_predicted_label(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
    pdp_config,
):
    probability = "pr"
    label_headers = ["success"]
    model_scores = ModelPredictedLabelConfig(
        probability=probability,
        label_headers=label_headers,
    )
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
        "probability": probability,
        "label_headers": label_headers,
    }
    _run_test_explain(
        name_from_base,
        clarify_processor,
        clarify_processor_with_job_name_prefix,
        data_config,
        model_config,
        shap_config,
        pdp_config,
        model_scores,
        expected_predictor_config,
    )
