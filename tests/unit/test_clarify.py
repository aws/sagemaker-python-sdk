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

from __future__ import absolute_import, print_function

import copy

import pytest
from mock import MagicMock, Mock, patch

from sagemaker import Processor, image_uris
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    PDPConfig,
    SageMakerClarifyProcessor,
    SHAPConfig,
    TextConfig,
    ImageConfig,
    _AnalysisConfigGenerator,
    DatasetType,
    ProcessingOutputHandler,
)

JOB_NAME_PREFIX = "my-prefix"
TIMESTAMP = "2021-06-17-22-29-54-685"
JOB_NAME = "{}-{}".format(JOB_NAME_PREFIX, TIMESTAMP)


def test_uri():
    uri = image_uris.retrieve("clarify", "us-west-2")
    assert "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0" == uri


@pytest.mark.parametrize(
    ("dataset_type", "features", "excluded_columns", "predicted_label"),
    [
        ("text/csv", None, ["F4"], "Predicted Label"),
        ("application/jsonlines", None, ["F4"], "Predicted Label"),
        ("application/json", "[*].[F1,F2,F3]", ["F4"], "Predicted Label"),
        ("application/x-parquet", None, ["F4"], "Predicted Label"),
    ],
)
def test_data_config(dataset_type, features, excluded_columns, predicted_label):
    # facets in input dataset
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    label_name = "Label"
    headers = ["Label", "F1", "F2", "F3", "F4", "Predicted Label"]
    data_config = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=s3_output_path,
        features=features,
        label=label_name,
        headers=headers,
        dataset_type=dataset_type,
        excluded_columns=excluded_columns,
        predicted_label=predicted_label,
    )

    expected_config = {
        "dataset_type": dataset_type,
        "headers": headers,
        "label": "Label",
    }
    if features:
        expected_config["features"] = features
    if excluded_columns:
        expected_config["excluded_columns"] = excluded_columns
    if predicted_label:
        expected_config["predicted_label"] = predicted_label

    assert expected_config == data_config.get_config()
    assert s3_data_input_path == data_config.s3_data_input_path
    assert s3_output_path == data_config.s3_output_path
    assert "None" == data_config.s3_compression_type
    assert "FullyReplicated" == data_config.s3_data_distribution_type


def test_data_config_with_separate_facet_dataset():
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    label_name = "Label"
    headers = ["Label", "F1", "F2", "F3", "F4"]

    # facets NOT in input dataset
    joinsource = 5
    facet_dataset_uri = "s3://path/to/facet.csv"
    facet_headers = ["Age"]
    predicted_label_dataset_uri = "s3://path/to/pred.csv"
    predicted_label_headers = ["Label", "F1", "F2", "F3", "F4", "Age"]
    predicted_label = "predicted_label"
    excluded_columns = "F4"

    data_config_no_facet = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=s3_output_path,
        label=label_name,
        headers=headers,
        dataset_type="text/csv",
        joinsource=joinsource,
        facet_dataset_uri=facet_dataset_uri,
        facet_headers=facet_headers,
        predicted_label_dataset_uri=predicted_label_dataset_uri,
        predicted_label_headers=predicted_label_headers,
        predicted_label=predicted_label,
        excluded_columns=excluded_columns,
    )

    expected_config_no_facet = {
        "dataset_type": "text/csv",
        "headers": headers,
        "label": label_name,
        "joinsource_name_or_index": joinsource,
        "facet_dataset_uri": facet_dataset_uri,
        "facet_headers": facet_headers,
        "predicted_label_dataset_uri": predicted_label_dataset_uri,
        "predicted_label_headers": predicted_label_headers,
        "predicted_label": predicted_label,
        "excluded_columns": excluded_columns,
    }

    assert expected_config_no_facet == data_config_no_facet.get_config()
    assert joinsource == data_config_no_facet.analysis_config["joinsource_name_or_index"]
    assert facet_dataset_uri == data_config_no_facet.facet_dataset_uri
    assert facet_headers == data_config_no_facet.facet_headers
    assert predicted_label_dataset_uri == data_config_no_facet.predicted_label_dataset_uri
    assert predicted_label_headers == data_config_no_facet.predicted_label_headers
    assert predicted_label == data_config_no_facet.predicted_label

    excluded_columns = "F4"
    data_config_excluded_cols = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=s3_output_path,
        label=label_name,
        headers=headers,
        dataset_type="text/csv",
        joinsource=joinsource,
        excluded_columns=excluded_columns,
    )

    expected_config_excluded_cols = {
        "dataset_type": "text/csv",
        "headers": headers,
        "label": label_name,
        "joinsource_name_or_index": joinsource,
        "excluded_columns": excluded_columns,
    }

    assert expected_config_excluded_cols == data_config_excluded_cols.get_config()
    assert joinsource == data_config_excluded_cols.analysis_config["joinsource_name_or_index"]
    assert excluded_columns == data_config_excluded_cols.excluded_columns


def test_invalid_data_config():
    # facets included in input dataset
    with pytest.raises(ValueError, match=r"^Invalid dataset_type"):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="whatnot_type",
        )
    # facets NOT included in input dataset
    error_msg = r"^The parameter 'predicted_label' is not supported for dataset_type"
    with pytest.raises(ValueError, match=error_msg):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="application/x-image",
            predicted_label="label",
        )
    error_msg = r"^The parameter 'excluded_columns' is not supported for dataset_type"
    with pytest.raises(ValueError, match=error_msg):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="application/x-image",
            excluded_columns="excluded",
        )
    error_msg = r"^The parameters 'facet_dataset_uri' and 'facet_headers' are not supported for dataset_type"  # noqa E501  # pylint: disable=c0301
    with pytest.raises(ValueError, match=error_msg):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="application/x-image",
            facet_dataset_uri="facet_dataset/URI",
            facet_headers="facet",
        )
    error_msg = r"^The parameters 'predicted_label_dataset_uri' and 'predicted_label_headers' are not supported for dataset_type"  # noqa E501  # pylint: disable=c0301
    with pytest.raises(ValueError, match=error_msg):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="application/jsonlines",
            predicted_label_dataset_uri="pred_dataset/URI",
            predicted_label_headers="prediction",
        )


# features JMESPath is required for JSON dataset types
def test_json_type_data_config_missing_features():
    # facets in input dataset
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    label_name = "Label"
    headers = ["Label", "F1", "F2", "F3", "F4", "Predicted Label"]
    with pytest.raises(
        ValueError, match="features JMESPath is required for application/json dataset_type"
    ):
        DataConfig(
            s3_data_input_path=s3_data_input_path,
            s3_output_path=s3_output_path,
            features=None,
            label=label_name,
            headers=headers,
            dataset_type="application/json",
            excluded_columns=["F4"],
            predicted_label="Predicted Label",
        )


def test_s3_data_distribution_type_ignorance():
    data_config = DataConfig(
        s3_data_input_path="s3://input/train.csv",
        s3_output_path="s3://output/analysis_test_result",
        label="Label",
        headers=["Label", "F1", "F2", "F3", "F4"],
        dataset_type="text/csv",
        joinsource="F4",
    )
    assert data_config.s3_data_distribution_type == "FullyReplicated"


def test_bias_config():
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


def test_invalid_bias_config():
    # Empty facet list,
    with pytest.raises(AssertionError, match="Please provide at least one facet"):
        BiasConfig(
            label_values_or_threshold=[1],
            facet_name=[],
        )

    # Two facets but only one value
    with pytest.raises(
        ValueError,
        match="The number of facet names doesn't match the number of facet values",
    ):
        BiasConfig(
            label_values_or_threshold=[1],
            facet_name=["Feature1", "Feature2"],
            facet_values_or_threshold=[[1]],
        )


@pytest.mark.parametrize(
    "facet_name,facet_values_or_threshold,expected_result",
    [
        # One facet, assume that it is binary and value 1 indicates the sensitive group
        [
            "Feature1",
            [1],
            {
                "facet": [{"name_or_index": "Feature1", "value_or_threshold": [1]}],
            },
        ],
        # The same facet as above, facet value is not specified. (Clarify will compute bias metrics
        # for each binary value).
        [
            "Feature1",
            None,
            {
                "facet": [{"name_or_index": "Feature1"}],
            },
        ],
        # Assume that the 2nd column (index 1, zero-based) of the dataset as facet, it has
        # four categories and two of them indicate the sensitive group.
        [
            1,
            ["category1, category2"],
            {
                "facet": [{"name_or_index": 1, "value_or_threshold": ["category1, category2"]}],
            },
        ],
        # The same facet as above, facet values are not specified. (Clarify will iterate
        # the categories and compute bias metrics for each category).
        [
            1,
            None,
            {
                "facet": [{"name_or_index": 1}],
            },
        ],
        # Assume that the facet is numeric value in range [0.0, 1.0]. Given facet threshold 0.5,
        # interval (0.5, 1.0] indicates the sensitive group.
        [
            "Feature3",
            [0.5],
            {
                "facet": [{"name_or_index": "Feature3", "value_or_threshold": [0.5]}],
            },
        ],
        # Multiple facets
        [
            ["Feature1", 1, "Feature3"],
            [[1], ["category1, category2"], [0.5]],
            {
                "facet": [
                    {"name_or_index": "Feature1", "value_or_threshold": [1]},
                    {
                        "name_or_index": 1,
                        "value_or_threshold": ["category1, category2"],
                    },
                    {"name_or_index": "Feature3", "value_or_threshold": [0.5]},
                ],
            },
        ],
        # Multiple facets, no value or threshold
        [
            ["Feature1", 1, "Feature3"],
            None,
            {
                "facet": [
                    {"name_or_index": "Feature1"},
                    {"name_or_index": 1},
                    {"name_or_index": "Feature3"},
                ],
            },
        ],
        # Multiple facets, specify values or threshold for some of them
        [
            ["Feature1", 1, "Feature3"],
            [[1], None, [0.5]],
            {
                "facet": [
                    {"name_or_index": "Feature1", "value_or_threshold": [1]},
                    {"name_or_index": 1},
                    {"name_or_index": "Feature3", "value_or_threshold": [0.5]},
                ],
            },
        ],
    ],
)
def test_facet_of_bias_config(facet_name, facet_values_or_threshold, expected_result):
    label_values = [1]
    bias_config = BiasConfig(
        label_values_or_threshold=label_values,
        facet_name=facet_name,
        facet_values_or_threshold=facet_values_or_threshold,
    )
    expected_config = {
        "label_values_or_threshold": label_values,
        **expected_result,
    }
    assert bias_config.get_config() == expected_config


@pytest.mark.parametrize(
    ("content_type", "accept_type"),
    [
        # All the combinations of content_type and accept_type should be acceptable
        ("text/csv", "text/csv"),
        ("application/jsonlines", "application/jsonlines"),
        ("text/csv", "application/json"),
        ("application/jsonlines", "application/json"),
        ("application/jsonlines", "text/csv"),
        ("application/json", "application/json"),
        ("application/json", "application/jsonlines"),
        ("application/json", "text/csv"),
        ("image/jpeg", "text/csv"),
        ("image/jpg", "text/csv"),
        ("image/png", "text/csv"),
        ("application/x-npy", "text/csv"),
    ],
)
def test_valid_model_config(content_type, accept_type):
    model_name = "xgboost-model"
    instance_type = "ml.c5.xlarge"
    instance_count = 1
    custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"
    target_model = "target_model_name"
    accelerator_type = "ml.eia1.medium"
    content_template = (
        '{"instances":$features}'
        if content_type == "application/jsonlines"
        else "$records"
        if content_type == "application/json"
        else None
    )
    record_template = "$features_kvp" if content_type == "application/json" else None
    model_config = ModelConfig(
        model_name=model_name,
        instance_type=instance_type,
        instance_count=instance_count,
        accept_type=accept_type,
        content_type=content_type,
        content_template=content_template,
        record_template=record_template,
        custom_attributes=custom_attributes,
        accelerator_type=accelerator_type,
        target_model=target_model,
    )
    expected_config = {
        "model_name": model_name,
        "instance_type": instance_type,
        "initial_instance_count": instance_count,
        "accept_type": accept_type,
        "content_type": content_type,
        "custom_attributes": custom_attributes,
        "accelerator_type": accelerator_type,
        "target_model": target_model,
    }
    if content_template is not None:
        expected_config["content_template"] = content_template
    if record_template is not None:
        expected_config["record_template"] = record_template
    assert expected_config == model_config.get_predictor_config()


@pytest.mark.parametrize(
    ("error", "content_type", "accept_type", "content_template", "record_template"),
    [
        (
            "Invalid accept_type invalid_accept_type. Please choose text/csv or application/jsonlines.",
            "text/csv",
            "invalid_accept_type",
            None,
            None,
        ),
        (
            "Invalid content_type invalid_content_type. Please choose text/csv or application/jsonlines.",
            "invalid_content_type",
            "text/csv",
            None,
            None,
        ),
        (
            "content_template field is required for content_type",
            "application/jsonlines",
            "text/csv",
            None,
            None,
        ),
        (
            "content_template and record_template are required for content_type",
            "application/json",
            "text/csv",
            None,
            None,
        ),
        (
            "content_template and record_template are required for content_type",
            "application/json",
            "text/csv",
            "$records",
            None,
        ),
        (
            r"Invalid content_template invalid_content_template. Please include a placeholder \$features.",
            "application/jsonlines",
            "text/csv",
            "invalid_content_template",
            None,
        ),
        (
            r"Invalid content_template invalid_content_template. Please include either placeholder "
            r"\$records or \$record.",
            "application/json",
            "text/csv",
            "invalid_content_template",
            "$features",
        ),
    ],
)
def test_invalid_model_config(error, content_type, accept_type, content_template, record_template):
    with pytest.raises(ValueError, match=error):
        ModelConfig(
            model_name="xgboost-model",
            instance_type="ml.c5.xlarge",
            instance_count=1,
            content_type=content_type,
            accept_type=accept_type,
            content_template=content_template,
            record_template=record_template,
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


@pytest.mark.parametrize(
    "baseline",
    [
        ([[0.26124998927116394, 0.2824999988079071, 0.06875000149011612]]),
        (
            {
                "instances": [
                    {"features": [0.26124998927116394, 0.2824999988079071, 0.06875000149011612]}
                ]
            }
        ),
    ],
)
def test_valid_shap_config(baseline):
    num_samples = 100
    agg_method = "mean_sq"
    use_logit = True
    seed = 123
    granularity = "sentence"
    language = "german"
    model_type = "IMAGE_CLASSIFICATION"
    num_segments = 2
    feature_extraction_method = "segmentation"
    segment_compactness = 10
    max_objects = 4
    iou_threshold = 0.5
    context = 1.0
    text_config = TextConfig(
        granularity=granularity,
        language=language,
    )
    image_config = ImageConfig(
        model_type=model_type,
        num_segments=num_segments,
        feature_extraction_method=feature_extraction_method,
        segment_compactness=segment_compactness,
        max_objects=max_objects,
        iou_threshold=iou_threshold,
        context=context,
    )
    shap_config = SHAPConfig(
        baseline=baseline,
        num_samples=num_samples,
        agg_method=agg_method,
        use_logit=use_logit,
        seed=seed,
        text_config=text_config,
        image_config=image_config,
    )
    expected_config = {
        "shap": {
            "baseline": baseline,
            "num_samples": num_samples,
            "agg_method": agg_method,
            "use_logit": use_logit,
            "save_local_shap_values": True,
            "seed": seed,
            "text_config": {
                "granularity": granularity,
                "language": language,
            },
            "image_config": {
                "model_type": model_type,
                "num_segments": num_segments,
                "feature_extraction_method": feature_extraction_method,
                "segment_compactness": segment_compactness,
                "max_objects": max_objects,
                "iou_threshold": iou_threshold,
                "context": context,
            },
        }
    }
    assert expected_config == shap_config.get_explainability_config()


def test_shap_config_features_to_explain():
    baseline = [1, 2, 3]
    num_samples = 100
    agg_method = "mean_sq"
    use_logit = True
    save_local_shap_values = True
    seed = 123
    features_to_explain = [0, 1]
    shap_config = SHAPConfig(
        baseline=baseline,
        num_samples=num_samples,
        agg_method=agg_method,
        use_logit=use_logit,
        save_local_shap_values=save_local_shap_values,
        seed=seed,
        features_to_explain=features_to_explain,
    )
    expected_config = {
        "shap": {
            "baseline": baseline,
            "num_samples": num_samples,
            "agg_method": agg_method,
            "use_logit": use_logit,
            "save_local_shap_values": save_local_shap_values,
            "seed": seed,
            "features_to_explain": features_to_explain,
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


def test_text_config():
    granularity = "sentence"
    language = "german"
    text_config = TextConfig(
        granularity=granularity,
        language=language,
    )
    expected_config = {
        "granularity": granularity,
        "language": language,
    }
    assert expected_config == text_config.get_text_config()


def test_invalid_text_config():
    with pytest.raises(ValueError) as error:
        TextConfig(
            granularity="invalid",
            language="english",
        )
    assert (
        "Invalid granularity invalid. Please choose among ['token', 'sentence', 'paragraph']"
        in str(error.value)
    )
    with pytest.raises(ValueError) as error:
        TextConfig(
            granularity="token",
            language="invalid",
        )
    assert "Invalid language invalid. Please choose among ['chinese'," in str(error.value)


def test_image_config():
    model_type = "IMAGE_CLASSIFICATION"
    num_segments = 2
    feature_extraction_method = "segmentation"
    segment_compactness = 10
    max_objects = 4
    iou_threshold = 0.5
    context = 1.0
    image_config = ImageConfig(
        model_type=model_type,
        num_segments=num_segments,
        feature_extraction_method=feature_extraction_method,
        segment_compactness=segment_compactness,
        max_objects=max_objects,
        iou_threshold=iou_threshold,
        context=context,
    )
    expected_config = {
        "model_type": model_type,
        "num_segments": num_segments,
        "feature_extraction_method": feature_extraction_method,
        "segment_compactness": segment_compactness,
        "max_objects": max_objects,
        "iou_threshold": iou_threshold,
        "context": context,
    }

    assert expected_config == image_config.get_image_config()


def test_invalid_image_config():
    model_type = "OBJECT_SEGMENTATION"
    num_segments = 2
    with pytest.raises(ValueError) as error:
        ImageConfig(
            model_type=model_type,
            num_segments=num_segments,
        )
    assert (
        "Clarify SHAP only supports object detection and image classification methods. "
        "Please set model_type to OBJECT_DETECTION or IMAGE_CLASSIFICATION." in str(error.value)
    )


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
    with pytest.raises(ValueError) as error:
        SHAPConfig(
            baseline=[[1, 2]],
            num_samples=1,
            text_config=TextConfig(granularity="token", language="english"),
            features_to_explain=[0],
        )
    assert (
        "`features_to_explain` is not supported for datasets containing text features or images."
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
        default_bucket_prefix=None,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value="mybucket")
    session_mock.upload_data = Mock(
        name="upload_data", return_value="mocked_s3_uri_from_upload_data"
    )
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = "arn:aws:iam::012345678901:role/SageMakerRole"
    session_mock.sagemaker_config = {}
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


def test_model_config_validations():
    new_model_endpoint_definition = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "instance_count": 1,
    }
    existing_endpoint_definition = {"endpoint_name": "existing_endpoint"}

    with pytest.raises(AssertionError):
        # should be one of them
        ModelConfig(
            **new_model_endpoint_definition,
            **existing_endpoint_definition,
        )

    with pytest.raises(AssertionError):
        # should be one of them
        ModelConfig(
            endpoint_name_prefix="prefix",
            **existing_endpoint_definition,
        )

    # success path for new model
    assert ModelConfig(**new_model_endpoint_definition).predictor_config == {
        "initial_instance_count": 1,
        "instance_type": "ml.c5.xlarge",
        "model_name": "xgboost-model",
    }

    # success path for existing endpoint
    assert (
        ModelConfig(**existing_endpoint_definition).predictor_config == existing_endpoint_definition
    )


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
            "methods": {
                "report": {"name": "report", "title": "Analysis Report"},
                "pre_training_bias": {"methods": "all"},
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
            "methods": {
                "report": {"name": "report", "title": "Analysis Report"},
                "post_training_bias": {"methods": "all"},
            },
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
        "dataset_type": "text/csv",
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
    expected_text_config=None,
    expected_image_config=None,
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
            if expected_text_config:
                expected_explanation_configs["shap"]["text_config"] = expected_text_config
            if expected_image_config:
                expected_explanation_configs["shap"]["image_config"] = expected_image_config
        if pdp_config:
            expected_explanation_configs["pdp"] = {
                "features": ["F1", "F2"],
                "grid_resolution": 20,
                "top_k_features": 10,
            }
        expected_analysis_config["methods"] = {
            "report": {"name": "report", "title": "Analysis Report"},
            **expected_explanation_configs,
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
        AttributeError,
        match="analysis_config must have at least one working method: "
        "One of the `pre_training_methods`, `post_training_methods`, `explainability_config`.",
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


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_shap_with_text_config(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
):
    granularity = "paragraph"
    language = "ukrainian"

    shap_config = SHAPConfig(
        baseline=[
            [
                0.26124998927116394,
                0.2824999988079071,
                0.06875000149011612,
            ]
        ],
        num_samples=100,
        agg_method="mean_sq",
        text_config=TextConfig(granularity, language),
    )

    expected_text_config = {
        "granularity": granularity,
        "language": language,
    }
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
        expected_text_config=expected_text_config,
    )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_shap_with_image_config(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
):
    model_type = "IMAGE_CLASSIFICATION"
    num_segments = 2
    feature_extraction_method = "segmentation"
    segment_compactness = 10
    max_objects = 4
    iou_threshold = 0.5
    context = 1.0
    image_config = ImageConfig(
        model_type=model_type,
        num_segments=num_segments,
        feature_extraction_method=feature_extraction_method,
        segment_compactness=segment_compactness,
        max_objects=max_objects,
        iou_threshold=iou_threshold,
        context=context,
    )

    shap_config = SHAPConfig(
        baseline=[
            [
                0.26124998927116394,
                0.2824999988079071,
                0.06875000149011612,
            ]
        ],
        num_samples=100,
        agg_method="mean_sq",
        image_config=image_config,
    )

    expected_image_config = {
        "model_type": model_type,
        "num_segments": num_segments,
        "feature_extraction_method": feature_extraction_method,
        "segment_compactness": segment_compactness,
        "max_objects": max_objects,
        "iou_threshold": iou_threshold,
        "context": context,
    }
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
        expected_image_config=expected_image_config,
    )


def test_analysis_config_generator_for_explainability(data_config, model_config):
    model_scores = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.explainability(
        data_config,
        model_config,
        model_scores,
        SHAPConfig(),
    )
    expected = {
        "dataset_type": "text/csv",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "shap": {"save_local_shap_values": True, "use_logit": False},
        },
        "predictor": {
            "initial_instance_count": 1,
            "instance_type": "ml.c5.xlarge",
            "label_headers": ["success"],
            "model_name": "xgboost-model",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_explainability_failing(data_config, model_config):
    model_scores = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    with pytest.raises(
        ValueError,
        match="PDP features must be provided when ShapConfig is not provided",
    ):
        _AnalysisConfigGenerator.explainability(
            data_config,
            model_config,
            model_scores,
            PDPConfig(),
        )

    with pytest.raises(ValueError, match="Duplicate explainability configs are provided"):
        _AnalysisConfigGenerator.explainability(
            data_config,
            model_config,
            model_scores,
            [SHAPConfig(), SHAPConfig()],
        )

    with pytest.raises(
        AttributeError,
        match="analysis_config must have at least one working method: "
        "One of the "
        "`pre_training_methods`, `post_training_methods`, `explainability_config`.",
    ):
        _AnalysisConfigGenerator.explainability(
            data_config,
            model_config,
            model_scores,
            [],
        )


def test_analysis_config_generator_for_bias_explainability(
    data_config, data_bias_config, model_config
):
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias_and_explainability(
        data_config,
        model_config,
        model_predicted_label_config,
        [SHAPConfig(), PDPConfig()],
        data_bias_config,
        pre_training_methods="all",
        post_training_methods="all",
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "pdp": {"grid_resolution": 15, "top_k_features": 10},
            "post_training_bias": {"methods": "all"},
            "pre_training_bias": {"methods": "all"},
            "report": {"name": "report", "title": "Analysis Report"},
            "shap": {"save_local_shap_values": True, "use_logit": False},
        },
        "predictor": {
            "initial_instance_count": 1,
            "instance_type": "ml.c5.xlarge",
            "label_headers": ["success"],
            "model_name": "xgboost-model",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_bias_explainability_with_existing_endpoint(
    data_config, data_bias_config
):
    model_config = ModelConfig(endpoint_name="existing_endpoint_name")
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias_and_explainability(
        data_config,
        model_config,
        model_predicted_label_config,
        [SHAPConfig(), PDPConfig()],
        data_bias_config,
        pre_training_methods="all",
        post_training_methods="all",
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "pdp": {"grid_resolution": 15, "top_k_features": 10},
            "post_training_bias": {"methods": "all"},
            "pre_training_bias": {"methods": "all"},
            "report": {"name": "report", "title": "Analysis Report"},
            "shap": {"save_local_shap_values": True, "use_logit": False},
        },
        "predictor": {
            "label_headers": ["success"],
            "endpoint_name": "existing_endpoint_name",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_bias_pre_training(data_config, data_bias_config):
    actual = _AnalysisConfigGenerator.bias_pre_training(
        data_config, data_bias_config, methods="all"
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "pre_training_bias": {"methods": "all"},
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_bias_post_training(
    data_config, data_bias_config, model_config
):
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias_post_training(
        data_config,
        data_bias_config,
        model_predicted_label_config,
        methods="all",
        model_config=model_config,
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "post_training_bias": {"methods": "all"},
        },
        "predictor": {
            "initial_instance_count": 1,
            "instance_type": "ml.c5.xlarge",
            "label_headers": ["success"],
            "model_name": "xgboost-model",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_bias(data_config, data_bias_config, model_config):
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias(
        data_config,
        data_bias_config,
        model_config,
        model_predicted_label_config,
        pre_training_methods="all",
        post_training_methods="all",
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "post_training_bias": {"methods": "all"},
            "pre_training_bias": {"methods": "all"},
        },
        "predictor": {
            "initial_instance_count": 1,
            "instance_type": "ml.c5.xlarge",
            "label_headers": ["success"],
            "model_name": "xgboost-model",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_for_bias_no_model_config(data_bias_config):
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    predicted_labels_uri = "s3://path/to/predicted_labels.csv"
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
        predicted_label_dataset_uri=predicted_labels_uri,
        predicted_label_headers=["PredictedLabel"],
        predicted_label="PredictedLabel",
    )
    model_config = None
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        probability_threshold=0.8,
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias(
        data_config,
        data_bias_config,
        model_config,
        model_predicted_label_config,
        pre_training_methods="all",
        post_training_methods="all",
    )
    expected = {
        "dataset_type": "text/csv",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "label": "Label",
        "predicted_label_dataset_uri": "s3://path/to/predicted_labels.csv",
        "predicted_label_headers": ["PredictedLabel"],
        "predicted_label": "PredictedLabel",
        "label_values_or_threshold": [1],
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "pre_training_bias": {"methods": "all"},
            "post_training_bias": {"methods": "all"},
        },
        "probability_threshold": 0.8,
    }
    assert actual == expected


def test_invalid_analysis_config(data_config, data_bias_config, model_config):
    with pytest.raises(
        ValueError, match="model_config must be provided when explainability methods are selected."
    ):
        _AnalysisConfigGenerator.bias_and_explainability(
            data_config=data_config,
            model_config=None,
            model_predicted_label_config=ModelPredictedLabelConfig(),
            explainability_config=SHAPConfig(),
            bias_config=data_bias_config,
            pre_training_methods="all",
            post_training_methods="all",
        )

    with pytest.raises(
        ValueError,
        match="model_config must be provided when `predicted_label_dataset_uri` or "
        "`predicted_label` are not provided in data_config.",
    ):
        _AnalysisConfigGenerator.bias(
            data_config=data_config,
            model_config=None,
            model_predicted_label_config=ModelPredictedLabelConfig(),
            bias_config=data_bias_config,
            pre_training_methods="all",
            post_training_methods="all",
        )


class TestProcessingOutputHandler:
    def test_get_s3_upload_mode_image(self):
        analysis_config = {"dataset_type": DatasetType.IMAGE.value}
        s3_upload_mode = ProcessingOutputHandler.get_s3_upload_mode(analysis_config)
        assert s3_upload_mode == ProcessingOutputHandler.S3UploadMode.CONTINUOUS.value

    def test_get_s3_upload_mode_text(self):
        analysis_config = {"dataset_type": DatasetType.TEXTCSV.value}
        s3_upload_mode = ProcessingOutputHandler.get_s3_upload_mode(analysis_config)
        assert s3_upload_mode == ProcessingOutputHandler.S3UploadMode.ENDOFJOB.value
