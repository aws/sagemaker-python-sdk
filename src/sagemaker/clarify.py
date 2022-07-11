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
"""This module configures the SageMaker Clarify bias and model explainability processor jobs.

SageMaker Clarify
==================
"""
from __future__ import absolute_import, print_function

import copy
import json
import logging
import os
import re

import tempfile
from abc import ABC, abstractmethod
from sagemaker import image_uris, s3, utils
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

logger = logging.getLogger(__name__)


class DataConfig:
    """Config object related to configurations of the input and output dataset."""

    def __init__(
        self,
        s3_data_input_path,
        s3_output_path,
        s3_analysis_config_output_path=None,
        label=None,
        headers=None,
        features=None,
        dataset_type="text/csv",
        s3_compression_type="None",
        joinsource=None,
    ):
        """Initializes a configuration of both input and output datasets.

        Args:
            s3_data_input_path (str): Dataset S3 prefix/object URI.
            s3_output_path (str): S3 prefix to store the output.
            s3_analysis_config_output_path (str): S3 prefix to store the analysis config output.
                If this field is None, then the ``s3_output_path`` will be used
                to store the ``analysis_config`` output.
            label (str): Target attribute of the model **required** for bias metrics (both pre-
                and post-training). Optional when running SHAP explainability.
                Specified as column name or index for CSV dataset, or as JSONPath for JSONLines.
            headers (list[str]): A list of column names in the input dataset.
            features (str): JSONPath for locating the feature columns for bias metrics if the
                dataset format is JSONLines.
            dataset_type (str): Format of the dataset. Valid values are ``"text/csv"`` for CSV,
                ``"application/jsonlines"`` for JSONLines, and
                ``"application/x-parquet"`` for Parquet.
            s3_compression_type (str): Valid options are "None" or ``"Gzip"``.
            joinsource (str): The name or index of the column in the dataset that acts as an
                identifier column (for instance, while performing a join). This column is only
                used as an identifier, and not used for any other computations. This is an
                optional field in all cases except when the dataset contains more than one file,
                and ``save_local_shap_values`` is set to True
                in :class:`~sagemaker.clarify.SHAPConfig`.
        """
        if dataset_type not in [
            "text/csv",
            "application/jsonlines",
            "application/x-parquet",
            "application/x-image",
        ]:
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'."
                f" Please check the API documentation for the supported dataset types."
            )
        self.s3_data_input_path = s3_data_input_path
        self.s3_output_path = s3_output_path
        self.s3_analysis_config_output_path = s3_analysis_config_output_path
        self.s3_data_distribution_type = "FullyReplicated"
        self.s3_compression_type = s3_compression_type
        self.label = label
        self.headers = headers
        self.features = features
        self.analysis_config = {
            "dataset_type": dataset_type,
        }
        _set(features, "features", self.analysis_config)
        _set(headers, "headers", self.analysis_config)
        _set(label, "label", self.analysis_config)
        _set(joinsource, "joinsource_name_or_index", self.analysis_config)

    def get_config(self):
        """Returns part of an analysis config dictionary."""
        return copy.deepcopy(self.analysis_config)


class BiasConfig:
    """Config object with user-defined bias configurations of the input dataset."""

    def __init__(
        self,
        label_values_or_threshold,
        facet_name,
        facet_values_or_threshold=None,
        group_name=None,
    ):
        """Initializes a configuration of the sensitive groups in the dataset.

        Args:
            label_values_or_threshold ([int or float or str]): List of label value(s) or threshold
                to indicate positive outcome used for bias metrics.
                The appropriate threshold depends on the problem type:

                * Binary: The list has one positive value.
                * Categorical:The list has one or more (but not all) categories
                  which are the positive values.
                * Regression: The list should include one threshold that defines the **exclusive**
                  lower bound of positive values.

            facet_name (str or int or list[str] or list[int]): Sensitive attribute column name
                (or index in the input data) to use when computing bias metrics. It can also be a
                list of names (or indexes) for computing metrics for multiple sensitive attributes.
            facet_values_or_threshold ([int or float or str] or [[int or float or str]]):
                The parameter controls the values of the sensitive group.
                If ``facet_name`` is a scalar, then it can be None or a list.
                Depending on the data type of the facet column, the values mean:

                * Binary data: None means computing the bias metrics for each binary value.
                  Or add one binary value to the list, to compute its bias metrics only.
                * Categorical data: None means computing the bias metrics for each category. Or add
                  one or more (but not all) categories to the list, to compute their
                  bias metrics v.s. the other categories.
                * Continuous data: The list should include one and only one threshold which defines
                  the **exclusive** lower bound of a sensitive group.

                If ``facet_name`` is a list, then ``facet_values_or_threshold`` can be None
                if all facets are of binary or categorical type.
                Otherwise, ``facet_values_or_threshold`` should be a list, and each element
                is the value or threshold of the corresponding facet.
            group_name (str): Optional column name or index to indicate a group column to be used
                for the bias metric
                `Conditional Demographic Disparity in Labels `(CDDL) <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-cddl.html>`_
                or
                `Conditional Demographic Disparity in Predicted Labels (CDDPL) <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-cddpl.html>`_.

        Raises:
            ValueError: If the number of ``facet_names`` doesn't equal number of ``facet values``
        """  # noqa E501  # pylint: disable=c0301
        if isinstance(facet_name, list):
            assert len(facet_name) > 0, "Please provide at least one facet"
            if facet_values_or_threshold is None:
                facet_list = [
                    {"name_or_index": single_facet_name} for single_facet_name in facet_name
                ]
            elif len(facet_values_or_threshold) == len(facet_name):
                facet_list = []
                for i, single_facet_name in enumerate(facet_name):
                    facet = {"name_or_index": single_facet_name}
                    if facet_values_or_threshold is not None:
                        _set(facet_values_or_threshold[i], "value_or_threshold", facet)
                    facet_list.append(facet)
            else:
                raise ValueError(
                    "The number of facet names doesn't match the number of facet values"
                )
        else:
            facet = {"name_or_index": facet_name}
            _set(facet_values_or_threshold, "value_or_threshold", facet)
            facet_list = [facet]
        self.analysis_config = {
            "label_values_or_threshold": label_values_or_threshold,
            "facet": facet_list,
        }
        _set(group_name, "group_variable", self.analysis_config)

    def get_config(self):
        """Returns a dictionary of bias detection configurations, part of the analysis config"""
        return copy.deepcopy(self.analysis_config)


class ModelConfig:
    """Config object related to a model and its endpoint to be created."""

    def __init__(
        self,
        model_name,
        instance_count,
        instance_type,
        accept_type=None,
        content_type=None,
        content_template=None,
        custom_attributes=None,
        accelerator_type=None,
        endpoint_name_prefix=None,
        target_model=None,
    ):
        r"""Initializes a configuration of a model and the endpoint to be created for it.

        Args:
            model_name (str): Model name (as created by 'CreateModel').
            instance_count (int): The number of instances of a new endpoint for model inference.
            instance_type (str): The type of EC2 instance to use for model inference,
                for example, ``"ml.c5.xlarge"``.
            accept_type (str): The model output format to be used for getting inferences with the
                shadow endpoint. Valid values are "text/csv" for CSV and "application/jsonlines".
                Default is the same as content_type.
            content_type (str): The model input format to be used for getting inferences with the
                shadow endpoint. Valid values are "text/csv" for CSV and "application/jsonlines".
                Default is the same as dataset format.
            content_template (str): A template string to be used to construct the model input from
                dataset instances. It is only used when ``model_content_type`` is
                ``"application/jsonlines"``. The template should have one and only one placeholder,
                "features", which will be replaced by a features list to form the model inference
                input.
            custom_attributes (str): Provides additional information about a request for an
                inference submitted to a model hosted at an Amazon SageMaker endpoint. The
                information is an opaque value that is forwarded verbatim. You could use this
                value, for example, to provide an ID that you can use to track a request or to
                provide other metadata that a service endpoint was programmed to process. The value
                must consist of no more than 1024 visible US-ASCII characters as specified in
                Section 3.3.6.
                `Field Value Components <https://tools.ietf.org/html/rfc7230#section-3.2.6>`_
                of the Hypertext Transfer Protocol (HTTP/1.1).
            accelerator_type (str): SageMaker
                `Elastic Inference <https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html>`_
                accelerator type to deploy to the model endpoint instance
                for making inferences to the model.
            endpoint_name_prefix (str): The endpoint name prefix of a new endpoint. Must follow
                pattern ``^[a-zA-Z0-9](-\*[a-zA-Z0-9]``.
            target_model (str): Sets the target model name when using a multi-model endpoint. For
                more information about multi-model endpoints, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html

        Raises:
            ValueError: when the ``endpoint_name_prefix`` is invalid, ``accept_type`` is invalid,
                 ``content_type`` is invalid, or ``content_template`` has no placeholder "features"
        """
        self.predictor_config = {
            "model_name": model_name,
            "instance_type": instance_type,
            "initial_instance_count": instance_count,
        }
        if endpoint_name_prefix is not None:
            if re.search("^[a-zA-Z0-9](-*[a-zA-Z0-9])", endpoint_name_prefix) is None:
                raise ValueError(
                    "Invalid endpoint_name_prefix."
                    " Please follow pattern ^[a-zA-Z0-9](-*[a-zA-Z0-9])."
                )
            self.predictor_config["endpoint_name_prefix"] = endpoint_name_prefix
        if accept_type is not None:
            if accept_type not in ["text/csv", "application/jsonlines"]:
                raise ValueError(
                    f"Invalid accept_type {accept_type}."
                    f" Please choose text/csv or application/jsonlines."
                )
            self.predictor_config["accept_type"] = accept_type
        if content_type is not None:
            if content_type not in [
                "text/csv",
                "application/jsonlines",
                "image/jpeg",
                "image/jpg",
                "image/png",
                "application/x-npy",
            ]:
                raise ValueError(
                    f"Invalid content_type {content_type}."
                    f" Please choose text/csv or application/jsonlines."
                )
            self.predictor_config["content_type"] = content_type
        if content_template is not None:
            if "$features" not in content_template:
                raise ValueError(
                    f"Invalid content_template {content_template}."
                    f" Please include a placeholder $features."
                )
            self.predictor_config["content_template"] = content_template
        _set(custom_attributes, "custom_attributes", self.predictor_config)
        _set(accelerator_type, "accelerator_type", self.predictor_config)
        _set(target_model, "target_model", self.predictor_config)

    def get_predictor_config(self):
        """Returns part of the predictor dictionary of the analysis config."""
        return copy.deepcopy(self.predictor_config)


class ModelPredictedLabelConfig:
    """Config object to extract a predicted label from the model output."""

    def __init__(
        self,
        label=None,
        probability=None,
        probability_threshold=None,
        label_headers=None,
    ):
        """Initializes a model output config to extract the predicted label or predicted score(s).

        The following examples show different parameter configurations depending on the endpoint:

        * **Regression task:**
          The model returns the score, e.g. ``1.2``. We don't need to specify
          anything. For json output, e.g. ``{'score': 1.2}``, we can set ``label='score'``.
        * **Binary classification:**

          * The model returns a single probability score. We want to classify as ``"yes"``
            predictions with a probability score over ``0.2``.
            We can set ``probability_threshold=0.2`` and ``label_headers="yes"``.
          * The model returns ``{"probability": 0.3}``, for which we would like to apply a
            threshold of ``0.5`` to obtain a predicted label in ``{0, 1}``.
            In this case we can set ``label="probability"``.
          * The model returns a tuple of the predicted label and the probability.
            In this case we can set ``label = 0``.
        * **Multiclass classification:**

          * The model returns ``{'labels': ['cat', 'dog', 'fish'],
            'probabilities': [0.35, 0.25, 0.4]}``. In this case we would set
            ``probability='probabilities'``, ``label='labels'``,
            and infer the predicted label to be ``'fish'``.
          * The model returns ``{'predicted_label': 'fish', 'probabilities': [0.35, 0.25, 0.4]}``.
            In this case we would set the ``label='predicted_label'``.
          * The model returns ``[0.35, 0.25, 0.4]``. In this case, we can set
            ``label_headers=['cat','dog','fish']`` and infer the predicted label to be ``'fish'``.

        Args:
            label (str or int): Index or JSONPath location in the model output for the prediction.
                In case, this is a predicted label of the same type as the label in the dataset,
                no further arguments need to be specified.
            probability (str or int): Index or JSONPath location in the model output
                for the predicted score(s).
            probability_threshold (float): An optional value for binary prediction tasks in which
                the model returns a probability, to indicate the threshold to convert the
                prediction to a boolean value. Default is ``0.5``.
            label_headers (list[str]): List of headers, each for a predicted score in model output.
                For bias analysis, it is used to extract the label value with the highest score as
                predicted label. For explainability jobs, it is used to beautify the analysis report
                by replacing placeholders like ``'label0'``.

        Raises:
            TypeError: when the ``probability_threshold`` cannot be cast to a float
        """
        self.label = label
        self.probability = probability
        self.probability_threshold = probability_threshold
        self.label_headers = label_headers
        if probability_threshold is not None:
            try:
                float(probability_threshold)
            except ValueError:
                raise TypeError(
                    f"Invalid probability_threshold {probability_threshold}. "
                    f"Please choose one that can be cast to float."
                )
        self.predictor_config = {}
        _set(label, "label", self.predictor_config)
        _set(probability, "probability", self.predictor_config)
        _set(label_headers, "label_headers", self.predictor_config)

    def get_predictor_config(self):
        """Returns ``probability_threshold`` and predictor config dictionary."""
        return self.probability_threshold, copy.deepcopy(self.predictor_config)


class ExplainabilityConfig(ABC):
    """Abstract config class to configure an explainability method."""

    @abstractmethod
    def get_explainability_config(self):
        """Returns config."""
        return None


class PDPConfig(ExplainabilityConfig):
    """Config class for Partial Dependence Plots (PDP).

    `PDPs <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-partial-dependence-plots.html>`_
    show the marginal effect (the dependence) a subset of features has on the predicted
    outcome of an ML model.

    When PDP is requested (by passing in a :class:`~sagemaker.clarify.PDPConfig` to the
    ``explainability_config`` parameter of :class:`~sagemaker.clarify.SageMakerClarifyProcessor`),
    the Partial Dependence Plots are included in the output
    `report <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-feature-attribute-baselines-reports.html>`__
    and the corresponding values are included in the analysis output.
    """  # noqa E501

    def __init__(self, features=None, grid_resolution=15, top_k_features=10):
        """Initializes PDP config.

        Args:
            features (None or list): List of feature names or indices for which partial dependence
                plots are computed and plotted. When :class:`~sagemaker.clarify.ShapConfig`
                is provided, this parameter is optional, as Clarify will compute the
                partial dependence plots for top features based on
                `SHAP <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-shapley-values.html>`__
                attributions. When :class:`~sagemaker.clarify.ShapConfig` is not provided,
                ``features`` must be provided.
            grid_resolution (int): When using numerical features, this integer represents the
                number of buckets that the range of values must be divided into. This decides the
                granularity of the grid in which the PDP are plotted.
            top_k_features (int): Sets the number of top SHAP attributes used to compute
                partial dependence plots.
        """  # noqa E501
        self.pdp_config = {"grid_resolution": grid_resolution, "top_k_features": top_k_features}
        if features is not None:
            self.pdp_config["features"] = features

    def get_explainability_config(self):
        """Returns PDP config dictionary."""
        return copy.deepcopy({"pdp": self.pdp_config})


class TextConfig:
    """Config object to handle text features for text explainability

    `SHAP analysis <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html>`__
    breaks down longer text into chunks (e.g. tokens, sentences, or paragraphs)
    and replaces them with the strings specified in the baseline for that feature.
    The `shap value <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-shapley-values.html>`_
    of a chunk then captures how much replacing it affects the prediction.
    """  # noqa E501  # pylint: disable=c0301

    _SUPPORTED_GRANULARITIES = ["token", "sentence", "paragraph"]
    _SUPPORTED_LANGUAGES = [
        "chinese",
        "danish",
        "dutch",
        "english",
        "french",
        "german",
        "greek",
        "italian",
        "japanese",
        "lithuanian",
        "multi-language",
        "norwegian bokmål",
        "polish",
        "portuguese",
        "romanian",
        "russian",
        "spanish",
        "afrikaans",
        "albanian",
        "arabic",
        "armenian",
        "basque",
        "bengali",
        "bulgarian",
        "catalan",
        "croatian",
        "czech",
        "estonian",
        "finnish",
        "gujarati",
        "hebrew",
        "hindi",
        "hungarian",
        "icelandic",
        "indonesian",
        "irish",
        "kannada",
        "kyrgyz",
        "latvian",
        "ligurian",
        "luxembourgish",
        "macedonian",
        "malayalam",
        "marathi",
        "nepali",
        "persian",
        "sanskrit",
        "serbian",
        "setswana",
        "sinhala",
        "slovak",
        "slovenian",
        "swedish",
        "tagalog",
        "tamil",
        "tatar",
        "telugu",
        "thai",
        "turkish",
        "ukrainian",
        "urdu",
        "vietnamese",
        "yoruba",
    ]

    def __init__(
        self,
        granularity,
        language,
    ):
        """Initializes a text configuration.

        Args:
            granularity (str): Determines the granularity in which text features are broken down
                to. Accepted values are ``"token"``, ``"sentence"``, or ``"paragraph"``.
                Computes `shap values <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-shapley-values.html>`_
                for these units.
            language (str): Specifies the language of the text features. Accepted values are
                one of the following:
                "chinese", "danish", "dutch", "english", "french", "german", "greek", "italian",
                "japanese", "lithuanian", "multi-language", "norwegian bokmål", "polish",
                "portuguese", "romanian", "russian", "spanish", "afrikaans", "albanian", "arabic",
                "armenian", "basque", "bengali", "bulgarian", "catalan", "croatian", "czech",
                "estonian", "finnish", "gujarati", "hebrew", "hindi", "hungarian", "icelandic",
                "indonesian", "irish", "kannada", "kyrgyz", "latvian", "ligurian", "luxembourgish",
                "macedonian", "malayalam", "marathi", "nepali", "persian", "sanskrit", "serbian",
                "setswana", "sinhala", "slovak", "slovenian", "swedish", "tagalog", "tamil",
                "tatar", "telugu", "thai", "turkish", "ukrainian", "urdu", "vietnamese", "yoruba".
                Use "multi-language" for a mix of multiple languages.

        Raises:
            ValueError: when ``granularity`` is not in list of supported values
                or ``language`` is not in list of supported values
        """  # noqa E501  # pylint: disable=c0301
        if granularity not in TextConfig._SUPPORTED_GRANULARITIES:
            raise ValueError(
                f"Invalid granularity {granularity}. Please choose among "
                f"{TextConfig._SUPPORTED_GRANULARITIES}"
            )
        if language not in TextConfig._SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Invalid language {language}. Please choose among "
                f"{TextConfig._SUPPORTED_LANGUAGES}"
            )
        self.text_config = {
            "granularity": granularity,
            "language": language,
        }

    def get_text_config(self):
        """Returns a text config dictionary, part of the analysis config dictionary."""
        return copy.deepcopy(self.text_config)


class ImageConfig:
    """Config object for handling images"""

    def __init__(
        self,
        model_type,
        num_segments=None,
        feature_extraction_method=None,
        segment_compactness=None,
        max_objects=None,
        iou_threshold=None,
        context=None,
    ):
        """Initializes a config object for Computer Vision (CV) Image explainability.

        `SHAP for CV explainability <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability-computer-vision.html>`__.
        generating heat maps that visualize feature attributions for input images.
        These heat maps highlight the image's features according
        to how much they contribute to the CV model prediction.

        ``"IMAGE_CLASSIFICATION"`` and ``"OBJECT_DETECTION"`` are the two supported CV use cases.

        Args:
            model_type (str): Specifies the type of CV model and use case. Accepted options:
                ``"IMAGE_CLASSIFICATION"`` or ``"OBJECT_DETECTION"``.
            num_segments (None or int): Approximate number of segments to generate when running
                SKLearn's `SLIC method <https://scikit-image.org/docs/dev/api/skimage.segmentation.html?highlight=slic#skimage.segmentation.slic>`_
                for image segmentation to generate features/superpixels.
                The default is None. When set to None, runs SLIC with 20 segments.
            feature_extraction_method (None or str): method used for extracting features from the
                image (ex: "segmentation"). Default is ``"segmentation"``.
            segment_compactness (None or float): Balances color proximity and space proximity.
                Higher values give more weight to space proximity, making superpixel
                shapes more square/cubic. We recommend exploring possible values on a log
                scale, e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen value.
                The default is None. When set to None, runs with the default value of ``5``.
            max_objects (None or int): Maximum number of objects displayed when running SHAP
                with an ``"OBJECT_DETECTION"`` model. The Object detection algorithm may detect
                more than the ``max_objects`` number of objects in a single image.
                In that case, the algorithm displays the top ``max_objects`` number of objects
                according to confidence score. Default value is None. In the ``"OBJECT_DETECTION"``
                case, passing in None leads to a default value of ``3``.
            iou_threshold (None or float): Minimum intersection over union for the object
                bounding box to consider its confidence score for computing SHAP values,
                in the range ``[0.0, 1.0]``. Used only for the ``"OBJECT_DETECTION"`` case,
                where passing in None sets the default value of ``0.5``.
            context (None or float): The portion of the image outside the bounding box used
                in SHAP analysis, in the range ``[0.0, 1.0]``. If set to ``1.0``, the whole image
                is considered; if set to ``0.0`` only the image inside bounding box is considered.
                Only used for the ``"OBJECT_DETECTION"`` case,
                when passing in None sets the default value of ``1.0``.

        """  # noqa E501  # pylint: disable=c0301
        self.image_config = {}

        if model_type not in ["OBJECT_DETECTION", "IMAGE_CLASSIFICATION"]:
            raise ValueError(
                "Clarify SHAP only supports object detection and image classification methods. "
                "Please set model_type to OBJECT_DETECTION or IMAGE_CLASSIFICATION."
            )
        self.image_config["model_type"] = model_type
        _set(num_segments, "num_segments", self.image_config)
        _set(feature_extraction_method, "feature_extraction_method", self.image_config)
        _set(segment_compactness, "segment_compactness", self.image_config)
        _set(max_objects, "max_objects", self.image_config)
        _set(iou_threshold, "iou_threshold", self.image_config)
        _set(context, "context", self.image_config)

    def get_image_config(self):
        """Returns the image config part of an analysis config dictionary."""
        return copy.deepcopy(self.image_config)


class SHAPConfig(ExplainabilityConfig):
    """Config class for `SHAP <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html>`__.

    The SHAP algorithm calculates feature attributions by computing
    the contribution of each feature to the prediction outcome, using the concept of
    `Shapley values <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-shapley-values.html>`_.

    These attributions can be provided for specific predictions (locally)
    and at a global level for the model as a whole.
    """  # noqa E501  # pylint: disable=c0301

    def __init__(
        self,
        baseline=None,
        num_samples=None,
        agg_method=None,
        use_logit=False,
        save_local_shap_values=True,
        seed=None,
        num_clusters=None,
        text_config=None,
        image_config=None,
    ):
        """Initializes config for SHAP analysis.

        Args:
            baseline (None or str or list): `Baseline dataset <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-feature-attribute-shap-baselines.html>`_
                for the Kernel SHAP algorithm, accepted in the form of:
                S3 object URI, a list of rows (with at least one element),
                or None (for no input baseline). The baseline dataset must have the same format
                as the input dataset specified in :class:`~sagemaker.clarify.DataConfig`.
                Each row must have only the feature columns/values and omit the label column/values.
                If None, a baseline will be calculated automatically on the input dataset
                using K-means (for numerical data) or K-prototypes (if there is categorical data).
            num_samples (None or int): Number of samples to be used in the Kernel SHAP algorithm.
                This number determines the size of the generated synthetic dataset to compute the
                SHAP values. If not provided then Clarify job will choose a proper value according
                to the count of features.
            agg_method (None or str): Aggregation method for global SHAP values. Valid values are
                ``"mean_abs"`` (mean of absolute SHAP values for all instances),
                ``"median"`` (median of SHAP values for all instances) and
                ``"mean_sq"`` (mean of squared SHAP values for all instances).
                If None is provided, then Clarify job uses the method ``"mean_abs"``.
            use_logit (bool): Indicates whether to apply the logit function to model predictions.
                Default is False. If ``use_logit`` is true then the SHAP values will
                have log-odds units.
            save_local_shap_values (bool): Indicates whether to save the local SHAP values
                in the output location. Default is True.
            seed (int): Seed value to get deterministic SHAP values. Default is None.
            num_clusters (None or int): If a ``baseline`` is not provided, Clarify automatically
                computes a baseline dataset via a clustering algorithm (K-means/K-prototypes), which
                takes ``num_clusters`` as a parameter. ``num_clusters`` will be the resulting size
                of the baseline dataset. If not provided, Clarify job uses a default value.
            text_config (:class:`~sagemaker.clarify.TextConfig`): Config object for handling
                text features. Default is None.
            image_config (:class:`~sagemaker.clarify.ImageConfig`): Config for handling image
                features. Default is None.
        """  # noqa E501  # pylint: disable=c0301
        if agg_method is not None and agg_method not in ["mean_abs", "median", "mean_sq"]:
            raise ValueError(
                f"Invalid agg_method {agg_method}." f" Please choose mean_abs, median, or mean_sq."
            )
        if num_clusters is not None and baseline is not None:
            raise ValueError(
                "Baseline and num_clusters cannot be provided together. "
                "Please specify one of the two."
            )
        self.shap_config = {
            "use_logit": use_logit,
            "save_local_shap_values": save_local_shap_values,
        }
        _set(baseline, "baseline", self.shap_config)
        _set(num_samples, "num_samples", self.shap_config)
        _set(agg_method, "agg_method", self.shap_config)
        _set(seed, "seed", self.shap_config)
        _set(num_clusters, "num_clusters", self.shap_config)
        if text_config:
            _set(text_config.get_text_config(), "text_config", self.shap_config)
            if not save_local_shap_values:
                logger.warning(
                    "Global aggregation is not yet supported for text features. "
                    "Consider setting save_local_shap_values=True to inspect local text "
                    "explanations."
                )
        if image_config:
            _set(image_config.get_image_config(), "image_config", self.shap_config)

    def get_explainability_config(self):
        """Returns a shap config dictionary."""
        return copy.deepcopy({"shap": self.shap_config})


class SageMakerClarifyProcessor(Processor):
    """Handles SageMaker Processing tasks to compute bias metrics and model explanations."""

    _CLARIFY_DATA_INPUT = "/opt/ml/processing/input/data"
    _CLARIFY_CONFIG_INPUT = "/opt/ml/processing/input/config"
    _CLARIFY_OUTPUT = "/opt/ml/processing/output"

    def __init__(
        self,
        role,
        instance_count,
        instance_type,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
        job_name_prefix=None,
        version=None,
    ):
        """Initializes a SageMakerClarifyProcessor to compute bias metrics and model explanations.

        Instance of :class:`~sagemaker.processing.Processor`.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            instance_count (int): The number of instances to run
                a processing job with.
            instance_type (str): The type of EC2 instance to use for
                processing, for example, ``'ml.c4.xlarge'``.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume (default: None).
            output_kms_key (str): The KMS key ID for processing job outputs (default: None).
            max_runtime_in_seconds (int): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If ``max_runtime_in_seconds`` is not
                specified, the default value is ``86400`` seconds (24 hours).
            sagemaker_session (:class:`~sagemaker.session.Session`):
                :class:`~sagemaker.session.Session` object which manages interactions
                with Amazon SageMaker and any other AWS services needed. If not specified,
                the Processor creates a :class:`~sagemaker.session.Session`
                using the default AWS configuration chain.
            env (dict[str, str]): Environment variables to be passed to
                the processing jobs (default: None).
            tags (list[dict]): List of tags to be passed to the processing job
                (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
            job_name_prefix (str): Processing job name prefix.
            version (str): Clarify version to use.
        """
        container_uri = image_uris.retrieve("clarify", sagemaker_session.boto_region_name, version)
        self.job_name_prefix = job_name_prefix
        super(SageMakerClarifyProcessor, self).__init__(
            role,
            container_uri,
            instance_count,
            instance_type,
            None,  # We manage the entrypoint.
            volume_size_in_gb,
            volume_kms_key,
            output_kms_key,
            max_runtime_in_seconds,
            None,  # We set method-specific job names below.
            sagemaker_session,
            env,
            tags,
            network_config,
        )

    def run(self, **_):
        """Overriding the base class method but deferring to specific run_* methods."""
        raise NotImplementedError(
            "Please choose a method of run_pre_training_bias, run_post_training_bias or "
            "run_explainability."
        )

    def _run(
        self,
        data_config,
        analysis_config,
        wait,
        logs,
        job_name,
        kms_key,
        experiment_config,
    ):
        """Runs a :class:`~sagemaker.processing.ProcessingJob` with the SageMaker Clarify container

        and analysis config.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            analysis_config (dict): Config following the analysis_config.json format.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                ``'ExperimentName'``, ``'TrialName'``, and ``'TrialComponentDisplayName'``.

                The behavior of setting these keys is as follows:
                * If ``'ExperimentName'`` is supplied but ``'TrialName'`` is not, a Trial will be
                  automatically created and the job's Trial Component associated with the Trial.
                * If ``'TrialName'`` is supplied and the Trial already exists,
                  the job's Trial Component will be associated with the Trial.
                * If both ``'ExperimentName'`` and ``'TrialName'`` are not supplied,
                  the Trial Component will be unassociated.
                * ``'TrialComponentDisplayName'`` is used for display in Amazon SageMaker Studio.
        """
        analysis_config["methods"]["report"] = {
            "name": "report",
            "title": "Analysis Report",
        }
        with tempfile.TemporaryDirectory() as tmpdirname:
            analysis_config_file = os.path.join(tmpdirname, "analysis_config.json")
            with open(analysis_config_file, "w") as f:
                json.dump(analysis_config, f)
            s3_analysis_config_file = _upload_analysis_config(
                analysis_config_file,
                data_config.s3_analysis_config_output_path or data_config.s3_output_path,
                self.sagemaker_session,
                kms_key,
            )
            config_input = ProcessingInput(
                input_name="analysis_config",
                source=s3_analysis_config_file,
                destination=self._CLARIFY_CONFIG_INPUT,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_compression_type="None",
            )
            data_input = ProcessingInput(
                input_name="dataset",
                source=data_config.s3_data_input_path,
                destination=self._CLARIFY_DATA_INPUT,
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type=data_config.s3_data_distribution_type,
                s3_compression_type=data_config.s3_compression_type,
            )
            result_output = ProcessingOutput(
                source=self._CLARIFY_OUTPUT,
                destination=data_config.s3_output_path,
                output_name="analysis_result",
                s3_upload_mode="EndOfJob",
            )

            return super().run(
                inputs=[data_input, config_input],
                outputs=[result_output],
                wait=wait,
                logs=logs,
                job_name=job_name,
                kms_key=kms_key,
                experiment_config=experiment_config,
            )

    def run_pre_training_bias(
        self,
        data_config,
        data_bias_config,
        methods="all",
        wait=True,
        logs=True,
        job_name=None,
        kms_key=None,
        experiment_config=None,
    ):
        """Runs a :class:`~sagemaker.processing.ProcessingJob` to compute pre-training bias methods

        Computes the requested ``methods`` on the input data. The ``methods`` compare
        metrics (e.g. fraction of examples) for the sensitive group(s) vs. the other examples.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            data_bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
            methods (str or list[str]): Selects a subset of potential metrics:
                ["`CI <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-bias-metric-class-imbalance.html>`_",
                "`DPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-true-label-imbalance.html>`_",
                "`KL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kl-divergence.html>`_",
                "`JS <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-jensen-shannon-divergence.html>`_",
                "`LP <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-lp-norm.html>`_",
                "`TVD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-total-variation-distance.html>`_",
                "`KS <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kolmogorov-smirnov.html>`_",
                "`CDDL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-cddl.html>`_"].
                Defaults to str "all" to run all metrics if left unspecified.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. When ``job_name`` is not specified,
                if ``job_name_prefix`` in :class:`~sagemaker.clarify.SageMakerClarifyProcessor` is
                specified, the job name will be the ``job_name_prefix`` and current timestamp;
                otherwise use ``"Clarify-Pretraining-Bias"`` as prefix.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                ``'ExperimentName'``, ``'TrialName'``, and ``'TrialComponentDisplayName'``.

                The behavior of setting these keys is as follows:

                * If ``'ExperimentName'`` is supplied but ``'TrialName'`` is not, a Trial will be
                  automatically created and the job's Trial Component associated with the Trial.
                * If ``'TrialName'`` is supplied and the Trial already exists,
                  the job's Trial Component will be associated with the Trial.
                * If both ``'ExperimentName'`` and ``'TrialName'`` are not supplied,
                  the Trial Component will be unassociated.
                * ``'TrialComponentDisplayName'`` is used for display in Amazon SageMaker Studio.
        """  # noqa E501  # pylint: disable=c0301
        analysis_config = data_config.get_config()
        analysis_config.update(data_bias_config.get_config())
        analysis_config["methods"] = {"pre_training_bias": {"methods": methods}}
        if job_name is None:
            if self.job_name_prefix:
                job_name = utils.name_from_base(self.job_name_prefix)
            else:
                job_name = utils.name_from_base("Clarify-Pretraining-Bias")
        return self._run(
            data_config,
            analysis_config,
            wait,
            logs,
            job_name,
            kms_key,
            experiment_config,
        )

    def run_post_training_bias(
        self,
        data_config,
        data_bias_config,
        model_config,
        model_predicted_label_config,
        methods="all",
        wait=True,
        logs=True,
        job_name=None,
        kms_key=None,
        experiment_config=None,
    ):
        """Runs a :class:`~sagemaker.processing.ProcessingJob` to compute posttraining bias

        Spins up a model endpoint and runs inference over the input dataset in
        the ``s3_data_input_path`` (from the :class:`~sagemaker.clarify.DataConfig`) to obtain
        predicted labels. Using model predictions, computes the requested posttraining bias
        ``methods`` that compare metrics (e.g. accuracy, precision, recall) for the
        sensitive group(s) versus the other examples.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            data_bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
            model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
                endpoint to be created.
            model_predicted_label_config (:class:`~sagemaker.clarify.ModelPredictedLabelConfig`):
                Config of how to extract the predicted label from the model output.
            methods (str or list[str]): Selector of a subset of potential metrics:
                ["`DPPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dppl.html>`_"
                , "`DI <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-di.html>`_",
                "`DCA <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dca.html>`_",
                "`DCR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dcr.html>`_",
                "`RD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-rd.html>`_",
                "`DAR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dar.html>`_",
                "`DRR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-drr.html>`_",
                "`AD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-ad.html>`_",
                "`CDDPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-cddpl.html>`_
                ", "`TE <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-te.html>`_",
                "`FT <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-ft.html>`_"].
                Defaults to str "all" to run all metrics if left unspecified.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. When ``job_name`` is not specified,
                if ``job_name_prefix`` in :class:`~sagemaker.clarify.SageMakerClarifyProcessor`
                is specified, the job name will be the ``job_name_prefix`` and current timestamp;
                otherwise use ``"Clarify-Posttraining-Bias"`` as prefix.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                ``'ExperimentName'``, ``'TrialName'``, and ``'TrialComponentDisplayName'``.

                The behavior of setting these keys is as follows:

                * If ``'ExperimentName'`` is supplied but ``'TrialName'`` is not, a Trial will be
                  automatically created and the job's Trial Component associated with the Trial.
                * If ``'TrialName'`` is supplied and the Trial already exists,
                  the job's Trial Component will be associated with the Trial.
                * If both ``'ExperimentName'`` and ``'TrialName'`` are not supplied,
                  the Trial Component will be unassociated.
                * ``'TrialComponentDisplayName'`` is used for display in Amazon SageMaker Studio.
        """  # noqa E501  # pylint: disable=c0301
        analysis_config = data_config.get_config()
        analysis_config.update(data_bias_config.get_config())
        (
            probability_threshold,
            predictor_config,
        ) = model_predicted_label_config.get_predictor_config()
        predictor_config.update(model_config.get_predictor_config())
        analysis_config["methods"] = {"post_training_bias": {"methods": methods}}
        analysis_config["predictor"] = predictor_config
        _set(probability_threshold, "probability_threshold", analysis_config)
        if job_name is None:
            if self.job_name_prefix:
                job_name = utils.name_from_base(self.job_name_prefix)
            else:
                job_name = utils.name_from_base("Clarify-Posttraining-Bias")
        return self._run(
            data_config,
            analysis_config,
            wait,
            logs,
            job_name,
            kms_key,
            experiment_config,
        )

    def run_bias(
        self,
        data_config,
        bias_config,
        model_config,
        model_predicted_label_config=None,
        pre_training_methods="all",
        post_training_methods="all",
        wait=True,
        logs=True,
        job_name=None,
        kms_key=None,
        experiment_config=None,
    ):
        """Runs a :class:`~sagemaker.processing.ProcessingJob` to compute the requested bias methods

        Computes metrics for both the pre-training and the post-training methods.
        To calculate post-training methods, it spins up a model endpoint and runs inference over the
        input examples in 's3_data_input_path' (from the :class:`~sagemaker.clarify.DataConfig`)
        to obtain predicted labels.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
            model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
                endpoint to be created.
            model_predicted_label_config (:class:`~sagemaker.clarify.ModelPredictedLabelConfig`):
                Config of how to extract the predicted label from the model output.
            pre_training_methods (str or list[str]): Selector of a subset of potential metrics:
                ["`CI <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-bias-metric-class-imbalance.html>`_",
                "`DPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-true-label-imbalance.html>`_",
                "`KL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kl-divergence.html>`_",
                "`JS <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-jensen-shannon-divergence.html>`_",
                "`LP <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-lp-norm.html>`_",
                "`TVD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-total-variation-distance.html>`_",
                "`KS <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kolmogorov-smirnov.html>`_",
                "`CDDL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-cddl.html>`_"].
                Defaults to str "all" to run all metrics if left unspecified.
            post_training_methods (str or list[str]): Selector of a subset of potential metrics:
                ["`DPPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dppl.html>`_"
                , "`DI <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-di.html>`_",
                "`DCA <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dca.html>`_",
                "`DCR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dcr.html>`_",
                "`RD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-rd.html>`_",
                "`DAR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-dar.html>`_",
                "`DRR <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-drr.html>`_",
                "`AD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-ad.html>`_",
                "`CDDPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-cddpl.html>`_
                ", "`TE <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-te.html>`_",
                "`FT <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-post-training-bias-metric-ft.html>`_"].
                Defaults to str "all" to run all metrics if left unspecified.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. When ``job_name`` is not specified,
                if ``job_name_prefix`` in :class:`~sagemaker.clarify.SageMakerClarifyProcessor` is
                specified, the job name will be ``job_name_prefix`` and the current timestamp;
                otherwise use ``"Clarify-Bias"`` as prefix.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                ``'ExperimentName'``, ``'TrialName'``, and ``'TrialComponentDisplayName'``.

                The behavior of setting these keys is as follows:

                * If ``'ExperimentName'`` is supplied but ``'TrialName'`` is not, a Trial will be
                  automatically created and the job's Trial Component associated with the Trial.
                * If ``'TrialName'`` is supplied and the Trial already exists,
                  the job's Trial Component will be associated with the Trial.
                * If both ``'ExperimentName'`` and ``'TrialName'`` are not supplied,
                  the Trial Component will be unassociated.
                * ``'TrialComponentDisplayName'`` is used for display in Amazon SageMaker Studio.
        """  # noqa E501  # pylint: disable=c0301
        analysis_config = data_config.get_config()
        analysis_config.update(bias_config.get_config())
        analysis_config["predictor"] = model_config.get_predictor_config()
        if model_predicted_label_config:
            (
                probability_threshold,
                predictor_config,
            ) = model_predicted_label_config.get_predictor_config()
            if predictor_config:
                analysis_config["predictor"].update(predictor_config)
            if probability_threshold is not None:
                analysis_config["probability_threshold"] = probability_threshold

        analysis_config["methods"] = {
            "pre_training_bias": {"methods": pre_training_methods},
            "post_training_bias": {"methods": post_training_methods},
        }
        if job_name is None:
            if self.job_name_prefix:
                job_name = utils.name_from_base(self.job_name_prefix)
            else:
                job_name = utils.name_from_base("Clarify-Bias")
        return self._run(
            data_config,
            analysis_config,
            wait,
            logs,
            job_name,
            kms_key,
            experiment_config,
        )

    def run_explainability(
        self,
        data_config,
        model_config,
        explainability_config,
        model_scores=None,
        wait=True,
        logs=True,
        job_name=None,
        kms_key=None,
        experiment_config=None,
    ):
        """Runs a :class:`~sagemaker.processing.ProcessingJob` computing feature attributions.

        Spins up a model endpoint.

        Currently, only SHAP and  Partial Dependence Plots (PDP) are supported
        as explainability methods.

        When SHAP is requested in the ``explainability_config``,
        the SHAP algorithm calculates the feature importance for each input example
        in the ``s3_data_input_path`` of the :class:`~sagemaker.clarify.DataConfig`,
        by creating ``num_samples`` copies of the example with a subset of features
        replaced with values from the ``baseline``.
        It then runs model inference to see how the model's prediction changes with the replaced
        features. If the model output returns multiple scores importance is computed for each score.
        Across examples, feature importance is aggregated using ``agg_method``.

        When PDP is requested in the ``explainability_config``,
        the PDP algorithm calculates the dependence of the target response
        on the input features and marginalizes over the values of all other input features.
        The Partial Dependence Plots are included in the output
        `report <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-feature-attribute-baselines-reports.html>`__
        and the corresponding values are included in the analysis output.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
                endpoint to be created.
            explainability_config (:class:`~sagemaker.clarify.ExplainabilityConfig` or list):
                Config of the specific explainability method or a list of
                :class:`~sagemaker.clarify.ExplainabilityConfig` objects.
                Currently, SHAP and PDP are the two methods supported.
            model_scores (int or str or :class:`~sagemaker.clarify.ModelPredictedLabelConfig`):
                Index or JSONPath to locate the predicted scores in the model output. This is not
                required if the model output is a single score. Alternatively, it can be an instance
                of :class:`~sagemaker.clarify.SageMakerClarifyProcessor`
                to provide more parameters like ``label_headers``.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. When ``job_name`` is not specified,
                if ``job_name_prefix`` in :class:`~sagemaker.clarify.SageMakerClarifyProcessor`
                is specified, the job name will be composed of ``job_name_prefix`` and current
                timestamp; otherwise use ``"Clarify-Explainability"`` as prefix.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                ``'ExperimentName'``, ``'TrialName'``, and ``'TrialComponentDisplayName'``.

                The behavior of setting these keys is as follows:

                * If ``'ExperimentName'`` is supplied but ``'TrialName'`` is not, a Trial will be
                  automatically created and the job's Trial Component associated with the Trial.
                * If ``'TrialName'`` is supplied and the Trial already exists,
                  the job's Trial Component will be associated with the Trial.
                * If both ``'ExperimentName'`` and ``'TrialName'`` are not supplied,
                  the Trial Component will be unassociated.
                * ``'TrialComponentDisplayName'`` is used for display in Amazon SageMaker Studio.
        """  # noqa E501  # pylint: disable=c0301
        analysis_config = data_config.get_config()
        predictor_config = model_config.get_predictor_config()
        if isinstance(model_scores, ModelPredictedLabelConfig):
            (
                probability_threshold,
                predicted_label_config,
            ) = model_scores.get_predictor_config()
            _set(probability_threshold, "probability_threshold", analysis_config)
            predictor_config.update(predicted_label_config)
        else:
            _set(model_scores, "label", predictor_config)

        explainability_methods = {}
        if isinstance(explainability_config, list):
            if len(explainability_config) == 0:
                raise ValueError("Please provide at least one explainability config.")
            for config in explainability_config:
                explain_config = config.get_explainability_config()
                explainability_methods.update(explain_config)
            if not len(explainability_methods.keys()) == len(explainability_config):
                raise ValueError("Duplicate explainability configs are provided")
            if (
                "shap" not in explainability_methods
                and explainability_methods["pdp"].get("features", None) is None
            ):
                raise ValueError("PDP features must be provided when ShapConfig is not provided")
        else:
            if (
                isinstance(explainability_config, PDPConfig)
                and explainability_config.get_explainability_config()["pdp"].get("features", None)
                is None
            ):
                raise ValueError("PDP features must be provided when ShapConfig is not provided")
            explainability_methods = explainability_config.get_explainability_config()
        analysis_config["methods"] = explainability_methods
        analysis_config["predictor"] = predictor_config
        if job_name is None:
            if self.job_name_prefix:
                job_name = utils.name_from_base(self.job_name_prefix)
            else:
                job_name = utils.name_from_base("Clarify-Explainability")
        return self._run(
            data_config,
            analysis_config,
            wait,
            logs,
            job_name,
            kms_key,
            experiment_config,
        )


def _upload_analysis_config(analysis_config_file, s3_output_path, sagemaker_session, kms_key):
    """Uploads the local ``analysis_config_file`` to the ``s3_output_path``.

    Args:
        analysis_config_file (str): File path to the local analysis config file.
        s3_output_path (str): S3 prefix to store the analysis config file.
        sagemaker_session (:class:`~sagemaker.session.Session`):
            :class:`~sagemaker.session.Session` object which manages interactions with
            Amazon SageMaker and any other AWS services needed. If not specified,
            the processor creates a :class:`~sagemaker.session.Session`
            using the default AWS configuration chain.
        kms_key (str): The ARN of the KMS key that is used to encrypt the
            user code file (default: None).

    Returns:
        The S3 URI of the uploaded file.
    """
    return s3.S3Uploader.upload(
        local_path=analysis_config_file,
        desired_s3_uri=s3_output_path,
        sagemaker_session=sagemaker_session,
        kms_key=kms_key,
    )


def _set(value, key, dictionary):
    """Sets dictionary[key] = value if value is not None."""
    if value is not None:
        dictionary[key] = value
