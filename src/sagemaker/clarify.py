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
"""This module configures the SageMaker Clarify bias and model explainability processor job."""
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
        s3_data_distribution_type="FullyReplicated",
        s3_compression_type="None",
        joinsource=None,
    ):
        """Initializes a configuration of both input and output datasets.

        Args:
            s3_data_input_path (str): Dataset S3 prefix/object URI.
            s3_output_path (str): S3 prefix to store the output.
            s3_analysis_config_output_path (str): S3 prefix to store the analysis_config output
                If this field is None, then the s3_output_path will be used
                to store the analysis_config output
            label (str): Target attribute of the model required by bias metrics (optional for SHAP)
                Specified as column name or index for CSV dataset, or as JSONPath for JSONLines.
            headers (list[str]): A list of column names in the input dataset.
            features (str): JSONPath for locating the feature columns for bias metrics if the
                dataset format is JSONLines.
            dataset_type (str): Format of the dataset. Valid values are "text/csv" for CSV,
                "application/jsonlines" for JSONLines, and "application/x-parquet" for Parquet.
            s3_data_distribution_type (str): Valid options are "FullyReplicated" or
                "ShardedByS3Key".
            s3_compression_type (str): Valid options are "None" or "Gzip".
            joinsource (str): The name or index of the column in the dataset that acts as an
                identifier column (for instance, while performing a join). This column is only
                used as an identifier, and not used for any other computations. This is an
                optional field in all cases except when the dataset contains more than one file,
                and `save_local_shap_values` is set to true in SHAPConfig.
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
        self.s3_data_distribution_type = s3_data_distribution_type
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
    """Config object related to bias configurations of the input dataset."""

    def __init__(
        self,
        label_values_or_threshold,
        facet_name,
        facet_values_or_threshold=None,
        group_name=None,
    ):
        """Initializes a configuration of the sensitive groups in the dataset.

        Args:
            label_values_or_threshold (Any): List of label values or threshold to indicate positive
                outcome used for bias metrics.
            facet_name (str or [str]): String or List of strings of sensitive attribute(s) in the
            input data for which we like to compare metrics.
            facet_values_or_threshold (list): Optional list of values to form a sensitive group or
                threshold for a numeric facet column that defines the lower bound of a sensitive
                group. Defaults to considering each possible value as sensitive group and
                computing metrics vs all the other examples.
                If facet_name is a list, this needs to be None or a List consisting of lists or None
                with the same length as facet_name list.
            group_name (str): Optional column name or index to indicate a group column to be used
                for the bias metric 'Conditional Demographic Disparity in Labels - CDDL' or
                'Conditional Demographic Disparity in Predicted Labels - CDDPL'.
        """
        if isinstance(facet_name, str):
            facet = {"name_or_index": facet_name}
            _set(facet_values_or_threshold, "value_or_threshold", facet)
            facet_list = [facet]
        elif facet_values_or_threshold is None or len(facet_name) == len(facet_values_or_threshold):
            facet_list = []
            for i, single_facet_name in enumerate(facet_name):
                facet = {"name_or_index": single_facet_name}
                if facet_values_or_threshold is not None:
                    _set(facet_values_or_threshold[i], "value_or_threshold", facet)
                facet_list.append(facet)
        else:
            raise ValueError("Wrong combination of argument values passed")
        self.analysis_config = {
            "label_values_or_threshold": label_values_or_threshold,
            "facet": facet_list,
        }
        _set(group_name, "group_variable", self.analysis_config)

    def get_config(self):
        """Returns part of an analysis config dictionary."""
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
    ):
        r"""Initializes a configuration of a model and the endpoint to be created for it.

        Args:
            model_name (str): Model name (as created by 'CreateModel').
            instance_count (int): The number of instances of a new endpoint for model inference.
            instance_type (str): The type of EC2 instance to use for model inference,
                for example, 'ml.c5.xlarge'.
            accept_type (str): The model output format to be used for getting inferences with the
                shadow endpoint. Valid values are "text/csv" for CSV and "application/jsonlines".
                Default is the same as content_type.
            content_type (str): The model input format to be used for getting inferences with the
                shadow endpoint. Valid values are "text/csv" for CSV and "application/jsonlines".
                Default is the same as dataset format.
            content_template (str): A template string to be used to construct the model input from
                dataset instances. It is only used when "model_content_type" is
                "application/jsonlines". The template should have one and only one placeholder
                $features which will be replaced by a features list for to form the model inference
                input.
            custom_attributes (str): Provides additional information about a request for an
                inference submitted to a model hosted at an Amazon SageMaker endpoint. The
                information is an opaque value that is forwarded verbatim. You could use this
                value, for example, to provide an ID that you can use to track a request or to
                provide other metadata that a service endpoint was programmed to process. The value
                must consist of no more than 1024 visible US-ASCII characters as specified in
                Section 3.3.6. Field Value Components (
                https://tools.ietf.org/html/rfc7230#section-3.2.6) of the Hypertext Transfer
                Protocol (HTTP/1.1).
            accelerator_type (str): The Elastic Inference accelerator type to deploy to the model
                endpoint instance for making inferences to the model, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
            endpoint_name_prefix (str): The endpoint name prefix of a new endpoint. Must follow
                pattern "^[a-zA-Z0-9](-\*[a-zA-Z0-9]".
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
            * Regression Task: The model returns the score, e.g. 1.2. we don't need to specify
                anything. For json output, e.g. {'score': 1.2} we can set 'label='score''.

            * Binary classification:
                * The model returns a single probability and we would like to classify as 'yes'
                    those with a probability exceeding 0.2.
                    We can set 'probability_threshold=0.2, label_headers='yes''.
                * The model returns {'probability': 0.3}, for which we would like to apply a
                    threshold of 0.5 to obtain a predicted label in {0, 1}. In this case we can set
                    'label='probability''.
                * The model returns a tuple of the predicted label and the probability.
                    In this case we can set 'label=0'.

            * Multiclass classification:
                * The model returns
                    {'labels': ['cat', 'dog', 'fish'], 'probabilities': [0.35, 0.25, 0.4]}.
                    In this case we would set the 'probability='probabilities'' and
                    'label='labels'' and infer the predicted label to be 'fish.'
                * The model returns {'predicted_label': 'fish', 'probabilities': [0.35, 0.25, 0.4]}.
                    In this case we would set the 'label='predicted_label''.
                * The model returns [0.35, 0.25, 0.4]. In this case, we can set
                    'label_headers=['cat','dog','fish']' and infer the predicted label to be 'fish.'

        Args:
            label (str or int): Index or JSONPath location in the model output for the prediction.
                In case, this is a predicted label of the same type as the label in the dataset,
                no further arguments need to be specified.
            probability (str or int): Index or JSONPath location in the model output
                for the predicted score(s).
            probability_threshold (float): An optional value for binary prediction tasks in which
                the model returns a probability, to indicate the threshold to convert the
                prediction to a boolean value. Default is 0.5.
            label_headers (list): List of label values - one for each score of the ``probability``.
        """
        self.label = label
        self.probability = probability
        self.probability_threshold = probability_threshold
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
        """Returns probability_threshold, predictor config."""
        return self.probability_threshold, copy.deepcopy(self.predictor_config)


class ExplainabilityConfig(ABC):
    """Abstract config class to configure an explainability method."""

    @abstractmethod
    def get_explainability_config(self):
        """Returns config."""
        return None


class PDPConfig(ExplainabilityConfig):
    """Config class for Partial Dependence Plots (PDP).

    If PDP is requested, the Partial Dependence Plots will be included in the report, and the
    corresponding values will be included in the analysis output.
    """

    def __init__(self, features=None, grid_resolution=15, top_k_features=10):
        """Initializes config for PDP.

        Args:
            features (None or list): List of features names or indices for which partial dependence
                plots must be computed and plotted. When ShapConfig is provided, this parameter is
                optional as Clarify will try to compute the partial dependence plots for top
                feature based on SHAP attributions. When ShapConfig is not provided, 'features'
                must be provided.
            grid_resolution (int): In case of numerical features, this number represents that
                number of buckets that range of values must be divided into. This decides the
                granularity of the grid in which the PDP are plotted.
            top_k_features (int): Set the number of top SHAP attributes to be selected to compute
                partial dependence plots.
        """
        self.pdp_config = {"grid_resolution": grid_resolution, "top_k_features": top_k_features}
        if features is not None:
            self.pdp_config["features"] = features

    def get_explainability_config(self):
        """Returns config."""
        return copy.deepcopy({"pdp": self.pdp_config})


class TextConfig:
    """Config object to handle text features.

    The SHAP analysis will break down longer text into chunks (e.g. tokens, sentences, or paragraphs
    ) and replace them with the strings specified in the baseline for that feature. The shap value
    of a chunk then captures how much replacing it affects the prediction.
    """

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

        Args: granularity (str): Determines the granularity in which text features are broken down
        to, can be "token", "sentence", or "paragraph". Shap values are computed for these units.
        language (str): Specifies the language of the text features, can be "chinese", "danish",
        "dutch", "english", "french", "german", "greek", "italian", "japanese", "lithuanian",
        "multi-language", "norwegian bokmål", "polish", "portuguese", "romanian", "russian",
        "spanish", "afrikaans", "albanian", "arabic", "armenian", "basque", "bengali", "bulgarian",
        "catalan", "croatian", "czech", "estonian", "finnish", "gujarati", "hebrew", "hindi",
        "hungarian", "icelandic", "indonesian", "irish", "kannada", "kyrgyz", "latvian", "ligurian",
        "luxembourgish", "macedonian", "malayalam", "marathi", "nepali", "persian", "sanskrit",
        "serbian", "setswana", "sinhala", "slovak", "slovenian", "swedish", "tagalog", "tamil",
        "tatar", "telugu", "thai", "turkish", "ukrainian", "urdu", "vietnamese", "yoruba". Use
        "multi-language" for a mix of mulitple languages.
        """
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
        """Returns part of an analysis config dictionary."""
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
        """Initializes all configuration parameters needed for SHAP CV explainability

        Args:
            model_type (str): Specifies the type of CV model. Options:
            (IMAGE_CLASSIFICATION | OBJECT_DETECTION).
            num_segments (None or int): Clarify uses SKLearn's SLIC method for image segmentation
            to generate features/superpixels. num_segments specifies approximate
            number of segments to be generated. Default is None. SLIC will default to
            100 segments.
            feature_extraction_method (None or str): method used for extracting features from the
            image.ex. "segmentation". Default is segmentation.
            segment_compactness (None or float): Balances color proximity and space proximity.
            Higher values give more weight to space proximity, making superpixel
            shapes more square/cubic. We recommend exploring possible values on a log
            scale, e.g., 0.01, 0.1, 1, 10, 100, before refining around a chosen value.
            max_objects (None or int): maximum number of objects displayed. Object detection
            algorithm may detect more than max_objects number of objects in a single
            image. The top max_objects number of objects according to confidence score
            will be displayed.
            iou_threshold (None or float): minimum intersection over union for the object
            bounding box to consider its confidence score for computing SHAP values [0.0, 1.0].
            This parameter is used for the object detection case.
            context (None or float): refers to the portion of the image outside of the bounding box.
            Scale is [0.0, 1.0]. If set to 1.0, whole image is considered, if set to
            0.0 only the image inside bounding box is considered.
        """
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
    """Config class of SHAP."""

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
        """Initializes config for SHAP.

        Args:
            baseline (None or str or list): None or S3 object Uri or A list of rows (at least one)
                to be used asthe baseline dataset in the Kernel SHAP algorithm. The format should
                be the same as the dataset format. Each row should contain only the feature
                columns/values and omit the label column/values. If None a baseline will be
                calculated automatically by using K-means or K-prototypes in the input dataset.
            num_samples (None or int): Number of samples to be used in the Kernel SHAP algorithm.
                This number determines the size of the generated synthetic dataset to compute the
                SHAP values. If not provided then Clarify job will choose a proper value according
                to the count of features.
            agg_method (None or str): Aggregation method for global SHAP values. Valid values are
                "mean_abs" (mean of absolute SHAP values for all instances),
                "median" (median of SHAP values for all instances) and
                "mean_sq" (mean of squared SHAP values for all instances).
                If not provided then Clarify job uses method "mean_abs"
            use_logit (bool): Indicator of whether the logit function is to be applied to the model
                predictions. Default is False. If "use_logit" is true then the SHAP values will
                have log-odds units.
            save_local_shap_values (bool): Indicator of whether to save the local SHAP values
                in the output location. Default is True.
            seed (int): seed value to get deterministic SHAP values. Default is None.
            num_clusters (None or int): If a baseline is not provided, Clarify automatically
                computes a baseline dataset via a clustering algorithm (K-means/K-prototypes).
                num_clusters is a parameter for this algorithm. num_clusters will be the resulting
                size of the baseline dataset. If not provided, Clarify job will use a default value.
            text_config (:class:`~sagemaker.clarify.TextConfig`): Config to handle text features.
                Default is None
            image_config (:class:`~sagemaker.clarify.ImageConfig`): Config to handle image features.
                Default is None
        """
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
        """Returns config."""
        return copy.deepcopy({"shap": self.shap_config})


class SageMakerClarifyProcessor(Processor):
    """Handles SageMaker Processing task to compute bias metrics and explain a model."""

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
        """Initializes a ``Processor`` instance, computing bias metrics and model explanations.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            instance_count (int): The number of instances to run
                a processing job with.
            instance_type (str): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume (default: None).
            output_kms_key (str): The KMS key ID for processing job outputs (default: None).
            max_runtime_in_seconds (int): Timeout in seconds (default: None).
                After this amount of time, Amazon SageMaker terminates the job,
                regardless of its current status. If `max_runtime_in_seconds` is not
                specified, the default value is 24 hours.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed. If not specified, the processor creates
                one using the default AWS configuration chain.
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
            version (str): Clarify version want to be used.
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
        """Runs a ProcessingJob with the Sagemaker Clarify container and an analysis config.

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
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
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
            super().run(
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
        """Runs a ProcessingJob to compute the pre-training bias methods of the input data.

        Computes the requested methods that compare 'methods' (e.g. fraction of examples) for the
        sensitive group vs the other examples.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            data_bias_config (:class:`~sagemaker.clarify.BiasConfig`): Config of sensitive groups.
            methods (str or list[str]): Selector of a subset of potential metrics:
                ["`CI <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-bias-metric-class-imbalance.html>`_",
                "`DPL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-true-label-imbalance.html>`_",
                "`KL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kl-divergence.html>`_",
                "`JS <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-jensen-shannon-divergence.html>`_",
                "`LP <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-lp-norm.html>`_",
                "`TVD <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-total-variation-distance.html>`_",
                "`KS <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kolmogorov-smirnov.html>`_",
                "`CDDL <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-cddl.html>`_"].
                Defaults to computing all.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. When ``job_name`` is not specified, if
                ``job_name_prefix`` in :class:`SageMakerClarifyProcessor` specified, the job name
                will be composed of ``job_name_prefix`` and current timestamp; otherwise use
                "Clarify-Pretraining-Bias" as prefix.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
        """  # noqa E501
        analysis_config = data_config.get_config()
        analysis_config.update(data_bias_config.get_config())
        analysis_config["methods"] = {"pre_training_bias": {"methods": methods}}
        if job_name is None:
            if self.job_name_prefix:
                job_name = utils.name_from_base(self.job_name_prefix)
            else:
                job_name = utils.name_from_base("Clarify-Pretraining-Bias")
        self._run(
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
        """Runs a ProcessingJob to compute the post-training bias methods of the model predictions.

        Spins up a model endpoint, runs inference over the input example in the
        's3_data_input_path' to obtain predicted labels. Computes a the requested methods that
        compare 'methods' (e.g. accuracy, precision, recall) for the sensitive group vs the other
        examples.

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
                Defaults to computing all.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. When ``job_name`` is not specified, if
                ``job_name_prefix`` in :class:`SageMakerClarifyProcessor` specified, the job name
                will be composed of ``job_name_prefix`` and current timestamp; otherwise use
                "Clarify-Posttraining-Bias" as prefix.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
        """
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
        self._run(
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
        """Runs a ProcessingJob to compute the requested bias methods.

        It computes the metrics of both the pre-training methods and the post-training methods.
        To calculate post-training methods, it needs to spin up a model endpoint, runs inference
        over the input example in the 's3_data_input_path' to obtain predicted labels.

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
                Defaults to computing all.
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
                Defaults to computing all.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. When ``job_name`` is not specified, if
                ``job_name_prefix`` in :class:`SageMakerClarifyProcessor` specified, the job name
                will be composed of ``job_name_prefix`` and current timestamp; otherwise use
                "Clarify-Bias" as prefix.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
        """  # noqa E501
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
        self._run(
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
        """Runs a ProcessingJob computing for each example in the input the feature importance.

        Currently, only SHAP is supported as explainability method.

        Spins up a model endpoint.
        For each input example in the 's3_data_input_path' the SHAP algorithm determines
        feature importance, by creating 'num_samples' copies of the example with a subset
        of features replaced with values from the 'baseline'.
        Model inference is run to see how the prediction changes with the replaced features.
        If the model output returns multiple scores importance is computed for each of them.
        Across examples, feature importance is aggregated using 'agg_method'.

        Args:
            data_config (:class:`~sagemaker.clarify.DataConfig`): Config of the input/output data.
            model_config (:class:`~sagemaker.clarify.ModelConfig`): Config of the model and its
                endpoint to be created.
            explainability_config (:class:`~sagemaker.clarify.ExplainabilityConfig` or list):
                Config of the specific explainability method or a list of ExplainabilityConfig
                objects. Currently, SHAP and PDP are the two methods supported.
            model_scores(str|int|ModelPredictedLabelConfig):  Index or JSONPath location in the
                model output for the predicted scores to be explained. This is not required if the
                model output is a single score. Alternatively, an instance of
                ModelPredictedLabelConfig can be provided.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when ``wait`` is True (default: True).
            job_name (str): Processing job name. When ``job_name`` is not specified, if
                ``job_name_prefix`` in :class:`SageMakerClarifyProcessor` specified, the job name
                will be composed of ``job_name_prefix`` and current timestamp; otherwise use
                "Clarify-Explainability" as prefix.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
        """
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
        self._run(
            data_config,
            analysis_config,
            wait,
            logs,
            job_name,
            kms_key,
            experiment_config,
        )


def _upload_analysis_config(analysis_config_file, s3_output_path, sagemaker_session, kms_key):
    """Uploads the local analysis_config_file to the s3_output_path.

    Args:
        analysis_config_file (str): File path to the local analysis config file.
        s3_output_path (str): S3 prefix to store the analysis config file.
        sagemaker_session (:class:`~sagemaker.session.Session`):
            Session object which manages interactions with Amazon SageMaker and
            any other AWS services needed. If not specified, the processor creates
            one using the default AWS configuration chain.
        kms_key (str): The ARN of the KMS key that is used to encrypt the
            user code file (default: None).

    Returns:
        The S3 uri of the uploaded file.
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
