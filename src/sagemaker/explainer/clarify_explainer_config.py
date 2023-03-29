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
"""A class for SageMaker Clarify explainer config.

Used for configuring online explainability with SageMaker Clarify.
"""

from __future__ import print_function, absolute_import
from typing import List, Optional


class ClarifyTextConfig(object):
    """Configuration to explain natural language processing (NLP) explainability."""

    def __init__(
        self,
        granularity: str,
        language: str,
    ):
        """Initialize a config object for text explainability.

        Args:
            granularity (str): The unit of granularity for the analysis of text features. For
                example, if the unit is ``"token"``, then each token (like a word in English) of the
                text is treated as a feature. SHAP values are computed for each unit/feature.
                Accepted values are ``"token"``, ``"sentence"``, or ``"paragraph"``.
            language (str): Specifies the language of the text features in ISO 639-1 or ISO 639-3
                code of a supported language. See valid values `here
                <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ClarifyTextConfig.html#sagemaker-Type-ClarifyTextConfig-Language>`_.
        """  # noqa E501  # pylint: disable=line-too-long
        self.granularity = granularity
        self.language = language

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {"Granularity": self.granularity, "Language": self.language}
        return request_dict


class ClarifyShapBaselineConfig(object):
    """Configuration for Shap baseline of the Kernal SHAP algorithm.

    `SHAP baseline
    <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-feature-attribute-shap-baselines.html>`_
    also called the background or reference dataset. The number of records in the baseline data
    determines the size of the synthetic dataset, which has an impact on latency of explainability
    requests. For more information, see the `Synthetic data
    <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-online-explainability-create-endpoint.html#clarify-online-explainability-create-endpoint-synthetic>`_.
    """  # noqa E501  # pylint: disable=line-too-long

    def __init__(
        self,
        mime_type: Optional[str] = "text/csv",
        shap_baseline: Optional[str] = None,
        shap_baseline_uri: Optional[str] = None,
    ):
        """Initialize a config object for SHAP baseline.

        Args:
            mime_type (str): Optional. The MIME type of the baseline data. Choose
                from ``"text/csv"`` or ``"application/jsonlines"``. (Default: ``"text/csv"``)
            shap_baseline (str): Optional. The inline SHAP baseline data in string format.
                ShapBaseline can have one or multiple records to be used as the baseline dataset.
                The format of the SHAP baseline file should be the same format as the training
                dataset. For example, if the training dataset is in CSV format and each record
                contains four features, and all features are numerical, then the format of the
                baseline data should also share these characteristics. For NLP of text columns, the
                baseline value should be the value used to replace the unit of text specified by
                the ``granularity`` of the
                :class:`~sagemaker.explainer.clarify_explainer_config.ClarifyTextConfig`
                parameter. The size limit for ``shap_baseline`` is 4 KB. Use the
                ``shap_baseline_uri`` parameter if you want to provide more than 4 KB of baseline
                data.
            shap_baseline_uri (str): Optional. The S3 URI where the SHAP baseline file is stored.
                The format of the SHAP baseline file should be the same format as the format of
                the training dataset. For example, if the training dataset is in CSV format,
                and each record in the training dataset has four features, and all features are
                numerical, then the baseline file should also have this same format. Each record
                should contain only the features.
        """
        self.mime_type = mime_type
        self.shap_baseline = shap_baseline
        self.shap_baseline_uri = shap_baseline_uri

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {"MimeType": self.mime_type}
        if self.shap_baseline is not None:
            request_dict["ShapBaseline"] = self.shap_baseline
        if self.shap_baseline_uri is not None:
            request_dict["ShapBaselineUri"] = self.shap_baseline_uri

        return request_dict


class ClarifyShapConfig(object):
    """Configuration for SHAP analysis using SageMaker Clarify Explainer."""

    def __init__(
        self,
        shap_baseline_config: ClarifyShapBaselineConfig,
        number_of_samples: Optional[int] = None,
        seed: Optional[int] = None,
        use_logit: Optional[bool] = False,
        text_config: Optional[ClarifyTextConfig] = None,
    ):
        """Initialize a config object for SHAP analysis.

        Args:
            shap_baseline_config (:class:`~sagemaker.explainer.clarify_explainer_config.ClarifyShapBaselineConfig`):
                The configuration for the SHAP baseline of the Kernal SHAP algorithm.
            number_of_samples (int): Optional. Number of samples to be used for analysis by the
                Kernal SHAP algorithm.
            seed (int): Optional. Seed value to get deterministic SHAP result.
            use_logit (bool): Optional. A Boolean toggle to indicate if you want to use the logit
                function (true) or log-odds units (false) for model predictions. (Default: false)
            text_config (:class:`~sagemaker.explainer.clarify_explainer_config.ClarifyTextConfig`):
                Optional. A parameter that indicates if text features are treated as text and
                explanations are provided for individual units of text. Required for NLP
                explainability only.
        """  # noqa E501  # pylint: disable=line-too-long
        self.number_of_samples = number_of_samples
        self.seed = seed
        self.shap_baseline_config = shap_baseline_config
        self.text_config = text_config
        self.use_logit = use_logit

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {
            "ShapBaselineConfig": self.shap_baseline_config._to_request_dict(),
            "UseLogit": self.use_logit,
        }
        if self.number_of_samples is not None:
            request_dict["NumberOfSamples"] = self.number_of_samples

        if self.seed is not None:
            request_dict["Seed"] = self.seed

        if self.text_config is not None:
            request_dict["TextConfig"] = self.text_config._to_request_dict()

        return request_dict


class ClarifyInferenceConfig(object):
    """Configuration object for the model container."""

    def __init__(
        self,
        content_template: Optional[str] = None,
        feature_headers: Optional[List[str]] = None,
        features_attribute: Optional[str] = None,
        feature_types: Optional[List[str]] = None,
        label_attribute: Optional[str] = None,
        label_headers: Optional[List[str]] = None,
        label_index: Optional[int] = None,
        max_payload_in_mb: Optional[int] = 6,
        max_record_count: Optional[int] = None,
        probability_attribute: Optional[str] = None,
        probability_index: Optional[int] = None,
    ):
        """Initialize a config object for model container.

        Args:
            content_template (str): Optional. A template string used to format a JSON record into an
                acceptable model container input. For example, a ContentTemplate string ``'{
                "myfeatures":$features}'`` will format a list of features ``[1,2,3]`` into the
                record string ``'{"myfeatures":[1,2,3]}'``. Required only when the model
                container input is in JSON Lines format.
            feature_headers (list[str]): Optional. The names of the features. If provided, these are
                included in the endpoint response payload to help readability of the
                ``InvokeEndpoint`` output.
            features_attribute (str): Optional. Provides the JMESPath expression to extract the
                features from a model container input in JSON Lines format. For example,
                if ``features_attribute`` is the JMESPath expression ``'myfeatures'``, it extracts a
                list of features ``[1,2,3]`` from request data ``'{"myfeatures":[1,2,3]}'``.
            feature_types (list[str]): Optional. A list of data types of the features. Applicable
                only to NLP explainability. If provided, ``feature_types`` must have at least one
                ``'text'`` string (for example, ``['text']``). If ``feature_types`` is not provided,
                the explainer infers the feature types based on the baseline data. The feature
                types are included in the endpoint response payload.
            label_attribute (str): Optional. A JMESPath expression used to locate the list of label
                headers in the model container output.
            label_headers (list[str]): Optional. For multiclass classification problems, the label
                headers are the names of the classes. Otherwise, the label header is the name of
                the predicted label. These are used to help readability for the output of the
                ``InvokeEndpoint`` API.
            label_index (int): Optional. A zero-based index used to extract a label header or list
                of label headers from model container output in CSV format.
            max_payload_in_mb (int): Optional. The maximum payload size (MB) allowed of a request
                from the explainer to the model container. (Default: 6)
            max_record_count (int): Optional. The maximum number of records in a request that the
                model container can process when querying the model container for the predictions
                of a synthetic dataset. A record is a unit of input data that inference can be made
                on, for example, a single line in CSV data. If ``max_record_count`` is 1, the model
                container expects one record per request. A value of 2 or greater means that the
                model expects batch requests, which can reduce overhead and speed up the
                inferencing process. If this parameter is not provided, the explainer will tune
                the record count per request according to the model container's capacity at runtime.
            probability_attribute (str): Optional. A JMESPath expression used to extract the
                probability (or score) from the model container output if the model container
                is in JSON Lines format.
            probability_index (int): Optional. A zero-based index used to extract a probability
                value (score) or list from model container output in CSV format. If this value is
                not provided, the entire model container output will be treated as a probability
                value (score) or list.
        """
        self.content_template = content_template
        self.feature_headers = feature_headers
        self.features_attribute = features_attribute
        self.feature_types = feature_types
        self.label_attribute = label_attribute
        self.label_headers = label_headers
        self.label_index = label_index
        self.max_payload_in_mb = max_payload_in_mb
        self.max_record_count = max_record_count
        self.probability_attribute = probability_attribute
        self.probability_index = probability_index

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {}
        if self.content_template is not None:
            request_dict["ContentTemplate"] = self.content_template
        if self.feature_headers is not None and self.feature_headers:
            request_dict["FeatureHeaders"] = self.feature_headers
        if self.features_attribute is not None:
            request_dict["FeaturesAttribute"] = self.features_attribute
        if self.feature_types is not None:
            request_dict["FeatureTypes"] = self.feature_types
        if self.label_attribute is not None:
            request_dict["LabelAttribute"] = self.label_attribute
        if self.label_headers is not None:
            request_dict["LabelHeaders"] = self.label_headers
        if self.label_index is not None:
            request_dict["LabelIndex"] = self.label_index
        if self.max_payload_in_mb is not None:
            request_dict["MaxPayloadInMB"] = self.max_payload_in_mb
        if self.max_record_count is not None:
            request_dict["MaxRecordCount"] = self.max_record_count
        if self.probability_attribute is not None:
            request_dict["ProbabilityAttribute"] = self.probability_attribute
        if self.probability_index is not None:
            request_dict["ProbabilityIndex"] = self.probability_index
        return request_dict


class ClarifyExplainerConfig(object):
    """A member of :class:`~sagemaker.explainer.explainer_config.ExplainerConfig`.

    Configuration to analyze explainability with SageMaker Clarify explainer.
    """

    def __init__(
        self,
        shap_config: ClarifyShapConfig,
        enable_explanations: Optional[str] = None,
        inference_config: Optional[ClarifyInferenceConfig] = None,
    ):
        """Initialize a config object for online explainability with AWS SageMaker Clarify.

        Args:
            shap_config (:class:`~sagemaker.explainer.clarify_explainer_config.ClarifyShapConfig`):
                The configuration for SHAP analysis.
            enable_explanations (str): Optional. A `JMESPath boolean expression
                <https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-online-explainability-create-endpoint.html#clarify-online-explainability-create-endpoint-enable>`_
                used to filter which records to explain (Default: None). If not specified,
                explanations are activated by default.
            inference_config (:class:`~sagemaker.explainer.clarify_explainer_config.ClarifyInferenceConfig`):
                The inference configuration parameter for the model container. (Default: None)
        """  # noqa E501  # pylint: disable=line-too-long
        self.enable_explanations = enable_explanations
        self.shap_config = shap_config
        self.inference_config = inference_config

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        request_dict = {
            "ShapConfig": self.shap_config._to_request_dict(),
        }

        if self.enable_explanations is not None:
            request_dict["EnableExplanations"] = self.enable_explanations

        if self.inference_config is not None:
            request_dict["InferenceConfig"] = self.inference_config._to_request_dict()

        return request_dict
