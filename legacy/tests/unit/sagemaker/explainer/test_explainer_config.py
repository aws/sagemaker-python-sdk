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

from sagemaker.explainer import (
    ExplainerConfig,
    ClarifyExplainerConfig,
    ClarifyShapConfig,
    ClarifyInferenceConfig,
    ClarifyShapBaselineConfig,
    ClarifyTextConfig,
)


OPTIONAL_MIME_TYPE = "application/jsonlines"
DEFAULT_MIME_TYPE = "text/csv"
SHAP_BASELINE = '1,2,3,"good product"'
SHAP_BASELINE_PATH = "s3://testbucket/baseline.csv"
OPTIONAL_LABEL_HEADERS = ["Label1", "Label2", "Label3"]
OPTIONAL_FEATURE_HEADERS = ["Feature1", "Feature2", "Feature3", "Feature4"]
OPTIONAL_FEATURE_TYPES = ["numerical", "numerical", "categorical", "text"]
OPTIONAL_CONTENT_TEMPLATE = '{"features":$features}'
OPTIONAL_FEATURE_ATTRIBUTION = "features"
OPTIONAL_ENABLE_EXPLAINABITIONS = "`true`"
OPTIONAL_MAX_RECORD_COUNT = 2
OPTIONAL_MAX_PAYLOAD_IN_MB = 5
DEFAULT_MAX_PAYLOAD_IN_MB = 6
OPTIONAL_PROBABILITY_INDEX = 0
OPTIONAL_LABEL_INDEX = 1
OPTIONAL_PROBABILITY_ATTRIBUTE = "probabilities"
OPTIONAL_LABEL_ATTRIBUTE = "labels"
OPTIONAL_NUM_OF_SAMPLES = 100
OPTIONAL_USE_LOGIT = True
DEFAULT_USE_LOGIT = False
OPTIONAL_SEED = 987
GRANULARITY = "token"
LANGUAGE = "en"

BASIC_CLARIFY_EXPLAINER_CONFIG_DICT = {
    "ShapConfig": {
        "ShapBaselineConfig": {
            "MimeType": DEFAULT_MIME_TYPE,
            "ShapBaseline": SHAP_BASELINE,
        },
        "UseLogit": DEFAULT_USE_LOGIT,
    }
}

CLARIFY_EXPLAINER_CONFIG_DICT_WITH_ALL_OPTIONAL = {
    "EnableExplanations": OPTIONAL_ENABLE_EXPLAINABITIONS,
    "InferenceConfig": {
        "FeaturesAttribute": OPTIONAL_FEATURE_ATTRIBUTION,
        "ContentTemplate": OPTIONAL_CONTENT_TEMPLATE,
        "MaxRecordCount": OPTIONAL_MAX_RECORD_COUNT,
        "MaxPayloadInMB": OPTIONAL_MAX_PAYLOAD_IN_MB,
        "ProbabilityIndex": OPTIONAL_PROBABILITY_INDEX,
        "LabelIndex": OPTIONAL_LABEL_INDEX,
        "ProbabilityAttribute": OPTIONAL_PROBABILITY_ATTRIBUTE,
        "LabelAttribute": OPTIONAL_LABEL_ATTRIBUTE,
        "LabelHeaders": OPTIONAL_LABEL_HEADERS,
        "FeatureHeaders": OPTIONAL_FEATURE_HEADERS,
        "FeatureTypes": OPTIONAL_FEATURE_TYPES,
    },
    "ShapConfig": {
        "ShapBaselineConfig": {
            "MimeType": OPTIONAL_MIME_TYPE,
            "ShapBaseline": SHAP_BASELINE,
            "ShapBaselineUri": SHAP_BASELINE_PATH,
        },
        "NumberOfSamples": OPTIONAL_NUM_OF_SAMPLES,
        "UseLogit": OPTIONAL_USE_LOGIT,
        "Seed": OPTIONAL_SEED,
        "TextConfig": {
            "Granularity": GRANULARITY,
            "Language": LANGUAGE,
        },
    },
}


def test_init_with_basic_input():
    shap_baseline_config = ClarifyShapBaselineConfig(shap_baseline=SHAP_BASELINE)
    shap_config = ClarifyShapConfig(shap_baseline_config=shap_baseline_config)
    clarify_explainer_config = ClarifyExplainerConfig(
        shap_config=shap_config,
    )
    explainer_config = ExplainerConfig(clarify_explainer_config=clarify_explainer_config)
    assert (
        explainer_config.clarify_explainer_config._to_request_dict()
        == BASIC_CLARIFY_EXPLAINER_CONFIG_DICT
    )


def test_init_with_all_optionals():
    shap_baseline_config = ClarifyShapBaselineConfig(
        mime_type=OPTIONAL_MIME_TYPE,
        # the config won't take shap_baseline and shap_baseline_uri both but we have both
        # here for testing purpose
        shap_baseline=SHAP_BASELINE,
        shap_baseline_uri=SHAP_BASELINE_PATH,
    )
    test_config = ClarifyTextConfig(granularity=GRANULARITY, language=LANGUAGE)
    shap_config = ClarifyShapConfig(
        shap_baseline_config=shap_baseline_config,
        number_of_samples=OPTIONAL_NUM_OF_SAMPLES,
        seed=OPTIONAL_SEED,
        use_logit=OPTIONAL_USE_LOGIT,
        text_config=test_config,
    )
    inference_config = ClarifyInferenceConfig(
        content_template=OPTIONAL_CONTENT_TEMPLATE,
        feature_headers=OPTIONAL_FEATURE_HEADERS,
        features_attribute=OPTIONAL_FEATURE_ATTRIBUTION,
        feature_types=OPTIONAL_FEATURE_TYPES,
        label_attribute=OPTIONAL_LABEL_ATTRIBUTE,
        label_headers=OPTIONAL_LABEL_HEADERS,
        label_index=OPTIONAL_LABEL_INDEX,
        max_payload_in_mb=OPTIONAL_MAX_PAYLOAD_IN_MB,
        max_record_count=OPTIONAL_MAX_RECORD_COUNT,
        probability_attribute=OPTIONAL_PROBABILITY_ATTRIBUTE,
        probability_index=OPTIONAL_PROBABILITY_INDEX,
    )
    clarify_explainer_config = ClarifyExplainerConfig(
        shap_config=shap_config,
        inference_config=inference_config,
        enable_explanations=OPTIONAL_ENABLE_EXPLAINABITIONS,
    )
    explainer_config = ExplainerConfig(clarify_explainer_config=clarify_explainer_config)
    assert (
        explainer_config.clarify_explainer_config._to_request_dict()
        == CLARIFY_EXPLAINER_CONFIG_DICT_WITH_ALL_OPTIONAL
    )
