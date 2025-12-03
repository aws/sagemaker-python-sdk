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
"""Unit tests for sagemaker.core.inference_recommender.inference_recommender_mixin module."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.core.inference_recommender.inference_recommender_mixin import (
    Phase,
    ModelLatencyThreshold,
    InferenceRecommenderMixin,
)
from sagemaker.core.parameter import CategoricalParameter


class TestPhase:
    """Test Phase class."""

    def test_init(self):
        """Test Phase initialization."""
        phase = Phase(duration_in_seconds=300, initial_number_of_users=10, spawn_rate=2)
        assert phase.to_json["DurationInSeconds"] == 300
        assert phase.to_json["InitialNumberOfUsers"] == 10
        assert phase.to_json["SpawnRate"] == 2

    def test_to_json_structure(self):
        """Test Phase to_json structure."""
        phase = Phase(100, 5, 1)
        json_data = phase.to_json
        assert "DurationInSeconds" in json_data
        assert "InitialNumberOfUsers" in json_data
        assert "SpawnRate" in json_data


class TestModelLatencyThreshold:
    """Test ModelLatencyThreshold class."""

    def test_init(self):
        """Test ModelLatencyThreshold initialization."""
        threshold = ModelLatencyThreshold(percentile="P95", value_in_milliseconds=500)
        assert threshold.to_json["Percentile"] == "P95"
        assert threshold.to_json["ValueInMilliseconds"] == 500

    def test_to_json_structure(self):
        """Test ModelLatencyThreshold to_json structure."""
        threshold = ModelLatencyThreshold("P99", 1000)
        json_data = threshold.to_json
        assert "Percentile" in json_data
        assert "ValueInMilliseconds" in json_data


class TestInferenceRecommenderMixin:
    """Test InferenceRecommenderMixin class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mixin = InferenceRecommenderMixin()
        self.mixin.role = "arn:aws:iam::123456789:role/SageMakerRole"
        self.mixin.name = "test-model"
        self.mixin.sagemaker_session = Mock()

    def test_convert_to_endpoint_configurations_json_valid(self):
        """Test _convert_to_endpoint_configurations_json with valid input."""
        hyperparameter_ranges = [
            {
                "instance_types": CategoricalParameter(["ml.c5.xlarge", "ml.c5.2xlarge"]),
                "OMP_NUM_THREADS": CategoricalParameter(["1", "2"]),
            }
        ]

        result = self.mixin._convert_to_endpoint_configurations_json(hyperparameter_ranges)

        assert len(result) == 2
        assert result[0]["InstanceType"] == "ml.c5.xlarge"
        assert result[1]["InstanceType"] == "ml.c5.2xlarge"
        assert "EnvironmentParameterRanges" in result[0]

    def test_convert_to_endpoint_configurations_json_missing_instance_types(self):
        """Test _convert_to_endpoint_configurations_json without instance_types."""
        hyperparameter_ranges = [{"OMP_NUM_THREADS": CategoricalParameter(["1", "2"])}]

        with pytest.raises(ValueError, match="instance_type must be defined"):
            self.mixin._convert_to_endpoint_configurations_json(hyperparameter_ranges)

    def test_convert_to_endpoint_configurations_json_none(self):
        """Test _convert_to_endpoint_configurations_json with None."""
        result = self.mixin._convert_to_endpoint_configurations_json(None)
        assert result is None

    def test_convert_to_traffic_pattern_json_valid(self):
        """Test _convert_to_traffic_pattern_json with valid input."""
        phases = [Phase(300, 10, 2), Phase(600, 20, 5)]

        result = self.mixin._convert_to_traffic_pattern_json("PHASES", phases)

        assert result["TrafficType"] == "PHASES"
        assert len(result["Phases"]) == 2
        assert result["Phases"][0]["DurationInSeconds"] == 300

    def test_convert_to_traffic_pattern_json_default_traffic_type(self):
        """Test _convert_to_traffic_pattern_json with default traffic type."""
        phases = [Phase(300, 10, 2)]

        result = self.mixin._convert_to_traffic_pattern_json(None, phases)

        assert result["TrafficType"] == "PHASES"

    def test_convert_to_traffic_pattern_json_none(self):
        """Test _convert_to_traffic_pattern_json with None."""
        result = self.mixin._convert_to_traffic_pattern_json(None, None)
        assert result is None

    def test_convert_to_resource_limit_json_valid(self):
        """Test _convert_to_resource_limit_json with valid input."""
        result = self.mixin._convert_to_resource_limit_json(10, 5)

        assert result["MaxNumberOfTests"] == 10
        assert result["MaxParallelOfTests"] == 5

    def test_convert_to_resource_limit_json_partial(self):
        """Test _convert_to_resource_limit_json with partial input."""
        result = self.mixin._convert_to_resource_limit_json(10, None)

        assert result["MaxNumberOfTests"] == 10
        assert "MaxParallelOfTests" not in result

    def test_convert_to_resource_limit_json_none(self):
        """Test _convert_to_resource_limit_json with None."""
        result = self.mixin._convert_to_resource_limit_json(None, None)
        assert result is None

    def test_convert_to_stopping_conditions_json_valid(self):
        """Test _convert_to_stopping_conditions_json with valid input."""
        thresholds = [ModelLatencyThreshold("P95", 500), ModelLatencyThreshold("P99", 1000)]

        result = self.mixin._convert_to_stopping_conditions_json(1000, thresholds)

        assert result["MaxInvocations"] == 1000
        assert len(result["ModelLatencyThresholds"]) == 2

    def test_convert_to_stopping_conditions_json_partial(self):
        """Test _convert_to_stopping_conditions_json with partial input."""
        result = self.mixin._convert_to_stopping_conditions_json(1000, None)

        assert result["MaxInvocations"] == 1000
        assert "ModelLatencyThresholds" not in result

    def test_convert_to_stopping_conditions_json_none(self):
        """Test _convert_to_stopping_conditions_json with None."""
        result = self.mixin._convert_to_stopping_conditions_json(None, None)
        assert result is None

    def test_search_recommendation_found(self):
        """Test _search_recommendation when recommendation is found."""
        recommendations = [
            {"RecommendationId": "rec-1", "InstanceType": "ml.c5.xlarge"},
            {"RecommendationId": "rec-2", "InstanceType": "ml.c5.2xlarge"},
        ]

        result = self.mixin._search_recommendation(recommendations, "rec-2")

        assert result is not None
        assert result["InstanceType"] == "ml.c5.2xlarge"

    def test_search_recommendation_not_found(self):
        """Test _search_recommendation when recommendation is not found."""
        recommendations = [{"RecommendationId": "rec-1", "InstanceType": "ml.c5.xlarge"}]

        result = self.mixin._search_recommendation(recommendations, "rec-999")

        assert result is None

    def test_filter_recommendations_for_realtime(self):
        """Test _filter_recommendations_for_realtime."""
        self.mixin.inference_recommendations = [
            {
                "EndpointConfiguration": {
                    "ServerlessConfig": {},
                    "InstanceType": "ml.c5.xlarge",
                    "InitialInstanceCount": 1,
                }
            },
            {"EndpointConfiguration": {"InstanceType": "ml.c5.2xlarge", "InitialInstanceCount": 2}},
        ]

        instance_type, count = self.mixin._filter_recommendations_for_realtime()

        assert instance_type == "ml.c5.2xlarge"
        assert count == 2

    def test_update_params_for_right_size_with_accelerator_raises_error(self):
        """Test _update_params_for_right_size with accelerator_type raises error."""
        with pytest.raises(ValueError, match="accelerator_type is not compatible"):
            self.mixin._update_params_for_right_size(accelerator_type="ml.eia1.medium")

    def test_update_params_for_right_size_with_instance_type_override(self):
        """Test _update_params_for_right_size with instance_type override."""
        result = self.mixin._update_params_for_right_size(
            instance_type="ml.m5.xlarge", initial_instance_count=1
        )

        assert result is None

    def test_update_params_for_right_size_with_serverless_override(self):
        """Test _update_params_for_right_size with serverless config override."""
        from sagemaker.core.serverless_inference_config import ServerlessInferenceConfig

        config = ServerlessInferenceConfig()

        result = self.mixin._update_params_for_right_size(serverless_inference_config=config)

        assert result is None

    def test_update_params_for_recommendation_id_invalid_format(self):
        """Test _update_params_for_recommendation_id with invalid format."""
        with pytest.raises(ValueError, match="inference_recommendation_id is not valid"):
            self.mixin._update_params_for_recommendation_id(
                instance_type=None,
                initial_instance_count=None,
                accelerator_type=None,
                async_inference_config=None,
                serverless_inference_config=None,
                inference_recommendation_id="invalid-format",
                explainer_config=None,
            )

    def test_update_params_for_recommendation_id_with_accelerator_raises_error(self):
        """Test _update_params_for_recommendation_id with accelerator raises error."""
        with pytest.raises(ValueError, match="accelerator_type is not compatible"):
            self.mixin._update_params_for_recommendation_id(
                instance_type=None,
                initial_instance_count=None,
                accelerator_type="ml.eia1.medium",
                async_inference_config=None,
                serverless_inference_config=None,
                inference_recommendation_id="test-job/12345678",
                explainer_config=None,
            )

    def test_update_params_with_both_instance_params_and_job_results(self):
        """Test _update_params with both instance_type and initial_instance_count and job results."""
        # Set up job results
        self.mixin.inference_recommender_job_results = {"status": "completed"}
        self.mixin.inference_recommendations = [
            {"EndpointConfiguration": {"InstanceType": "ml.c5.2xlarge", "InitialInstanceCount": 2}}
        ]

        # When both params are provided, it should override recommendations
        result = self.mixin._update_params(
            instance_type="ml.m5.xlarge",
            initial_instance_count=1,
            accelerator_type=None,
            async_inference_config=None,
            serverless_inference_config=None,
            explainer_config=None,
            inference_recommendation_id=None,
            inference_recommender_job_results=self.mixin.inference_recommender_job_results,
        )

        # When instance params are provided, they override recommendations
        # The function returns the provided params
        assert result == ("ml.m5.xlarge", 1)
