"""
Unit tests for inference_recommendation_mixin.py module.
Tests Phase, ModelLatencyThreshold, and _InferenceRecommenderMixin classes.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.serve.inference_recommendation_mixin import (
    Phase,
    ModelLatencyThreshold,
    _InferenceRecommenderMixin,
    INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING,
)
from sagemaker.core.parameter import CategoricalParameter


class TestPhase(unittest.TestCase):
    """Test Phase class for traffic pattern configuration."""

    def test_phase_initialization(self):
        """Test Phase initialization with valid parameters."""
        phase = Phase(
            duration_in_seconds=300,
            initial_number_of_users=1,
            spawn_rate=2
        )
        
        self.assertEqual(phase.to_json["DurationInSeconds"], 300)
        self.assertEqual(phase.to_json["InitialNumberOfUsers"], 1)
        self.assertEqual(phase.to_json["SpawnRate"], 2)

    def test_phase_to_json_structure(self):
        """Test Phase to_json structure."""
        phase = Phase(duration_in_seconds=600, initial_number_of_users=5, spawn_rate=10)
        
        expected_keys = {"DurationInSeconds", "InitialNumberOfUsers", "SpawnRate"}
        self.assertEqual(set(phase.to_json.keys()), expected_keys)


class TestModelLatencyThreshold(unittest.TestCase):
    """Test ModelLatencyThreshold class."""

    def test_latency_threshold_initialization(self):
        """Test ModelLatencyThreshold initialization."""
        threshold = ModelLatencyThreshold(percentile="P95", value_in_milliseconds=100)
        
        self.assertEqual(threshold.to_json["Percentile"], "P95")
        self.assertEqual(threshold.to_json["ValueInMilliseconds"], 100)

    def test_latency_threshold_p99(self):
        """Test ModelLatencyThreshold with P99 percentile."""
        threshold = ModelLatencyThreshold(percentile="P99", value_in_milliseconds=200)
        
        self.assertEqual(threshold.to_json["Percentile"], "P99")
        self.assertEqual(threshold.to_json["ValueInMilliseconds"], 200)


class TestInferenceRecommenderMixin(unittest.TestCase):
    """Test _InferenceRecommenderMixin class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mixin = _InferenceRecommenderMixin()
        self.mixin.sagemaker_session = Mock()
        self.mixin.role_arn = "arn:aws:iam::123456789012:role/TestRole"
        self.mixin.model_name = "test-model"
        self.mixin.image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.9.0-cpu-py38"

    def test_convert_to_endpoint_configurations_json_none(self):
        """Test _convert_to_endpoint_configurations_json with None input."""
        result = self.mixin._convert_to_endpoint_configurations_json(None)
        
        self.assertIsNone(result)

    def test_convert_to_endpoint_configurations_json_valid(self):
        """Test _convert_to_endpoint_configurations_json with valid input."""
        hyperparameter_ranges = [{
            'instance_types': CategoricalParameter(['ml.c5.xlarge', 'ml.c5.2xlarge']),
            'OMP_NUM_THREADS': CategoricalParameter(['1', '2', '4'])
        }]
        
        result = self.mixin._convert_to_endpoint_configurations_json(hyperparameter_ranges)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)  # Two instance types
        self.assertEqual(result[0]['InstanceType'], 'ml.c5.xlarge')
        self.assertEqual(result[1]['InstanceType'], 'ml.c5.2xlarge')

    def test_convert_to_endpoint_configurations_json_missing_instance_types(self):
        """Test _convert_to_endpoint_configurations_json without instance_types."""
        hyperparameter_ranges = [{
            'OMP_NUM_THREADS': CategoricalParameter(['1', '2'])
        }]
        
        with self.assertRaises(ValueError) as context:
            self.mixin._convert_to_endpoint_configurations_json(hyperparameter_ranges)
        
        self.assertIn("instance_types must be defined", str(context.exception))

    def test_convert_to_traffic_pattern_json_none(self):
        """Test _convert_to_traffic_pattern_json with None input."""
        result = self.mixin._convert_to_traffic_pattern_json(None, None)
        
        self.assertIsNone(result)

    def test_convert_to_traffic_pattern_json_valid(self):
        """Test _convert_to_traffic_pattern_json with valid phases."""
        phases = [
            Phase(duration_in_seconds=300, initial_number_of_users=1, spawn_rate=2),
            Phase(duration_in_seconds=600, initial_number_of_users=10, spawn_rate=5)
        ]
        
        result = self.mixin._convert_to_traffic_pattern_json("PHASES", phases)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['TrafficType'], 'PHASES')
        self.assertEqual(len(result['Phases']), 2)
        self.assertEqual(result['Phases'][0]['DurationInSeconds'], 300)

    def test_convert_to_traffic_pattern_json_default_traffic_type(self):
        """Test _convert_to_traffic_pattern_json with default traffic type."""
        phases = [Phase(duration_in_seconds=300, initial_number_of_users=1, spawn_rate=2)]
        
        result = self.mixin._convert_to_traffic_pattern_json(None, phases)
        
        self.assertEqual(result['TrafficType'], 'PHASES')

    def test_convert_to_resource_limit_json_none(self):
        """Test _convert_to_resource_limit_json with None inputs."""
        result = self.mixin._convert_to_resource_limit_json(None, None)
        
        self.assertIsNone(result)

    def test_convert_to_resource_limit_json_max_tests_only(self):
        """Test _convert_to_resource_limit_json with max_tests only."""
        result = self.mixin._convert_to_resource_limit_json(max_tests=10, max_parallel_tests=None)
        
        self.assertEqual(result['MaxNumberOfTests'], 10)
        self.assertNotIn('MaxParallelOfTests', result)

    def test_convert_to_resource_limit_json_both_limits(self):
        """Test _convert_to_resource_limit_json with both limits."""
        result = self.mixin._convert_to_resource_limit_json(max_tests=10, max_parallel_tests=3)
        
        self.assertEqual(result['MaxNumberOfTests'], 10)
        self.assertEqual(result['MaxParallelOfTests'], 3)

    def test_convert_to_stopping_conditions_json_none(self):
        """Test _convert_to_stopping_conditions_json with None inputs."""
        result = self.mixin._convert_to_stopping_conditions_json(None, None)
        
        self.assertIsNone(result)

    def test_convert_to_stopping_conditions_json_max_invocations(self):
        """Test _convert_to_stopping_conditions_json with max_invocations."""
        result = self.mixin._convert_to_stopping_conditions_json(max_invocations=1000, model_latency_thresholds=None)
        
        self.assertEqual(result['MaxInvocations'], 1000)
        self.assertNotIn('ModelLatencyThresholds', result)

    def test_convert_to_stopping_conditions_json_with_thresholds(self):
        """Test _convert_to_stopping_conditions_json with latency thresholds."""
        thresholds = [
            ModelLatencyThreshold(percentile="P95", value_in_milliseconds=100),
            ModelLatencyThreshold(percentile="P99", value_in_milliseconds=200)
        ]
        
        result = self.mixin._convert_to_stopping_conditions_json(None, thresholds)
        
        self.assertEqual(len(result['ModelLatencyThresholds']), 2)
        self.assertEqual(result['ModelLatencyThresholds'][0]['Percentile'], 'P95')

    def test_search_recommendation_found(self):
        """Test _search_recommendation when recommendation is found."""
        recommendations = [
            {'RecommendationId': 'rec-1', 'InstanceType': 'ml.m5.large'},
            {'RecommendationId': 'rec-2', 'InstanceType': 'ml.m5.xlarge'}
        ]
        
        result = self.mixin._search_recommendation(recommendations, 'rec-2')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['InstanceType'], 'ml.m5.xlarge')

    def test_search_recommendation_not_found(self):
        """Test _search_recommendation when recommendation is not found."""
        recommendations = [
            {'RecommendationId': 'rec-1', 'InstanceType': 'ml.m5.large'}
        ]
        
        result = self.mixin._search_recommendation(recommendations, 'rec-999')
        
        self.assertIsNone(result)

    def test_filter_recommendations_for_realtime(self):
        """Test _filter_recommendations_for_realtime."""
        self.mixin.inference_recommendations = [
            {
                'EndpointConfiguration': {
                    'ServerlessConfig': {'MemorySizeInMB': 2048}
                }
            },
            {
                'EndpointConfiguration': {
                    'InstanceType': 'ml.m5.large',
                    'InitialInstanceCount': 2
                }
            }
        ]
        
        instance_type, instance_count = self.mixin._filter_recommendations_for_realtime()
        
        self.assertEqual(instance_type, 'ml.m5.large')
        self.assertEqual(instance_count, 2)

    def test_filter_recommendations_for_realtime_no_realtime(self):
        """Test _filter_recommendations_for_realtime with only serverless."""
        self.mixin.inference_recommendations = [
            {
                'EndpointConfiguration': {
                    'ServerlessConfig': {'MemorySizeInMB': 2048}
                }
            }
        ]
        
        instance_type, instance_count = self.mixin._filter_recommendations_for_realtime()
        
        self.assertIsNone(instance_type)
        self.assertIsNone(instance_count)

    def test_update_params_for_right_size_with_accelerator(self):
        """Test _update_params_for_right_size rejects accelerator_type."""
        with self.assertRaises(ValueError) as context:
            self.mixin._update_params_for_right_size(accelerator_type="ml.eia1.medium")
        
        self.assertIn("accelerator_type is not compatible", str(context.exception))

    def test_update_params_for_right_size_with_instance_type_override(self):
        """Test _update_params_for_right_size with instance_type override."""
        result = self.mixin._update_params_for_right_size(
            instance_type="ml.m5.large",
            initial_instance_count=1
        )
        
        self.assertIsNone(result)

    def test_update_params_for_right_size_with_async_config(self):
        """Test _update_params_for_right_size with async_inference_config."""
        result = self.mixin._update_params_for_right_size(
            async_inference_config=Mock()
        )
        
        self.assertIsNone(result)

    def test_update_params_returns_provided_params(self):
        """Test _update_params returns provided parameters when no recommendations."""
        result = self.mixin._update_params(
            instance_type="ml.m5.large",
            initial_instance_count=2
        )
        
        self.assertEqual(result, ("ml.m5.large", 2))

    def test_update_params_for_recommendation_id_invalid_format(self):
        """Test _update_params_for_recommendation_id with invalid ID format."""
        with self.assertRaises(ValueError) as context:
            self.mixin._update_params_for_recommendation_id(
                instance_type=None,
                initial_instance_count=None,
                accelerator_type=None,
                async_inference_config=None,
                serverless_inference_config=None,
                inference_recommendation_id="invalid-format",
                explainer_config=None
            )
        
        self.assertIn("Invalid inference_recommendation_id format", str(context.exception))

    def test_update_params_for_recommendation_id_with_accelerator(self):
        """Test _update_params_for_recommendation_id rejects accelerator_type."""
        with self.assertRaises(ValueError) as context:
            self.mixin._update_params_for_recommendation_id(
                instance_type=None,
                initial_instance_count=None,
                accelerator_type="ml.eia1.medium",
                async_inference_config=None,
                serverless_inference_config=None,
                inference_recommendation_id="job-name/12345678",
                explainer_config=None
            )
        
        self.assertIn("accelerator_type is not compatible", str(context.exception))

    def test_framework_mapping_constants(self):
        """Test INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING constants."""
        self.assertEqual(INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING['xgboost'], 'XGBOOST')
        self.assertEqual(INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING['sklearn'], 'SAGEMAKER-SCIKIT-LEARN')
        self.assertEqual(INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING['pytorch'], 'PYTORCH')
        self.assertEqual(INFERENCE_RECOMMENDER_FRAMEWORK_MAPPING['tensorflow'], 'TENSORFLOW')


if __name__ == "__main__":
    unittest.main()
