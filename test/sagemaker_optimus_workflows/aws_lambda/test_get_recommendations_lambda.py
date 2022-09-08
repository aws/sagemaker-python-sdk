from unittest import TestCase
from unittest.mock import Mock

from sagemaker_titan_workflows.aws_lambda.get_recommendations_lambda import \
    get_recommendations_handler


class TestGetRecommendationLambda(TestCase):
    def test_get_recommendation_handler(self):
        recommendations = get_recommendations_handler(
            {
                "CustomerModelDetails": {
                    "NearestModelName": "resnet152-torchvision",
                    "Framework": "pytorch",
                },
                "Count": 1,
                "InstanceTypes": ["ml.g4dn.2xlarge"],
            },
            Mock(),
        )

        self.assertEqual(1, len(recommendations))
        self.assertEquals("ml.g4dn.2xlarge", recommendations[0]["instanceType"])
        self.assertTrue("OMP_NUM_THREADS" not in recommendations[0]["env"])
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[0]["env"])

    def test_get_recommendation_handler_cpu(self):
        recommendations = get_recommendations_handler(
            {
                "CustomerModelDetails": {
                    "NearestModelName": "resnet152-torchvision",
                    "Framework": "pytorch",
                },
                "Count": 1,
                "InstanceTypes": ["ml.c5.xlarge"],
            },
            Mock(),
        )

        self.assertEqual(1, len(recommendations))
        self.assertEquals("ml.c5.xlarge", recommendations[0]["instanceType"])
        self.assertTrue("OMP_NUM_THREADS" in recommendations[0]["env"])
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[0]["env"])

    def test_get_recommendation_without_nearest_model_name(self):
        recommendations = get_recommendations_handler(
            {
                "CustomerModelDetails": {
                    "Framework": "pytorch",
                },
                "Count": 1,
                "InstanceTypes": ["ml.c5.xlarge"],
            },
            Mock(),
        )

        self.assertEqual(1, len(recommendations))
        self.assertEquals("ml.c5.xlarge", recommendations[0]["instanceType"])
        self.assertTrue("OMP_NUM_THREADS" in recommendations[0]["env"])
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[0]["env"])

    def test_get_recommendation_without_framework(self):
        recommendations = get_recommendations_handler(
            {
                "CustomerModelDetails": {
                    "NearestModelName": "resnet152-torchvision",
                },
                "Count": 1,
                "InstanceTypes": ["ml.c5.xlarge"],
            },
            Mock(),
        )

        self.assertEqual(1, len(recommendations))
        self.assertEquals("ml.c5.xlarge", recommendations[0]["instanceType"])
        self.assertTrue("OMP_NUM_THREADS" in recommendations[0]["env"])
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[0]["env"])
