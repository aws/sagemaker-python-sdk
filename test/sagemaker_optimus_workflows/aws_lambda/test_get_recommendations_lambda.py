from unittest import TestCase
from unittest.mock import Mock

from sagemaker_titan_workflows.aws_lambda.get_recommendations_lambda import (
    get_recommendations_handler,
)


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
        self.assertEquals(
            recommendations[0]["env"]["TS_DEFAULT_WORKERS_PER_MODEL"], "5"
        )

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
        self.assertEquals(recommendations[0]["env"]["OMP_NUM_THREADS"], "2")
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[0]["env"])
        self.assertEquals(
            recommendations[0]["env"]["TS_DEFAULT_WORKERS_PER_MODEL"], "1"
        )

    def test_get_recommendation_handler_unknown_framework(self):
        recommendations = get_recommendations_handler(
            {
                "CustomerModelDetails": {
                    "NearestModelName": "unknown-model",
                    "Framework": "UNKNOWN_FRAMEWORK",
                },
                "Count": 2,
                "InstanceTypes": ["ml.c5.xlarge", "ml.g4dn.xlarge"],
            },
            Mock(),
        )

        # check for 2 recommendations
        self.assertEqual(2, len(recommendations))

        # one for each type of instance
        self.assertEquals("ml.c5.xlarge", recommendations[0]["instanceType"])
        self.assertEquals("ml.g4dn.xlarge", recommendations[1]["instanceType"])

        # cpu instance should have expected environnment variables
        self.assertTrue("OMP_NUM_THREADS" in recommendations[0]["env"])
        self.assertEquals(recommendations[0]["env"]["OMP_NUM_THREADS"], "2")
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[0]["env"])
        self.assertEquals(
            recommendations[0]["env"]["TS_DEFAULT_WORKERS_PER_MODEL"], "2"
        )

        # gpu instance should have expected environnment variables
        self.assertTrue("OMP_NUM_THREADS" not in recommendations[1]["env"])
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[1]["env"])
        self.assertEquals(
            recommendations[1]["env"]["TS_DEFAULT_WORKERS_PER_MODEL"], "3"
        )

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
        self.assertEquals(recommendations[0]["env"]["OMP_NUM_THREADS"], "1")
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[0]["env"])
        self.assertEquals(
            recommendations[0]["env"]["TS_DEFAULT_WORKERS_PER_MODEL"], "1"
        )

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
        self.assertEquals(recommendations[0]["env"]["OMP_NUM_THREADS"], "1")
        self.assertTrue("TS_DEFAULT_WORKERS_PER_MODEL" in recommendations[0]["env"])
        self.assertEquals(
            recommendations[0]["env"]["TS_DEFAULT_WORKERS_PER_MODEL"], "3"
        )
