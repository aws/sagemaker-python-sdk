from unittest import TestCase
from unittest.mock import Mock

from sagemaker_titan_workflows.aws_lambda.get_recommendations_lambda import \
    get_recommendations_handler


class TestGetRecommendationLambda(TestCase):
    def test_get_recommendation_handler(self):
        recommendations = get_recommendations_handler(
            {
                "NearestModelName": "densenet201-keras",
                "Framework": "tensorflow:1.15.5:py36",
            },
            Mock(),
        )

        expected = ["ml.c5.large", "ml.c5d.large"]

        self.assertTrue(set(expected) <= set(recommendations))
