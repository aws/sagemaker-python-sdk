from unittest import TestCase
from unittest.mock import Mock

from sagemaker_titan_workflows.aws_lambda.get_recommendations_lambda import \
    get_recommendations_handler


class TestGetRecommendationLambda(TestCase):
    def test_get_recommendation_handler(self):
        recommendations = get_recommendations_handler(
            {
                "NearestModelName": "resnet152-torchvision",
                "Framework": "pytorch",
            },
            Mock(),
        )

        expected = [
            {
                "instanceType": "ml.c5.large",
                "env": {
                    "OMP_NUM_THREADS": "1",
                    "TS_DEFAULT_WORKERS_PER_MODEL": "1",
                },
            },
            {
                "instanceType": "ml.c5d.large",
                "env": {
                    "OMP_NUM_THREADS": "1",
                    "TS_DEFAULT_WORKERS_PER_MODEL": "1",
                },
            }
        ]

        self.assertTrue(all([x in recommendations for x in expected]))
