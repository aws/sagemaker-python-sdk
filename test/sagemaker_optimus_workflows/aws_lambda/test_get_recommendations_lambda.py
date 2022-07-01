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
                "InstanceType": "ml.c5.large",
                "EnvironmentVariables": {
                    "OMP_NUM_THREADS": "1",
                    "TS_DEFAULT_WORKERS_PER_MODEL": "1",
                },
            },
            {
                "InstanceType": "ml.c5d.large",
                "EnvironmentVariables": {
                    "OMP_NUM_THREADS": "1",
                    "TS_DEFAULT_WORKERS_PER_MODEL": "1",
                },
            }
        ]

        self.assertTrue(all([x in recommendations for x in expected]))
