"""Unit tests for sagemaker.serve.utils.lineage_constants module."""
import unittest
import re
from sagemaker.serve.utils.lineage_constants import (
    LINEAGE_POLLER_INTERVAL_SECS,
    LINEAGE_POLLER_MAX_TIMEOUT_SECS,
    TRACKING_SERVER_ARN_REGEX,
    TRACKING_SERVER_CREATION_TIME_FORMAT,
    MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE,
    MLFLOW_S3_PATH,
    MLFLOW_MODEL_PACKAGE_PATH,
    MLFLOW_RUN_ID,
    MLFLOW_LOCAL_PATH,
    MLFLOW_REGISTRY_PATH,
    ERROR,
    CODE,
    CONTRIBUTED_TO,
    VALIDATION_EXCEPTION,
)


class TestLineageConstants(unittest.TestCase):
    """Test cases for lineage constants."""

    def test_lineage_poller_interval_secs(self):
        """Test LINEAGE_POLLER_INTERVAL_SECS constant."""
        self.assertEqual(LINEAGE_POLLER_INTERVAL_SECS, 15)
        self.assertIsInstance(LINEAGE_POLLER_INTERVAL_SECS, int)

    def test_lineage_poller_max_timeout_secs(self):
        """Test LINEAGE_POLLER_MAX_TIMEOUT_SECS constant."""
        self.assertEqual(LINEAGE_POLLER_MAX_TIMEOUT_SECS, 120)
        self.assertIsInstance(LINEAGE_POLLER_MAX_TIMEOUT_SECS, int)

    def test_tracking_server_arn_regex(self):
        """Test TRACKING_SERVER_ARN_REGEX constant."""
        self.assertEqual(
            TRACKING_SERVER_ARN_REGEX,
            r"arn:(.*?):sagemaker:(.*?):(.*?):mlflow-tracking-server/(.*?)$"
        )
        # Test that it's a valid regex
        pattern = re.compile(TRACKING_SERVER_ARN_REGEX)
        self.assertIsNotNone(pattern)
        
        # Test matching a valid ARN
        valid_arn = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
        match = pattern.match(valid_arn)
        self.assertIsNotNone(match)

    def test_tracking_server_creation_time_format(self):
        """Test TRACKING_SERVER_CREATION_TIME_FORMAT constant."""
        self.assertEqual(TRACKING_SERVER_CREATION_TIME_FORMAT, "%Y-%m-%dT%H:%M:%S.%fZ")

    def test_model_builder_mlflow_model_path_lineage_artifact_type(self):
        """Test MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE constant."""
        self.assertEqual(
            MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE,
            "ModelBuilderInputModelData"
        )

    def test_mlflow_path_constants(self):
        """Test MLflow path constants."""
        self.assertEqual(MLFLOW_S3_PATH, "S3")
        self.assertEqual(MLFLOW_MODEL_PACKAGE_PATH, "ModelPackage")
        self.assertEqual(MLFLOW_RUN_ID, "MLflowRunId")
        self.assertEqual(MLFLOW_LOCAL_PATH, "Local")
        self.assertEqual(MLFLOW_REGISTRY_PATH, "MLflowRegistry")

    def test_error_constants(self):
        """Test error-related constants."""
        self.assertEqual(ERROR, "Error")
        self.assertEqual(CODE, "Code")
        self.assertEqual(VALIDATION_EXCEPTION, "ValidationException")

    def test_contributed_to_constant(self):
        """Test CONTRIBUTED_TO constant."""
        self.assertEqual(CONTRIBUTED_TO, "ContributedTo")


if __name__ == "__main__":
    unittest.main()
