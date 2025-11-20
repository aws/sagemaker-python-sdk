"""Unit tests for sagemaker.serve.utils.lineage_utils module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.serve.utils.lineage_utils import _get_mlflow_model_path_type
from sagemaker.serve.utils.lineage_constants import (
    MLFLOW_RUN_ID,
    MLFLOW_REGISTRY_PATH,
    MLFLOW_MODEL_PACKAGE_PATH,
    MLFLOW_S3_PATH,
    MLFLOW_LOCAL_PATH,
)


class TestGetMlflowModelPathType(unittest.TestCase):
    """Test cases for _get_mlflow_model_path_type function."""

    def test_mlflow_run_id_pattern(self):
        """Test MLflow run ID pattern detection."""
        run_id_path = "runs:/abc123def456/model"
        result = _get_mlflow_model_path_type(run_id_path)
        self.assertEqual(result, MLFLOW_RUN_ID)

    def test_mlflow_registry_pattern(self):
        """Test MLflow registry pattern detection."""
        registry_path = "models:/my-model/1"
        result = _get_mlflow_model_path_type(registry_path)
        self.assertEqual(result, MLFLOW_REGISTRY_PATH)

    def test_model_package_arn_pattern(self):
        """Test SageMaker model package ARN pattern detection."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1"
        result = _get_mlflow_model_path_type(arn)
        self.assertEqual(result, MLFLOW_MODEL_PACKAGE_PATH)

    def test_s3_path_pattern(self):
        """Test S3 path pattern detection."""
        s3_path = "s3://my-bucket/path/to/model"
        result = _get_mlflow_model_path_type(s3_path)
        self.assertEqual(result, MLFLOW_S3_PATH)

    @patch('os.path.exists')
    def test_local_path_pattern(self, mock_exists):
        """Test local path pattern detection."""
        mock_exists.return_value = True
        local_path = "/local/path/to/model"
        result = _get_mlflow_model_path_type(local_path)
        self.assertEqual(result, MLFLOW_LOCAL_PATH)
        mock_exists.assert_called_once_with(local_path)

    @patch('os.path.exists')
    def test_invalid_path_raises_error(self, mock_exists):
        """Test that invalid path raises ValueError."""
        mock_exists.return_value = False
        invalid_path = "invalid://path"
        with self.assertRaises(ValueError) as context:
            _get_mlflow_model_path_type(invalid_path)
        self.assertIn("Invalid MLflow model path", str(context.exception))


# Note: Other functions in lineage_utils involve complex AWS API interactions
# with Artifact, Association, and Session objects. These are better tested
# through integration tests rather than unit tests to avoid brittle mocks.
#
# Functions not unit tested here (but covered by integration tests):
# - _load_artifact_by_source_uri
# - _poll_lineage_artifact
# - _create_mlflow_model_path_lineage_artifact
# - _retrieve_and_create_if_not_exist_mlflow_model_path_lineage_artifact
# - _add_association_between_artifacts
# - _maintain_lineage_tracking_for_mlflow_model


if __name__ == "__main__":
    unittest.main()
