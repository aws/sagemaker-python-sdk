"""Unit tests for the session ModelBuilder uses to reload the created Model (issue #6459)."""

import unittest
from unittest.mock import Mock, patch

from sagemaker.serve.model_builder import ModelBuilder

ROLE_ARN = "arn:aws:iam::123456789012:role/TestRole"
IMAGE_URI = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"


def _mock_session():
    session = Mock()
    session.boto_region_name = "us-west-2"
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = None
    session.config = {}
    session.sagemaker_config = {}
    return session


@patch("sagemaker.serve.model_builder.Model")
@patch("sagemaker.serve.model_builder.resolve_nested_dict_value_from_config")
@patch("sagemaker.serve.model_builder.resolve_value_from_config")
@patch.object(ModelBuilder, "_init_sagemaker_session_if_does_not_exist")
@patch.object(ModelBuilder, "_prepare_container_def")
class TestCreateSageMakerModelSession(unittest.TestCase):
    def test_model_get_uses_configured_boto_session(
        self, mock_prepare, mock_init, mock_resolve, mock_resolve_nested, mock_model
    ):
        mock_prepare.return_value = {"Image": IMAGE_URI, "Environment": {}}
        mock_resolve.side_effect = lambda value, *args, **kwargs: value
        mock_resolve_nested.side_effect = lambda value, *args, **kwargs: value

        session = _mock_session()
        builder = ModelBuilder(image_uri=IMAGE_URI, role_arn=ROLE_ARN, sagemaker_session=session)
        builder.model_name = "test-model"

        result = builder._create_sagemaker_model()

        session.create_model.assert_called_once()
        mock_model.get.assert_called_once_with(
            model_name="test-model",
            region=builder.region,
            session=session.boto_session,
        )
        self.assertIs(result, mock_model.get.return_value)
