import unittest
from unittest.mock import patch

from sagemaker.image_utils import get_latest_container_image


class TestImageUtils(unittest.TestCase):
    @patch('sagemaker.image_utils.config_for_framework')
    @patch('sagemaker.image_utils.retrieve')
    def test_get_latest_container_image(self,
                                        mock_image_retrieve,
                                        mock_config_for_framework):
        mock_config_for_framework.return_value = {
            "inference": {
                "version_aliases": {
                    "latest": "1"
                }
            }
        }
        mock_image_retrieve.return_value = "latest-image"

        image, version = get_latest_container_image("xgboost", "inference")
        assert image == "latest-image"
        assert version == "1"

    @patch('sagemaker.image_utils.config_for_framework')
    @patch('sagemaker.image_utils.retrieve')
    def test_get_latest_container_image_invalid_framework(self,
                                                          mock_image_retrieve,
                                                          mock_config_for_framework):
        mock_config_for_framework.side_effect = FileNotFoundError

        with self.assertRaises(ValueError) as e:
            get_latest_container_image("xgboost", "inference")
            assert "No framework config for framework" in str(e.exception)

    @patch('sagemaker.image_utils.config_for_framework')
    @patch('sagemaker.image_utils.retrieve')
    def test_get_latest_container_image_no_framework(self,
                                                          mock_image_retrieve,
                                                          mock_config_for_framework):
        mock_config_for_framework.return_value = {}

        with self.assertRaises(ValueError) as e:
            get_latest_container_image("xgboost", "inference")
            assert "No framework config for framework" in str(e.exception)