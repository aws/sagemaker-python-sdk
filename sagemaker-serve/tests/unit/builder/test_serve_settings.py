import unittest
from unittest.mock import Mock, patch
from sagemaker.serve.builder.serve_settings import _ServeSettings


class TestServeSettings(unittest.TestCase):
    @patch("sagemaker.serve.builder.serve_settings.Session")
    @patch("sagemaker.serve.builder.serve_settings.resolve_value_from_config")
    def test_init_with_defaults(self, mock_resolve, mock_session):
        mock_resolve.return_value = None
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        settings = _ServeSettings()
        
        self.assertIsNotNone(settings.sagemaker_session)
        self.assertEqual(mock_resolve.call_count, 5)

    @patch("sagemaker.serve.builder.serve_settings.Session")
    @patch("sagemaker.serve.builder.serve_settings.resolve_value_from_config")
    def test_init_with_custom_values(self, mock_resolve, mock_session):
        mock_resolve.side_effect = lambda direct_input, **kwargs: direct_input
        mock_session_instance = Mock()
        
        settings = _ServeSettings(
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            s3_model_data_url="s3://bucket/model.tar.gz",
            instance_type="ml.m5.large",
            env_vars={"KEY": "VALUE"},
            sagemaker_session=mock_session_instance
        )
        
        self.assertEqual(settings.role_arn, "arn:aws:iam::123456789012:role/TestRole")
        self.assertEqual(settings.s3_model_data_url, "s3://bucket/model.tar.gz")
        self.assertEqual(settings.instance_type, "ml.m5.large")
        self.assertEqual(settings.env_vars, {"KEY": "VALUE"})

    @patch("sagemaker.serve.builder.serve_settings.Session")
    @patch("sagemaker.serve.builder.serve_settings.resolve_value_from_config")
    def test_telemetry_opt_out(self, mock_resolve, mock_session):
        mock_resolve.side_effect = lambda direct_input, default_value=None, **kwargs: default_value
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        settings = _ServeSettings()
        
        # Telemetry opt out should default to False
        self.assertFalse(settings.telemetry_opt_out)


if __name__ == "__main__":
    unittest.main()
