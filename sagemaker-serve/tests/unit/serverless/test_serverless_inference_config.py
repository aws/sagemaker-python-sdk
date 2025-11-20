import unittest
import warnings


class TestServerlessInferenceConfig(unittest.TestCase):
    def test_import_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Force reimport to trigger warning
            import sys
            if 'sagemaker.serve.serverless.serverless_inference_config' in sys.modules:
                del sys.modules['sagemaker.serve.serverless.serverless_inference_config']
            from sagemaker.serve.serverless.serverless_inference_config import ServerlessInferenceConfig
            self.assertGreaterEqual(len(w), 1)
            # Check if any warning is a DeprecationWarning
            has_deprecation = any(issubclass(warning.category, DeprecationWarning) for warning in w)
            self.assertTrue(has_deprecation)


if __name__ == "__main__":
    unittest.main()
