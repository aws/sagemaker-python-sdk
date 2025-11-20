import unittest
from sagemaker.serve.model_format.mlflow.constants import (
    DEFAULT_FW_USED_FOR_DEFAULT_IMAGE,
    DEFAULT_PYTORCH_VERSION,
    MLFLOW_FLAVOR_TO_PYTHON_PACKAGE_MAP
)


class TestMLflowConstants(unittest.TestCase):
    def test_default_framework(self):
        self.assertEqual(DEFAULT_FW_USED_FOR_DEFAULT_IMAGE, "pytorch")

    def test_default_pytorch_versions(self):
        self.assertIn("py38", DEFAULT_PYTORCH_VERSION)
        self.assertIn("py39", DEFAULT_PYTORCH_VERSION)
        self.assertIn("py310", DEFAULT_PYTORCH_VERSION)

    def test_flavor_mapping(self):
        self.assertEqual(MLFLOW_FLAVOR_TO_PYTHON_PACKAGE_MAP["sklearn"], "scikit-learn")
        self.assertEqual(MLFLOW_FLAVOR_TO_PYTHON_PACKAGE_MAP["pytorch"], "torch")


if __name__ == "__main__":
    unittest.main()
