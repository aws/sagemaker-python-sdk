import unittest
from sagemaker.serve.compute_resource_requirements.resource_requirements import ResourceRequirements


class TestResourceRequirements(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(ResourceRequirements)


if __name__ == "__main__":
    unittest.main()
