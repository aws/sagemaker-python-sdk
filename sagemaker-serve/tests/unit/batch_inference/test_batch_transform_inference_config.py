import unittest
from sagemaker.serve.batch_inference.batch_transform_inference_config import BatchTransformInferenceConfig


class TestBatchTransformInferenceConfig(unittest.TestCase):
    def test_init(self):
        config = BatchTransformInferenceConfig(
            instance_count=2,
            instance_type="ml.m5.large",
            output_path="s3://bucket/output"
        )
        self.assertEqual(config.instance_count, 2)
        self.assertEqual(config.instance_type, "ml.m5.large")
        self.assertEqual(config.output_path, "s3://bucket/output")

    def test_validation(self):
        with self.assertRaises(Exception):
            BatchTransformInferenceConfig(instance_count="invalid", instance_type="ml.m5.large", output_path="s3://bucket")


if __name__ == "__main__":
    unittest.main()
