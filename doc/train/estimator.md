# Estimator

The `Estimator` is the foundational interface in the SageMaker Python SDK for training models. It provides full control over the configuration of your training jobs, making it suitable for advanced use cases or when using custom training scripts and containers.

---

## 🚀 Quick Start

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="2.0.1",
    py_version="py310",
    hyperparameters={
        "epochs": 3,
        "learning_rate": 0.001
    },
    output_path="s3://my-bucket/output"
)

estimator.fit({"training": "s3://my-bucket/data/train"})
```

## Key Parameters

| Parameter           | Description                                         |
| ------------------- | --------------------------------------------------- |
| `entry_point`       | Path to the training script                         |
| `source_dir`        | Directory containing `entry_point` and dependencies |
| `role`              | IAM role ARN for SageMaker                          |
| `instance_type`     | EC2 instance type (e.g., `ml.g4dn.xlarge`)          |
| `instance_count`    | Number of instances                                 |
| `hyperparameters`   | Script arguments passed at runtime                  |
| `output_path`       | S3 path to store model artifacts                    |
| `py_version`        | Python version (`py38`, `py310`)                    |
| `framework_version` | Framework version (e.g., `2.0.1` for PyTorch)       |
