# Remote Decorator

The `@remote` decorator in the SageMaker Python SDK lets you run a Python function remotely as a SageMaker training job, without needing to define an Estimator or training script explicitly.

This interface is ideal for notebook-first workflows and quick experimentation.

## 🚀 Quick Start

```python
from sagemaker.remote_function import remote

@remote
def train_fn():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Dummy model
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(5):
        print(f"Epoch {epoch}")
    return "Training Complete"

future = train_fn.submit()
future.result()
```

## Configuration
You can pass configuration using .configure() before .submit():
```python
train_fn.configure(
    instance_type="ml.m5.large",
    instance_count=1,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310"
)

future = train_fn.submit()

```

## Parameters for .configure()

| Parameter        | Description                               |
| ---------------- | ----------------------------------------- |
| `role`           | IAM role ARN                              |
| `instance_type`  | EC2 instance (e.g., `ml.m5.large`)        |
| `instance_count` | Number of nodes                           |
| `image_uri`      | Docker image to run remotely              |
| `dependencies`   | List of local files or folders to include |
| `job_name`       | Optional job name prefix                  |
