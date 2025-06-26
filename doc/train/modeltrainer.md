# ModelTrainer

`ModelTrainer` is a high-level API in the SageMaker Python SDK that simplifies launching training jobs on Amazon SageMaker. It reduces boilerplate by abstracting away most of the low-level Estimator configuration and is ideal for users who prefer a configuration-driven approach.

---

## Quick Start

```python
from sagemaker.model_trainer import ModelTrainer
from sagemaker.inputs import TrainingInput

inputs = {
    "train": TrainingInput("s3://my-bucket/data/train"),
    "validation": TrainingInput("s3://my-bucket/data/validation")
}

trainer = ModelTrainer(
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework="pytorch",
    framework_version="2.0.1",
    py_version="py310",
    output_path="s3://my-bucket/output",
    hyperparameters={
        "epochs": 3,
        "learning_rate": 0.001
    }
)

job = trainer.train(inputs)
print(f"Started training job: {job.job_name}")
```

## Constructor Parameters

| Parameter           | Type | Description                                                      |
| ------------------- | ---- | ---------------------------------------------------------------- |
| `role`              | str  | IAM role ARN used by the training job                            |
| `instance_count`    | int  | Number of compute instances                                      |
| `instance_type`     | str  | Compute instance type (e.g., `ml.m5.xlarge`)                     |
| `framework`         | str  | Framework name: `pytorch`, `tensorflow`, `xgboost`, or `sklearn` |
| `framework_version` | str  | Framework version (e.g., `2.0.1`)                                |
| `py_version`        | str  | Python version (e.g., `py38`, `py310`)                           |
| `hyperparameters`   | dict | Key-value pairs passed to your training script                   |
| `output_path`       | str  | S3 URI for storing the model artifact                            |
| `entry_point`       | str  | Path to the training script (default: `train.py`)                |
| `source_dir`        | str  | Directory containing code dependencies                           |
| `environment`       | dict | Environment variables for the container                          |
| `base_job_name`     | str  | Optional prefix for the training job name                        |

## Training Method

```python
job = trainer.train(inputs)
```
This launches a SageMaker training job and returns a TrainingJob object containing metadata:
```python
print(job.job_name)
print(job.model_artifact_uri)

```

## Advanced Features
### Custom Script or Source Directory
```python
trainer = ModelTrainer(
    entry_point="train.py",
    source_dir="src/",
    ...
)

```
### Environment Variables
```python
trainer = ModelTrainer(
    environment={"MY_ENV_VAR": "value"},
    ...
)

```
### Experiment Tracking
```python
trainer = ModelTrainer(
    experiment_config={
        "ExperimentName": "experiment-001",
        "TrialName": "trial-abc"
    },
    ...
)

```

## Example Notebooks
Github Repo - https://github.com/aws/amazon-sagemaker-examples/tree/default/%20%20%20%20%20%20build_and_train_models/sm-model_trainer

## Related Pages
