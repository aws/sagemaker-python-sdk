# SageMaker ModelBuilder

## Overview

SageMaker ModelBuilder is a high-level interface for building and deploying machine learning models on Amazon SageMaker. It provides a streamlined workflow for model deployment with support for various deployment options including real-time endpoints, serverless, asynchronous inference, batch transforms, and multi-model endpoints.

ModelBuilder integrates seamlessly with other SageMaker components like ModelTrainer, making it easier to build and deploy models with minimal code. This unified deployment interface simplifies the process of moving from model training to production deployment.

## Key Features

- **Integration with ModelTrainer**: Direct handshake between ModelTrainer and ModelBuilder for seamless workflow
- **Latest Container Image Utility**: Enhanced `image_uris.retrieve()` method to fetch the latest version of an image automatically
- **Unified Deployment Interface**: Single interface for deploying models to different types of endpoints
- **Support for Multiple Deployment Types**:
  - Real-time endpoints for synchronous, low-latency inference
  - Serverless endpoints for auto-scaling with no infrastructure management
  - Asynchronous inference endpoints for processing large payloads
  - Batch transforms for offline inference on large datasets
  - Multi-model endpoints for hosting multiple models efficiently

## Basic Usage

### Integration with ModelTrainer

ModelBuilder can directly use a ModelTrainer object as its model source:

```python
from sagemaker import image_uris
from sagemaker.modules.train.model_trainer import ModelTrainer
from sagemaker_core.main.shapes import (
    Channel,
    DataSource,
    S3DataSource,
    OutputDataConfig,
    StoppingCondition,
)

# Get the latest XGBoost container image
xgboost_image = image_uris.retrieve(framework="xgboost", region="us-west-2", image_scope="training")

# Create a ModelTrainer instance
model_trainer = ModelTrainer(
    base_job_name="my-model-training-job",
    hyperparameters={
        "objective": "multi:softmax",
        "num_class": "3",
        "num_round": "10",
        "eval_metric": "merror",
    },
    training_image=xgboost_image,
    training_input_mode="File",
    role=role,
    output_data_config=OutputDataConfig(s3_output_path=s3_output_path),
    stopping_condition=StoppingCondition(max_runtime_in_seconds=600),
)

# Train the model
model_trainer.train(
    input_data_config=[
        Channel(
            channel_name="train",
            content_type="csv",
            compression_type="None",
            record_wrapper_type="None",
            data_source=DataSource(
                s3_data_source=S3DataSource(
                    s3_data_type="S3Prefix",
                    s3_uri=s3_input_path,
                    s3_data_distribution_type="FullyReplicated",
                )
            ),
        )
    ],
)
```

### Creating a ModelBuilder Instance

To create a ModelBuilder instance, you need to define an InferenceSpec class that specifies how to load and invoke your model:

```python
import numpy as np
from sagemaker.serve.builder.schema_builder import SchemaBuilder
import pandas as pd
from xgboost import XGBClassifier
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve import ModelBuilder

# Create a schema builder with sample input and output
data = {"Name": ["Alice", "Bob", "Charlie"]}
df = pd.DataFrame(data)
schema_builder = SchemaBuilder(sample_input=df, sample_output=df)

# Define an InferenceSpec for XGBoost
class XGBoostSpec(InferenceSpec):
    def load(self, model_dir: str):
        print(model_dir)
        model = XGBClassifier()
        model.load_model(model_dir + "/xgboost-model")
        return model

    def invoke(self, input_object: object, model: object):
        prediction_probabilities = model.predict_proba(input_object)
        predictions = np.argmax(prediction_probabilities, axis=1)
        return predictions

# Create a ModelBuilder instance
model_builder = ModelBuilder(
    model=model_trainer,  # ModelTrainer object passed directly
    role_arn=role,
    image_uri=xgboost_image,
    inference_spec=XGBoostSpec(),
    schema_builder=schema_builder,
    instance_type="ml.c6i.xlarge",
)

# Build the model
model = model_builder.build()
```

## Deployment Options

Once the model has been built, it can be deployed using the `model_builder.deploy()` method. This method supports various deployment configurations through the optional `inference_config` parameter.

### Real-Time Deployment

Deploy a model to a real-time endpoint:

```python
# Deploy to a real-time endpoint
predictor = model_builder.deploy(endpoint_name="my-xgboost-endpoint")

# Make a prediction
sklearn_input = np.array([1.0, 2.0, 3.0, 4.0])
result = predictor.predict(sklearn_input)
print(result)
```

Update an existing endpoint:

```python
# Update existing endpoint
predictor = model_builder.deploy(
    endpoint_name="my-xgboost-endpoint",
    initial_instance_count=3,
    update_endpoint=True,  # Updates existing endpoint
)
```

### Serverless Deployment

Deploy a model to a serverless endpoint:

```python
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig

# Deploy to a serverless endpoint
predictor = model_builder.deploy(
    endpoint_name="my-xgboost-serverless",
    inference_config=ServerlessInferenceConfig(memory_size_in_mb=2048),
)

# Make a prediction
sklearn_input = np.array([1.0, 2.0, 3.0, 4.0])
result = predictor.predict(sklearn_input)
print(result)
```

Update an existing serverless endpoint:

```python
# Update existing serverless endpoint
predictor = model_builder.deploy(
    endpoint_name="my-xgboost-serverless",
    inference_config=ServerlessInferenceConfig(memory_size_in_mb=1024),
    update_endpoint=True,
)
```

### Asynchronous Inference Deployment

Deploy a model for asynchronous inference:

```python
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.s3_utils import s3_path_join

# Deploy for asynchronous inference
predictor = model_builder.deploy(
    endpoint_name="my-xgboost-async",
    inference_config=AsyncInferenceConfig(
        output_path=s3_path_join(
            "s3://", bucket, "async_inference/output"
        )
    ),
)

# Make a prediction
sklearn_input = np.array([1.0, 2.0, 3.0, 4.0])
result = predictor.predict(sklearn_input)
print(result)
```

Update an existing asynchronous inference endpoint:

```python
# Update existing asynchronous inference endpoint
predictor = model_builder.deploy(
    endpoint_name="my-xgboost-async",
    inference_config=AsyncInferenceConfig(
        output_path=s3_path_join(
            "s3://", bucket, "async_inference/update_output_prefix"
        )
    ),
    update_endpoint=True,
)
```

### Batch Deployment

Deploy a model for batch inference:

```python
from sagemaker.batch_inference.batch_transform_inference_config import BatchTransformInferenceConfig
from sagemaker.s3_utils import s3_path_join

# Deploy for batch inference
transformer = model_builder.deploy(
    endpoint_name="my-xgboost-batch",
    inference_config=BatchTransformInferenceConfig(
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=s3_path_join(
            "s3://", bucket, "batch_inference/output"
        ),
        test_data_s3_path=s3_test_path,
    ),
)

print(transformer)
```

### Multi-Model Endpoint Deployment

Deploy a model to a multi-model endpoint:

```python
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements

# Deploy to a multi-model endpoint
predictor = model_builder.deploy(
    endpoint_name="my-xgboost-multi-model",
    inference_config=ResourceRequirements(
        requests={
            "num_cpus": 0.5,
            "memory": 512,
            "copies": 2,
        },
        limits={},
    ),
)

# Make a prediction
sklearn_input = np.array([1.0, 2.0, 3.0, 4.0])
result = predictor.predict(sklearn_input)
print(result)
```

## Key Components

### InferenceSpec

The `InferenceSpec` class defines how to load and invoke your model. It requires implementing two methods:

- `load(model_dir: str)`: Loads the model from the specified directory
- `invoke(input_object: object, model: object)`: Processes input data using the loaded model

### SchemaBuilder

The `SchemaBuilder` class helps define the input and output schemas for your model:

```python
schema_builder = SchemaBuilder(sample_input=df, sample_output=df)
```

### Deployment Configurations

Different deployment types require different configuration objects:

- **Real-time**: No special configuration needed
- **Serverless**: `ServerlessInferenceConfig`
- **Asynchronous**: `AsyncInferenceConfig`
- **Batch**: `BatchTransformInferenceConfig`
- **Multi-model**: `ResourceRequirements`

## Best Practices

1. **Use the latest container images**: Let the `image_uris.retrieve()` method fetch the latest version automatically.
2. **Choose the right deployment type**: Select the deployment type based on your use case requirements.
3. **Define proper InferenceSpec**: Ensure your `load()` and `invoke()` methods correctly handle your model and data.
4. **Update endpoints when needed**: Use the `update_endpoint=True` parameter to update existing endpoints.
5. **Provide sample inputs and outputs**: Use `SchemaBuilder` with representative samples to define your model's interface.

## Conclusion

SageMaker ModelBuilder provides a unified interface for building and deploying machine learning models on Amazon SageMaker. It simplifies the deployment process by abstracting away the complexities of different deployment types and providing a consistent API for all deployment options.

By integrating directly with ModelTrainer and supporting various deployment configurations, ModelBuilder helps streamline the ML workflow from training to production, reducing the time and effort required to deploy models at scale.
