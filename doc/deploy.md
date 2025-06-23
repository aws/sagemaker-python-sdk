# Deploy Models with SageMaker Python SDK

Amazon SageMaker provides flexible deployment options to serve your machine learning models for inference. This guide covers the key deployment interfaces and helps you choose the right approach for your workload requirements.

## Deployment Options Overview

| Deployment Option | Description | Best For |
|-------------------|-------------|----------|
| **Real-time Endpoints** | Synchronous, low-latency inference | Interactive applications, real-time predictions |
| **Serverless Inference** | Auto-scaling, pay-per-use endpoints | Intermittent traffic, cost optimization |
| **Asynchronous Inference** | Process requests in queue | Large payloads, long processing times |
| **Batch Transform** | Offline predictions for datasets | Bulk inference, ETL workflows |
| **Multi-model Endpoints** | Host multiple models on shared resources | Cost efficiency, similar model types |

## Deployment Workflow

Regardless of which deployment option you choose, the general workflow follows these steps:

1. **Prepare your model artifacts** - Ensure your trained model is in a compatible format
2. **Create a model in SageMaker** - Register your model artifacts with SageMaker
3. **Configure the deployment** - Choose deployment options and resources
4. **Deploy the model** - Create an endpoint or batch transform job
5. **Invoke for predictions** - Send requests to your deployed model

## Choosing the Right Deployment Option

### Real-time Endpoints

Real-time endpoints provide synchronous, low-latency inference via a REST API. They're ideal for applications that require immediate responses.

Key features:
- **Low-latency responses** for real-time applications
- **Auto-scaling** based on traffic patterns
- **High availability** with multi-AZ deployment
- **Monitoring** for performance and data quality

```python
import sagemaker
from sagemaker.model import Model

# Create a model from trained artifacts
model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39",
    model_data="s3://my-bucket/model.tar.gz",
    role=sagemaker.get_execution_role(),
    env={"SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600"}
)

# Deploy to a real-time endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="my-endpoint"
)

# Make a prediction
response = predictor.predict(data)
```

### Serverless Inference

Serverless Inference automatically provisions and scales compute capacity based on the volume of inference requests. You pay only for the compute time used to process requests, making it ideal for intermittent workloads.

Key features:
- **No infrastructure management** required
- **Pay-per-use** pricing model
- **Auto-scaling** from zero to handle traffic spikes
- **Configurable memory** sizes

```python
import sagemaker
from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

# Configure serverless settings
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=5
)

# Create a model
model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39",
    model_data="s3://my-bucket/model.tar.gz",
    role=sagemaker.get_execution_role()
)

# Deploy with serverless configuration
predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="my-serverless-endpoint"
)

# Make a prediction
response = predictor.predict(data)
```

### Asynchronous Inference

Asynchronous Inference queues incoming requests and processes them asynchronously. This is ideal for workloads with large payloads or long processing times.

Key features:
- **Queue-based processing** for handling traffic spikes
- **Support for large payloads** up to 1GB
- **Long processing times** without timeout concerns
- **Cost optimization** through efficient resource utilization

```python
import sagemaker
from sagemaker.model import Model
from sagemaker.async_inference import AsyncInferenceConfig

# Configure async inference
async_config = AsyncInferenceConfig(
    output_path="s3://my-bucket/async-results/",
    max_concurrent_invocations_per_instance=4,
    notification_config={
        "SuccessTopic": "arn:aws:sns:us-west-2:123456789012:success-topic",
        "ErrorTopic": "arn:aws:sns:us-west-2:123456789012:error-topic"
    }
)

# Create a model
model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39",
    model_data="s3://my-bucket/model.tar.gz",
    role=sagemaker.get_execution_role()
)

# Deploy with async configuration
async_predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    async_inference_config=async_config,
    endpoint_name="my-async-endpoint"
)

# Submit a request for async processing
response = async_predictor.predict_async(
    data="s3://my-bucket/input/data.csv",
    input_content_type="text/csv"
)
```

### Batch Transform

Batch Transform is designed for offline processing of large datasets. It's ideal for scenarios where you need to generate predictions for a complete dataset rather than individual requests.

Key features:
- **Efficient processing** of large datasets
- **Automatic scaling** to handle large workloads
- **Cost-effective** for bulk inference
- **Integration with data processing pipelines**

```python
import sagemaker
from sagemaker.model import Model

# Create a model
model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39",
    model_data="s3://my-bucket/model.tar.gz",
    role=sagemaker.get_execution_role()
)

# Create a transformer for batch processing
transformer = model.transformer(
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://my-bucket/batch-output/",
    strategy="MultiRecord",
    assemble_with="Line",
    accept="text/csv"
)

# Start a batch transform job
transformer.transform(
    data="s3://my-bucket/batch-input/",
    content_type="text/csv",
    split_type="Line"
)
```

### Multi-model Endpoints

Multi-model Endpoints allow you to host multiple models on a single endpoint, sharing compute resources among them. This is ideal for scenarios where you have many similar models that don't all need to be active simultaneously.

Key features:
- **Cost efficiency** through resource sharing
- **Dynamic loading** of models into memory
- **Simplified management** of many models
- **Reduced overhead** compared to individual endpoints

```python
import sagemaker
from sagemaker.multidatamodel import MultiDataModel

# Create a multi-model endpoint
mme = MultiDataModel(
    name="my-multi-model-endpoint",
    model_data_prefix="s3://my-bucket/models/",
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39",
    role=sagemaker.get_execution_role()
)

# Deploy the multi-model endpoint
predictor = mme.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)

# Make a prediction with a specific model
response = predictor.predict(
    target_model="model1.tar.gz",
    data=data
)
```

## Inference APIs

### ModelBuilder

The `ModelBuilder` class provides a simplified interface for creating and deploying models. It handles the complexity of model creation and deployment with a focus on ease of use.

```python
from sagemaker.model_builder import ModelBuilder
from sagemaker.schemas import SchemaBuilder

# Define input and output schemas
schema_builder = SchemaBuilder()
schema_builder.add_request_content_type("application/json")
schema_builder.add_response_content_type("application/json")

# Create a model with ModelBuilder
model_builder = ModelBuilder(
    model_name="my-model",
    role=sagemaker.get_execution_role(),
    schema=schema_builder.build(),
    container_config={
        "image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39",
        "model_data_url": "s3://my-bucket/model.tar.gz"
    }
)

# Deploy the model
predictor = model_builder.deploy(
    instance_type="ml.m5.xlarge",
    initial_instance_count=1
)
```

### Model

The `Model` class is the foundation for all deployment options in SageMaker. It encapsulates the model artifacts, inference code, and dependencies needed for deployment.

```python
from sagemaker.model import Model

model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-cpu-py39",
    model_data="s3://my-bucket/model.tar.gz",
    role=sagemaker.get_execution_role(),
    env={"MODEL_SERVER_WORKERS": "2"}
)
```

### Predictors

Predictors are client objects that provide a convenient interface for making inference requests to deployed endpoints.

```python
# After deploying a model
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)

# Make predictions with different content types
json_response = predictor.predict(data, content_type="application/json")
csv_response = predictor.predict(data, content_type="text/csv")

# Serialize and deserialize data automatically
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor.serializer = JSONSerializer()
predictor.deserializer = JSONDeserializer()

# Now predictor automatically handles serialization/deserialization
response = predictor.predict({"data": [1, 2, 3, 4]})
```

## Advanced Deployment Features

### Inference Optimization

SageMaker provides several options to optimize your model for inference:

- **SageMaker Neo** - Automatically optimize models for specific hardware
- **Elastic Inference** - Add GPU acceleration to CPU instances
- **Inference Recommender** - Get recommendations for optimal deployment configurations

### Inference Pipelines

Inference Pipelines allow you to chain multiple models and preprocessing/postprocessing steps into a single endpoint:

```python
from sagemaker.pipeline import PipelineModel

# Create individual models
preprocessor_model = Model(...)
inference_model = Model(...)
postprocessor_model = Model(...)

# Create a pipeline of models
pipeline_model = PipelineModel(
    models=[preprocessor_model, inference_model, postprocessor_model],
    role=sagemaker.get_execution_role()
)

# Deploy the pipeline
pipeline_predictor = pipeline_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge"
)
```

### A/B Testing

SageMaker supports A/B testing through production variants, allowing you to test different models or configurations:

```python
from sagemaker.model import Model

# Create two model variants
model_a = Model(...)
model_b = Model(...)

# Deploy both models to the same endpoint with traffic splitting
production_variants = [
    {
        "VariantName": "ModelA",
        "ModelName": model_a.name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.xlarge",
        "InitialVariantWeight": 0.7
    },
    {
        "VariantName": "ModelB",
        "ModelName": model_b.name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.xlarge",
        "InitialVariantWeight": 0.3
    }
]

# Create the endpoint with both variants
sagemaker_session = sagemaker.Session()
sagemaker_session.endpoint_from_production_variants(
    name="ab-test-endpoint",
    production_variants=production_variants
)
```

### Model Monitoring

SageMaker Model Monitor allows you to monitor the quality of your deployed models:

```python
from sagemaker.model_monitor import DataCaptureConfig, ModelMonitor

# Configure data capture for the endpoint
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri="s3://my-bucket/monitor-data/"
)

# Deploy model with monitoring enabled
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    data_capture_config=data_capture_config
)

# Create a model monitor
monitor = ModelMonitor(
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.m5.xlarge"
)

# Create a monitoring schedule
monitor.create_monitoring_schedule(
    monitor_schedule_name="my-monitoring-schedule",
    endpoint_input=predictor.endpoint_name,
    statistics="s3://my-bucket/baseline/statistics.json",
    constraints="s3://my-bucket/baseline/constraints.json",
    schedule_cron_expression="cron(0 * ? * * *)"  # Hourly
)
```
