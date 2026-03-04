# SageMaker Python SDK V2 to V3 Migration Guide

## Benefits of Migration

### 1. Improved Developer Experience

* **Type Safety**: Full type hints and IDE support
* **Auto-completion**: Better code completion in IDEs
* **Validation**: Early validation of configurations
* **Documentation**: Inline documentation and examples

### 2. Better Architecture

* **Separation of Concerns**: Clear boundaries between training, serving, and orchestration
* **Modularity**: Install only what you need
* **Extensibility**: Easier to extend and customize
* **Maintainability**: Cleaner, more maintainable code

### 3. Enhanced Functionality

* **Resource Chaining**: Seamless workflows between components
* **Structured Configuration**: Better organization of parameters
* **Intelligent Defaults**: Smart default values based on context
* **Advanced Features**: Access to latest SageMaker capabilities

## Key Architectural Changes

### Package Structure

**V2 (Monolithic)**

```
sagemaker/
├── estimator.py
├── model.py
├── predictor.py
├── processing.py
├── workflow/
└── ...
```

**V3 (Modular)**

```
sagemaker-core/          # Foundation primitives and low-level API access
sagemaker-train/         # Training functionality
sagemaker-serve/         # Model serving and deployment
sagemaker-mlops/         # Workflow orchestration (Pipelines, Steps)
```

### Core Philosophy Changes

| Aspect | V2 | V3 |
|--------|----|----|  
| Architecture | Monolithic package | Modular packages |
| API Style | Class-based (Estimator) | Configuration-based (ModelTrainer) |
| Code Completion | Limited | Full IDE support with type hints |
| Configuration | Scattered parameters | Structured config objects |

## Migration Patterns

### 1. Training Jobs

**V2 Estimator → V3 ModelTrainer**

**V2 Code:**

```python
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

estimator = Estimator(
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-training-image",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size=30,
    max_run=3600,
    output_path="s3://my-bucket/output",
    hyperparameters={
        "epochs": 10,
        "batch_size": 32
    }
)
train_input = TrainingInput("s3://my-bucket/train")
estimator.fit(train_input)
```

**V3 Code:**

```python
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute, InputData

# Configuration objects provide structure and validation
compute = Compute(
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size_in_gb=30
)
source_code = SourceCode(
    source_dir="./src",
    entry_script="train.py"
)
model_trainer = ModelTrainer(
    training_image="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-training-image",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    source_code=source_code,
    compute=compute,
    hyperparameters={
        "epochs": 10,
        "batch_size": 32
    }
)
train_data = InputData(channel_name="train", data_source="s3://my-bucket/train")
model_trainer.train(input_data_config=[train_data])
```

### Framework Estimators

**V2 PyTorch:**

```python
from sagemaker.pytorch import PyTorch

pytorch_estimator = PyTorch(
    entry_point="train.py",
    source_dir="./src",
    role=role,
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    framework_version="1.12.0",
    py_version="py38",
    hyperparameters={
        "epochs": 10,
        "lr": 0.001
    }
)
```

**V3 Equivalent:**

```python
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute
from sagemaker.core import image_uris

# Get framework image
training_image = image_uris.retrieve(
    framework="pytorch",
    region="us-west-2",
    version="1.12.0",
    py_version="py38",
    instance_type="ml.p3.2xlarge",
    image_scope="training"
)

source_code = SourceCode(
    source_dir="./src",
    entry_script="train.py"
)

compute = Compute(
    instance_type="ml.p3.2xlarge",
    instance_count=1
)

model_trainer = ModelTrainer(
    training_image=training_image,
    source_code=source_code,
    compute=compute,
    hyperparameters={
        "epochs": 10,
        "lr": 0.001
    }
)
```

### 2. Model Deployment

**V2 Model → V3 ModelBuilder**

**V2 Code:**

```python
from sagemaker.model import Model

model = Model(
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-inference-image",
    model_data="s3://my-bucket/model.tar.gz",
    role=role
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)
```

**V3 Code:**

```python
from sagemaker.serve import ModelBuilder
from sagemaker.serve.configs import InferenceSpec

inference_spec = InferenceSpec(
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-inference-image",
    model_data_url="s3://my-bucket/model.tar.gz"
)

model_builder = ModelBuilder(
    inference_spec=inference_spec,
    role=role
)

predictor = model_builder.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1
)
```

### 3. Processing Jobs

**V2 Processor:**

```python
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    command=["python3"],
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/processing",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge"
)

processor.run(
    code="preprocess.py",
    inputs=[ProcessingInput(source="s3://bucket/input", destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination="s3://bucket/output")]
)
```

**V3 Code:**

```python
from sagemaker.mlops.processing import DataProcessor
from sagemaker.mlops.configs import ProcessingConfig, ProcessingInput, ProcessingOutput

processing_config = ProcessingConfig(
    instance_type="ml.m5.xlarge",
    instance_count=1,
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/processing"
)

processor = DataProcessor(
    processing_config=processing_config,
    role=role
)

processor.run(
    code="preprocess.py",
    inputs=[ProcessingInput(source="s3://bucket/input", destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination="s3://bucket/output")]
)
```

### 4. Pipelines and Workflows

**V2 Pipeline:**

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": TrainingInput(s3_data="s3://bucket/train")}
)

pipeline = Pipeline(
    name="MyPipeline",
    steps=[training_step]
)
```

**V3 Code:**

```python
from sagemaker.mlops.pipeline import Pipeline
from sagemaker.mlops.steps import TrainingStep

training_step = TrainingStep(
    name="TrainModel",
    model_trainer=model_trainer,
    inputs={"training": "s3://bucket/train"}
)

pipeline = Pipeline(
    name="MyPipeline",
    steps=[training_step]
)
```

## Feature Mapping

### Training Features

| V2 Feature | V3 Equivalent | Notes |
|------------|---------------|-------|
| Estimator | ModelTrainer | Core training interface |
| PyTorch, TensorFlow, etc. | ModelTrainer + image_uris | Framework-specific estimators replaced by generic trainer |
| fit() method | train() method | Method renamed for clarity |
| hyperparameters dict | hyperparameters dict | Same naming |
| TrainingInput | InputData | Simplified input configuration |
| instance_type/count | Compute config | Structured compute configuration |
| output_path | OutputDataConfig | More explicit output configuration |

### Serving Features

| V2 Feature | V3 Equivalent | Notes |
|------------|---------------|-------|
| Model | ModelBuilder | Enhanced model building |
| deploy() | deploy() | Similar interface |
| Predictor | Endpoint | Replaced with sagemaker-core |
| MultiDataModel | ModelBuilder | Multi-model endpoints |
| AsyncPredictor | ModelBuilder | Async inference |

### Processing Features

| V2 Feature | V3 Equivalent | Notes |
|------------|---------------|-------|
| Processor | ProcessingJob | Processing jobs |
| ScriptProcessor | ProcessingJob | Script-based processing |
| FrameworkProcessor | ProcessingJob | Framework-specific processing |

## Functionality Level Mapping

### Training

| User Journey | V2 Interface | V3 Interface | Status |
|--------------|--------------|--------------|--------|
| Train with custom Python scripts | `Estimator(entry_point='train.py')` | `ModelTrainer` | 🔄 REPLACED |
| Train with PyTorch | `PyTorch(entry_point=..., framework_version=...)` | `ModelTrainer` with PyTorch config | 🔄 REPLACED |
| Train with TensorFlow | `TensorFlow(entry_point=...)` | `ModelTrainer` with TF config | 🔄 REPLACED |
| Train with HuggingFace | `HuggingFace(entry_point=...)` | `ModelTrainer` + HuggingFace config | 🔄 REPLACED |
| Train with XGBoost / Built-in Algorithms | `sagemaker.amazon.XGBoost(...)` | `ModelTrainer` with algorithm config | 🔄 REPLACED |
| Train with SKLearn | `SKLearn(entry_point=...)` | `ModelTrainer` | 🔄 REPLACED |
| Train with custom Docker images | `Estimator(image_uri='...')` | `ModelTrainer(image_uri='...')` | 🔄 REPLACED |
| Train with SageMaker containers | Framework estimators (auto image) | `ModelTrainer` with framework config | 🔄 REPLACED |
| Distributed training | `Estimator(distribution={...})` | V3 example: custom-distributed-training-example.ipynb | 🔄 REPLACED |
| Local mode training | `Estimator(instance_type='local')` | V3 example: local-training-example.ipynb | 🔄 REPLACED |
| Hyperparameter tuning | `HyperparameterTuner(estimator=...)` | V3: 17 HPT classes in sagemaker-core | 🔄 REPLACED |
| Remote function (@remote) | `@remote` decorator | Still supported (v3.4.0+) | ✅ SUPPORTED |
| AWS Batch training | aws_batch/ | V3 example: aws_batch/ | 🔄 REPLACED |
| Train with MXNet | `MXNet(entry_point=...)` | Not available | ❌ REMOVED |
| Train with Chainer | `Chainer(entry_point=...)` | Not available | ❌ REMOVED |
| Train with RL | `RLEstimator(...)` | Not available | ❌ REMOVED |
| Training Compiler | `TrainingCompilerConfig(...)` | Not available | ❌ REMOVED |
| Fine-tuning SDK (SFT, RLVR, RLAIF) | Not available | V3.1.0: Standardized fine-tuning techniques | 🆕 NEW IN V3 |
| Model customization | Manual training scripts | model-customization-examples/ | 🆕 NEW IN V3 |
| Nova recipe training | Not available | V3.4.0: Nova recipe support in ModelTrainer | 🆕 NEW IN V3 |

### Data Processing

| User Journey | V2 Interface | V3 Interface | Status |
|--------------|--------------|--------------|--------|
| Run processing jobs | `Processor(...)`, `ScriptProcessor(...)` | `sagemaker.core.resources.ProcessingJob` | ❌ REMOVED |
| PySpark processing | `PySparkProcessor(...)` | `sagemaker.core.resources.ProcessingJob` | ❌ REMOVED |
| Feature Store | `FeatureGroup`, `FeatureStore` | `sagemaker.core.resources.ProcessingJob` | ❌ REMOVED |
| Data Wrangler | `sagemaker.wrangler` | `sagemaker.core.resources.ProcessingJob` | ❌ REMOVED |
| Ground Truth (Labeling) | Not directly in V2 SDK (AWS console) | `sagemaker.core.resources.GroundTruthJob` (28 classes) | 🆕 NEW IN V3 |

### Inference

| User Journey | V2 Interface | V3 Interface | Status |
|--------------|--------------|--------------|--------|
| Deploy to real-time endpoints | `model.deploy(...)` → `Predictor` | `ModelBuilder.build().deploy()` | 🔄 REPLACED |
| Make real-time predictions | `predictor.predict(data)` | Via ModelBuilder endpoint | 🔄 REPLACED |
| JumpStart model deployment | `JumpStartModel(model_id=...).deploy()` | V3 example: jumpstart-example.ipynb | 🔄 REPLACED |
| HuggingFace inference | `HuggingFaceModel(...).deploy()` | V3 example: huggingface-example.ipynb | 🔄 REPLACED |
| In-process mode inference | Not available | V3 example: in-process-mode-example.ipynb | 🆕 NEW IN V3 |
| InferenceSpec | Not available | V3 example: inference-spec-example.ipynb | 🆕 NEW IN V3 |
| Batch transform | `Transformer(model_name=...)` | `sagemaker.core.shapes.TransformJob` (6 classes) | 🔄 REPLACED |
| Serverless inference | `ServerlessInferenceConfig(...)` | Via ModelBuilder serverless config | 🔄 REPLACED |
| Async inference | `AsyncPredictor(...)` | Via ModelBuilder.deploy | 🔄 REPLACED |
| Multi-model endpoints | `MultiDataModel(...)` | Via ModelBuilder.deploy | 🔄 REPLACED |
| A/B testing (multi-variant) | Production variants with traffic splitting | ProductionVariant + traffic routing | 🔄 REPLACED |
| Endpoint auto-scaling | Via boto3 Application Auto Scaling | Via ModelBuilder.deploy | 🔄 REPLACED |
| Model compilation (Neo) | `Estimator.compile_model(...)` | `ModelBuilder.optimize()` | 🔄 REPLACED |

### MLOps and Workflows

| User Journey | V2 Interface | V3 Interface | Status |
|--------------|--------------|--------------|--------|
| Build ML pipelines | `Pipeline(steps=[...])` | V3: 31 Pipeline classes + PipelineVariables | 🔄 REPLACED |
| Model registry | `ModelPackage`, `ModelPackageGroup` | Via sagemaker-core ModelPackage | ❌ REMOVED |
| Experiment tracking | `Experiment`, `Trial`, `Run` | `sagemaker.core.experiments` (24 classes) | ❌ REMOVED |
| MLFlow integration | `sagemaker.mlflow` (limited) | V3: 8 MLFlow classes + metrics tracking | 🔄 REPLACED |
| Model monitoring | `ModelMonitor`, `DataQualityMonitor` | `sagemaker.core.shapes.MonitoringSchedule` (4 classes) | ❌ REMOVED |
| Lineage tracking | `sagemaker.lineage` | Still in V3 unchanged | ✅ UNCHANGED |
| Model cards | `ModelCard(...)` | Via sagemaker-core | ❌ REMOVED |
| Model dashboard | Limited | `sagemaker.core.shapes.ModelDashboard` (5 classes) | 🔄 REPLACED |
| AIRegistry | Not available | V3.1.0: Datasets and evaluators CRUD | 🆕 NEW IN V3 |
| Evaluator framework | Not available | V3.2.0: Evaluator + trainer handshake | 🆕 NEW IN V3 |
| EMR Serverless in Pipelines | Not available | V3.4.0: EMR-serverless step | 🆕 NEW IN V3 |

### JumpStart and Foundation Models

| User Journey | V2 Interface | V3 Interface | Status |
|--------------|--------------|--------------|--------|
| Deploy foundation models | `JumpStartModel(model_id=...).deploy()` | V3 examples + JumpStart module | 🔄 REPLACED |
| Fine-tune foundation models | `JumpStartEstimator(model_id=...).fit()` | V3: jumpstart-training-example.ipynb | 🔄 REPLACED |
| E2E training + inference | Separate steps | jumpstart-e2e-training-example.ipynb | 🔄 REPLACED |
| Marketplace algorithms | `sagemaker.algorithm.AlgorithmEstimator` | `sagemaker.core.resources.Algorithm` (13 classes) | 🔄 REPLACED |

## Configuration Management

### V2 Scattered Parameters

```python
estimator = Estimator(
    image_uri="...",
    role="...",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size=30,
    max_run=3600,
    use_spot_instances=True,
    max_wait=7200,
    subnets=["subnet-12345"],
    security_group_ids=["sg-12345"],
    encrypt_inter_container_traffic=True
)
```

### V3 Structured Configuration

```python
compute = Compute(
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size_in_gb=30,
    use_spot_instances=True,
    max_wait_time_in_seconds=7200
)
networking = Networking(
    subnets=["subnet-12345"],
    security_group_ids=["sg-12345"],
    enable_inter_container_traffic_encryption=True
)
stopping_condition = StoppingCondition(
    max_runtime_in_seconds=3600
)
model_trainer = ModelTrainer(
    training_image="...",
    role="...",
    compute=compute,
    networking=networking,
    stopping_condition=stopping_condition
)
```

## Advanced Features

### 1. Resource Chaining (V3 Only)

V3 introduces resource chaining for seamless workflows:

```python
# Train a model
model_trainer = ModelTrainer(...)
model_trainer.train()

# Chain training output to model builder
model_builder = ModelBuilder(model=model_trainer)

# Deploy the trained model
endpoint = model_builder.deploy()
```

### 2. Local Mode

**V2:**

```python
estimator = Estimator(
    instance_type="local",
    # ... other params
)
```

**V3:**

```python
from sagemaker.train import ModelTrainer, Mode

model_trainer = ModelTrainer(
    training_mode=Mode.LOCAL_CONTAINER,
    # ... other params
)
```

### 3. Distributed Training

**V2:**

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    # ... other params
    distribution={
        "smdistributed": {
            "dataparallel": {
                "enabled": True
            }
        }
    }
)
```

**V3:**

```python
from sagemaker.train.distributed import Distributed

distributed_config = Distributed(
    enabled=True
)
model_trainer = ModelTrainer(
    # ... other params
    distributed=distributed_config
)
```

## Migration Checklist

### Phase 1: Assessment

* Inventory current V2 usage patterns
* Identify framework-specific estimators in use
* Document custom configurations and extensions
* Review pipeline and workflow dependencies

### Phase 2: Package Installation

* Install required V3 packages: `pip install sagemaker-core sagemaker-train sagemaker-serve sagemaker-mlops`
  * You can also install `pip install sagemaker` to install all packages
* Update requirements.txt files
* Verify compatibility with existing dependencies

### Phase 3: Code Migration

* Replace Estimator with ModelTrainer
* Convert framework estimators to generic ModelTrainer + image URIs
* Restructure parameters into configuration objects
* Update method calls (fit() → train())
* Migrate model deployment code
* Update pipeline definitions

### Phase 4: Testing

* Test training jobs with new API
* Verify model deployment functionality
* Test pipeline execution
* Validate local mode functionality
* Performance testing and comparison

### Phase 5: Advanced Features

* Implement resource chaining where beneficial
* Leverage improved type hints and IDE support
* Optimize configurations using structured objects
* Explore new V3-specific features

## Common Migration Patterns

### 1. Simple Training Job

```python
# V2 → V3 transformation
def migrate_simple_training():
    # V2
    estimator = Estimator(
        image_uri=image,
        role=role,
        instance_type="ml.m5.xlarge",
        hyperparameters={"epochs": 10}
    )
    estimator.fit("s3://bucket/data")
    
    # V3
    model_trainer = ModelTrainer(
        training_image=image,
        role=role,
        compute=Compute(instance_type="ml.m5.xlarge"),
        hyperparameters={"epochs": 10}
    )
    
    train_data = InputData(channel_name="training", data_source="s3://bucket/data")
    model_trainer.train(input_data_config=[train_data])
```

### 2. Multi-Channel Training

```python
# V2 → V3 transformation
def migrate_multi_channel_training():
    # V2
    estimator.fit({
        "train": "s3://bucket/train",
        "validation": "s3://bucket/val"
    })
    
    # V3
    input_data_config = [
        InputData(channel_name="train", data_source="s3://bucket/train"),
        InputData(channel_name="validation", data_source="s3://bucket/val")
    ]
    model_trainer.train(input_data_config=input_data_config)
```

### 3. Custom Docker Images

```python
# V2 → V3 transformation remains similar
def migrate_custom_image():
    # V2
    estimator = Estimator(
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:latest",
        # ... other params
    )
    
    # V3
    model_trainer = ModelTrainer(
        training_image="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:latest",
        # ... other params
    )
```

## Troubleshooting Common Issues

### 1. Import Errors

```python
# V2 imports that need updating
from sagemaker.estimator import Estimator  # ❌
from sagemaker.train import ModelTrainer   # ✅

from sagemaker.model import Model          # ❌
from sagemaker.serve import ModelBuilder   # ✅
```

### 2. Parameter Mapping

```python
# V2 parameter that changed
estimator = Estimator(
    train_instance_type="ml.m5.xlarge",    # ❌ Old parameter name
    instance_type="ml.m5.xlarge"           # ✅ New parameter name
)

# V3 structured approach
compute = Compute(instance_type="ml.m5.xlarge")  # ✅
```

### 3. Method Name Changes

```python
# V2 method names
estimator.fit(inputs)           # ❌
model_trainer.train(inputs)     # ✅

# V2 deployment
model.deploy()                  # ❌ (different context)
model_builder.deploy()          # ✅
```

## Best Practices for V3

### 1. Use Configuration Objects

```python
# Preferred V3 pattern
compute = Compute(
    instance_type="ml.p3.2xlarge",
    instance_count=2,
    volume_size_in_gb=100
)
networking = Networking(
    enable_network_isolation=True,
    subnets=["subnet-12345"]
)

model_trainer = ModelTrainer(
    compute=compute,
    networking=networking,
    # ... other params
)
```

### 2. Leverage Resource Chaining

```python
# Chain resources for seamless workflows
training_job = model_trainer.train()
model_builder = ModelBuilder.from_training_job(training_job)
endpoint = model_builder.deploy()
```

### 3. Use Type Hints

```python
from typing import List
from sagemaker.train.configs import InputData

def setup_training_data() -> List[InputData]:
    return [
        InputData(channel_name="train", data_source="s3://bucket/train"),
        InputData(channel_name="val", data_source="s3://bucket/val")
    ]
```

## Conclusion

Migrating from SageMaker Python SDK V2 to V3 provides significant benefits in terms of developer experience, code organization, and access to new features. While the migration requires updating code patterns and imports, the structured approach of V3 leads to more maintainable and robust machine learning workflows.

The key to successful migration is understanding the new architectural patterns and gradually adopting the structured configuration approach that V3 promotes. Start with simple training jobs and gradually migrate more complex workflows, taking advantage of V3's enhanced features like resource chaining and improved type safety.