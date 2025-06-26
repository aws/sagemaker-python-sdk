---
title: Training with SageMaker Python SDK
description: Learn about different training approaches in the SageMaker Python SDK
---
(training-overview)=
# Training with SageMaker Python SDK

The SageMaker Python SDK provides multiple approaches to train machine learning models on Amazon SageMaker, from high-level abstractions to resource-level controls. This guide covers the key training interfaces and helps you choose the right approach for your needs.

::: {admonition} Choose the right interface
:class: tip

Each training interface offers different levels of abstraction and control. Consider your specific needs when selecting an interface.
:::

## Training Interfaces Overview

| Interface            | Description                                                  | Best For                                        |
|----------------------|--------------------------------------------------------------|-------------------------------------------------|
| **ModelTrainer**     | Modern, intuitive interface with simplified configuration    | New users, simplified workflows                 |
| **Estimator**        | Traditional interface with extensive framework support       | Framework-specific training, legacy workflows   |
| **Algorithm Estimator** | Specialized interface for SageMaker built-in algorithms | Using SageMaker's pre-built algorithms          |
| **SageMaker Core**   | Low-level resource abstractions over boto3                   | Advanced users, fine-grained control            |

## Training Workflow

Regardless of which interface you choose, the general workflow for training in SageMaker follows these steps:

1. **Prepare your training script** - Create a Python script containing your training logic
2. **Configure your training job** - Set up the training environment, resources, and inputs
3. **Launch training** - Execute the training job on SageMaker infrastructure
4. **Monitor progress** - Track metrics and logs during training
5. **Save the model** - Store trained model artifacts for deployment

## Choosing the Right Interface

::::{grid}
:gutter: 3

:::{grid-item-card}
:columns: 6

### ModelTrainer

Modern, intuitive interface with simplified configuration classes.

**Recommended for new users and simplified workflows**
:::

:::{grid-item-card}
:columns: 6

### Estimator

Traditional interface with extensive framework support.

**Best for framework-specific training and legacy workflows**
:::

:::{grid-item-card}
:columns: 6

### Algorithm Estimator

Specialized interface for SageMaker built-in algorithms.

**Ideal for using SageMaker's pre-built algorithms**
:::

:::{grid-item-card}
:columns: 6

### SageMaker Core

Low-level resource abstractions over boto3.

**For advanced users needing fine-grained control**
:::
::::

## ModelTrainer

The `ModelTrainer` is the newest and recommended interface for training in SageMaker. It provides an intuitive, simplified experience with these key benefits:

- **Improved usability** through configuration classes and minimal core parameters
- **Native script mode support** without requiring the SageMaker Training Toolkit
- **Simplified distributed training** with dedicated configuration classes
- **Bring your own container (BYOC)** without adaptation for SageMaker

```python
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode

# Define the training image and source code
pytorch_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"
source_code = SourceCode(command="python train.py --epochs 10")

# Create and launch the training job
model_trainer = ModelTrainer(
    training_image=pytorch_image,
    source_code=source_code,
    base_job_name="my-training-job",
)
model_trainer.train()
```

```{toctree}
:hidden:
:maxdepth: 1

train/modeltrainer
train/estimator
train/remote-decorator
train/others
```