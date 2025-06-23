# Quickstart

The SageMaker Python SDK provides a high-level interface for training and deploying machine learning models. The new `ModelTrainer` class simplifies the process even further.

Here's a simple example to get you started:

```python
import sagemaker
from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode

# Define a training image
pytorch_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"

# Define source code
source_code = SourceCode(
    source_dir="my_training_code",
    entry_script="train.py",
)

# Create the ModelTrainer
model_trainer = ModelTrainer(
    training_image=pytorch_image,
    source_code=source_code,
    base_job_name="my-first-training-job",
)

# Start the training job
model_trainer.train()
```