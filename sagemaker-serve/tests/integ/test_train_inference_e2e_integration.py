# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
import uuid
import tempfile
import os
import pytest
import logging

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import SourceCode
from sagemaker.core.resources import EndpointConfig

logger = logging.getLogger(__name__)

# Configuration - easily customizable
MODEL_NAME_PREFIX = "train-inf-v3-test-model"
ENDPOINT_NAME_PREFIX = "train-inf-v3-test-endpoint"
TRAINING_JOB_PREFIX = "e2e-v3-pytorch"

# Configuration
AWS_REGION = "us-west-2"
PYTORCH_TRAINING_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-cpu-py39"


@pytest.mark.slow_test
def test_train_inference_e2e_build_deploy_invoke_cleanup():
    """Integration test for Train-Inference E2E workflow"""
    logger.info("Starting Train-Inference E2E integration test...")
    
    model_trainer = None
    core_model = None
    core_endpoint = None
    
    try:
        # Step 1: Train model
        logger.info("Training model...")
        model_trainer, unique_id = train_model()
        
        # Step 2: Build and deploy
        logger.info("Building and deploying model...")
        core_model, core_endpoint = build_and_deploy(model_trainer, unique_id)
        
        # Step 3: Test inference
        logger.info("Making prediction...")
        make_prediction(core_endpoint)
        
        # Test passed successfully
        logger.info("Train-Inference E2E integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Train-Inference E2E integration test failed: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if core_model and core_endpoint:
            logger.info("Cleaning up resources...")
            cleanup_resources(core_model, core_endpoint)


def create_pytorch_training_code():
    """Create PyTorch training script."""
    temp_dir = tempfile.mkdtemp()
    
    train_script = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
    
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)

def train():
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic data
    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Train for 1 epoch
    model.train()
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Save model for TorchServe
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1, 4))
    
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    os.makedirs(model_dir, exist_ok=True)
    torch.jit.save(traced_model, os.path.join(model_dir, 'model.pth'))
    
    print("Training completed and model saved!")

if __name__ == "__main__":
    train()
'''
    
    with open(os.path.join(temp_dir, 'train.py'), 'w') as f:
        f.write(train_script)
    
    with open(os.path.join(temp_dir, 'requirements.txt'), 'w') as f:
        f.write('torch>=1.13.0,<2.0.0\n')
    
    return temp_dir


def create_schema_builder():
    """Create schema builder for tensor-based models."""
    sample_input = [[0.1, 0.2, 0.3, 0.4]]
    sample_output = [[0.8, 0.2]]
    return SchemaBuilder(sample_input, sample_output)


def train_model():
    """Train model using ModelTrainer."""
    from sagemaker.core.helper.session_helper import Session
    import boto3
    
    # Create SageMaker session with AWS region
    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = Session(boto_session=boto_session)
    
    training_code_dir = create_pytorch_training_code()
    unique_id = str(uuid.uuid4())[:8]
    
    model_trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=PYTORCH_TRAINING_IMAGE,
        source_code=SourceCode(
            source_dir=training_code_dir,
            entry_script="train.py",
            requirements="requirements.txt",
        ),
        base_job_name=f"{TRAINING_JOB_PREFIX}-{unique_id}"
    )
    
    model_trainer.train()
    logger.info("Model Training Completed!")
    
    return model_trainer, unique_id


def build_and_deploy(model_trainer, unique_id):
    """Build and deploy model using ModelBuilder."""
    from sagemaker.serve.spec.inference_spec import InferenceSpec
    
    class SimpleInferenceSpec(InferenceSpec):
        def load(self, model_dir):
            import torch
            return torch.jit.load(f"{model_dir}/model.pth")
        
        def invoke(self, input_object, model):
            import torch
            return model(torch.tensor(input_object)).tolist()

    schema_builder = create_schema_builder()
    
    model_builder = ModelBuilder(
        model=model_trainer,
        schema_builder=schema_builder,
        model_server=ModelServer.TORCHSERVE,
        inference_spec=SimpleInferenceSpec(),
        image_uri=PYTORCH_TRAINING_IMAGE.replace("training", "inference"),
        dependencies={"auto": False},
    )
    
    core_model = model_builder.build(model_name=f"{MODEL_NAME_PREFIX}-{unique_id}", region="us-west-2")
    logger.info(f"Model Successfully Created: {core_model.model_name}")
    
    core_endpoint = model_builder.deploy(
        endpoint_name=f"{ENDPOINT_NAME_PREFIX}-{unique_id}",
        initial_instance_count=1
    )
    logger.info(f"Endpoint Successfully Created: {core_endpoint.endpoint_name}")
    
    return core_model, core_endpoint


def make_prediction(core_endpoint):
    """Make prediction using the deployed endpoint - preserving exact logic from manual test"""
    test_data = [[0.1, 0.2, 0.3, 0.4]]
    
    result = core_endpoint.invoke(
        body=json.dumps(test_data),
        content_type="application/json"
    )

    prediction = json.loads(result.body.read().decode('utf-8'))
    logger.info(f"Result of invoking endpoint: {prediction}")


def cleanup_resources(core_model, core_endpoint):
    """Fully clean up model and endpoint creation - preserving exact logic from manual test"""
    core_endpoint_config = EndpointConfig.get(endpoint_config_name=core_endpoint.endpoint_name)
   
    core_model.delete()
    core_endpoint.delete()
    core_endpoint_config.delete()

    logger.info("Model and Endpoint Successfully Deleted!")