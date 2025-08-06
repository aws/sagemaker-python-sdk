#!/usr/bin/env python3

import json
import os
import sys
import boto3
import time
from datetime import datetime
# from urllib.parse import urlparse
from unittest.mock import patch

os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.enums import JumpStartModelType


def check_aws_account():
    """Check which AWS account and region we're using."""
    try:
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        
        account_id = identity['Account']
        user_arn = identity['Arn']
        region = boto3.Session().region_name or 'us-west-2'
        
        print(f" AWS Account: {account_id}")
        print(f" User/Role: {user_arn}")
        print(f" Region: {region}")
        print()
        
        return account_id, region
    except Exception as e:
        print(f" Error checking AWS account: {e}")
        return None, None


def monitor_endpoint(endpoint_name, region='us-west-2'):
    """Monitor endpoint deployment progress."""
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    print(f" Monitoring endpoint: {endpoint_name}")
    start_time = time.time()
    
    while True:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            elapsed = int(time.time() - start_time)
            
            print(f"  [{elapsed//60}m {elapsed%60}s] {endpoint_name}: {status}")
            
            if status == 'InService':
                print(f" {endpoint_name} is ready! (took {elapsed//60}m {elapsed%60}s)")
                break
            elif status == 'Failed':
                print(f" {endpoint_name} deployment failed!")
                print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
                break
                
        except Exception as e:
            print(f"Error checking {endpoint_name}: {e}")
            
        time.sleep(30)  # Check every 30 seconds

def load_custom_spec():
    """Load the custom spec file from src/sagemaker directory."""
    spec_path = os.path.join(os.path.dirname(__file__), 'specfileex')
    with open(spec_path, 'r') as f:
        return json.load(f)


# Check AWS account 
account_id, region = check_aws_account()

custom_spec = load_custom_spec()
mock_specs = JumpStartModelSpecs(custom_spec)

with patch('sagemaker.jumpstart.cache.JumpStartModelsCache.get_specs') as mock_get_specs, \
     patch('sagemaker.jumpstart.utils.validate_model_id_and_get_type') as mock_validate_model:
    
    mock_get_specs.return_value = mock_specs
    mock_validate_model.return_value = JumpStartModelType.OPEN_WEIGHTS
    
    model_id = "meta-textgeneration-llama-2-7b-f"
    model_version = "4.19.0"
    accept_eula = False
    
    # Create unique endpoint names with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    neuron_endpoint_name = f"llama-neuron-{timestamp}"
    gpu_endpoint_name = f"llama-gpu-{timestamp}"
    
    print(f" Neuron endpoint: {neuron_endpoint_name}")
    print(f" GPU endpoint: {gpu_endpoint_name}")
    print()

    
    model_neuron = JumpStartModel(
        model_id=model_id, 
        model_version=model_version,
        instance_type="ml.inf2.24xlarge",
        env={"HUGGING_FACE_HUB_TOKEN": "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"}
    )

    # Modify to use alpha us-west-2 bucket
    original_neuron_uri = model_neuron.model_data['S3DataSource']['S3Uri']
    # Replace with alpha us-west-2 bucket (handle both east-1 and west-2 original buckets)
    alpha_neuron_uri = original_neuron_uri.replace('jumpstart-private-cache-prod-us-east-1', 'jumpstart-private-cache-alpha-us-west-2')
    alpha_neuron_uri = alpha_neuron_uri.replace('jumpstart-private-cache-prod-us-west-2', 'jumpstart-private-cache-alpha-us-west-2')
    # Also handle regular cache buckets (without "private")
    alpha_neuron_uri = alpha_neuron_uri.replace('jumpstart-cache-prod-us-east-1', 'jumpstart-cache-alpha-us-west-2')
    alpha_neuron_uri = alpha_neuron_uri.replace('jumpstart-cache-prod-us-west-2', 'jumpstart-cache-alpha-us-west-2')
    model_neuron.model_data['S3DataSource']['S3Uri'] = alpha_neuron_uri
    print(f"Original neuron URI: {original_neuron_uri}")
    print(f"Alpha neuron URI: {alpha_neuron_uri}")
    print(model_neuron.model_data)
    neuron_location = model_neuron.model_data['S3DataSource']['S3Uri']
    print(f"Neuron location: {neuron_location}")

    print("Deploying neuron model...")
    neuron_predictor = model_neuron.deploy(
        initial_instance_count=1,
        instance_type="ml.inf2.24xlarge",
        endpoint_name=neuron_endpoint_name,
        accept_eula=True,
        wait=False 
    )
    
    # Monitor neuron deployment
    monitor_endpoint(neuron_endpoint_name, 'us-west-2')



    model_gpu = JumpStartModel(
        model_id=model_id, 
        model_version=model_version, 
        instance_type="ml.g5.12xlarge",
        env={"HUGGING_FACE_HUB_TOKEN": "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"}
    )

    # Modify to use alpha us-west-2 bucket  
    original_gpu_uri = model_gpu.model_data['S3DataSource']['S3Uri']
    # Replace with alpha us-west-2 bucket (handle both east-1 and west-2 original buckets)
    alpha_gpu_uri = original_gpu_uri.replace('jumpstart-private-cache-prod-us-east-1', 'jumpstart-private-cache-alpha-us-west-2')
    alpha_gpu_uri = alpha_gpu_uri.replace('jumpstart-private-cache-prod-us-west-2', 'jumpstart-private-cache-alpha-us-west-2')
    # Also handle regular cache buckets (without "private")
    alpha_gpu_uri = alpha_gpu_uri.replace('jumpstart-cache-prod-us-east-1', 'jumpstart-cache-alpha-us-west-2')
    alpha_gpu_uri = alpha_gpu_uri.replace('jumpstart-cache-prod-us-west-2', 'jumpstart-cache-alpha-us-west-2')
    model_gpu.model_data['S3DataSource']['S3Uri'] = alpha_gpu_uri
    print(f"Original GPU URI: {original_gpu_uri}")
    print(f"Alpha GPU URI: {alpha_gpu_uri}")
    print(model_gpu.model_data)
    gpu_location = model_gpu.model_data['S3DataSource']['S3Uri']
    print(f"GPU location: {gpu_location}")

    print("Deploying GPU model...")
    gpu_predictor = model_gpu.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.12xlarge",
        endpoint_name=gpu_endpoint_name,
        accept_eula=True,
        wait=False  
    )
    
    # Monitor GPU deployment  
    monitor_endpoint(gpu_endpoint_name, 'us-west-2')

    test_payload = {
        "inputs": "The meaning of life is",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7
        }
    }

    print("Testing neuron endpoint...")
    neuron_response = neuron_predictor.predict(test_payload)
    print(f"Neuron response: {neuron_response}")

    print("Testing GPU endpoint...")
    gpu_response = gpu_predictor.predict(test_payload)
    print(f"GPU response: {gpu_response}")


    #print("Cleaning up endpoints...")
    #neuron_predictor.delete_endpoint()
    #gpu_predictor.delete_endpoint()
