#!/usr/bin/env python3
"""
ModelBuilder V3 Manual Testing - Step 1.2

This file tests the physical ModelBuilder.build() workflow with real AWS resources.
WARNING: This creates actual AWS resources that need cleanup!
"""

import tempfile
import os
import boto3
import torch
from sagemaker.serve.model_builder import ModelBuilder, Compute
# from sagemaker.utils.jumpstart.model import JumpStartModel
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.helper.session_helper import Session

# AWS Account Configuration
AWS_ACCOUNT_ID = "593793038179"
AWS_REGION = "us-east-1"  # Default region, can be changed

# Global list to track created resources for cleanup
created_models = []

def setup_aws_session():
    """Set up AWS session for the test account."""
    try:
        # Create boto3 session (assumes ada credentials are already set)
        boto_session = boto3.Session()
        
        # Verify we can access the account
        sts = boto_session.client('sts')
        identity = sts.get_caller_identity()
        
        print(f"AWS Account: {identity['Account']}")
        print(f"AWS Region: {boto_session.region_name or AWS_REGION}")
        print(f"AWS User/Role: {identity['Arn']}")
        
        if identity['Account'] != AWS_ACCOUNT_ID:
            print(f"⚠️  Warning: Expected account {AWS_ACCOUNT_ID}, got {identity['Account']}")
        
        return boto_session
        
    except Exception as e:
        print(f"❌ Failed to set up AWS session: {e}")
        print("Please run: ada credentials update --account=593793038179 --provider=isengard --role=Admin --once")
        raise

def cleanup_resources():
    """Clean up all created AWS resources."""
    print("\n=== CLEANUP PHASE ===")
    for model in created_models:
        try:
            print(f"Deleting model: {model.model_name}")
            model.delete()
            print(f"✅ Successfully deleted {model.model_name}")
        except Exception as e:
            print(f"❌ Failed to delete {model.model_name}: {e}")
    
    print(f"Cleanup complete. Attempted to delete {len(created_models)} models.")

# Removed complex helper functions - using simple JumpStart models instead

def test_basic_build():
    """Test 1: Basic ModelBuilder.build() with JumpStart model (simplest pattern)."""
    print("\n=== TEST 1: Basic Build with JumpStart model ===")
    
    # Debug version information
    try:
        import sagemaker
        version = getattr(sagemaker, '__version__', 'dev')
        print(f"SageMaker version: {version}")
    except Exception as e:
        print(f"Could not get SageMaker version: {e}")
    
    try:
        # Simple sample input/output for text generation
        sample_input = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}
        sample_output = [{
            "generated_text": "Falcons are small to medium-sized birds of prey related to hawks and eagles."
        }]
        
        # Create schema builder with simple text data
        from sagemaker.serve.builder.schema_builder import SchemaBuilder
        schema_builder = SchemaBuilder(sample_input, sample_output)

        boto_session = boto3.Session(region_name="us-east-1")
        sagemaker_session = Session(boto_session=boto_session)

        compute=Compute(instance_type="ml.m5.large")
        
        # Simplest pattern: JumpStart model ID with explicit image_uri
        model_builder = ModelBuilder(
            model="gpt2",  # Simple JumpStart model
            schema_builder=schema_builder,
            # Use HuggingFace DLC for text generation
            image_uri="763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
            # role_arn="arn:aws:iam::593793038179:role/SageMakerExecutionRole",
            compute=compute,
            sagemaker_session=sagemaker_session
        )
        
        print("Building model (auto-detecting container)...")
        core_model = model_builder.build()
        
        print(f"✅ Build successful!")
        print(f"Model type: {type(core_model)}")
        print(f"Model name: {core_model.model_name}")
        # print(f"Model name: {core_model.name}")
        print(f"Model ARN: {getattr(core_model, 'model_arn', 'Not available')}")
        print(f"Primary container image: {getattr(core_model.primary_container, 'image', 'Not available')}")
        
        # Track for cleanup
        created_models.append(core_model)
        
        return core_model
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        # return test_basic_build_with_explicit_image()
        return None

def test_basic_build_with_explicit_image():
    """Test 1b: Basic ModelBuilder.build() with different JumpStart model (fallback)."""
    print("\n=== TEST 1b: Basic Build with different model ===")
    
    try:
        # Simple sample input/output
        sample_input = {"inputs": "Hello world"}
        sample_output = [{"generated_text": "Hello world, how are you?"}]
        
        # Create schema builder
        from sagemaker.serve.builder.schema_builder import SchemaBuilder
        schema_builder = SchemaBuilder(sample_input, sample_output)

        boto_session = boto3.Session(region_name="us-east-1")
        sagemaker_session = Session(boto_session=boto_session)
        
        # Try a different simple model
        model_builder = ModelBuilder(
            model="gpt2",  # Different JumpStart model
            schema_builder=schema_builder,
            instance_type="ml.m5.xlarge",
            sagemaker_session=sagemaker_session
        )
        
        print("Building model with explicit image_uri...")
        core_model = model_builder.build()
        
        print(f"✅ Build successful!")
        print(f"Model type: {type(core_model)}")
        print(f"Model name: {core_model.model_name}")
        print(f"Model ARN: {getattr(core_model, 'model_arn', 'Not available')}")
        print(f"Primary container image: {getattr(core_model.primary_container, 'image', 'Not available')}")
        
        # Track for cleanup
        created_models.append(core_model)
        
        return core_model
        
    except Exception as e:
        print(f"❌ Test 1b failed: {e}")
        return None

def test_build_with_vpc():
    """Test 2: ModelBuilder.build() with VPC configuration."""
    print("\n=== TEST 2: Build with VPC Config ===")
    
    try:
        # Same setup as test 1
        sample_input = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}
        sample_output = [{"generated_text": "Falcons are small to medium-sized birds of prey."}]
        
        from sagemaker.serve.builder.schema_builder import SchemaBuilder
        schema_builder = SchemaBuilder(sample_input, sample_output)

        boto_session = boto3.Session(region_name="us-east-1")
        sagemaker_session = Session(boto_session=boto_session)

        # VPC configuration using Network dataclass
        from sagemaker.serve.model_builder import Network
        network = Network(
            security_group_ids=["sg-12345678"],
            subnets=["subnet-12345678", "subnet-87654321"]
        )
        
        model_builder = ModelBuilder(
            model="gpt2",
            schema_builder=schema_builder,
            image_uri="763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
            network=network,  # Add VPC config
            sagemaker_session=sagemaker_session
        )
        
        print("Building model with VPC config...")
        core_model = model_builder.build()
        
        print(f"✅ VPC build successful!")
        print(f"Model name: {core_model.model_name}")
        print(f"VPC config: {getattr(core_model, 'vpc_config', 'Not available')}")
        
        created_models.append(core_model)
        return core_model
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return None

def test_build_with_custom_role():
    """Test 3: ModelBuilder.build() with custom execution role."""
    print("\n=== TEST 3: Build with Custom Role ===")
    
    try:
        # Same setup as test 1
        sample_input = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}
        sample_output = [{"generated_text": "Falcons are small to medium-sized birds of prey."}]
        
        from sagemaker.serve.builder.schema_builder import SchemaBuilder
        schema_builder = SchemaBuilder(sample_input, sample_output)

        boto_session = boto3.Session(region_name="us-east-1")
        sagemaker_session = Session(boto_session=boto_session)
        
        model_builder = ModelBuilder(
            model="gpt2",
            schema_builder=schema_builder,
            image_uri="763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
            role_arn=f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/SageMakerExecutionRole",  # Custom role
            sagemaker_session=sagemaker_session
        )
        
        print("Building model with custom role...")
        core_model = model_builder.build()
        
        print(f"✅ Custom role build successful!")
        print(f"Model name: {core_model.model_name}")
        print(f"Execution role: {getattr(core_model, 'execution_role_arn', 'Not available')}")
        
        created_models.append(core_model)
        return core_model
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return None


def test_core_model_operations():
    """Test 4: Test Core Model operations (get, refresh, etc.)."""
    print("\n=== TEST 4: Core Model Operations ===")
    
    # Use model from Test 1
    if not created_models:
        print("❌ No models available for operations test")
        return
    
    try:
        core_model = created_models[0]
        
        print(f"Testing operations on model: {core_model.model_name}")
        
        # Test refresh
        print("Refreshing model...")
        refreshed_model = core_model.refresh()
        print(f"✅ Refresh successful: {refreshed_model.model_name}")
        
        # Test get_name
        print("Getting model name...")
        name = core_model.get_name()
        print(f"✅ Model name: {name}")
        
        # Test attributes
        print("Model attributes:")
        print(f"  - Creation time: {getattr(core_model, 'creation_time', 'Not available')}")
        print(f"  - Primary container: {getattr(core_model, 'primary_container', 'Not available')}")
        print(f"  - Enable network isolation: {getattr(core_model, 'enable_network_isolation', 'Not available')}")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")

def main():
    """Run all manual tests for ModelBuilder.build()."""
    print("🚀 Starting ModelBuilder V3 Manual Testing (Step 1.2)")
    print("⚠️  WARNING: This will create real AWS resources!")
    
    # Set up AWS session
    print("\n=== AWS SESSION SETUP ===")
    try:
        boto_session = setup_aws_session()
        print("✅ AWS session configured successfully")
    except Exception as e:
        print(f"❌ Failed to set up AWS session: {e}")
        return
    
    # Confirm with user
    response = input("\nDo you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Testing cancelled.")
        return
    
    try:
        # Run realistic tests (no mocks)
        test_basic_build()  # Will try auto-detection first, then fallback
        # test_build_with_vpc()
        # test_build_with_custom_role()
        test_core_model_operations()
        
        print("\n🎉 All tests completed!")
        print(f"Created {len(created_models)} models for testing.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user.")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        # Always attempt cleanup
        cleanup_response = input("\nDo you want to clean up created resources? (Y/n): ")
        if cleanup_response.lower() != 'n':
            cleanup_resources()
        else:
            print("⚠️  Resources left for manual cleanup:")
            for model in created_models:
                print(f"  - {model.model_name}")

if __name__ == "__main__":
    main()