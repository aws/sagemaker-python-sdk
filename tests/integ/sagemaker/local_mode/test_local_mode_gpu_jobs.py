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
import time
from typing import Union


import os
import re
import pytest
import subprocess
import logging
import sagemaker
import boto3
import urllib3
from pathlib import Path
from sagemaker.huggingface import (
    HuggingFaceModel, 
    get_huggingface_llm_image_uri
)
from sagemaker.deserializers import JSONDeserializer
from sagemaker.local import LocalSession
from sagemaker.serializers import JSONSerializer


# Replace this role ARN with an appropriate role for your environment
ROLE = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"


def ensure_docker_compose_installed():
    """
    Downloads the Docker Compose plugin if not present, and verifies installation
    by checking the output of 'docker compose version' matches the pattern:
    'Docker Compose version vX.Y.Z'
    """

    cli_plugins_path = Path.home() / ".docker" / "cli-plugins"
    cli_plugins_path.mkdir(parents=True, exist_ok=True)

    compose_binary_path = cli_plugins_path / "docker-compose"
    if not compose_binary_path.exists():
        subprocess.run(
            [
                "curl",
                "-SL",
                "https://github.com/docker/compose/releases/download/v2.3.3/docker-compose-linux-x86_64",
                "-o",
                str(compose_binary_path),
            ],
            check=True,
        )
        subprocess.run(["chmod", "+x", str(compose_binary_path)], check=True)

    # Verify Docker Compose version
    try:
        output = subprocess.check_output(["docker", "compose", "version"], stderr=subprocess.STDOUT)
        output_decoded = output.decode("utf-8").strip()
        logging.info(f"'docker compose version' output: {output_decoded}")

        # Example expected format: "Docker Compose version vxxx"
        pattern = r"Docker Compose version+"
        match = re.search(pattern, output_decoded)
        assert (
            match is not None
        ), f"Could not find a Docker Compose version string matching '{pattern}' in: {output_decoded}"

    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Failed to verify Docker Compose: {e}")


"""
Local Model: HuggingFace LLM Inference
"""
@pytest.mark.local
def test_huggingfacellm_local_model_inference():
    """
    Test local mode inference with DJL-LMI inference containers
    without a model_data path provided at runtime. This test should 
    be run on a GPU only machine with instance set to local_gpu.
    """
    ensure_docker_compose_installed()

    # 1. Create a local session for inference
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}
    
    djllmi_model = sagemaker.Model(
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.31.0-lmi13.0.0-cu124",
        env={
            "HF_MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "OPTION_MAX_MODEL_LEN": "10000",
            "OPTION_GPU_MEMORY_UTILIZATION": "0.95",
            "OPTION_ENABLE_STREAMING": "false",
            "OPTION_ROLLING_BATCH": "auto",
            "OPTION_MODEL_LOADING_TIMEOUT": "3600",
            "OPTION_PAGED_ATTENTION": "false",
            "OPTION_DTYPE": "fp16",
        },
        role=ROLE,
        sagemaker_session=sagemaker_session
    )

    logging.warning('Deploying endpoint in local mode')
    logging.warning(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.'
    )

    endpoint_name = "test-djl"
    djllmi_model.deploy(
         endpoint_name=endpoint_name,
         initial_instance_count=1,
         instance_type="local_gpu",
         container_startup_health_check_timeout=600,
    )
    predictor = sagemaker.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )
    test_response = predictor.predict(
        {
            "inputs": """<|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are a helpful assistant that thinks and reasons before answering.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            What's 2x2?
            <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>
            """
        }
    )
    logging.warning(test_response)
    gen_text = test_response['generated_text']
    logging.warning(f"\n=======\nmodel response: {gen_text}\n=======\n")
    
    assert type(test_response) == dict, f"invalid model response format: {gen_text}"
    assert type(gen_text) == str, f"assistant response format: {gen_text}"
    
    logging.warning('About to delete the endpoint')
    predictor.delete_endpoint()


"""
Local Model: HuggingFace TGI Inference
"""
@pytest.mark.local
def test_huggingfacetgi_local_model_inference():
    """
    Test local mode inference with HuggingFace TGI inference containers
    without a model_data path provided at runtime. This test should 
    be run on a GPU only machine with instance set to local_gpu.
    """
    ensure_docker_compose_installed()

    # 1. Create a local session for inference
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    huggingface_model = HuggingFaceModel(
         image_uri=get_huggingface_llm_image_uri(
             "huggingface", 
             version="2.3.1"
         ),
        env={
            "HF_MODEL_ID": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "ENDPOINT_SERVER_TIMEOUT": "3600",
            "MESSAGES_API_ENABLED": "true",
            "OPTION_ENTRYPOINT": "inference.py",
            "SAGEMAKER_ENV": "1",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            "SAGEMAKER_PROGRAM": "inference.py",
            "SM_NUM_GPUS": "1",
            "MAX_TOTAL_TOKENS": "1024",
            "MAX_INPUT_TOKENS": "800",
            "MAX_BATCH_PREFILL_TOKENS": "900",
            "DTYPE": "bfloat16",
            "PORT": "8080"
        },
        role=ROLE,
        sagemaker_session=sagemaker_session
    )

    logging.warning('Deploying endpoint in local mode')
    logging.warning(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.'
    )

    endpoint_name = "test-hf"
    huggingface_model.deploy(
         endpoint_name=endpoint_name,
         initial_instance_count=1,
         instance_type="local_gpu",
         container_startup_health_check_timeout=600,
    )
    predictor = sagemaker.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )
    test_response = predictor.predict(
        {
            "messages": [
                    {"role": "system", "content": "You are a helpful assistant." },
                    {"role": "user", "content": "What is 2x2?"}
                ]
        }
    )
    logging.warning(test_response)
    gen_text = test_response['choices'][0]['message']
    logging.warning(f"\n=======\nmodel response: {gen_text}\n=======\n")
    
    assert type(gen_text) == dict, f"invalid model response: {gen_text}"
    assert gen_text['role'] == 'assistant', f"assistant response missing: {gen_text}"
    
    logging.warning('About to delete the endpoint')
    predictor.delete_endpoint()

    

