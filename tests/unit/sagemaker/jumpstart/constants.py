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

# flake8: noqa: E501

SPECIAL_MODEL_SPECS_DICT = {
    "js-model-class-model-prepacked": {
        "model_id": "huggingface-txt2img-conflictx-complex-lineart",
        "url": "https://huggingface.co/Conflictx/Complex-Lineart",
        "version": "2.0.3",
        "min_sdk_version": "2.189.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.10.2",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
        },
        "hosting_artifact_key": "huggingface-txt2img/huggingface-txt2img-conflictx-complex-lineart/artifacts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/txt2img/v1.1.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-txt2img/huggingface-txt2img-conflictx-complex-lineart/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [
            "accelerate==0.16.0",
            "diffusers==0.12.1",
            "huggingface_hub==0.12.0",
            "transformers==4.26.0",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.p3.2xlarge",
        "supported_inference_instance_types": [
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json", "application/x-text"],
            "supported_accept_types": [
                "application/json",
                "application/json;verbose",
                "application/json;jpeg",
            ],
            "default_content_type": "application/x-text",
            "default_accept_type": "application/json;jpeg",
        },
        "inference_enable_network_isolation": True,
        "validation_supported": False,
        "fine_tuning_supported": False,
        "resource_name_base": "hf-txt2img-conflictx-complex-lineart",
        "default_payloads": {
            "Astronaut": {"content_type": "application/x-text", "body": "astronaut on a horse"}
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "gemma-model": {
        "model_id": "huggingface-llm-gemma-7b-instruct",
        "url": "https://huggingface.co/google/gemma-7b-it",
        "version": "1.1.0",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "1.4.2",
            "py_version": "py310",
            "huggingface_transformers_version": "4.33.2",
        },
        "hosting_artifact_key": "huggingface-llm/huggingface-llm-gemma-7b-instruct/artifacts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/llm/v1.0.1/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-llm/huggingface-llm-gemma-7b-i"
        "nstruct/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "hosting_eula_key": "fmhMetadata/terms/gemmaTerms.txt",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "accelerate==0.26.1",
            "bitsandbytes==0.42.0",
            "deepspeed==0.10.3",
            "docstring-parser==0.15",
            "flash_attn==2.5.5",
            "ninja==1.11.1",
            "packaging==23.2",
            "peft==0.8.2",
            "py_cpuinfo==9.0.0",
            "rich==13.7.0",
            "safetensors==0.4.2",
            "sagemaker_jumpstart_huggingface_script_utilities==1.2.1",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
            "shtab==1.6.5",
            "tokenizers==0.15.1",
            "transformers==4.38.1",
            "trl==0.7.10",
            "tyro==0.7.2",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "peft_type",
                "type": "text",
                "default": "lora",
                "options": ["lora", "None"],
                "scope": "algorithm",
            },
            {
                "name": "instruction_tuned",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "chat_dataset",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "epoch",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "lora_r",
                "type": "int",
                "default": 64,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {"name": "lora_alpha", "type": "int", "default": 16, "min": 0, "scope": "algorithm"},
            {
                "name": "lora_dropout",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "bits", "type": "int", "default": 4, "scope": "algorithm"},
            {
                "name": "double_quant",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "quant_type",
                "type": "text",
                "default": "nf4",
                "options": ["fp4", "nf4"],
                "scope": "algorithm",
            },
            {
                "name": "per_device_train_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "per_device_eval_batch_size",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "warmup_ratio",
                "type": "float",
                "default": 0.1,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_from_scratch",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "fp16",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "bf16",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "evaluation_strategy",
                "type": "text",
                "default": "steps",
                "options": ["steps", "epoch", "no"],
                "scope": "algorithm",
            },
            {
                "name": "eval_steps",
                "type": "int",
                "default": 20,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "gradient_accumulation_steps",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "logging_steps",
                "type": "int",
                "default": 8,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "weight_decay",
                "type": "float",
                "default": 0.2,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "load_best_model_at_end",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_val_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "seed",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": 2048,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_data_split_seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {"name": "max_steps", "type": "int", "default": -1, "scope": "algorithm"},
            {
                "name": "gradient_checkpointing",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_patience",
                "type": "int",
                "default": 3,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_threshold",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta1",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta2",
                "type": "float",
                "default": 0.999,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_epsilon",
                "type": "float",
                "default": 1e-08,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "max_grad_norm",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "label_smoothing_factor",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_first_step",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "logging_nan_inf_filter",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "save_strategy",
                "type": "text",
                "default": "steps",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "save_steps", "type": "int", "default": 500, "min": 1, "scope": "algorithm"},
            {"name": "save_total_limit", "type": "int", "default": 1, "scope": "algorithm"},
            {
                "name": "dataloader_drop_last",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "dataloader_num_workers",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "eval_accumulation_steps",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "auto_find_batch_size",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "lr_scheduler_type",
                "type": "text",
                "default": "constant_with_warmup",
                "options": ["constant_with_warmup", "linear"],
                "scope": "algorithm",
            },
            {"name": "warmup_steps", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "deepspeed",
                "type": "text",
                "default": "False",
                "options": ["False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/llm/v1.1.1/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/"
        "llm/prepack/v1.1.1/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.1",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-llm-gemma-7b-instruct.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "8191",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "8192",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_BATCH_PREFILL_TOKENS",
                "type": "text",
                "default": "8191",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {
                "Name": "huggingface-textgeneration:eval-loss",
                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
            },
            {"Name": "huggingface-textgeneration:train-loss", "Regex": "'loss': ([0-9]+\\.[0-9]+)"},
        ],
        "default_inference_instance_type": "ml.g5.12xlarge",
        "supported_inference_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "default_training_instance_type": "ml.g5.12xlarge",
        "supported_training_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 1200,
            "container_startup_health_check_timeout": 1200,
        },
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
            "disable_output_compression": True,
            "max_run": 360000,
        },
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 512,
        "training_volume_size": 512,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/oasst_top/train/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "hf-llm-gemma-7b-instruct",
        "gated_bucket": True,
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/h"
                    "uggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:"
                    "2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "huggingface-training/g4dn/v1.0.0/train-hugg"
                        "ingface-llm-gemma-7b-instruct.tar.gz"
                    },
                },
                "g5": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "huggingface-training/g5/v1.0.0/train-huggingf"
                        "ace-llm-gemma-7b-instruct.tar.gz"
                    },
                },
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "huggingface-training/p3dn/v1.0.0/train-hugg"
                        "ingface-llm-gemma-7b-instruct.tar.gz"
                    },
                },
                "p4d": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "huggingface-training/"
                        "p4d/v1.0.0/train-huggingface-llm-gemma-7b-instruct.tar.gz"
                    },
                },
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
            "hosting_artifact_s3_data_type": "S3Prefix",
            "hosting_artifact_compression_type": "None",
            "hosting_resource_requirements": {"min_memory_mb": 98304, "num_accelerators": 4},
            "dynamic_container_deployment_supported": True,
        },
    },
    "gemma-model-1-artifact": {
        "model_id": "huggingface-llm-gemma-7b-instruct",
        "url": "https://huggingface.co/google/gemma-7b-it",
        "version": "1.1.0",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "1.4.2",
            "py_version": "py310",
            "huggingface_transformers_version": "4.33.2",
        },
        "hosting_artifact_key": "huggingface-llm/huggingface-llm-gemma-7b-instruct/artifacts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/llm/v1.0.1/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-llm/huggingface-llm-gemma-7b-i"
        "nstruct/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "hosting_eula_key": "fmhMetadata/terms/gemmaTerms.txt",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "accelerate==0.26.1",
            "bitsandbytes==0.42.0",
            "deepspeed==0.10.3",
            "docstring-parser==0.15",
            "flash_attn==2.5.5",
            "ninja==1.11.1",
            "packaging==23.2",
            "peft==0.8.2",
            "py_cpuinfo==9.0.0",
            "rich==13.7.0",
            "safetensors==0.4.2",
            "sagemaker_jumpstart_huggingface_script_utilities==1.2.1",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
            "shtab==1.6.5",
            "tokenizers==0.15.1",
            "transformers==4.38.1",
            "trl==0.7.10",
            "tyro==0.7.2",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "peft_type",
                "type": "text",
                "default": "lora",
                "options": ["lora", "None"],
                "scope": "algorithm",
            },
            {
                "name": "instruction_tuned",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "chat_dataset",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "epoch",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "lora_r",
                "type": "int",
                "default": 64,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {"name": "lora_alpha", "type": "int", "default": 16, "min": 0, "scope": "algorithm"},
            {
                "name": "lora_dropout",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "bits", "type": "int", "default": 4, "scope": "algorithm"},
            {
                "name": "double_quant",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "quant_type",
                "type": "text",
                "default": "nf4",
                "options": ["fp4", "nf4"],
                "scope": "algorithm",
            },
            {
                "name": "per_device_train_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "per_device_eval_batch_size",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "warmup_ratio",
                "type": "float",
                "default": 0.1,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_from_scratch",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "fp16",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "bf16",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "evaluation_strategy",
                "type": "text",
                "default": "steps",
                "options": ["steps", "epoch", "no"],
                "scope": "algorithm",
            },
            {
                "name": "eval_steps",
                "type": "int",
                "default": 20,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "gradient_accumulation_steps",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "logging_steps",
                "type": "int",
                "default": 8,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "weight_decay",
                "type": "float",
                "default": 0.2,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "load_best_model_at_end",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_val_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "seed",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": 2048,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_data_split_seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {"name": "max_steps", "type": "int", "default": -1, "scope": "algorithm"},
            {
                "name": "gradient_checkpointing",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_patience",
                "type": "int",
                "default": 3,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_threshold",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta1",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta2",
                "type": "float",
                "default": 0.999,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_epsilon",
                "type": "float",
                "default": 1e-08,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "max_grad_norm",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "label_smoothing_factor",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_first_step",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "logging_nan_inf_filter",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "save_strategy",
                "type": "text",
                "default": "steps",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "save_steps", "type": "int", "default": 500, "min": 1, "scope": "algorithm"},
            {"name": "save_total_limit", "type": "int", "default": 1, "scope": "algorithm"},
            {
                "name": "dataloader_drop_last",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "dataloader_num_workers",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "eval_accumulation_steps",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "auto_find_batch_size",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "lr_scheduler_type",
                "type": "text",
                "default": "constant_with_warmup",
                "options": ["constant_with_warmup", "linear"],
                "scope": "algorithm",
            },
            {"name": "warmup_steps", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "deepspeed",
                "type": "text",
                "default": "False",
                "options": ["False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/llm/v1.1.1/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/"
        "llm/prepack/v1.1.1/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.1",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-llm-gemma-7b-instruct.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "8191",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "8192",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_BATCH_PREFILL_TOKENS",
                "type": "text",
                "default": "8191",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {
                "Name": "huggingface-textgeneration:eval-loss",
                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
            },
            {"Name": "huggingface-textgeneration:train-loss", "Regex": "'loss': ([0-9]+\\.[0-9]+)"},
        ],
        "default_inference_instance_type": "ml.g5.12xlarge",
        "supported_inference_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "default_training_instance_type": "ml.g5.12xlarge",
        "supported_training_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 1200,
            "container_startup_health_check_timeout": 1200,
        },
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
            "disable_output_compression": True,
            "max_run": 360000,
        },
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 512,
        "training_volume_size": 512,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/oasst_top/train/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "hf-llm-gemma-7b-instruct",
        "gated_bucket": True,
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/h"
                    "uggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:"
                    "2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "huggingface-training/train-hugg"
                        "ingface-llm-gemma-7b-instruct.tar.gz"
                    },
                },
                "g5": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "huggingface-training/train-hugg"
                        "ingface-llm-gemma-7b-instruct.tar.gz"
                    },
                },
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "huggingface-training/train-hugg"
                        "ingface-llm-gemma-7b-instruct.tar.gz"
                    },
                },
                "p4d": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "huggingface-training/train-hugg"
                        "ingface-llm-gemma-7b-instruct.tar.gz"
                    },
                },
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
            "hosting_artifact_s3_data_type": "S3Prefix",
            "hosting_artifact_compression_type": "None",
            "hosting_resource_requirements": {"min_memory_mb": 98304, "num_accelerators": 4},
            "dynamic_container_deployment_supported": True,
        },
    },
    # noqa: E501
    "gemma-model-2b-v1_1_0": {
        "model_id": "huggingface-llm-gemma-2b-instruct",
        "url": "https://huggingface.co/google/gemma-2b-it",
        "version": "1.1.0",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "1.4.2",
            "py_version": "py310",
            "huggingface_transformers_version": "4.33.2",
        },
        "hosting_artifact_key": "huggingface-llm/huggingface-llm-gemma-2b-instruct/artifacts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/llm/v1.0.1/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": (
            "huggingface-llm/huggingface-llm-gemma-2b-instruct/artifacts/inference-prepack/v1.0.0/"
        ),
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "hosting_eula_key": "fmhMetadata/terms/gemmaTerms.txt",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "accelerate==0.26.1",
            "bitsandbytes==0.42.0",
            "deepspeed==0.10.3",
            "docstring-parser==0.15",
            "flash_attn==2.5.5",
            "ninja==1.11.1",
            "packaging==23.2",
            "peft==0.8.2",
            "py_cpuinfo==9.0.0",
            "rich==13.7.0",
            "safetensors==0.4.2",
            "sagemaker_jumpstart_huggingface_script_utilities==1.2.1",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
            "shtab==1.6.5",
            "tokenizers==0.15.1",
            "transformers==4.38.1",
            "trl==0.7.10",
            "tyro==0.7.2",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "peft_type",
                "type": "text",
                "default": "lora",
                "options": ["lora", "None"],
                "scope": "algorithm",
            },
            {
                "name": "instruction_tuned",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "chat_dataset",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "epoch",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "lora_r",
                "type": "int",
                "default": 64,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {"name": "lora_alpha", "type": "int", "default": 16, "min": 0, "scope": "algorithm"},
            {
                "name": "lora_dropout",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "bits", "type": "int", "default": 4, "scope": "algorithm"},
            {
                "name": "double_quant",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "quant_type",
                "type": "text",
                "default": "nf4",
                "options": ["fp4", "nf4"],
                "scope": "algorithm",
            },
            {
                "name": "per_device_train_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "per_device_eval_batch_size",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "warmup_ratio",
                "type": "float",
                "default": 0.1,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_from_scratch",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "fp16",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "bf16",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "evaluation_strategy",
                "type": "text",
                "default": "steps",
                "options": ["steps", "epoch", "no"],
                "scope": "algorithm",
            },
            {
                "name": "eval_steps",
                "type": "int",
                "default": 20,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "gradient_accumulation_steps",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "logging_steps",
                "type": "int",
                "default": 8,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "weight_decay",
                "type": "float",
                "default": 0.2,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "load_best_model_at_end",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_val_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "seed",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": 1024,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_data_split_seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {"name": "max_steps", "type": "int", "default": -1, "scope": "algorithm"},
            {
                "name": "gradient_checkpointing",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_patience",
                "type": "int",
                "default": 3,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_threshold",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta1",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta2",
                "type": "float",
                "default": 0.999,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_epsilon",
                "type": "float",
                "default": 1e-08,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "max_grad_norm",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "label_smoothing_factor",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_first_step",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "logging_nan_inf_filter",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "save_strategy",
                "type": "text",
                "default": "steps",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "save_steps", "type": "int", "default": 500, "min": 1, "scope": "algorithm"},
            {"name": "save_total_limit", "type": "int", "default": 1, "scope": "algorithm"},
            {
                "name": "dataloader_drop_last",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "dataloader_num_workers",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "eval_accumulation_steps",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "auto_find_batch_size",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "lr_scheduler_type",
                "type": "text",
                "default": "constant_with_warmup",
                "options": ["constant_with_warmup", "linear"],
                "scope": "algorithm",
            },
            {"name": "warmup_steps", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "deepspeed",
                "type": "text",
                "default": "False",
                "options": ["False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/llm/v1.1.1/sourcedir.tar.gz",
        "training_prepacked_script_key": (
            "source-directory-tarballs/huggingface/transfer_learning/llm/prepack/v1.1.1/sourcedir.tar.gz"
        ),
        "training_prepacked_script_version": "1.1.1",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-llm-gemma-2b-instruct.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "8191",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "8192",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_BATCH_PREFILL_TOKENS",
                "type": "text",
                "default": "8191",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SM_NUM_GPUS",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {
                "Name": "huggingface-textgeneration:eval-loss",
                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
            },
            {"Name": "huggingface-textgeneration:train-loss", "Regex": "'loss': ([0-9]+\\.[0-9]+)"},
        ],
        "default_inference_instance_type": "ml.g5.xlarge",
        "supported_inference_instance_types": [
            "ml.g5.xlarge",
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.16xlarge",
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "default_training_instance_type": "ml.g5.2xlarge",
        "supported_training_instance_types": [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.16xlarge",
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 1200,
            "container_startup_health_check_timeout": 1200,
        },
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
            "disable_output_compression": True,
            "max_run": 360000,
        },
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 512,
        "training_volume_size": 512,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/oasst_top/train/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "hf-llm-gemma-2b-instruct",
        "default_payloads": {
            "HelloWorld": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": (
                        "<bos><start_of_turn>user\nWrite a hello world program<end_of_turn>\n<start_of_turn>model"
                    ),
                    "parameters": {
                        "max_new_tokens": 256,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "MachineLearningPoem": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "Write me a poem about Machine Learning.",
                    "parameters": {
                        "max_new_tokens": 256,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
        },
        "gated_bucket": True,
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": (
                        "626614931356.dkr.ecr.af-south-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": (
                        "871362719292.dkr.ecr.ap-east-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-south-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ca-central-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": (
                        "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-central-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-north-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": (
                        "692866216735.dkr.ecr.eu-south-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-west-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-west-2.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-west-3.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": (
                        "780543022126.dkr.ecr.il-central-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": (
                        "217643126080.dkr.ecr.me-south-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.sa-east-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.us-east-2.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.us-west-1.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
                    )
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g4dn.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": (
                        "626614931356.dkr.ecr.af-south-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": (
                        "871362719292.dkr.ecr.ap-east-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-south-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.ca-central-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": (
                        "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-central-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-north-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": (
                        "692866216735.dkr.ecr.eu-south-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-west-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-west-2.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.eu-west-3.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": (
                        "780543022126.dkr.ecr.il-central-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": (
                        "217643126080.dkr.ecr.me-south-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.sa-east-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.us-east-2.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.us-west-1.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": (
                        "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    )
                },
            },
            "variants": {
                "g4dn": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": (
                            "huggingface-training/g4dn/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz"
                        )
                    },
                },
                "g5": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": (
                            "huggingface-training/g5/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz"
                        )
                    },
                },
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": (
                            "huggingface-training/p3dn/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz"
                        )
                    },
                },
                "p4d": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": (
                            "huggingface-training/p4d/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz"
                        )
                    },
                },
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "hosting_resource_requirements": {"min_memory_mb": 8192, "num_accelerators": 1},
        "dynamic_container_deployment_supported": True,
    },
    # noqa: E501
    "env-var-variant-model": {
        "model_id": "huggingface-llm-falcon-180b-bf16",
        "url": "https://huggingface.co/tiiuae/falcon-180B",
        "version": "1.6.2",
        "min_sdk_version": "2.188.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "1.4.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.33.2",
        },
        "hosting_artifact_key": "huggingface-infer/v1.2.0/infer-huggingface-llm-falcon-180b-bf16.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/llm/v1.0.1/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.2.0/infer-prepack-huggingface-llm-falcon-180b-bf16.tar.gz",
        "hosting_prepacked_artifact_version": "1.2.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SM_NUM_GPUS",
                "type": "text",
                "default": "8",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "1024",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "2048",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.p4de.24xlarge",
        "supported_inference_instance_types": ["ml.p4de.24xlarge", "ml.p5.48xlarge"],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 3600,
            "container_startup_health_check_timeout": 3600,
        },
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 512,
        "inference_enable_network_isolation": True,
        "validation_supported": False,
        "fine_tuning_supported": False,
        "resource_name_base": "hf-llm-falcon-180b-bf16",
        "default_payloads": {
            "Girafatron": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
                    "parameters": {
                        "max_new_tokens": 50,
                        "return_full_text": False,
                        "do_sample": True,
                        "top_k": 10,
                        "stop": ["Daniel:"],
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "Factorial": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "Write a program to compute factorial in python:",
                    "parameters": {
                        "max_new_tokens": 200,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "Website": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "Building a website can be done in 10 simple steps:",
                    "parameters": {
                        "max_new_tokens": 256,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "TranslateEnglishToFrench": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "Translate English to French:\n\nsea otter => loutre de mer\n\npeppermint => menthe poivr\u00e9e\n\nplush girafe => girafe peluche\n\ncheese =>",
                    "parameters": {
                        "max_new_tokens": 3,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "SentimentAnalysis": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": '"I hate it when my phone battery dies."\nSentiment: Negative\n###\nTweet: "My day has been :+1:"\nSentiment: Positive\n###\nTweet: "This is the link to the article"\nSentiment: Neutral\n###\nTweet: "This new music video was incredibile"\nSentiment:',
                    "parameters": {
                        "max_new_tokens": 2,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "QuestionAnswering": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "Could you remind me when was the C programming language invented?",
                    "parameters": {
                        "max_new_tokens": 50,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "RecipeGeneration": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "What is the recipe for a delicious lemon cheesecake?",
                    "parameters": {
                        "max_new_tokens": 256,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "Summarization": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "Starting today, the state-of-the-art Falcon 40B foundation model from Technology\nInnovation Institute (TII) is available on Amazon SageMaker JumpStart, SageMaker's machine learning (ML) hub\nthat offers pre-trained models, built-in algorithms, and pre-built solution templates to help you quickly get\nstarted with ML. You can deploy and use this Falcon LLM with a few clicks in SageMaker Studio or\nprogrammatically through the SageMaker Python SDK.\nFalcon 40B is a 40-billion-parameter large language model (LLM) available under the Apache 2.0 license that\nranked #1 in Hugging Face Open LLM leaderboard, which tracks, ranks, and evaluates LLMs across multiple\nbenchmarks to identify top performing models. Since its release in May 2023, Falcon 40B has demonstrated\nexceptional performance without specialized fine-tuning. To make it easier for customers to access this\nstate-of-the-art model, AWS has made Falcon 40B available to customers via Amazon SageMaker JumpStart.\nNow customers can quickly and easily deploy their own Falcon 40B model and customize it to fit their specific\nneeds for applications such as translation, question answering, and summarizing information.\nFalcon 40B are generally available today through Amazon SageMaker JumpStart in US East (Ohio),\nUS East (N. Virginia), US West (Oregon), Asia Pacific (Tokyo), Asia Pacific (Seoul), Asia Pacific (Mumbai),\nEurope (London), Europe (Frankfurt), Europe (Ireland), and Canada (Central),\nwith availability in additional AWS Regions coming soon. To learn how to use this new feature,\nplease see SageMaker JumpStart documentation, the Introduction to SageMaker JumpStart \u2013\nText Generation with Falcon LLMs example notebook, and the blog Technology Innovation Institute trains\nthe state-of-the-art Falcon LLM 40B foundation model on Amazon SageMaker. Summarize the article above:",
                    "parameters": {
                        "max_new_tokens": 256,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-3": {
                    "gpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-3": {
                    "gpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "cn-northwest-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-central-2": {
                    "gpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "me-central-1": {
                    "gpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-gov-east-1": {
                    "gpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-gov-west-1": {
                    "gpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g4dn.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4de.24xlarge": {
                    "properties": {
                        "environment_variables": {"SM_NUM_GPUS": "8"},
                        "resource_requirements": {"min_memory_mb": 589824, "num_accelerators": 8},
                    }
                },
                "ml.p5.48xlarge": {
                    "properties": {
                        "resource_requirements": {"min_memory_mb": 1048576, "num_accelerators": 8}
                    }
                },
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "80"}}},
                "ml.p4d.24xlarge": {
                    "properties": {
                        "environment_variables": {
                            "YODEL": "NACEREMA",
                        }
                    }
                },
            },
        },
        "hosting_resource_requirements": {"min_memory_mb": 589824, "num_accelerators": 8},
        "dynamic_container_deployment_supported": True,
        "bedrock_console_supported": True,
        "bedrock_io_mapping_id": "tgi_default_1.0.0",
    },
    "inference-instance-types-variant-model": {
        "model_id": "huggingface-llm-falcon-180b-bf16",
        "url": "https://huggingface.co/tiiuae/falcon-180B",
        "version": "1.0.0",
        "min_sdk_version": "2.175.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "0.9.3",
            "py_version": "py39",
            "huggingface_transformers_version": "4.29.2",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-llm-falcon-180b-bf16.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/llm/v1.0.1/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.0.1/infer-prepack"
        "-huggingface-llm-falcon-180b-bf16.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.1",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SM_NUM_GPUS",
                "type": "text",
                "default": "8",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "1024",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "2048",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.p4de.24xlarge",
        "supported_inference_instance_types": ["ml.p4de.24xlarge"],
        "default_training_instance_type": "ml.p4de.24xlarge",
        "supported_training_instance_types": ["ml.p4de.24xlarge"],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 3600,
            "container_startup_health_check_timeout": 3600,
        },
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 512,
        "inference_enable_network_isolation": True,
        "validation_supported": False,
        "fine_tuning_supported": False,
        "resource_name_base": "hf-llm-falcon-180b-bf16",
        "training_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                    "huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                    "gpu_image_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/stud-gpu",
                    "cpu_image_uri": "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah",
                }
            },
            "variants": {
                "ml.p2.12xlarge": {
                    "properties": {
                        "environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"},
                        "supported_inference_instance_types": ["ml.p5.xlarge"],
                        "default_inference_instance_type": "ml.p5.xlarge",
                        "metrics": [
                            {
                                "Name": "huggingface-textgeneration:eval-loss",
                                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:instance-typemetric-loss",
                                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:train-loss",
                                "Regex": "'instance type specific': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:noneyourbusiness-loss",
                                "Regex": "'loss-noyb instance specific': ([0-9]+\\.[0-9]+)",
                            },
                        ],
                    }
                },
                "p2": {
                    "regional_properties": {"image_uri": "$gpu_image_uri"},
                    "properties": {
                        "supported_inference_instance_types": ["ml.p2.xlarge", "ml.p3.xlarge"],
                        "default_inference_instance_type": "ml.p2.xlarge",
                        "metrics": [
                            {
                                "Name": "huggingface-textgeneration:wtafigo",
                                "Regex": "'evasadfasdl_loss': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:eval-loss",
                                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:train-loss",
                                "Regex": "'instance family specific': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:noneyourbusiness-loss",
                                "Regex": "'loss-noyb': ([0-9]+\\.[0-9]+)",
                            },
                        ],
                    },
                },
                "p3": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
                "ml.p3.200xlarge": {"regional_properties": {"image_uri": "$gpu_image_uri_2"}},
                "p4": {
                    "regional_properties": {"image_uri": "$gpu_image_uri"},
                    "properties": {
                        "prepacked_artifact_key": "path/to/prepacked/inference/artifact/prefix/number2/"
                    },
                },
                "g4": {
                    "regional_properties": {"image_uri": "$gpu_image_uri"},
                    "properties": {
                        "artifact_key": "path/to/prepacked/training/artifact/prefix/number2/"
                    },
                },
                "g4dn": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
                "g9": {
                    "regional_properties": {"image_uri": "$gpu_image_uri"},
                    "properties": {
                        "prepacked_artifact_key": "asfs/adsf/sda/f",
                        "hyperparameters": [
                            {
                                "name": "num_bag_sets",
                                "type": "int",
                                "default": 5,
                                "min": 5,
                                "scope": "algorithm",
                            },
                            {
                                "name": "num_stack_levels",
                                "type": "int",
                                "default": 6,
                                "min": 7,
                                "max": 3,
                                "scope": "algorithm",
                            },
                            {
                                "name": "refit_full",
                                "type": "text",
                                "default": "False",
                                "options": ["True", "False"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "set_best_to_refit_full",
                                "type": "text",
                                "default": "False",
                                "options": ["True", "False"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "save_space",
                                "type": "text",
                                "default": "False",
                                "options": ["True", "False"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "verbosity",
                                "type": "int",
                                "default": 2,
                                "min": 0,
                                "max": 4,
                                "scope": "algorithm",
                            },
                            {
                                "name": "sagemaker_submit_directory",
                                "type": "text",
                                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                                "scope": "container",
                            },
                            {
                                "name": "sagemaker_program",
                                "type": "text",
                                "default": "transfer_learning.py",
                                "scope": "container",
                            },
                            {
                                "name": "sagemaker_container_log_level",
                                "type": "text",
                                "default": "20",
                                "scope": "container",
                            },
                        ],
                    },
                },
                "p9": {
                    "regional_properties": {"image_uri": "$gpu_image_uri"},
                    "properties": {"artifact_key": "do/re/mi"},
                },
                "m2": {
                    "regional_properties": {"image_uri": "$cpu_image_uri"},
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "400"}},
                },
                "c2": {"regional_properties": {"image_uri": "$cpu_image_uri"}},
                "local": {"regional_properties": {"image_uri": "$cpu_image_uri"}},
                "ml.g5.48xlarge": {
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "8"}}
                },
                "ml.g5.12xlarge": {
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"}}
                },
                "g5": {
                    "properties": {
                        "environment_variables": {"TENSOR_PARALLEL_DEGREE": "4", "JOHN": "DOE"}
                    }
                },
                "ml.g9.12xlarge": {
                    "properties": {
                        "environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"},
                        "prepacked_artifact_key": "nlahdasf/asdf/asd/f",
                        "hyperparameters": [
                            {
                                "name": "eval_metric",
                                "type": "text",
                                "default": "auto",
                                "scope": "algorithm",
                            },
                            {
                                "name": "presets",
                                "type": "text",
                                "default": "medium_quality",
                                "options": [
                                    "best_quality",
                                    "high_quality",
                                    "good_quality",
                                    "medium_quality",
                                    "optimize_for_deployment",
                                    "interpretable",
                                ],
                                "scope": "algorithm",
                            },
                            {
                                "name": "auto_stack",
                                "type": "text",
                                "default": "False",
                                "options": ["True", "False"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "num_bag_folds",
                                "type": "text",
                                "default": "0",
                                "options": ["0", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "num_bag_sets",
                                "type": "int",
                                "default": 1,
                                "min": 1,
                                "scope": "algorithm",
                            },
                            {
                                "name": "num_stack_levels",
                                "type": "int",
                                "default": 0,
                                "min": 0,
                                "max": 3,
                                "scope": "algorithm",
                            },
                        ],
                    }
                },
                "ml.p9.12xlarge": {
                    "properties": {
                        "environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"},
                        "artifact_key": "you/not/entertained",
                    }
                },
                "g6": {
                    "properties": {
                        "environment_variables": {"BLAH": "4"},
                        "artifact_key": "path/to/training/artifact.tar.gz",
                        "prepacked_artifact_key": "path/to/prepacked/inference/artifact/prefix/",
                    }
                },
                "trn1": {
                    "properties": {
                        "supported_inference_instance_types": ["ml.inf1.xlarge", "ml.inf1.2xlarge"],
                        "default_inference_instance_type": "ml.inf1.xlarge",
                    }
                },
            },
        },
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "training_artifact_key": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": None,
        "training_model_package_artifact_uris": None,
        "deprecate_warn_message": None,
        "deprecated_message": None,
        "hosting_eula_key": None,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "adam-learning-rate",
                "type": "float",
                "default": 0.05,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch-size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_vulnerable": False,
        "deprecated": False,
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
        },
        "training_volume_size": 456,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": False,
    },
    # noqa: E501
    "variant-model": {
        "model_id": "pytorch-ic-mobilenet-v2",
        "url": "https://pytorch.org/hub/pytorch_vision_mobilenet_v2/",
        "version": "1.0.0",
        "min_sdk_version": "2.49.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_model_package_arns": {
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/ll"
            "ama2-7b-v3-740347e540da35b4ab9f6fc0ab3fed2c"
        },
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "training_instance_type_variants": {
            "regional_aliases": {},
            "variants": {
                "ml.p2.12xlarge": {
                    "properties": {
                        "environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"},
                        "hyperparameters": [
                            {
                                "name": "eval_metric",
                                "type": "text",
                                "default": "auto",
                                "scope": "algorithm",
                            },
                            {
                                "name": "presets",
                                "type": "text",
                                "default": "medium_quality",
                                "options": [
                                    "best_quality",
                                    "high_quality",
                                    "good_quality",
                                    "medium_quality",
                                    "optimize_for_deployment",
                                    "interpretable",
                                ],
                                "scope": "algorithm",
                            },
                            {
                                "name": "auto_stack",
                                "type": "text",
                                "default": "False",
                                "options": ["True", "False"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "num_bag_folds",
                                "type": "text",
                                "default": "0",
                                "options": ["0", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "num_bag_sets",
                                "type": "int",
                                "default": 1,
                                "min": 1,
                                "scope": "algorithm",
                            },
                            {
                                "name": "batch-size",
                                "type": "int",
                                "default": 1,
                                "min": 1,
                                "scope": "algorithm",
                            },
                            {
                                "name": "num_stack_levels",
                                "type": "int",
                                "default": 0,
                                "min": 0,
                                "max": 3,
                                "scope": "algorithm",
                            },
                        ],
                        "metrics": [
                            {
                                "Name": "huggingface-textgeneration:instance-typemetric-loss",
                                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:eval-loss",
                                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:train-loss",
                                "Regex": "'instance type specific': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:noneyourbusiness-loss",
                                "Regex": "'loss-noyb instance specific': ([0-9]+\\.[0-9]+)",
                            },
                        ],
                    }
                },
                "p2": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_2"},
                    "properties": {
                        "hyperparameters": [
                            {
                                "name": "num_bag_sets",
                                "type": "int",
                                "default": 5,
                                "min": 5,
                                "scope": "algorithm",
                            },
                            {
                                "name": "num_stack_levels",
                                "type": "int",
                                "default": 6,
                                "min": 7,
                                "max": 3,
                                "scope": "algorithm",
                            },
                            {
                                "name": "refit_full",
                                "type": "text",
                                "default": "False",
                                "options": ["True", "False"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "set_best_to_refit_full",
                                "type": "text",
                                "default": "False",
                                "options": ["True", "False"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "save_space",
                                "type": "text",
                                "default": "False",
                                "options": ["True", "False"],
                                "scope": "algorithm",
                            },
                            {
                                "name": "verbosity",
                                "type": "int",
                                "default": 2,
                                "min": 0,
                                "max": 4,
                                "scope": "algorithm",
                            },
                            {
                                "name": "sagemaker_submit_directory",
                                "type": "text",
                                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                                "scope": "container",
                            },
                            {
                                "name": "sagemaker_program",
                                "type": "text",
                                "default": "transfer_learning.py",
                                "scope": "container",
                            },
                            {
                                "name": "sagemaker_container_log_level",
                                "type": "text",
                                "default": "20",
                                "scope": "container",
                            },
                        ],
                        "metrics": [
                            {
                                "Name": "huggingface-textgeneration:wtafigo",
                                "Regex": "'evasadfasdl_loss': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:eval-loss",
                                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:train-loss",
                                "Regex": "'instance family specific': ([0-9]+\\.[0-9]+)",
                            },
                            {
                                "Name": "huggingface-textgeneration:noneyourbusiness-loss",
                                "Regex": "'loss-noyb': ([0-9]+\\.[0-9]+)",
                            },
                        ],
                    },
                },
            },
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                    "huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                    "cpu_image_uri": "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah",
                    "inf_model_package_arn": "us-west-2/blah/blah/blah/inf",
                    "gpu_model_package_arn": "us-west-2/blah/blah/blah/gpu",
                }
            },
            "variants": {
                "p2": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                        "model_package_arn": "$gpu_model_package_arn",
                    }
                },
                "p3": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                        "model_package_arn": "$gpu_model_package_arn",
                    }
                },
                "p4": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                        "model_package_arn": "$gpu_model_package_arn",
                    }
                },
                "g4dn": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                        "model_package_arn": "$gpu_model_package_arn",
                    }
                },
                "g5": {
                    "properties": {
                        "resource_requirements": {
                            "num_accelerators": 888810,
                            "randon-field-2": 2222,
                        }
                    }
                },
                "m2": {"regional_properties": {"image_uri": "$cpu_image_uri"}},
                "c2": {"regional_properties": {"image_uri": "$cpu_image_uri"}},
                "ml.g5.xlarge": {
                    "properties": {
                        "environment_variables": {"TENSOR_PARALLEL_DEGREE": "8"},
                        "resource_requirements": {"num_accelerators": 10},
                    }
                },
                "ml.g5.48xlarge": {
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "8"}}
                },
                "ml.g5.12xlarge": {
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"}}
                },
                "inf1": {"regional_properties": {"model_package_arn": "$inf_model_package_arn"}},
                "inf2": {"regional_properties": {"model_package_arn": "$inf_model_package_arn"}},
            },
        },
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "dynamic_container_deployment_supported": True,
        "hosting_resource_requirements": {
            "min_memory_mb": 81999,
            "num_accelerators": 1,
            "random_field_1": 1,
        },
        "hosting_artifact_key": "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
        "training_artifact_key": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
        "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz",
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": None,
        "hosting_prepacked_artifact_key": None,
        "training_model_package_artifact_uris": None,
        "deprecate_warn_message": None,
        "deprecated_message": None,
        "hosting_eula_key": None,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "adam-learning-rate",
                "type": "float",
                "default": 0.05,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch-size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "default_inference_instance_type": "ml.p2.xlarge",
        "supported_inference_instance_types": [
            "ml.p2.xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
        ],
        "default_training_instance_type": "ml.p3.2xlarge",
        "supported_training_instance_types": [
            "ml.p3.2xlarge",
            "ml.p2.xlarge",
            "ml.g4dn.2xlarge",
            "ml.m5.xlarge",
            "ml.c5.2xlarge",
        ],
        "hosting_use_script_uri": True,
        "metrics": [
            {
                "Name": "huggingface-textgeneration:train-loss",
                "Regex": "'loss default': ([0-9]+\\.[0-9]+)",
            },
            {
                "Name": "huggingface-textgeyyyuyuyuyneration:train-loss",
                "Regex": "'loss default': ([0-9]+\\.[0-9]+)",
            },
        ],
        "model_kwargs": {"some-model-kwarg-key": "some-model-kwarg-value"},
        "deploy_kwargs": {"some-model-deploy-kwarg-key": "some-model-deploy-kwarg-value"},
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
        },
        "fit_kwargs": {"some-estimator-fit-key": "some-estimator-fit-value"},
        "predictor_specs": {
            "supported_content_types": ["application/x-image"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-image",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 123,
        "training_volume_size": 456,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": False,
        "resource_name_base": "dfsdfsds",
    },
    "gated_llama_neuron_model": {
        "model_id": "meta-textgenerationneuron-llama-2-7b",
        "url": "https://ai.meta.com/resources/models-and-libraries/llama-downloads/",
        "version": "1.0.0",
        "min_sdk_version": "2.198.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "djl-neuronx",
            "framework_version": "0.24.0",
            "py_version": "py39",
        },
        "hosting_artifact_key": "meta-textgenerationneuron/meta-textgenerationneuron-llama-2-7b/artifac"
        "ts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/meta/inference/textgenerationneuron/v1.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "meta-textgenerationneuron/meta-textgenerationneuro"
        "n-llama-2-7b/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "hosting_eula_key": "fmhMetadata/eula/llamaEula.txt",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "sagemaker_jumpstart_huggingface_script_utilities==1.0.8",
            "sagemaker_jumpstart_script_utilities==1.1.8",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "sagemaker_jumpstart_huggingface_script_utilities==1.1.3",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "max_input_length",
                "type": "int",
                "default": 2048,
                "min": 128,
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 6e-06,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "min_learning_rate",
                "type": "float",
                "default": 1e-06,
                "min": 1e-12,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "max_steps", "type": "int", "default": 20, "min": 2, "scope": "algorithm"},
            {
                "name": "global_train_batch_size",
                "type": "int",
                "default": 256,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "per_device_train_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "layer_norm_epilson",
                "type": "float",
                "default": 1e-05,
                "min": 1e-12,
                "scope": "algorithm",
            },
            {
                "name": "weight_decay",
                "type": "float",
                "default": 0.1,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "lr_scheduler_type",
                "type": "text",
                "default": "CosineAnnealing",
                "options": ["CosineAnnealing"],
                "scope": "algorithm",
            },
            {"name": "warmup_steps", "type": "int", "default": 10, "min": 0, "scope": "algorithm"},
            {"name": "constant_steps", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "adam_beta1",
                "type": "float",
                "default": 0.9,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta2",
                "type": "float",
                "default": 0.95,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "mixed_precision",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "tensor_parallel_degree",
                "type": "text",
                "default": "8",
                "options": ["8"],
                "scope": "algorithm",
            },
            {
                "name": "pipeline_parallel_degree",
                "type": "text",
                "default": "1",
                "options": ["1"],
                "scope": "algorithm",
            },
            {
                "name": "append_eod",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/meta/transfer_learning/textgenerati"
        "onneuron/v1.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/meta/tra"
        "nsfer_learning/textgenerationneuron/prepack/v1.0.0/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.0.0",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "meta-training/train-meta-textgenerationneuron-llama-2-7b.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {
                "Name": "meta-textgenerationneuron:train-loss",
                "Regex": "reduced_train_loss=([0-9]+\\.[0-9]+)",
            }
        ],
        "default_inference_instance_type": "ml.inf2.xlarge",
        "supported_inference_instance_types": [
            "ml.inf2.xlarge",
            "ml.inf2.8xlarge",
            "ml.inf2.24xlarge",
            "ml.inf2.48xlarge",
        ],
        "default_training_instance_type": "ml.trn1.32xlarge",
        "supported_training_instance_types": ["ml.trn1.32xlarge", "ml.trn1n.32xlarge"],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 3600,
            "container_startup_health_check_timeout": 3600,
        },
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
            "disable_output_compression": True,
            "max_run": 360000,
        },
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 256,
        "training_volume_size": 256,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/sec_amazon/",
        "validation_supported": False,
        "fine_tuning_supported": True,
        "resource_name_base": "meta-textgenerationneuron-llama-2-7b",
        "default_payloads": {
            "meaningOfLife": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {"generated_text": "generated_text"},
                "body": {
                    "inputs": "I believe the meaning of life is",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
            "theoryOfRelativity": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {"generated_text": "generated_text"},
                "body": {
                    "inputs": "Simply put, the theory of relativity states that ",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
            "teamMessage": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {"generated_text": "generated_text"},
                "body": {
                    "inputs": "A brief message congratulating the team on the launch:\n\nHi "
                    "everyone,\n\nI just ",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
            "englishToFrench": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {"generated_text": "generated_text"},
                "body": {
                    "inputs": "Translate English to French:\nsea otter => loutre de mer\npep"
                    "permint => menthe poivr\u00e9e\nplush girafe => girafe peluche\ncheese =>",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
        },
        "gated_bucket": True,
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "alias_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/djl-in"
                    "ference:0.24.0-neuronx-sdk2.14.1"
                },
                "ap-east-1": {
                    "alias_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/djl-in"
                    "ference:0.24.0-neuronx-sdk2.14.1"
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/d"
                    "jl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "ap-northeast-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com"
                    "/djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "ap-south-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/"
                    "djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com"
                    "/djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com"
                    "/djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "ca-central-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/"
                    "djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "cn-north-1": {
                    "alias_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/"
                    "djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "eu-central-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/"
                    "djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "eu-north-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/"
                    "djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "eu-south-1": {
                    "alias_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/"
                    "djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "eu-west-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/d"
                    "jl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "eu-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/d"
                    "jl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "eu-west-3": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/d"
                    "jl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "me-south-1": {
                    "alias_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com"
                    "/djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "sa-east-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com"
                    "/djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "us-east-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
                    "djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "us-east-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com"
                    "/djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "us-west-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.co"
                    "m/djl-inference:0.24.0-neuronx-sdk2.14.1"
                },
                "us-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-"
                    "inference:0.24.0-neuronx-sdk2.14.1"
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf1": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "ml.inf2.xlarge": {
                    "properties": {
                        "environment_variables": {
                            "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                            "OPTION_N_POSITIONS": "1024",
                            "OPTION_DTYPE": "fp16",
                            "OPTION_ROLLING_BATCH": "auto",
                            "OPTION_MAX_ROLLING_BATCH_SIZE": "1",
                            "OPTION_NEURON_OPTIMIZE_LEVEL": "2",
                        }
                    }
                },
                "ml.inf2.8xlarge": {
                    "properties": {
                        "environment_variables": {
                            "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                            "OPTION_N_POSITIONS": "2048",
                            "OPTION_DTYPE": "fp16",
                            "OPTION_ROLLING_BATCH": "auto",
                            "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
                            "OPTION_NEURON_OPTIMIZE_LEVEL": "2",
                        }
                    }
                },
                "ml.inf2.24xlarge": {
                    "properties": {
                        "environment_variables": {
                            "OPTION_TENSOR_PARALLEL_DEGREE": "12",
                            "OPTION_N_POSITIONS": "4096",
                            "OPTION_DTYPE": "fp16",
                            "OPTION_ROLLING_BATCH": "auto",
                            "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
                            "OPTION_NEURON_OPTIMIZE_LEVEL": "2",
                        }
                    }
                },
                "ml.inf2.48xlarge": {
                    "properties": {
                        "environment_variables": {
                            "OPTION_TENSOR_PARALLEL_DEGREE": "24",
                            "OPTION_N_POSITIONS": "4096",
                            "OPTION_DTYPE": "fp16",
                            "OPTION_ROLLING_BATCH": "auto",
                            "OPTION_MAX_ROLLING_BATCH_SIZE": "4",
                            "OPTION_NEURON_OPTIMIZE_LEVEL": "2",
                        }
                    }
                },
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorc"
                    "h-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch"
                    "-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-"
                    "pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingfa"
                    "ce-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch"
                    "-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/"
                    "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/"
                    "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/hu"
                    "ggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/"
                    "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/hug"
                    "gingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/hu"
                    "ggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/hu"
                    "ggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/hug"
                    "gingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/hug"
                    "gingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggi"
                    "ngface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggin"
                    "gface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/hugg"
                    "ingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/hu"
                    "ggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
                    "neuron_ecr_uri": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-"
                    "training-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04",
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-py"
                    "torch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
                    "neuron_ecr_uri": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-trai"
                    "ning-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04",
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytor"
                    "ch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface"
                    "-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",
                    "neuron_ecr_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch"
                    "-training-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04",
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "trn1": {
                    "regional_properties": {"image_uri": "$neuron_ecr_uri"},
                    "properties": {
                        "gated_model_key_env_var_value": "meta-training/trn1/v1.0."
                        "0/train-meta-textgenerationneuron-llama-2-7b.tar.gz"
                    },
                },
                "trn1n": {
                    "regional_properties": {"image_uri": "$neuron_ecr_uri"},
                    "properties": {
                        "gated_model_key_env_var_value": "meta-training/trn1n/v1.0.0"
                        "/train-meta-textgenerationneuron-llama-2-7b.tar.gz"
                    },
                },
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "hosting_resource_requirements": {"min_memory_mb": 8192, "num_accelerators": 1},
        "dynamic_container_deployment_supported": True,
    },
    "gated_variant-model": {
        "model_id": "pytorch-ic-mobilenet-v2",
        "gated_bucket": True,
        "url": "https://pytorch.org/hub/pytorch_vision_mobilenet_v2/",
        "version": "1.0.0",
        "min_sdk_version": "2.49.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "training_instance_type_variants": None,
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                    "huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                    "cpu_image_uri": "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah",
                }
            },
            "variants": {
                "p2": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                    },
                    "properties": {
                        "prepacked_artifact_key": "some-instance-specific/model/prefix/"
                    },
                },
                "p3": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                    }
                },
                "p4": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                    }
                },
                "g4dn": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                    }
                },
                "m2": {"regional_properties": {"image_uri": "$cpu_image_uri"}},
                "c2": {"regional_properties": {"image_uri": "$cpu_image_uri"}},
                "ml.g5.48xlarge": {
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "8"}}
                },
                "ml.g5.12xlarge": {
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"}}
                },
            },
        },
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "hosting_artifact_key": "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
        "training_artifact_key": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
        "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz",
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": None,
        "hosting_prepacked_artifact_key": None,
        "training_model_package_artifact_uris": None,
        "deprecate_warn_message": None,
        "deprecated_message": None,
        "hosting_eula_key": None,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "adam-learning-rate",
                "type": "float",
                "default": 0.05,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch-size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "default_inference_instance_type": "ml.p2.xlarge",
        "supported_inference_instance_types": [
            "ml.p2.xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
        ],
        "default_training_instance_type": "ml.p3.2xlarge",
        "supported_training_instance_types": [
            "ml.p3.2xlarge",
            "ml.p2.xlarge",
            "ml.g4dn.2xlarge",
            "ml.m5.xlarge",
            "ml.c5.2xlarge",
        ],
        "hosting_use_script_uri": False,
        "metrics": [
            {
                "Name": "huggingface-textgeneration:train-loss",
                "Regex": "'loss default': ([0-9]+\\.[0-9]+)",
            },
            {
                "Name": "huggingface-textgeyyyuyuyuyneration:train-loss",
                "Regex": "'loss default': ([0-9]+\\.[0-9]+)",
            },
        ],
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
        },
        "fit_kwargs": {"some-estimator-fit-key": "some-estimator-fit-value"},
        "predictor_specs": {
            "supported_content_types": ["application/x-image"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-image",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 123,
        "training_volume_size": 456,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": False,
        "resource_name_base": "dfsdfsds",
    },
    "model-artifact-variant-model": {
        "model_id": "pytorch-ic-mobilenet-v2",
        "url": "https://pytorch.org/hub/pytorch_vision_mobilenet_v2/",
        "version": "3.0.6",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        },
        "hosting_artifact_key": "pytorch-ic/pytorch-ic-mobilenet-v2/artifacts/inference/v2.0.0/",
        "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v2.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "pytorch-ic/pytorch-ic-mobilenet-v2/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": ["sagemaker_jumpstart_prepack_script_utilities==1.0.0"],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "train_only_top_layer",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "epochs",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch_size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "reinitialize_top_layer",
                "type": "text",
                "default": "Auto",
                "options": ["Auto", "True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v2.3.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/prepack/v1.1.0/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.0",
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        },
        "training_artifact_key": "pytorch-training/v2.0.0/train-pytorch-ic-mobilenet-v2.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [{"Name": "pytorch-ic:val-accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"}],
        "default_inference_instance_type": "ml.m5.large",
        "supported_inference_instance_types": [
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.m4.large",
            "ml.m4.xlarge",
        ],
        "default_training_instance_type": "ml.m5.xlarge",
        "supported_training_instance_types": ["ml.m5.xlarge", "ml.c5.2xlarge", "ml.m4.xlarge"],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/x-image"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-image",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/tf_flowers/",
        "validation_supported": False,
        "fine_tuning_supported": True,
        "resource_name_base": "pt-ic-mobilenet-v2",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-south-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:1.10.0-gpu-py38",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-central-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-west-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "sa-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-east-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-west-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_2"},
                    "properties": {"prepacked_artifact_key": "hello-world-1"},
                },
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "trn1": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
                "trn1n": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-south-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.10.0-gpu-py38",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-central-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-west-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "sa-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-east-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-west-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {
                    "regional_properties": {"image_uri": "$cpu_ecr_uri_1"},
                    "properties": {"artifact_key": "hello-world-1"},
                },
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "trn1": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
                "trn1n": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "private-model": {
        "model_id": "pytorch-ic-mobilenet-v2",
        "gated_bucket": True,
        "url": "https://pytorch.org/hub/pytorch_vision_mobilenet_v2/",
        "version": "1.0.0",
        "min_sdk_version": "2.49.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_model_package_arns": {
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/ll"
            "ama2-7b-v3-740347e540da35b4ab9f6fc0ab3fed2c"
        },
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                    "huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                    "cpu_image_uri": "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah",
                    "inf_model_package_arn": "us-west-2/blah/blah/blah/inf",
                    "gpu_model_package_arn": "us-west-2/blah/blah/blah/gpu",
                }
            },
            "variants": {
                "p2": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                        "model_package_arn": "$gpu_model_package_arn",
                    }
                },
                "p3": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                        "model_package_arn": "$gpu_model_package_arn",
                    }
                },
                "p4": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                        "model_package_arn": "$gpu_model_package_arn",
                    }
                },
                "g4dn": {
                    "regional_properties": {
                        "image_uri": "$gpu_image_uri",
                        "model_package_arn": "$gpu_model_package_arn",
                    }
                },
                "m2": {"regional_properties": {"image_uri": "$cpu_image_uri"}},
                "c2": {"regional_properties": {"image_uri": "$cpu_image_uri"}},
                "ml.g5.48xlarge": {
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "8"}}
                },
                "ml.g5.12xlarge": {
                    "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"}}
                },
                "inf1": {"regional_properties": {"model_package_arn": "$inf_model_package_arn"}},
                "inf2": {"regional_properties": {"model_package_arn": "$inf_model_package_arn"}},
            },
        },
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "training_instance_type_variants": None,
        "hosting_artifact_key": "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
        "training_artifact_key": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
        "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz",
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": None,
        "hosting_prepacked_artifact_key": None,
        "training_model_package_artifact_uris": None,
        "deprecate_warn_message": None,
        "deprecated_message": None,
        "hosting_eula_key": None,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "adam-learning-rate",
                "type": "float",
                "default": 0.05,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch-size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "default_inference_instance_type": "ml.p2.xlarge",
        "supported_inference_instance_types": [
            "ml.p2.xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
        ],
        "default_training_instance_type": "ml.p3.2xlarge",
        "supported_training_instance_types": [
            "ml.p3.2xlarge",
            "ml.p2.xlarge",
            "ml.g4dn.2xlarge",
            "ml.m5.xlarge",
            "ml.c5.2xlarge",
        ],
        "hosting_use_script_uri": True,
        "metrics": [{"Regex": "val_accuracy: ([0-9\\.]+)", "Name": "pytorch-ic:val-accuracy"}],
        "model_kwargs": {"some-model-kwarg-key": "some-model-kwarg-value"},
        "deploy_kwargs": {"some-model-deploy-kwarg-key": "some-model-deploy-kwarg-value"},
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
        },
        "fit_kwargs": {"some-estimator-fit-key": "some-estimator-fit-value"},
        "predictor_specs": {
            "supported_content_types": ["application/x-image"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-image",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 123,
        "training_volume_size": 456,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": False,
        "resource_name_base": "dfsdfsds",
    },
    "js-model-package-arn": {
        "model_id": "meta-textgeneration-llama-2-7b-f",
        "url": "https://ai.meta.com/resources/models-and-libraries/llama-downloads/",
        "version": "2.0.4",
        "min_sdk_version": "2.174.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "djl-deepspeed",
            "framework_version": "0.23.0",
            "py_version": "py39",
        },
        "hosting_artifact_key": "meta-infer/infer-meta-textgeneration-llama-2-7b-f.tar.gz",
        "hosting_script_key": "source-directory-tarballs/meta/inference/textgeneration/v1.2.2/sourcedir.tar.gz",
        "hosting_use_script_uri": False,
        "hosting_eula_key": "fmhMetadata/eula/llamaEula.txt",
        "hosting_model_package_arns": {
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
        },
        "training_model_package_artifact_uris": {
            "us-west-2": "s3://sagemaker-repository-pdx/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "us-east-2": "s3://sagemaker-repository-cmh/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "us-east-1": "s3://sagemaker-repository-iad/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "eu-west-1": "s3://sagemaker-repository-dub/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "ap-southeast-1": "s3://sagemaker-repository-sin/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "ap-southeast-2": "s3://sagemaker-repository-syd/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
        },
        "inference_vulnerable": False,
        "inference_dependencies": [
            "sagemaker_jumpstart_huggingface_script_utilities==1.0.8",
            "sagemaker_jumpstart_script_utilities==1.1.8",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": True,
        "training_dependencies": [
            "accelerate==0.21.0",
            "bitsandbytes==0.39.1",
            "black==23.7.0",
            "brotli==1.0.9",
            "datasets==2.14.1",
            "fire==0.5.0",
            "inflate64==0.3.1",
            "loralib==0.1.1",
            "multivolumefile==0.2.3",
            "mypy-extensions==1.0.0",
            "pathspec==0.11.1",
            "peft==0.4.0",
            "py7zr==0.20.5",
            "pybcj==1.0.1",
            "pycryptodomex==3.18.0",
            "pyppmd==1.0.0",
            "pytorch-triton==2.1.0+6e4932cda8",
            "pyzstd==0.15.9",
            "safetensors==0.3.1",
            "sagemaker_jumpstart_huggingface_script_utilities==1.1.3",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "scipy==1.11.1",
            "termcolor==2.3.0",
            "texttable==1.6.7",
            "tokenize-rt==5.1.0",
            "tokenizers==0.13.3",
            "torch==2.2.0.dev20231104+cu118",
            "transformers==4.31.0",
        ],
        "training_vulnerabilities": ["transformers==4.31.0"],
        "deprecated": False,
        "deprecate_warn_message": "For forward compatibility, pin to model_version='2.*' in your JumpStartModel or JumpStartEstimator definitions. Note that major version upgrades may have different EULA acceptance terms and input/output signatures.",
        "hyperparameters": [
            {
                "name": "int8_quantization",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "enable_fsdp",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "epoch",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "lora_r", "type": "int", "default": 8, "min": 1, "scope": "algorithm"},
            {"name": "lora_alpha", "type": "int", "default": 32, "min": 1, "scope": "algorithm"},
            {
                "name": "lora_dropout",
                "type": "float",
                "default": 0.05,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "instruction_tuned",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "chat_dataset",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "add_input_output_demarcation_key",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "per_device_train_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "per_device_eval_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_val_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "seed",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_data_split_seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/meta/transfer_learning/textgeneration/v1.0.6/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "meta-training/train-meta-textgeneration-llama-2-7b-f.tar.gz",
        "inference_environment_variables": [],
        "metrics": [
            {
                "Name": "huggingface-textgeneration:eval-loss",
                "Regex": "eval_epoch_loss=tensor\\(([0-9\\.]+)",
            },
            {
                "Name": "huggingface-textgeneration:eval-ppl",
                "Regex": "eval_ppl=tensor\\(([0-9\\.]+)",
            },
            {
                "Name": "huggingface-textgeneration:train-loss",
                "Regex": "train_epoch_loss=([0-9\\.]+)",
            },
        ],
        "default_inference_instance_type": "ml.g5.2xlarge",
        "supported_inference_instance_types": [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "default_training_instance_type": "ml.g5.12xlarge",
        "supported_training_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p3dn.24xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 3600,
            "container_startup_health_check_timeout": 3600,
        },
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 256,
        "training_volume_size": 256,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/oasst_top/train/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "meta-textgeneration-llama-2-7b-f",
        "default_payloads": {
            "Mayo": {
                "content_type": "application/json",
                "body": {
                    "inputs": [[{"role": "user", "content": "what is the recipe of mayonnaise?"}]],
                    "parameters": {"max_new_tokens": 512, "top_p": 0.9, "temperature": 0.6},
                },
            }
        },
        "gated_bucket": True,
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "alias_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-east-1": {
                    "alias_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-northeast-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-south-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ca-central-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "cn-north-1": {
                    "alias_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-central-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-north-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-south-1": {
                    "alias_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-west-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-west-3": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "me-south-1": {
                    "alias_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "sa-east-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "us-east-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "us-east-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "us-west-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "us-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf1": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "dynamic_container_deployment_supported": False,
    },
    "js-trainable-model-prepacked": {
        "model_id": "huggingface-text2text-flan-t5-base",
        "url": "https://huggingface.co/google/flan-t5-base",
        "version": "2.2.3",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "1.4.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.33.2",
        },
        "hosting_artifact_key": "huggingface-text2text/huggingface-text2text-flan-t5-base/artifacts/inference/v2.0.0/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v2.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-text2text/huggingface-text2text-flan-t5-base/artifacts/inference-prepack/v2.0.0/",
        "hosting_prepacked_artifact_version": "2.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "accelerate==0.23.0",
            "datasets==2.12.0",
            "deepspeed==0.10.3",
            "peft==0.5.0",
            "safetensors==0.3.3",
            "sagemaker_jumpstart_huggingface_script_utilities==1.1.2",
            "sagemaker_jumpstart_script_utilities==1.1.8",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {"name": "max_steps", "type": "int", "default": -1, "scope": "algorithm"},
            {
                "name": "seed",
                "type": "int",
                "default": 42,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "batch_size",
                "type": "int",
                "default": 64,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "lr_scheduler_type",
                "type": "text",
                "default": "constant_with_warmup",
                "options": ["constant_with_warmup", "linear"],
                "scope": "algorithm",
            },
            {
                "name": "warmup_ratio",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "warmup_steps", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.05,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "train_data_split_seed", "type": "int", "default": 0, "scope": "algorithm"},
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_eval_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_output_length",
                "type": "int",
                "default": 128,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "pad_to_max_length",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "gradient_accumulation_steps",
                "type": "int",
                "default": 1,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "weight_decay",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta1",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta2",
                "type": "float",
                "default": 0.999,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_epsilon",
                "type": "float",
                "default": 1e-08,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "max_grad_norm",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "load_best_model_at_end",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_patience",
                "type": "int",
                "default": 3,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_threshold",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "label_smoothing_factor",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_strategy",
                "type": "text",
                "default": "steps",
                "options": ["no", "steps", "epoch"],
                "scope": "algorithm",
            },
            {
                "name": "logging_first_step",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "logging_steps",
                "type": "int",
                "default": 500,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_nan_inf_filter",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "save_strategy",
                "type": "text",
                "default": "epoch",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "save_steps", "type": "int", "default": 500, "min": 1, "scope": "algorithm"},
            {"name": "save_total_limit", "type": "text", "default": "2", "scope": "algorithm"},
            {
                "name": "dataloader_drop_last",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "dataloader_num_workers",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "evaluation_strategy",
                "type": "text",
                "default": "epoch",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "eval_steps", "type": "text", "default": "500", "scope": "algorithm"},
            {
                "name": "eval_accumulation_steps",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "gradient_checkpointing",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "auto_find_batch_size",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "peft_type",
                "type": "text",
                "default": "none",
                "options": ["lora", "none"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/v2.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/prepack/v2.0.0/sourcedir.tar.gz",
        "training_prepacked_script_version": "2.0.0",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-text2text-flan-t5-base.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SM_NUM_GPUS",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "1024",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "2048",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {"Name": "huggingface-text2text:eval-loss", "Regex": "'eval_loss': ([0-9\\.]+)"}
        ],
        "default_inference_instance_type": "ml.g5.2xlarge",
        "supported_inference_instance_types": [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.16xlarge",
        ],
        "default_training_instance_type": "ml.p3.16xlarge",
        "supported_training_instance_types": [
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.p3dn.24xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 1200,
            "container_startup_health_check_timeout": 1200,
        },
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
            "disable_output_compression": True,
            "max_run": 360000,
        },
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 512,
        "training_volume_size": 512,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/genuq/dev/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "hf-text2text-flan-t5-base",
        "default_payloads": {
            "Summarization": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "body": {
                    "inputs": "Summarize this content - Amazon Comprehend uses natural language processing (NLP) to extract insights about the content of documents. It develops insights by recognizing the entities, key phrases, language, sentiments, and other common elements in a document. Use Amazon Comprehend to create new products based on understanding the structure of documents. For example, using Amazon Comprehend you can search social networking feeds for mentions of products or scan an entire document repository for key phrases. You can access Amazon Comprehend document analysis capabilities using the Amazon Comprehend console or using the Amazon Co",
                    "parameters": {
                        "max_new_tokens": 400,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
            }
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-3": {
                    "gpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-3": {
                    "gpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "cn-northwest-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-central-2": {
                    "gpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "me-central-1": {
                    "gpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-gov-east-1": {
                    "gpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-gov-west-1": {
                    "gpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-3": {
                    "gpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-3": {
                    "gpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-northwest-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-central-1": {
                    "gpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-gov-east-1": {
                    "gpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-gov-west-1": {
                    "gpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
        "bedrock_console_supported": True,
        "bedrock_io_mapping_id": "tgi_default_1.0.0",
    },
    "deprecated_model": {
        "model_id": "huggingface-text2text-flan-t5-base",
        "url": "https://huggingface.co/google/flan-t5-base",
        "version": "2.2.3",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "1.4.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.33.2",
        },
        "hosting_artifact_key": "huggingface-text2text/huggingface-text2text-flan-t5-base/artifacts/inference/v2.0.0/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v2.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-text2text/huggingface-text2text-flan-t5-base/artifacts/inference-prepack/v2.0.0/",
        "hosting_prepacked_artifact_version": "2.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "accelerate==0.23.0",
            "datasets==2.12.0",
            "deepspeed==0.10.3",
            "peft==0.5.0",
            "safetensors==0.3.3",
            "sagemaker_jumpstart_huggingface_script_utilities==1.1.2",
            "sagemaker_jumpstart_script_utilities==1.1.8",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": True,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {"name": "max_steps", "type": "int", "default": -1, "scope": "algorithm"},
            {
                "name": "seed",
                "type": "int",
                "default": 42,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "batch_size",
                "type": "int",
                "default": 64,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "lr_scheduler_type",
                "type": "text",
                "default": "constant_with_warmup",
                "options": ["constant_with_warmup", "linear"],
                "scope": "algorithm",
            },
            {
                "name": "warmup_ratio",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "warmup_steps", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.05,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "train_data_split_seed", "type": "int", "default": 0, "scope": "algorithm"},
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_eval_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_output_length",
                "type": "int",
                "default": 128,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "pad_to_max_length",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "gradient_accumulation_steps",
                "type": "int",
                "default": 1,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "weight_decay",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta1",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta2",
                "type": "float",
                "default": 0.999,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_epsilon",
                "type": "float",
                "default": 1e-08,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "max_grad_norm",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "load_best_model_at_end",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_patience",
                "type": "int",
                "default": 3,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_threshold",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "label_smoothing_factor",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_strategy",
                "type": "text",
                "default": "steps",
                "options": ["no", "steps", "epoch"],
                "scope": "algorithm",
            },
            {
                "name": "logging_first_step",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "logging_steps",
                "type": "int",
                "default": 500,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_nan_inf_filter",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "save_strategy",
                "type": "text",
                "default": "epoch",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "save_steps", "type": "int", "default": 500, "min": 1, "scope": "algorithm"},
            {"name": "save_total_limit", "type": "text", "default": "2", "scope": "algorithm"},
            {
                "name": "dataloader_drop_last",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "dataloader_num_workers",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "evaluation_strategy",
                "type": "text",
                "default": "epoch",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "eval_steps", "type": "text", "default": "500", "scope": "algorithm"},
            {
                "name": "eval_accumulation_steps",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "gradient_checkpointing",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "auto_find_batch_size",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "peft_type",
                "type": "text",
                "default": "none",
                "options": ["lora", "none"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/v2.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/prepack/v2.0.0/sourcedir.tar.gz",
        "training_prepacked_script_version": "2.0.0",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-text2text-flan-t5-base.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SM_NUM_GPUS",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "1024",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "2048",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {"Name": "huggingface-text2text:eval-loss", "Regex": "'eval_loss': ([0-9\\.]+)"}
        ],
        "default_inference_instance_type": "ml.g5.2xlarge",
        "supported_inference_instance_types": [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.16xlarge",
        ],
        "default_training_instance_type": "ml.p3.16xlarge",
        "supported_training_instance_types": [
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.p3dn.24xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 1200,
            "container_startup_health_check_timeout": 1200,
        },
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
            "disable_output_compression": True,
            "max_run": 360000,
        },
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 512,
        "training_volume_size": 512,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/genuq/dev/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "hf-text2text-flan-t5-base",
        "default_payloads": {
            "Summarization": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "body": {
                    "inputs": "Summarize this content - Amazon Comprehend uses natural language processing (NLP) to extract insights about the content of documents. It develops insights by recognizing the entities, key phrases, language, sentiments, and other common elements in a document. Use Amazon Comprehend to create new products based on understanding the structure of documents. For example, using Amazon Comprehend you can search social networking feeds for mentions of products or scan an entire document repository for key phrases. You can access Amazon Comprehend document analysis capabilities using the Amazon Comprehend console or using the Amazon Co",
                    "parameters": {
                        "max_new_tokens": 400,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
            }
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-3": {
                    "gpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-3": {
                    "gpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "cn-northwest-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-central-2": {
                    "gpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "me-central-1": {
                    "gpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-gov-east-1": {
                    "gpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-gov-west-1": {
                    "gpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-3": {
                    "gpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-3": {
                    "gpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-northwest-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-central-1": {
                    "gpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-gov-east-1": {
                    "gpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-gov-west-1": {
                    "gpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
        "bedrock_console_supported": True,
        "bedrock_io_mapping_id": "tgi_default_1.0.0",
    },
    "vulnerable_model": {
        "model_id": "huggingface-text2text-flan-t5-base",
        "url": "https://huggingface.co/google/flan-t5-base",
        "version": "2.2.3",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "1.4.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.33.2",
        },
        "hosting_artifact_key": "huggingface-text2text/huggingface-text2text-flan-t5-base/artifacts/inference/v2.0.0/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v2.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-text2text/huggingface-text2text-flan-t5-base/artifacts/inference-prepack/v2.0.0/",
        "hosting_prepacked_artifact_version": "2.0.0",
        "hosting_use_script_uri": False,
        "inference_dependencies": [],
        "training_vulnerable": True,
        "training_dependencies": [
            "accelerate==0.23.0",
            "datasets==2.12.0",
            "deepspeed==0.10.3",
            "peft==0.5.0",
            "safetensors==0.3.3",
            "sagemaker_jumpstart_huggingface_script_utilities==1.1.2",
            "sagemaker_jumpstart_script_utilities==1.1.8",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
        ],
        "inference_vulnerable": True,
        "training_vulnerabilities": ["accelerate==0.23.0"],
        "training_vulnerabilities": ["accelerate==0.23.0"],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {"name": "max_steps", "type": "int", "default": -1, "scope": "algorithm"},
            {
                "name": "seed",
                "type": "int",
                "default": 42,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "batch_size",
                "type": "int",
                "default": 64,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "lr_scheduler_type",
                "type": "text",
                "default": "constant_with_warmup",
                "options": ["constant_with_warmup", "linear"],
                "scope": "algorithm",
            },
            {
                "name": "warmup_ratio",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "warmup_steps", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.05,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "train_data_split_seed", "type": "int", "default": 0, "scope": "algorithm"},
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_eval_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_output_length",
                "type": "int",
                "default": 128,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "pad_to_max_length",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "gradient_accumulation_steps",
                "type": "int",
                "default": 1,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "weight_decay",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta1",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta2",
                "type": "float",
                "default": 0.999,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_epsilon",
                "type": "float",
                "default": 1e-08,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "max_grad_norm",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "load_best_model_at_end",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_patience",
                "type": "int",
                "default": 3,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_threshold",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "label_smoothing_factor",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_strategy",
                "type": "text",
                "default": "steps",
                "options": ["no", "steps", "epoch"],
                "scope": "algorithm",
            },
            {
                "name": "logging_first_step",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "logging_steps",
                "type": "int",
                "default": 500,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_nan_inf_filter",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "save_strategy",
                "type": "text",
                "default": "epoch",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "save_steps", "type": "int", "default": 500, "min": 1, "scope": "algorithm"},
            {"name": "save_total_limit", "type": "text", "default": "2", "scope": "algorithm"},
            {
                "name": "dataloader_drop_last",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "dataloader_num_workers",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "evaluation_strategy",
                "type": "text",
                "default": "epoch",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "eval_steps", "type": "text", "default": "500", "scope": "algorithm"},
            {
                "name": "eval_accumulation_steps",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "gradient_checkpointing",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "auto_find_batch_size",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "peft_type",
                "type": "text",
                "default": "none",
                "options": ["lora", "none"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/v2.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/prepack/v2.0.0/sourcedir.tar.gz",
        "training_prepacked_script_version": "2.0.0",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-text2text-flan-t5-base.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SM_NUM_GPUS",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "1024",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "2048",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {"Name": "huggingface-text2text:eval-loss", "Regex": "'eval_loss': ([0-9\\.]+)"}
        ],
        "default_inference_instance_type": "ml.g5.2xlarge",
        "supported_inference_instance_types": [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.16xlarge",
        ],
        "default_training_instance_type": "ml.p3.16xlarge",
        "supported_training_instance_types": [
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.p3dn.24xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 1200,
            "container_startup_health_check_timeout": 1200,
        },
        "estimator_kwargs": {
            "encrypt_inter_container_traffic": True,
            "disable_output_compression": True,
            "max_run": 360000,
        },
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 512,
        "training_volume_size": 512,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/genuq/dev/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "hf-text2text-flan-t5-base",
        "default_payloads": {
            "Summarization": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "body": {
                    "inputs": "Summarize this content - Amazon Comprehend uses natural language processing (NLP) to extract insights about the content of documents. It develops insights by recognizing the entities, key phrases, language, sentiments, and other common elements in a document. Use Amazon Comprehend to create new products based on understanding the structure of documents. For example, using Amazon Comprehend you can search social networking feeds for mentions of products or scan an entire document repository for key phrases. You can access Amazon Comprehend document analysis capabilities using the Amazon Comprehend console or using the Amazon Co",
                    "parameters": {
                        "max_new_tokens": 400,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
            }
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-northeast-3": {
                    "gpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ap-southeast-3": {
                    "gpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "cn-northwest-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-central-2": {
                    "gpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "me-central-1": {
                    "gpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-gov-east-1": {
                    "gpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-gov-west-1": {
                    "gpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-3": {
                    "gpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-3": {
                    "gpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-northwest-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-central-1": {
                    "gpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-gov-east-1": {
                    "gpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-gov-west-1": {
                    "gpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
        "bedrock_console_supported": True,
        "bedrock_io_mapping_id": "tgi_default_1.0.0",
    },
    "js-gated-artifact-non-model-package-trainable-model": {
        "model_id": "meta-textgeneration-llama-2-7b",
        "url": "https://ai.meta.com/resources/models-and-libraries/llama-downloads/",
        "version": "3.0.0",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface-llm",
            "framework_version": "1.1.0",
            "py_version": "py39",
        },
        "training_artifact_key": "some/dummy/key",
        "hosting_artifact_key": "meta-textgeneration/meta-textgeneration-llama-2-7b/artifacts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/meta/inference/textgeneration/v1.2.3/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "meta-textgeneration/meta-textgen"
        "eration-llama-2-7b/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "hosting_eula_key": "fmhMetadata/eula/llamaEula.txt",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "sagemaker_jumpstart_huggingface_script_utilities==1.0.8",
            "sagemaker_jumpstart_script_utilities==1.1.8",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "accelerate==0.21.0",
            "bitsandbytes==0.39.1",
            "black==23.7.0",
            "brotli==1.0.9",
            "datasets==2.14.1",
            "fire==0.5.0",
            "inflate64==0.3.1",
            "loralib==0.1.1",
            "multivolumefile==0.2.3",
            "mypy-extensions==1.0.0",
            "pathspec==0.11.1",
            "peft==0.4.0",
            "py7zr==0.20.5",
            "pybcj==1.0.1",
            "pycryptodomex==3.18.0",
            "pyppmd==1.0.0",
            "pytorch-triton==2.1.0+e6216047b8",
            "pyzstd==0.15.9",
            "safetensors==0.3.1",
            "sagemaker_jumpstart_huggingface_script_utilities==1.1.3",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "scipy==1.11.1",
            "termcolor==2.3.0",
            "texttable==1.6.7",
            "tokenize-rt==5.1.0",
            "tokenizers==0.13.3",
            "torch==2.1.0.dev20230905+cu118",
            "transformers==4.31.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "int8_quantization",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "enable_fsdp",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "epoch",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "lora_r", "type": "int", "default": 8, "min": 1, "scope": "algorithm"},
            {"name": "lora_alpha", "type": "int", "default": 32, "min": 1, "scope": "algorithm"},
            {
                "name": "lora_dropout",
                "type": "float",
                "default": 0.05,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "instruction_tuned",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "chat_dataset",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "add_input_output_demarcation_key",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "per_device_train_batch_size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "per_device_eval_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_val_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "seed",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_data_split_seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/"
        "meta/transfer_learning/textgeneration/v1.0.4/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-"
        "tarballs/meta/transfer_learning/textgeneration/prepack/v1.0.1/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.0.1",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "HF_MODEL_ID",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_INPUT_LENGTH",
                "type": "text",
                "default": "4095",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MAX_TOTAL_TOKENS",
                "type": "text",
                "default": "4096",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SM_NUM_GPUS",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {
                "Name": "huggingface-textgeneration:eval-loss",
                "Regex": "eval_epoch_loss=tensor\\(([0-9\\.]+)",
            },
            {
                "Name": "huggingface-textgeneration:eval-ppl",
                "Regex": "eval_ppl=tensor\\(([0-9\\.]+)",
            },
            {
                "Name": "huggingface-textgeneration:train-loss",
                "Regex": "train_epoch_loss=([0-9\\.]+)",
            },
        ],
        "default_inference_instance_type": "ml.g5.2xlarge",
        "supported_inference_instance_types": [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "default_training_instance_type": "ml.g5.12xlarge",
        "supported_training_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p3dn.24xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 1200,
            "container_startup_health_check_timeout": 1200,
        },
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 256,
        "training_volume_size": 256,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/sec_amazon/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "meta-textgeneration-llama-2-7b",
        "default_payloads": {
            "meaningOfLife": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {"generated_text": "[0].generated_text"},
                "body": {
                    "inputs": "I believe the meaning of life is",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
            "theoryOfRelativity": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {"generated_text": "[0].generated_text"},
                "body": {
                    "inputs": "Simply put, the theory of relativity states that ",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
            "teamMessage": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {"generated_text": "[0].generated_text"},
                "body": {
                    "inputs": "A brief message congratulating the team on the launch:\n\nHi everyone,\n\nI just ",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
            "englishToFrench": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {"generated_text": "[0].generated_text"},
                "body": {
                    "inputs": "Translate English to French:\nsea o"
                    "tter => loutre de mer\npeppermint => ment"
                    "he poivr\u00e9e\nplush girafe => girafe peluche\ncheese =>",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
            "Story": {
                "content_type": "application/json",
                "prompt_key": "inputs",
                "output_keys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "body": {
                    "inputs": "Please tell me a story.",
                    "parameters": {
                        "max_new_tokens": 64,
                        "top_p": 0.9,
                        "temperature": 0.2,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
        },
        "gated_bucket": True,
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/h"
                    "uggingface-pytorch-tgi-inference:2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazon"
                    "aws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {
                    "regional_properties": {"image_uri": "$gpu_ecr_uri_1"},
                    "properties": {
                        "gated_model_key_env_var_value": "meta-training/train-meta-textgeneration-llama-2-7b.tar.gz",
                        "environment_variables": {"SELF_DESTRUCT": "true"},
                    },
                },
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "dynamic_container_deployment_supported": False,
    },
    "js-gated-artifact-trainable-model": {
        "model_id": "meta-textgeneration-llama-2-7b-f",
        "url": "https://ai.meta.com/resources/models-and-libraries/llama-downloads/",
        "version": "2.0.4",
        "min_sdk_version": "2.174.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "djl-deepspeed",
            "framework_version": "0.23.0",
            "py_version": "py39",
        },
        "hosting_artifact_key": "meta-infer/infer-meta-textgeneration-llama-2-7b-f.tar.gz",
        "hosting_script_key": "source-directory-tarballs/meta/inference/textgeneration/v1.2.2/sourcedir.tar.gz",
        "hosting_use_script_uri": False,
        "hosting_eula_key": "fmhMetadata/eula/llamaEula.txt",
        "hosting_model_package_arns": {
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
        },
        "training_model_package_artifact_uris": {
            "us-west-2": "s3://sagemaker-repository-pdx/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "us-east-2": "s3://sagemaker-repository-cmh/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "us-east-1": "s3://sagemaker-repository-iad/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "eu-west-1": "s3://sagemaker-repository-dub/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "ap-southeast-1": "s3://sagemaker-repository-sin/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
            "ap-southeast-2": "s3://sagemaker-repository-syd/model-data-model-package_llama2-7b-f-v4-71eeccf76ddf33f2a18d2e16b9c7f302",
        },
        "inference_vulnerable": False,
        "inference_dependencies": [
            "sagemaker_jumpstart_huggingface_script_utilities==1.0.8",
            "sagemaker_jumpstart_script_utilities==1.1.8",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "accelerate==0.21.0",
            "bitsandbytes==0.39.1",
            "black==23.7.0",
            "brotli==1.0.9",
            "datasets==2.14.1",
            "fire==0.5.0",
            "inflate64==0.3.1",
            "loralib==0.1.1",
            "multivolumefile==0.2.3",
            "mypy-extensions==1.0.0",
            "pathspec==0.11.1",
            "peft==0.4.0",
            "py7zr==0.20.5",
            "pybcj==1.0.1",
            "pycryptodomex==3.18.0",
            "pyppmd==1.0.0",
            "pytorch-triton==2.1.0+6e4932cda8",
            "pyzstd==0.15.9",
            "safetensors==0.3.1",
            "sagemaker_jumpstart_huggingface_script_utilities==1.1.3",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "scipy==1.11.1",
            "termcolor==2.3.0",
            "texttable==1.6.7",
            "tokenize-rt==5.1.0",
            "tokenizers==0.13.3",
            "torch==2.2.0.dev20231104+cu118",
            "transformers==4.31.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "deprecate_warn_message": "For forward compatibility, pin to model_version='2.*' in your JumpStartModel or JumpStartEstimator definitions. Note that major version upgrades may have different EULA acceptance terms and input/output signatures.",
        "hyperparameters": [
            {
                "name": "int8_quantization",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "enable_fsdp",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "epoch",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "lora_r", "type": "int", "default": 8, "min": 1, "scope": "algorithm"},
            {"name": "lora_alpha", "type": "int", "default": 32, "min": 1, "scope": "algorithm"},
            {
                "name": "lora_dropout",
                "type": "float",
                "default": 0.05,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "instruction_tuned",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "chat_dataset",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "add_input_output_demarcation_key",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "per_device_train_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "per_device_eval_batch_size",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_val_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "seed",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "train_data_split_seed",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/meta/transfer_learning/textgeneration/v1.0.6/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "2.0.0",
            "py_version": "py310",
            "huggingface_transformers_version": "4.28.1",
        },
        "training_artifact_key": "meta-training/train-meta-textgeneration-llama-2-7b-f.tar.gz",
        "inference_environment_variables": [],
        "metrics": [
            {
                "Name": "huggingface-textgeneration:eval-loss",
                "Regex": "eval_epoch_loss=tensor\\(([0-9\\.]+)",
            },
            {
                "Name": "huggingface-textgeneration:eval-ppl",
                "Regex": "eval_ppl=tensor\\(([0-9\\.]+)",
            },
            {
                "Name": "huggingface-textgeneration:train-loss",
                "Regex": "train_epoch_loss=([0-9\\.]+)",
            },
        ],
        "default_inference_instance_type": "ml.g5.2xlarge",
        "supported_inference_instance_types": [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "default_training_instance_type": "ml.g5.12xlarge",
        "supported_training_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p3dn.24xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 3600,
            "container_startup_health_check_timeout": 3600,
        },
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 256,
        "training_volume_size": 256,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/oasst_top/train/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "meta-textgeneration-llama-2-7b-f",
        "default_payloads": {
            "Mayo": {
                "content_type": "application/json",
                "body": {
                    "inputs": [[{"role": "user", "content": "what is the recipe of mayonnaise?"}]],
                    "parameters": {"max_new_tokens": 512, "top_p": 0.9, "temperature": 0.6},
                },
            }
        },
        "gated_bucket": True,
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "alias_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-east-1": {
                    "alias_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-northeast-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-south-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "ca-central-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "cn-north-1": {
                    "alias_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-central-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-north-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-south-1": {
                    "alias_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-west-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "eu-west-3": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "me-south-1": {
                    "alias_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "sa-east-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "us-east-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "us-east-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "us-west-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
                "us-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf1": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "dynamic_container_deployment_supported": False,
    },
    "js-trainable-model": {
        "model_id": "autogluon-classification-ensemble",
        "url": "https://auto.gluon.ai/stable/index.html",
        "version": "1.1.1",
        "min_sdk_version": "2.103.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "autogluon",
            "framework_version": "0.4.3",
            "py_version": "py38",
        },
        "hosting_artifact_key": "autogluon-infer/v1.1.0/infer-autogluon-classification-ensemble.tar.gz",
        "hosting_script_key": "source-directory-tarballs/autogluon/inference/classification/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": ["sagemaker_jumpstart_script_utilities==1.0.1"],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {"name": "eval_metric", "type": "text", "default": "auto", "scope": "algorithm"},
            {
                "name": "presets",
                "type": "text",
                "default": "medium_quality",
                "options": [
                    "best_quality",
                    "high_quality",
                    "good_quality",
                    "medium_quality",
                    "optimize_for_deployment",
                    "interpretable",
                ],
                "scope": "algorithm",
            },
            {
                "name": "auto_stack",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "num_bag_folds",
                "type": "text",
                "default": "0",
                "options": ["0", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                "scope": "algorithm",
            },
            {"name": "num_bag_sets", "type": "int", "default": 1, "min": 1, "scope": "algorithm"},
            {
                "name": "num_stack_levels",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 3,
                "scope": "algorithm",
            },
            {
                "name": "refit_full",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "set_best_to_refit_full",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "save_space",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "verbosity",
                "type": "int",
                "default": 2,
                "min": 0,
                "max": 4,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/autogluon/transfer_learning/classification/"
        "v1.0.2/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework": "autogluon",
            "framework_version": "0.4.3",
            "py_version": "py38",
        },
        "training_artifact_key": "autogluon-training/train-autogluon-classification-ensemble.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.p2.xlarge",
        "supported_inference_instance_types": [
            "ml.p2.xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
        ],
        "default_training_instance_type": "ml.p3.2xlarge",
        "supported_training_instance_types": [
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.p2.xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["text/csv"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "text/csv",
            "default_accept_type": "application/json",
        },
        "resource_name_base": "blahblahblah",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1"
                    ".amazonaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1"
                    ".amazonaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1."
                    "amazonaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1."
                    "amazonaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-"
                    "1.amazonaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-"
                    "1.amazonaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2"
                    ".amazonaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2"
                    ".amazonaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.ama"
                    "zonaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazo"
                    "naws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazo"
                    "naws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.a"
                    "mazonaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.am"
                    "azonaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazon"
                    "aws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amaz"
                    "onaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazon"
                    "aws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaw"
                    "s.com.cn/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws"
                    ".com.cn/autogluon-inference:0.4.3-gpu-py38",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazona"
                    "ws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaw"
                    "s.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazona"
                    "ws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazona"
                    "ws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amaz"
                    "onaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazo"
                    "naws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1."
                    "amazonaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.ama"
                    "zonaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazona"
                    "ws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaw"
                    "s.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws"
                    ".com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amaz"
                    "onaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amaz"
                    "onaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazo"
                    "naws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amaz"
                    "onaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.ama"
                    "zonaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.ama"
                    "zonaws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.am"
                    "azonaws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazo"
                    "naws.com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazon"
                    "aws.com/autogluon-inference:0.4.3-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws."
                    "com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws."
                    "com/autogluon-inference:0.4.3-gpu-py38",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws"
                    ".com/autogluon-inference:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws."
                    "com/autogluon-inference:0.4.3-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south"
                    "-1.amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amaz"
                    "naws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-"
                    "1.amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amaz"
                    "onaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.ama"
                    "zonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.a"
                    "mazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2."
                    "amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2."
                    "amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-sou"
                    "th-1.amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.ama"
                    "zonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-"
                    "1.amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast"
                    "-1.amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast"
                    "-2.amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2."
                    "amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.a"
                    "mazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.am"
                    "azonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.a"
                    "mazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.am"
                    "azonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.am"
                    "azonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazon"
                    "aws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amaz"
                    "onaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.ama"
                    "zonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amaz"
                    "onaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazo"
                    "naws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.am"
                    "azonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.am"
                    "azonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3."
                    "amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3."
                    "amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-sout"
                    "h-1.amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1."
                    "amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1."
                    "amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1"
                    ".amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1."
                    "amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1."
                    "amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2"
                    ".amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-"
                    "2.amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west"
                    "-1.amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-wes"
                    "t-1.amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west"
                    "-2.amazonaws.com/autogluon-training:0.4.3-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-w"
                    "est-2.amazonaws.com/autogluon-training:0.4.3-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
    },
    "response-keys": {
        "model_id": "model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16",
        "url": "https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth",
        "version": "1.0.0",
        "min_sdk_version": "2.144.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "djl-deepspeed",
            "framework_version": "0.21.0",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17",
        },
        "hosting_artifact_key": "stabilityai-infer/infer-model-depth2img-st"
        "able-diffusion-v1-5-controlnet-v1-1-fp16.tar.gz",
        "hosting_script_key": "source-directory-tarballs/stabilityai/inference/depth2img/v1.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "stabilityai-infer/prepack/v1.0.0/"
        "infer-prepack-model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.0",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "accelerate==0.18.0",
            "diffusers==0.14.0",
            "fsspec==2023.4.0",
            "huggingface-hub==0.14.1",
            "transformers==4.26.1",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.g5.8xlarge",
        "supported_inference_instance_types": [
            "ml.g5.8xlarge",
            "ml.g5.xlarge",
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.16xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.4xlarge",
            "ml.g4dn.8xlarge",
            "ml.g4dn.16xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "validation_supported": False,
        "fine_tuning_supported": False,
        "resource_name_base": "sd-1-5-controlnet-1-1-fp16",
        "default_payloads": {
            "Dog": {
                "content_type": "application/json",
                "prompt_key": "hello.prompt",
                "body": {
                    "hello": {"prompt": "a dog"},
                    "seed": 43,
                },
            }
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "alias_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/d"
                    "jl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf1": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
            },
        },
    },
    "default_payloads": {
        "model_id": "model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16",
        "url": "https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth",
        "version": "2.0.5",
        "min_sdk_version": "2.189.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "djl-deepspeed",
            "framework_version": "0.21.0",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17",
        },
        "hosting_artifact_key": "stabilityai-depth2img/model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16/artifacts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/stabilityai/inference/depth2img/v1.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "stabilityai-depth2img/model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [
            "accelerate==0.18.0",
            "diffusers==0.14.0",
            "fsspec==2023.4.0",
            "huggingface-hub==0.14.1",
            "transformers==4.26.1",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "default_payloads": {
            "Dog": {
                "content_type": "application/json",
                "body": {
                    "prompt": "a dog",
                    "num_images_per_prompt": 2,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "seed": 43,
                    "eta": 0.7,
                    "image": "$s3_b64<inference-notebook-assets/inpainting_cow.jpg>",
                },
            }
        },
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.g5.8xlarge",
        "supported_inference_instance_types": [
            "ml.g5.8xlarge",
            "ml.g5.xlarge",
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.16xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.4xlarge",
            "ml.g4dn.8xlarge",
            "ml.g4dn.16xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "validation_supported": False,
        "fine_tuning_supported": False,
        "resource_name_base": "sd-1-5-controlnet-1-1-fp16",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "alias_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ap-east-1": {
                    "alias_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ap-northeast-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ap-northeast-3": {
                    "alias_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ap-south-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ap-southeast-3": {
                    "alias_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "ca-central-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "cn-north-1": {
                    "alias_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "cn-northwest-1": {
                    "alias_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "eu-central-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "eu-north-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "eu-south-1": {
                    "alias_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "eu-west-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "eu-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "eu-west-3": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "il-central-1": {
                    "alias_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "me-south-1": {
                    "alias_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "sa-east-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "us-east-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "us-east-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "us-gov-east-1": {
                    "alias_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "us-gov-west-1": {
                    "alias_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "us-west-1": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
                "us-west-2": {
                    "alias_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "prompt-key": {
        "model_id": "model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16",
        "url": "https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth",
        "version": "1.0.0",
        "min_sdk_version": "2.144.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "djl-deepspeed",
            "framework_version": "0.21.0",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17",
        },
        "hosting_artifact_key": "stabilityai-infer/infer-model-depth2img-st"
        "able-diffusion-v1-5-controlnet-v1-1-fp16.tar.gz",
        "hosting_script_key": "source-directory-tarballs/stabilityai/inference/depth2img/v1.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "stabilityai-infer/prepack/v1.0.0/"
        "infer-prepack-model-depth2img-stable-diffusion-v1-5-controlnet-v1-1-fp16.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.0",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "accelerate==0.18.0",
            "diffusers==0.14.0",
            "fsspec==2023.4.0",
            "huggingface-hub==0.14.1",
            "transformers==4.26.1",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.g5.8xlarge",
        "supported_inference_instance_types": [
            "ml.g5.8xlarge",
            "ml.g5.xlarge",
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.16xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.4xlarge",
            "ml.g4dn.8xlarge",
            "ml.g4dn.16xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "validation_supported": False,
        "fine_tuning_supported": False,
        "resource_name_base": "sd-1-5-controlnet-1-1-fp16",
        "default_payloads": {
            "Dog": {
                "content_type": "application/json",
                "prompt_key": "hello.prompt",
                "body": {
                    "hello": {"prompt": "a dog"},
                    "seed": 43,
                },
            }
        },
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "alias_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/d"
                    "jl-inference:0.21.0-deepspeed0.8.3-cu117"
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf1": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "inf2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$alias_ecr_uri_1"}},
            },
        },
    },
    "predictor-specs-model": {
        "model_id": "huggingface-text2text-flan-t5-xxl-fp16",
        "url": "https://huggingface.co/google/flan-t5-xxl",
        "version": "1.0.1",
        "min_sdk_version": "2.130.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.12.0",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-text2text-flan-t5-xxl-fp16.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v1.0.3/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.0.1/infer-prepack-huggingface-"
        "text2text-flan-t5-xxl-fp16.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.1",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "accelerate==0.16.0",
            "bitsandbytes==0.37.0",
            "filelock==3.9.0",
            "huggingface_hub==0.12.0",
            "regex==2022.7.9",
            "tokenizers==0.13.2",
            "transformers==4.26.0",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
            },
            {"name": "SAGEMAKER_ENV", "type": "text", "default": "1", "scope": "container"},
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "text",
                "default": "1",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.g5.12xlarge",
        "supported_inference_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.g4dn.12xlarge",
        ],
        "predictor_specs": {
            "supported_content_types": ["application/x-text"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-text",
            "default_accept_type": "application/json",
        },
    },
    "model_data_s3_prefix_model": {
        "model_id": "huggingface-text2text-flan-t5-xxl-fp16",
        "url": "https://huggingface.co/google/flan-t5-xxl",
        "version": "1.1.2",
        "min_sdk_version": "2.144.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.13.1",
            "py_version": "py39",
            "huggingface_transformers_version": "4.26.0",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-text2text-flan-t5-xxl-fp16.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v1.1.2/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.1.2/infer-prepack-huggingface-text2text-flan-t5-xxl-fp16.tar.gz",
        "hosting_prepacked_artifact_version": "1.1.2",
        "inference_vulnerable": False,
        "inference_dependencies": ["accelerate==0.19.0", "bitsandbytes==0.38.1", "peft==0.3.0"],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "accelerate==0.19.0",
            "datasets==2.12.0",
            "deepspeed==0.9.2",
            "peft==0.3.0",
            "sagemaker_jumpstart_huggingface_script_utilities==1.0.2",
            "sagemaker_jumpstart_script_utilities==1.1.4",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {"name": "max_steps", "type": "int", "default": -1, "scope": "algorithm"},
            {
                "name": "seed",
                "type": "int",
                "default": 42,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "batch_size",
                "type": "int",
                "default": 64,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.0001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "lr_scheduler_type",
                "type": "text",
                "default": "constant_with_warmup",
                "options": ["constant_with_warmup", "linear"],
                "scope": "algorithm",
            },
            {
                "name": "warmup_ratio",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "warmup_steps", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.05,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "train_data_split_seed", "type": "int", "default": 0, "scope": "algorithm"},
            {
                "name": "max_train_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_eval_samples",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_input_length",
                "type": "int",
                "default": -1,
                "min": -1,
                "scope": "algorithm",
            },
            {
                "name": "max_output_length",
                "type": "int",
                "default": 128,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "pad_to_max_length",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "gradient_accumulation_steps",
                "type": "int",
                "default": 1,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "weight_decay",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta1",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_beta2",
                "type": "float",
                "default": 0.999,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "adam_epsilon",
                "type": "float",
                "default": 1e-08,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "max_grad_norm",
                "type": "float",
                "default": 1.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "load_best_model_at_end",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_patience",
                "type": "int",
                "default": 3,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_threshold",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "label_smoothing_factor",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_strategy",
                "type": "text",
                "default": "steps",
                "options": ["no", "steps", "epoch"],
                "scope": "algorithm",
            },
            {
                "name": "logging_first_step",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "logging_steps",
                "type": "int",
                "default": 500,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "logging_nan_inf_filter",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "save_strategy",
                "type": "text",
                "default": "epoch",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "save_steps", "type": "int", "default": 500, "min": 1, "scope": "algorithm"},
            {"name": "save_total_limit", "type": "text", "default": "2", "scope": "algorithm"},
            {
                "name": "dataloader_drop_last",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "dataloader_num_workers",
                "type": "int",
                "default": 0,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "evalaution_strategy",
                "type": "text",
                "default": "epoch",
                "options": ["no", "epoch", "steps"],
                "scope": "algorithm",
            },
            {"name": "eval_steps", "type": "text", "default": "500", "scope": "algorithm"},
            {
                "name": "eval_accumulation_steps",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "gradient_checkpointing",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "auto_find_batch_size",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "preprocessing_num_workers",
                "type": "text",
                "default": "None",
                "scope": "algorithm",
            },
            {
                "name": "peft_type",
                "type": "text",
                "default": "lora",
                "options": ["lora"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/v1.2.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/prepack/v1.1.2/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.2",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.13.1",
            "py_version": "py39",
            "huggingface_transformers_version": "4.26.0",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-text2text-flan-t5-xxl-fp16.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "TS_DEFAULT_WORKERS_PER_MODEL",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {"Name": "huggingface-text2text:eval-loss", "Regex": "'eval_loss': ([0-9\\.]+)"}
        ],
        "default_inference_instance_type": "ml.g5.12xlarge",
        "supported_inference_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.g4dn.12xlarge",
        ],
        "default_training_instance_type": "ml.g5.24xlarge",
        "supported_training_instance_types": ["ml.g5.24xlarge", "ml.g5.48xlarge"],
        "model_kwargs": {},
        "deploy_kwargs": {
            "model_data_download_timeout": 3600,
            "container_startup_health_check_timeout": 3600,
        },
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/x-text", "application/json"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-text",
            "default_accept_type": "application/json",
        },
        "inference_volume_size": 256,
        "training_volume_size": 256,
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/genuq/dev/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "hf-text2text-flan-t5-xxl-fp16",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "dynamic_container_deployment_supported": False,
    },
    "no-supported-instance-types-model": {
        "model_id": "pytorch-ic-mobilenet-v2",
        "url": "https://pytorch.org/hub/pytorch_vision_mobilenet_v2/",
        "version": "1.0.0",
        "min_sdk_version": "2.49.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "hosting_artifact_key": "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
        "training_artifact_key": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
        "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz",
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "adam-learning-rate",
                "type": "float",
                "default": 0.05,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch-size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
            },
            {"name": "SAGEMAKER_ENV", "type": "text", "default": "1", "scope": "container"},
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "text",
                "default": "1",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
            },
        ],
        "default_inference_instance_type": "",
        "supported_inference_instance_types": None,
        "default_training_instance_type": None,
        "supported_training_instance_types": [],
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "metrics": [],
    },
    "huggingface-text2text-flan-t5-xxl-fp16": {
        "model_id": "huggingface-text2text-flan-t5-xxl-fp16",
        "url": "https://huggingface.co/google/flan-t5-xxl",
        "version": "1.0.0",
        "min_sdk_version": "2.130.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.12.0",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-text2text-flan-t5-xxl-fp16.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v1.0.2/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.0.0/infer-prepack-huggingface-"
        "text2text-flan-t5-xxl-fp16.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.0",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "accelerate==0.16.0",
            "bitsandbytes==0.37.0",
            "filelock==3.9.0",
            "huggingface-hub==0.12.0",
            "regex==2022.7.9",
            "tokenizers==0.13.2",
            "transformers==4.26.0",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
            },
            {"name": "SAGEMAKER_ENV", "type": "text", "default": "1", "scope": "container"},
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "text",
                "default": "1",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
            },
        ],
        "inference_vulnerable": False,
        "training_vulnerable": False,
        "deprecated": False,
        "default_training_instance_type": None,
        "supported_training_instance_types": [],
        "metrics": [],
        "default_inference_instance_type": "ml.g5.12xlarge",
        "supported_inference_instance_types": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.g4dn.12xlarge",
        ],
    },
    "mock-model-training-prepacked-script-key": {
        "model_id": "sklearn-classification-linear",
        "url": "https://scikit-learn.org/stable/",
        "version": "1.0.0",
        "min_sdk_version": "2.68.1",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "sklearn",
            "framework_version": "0.23-1",
            "py_version": "py3",
        },
        "hosting_artifact_key": "sklearn-infer/infer-sklearn-classification-linear.tar.gz",
        "hosting_script_key": "source-directory-tarballs/sklearn/inference/classification/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "tol",
                "type": "float",
                "default": 0.0001,
                "min": 1e-20,
                "max": 50,
                "scope": "algorithm",
            },
            {
                "name": "penalty",
                "type": "text",
                "default": "l2",
                "options": ["l1", "l2", "elasticnet", "none"],
                "scope": "algorithm",
            },
            {
                "name": "alpha",
                "type": "float",
                "default": 0.0001,
                "min": 1e-20,
                "max": 999,
                "scope": "algorithm",
            },
            {
                "name": "l1_ratio",
                "type": "float",
                "default": 0.15,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/sklearn/transfer_learning/classification/"
        "v1.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "some/key/to/training_prepacked_script_key.tar.gz",
        "training_ecr_specs": {
            "framework_version": "0.23-1",
            "framework": "sklearn",
            "py_version": "py3",
        },
        "training_artifact_key": "sklearn-training/train-sklearn-classification-linear.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
            },
            {"name": "SAGEMAKER_ENV", "type": "text", "default": "1", "scope": "container"},
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "text",
                "default": "1",
                "scope": "container",
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
            },
        ],
    },
}


PROTOTYPICAL_MODEL_SPECS_DICT = {
    "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1": {
        "model_id": "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1",
        "url": "https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1",
        "version": "4.0.6",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "tensorflow",
            "framework_version": "2.8",
            "py_version": "py39",
        },
        "hosting_artifact_key": "tensorflow-ic/tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1/artifacts/inference/v3.0.0/",
        "hosting_script_key": "source-directory-tarballs/tensorflow/inference/ic/v2.0.3/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "tensorflow-ic/tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "sagemaker_jumpstart_prepack_script_utilities==1.0.0",
            "sagemaker_jumpstart_script_utilities==1.1.1",
            "sagemaker_jumpstart_tensorflow_script_utilities==1.0.1",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "train_only_top_layer",
                "type": "text",
                "default": "True",
                "options": ["False", "True"],
                "scope": "algorithm",
            },
            {
                "name": "epochs",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "batch_size",
                "type": "int",
                "default": 32,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "optimizer",
                "type": "text",
                "default": "adam",
                "options": ["adam", "sgd", "nesterov", "rmsprop", "adagrad", "adadelta"],
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.001,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "beta_1",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "beta_2",
                "type": "float",
                "default": 0.999,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "momentum",
                "type": "float",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "epsilon",
                "type": "float",
                "default": 1e-07,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "rho",
                "type": "float",
                "default": 0.95,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "initial_accumulator_value",
                "type": "float",
                "default": 0.1,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "reinitialize_top_layer",
                "type": "text",
                "default": "Auto",
                "options": ["Auto", "True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping",
                "type": "text",
                "default": "False",
                "options": ["False", "True"],
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_patience",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_min_delta",
                "type": "float",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "dropout_rate",
                "type": "float",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "regularizers_l2",
                "type": "float",
                "default": 0.0001,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "label_smoothing",
                "type": "float",
                "default": 0.1,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "image_resize_interpolation",
                "type": "text",
                "default": "bilinear",
                "options": [
                    "bilinear",
                    "nearest",
                    "bicubic",
                    "area",
                    "lanczos3",
                    "lanczos5",
                    "gaussian",
                    "mitchellcubic",
                ],
                "scope": "algorithm",
            },
            {
                "name": "augmentation",
                "type": "text",
                "default": "False",
                "options": ["False", "True"],
                "scope": "algorithm",
            },
            {
                "name": "augmentation_random_flip",
                "type": "text",
                "default": "horizontal_and_vertical",
                "options": ["horizontal_and_vertical", "horizontal", "vertical", "None"],
                "scope": "algorithm",
            },
            {
                "name": "augmentation_random_rotation",
                "type": "float",
                "default": 0.2,
                "min": -1,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "augmentation_random_zoom",
                "type": "float",
                "default": 0.1,
                "min": -1,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "binary_mode",
                "type": "text",
                "default": "False",
                "options": ["False", "True"],
                "scope": "algorithm",
            },
            {
                "name": "eval_metric",
                "type": "text",
                "default": "accuracy",
                "options": ["accuracy", "precision", "recall", "auc", "prc"],
                "scope": "algorithm",
            },
            {
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "random_seed", "type": "int", "default": 123, "min": 0, "scope": "algorithm"},
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/tensorflow/transfer_learning/ic/v2.1.2/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/tensorflow/transfer_learning/ic/prepack/v1.1.2/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.2",
        "training_ecr_specs": {
            "framework": "tensorflow",
            "framework_version": "2.9",
            "py_version": "py39",
        },
        "training_artifact_key": "tensorflow-training/v3.0.0/train-tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [{"Name": "tflow-ic:val-accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"}],
        "default_inference_instance_type": "ml.p3.2xlarge",
        "supported_inference_instance_types": [
            "ml.p3.2xlarge",
            "ml.p2.xlarge",
            "ml.g4dn.xlarge",
            "ml.m5.xlarge",
            "ml.m4.xlarge",
            "ml.m5.large",
            "ml.c5.2xlarge",
            "ml.c5.xlarge",
            "ml.r5.xlarge",
            "ml.r5.large",
            "ml.c6i.xlarge",
            "ml.c6i.large",
        ],
        "default_training_instance_type": "ml.p3.2xlarge",
        "supported_training_instance_types": [
            "ml.p3.2xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.p2.xlarge",
            "ml.p2.8xlarge",
            "ml.p2.16xlarge",
            "ml.g5.xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.4xlarge",
            "ml.g4dn.8xlarge",
            "ml.g4dn.16xlarge",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/x-image"],
            "supported_accept_types": ["application/json", "application/json;verbose"],
            "default_content_type": "application/x-image",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/tf_flowers/",
        "validation_supported": True,
        "fine_tuning_supported": True,
        "resource_name_base": "bit-m-r101x1-ilsvrc2012-classification",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/tensorflow-inference:2.8-gpu",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/tensorflow-inference:2.8-gpu",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.8-cpu",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.8-gpu",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/tensorflow-training:2.9-gpu-py39",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/tensorflow-training:2.9-gpu-py39",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.9-cpu-py39",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.9-gpu-py39",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "pytorch-ic-mobilenet-v2": {
        "model_id": "pytorch-ic-mobilenet-v2",
        "url": "https://pytorch.org/hub/pytorch_vision_mobilenet_v2/",
        "version": "3.0.6",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        },
        "hosting_artifact_key": "pytorch-ic/pytorch-ic-mobilenet-v2/artifacts/inference/v2.0.0/",
        "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v2.0.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "pytorch-ic/pytorch-ic-mobilenet-v2/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": ["sagemaker_jumpstart_prepack_script_utilities==1.0.0"],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "train_only_top_layer",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "epochs",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch_size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "reinitialize_top_layer",
                "type": "text",
                "default": "Auto",
                "options": ["Auto", "True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v2.3.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/prepack/v1.1.0/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.0",
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        },
        "training_artifact_key": "pytorch-training/v2.0.0/train-pytorch-ic-mobilenet-v2.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [{"Name": "pytorch-ic:val-accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"}],
        "default_inference_instance_type": "ml.m5.large",
        "supported_inference_instance_types": [
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.m4.large",
            "ml.m4.xlarge",
        ],
        "default_training_instance_type": "ml.m5.xlarge",
        "supported_training_instance_types": ["ml.m5.xlarge", "ml.c5.2xlarge", "ml.m4.xlarge"],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/x-image"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-image",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/tf_flowers/",
        "validation_supported": False,
        "fine_tuning_supported": True,
        "resource_name_base": "pt-ic-mobilenet-v2",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-south-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:1.10.0-gpu-py38",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-central-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-west-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "sa-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-east-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
                "us-west-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "trn1": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
                "trn1n": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-south-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.10.0-gpu-py38",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-central-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-west-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "sa-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-east-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
                "us-west-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "trn1": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
                "trn1n": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "mxnet-semseg-fcn-resnet50-ade": {
        "model_id": "mxnet-semseg-fcn-resnet50-ade",
        "url": "https://cv.gluon.ai/model_zoo/segmentation.html",
        "version": "2.0.3",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "mxnet",
            "framework_version": "1.9.0",
            "py_version": "py38",
        },
        "hosting_artifact_key": "mxnet-semseg/mxnet-semseg-fcn-resnet50-ade/artifacts/inference/v1.1.0/",
        "hosting_script_key": "source-directory-tarballs/mxnet/inference/semseg/v1.2.1/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "mxnet-semseg/mxnet-semseg-fcn-resnet50-ade/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "numpy==2.0",
            "opencv_python==4.7.0.68",
            "sagemaker_jumpstart_prepack_script_utilities==1.0.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "adam-learning-rate",
                "type": "float",
                "default": 0.001,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch-size",
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "train-only-top-layer",
                "type": "text",
                "default": "True",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/mxnet/transfer_learning/semseg/v1.5.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/mxnet/transfer_learning/semseg/prepack/v1.1.0/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.0",
        "training_ecr_specs": {
            "framework": "mxnet",
            "framework_version": "1.9.0",
            "py_version": "py38",
        },
        "training_artifact_key": "mxnet-training/train-mxnet-semseg-fcn-resnet50-ade.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [{"Name": "mxnet-semseg:val-loss", "Regex": "validation loss=([0-9\\.]+)"}],
        "default_inference_instance_type": "ml.p3.2xlarge",
        "supported_inference_instance_types": [
            "ml.p2.xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
        ],
        "default_training_instance_type": "ml.p3.2xlarge",
        "supported_training_instance_types": [
            "ml.p3.2xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.g4dn.8xlarge",
            "ml.g4dn.16xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/x-image"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-image",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/PennFudanPed_SemSeg/",
        "validation_supported": False,
        "fine_tuning_supported": True,
        "resource_name_base": "mx-semseg-fcn-resnet50-ade",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/mxnet-inference:1.9.0-gpu-py38",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/mxnet-inference:1.9.0-gpu-py38",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference:1.9.0-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/mxnet-training:1.9.0-gpu-py38",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/mxnet-training:1.9.0-gpu-py38",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-training:1.9.0-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "huggingface-spc-bert-base-cased": {
        "model_id": "huggingface-spc-bert-base-cased",
        "url": "https://huggingface.co/bert-base-cased",
        "version": "2.0.3",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.7.1",
            "py_version": "py36",
            "huggingface_transformers_version": "4.6.1",
        },
        "hosting_artifact_key": "huggingface-spc/huggingface-spc-bert-base-cased/artifacts/inference/v1.2.0/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/spc/v1.1.3/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-spc/huggingface-spc-bert-base-cased/artifacts/inference-prepack/v1.0.0/",
        "hosting_prepacked_artifact_version": "1.0.0",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": ["sagemaker_jumpstart_prepack_script_utilities==1.0.0"],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "epochs",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            },
            {
                "name": "adam-learning-rate",
                "type": "float",
                "default": 2e-05,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "batch-size",
                "type": "int",
                "default": 8,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            },
            {
                "name": "reinitialize-top-layer",
                "type": "text",
                "default": "Auto",
                "options": ["Auto", "True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "train-only-top-layer",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/spc/v1.3.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/spc/prepack/v1.1.0/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.0",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.6.0",
            "py_version": "py36",
            "huggingface_transformers_version": "4.4.2",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-spc-bert-base-cased.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {"Name": "hugginface-spc:eval-accuracy", "Regex": "'eval_accuracy': ([0-9\\.]+)"}
        ],
        "default_inference_instance_type": "ml.p3.2xlarge",
        "supported_inference_instance_types": [
            "ml.p2.xlarge",
            "ml.p3.2xlarge",
            "ml.g4dn.xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
        ],
        "default_training_instance_type": "ml.p3.2xlarge",
        "supported_training_instance_types": [
            "ml.p3.2xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.p2.xlarge",
            "ml.p2.8xlarge",
            "ml.p2.16xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.4xlarge",
            "ml.g4dn.8xlarge",
            "ml.g4dn.16xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/list-text"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/list-text",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/QNLI/",
        "validation_supported": False,
        "fine_tuning_supported": True,
        "resource_name_base": "hf-spc-bert-base-cased",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "gpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ap-east-1": {
                    "gpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ap-northeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ap-northeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ap-northeast-3": {
                    "gpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ap-south-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ap-southeast-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ap-southeast-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ap-southeast-3": {
                    "gpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "ca-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "cn-north-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "cn-northwest-1": {
                    "gpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "eu-central-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "eu-central-2": {
                    "gpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "eu-north-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "eu-south-1": {
                    "gpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "eu-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "eu-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "eu-west-3": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "il-central-1": {
                    "gpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "me-central-1": {
                    "gpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "me-south-1": {
                    "gpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "sa-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "us-east-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "us-east-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "us-gov-east-1": {
                    "gpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "us-gov-west-1": {
                    "gpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "us-west-1": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
                "us-west-2": {
                    "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
                },
            },
            "variants": {
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "lightgbm-classification-model": {
        "model_id": "lightgbm-classification-model",
        "url": "https://lightgbm.readthedocs.io/en/latest/",
        "version": "2.1.6",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "2.0.1",
            "py_version": "py310",
        },
        "hosting_artifact_key": "lightgbm-classification/lightgbm-classification-model/artifacts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/lightgbm/inference/classification/v1.2.2/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "lightgbm-classification/lightgbm-classification-model/artifacts/inference-prepack/v1.0.1/",
        "hosting_prepacked_artifact_version": "1.0.1",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": ["lightgbm==4.1.0"],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "HeapDict==1.0.1",
            "dask==2022.12.1",
            "distributed==2022.12.1",
            "graphviz==0.17",
            "lightgbm==3.3.3",
            "locket==1.0.0",
            "msgpack==1.0.4",
            "partd==1.3.0",
            "sagemaker_jumpstart_prepack_script_utilities==1.0.0",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
            "sortedcontainers==2.4.0",
            "tblib==1.7.0",
            "toolz==0.12.0",
            "zict==2.2.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "num_boost_round",
                "type": "int",
                "default": 5000,
                "min": 1,
                "max": 100000,
                "scope": "algorithm",
            },
            {"name": "early_stopping_rounds", "type": "int", "default": 30, "scope": "algorithm"},
            {"name": "metric", "type": "text", "default": "auto", "scope": "algorithm"},
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.009,
                "min": 1e-20,
                "scope": "algorithm",
            },
            {
                "name": "num_leaves",
                "type": "int",
                "default": 67,
                "min": 2,
                "max": 131072,
                "scope": "algorithm",
            },
            {
                "name": "feature_fraction",
                "type": "float",
                "default": 0.74,
                "min": 1e-20,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "bagging_fraction",
                "type": "float",
                "default": 0.53,
                "min": 1e-20,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "bagging_freq",
                "type": "int",
                "default": 5,
                "min": 0,
                "max": 100000,
                "scope": "algorithm",
            },
            {"name": "max_depth", "type": "int", "default": 11, "scope": "algorithm"},
            {
                "name": "min_data_in_leaf",
                "type": "int",
                "default": 26,
                "min": 0,
                "scope": "algorithm",
            },
            {"name": "max_delta_step", "type": "float", "default": 0.0, "scope": "algorithm"},
            {
                "name": "lambda_l1",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "scope": "algorithm",
            },
            {
                "name": "lambda_l2",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "scope": "algorithm",
            },
            {
                "name": "boosting",
                "type": "text",
                "default": "gbdt",
                "options": ["gbdt", "rf", "dart", "goss"],
                "scope": "algorithm",
            },
            {
                "name": "min_gain_to_split",
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "scope": "algorithm",
            },
            {
                "name": "scale_pos_weight",
                "type": "float",
                "default": 1.0,
                "min": 1e-20,
                "scope": "algorithm",
            },
            {
                "name": "tree_learner",
                "type": "text",
                "default": "serial",
                "options": ["serial", "feature", "data", "voting"],
                "scope": "algorithm",
            },
            {
                "name": "feature_fraction_bynode",
                "type": "float",
                "default": 1.0,
                "min": 1e-20,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "is_unbalance",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {"name": "max_bin", "type": "int", "default": 255, "min": 2, "scope": "algorithm"},
            {"name": "num_threads", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {"name": "verbosity", "type": "int", "default": 1, "scope": "algorithm"},
            {
                "name": "use_dask",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/lightgbm/transfer_learning/classification/v2.2.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/lightgbm/transfer_learning/classification/prepack/v1.1.1/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.1",
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.9.0",
            "py_version": "py38",
        },
        "training_artifact_key": "lightgbm-training/train-lightgbm-classification-model.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {
                "Name": "lightgbm-classification:multi-log-loss",
                "Regex": "multi_logloss: ([0-9\\.]+)",
            }
        ],
        "default_inference_instance_type": "ml.m5.4xlarge",
        "supported_inference_instance_types": [
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.m4.16xlarge",
        ],
        "default_training_instance_type": "ml.m5.12xlarge",
        "supported_training_instance_types": [
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.m4.16xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["text/csv"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "text/csv",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/tabular_multiclass/",
        "validation_supported": True,
        "fine_tuning_supported": False,
        "resource_name_base": "lgb-classification-model",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:2.0.1-gpu-py310",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "c6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "c6gn": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "m6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "r6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-south-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.9.0-gpu-py38",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-central-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-west-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "sa-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-east-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-west-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "trn1": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
                "trn1n": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "catboost-classification-model": {
        "model_id": "catboost-classification-model",
        "url": "https://catboost.ai/",
        "version": "2.1.6",
        "min_sdk_version": "2.189.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "2.0.1",
            "py_version": "py310",
        },
        "hosting_artifact_key": "catboost-classification/catboost-classification-model/artifacts/inference/v1.0.0/",
        "hosting_script_key": "source-directory-tarballs/catboost/inference/classification/v1.1.2/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "catboost-classification/catboost-classification-model/artifacts/inference-prepack/v1.0.1/",
        "hosting_prepacked_artifact_version": "1.0.1",
        "hosting_use_script_uri": False,
        "inference_vulnerable": False,
        "inference_dependencies": [
            "catboost==1.2.2",
            "graphviz==0.20.1",
            "plotly==5.18.0",
            "tenacity==8.2.3",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "catboost==1.0.1",
            "graphviz==0.17",
            "plotly==5.1.0",
            "sagemaker_jumpstart_prepack_script_utilities==1.0.0",
            "sagemaker_jumpstart_script_utilities==1.0.1",
            "tenacity==8.0.1",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "iterations",
                "type": "int",
                "default": 500,
                "min": 1,
                "max": 100000,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_rounds",
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 5000,
                "scope": "algorithm",
            },
            {"name": "eval_metric", "type": "text", "default": "Auto", "scope": "algorithm"},
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.03,
                "min": 1e-20,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "depth",
                "type": "int",
                "default": 6,
                "min": 1,
                "max": 16,
                "scope": "algorithm",
            },
            {
                "name": "l2_leaf_reg",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10000,
                "scope": "algorithm",
            },
            {
                "name": "random_strength",
                "type": "float",
                "default": 1.0,
                "min": 1e-20,
                "max": 10,
                "scope": "algorithm",
            },
            {"name": "max_leaves", "type": "int", "default": 31, "min": 2, "scope": "algorithm"},
            {
                "name": "rsm",
                "type": "float",
                "default": 1,
                "min": 1e-20,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "sampling_frequency",
                "type": "text",
                "default": "PerTreeLevel",
                "options": ["PerTreeLevel", "PerTree"],
                "scope": "algorithm",
            },
            {
                "name": "min_data_in_leaf",
                "type": "int",
                "default": 1,
                "min": 1,
                "scope": "algorithm",
            },
            {
                "name": "bagging_temperature",
                "type": "float",
                "default": 1,
                "min": 0,
                "scope": "algorithm",
            },
            {
                "name": "boosting_type",
                "type": "text",
                "default": "Auto",
                "options": ["Auto", "Ordered", "Plain"],
                "scope": "algorithm",
            },
            {
                "name": "scale_pos_weight",
                "type": "float",
                "default": 1.0,
                "min": 1e-20,
                "scope": "algorithm",
            },
            {"name": "max_bin", "type": "text", "default": "Auto", "scope": "algorithm"},
            {
                "name": "grow_policy",
                "type": "text",
                "default": "SymmetricTree",
                "options": ["SymmetricTree", "Depthwise", "Lossguide"],
                "scope": "algorithm",
            },
            {"name": "random_seed", "type": "int", "default": 0, "min": 0, "scope": "algorithm"},
            {"name": "thread_count", "type": "int", "default": -1, "min": -1, "scope": "algorithm"},
            {"name": "verbose", "type": "int", "default": 1, "min": 1, "scope": "algorithm"},
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/catboost/transfer_learning/classification/v1.2.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/catboost/transfer_learning/classification/prepack/v1.1.1/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.1.1",
        "training_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.9.0",
            "py_version": "py38",
        },
        "training_artifact_key": "catboost-training/train-catboost-classification-model.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [
            {"Name": "catboost-classification:multi-logloss", "Regex": "multi_logloss: ([0-9\\.]+)"}
        ],
        "default_inference_instance_type": "ml.m5.4xlarge",
        "supported_inference_instance_types": [
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.m4.16xlarge",
        ],
        "default_training_instance_type": "ml.m5.12xlarge",
        "supported_training_instance_types": [
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.m4.16xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["text/csv"],
            "supported_accept_types": ["application/json", "application/json;verbose"],
            "default_content_type": "text/csv",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/tabular_multiclass/",
        "validation_supported": True,
        "fine_tuning_supported": False,
        "resource_name_base": "cat-classification-model",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:2.0.1-gpu-py310",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-east-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-east-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
                "us-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
                    "cpu_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "c6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "c6gn": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "m6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "r6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_3"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-east-1": {
                    "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-northeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-south-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-southeast-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-southeast-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-southeast-3": {
                    "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ap-southeast-5": {
                    "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "ca-central-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "cn-north-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.9.0-gpu-py38",
                },
                "cn-northwest-1": {
                    "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-central-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-west-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "il-central-1": {
                    "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "me-central-1": {
                    "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "me-south-1": {
                    "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "sa-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-east-1": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-east-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-gov-west-1": {
                    "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-west-1": {
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
                "us-west-2": {
                    "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                    "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.9.0-cpu-py38",
                    "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "trn1": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
                "trn1n": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
            },
        },
        "hosting_artifact_s3_data_type": "S3Prefix",
        "hosting_artifact_compression_type": "None",
        "dynamic_container_deployment_supported": False,
    },
    "xgboost-classification-model": {
        "model_id": "xgboost-classification-model",
        "url": "https://xgboost.readthedocs.io/en/release_1.7.0/",
        "version": "2.1.1",
        "min_sdk_version": "2.188.0",
        "training_supported": True,
        "incremental_training_supported": True,
        "hosting_ecr_specs": {
            "framework": "xgboost",
            "framework_version": "1.7-1",
            "py_version": "py3",
        },
        "hosting_artifact_key": "xgboost-infer/infer-xgboost-classification-model.tar.gz",
        "hosting_script_key": "source-directory-tarballs/xgboost/inference/classification/v1.1.0/sourcedir.tar.gz",
        "hosting_use_script_uri": True,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "asn1crypto==1.5.1",
            "attrs==23.1.0",
            "boto3==1.26.158",
            "botocore==1.29.159",
            "certifi==2023.5.7",
            "cffi==1.15.1",
            "charset-normalizer==2.1.1",
            "cloudpickle==2.2.1",
            "contextlib2==21.6.0",
            "cryptography==40.0.2",
            "dill==0.3.6",
            "filelock==3.12.2",
            "google-pasta==0.2.0",
            "idna==3.4",
            "importlib-metadata==4.13.0",
            "importlib-resources==5.12.0",
            "jmespath==1.0.1",
            "jsonschema==4.17.3",
            "multiprocess==0.70.14",
            "numpy==2.0",
            "oscrypto==1.3.0",
            "packaging==23.1",
            "pandas==2.2.3",
            "pathos==0.3.0",
            "pkgutil-resolve-name==1.3.10",
            "platformdirs==3.8.0",
            "pox==0.3.2",
            "ppft==1.7.6.6",
            "protobuf3-to-dict==0.1.5",
            "protobuf==3.20.3",
            "pycparser==2.21",
            "pycryptodomex==3.12.0",
            "pyjwt==2.7.0",
            "pyopenssl==23.2.0",
            "pyrsistent==0.19.3",
            "python-dateutil==2.8.2",
            "pytz==2023.3",
            "pyyaml==6.0",
            "requests==2.31.0",
            "s3transfer==0.6.1",
            "sagemaker==2.164.0",
            "sagemaker_jumpstart_script_utilities==1.0.1",
            "sagemaker_jumpstart_snowflake_script_utilities==1.1.0",
            "schema==0.7.5",
            "six==1.16.0",
            "smdebug-rulesconfig==1.0.1",
            "snowflake-connector-python==3.12.3",
            "tblib==1.7.0",
            "typing-extensions==4.6.3",
            "tzdata==2023.3",
            "urllib3==1.26.16",
            "zipp==3.15.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "num_boost_round",
                "type": "int",
                "default": 5000,
                "min": 1,
                "max": 700000,
                "scope": "algorithm",
            },
            {
                "name": "early_stopping_rounds",
                "type": "int",
                "default": 30,
                "min": 1,
                "max": 5000,
                "scope": "algorithm",
            },
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.3,
                "min": 1e-20,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "gamma", "type": "float", "default": 0, "min": 0, "scope": "algorithm"},
            {
                "name": "min_child_weight",
                "type": "float",
                "default": 1,
                "min": 0,
                "scope": "algorithm",
            },
            {"name": "max_depth", "type": "int", "default": 6, "min": 1, "scope": "algorithm"},
            {
                "name": "subsample",
                "type": "float",
                "default": 1,
                "min": 1e-20,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "colsample_bytree",
                "type": "float",
                "default": 1,
                "min": 1e-20,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "reg_lambda",
                "type": "float",
                "default": 1,
                "min": 0,
                "max": 200,
                "scope": "algorithm",
            },
            {
                "name": "reg_alpha",
                "type": "float",
                "default": 0,
                "min": 0,
                "max": 200,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/training/xgboost-classification/v1.3.1/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework": "xgboost",
            "framework_version": "1.7-1",
            "py_version": "py3",
        },
        "training_artifact_key": "xgboost-training/train-xgboost-classification-model.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.m5.xlarge",
        "supported_inference_instance_types": [
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.c4.8xlarge",
        ],
        "default_training_instance_type": "ml.m5.4xlarge",
        "supported_training_instance_types": [
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.c4.8xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["text/csv"],
            "supported_accept_types": ["application/json", "application/json;verbose"],
            "default_content_type": "text/csv",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/tabular_multiclass/",
        "validation_supported": True,
        "fine_tuning_supported": False,
        "resource_name_base": "xgb-classification-model",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "xgb_ecr_uri_1": "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-east-1": {
                    "xgb_ecr_uri_1": "651117190479.dkr.ecr.ap-east-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-northeast-1": {
                    "xgb_ecr_uri_1": "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-northeast-2": {
                    "xgb_ecr_uri_1": "366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-northeast-3": {
                    "xgb_ecr_uri_1": "867004704886.dkr.ecr.ap-northeast-3.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-south-1": {
                    "xgb_ecr_uri_1": "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-southeast-1": {
                    "xgb_ecr_uri_1": "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-southeast-2": {
                    "xgb_ecr_uri_1": "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-southeast-3": {
                    "xgb_ecr_uri_1": "951798379941.dkr.ecr.ap-southeast-3.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ca-central-1": {
                    "xgb_ecr_uri_1": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "cn-north-1": {
                    "xgb_ecr_uri_1": "450853457545.dkr.ecr.cn-north-1.amazonaws.com.cn/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "cn-northwest-1": {
                    "xgb_ecr_uri_1": "451049120500.dkr.ecr.cn-northwest-1.amazonaws.com.cn/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-central-1": {
                    "xgb_ecr_uri_1": "492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-central-2": {
                    "xgb_ecr_uri_1": "680994064768.dkr.ecr.eu-central-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-north-1": {
                    "xgb_ecr_uri_1": "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-south-1": {
                    "xgb_ecr_uri_1": "978288397137.dkr.ecr.eu-south-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-west-1": {
                    "xgb_ecr_uri_1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-west-2": {
                    "xgb_ecr_uri_1": "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-west-3": {
                    "xgb_ecr_uri_1": "659782779980.dkr.ecr.eu-west-3.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "il-central-1": {
                    "xgb_ecr_uri_1": "898809789911.dkr.ecr.il-central-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "me-central-1": {
                    "xgb_ecr_uri_1": "272398656194.dkr.ecr.me-central-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "me-south-1": {
                    "xgb_ecr_uri_1": "801668240914.dkr.ecr.me-south-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "sa-east-1": {
                    "xgb_ecr_uri_1": "737474898029.dkr.ecr.sa-east-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-east-1": {
                    "xgb_ecr_uri_1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-east-2": {
                    "xgb_ecr_uri_1": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-gov-east-1": {
                    "xgb_ecr_uri_1": "237065988967.dkr.ecr.us-gov-east-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-gov-west-1": {
                    "xgb_ecr_uri_1": "414596584902.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-west-1": {
                    "xgb_ecr_uri_1": "746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-west-2": {
                    "xgb_ecr_uri_1": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "inf1": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "inf2": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "xgb_ecr_uri_1": "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-east-1": {
                    "xgb_ecr_uri_1": "651117190479.dkr.ecr.ap-east-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-northeast-1": {
                    "xgb_ecr_uri_1": "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-northeast-2": {
                    "xgb_ecr_uri_1": "366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-northeast-3": {
                    "xgb_ecr_uri_1": "867004704886.dkr.ecr.ap-northeast-3.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-south-1": {
                    "xgb_ecr_uri_1": "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-southeast-1": {
                    "xgb_ecr_uri_1": "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-southeast-2": {
                    "xgb_ecr_uri_1": "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ap-southeast-3": {
                    "xgb_ecr_uri_1": "951798379941.dkr.ecr.ap-southeast-3.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "ca-central-1": {
                    "xgb_ecr_uri_1": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "cn-north-1": {
                    "xgb_ecr_uri_1": "450853457545.dkr.ecr.cn-north-1.amazonaws.com.cn/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "cn-northwest-1": {
                    "xgb_ecr_uri_1": "451049120500.dkr.ecr.cn-northwest-1.amazonaws.com.cn/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-central-1": {
                    "xgb_ecr_uri_1": "492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-central-2": {
                    "xgb_ecr_uri_1": "680994064768.dkr.ecr.eu-central-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-north-1": {
                    "xgb_ecr_uri_1": "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-south-1": {
                    "xgb_ecr_uri_1": "978288397137.dkr.ecr.eu-south-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-west-1": {
                    "xgb_ecr_uri_1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-west-2": {
                    "xgb_ecr_uri_1": "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "eu-west-3": {
                    "xgb_ecr_uri_1": "659782779980.dkr.ecr.eu-west-3.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "il-central-1": {
                    "xgb_ecr_uri_1": "898809789911.dkr.ecr.il-central-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "me-central-1": {
                    "xgb_ecr_uri_1": "272398656194.dkr.ecr.me-central-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "me-south-1": {
                    "xgb_ecr_uri_1": "801668240914.dkr.ecr.me-south-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "sa-east-1": {
                    "xgb_ecr_uri_1": "737474898029.dkr.ecr.sa-east-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-east-1": {
                    "xgb_ecr_uri_1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-east-2": {
                    "xgb_ecr_uri_1": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-gov-east-1": {
                    "xgb_ecr_uri_1": "237065988967.dkr.ecr.us-gov-east-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-gov-west-1": {
                    "xgb_ecr_uri_1": "414596584902.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-west-1": {
                    "xgb_ecr_uri_1": "746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
                "us-west-2": {
                    "xgb_ecr_uri_1": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost@sha256:ba417ec6d8d3e0c6b5f463bc9202e3b498b42260a29b61875f34beb6d99d8444"
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c6i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "c7i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "g4dn": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "g5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "g6": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "g6e": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "inf1": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "inf2": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "local_gpu": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m6i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p2": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p3": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p3dn": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p4d": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p4de": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "p5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r6i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$xgb_ecr_uri_1"}},
            },
        },
        "dynamic_container_deployment_supported": False,
    },
    "sklearn-classification-linear": {
        "model_id": "sklearn-classification-linear",
        "url": "https://scikit-learn.org/stable/",
        "version": "1.3.1",
        "min_sdk_version": "2.188.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "sklearn",
            "framework_version": "1.2-1",
            "py_version": "py3",
        },
        "hosting_artifact_key": "sklearn-infer/infer-sklearn-classification-linear.tar.gz",
        "hosting_script_key": "source-directory-tarballs/sklearn/inference/classification/v1.1.0/sourcedir.tar.gz",
        "hosting_use_script_uri": True,
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "asn1crypto==1.5.1",
            "attrs==23.1.0",
            "boto3==1.26.158",
            "botocore==1.29.159",
            "certifi==2023.5.7",
            "cffi==1.15.1",
            "charset-normalizer==2.1.1",
            "cloudpickle==2.2.1",
            "contextlib2==21.6.0",
            "cryptography==40.0.2",
            "dill==0.3.6",
            "filelock==3.12.2",
            "google-pasta==0.2.0",
            "idna==3.4",
            "importlib-metadata==4.13.0",
            "importlib-resources==5.12.0",
            "jmespath==1.0.1",
            "jsonschema==4.17.3",
            "multiprocess==0.70.14",
            "numpy==2.0",
            "oscrypto==1.3.0",
            "packaging==23.1",
            "pandas==2.2.3",
            "pathos==0.3.0",
            "pkgutil-resolve-name==1.3.10",
            "platformdirs==3.8.0",
            "pox==0.3.2",
            "ppft==1.7.6.6",
            "protobuf3-to-dict==0.1.5",
            "protobuf==3.20.3",
            "pycparser==2.21",
            "pycryptodomex==3.12.0",
            "pyjwt==2.7.0",
            "pyopenssl==23.2.0",
            "pyrsistent==0.19.3",
            "python-dateutil==2.8.2",
            "pytz==2023.3",
            "pyyaml==6.0",
            "requests==2.31.0",
            "s3transfer==0.6.1",
            "sagemaker==2.164.0",
            "sagemaker_jumpstart_script_utilities==1.0.1",
            "sagemaker_jumpstart_snowflake_script_utilities==1.1.0",
            "schema==0.7.5",
            "six==1.16.0",
            "smdebug-rulesconfig==1.0.1",
            "snowflake-connector-python==3.12.3",
            "tblib==1.7.0",
            "typing-extensions==4.6.3",
            "tzdata==2023.3",
            "urllib3==1.26.16",
            "zipp==3.15.0",
        ],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
            {
                "name": "tol",
                "type": "float",
                "default": 0.0001,
                "min": 1e-20,
                "max": 50,
                "scope": "algorithm",
            },
            {
                "name": "penalty",
                "type": "text",
                "default": "l2",
                "options": ["l1", "l2", "elasticnet", "none"],
                "scope": "algorithm",
            },
            {
                "name": "alpha",
                "type": "float",
                "default": 0.0001,
                "min": 1e-20,
                "max": 999,
                "scope": "algorithm",
            },
            {
                "name": "l1_ratio",
                "type": "float",
                "default": 0.15,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            },
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            },
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            },
        ],
        "training_script_key": "source-directory-tarballs/training/sklearn-classification/v2.0.1/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework": "sklearn",
            "framework_version": "1.2-1",
            "py_version": "py3",
        },
        "training_artifact_key": "sklearn-training/train-sklearn-classification-linear.tar.gz",
        "inference_environment_variables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "metrics": [],
        "default_inference_instance_type": "ml.m5.xlarge",
        "supported_inference_instance_types": [
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.c4.8xlarge",
        ],
        "default_training_instance_type": "ml.m5.4xlarge",
        "supported_training_instance_types": [
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.c4.8xlarge",
        ],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["text/csv"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "text/csv",
            "default_accept_type": "application/json",
        },
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/tabular_multiclass/",
        "validation_supported": True,
        "fine_tuning_supported": False,
        "resource_name_base": "sklearn-classification-linear",
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "scikit_ecr_uri_1": "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "ap-east-1": {
                    "cpu_ecr_uri_2": "651117190479.dkr.ecr.ap-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "651117190479.dkr.ecr.ap-east-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_2": "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_2": "366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_2": "867004704886.dkr.ecr.ap-northeast-3.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "867004704886.dkr.ecr.ap-northeast-3.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_2": "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_2": "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_2": "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-southeast-3": {
                    "scikit_ecr_uri_1": "951798379941.dkr.ecr.ap-southeast-3.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "ca-central-1": {
                    "cpu_ecr_uri_2": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "cn-north-1": {
                    "scikit_ecr_uri_1": "450853457545.dkr.ecr.cn-north-1.amazonaws.com.cn/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "cn-northwest-1": {
                    "scikit_ecr_uri_1": "451049120500.dkr.ecr.cn-northwest-1.amazonaws.com.cn/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "eu-central-1": {
                    "cpu_ecr_uri_2": "492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_2": "680994064768.dkr.ecr.eu-central-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "680994064768.dkr.ecr.eu-central-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_2": "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_2": "978288397137.dkr.ecr.eu-south-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "978288397137.dkr.ecr.eu-south-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_2": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_2": "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_2": "659782779980.dkr.ecr.eu-west-3.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "659782779980.dkr.ecr.eu-west-3.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "il-central-1": {
                    "cpu_ecr_uri_2": "898809789911.dkr.ecr.il-central-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "898809789911.dkr.ecr.il-central-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "me-central-1": {
                    "cpu_ecr_uri_2": "272398656194.dkr.ecr.me-central-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "272398656194.dkr.ecr.me-central-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "me-south-1": {
                    "cpu_ecr_uri_2": "801668240914.dkr.ecr.me-south-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "801668240914.dkr.ecr.me-south-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_2": "737474898029.dkr.ecr.sa-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "737474898029.dkr.ecr.sa-east-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-east-1": {
                    "cpu_ecr_uri_2": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-east-2": {
                    "cpu_ecr_uri_2": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_2": "237065988967.dkr.ecr.us-gov-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "237065988967.dkr.ecr.us-gov-east-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-gov-west-1": {
                    "scikit_ecr_uri_1": "414596584902.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "us-west-1": {
                    "cpu_ecr_uri_2": "746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-west-2": {
                    "cpu_ecr_uri_2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "c6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "c6gn": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "c6i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c7g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "c7i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "m6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "m6i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "r6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "r6i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
            },
        },
        "training_instance_type_variants": {
            "regional_aliases": {
                "af-south-1": {
                    "scikit_ecr_uri_1": "510948584623.dkr.ecr.af-south-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "ap-east-1": {
                    "cpu_ecr_uri_2": "651117190479.dkr.ecr.ap-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "651117190479.dkr.ecr.ap-east-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-northeast-1": {
                    "cpu_ecr_uri_2": "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "354813040037.dkr.ecr.ap-northeast-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-northeast-2": {
                    "cpu_ecr_uri_2": "366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-northeast-3": {
                    "cpu_ecr_uri_2": "867004704886.dkr.ecr.ap-northeast-3.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "867004704886.dkr.ecr.ap-northeast-3.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-south-1": {
                    "cpu_ecr_uri_2": "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-southeast-1": {
                    "cpu_ecr_uri_2": "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-southeast-2": {
                    "cpu_ecr_uri_2": "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "ap-southeast-3": {
                    "scikit_ecr_uri_1": "951798379941.dkr.ecr.ap-southeast-3.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "ca-central-1": {
                    "cpu_ecr_uri_2": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "341280168497.dkr.ecr.ca-central-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "cn-north-1": {
                    "scikit_ecr_uri_1": "450853457545.dkr.ecr.cn-north-1.amazonaws.com.cn/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "cn-northwest-1": {
                    "scikit_ecr_uri_1": "451049120500.dkr.ecr.cn-northwest-1.amazonaws.com.cn/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "eu-central-1": {
                    "cpu_ecr_uri_2": "492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-central-2": {
                    "cpu_ecr_uri_2": "680994064768.dkr.ecr.eu-central-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "680994064768.dkr.ecr.eu-central-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-north-1": {
                    "cpu_ecr_uri_2": "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-south-1": {
                    "cpu_ecr_uri_2": "978288397137.dkr.ecr.eu-south-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "978288397137.dkr.ecr.eu-south-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-west-1": {
                    "cpu_ecr_uri_2": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-west-2": {
                    "cpu_ecr_uri_2": "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "eu-west-3": {
                    "cpu_ecr_uri_2": "659782779980.dkr.ecr.eu-west-3.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "659782779980.dkr.ecr.eu-west-3.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "il-central-1": {
                    "cpu_ecr_uri_2": "898809789911.dkr.ecr.il-central-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "898809789911.dkr.ecr.il-central-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "me-central-1": {
                    "cpu_ecr_uri_2": "272398656194.dkr.ecr.me-central-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "272398656194.dkr.ecr.me-central-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "me-south-1": {
                    "cpu_ecr_uri_2": "801668240914.dkr.ecr.me-south-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "801668240914.dkr.ecr.me-south-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "sa-east-1": {
                    "cpu_ecr_uri_2": "737474898029.dkr.ecr.sa-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "737474898029.dkr.ecr.sa-east-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-east-1": {
                    "cpu_ecr_uri_2": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-east-2": {
                    "cpu_ecr_uri_2": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-gov-east-1": {
                    "cpu_ecr_uri_2": "237065988967.dkr.ecr.us-gov-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "237065988967.dkr.ecr.us-gov-east-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-gov-west-1": {
                    "scikit_ecr_uri_1": "414596584902.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95"
                },
                "us-west-1": {
                    "cpu_ecr_uri_2": "746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "746614075791.dkr.ecr.us-west-1.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
                "us-west-2": {
                    "cpu_ecr_uri_2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1-arm64-cpu-py3",
                    "scikit_ecr_uri_1": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn@sha256:e09bbb7686077a1db23d316b699020a786a6e1636b2b89384be9651368c40f95",
                },
            },
            "variants": {
                "c4": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c5": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c5d": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c5n": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "c6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "c6gn": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "c6i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c6id": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "c7g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "c7i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "local": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m4": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m5": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m5d": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "m6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "m6i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m6id": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "m7i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r5": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r5d": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r6g": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "r6gd": {"regional_properties": {"image_uri": "$cpu_ecr_uri_2"}},
                "r6i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r6id": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "r7i": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "t2": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
                "t3": {"regional_properties": {"image_uri": "$scikit_ecr_uri_1"}},
            },
        },
        "dynamic_container_deployment_supported": False,
    },
}

BASE_SPEC = {
    "hosting_resource_requirements": {"num_accelerators": 1, "min_memory_mb": 34360},
    "inference_volume_size": 123,
    "training_volume_size": 456,
    "dynamic_container_deployment_supported": True,
    "model_id": "pytorch-ic-mobilenet-v2",
    "url": "https://pytorch.org/hub/pytorch_vision_mobilenet_v2/",
    "version": "3.0.6",
    "min_sdk_version": "2.189.0",
    "incremental_training_supported": True,
    "hosting_ecr_specs": {
        "framework": "pytorch",
        "framework_version": "1.10.0",
        "py_version": "py38",
    },
    "hosting_artifact_uri": None,
    "hosting_artifact_key": "pytorch-ic/pytorch-ic-mobilenet-v2/artifacts/inference/v2.0.0/",
    "hosting_script_key": "source-directory-tarballs/pytorch/inference/ic/v2.0.0/sourcedir.tar.gz",
    "training_supported": True,
    "training_ecr_specs": {
        "framework": "pytorch",
        "framework_version": "1.10.0",
        "py_version": "py38",
    },
    "training_artifact_key": "pytorch-training/v2.0.0/train-pytorch-ic-mobilenet-v2.tar.gz",
    "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v2.3.0/sourcedir.tar.gz",
    "hyperparameters": [
        {
            "name": "train_only_top_layer",
            "type": "text",
            "options": ["True", "False"],
            "default": "True",
            "scope": "algorithm",
        },
        {
            "name": "epochs",
            "type": "int",
            "default": 5,
            "scope": "algorithm",
            "min": 1,
            "max": 1000,
        },
        {
            "name": "learning_rate",
            "type": "float",
            "default": 0.001,
            "scope": "algorithm",
            "min": 1e-08,
            "max": 1,
        },
        {
            "name": "batch_size",
            "type": "int",
            "default": 4,
            "scope": "algorithm",
            "min": 1,
            "max": 1024,
        },
        {
            "name": "reinitialize_top_layer",
            "type": "text",
            "options": ["Auto", "True", "False"],
            "default": "Auto",
            "scope": "algorithm",
        },
        {
            "name": "sagemaker_submit_directory",
            "type": "text",
            "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
            "scope": "container",
        },
        {
            "name": "sagemaker_program",
            "type": "text",
            "default": "transfer_learning.py",
            "scope": "container",
        },
        {
            "name": "sagemaker_container_log_level",
            "type": "text",
            "default": "20",
            "scope": "container",
        },
    ],
    "inference_environment_variables": [
        {
            "name": "SAGEMAKER_PROGRAM",
            "type": "text",
            "default": "inference.py",
            "scope": "container",
            "required_for_model_class": True,
        },
        {
            "name": "SAGEMAKER_SUBMIT_DIRECTORY",
            "type": "text",
            "default": "/opt/ml/model/code",
            "scope": "container",
            "required_for_model_class": False,
        },
        {
            "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
            "type": "text",
            "default": "20",
            "scope": "container",
            "required_for_model_class": False,
        },
        {
            "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
            "type": "text",
            "default": "3600",
            "scope": "container",
            "required_for_model_class": False,
        },
        {
            "name": "ENDPOINT_SERVER_TIMEOUT",
            "type": "int",
            "default": 3600,
            "scope": "container",
            "required_for_model_class": True,
        },
        {
            "name": "MODEL_CACHE_ROOT",
            "type": "text",
            "default": "/opt/ml/model",
            "scope": "container",
            "required_for_model_class": True,
        },
        {
            "name": "SAGEMAKER_ENV",
            "type": "text",
            "default": "1",
            "scope": "container",
            "required_for_model_class": True,
        },
        {
            "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
            "type": "int",
            "default": 1,
            "scope": "container",
            "required_for_model_class": True,
        },
    ],
    "inference_vulnerable": False,
    "inference_dependencies": [],
    "inference_vulnerabilities": [],
    "training_vulnerable": False,
    "training_dependencies": ["sagemaker_jumpstart_prepack_script_utilities==1.0.0"],
    "training_vulnerabilities": [],
    "deprecated": False,
    "usage_info_message": None,
    "deprecated_message": None,
    "deprecate_warn_message": None,
    "default_inference_instance_type": "ml.m5.large",
    "supported_inference_instance_types": [
        "ml.m5.large",
        "ml.m5.xlarge",
        "ml.c5.xlarge",
        "ml.c5.2xlarge",
        "ml.m4.large",
        "ml.m4.xlarge",
    ],
    "default_training_instance_type": "ml.m5.xlarge",
    "supported_training_instance_types": ["ml.m5.xlarge", "ml.c5.2xlarge", "ml.m4.xlarge"],
    "metrics": [{"Name": "pytorch-ic:val-accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"}],
    "training_prepacked_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/prepack/v1.1.0/sourcedir.tar.gz",
    "hosting_prepacked_artifact_key": "pytorch-ic/pytorch-ic-mobilenet-v2/artifacts/inference-prepack/v1.0.0/",
    "model_kwargs": {},
    "deploy_kwargs": {},
    "estimator_kwargs": {"encrypt_inter_container_traffic": True, "max_run": 360000},
    "fit_kwargs": {},
    "predictor_specs": {
        "default_content_type": "application/x-image",
        "supported_content_types": ["application/x-image"],
        "default_accept_type": "application/json",
        "supported_accept_types": ["application/json;verbose", "application/json"],
    },
    "inference_enable_network_isolation": True,
    "training_enable_network_isolation": True,
    "resource_name_base": "pt-ic-mobilenet-v2",
    "hosting_eula_key": None,
    "hosting_model_package_arns": {},
    "training_model_package_artifact_uris": None,
    "hosting_use_script_uri": False,
    "hosting_instance_type_variants": {
        "regional_aliases": {
            "af-south-1": {
                "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-east-1": {
                "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-northeast-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-northeast-2": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-northeast-3": {
                "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-south-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-southeast-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-southeast-2": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-southeast-3": {
                "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ap-southeast-5": {
                "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "ca-central-1": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "cn-north-1": {
                "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-inference:1.10.0-gpu-py38",
            },
            "cn-northwest-1": {
                "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-inference:1.10.0-gpu-py38",
            },
            "eu-central-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "eu-central-2": {
                "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "eu-north-1": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "eu-south-1": {
                "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "eu-west-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "eu-west-2": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "eu-west-3": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "il-central-1": {
                "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "me-central-1": {
                "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "me-south-1": {
                "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "sa-east-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "us-east-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "us-east-2": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "us-gov-east-1": {
                "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "us-gov-west-1": {
                "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "us-west-1": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
            "us-west-2": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.10.0-gpu-py38",
            },
        },
        "aliases": None,
        "variants": {
            "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "trn1": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
            "trn1n": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
        },
    },
    "training_instance_type_variants": {
        "regional_aliases": {
            "af-south-1": {
                "cpu_ecr_uri_1": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "626614931356.dkr.ecr.af-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-east-1": {
                "cpu_ecr_uri_1": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "871362719292.dkr.ecr.ap-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-northeast-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-northeast-2": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-northeast-3": {
                "cpu_ecr_uri_1": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-south-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-southeast-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-southeast-2": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-southeast-3": {
                "cpu_ecr_uri_1": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ap-southeast-5": {
                "cpu_ecr_uri_1": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "ca-central-1": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.ca-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "cn-north-1": {
                "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/pytorch-training:1.10.0-gpu-py38",
            },
            "cn-northwest-1": {
                "cpu_ecr_uri_1": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/pytorch-training:1.10.0-gpu-py38",
            },
            "eu-central-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "eu-central-2": {
                "cpu_ecr_uri_1": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "380420809688.dkr.ecr.eu-central-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "eu-north-1": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "eu-south-1": {
                "cpu_ecr_uri_1": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "692866216735.dkr.ecr.eu-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "eu-west-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "eu-west-2": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "eu-west-3": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.eu-west-3.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "il-central-1": {
                "cpu_ecr_uri_1": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "780543022126.dkr.ecr.il-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "me-central-1": {
                "cpu_ecr_uri_1": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "914824155844.dkr.ecr.me-central-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "me-south-1": {
                "cpu_ecr_uri_1": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "217643126080.dkr.ecr.me-south-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "sa-east-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.sa-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "us-east-1": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "us-east-2": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "us-gov-east-1": {
                "cpu_ecr_uri_1": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "446045086412.dkr.ecr.us-gov-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "us-gov-west-1": {
                "cpu_ecr_uri_1": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "442386744353.dkr.ecr.us-gov-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "us-west-1": {
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
            "us-west-2": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.0-gpu-py38",
            },
        },
        "aliases": None,
        "variants": {
            "c4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c5n": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "c7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "g4dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "g5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "g6": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "g6e": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "local": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "local_gpu": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "m4": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "m7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "p2": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p3": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p3dn": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p4d": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p4de": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "p5": {"regional_properties": {"image_uri": "$gpu_ecr_uri_2"}},
            "r5": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "r5d": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "r6i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "r6id": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "r7i": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "t2": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "t3": {"regional_properties": {"image_uri": "$cpu_ecr_uri_1"}},
            "trn1": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
            "trn1n": {"regional_properties": {"image_uri": "$alias_ecr_uri_3"}},
        },
    },
    "default_payloads": None,
    "gated_bucket": False,
    "model_subscription_link": None,
    "hosting_additional_data_sources": None,
    "hosting_neuron_model_id": None,
    "hosting_neuron_model_version": None,
    "inference_configs": None,
    "inference_config_components": None,
    "inference_config_rankings": None,
    "training_configs": None,
    "training_config_components": None,
    "training_config_rankings": None,
}
BASE_HOSTING_ADDITIONAL_DATA_SOURCES = {
    "hosting_additional_data_sources": {
        "speculative_decoding": [
            {
                "channel_name": "speculative_decoding_channel",
                "artifact_version": "version",
                "s3_data_source": {
                    "compression_type": "None",
                    "s3_data_type": "S3Prefix",
                    "s3_uri": "s3://bucket/path1",
                    "hub_access_config": None,
                    "model_access_config": None,
                },
            }
        ],
        "scripts": [
            {
                "channel_name": "scripts_channel",
                "artifact_version": "version",
                "s3_data_source": {
                    "compression_type": "None",
                    "s3_data_type": "S3Prefix",
                    "s3_uri": "s3://bucket/path1",
                    "hub_access_config": None,
                    "model_access_config": None,
                },
            }
        ],
    },
}

BASE_HEADER = {
    "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
    "version": "1.0.0",
    "min_version": "2.49.0",
    "spec_key": "community_models_specs/tensorflow-ic-imagenet"
    "-inception-v3-classification-4/specs_v1.0.0.json",
}

BASE_MANIFEST = [
    {
        "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
        "version": "1.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/tensorflow-ic-imagenet"
        "-inception-v3-classification-4/specs_v1.0.0.json",
    },
    {
        "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
        "version": "2.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/tensorflow-ic-imagenet"
        "-inception-v3-classification-4/specs_v2.0.0.json",
    },
    {
        "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
        "version": "1.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/pytorch-ic-"
        "imagenet-inception-v3-classification-4/specs_v1.0.0.json",
    },
    {
        "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
        "version": "2.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/pytorch-ic-imagenet-"
        "inception-v3-classification-4/specs_v2.0.0.json",
    },
    {
        "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
        "version": "3.0.0",
        "min_version": "4.49.0",
        "spec_key": "community_models_specs/tensorflow-ic-"
        "imagenet-inception-v3-classification-4/specs_v3.0.0.json",
    },
]

BASE_PROPRIETARY_HEADER = {
    "model_id": "ai21-summarization",
    "version": "1.1.003",
    "min_version": "2.0.0",
    "spec_key": "proprietary-models/ai21-summarization/proprietary_specs_1.1.003.json",
    "search_keywords": ["Text2Text", "Generation"],
}

BASE_PROPRIETARY_MANIFEST = [
    {
        "model_id": "ai21-summarization",
        "version": "1.1.003",
        "min_version": "2.0.0",
        "spec_key": "proprietary-models/ai21-summarization/proprietary_specs_1.1.003.json",
        "search_keywords": ["Text2Text", "Generation"],
    },
    {
        "model_id": "lighton-mini-instruct40b",
        "version": "v1.0",
        "min_version": "2.0.0",
        "spec_key": "proprietary-models/lighton-mini-instruct40b/proprietary_specs_v1.0.json",
        "search_keywords": ["Text2Text", "Generation"],
    },
    {
        "model_id": "ai21-paraphrase",
        "version": "1.0.005",
        "min_version": "2.0.0",
        "spec_key": "proprietary-models/ai21-paraphrase/proprietary_specs_1.0.005.json",
        "search_keywords": ["Text2Text", "Generation"],
    },
    {
        "model_id": "ai21-paraphrase",
        "version": "v1.00-rc2-not-valid-version",
        "min_version": "2.0.0",
        "spec_key": "proprietary-models/ai21-paraphrase/proprietary_specs_1.0.005.json",
        "search_keywords": ["Text2Text", "Generation"],
    },
    {
        "model_id": "nc-soft-model-1",
        "version": "v3.0-not-valid-version!",
        "min_version": "2.0.0",
        "spec_key": "proprietary-models/nc-soft-model-1/proprietary_specs_1.0.005.json",
        "search_keywords": ["Text2Text", "Generation"],
    },
]

BASE_PROPRIETARY_SPEC = {
    "model_id": "ai21-jurassic-2-light",
    "version": "2.0.004",
    "min_sdk_version": "2.999.0",
    "listing_id": "prodview-roz6zicyvi666",
    "product_id": "1bd680a0-f29b-479d-91c3-9899743021cf",
    "model_subscription_link": "https://aws.amazon.com/marketplace/ai/procurement?productId=1bd680a0",
    "hosting_notebook_key": "pmm-notebooks/pmm-notebook-ai21-jurassic-2-light.ipynb",
    "deploy_kwargs": {
        "model_data_download_timeout": 3600,
        "container_startup_health_check_timeout": 600,
    },
    "default_payloads": {
        "Shakespeare": {
            "content_type": "application/json",
            "prompt_key": "prompt",
            "output_keys": {"generated_text": "[0].completions[0].data.text"},
            "body": {"prompt": "To be, or", "maxTokens": 1, "temperature": 0},
        }
    },
    "predictor_specs": {
        "supported_content_types": ["application/json"],
        "supported_accept_types": ["application/json"],
        "default_content_type": "application/json",
        "default_accept_type": "application/json",
    },
    "default_inference_instance_type": "ml.p4de.24xlarge",
    "supported_inference_instance_types": ["ml.p4de.24xlarge"],
    "hosting_model_package_arns": {
        "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/j2-light-v2-0-004",
        "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/j2-light-v2-0-004",
        "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/j2-light-v2-0-004",
        "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/j2-light-v2-0-004",
        "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/j2-light-v2-0-004",
        "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/j2-light-v2-0-004",
        "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/j2-light-v2-0-004",
        "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/j2-light-v2-0-004",
        "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:model-package/j2-light-v2-0-004",
        "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:model-package/j2-light-v2-0-004",
        "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/j2-light-v2-0-004",
        "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/j2-light-v2-0-004",
        "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/j2-light-v2-0-004",
        "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/j2-light-v2-0-004",
        "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/j2-light-v2-0-004",
        "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:model-package/j2-light-v2-0-004",
    },
}


INFERENCE_CONFIGS = {
    "inference_configs": {
        "neuron-inference": {
            "benchmark_metrics": {
                "ml.inf2.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
                ]
            },
            "component_names": ["neuron-inference"],
        },
        "neuron-inference-budget": {
            "benchmark_metrics": {
                "ml.inf2.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
                ]
            },
            "component_names": ["neuron-base"],
        },
        "gpu-inference-budget": {
            "benchmark_metrics": {
                "ml.p3.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
                ]
            },
            "component_names": ["gpu-inference-budget"],
        },
        "gpu-inference": {
            "benchmark_metrics": {
                "ml.p3.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
                ]
            },
            "component_names": ["gpu-inference"],
        },
        "gpu-inference-model-package": {
            "benchmark_metrics": {
                "ml.p3.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
                ]
            },
            "component_names": ["gpu-inference-model-package"],
        },
        "gpu-accelerated": {
            "benchmark_metrics": {
                "ml.p3.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
                ]
            },
            "component_names": ["gpu-accelerated"],
        },
    },
    "inference_config_components": {
        "neuron-base": {
            "supported_inference_instance_types": ["ml.inf2.xlarge", "ml.inf2.2xlarge"]
        },
        "neuron-inference": {
            "default_inference_instance_type": "ml.inf2.xlarge",
            "supported_inference_instance_types": ["ml.inf2.xlarge", "ml.inf2.2xlarge"],
            "hosting_ecr_specs": {
                "framework": "huggingface-llm-neuronx",
                "framework_version": "0.0.17",
                "py_version": "py310",
            },
            "hosting_artifact_key": "artifacts/meta-textgeneration-llama-2-7b/neuron-inference/model/",
            "hosting_instance_type_variants": {
                "regional_aliases": {
                    "us-west-2": {
                        "neuron-ecr-uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "pytorch-hosting-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04"
                    }
                },
                "variants": {"inf2": {"regional_properties": {"image_uri": "$neuron-ecr-uri"}}},
            },
        },
        "neuron-budget": {
            "inference_environment_variables": [
                {
                    "name": "SAGEMAKER_PROGRAM",
                    "type": "text",
                    "default": "inference.py",
                    "scope": "container",
                    "required_for_model_class": True,
                }
            ],
        },
        "gpu-inference": {
            "supported_inference_instance_types": ["ml.p2.xlarge", "ml.p3.2xlarge"],
            "hosting_artifact_key": "artifacts/meta-textgeneration-llama-2-7b/gpu-inference/model/",
            "hosting_instance_type_variants": {
                "regional_aliases": {
                    "us-west-2": {
                        "gpu-ecr-uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "huggingface-pytorch-hosting:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
                    }
                },
                "variants": {
                    "p3": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                    "p2": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                },
            },
        },
        "gpu-inference-model-package": {
            "default_inference_instance_type": "ml.p2.xlarge",
            "supported_inference_instance_types": ["ml.p2.xlarge", "ml.p3.2xlarge"],
            "hosting_model_package_arns": {
                "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/ll"
                "ama2-7b-v3-740347e540da35b4ab9f6fc0ab3fed2c"
            },
        },
        "gpu-inference-budget": {
            "supported_inference_instance_types": ["ml.p2.xlarge", "ml.p3.2xlarge"],
            "hosting_artifact_key": "artifacts/meta-textgeneration-llama-2-7b/gpu-inference-budget/model/",
            "hosting_instance_type_variants": {
                "regional_aliases": {
                    "us-west-2": {
                        "gpu-ecr-uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "pytorch-hosting:1.13.1-py310-sdk2.14.1-ubuntu20.04"
                    }
                },
                "variants": {
                    "p2": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                    "p3": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                },
            },
        },
        "gpu-accelerated": {
            "supported_inference_instance_types": ["ml.p2.xlarge", "ml.p3.2xlarge"],
            "hosting_instance_type_variants": {
                "regional_aliases": {
                    "us-west-2": {
                        "gpu-ecr-uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "pytorch-hosting-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04"
                    }
                },
                "variants": {
                    "p2": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                    "p3": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                },
            },
            "hosting_additional_data_sources": {
                "speculative_decoding": [
                    {
                        "channel_name": "draft_model_name",
                        "artifact_version": "1.2.1",
                        "s3_data_source": {
                            "compression_type": "None",
                            "model_access_config": {"accept_eula": False},
                            "s3_data_type": "S3Prefix",
                            "s3_uri": "key/to/draft/model/artifact/",
                        },
                    }
                ],
            },
        },
    },
}

TRAINING_CONFIGS = {
    "training_configs": {
        "neuron-training": {
            "benchmark_metrics": {
                "ml.tr1n1.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
                ],
                "ml.tr1n1.4xlarge": [
                    {"name": "Latency", "value": "50", "unit": "Tokens/S", "concurrency": 1}
                ],
            },
            "component_names": ["neuron-training"],
            "default_inference_config": "neuron-inference",
            "default_incremental_training_config": "neuron-training",
            "supported_inference_configs": ["neuron-inference", "neuron-inference-budget"],
            "supported_incremental_training_configs": ["neuron-training", "neuron-training-budget"],
        },
        "neuron-training-budget": {
            "benchmark_metrics": {
                "ml.tr1n1.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
                ],
                "ml.tr1n1.4xlarge": [
                    {"name": "Latency", "value": "50", "unit": "Tokens/S", "concurrency": 1}
                ],
            },
            "component_names": ["neuron-training-budget"],
            "default_inference_config": "neuron-inference-budget",
            "default_incremental_training_config": "neuron-training-budget",
            "supported_inference_configs": ["neuron-inference", "neuron-inference-budget"],
            "supported_incremental_training_configs": ["neuron-training", "neuron-training-budget"],
        },
        "gpu-training": {
            "benchmark_metrics": {
                "ml.p3.2xlarge": [
                    {"name": "Latency", "value": "200", "unit": "Tokens/S", "concurrency": "1"}
                ],
            },
            "component_names": ["gpu-training"],
            "default_inference_config": "gpu-inference",
            "default_incremental_training_config": "gpu-training",
            "supported_inference_configs": ["gpu-inference", "gpu-inference-budget"],
            "supported_incremental_training_configs": ["gpu-training", "gpu-training-budget"],
        },
        "gpu-training-budget": {
            "benchmark_metrics": {
                "ml.p3.2xlarge": [
                    {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": "1"}
                ]
            },
            "component_names": ["gpu-training-budget"],
            "default_inference_config": "gpu-inference-budget",
            "default_incremental_training_config": "gpu-training-budget",
            "supported_inference_configs": ["gpu-inference", "gpu-inference-budget"],
            "supported_incremental_training_configs": ["gpu-training", "gpu-training-budget"],
        },
    },
    "training_config_components": {
        "neuron-training": {
            "default_training_instance_type": "ml.trn1.2xlarge",
            "supported_training_instance_types": ["ml.trn1.xlarge", "ml.trn1.2xlarge"],
            "training_artifact_key": "artifacts/meta-textgeneration-llama-2-7b/neuron-training/model/",
            "training_ecr_specs": {
                "framework": "huggingface",
                "framework_version": "2.0.0",
                "py_version": "py310",
                "huggingface_transformers_version": "4.28.1",
            },
            "training_instance_type_variants": {
                "regional_aliases": {
                    "us-west-2": {
                        "neuron-ecr-uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04"
                    }
                },
                "variants": {"trn1": {"regional_properties": {"image_uri": "$neuron-ecr-uri"}}},
            },
        },
        "gpu-training": {
            "default_training_instance_type": "ml.p2.xlarge",
            "supported_training_instance_types": ["ml.p2.xlarge", "ml.p3.2xlarge"],
            "training_artifact_key": "artifacts/meta-textgeneration-llama-2-7b/gpu-training/model/",
            "training_instance_type_variants": {
                "regional_aliases": {
                    "us-west-2": {
                        "gpu-ecr-uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "huggingface-pytorch-training:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04"
                    }
                },
                "variants": {
                    "p2": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                    "p3": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                },
            },
        },
        "neuron-training-budget": {
            "default_training_instance_type": "ml.trn1.2xlarge",
            "supported_training_instance_types": ["ml.trn1.xlarge", "ml.trn1.2xlarge"],
            "training_artifact_key": "artifacts/meta-textgeneration-llama-2-7b/neuron-training-budget/model/",
            "training_instance_type_variants": {
                "regional_aliases": {
                    "us-west-2": {
                        "neuron-ecr-uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04"
                    }
                },
                "variants": {"trn1": {"regional_properties": {"image_uri": "$neuron-ecr-uri"}}},
            },
        },
        "gpu-training-budget": {
            "default_training_instance_type": "ml.p2.xlarge",
            "supported_training_instance_types": ["ml.p2.xlarge", "ml.p3.2xlarge"],
            "training_artifact_key": "artifacts/meta-textgeneration-llama-2-7b/gpu-training-budget/model/",
            "training_instance_type_variants": {
                "regional_aliases": {
                    "us-west-2": {
                        "gpu-ecr-uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                        "pytorch-training:1.13.1-py310-sdk2.14.1-ubuntu20.04"
                    }
                },
                "variants": {
                    "p2": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                    "p3": {"regional_properties": {"image_uri": "$gpu-ecr-uri"}},
                },
            },
        },
    },
}


INFERENCE_CONFIG_RANKINGS = {
    "inference_config_rankings": {
        "overall": {
            "description": "Overall rankings of configs",
            "rankings": [
                "neuron-inference",
                "neuron-inference-budget",
                "gpu-inference",
                "gpu-inference-budget",
                "gpu-accelerated",
            ],
        },
        "performance": {
            "description": "Configs ranked based on performance",
            "rankings": [
                "neuron-inference",
                "gpu-inference",
                "neuron-inference-budget",
                "gpu-inference-budget",
            ],
        },
        "cost": {
            "description": "Configs ranked based on cost",
            "rankings": [
                "neuron-inference-budget",
                "gpu-inference-budget",
                "neuron-inference",
                "gpu-inference",
            ],
        },
    }
}

TRAINING_CONFIG_RANKINGS = {
    "training_config_rankings": {
        "overall": {
            "description": "Overall rankings of configs",
            "rankings": [
                "neuron-training",
                "neuron-training-budget",
                "gpu-training",
                "gpu-training-budget",
            ],
        },
        "performance_training": {
            "description": "Configs ranked based on performance",
            "rankings": [
                "neuron-training",
                "gpu-training",
                "neuron-training-budget",
                "gpu-training-budget",
            ],
            "instance_type_overrides": {
                "ml.p2.xlarge": [
                    "neuron-training",
                    "neuron-training-budget",
                    "gpu-training",
                    "gpu-training-budget",
                ]
            },
        },
        "cost_training": {
            "description": "Configs ranked based on cost",
            "rankings": [
                "neuron-training-budget",
                "gpu-training-budget",
                "neuron-training",
                "gpu-training",
            ],
        },
    }
}


DEPLOYMENT_CONFIGS = [
    {
        "DeploymentConfigName": "neuron-inference",
        "DeploymentArgs": {
            "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
            ".0-gpu-py310-cu121-ubuntu20.04",
            "ModelData": {
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-cache-alpha-us-west-2/huggingface-textgeneration/huggingface"
                    "-textgeneration-bloom-1b1/artifacts/inference-prepack/v4.0.0/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "SM_NUM_GPUS": "1",
                "MAX_INPUT_LENGTH": "2047",
                "MAX_TOTAL_TOKENS": "2048",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            "InstanceType": "ml.p2.xlarge",
            "ComputeResourceRequirements": {"MinMemoryRequiredInMb": None},
            "ModelDataDownloadTimeout": None,
            "ContainerStartupHealthCheckTimeout": None,
        },
        "AccelerationConfigs": None,
        "BenchmarkMetrics": [
            {"name": "Instance Rate", "value": "0.0083000000", "unit": "USD/Hrs", "concurrency": 1}
        ],
    },
    {
        "DeploymentConfigName": "neuron-inference-budget",
        "DeploymentArgs": {
            "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
            ".0-gpu-py310-cu121-ubuntu20.04",
            "ModelData": {
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-cache-alpha-us-west-2/huggingface-textgeneration/huggingface"
                    "-textgeneration-bloom-1b1/artifacts/inference-prepack/v4.0.0/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "SM_NUM_GPUS": "1",
                "MAX_INPUT_LENGTH": "2047",
                "MAX_TOTAL_TOKENS": "2048",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            "InstanceType": "ml.p2.xlarge",
            "ComputeResourceRequirements": {"MinMemoryRequiredInMb": None},
            "ModelDataDownloadTimeout": None,
            "ContainerStartupHealthCheckTimeout": None,
        },
        "AccelerationConfigs": None,
        "BenchmarkMetrics": [
            {"name": "Instance Rate", "value": "0.0083000000", "unit": "USD/Hrs", "concurrency": 1}
        ],
    },
    {
        "DeploymentConfigName": "gpu-inference-budget",
        "DeploymentArgs": {
            "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
            ".0-gpu-py310-cu121-ubuntu20.04",
            "ModelData": {
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-cache-alpha-us-west-2/huggingface-textgeneration/huggingface"
                    "-textgeneration-bloom-1b1/artifacts/inference-prepack/v4.0.0/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "SM_NUM_GPUS": "1",
                "MAX_INPUT_LENGTH": "2047",
                "MAX_TOTAL_TOKENS": "2048",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            "InstanceType": "ml.p2.xlarge",
            "ComputeResourceRequirements": {"MinMemoryRequiredInMb": None},
            "ModelDataDownloadTimeout": None,
            "ContainerStartupHealthCheckTimeout": None,
        },
        "AccelerationConfigs": None,
        "BenchmarkMetrics": [
            {"name": "Instance Rate", "value": "0.0083000000", "unit": "USD/Hrs", "concurrency": 1}
        ],
    },
    {
        "DeploymentConfigName": "gpu-inference",
        "DeploymentArgs": {
            "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
            ".0-gpu-py310-cu121-ubuntu20.04",
            "ModelData": {
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-cache-alpha-us-west-2/huggingface-textgeneration/huggingface"
                    "-textgeneration-bloom-1b1/artifacts/inference-prepack/v4.0.0/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "SM_NUM_GPUS": "1",
                "MAX_INPUT_LENGTH": "2047",
                "MAX_TOTAL_TOKENS": "2048",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            "InstanceType": "ml.p2.xlarge",
            "ComputeResourceRequirements": {"MinMemoryRequiredInMb": None},
            "ModelDataDownloadTimeout": None,
            "ContainerStartupHealthCheckTimeout": None,
        },
        "AccelerationConfigs": None,
        "BenchmarkMetrics": [{"name": "Instance Rate", "value": "0.0083000000", "unit": "USD/Hrs"}],
    },
]


INIT_KWARGS = {
    "image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu"
    "-py310-cu121-ubuntu20.04",
    "model_data": {
        "S3DataSource": {
            "S3Uri": "s3://jumpstart-cache-alpha-us-west-2/huggingface-textgeneration/huggingface-textgeneration"
            "-bloom-1b1/artifacts/inference-prepack/v4.0.0/",
            "S3DataType": "S3Prefix",
            "CompressionType": "None",
        }
    },
    "instance_type": "ml.p2.xlarge",
    "env": {
        "SAGEMAKER_PROGRAM": "inference.py",
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_ENV": "1",
        "HF_MODEL_ID": "/opt/ml/model",
        "SM_NUM_GPUS": "1",
        "MAX_INPUT_LENGTH": "2047",
        "MAX_TOTAL_TOKENS": "2048",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
    },
    "role": "arn:aws:iam::312206380606:role/service-role/AmazonSageMaker-ExecutionRole-20230707T131628",
    "name": "hf-textgeneration-bloom-1b1-2024-04-22-20-23-48-799",
    "enable_network_isolation": True,
}

HUB_MODEL_DOCUMENT_DICTS = {
    "huggingface-llm-gemma-2b-instruct": {
        "Url": "https://huggingface.co/google/gemma-2b-it",
        "MinSdkVersion": "2.189.0",
        "TrainingSupported": True,
        "IncrementalTrainingSupported": False,
        "HostingEcrUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04",  # noqa: E501
        "HostingArtifactS3DataType": "S3Prefix",
        "HostingArtifactCompressionType": "None",
        "HostingArtifactUri": "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/huggingface-llm-gemma-2b-instruct/artifacts/inference/v1.0.0/",  # noqa: E501
        "HostingScriptUri": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/inference/llm/v1.0.1/sourcedir.tar.gz",  # noqa: E501
        "HostingPrepackedArtifactUri": "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/huggingface-llm-gemma-2b-instruct/artifacts/inference-prepack/v1.0.0/",  # noqa: E501
        "HostingPrepackedArtifactVersion": "1.0.0",
        "HostingUseScriptUri": False,
        "HostingEulaUri": "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/fmhMetadata/terms/gemmaTerms.txt",
        "TrainingScriptUri": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/transfer_learning/llm/v1.1.1/sourcedir.tar.gz",  # noqa: E501
        "TrainingPrepackedScriptUri": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/transfer_learning/llm/prepack/v1.1.1/sourcedir.tar.gz",  # noqa: E501
        "TrainingPrepackedScriptVersion": "1.1.1",
        "TrainingEcrUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04",  # noqa: E501
        "TrainingArtifactS3DataType": "S3Prefix",
        "TrainingArtifactCompressionType": "None",
        "TrainingArtifactUri": "s3://jumpstart-cache-prod-us-west-2/huggingface-training/train-huggingface-llm-gemma-2b-instruct.tar.gz",  # noqa: E501
        "ModelTypes": ["OPEN_WEIGHTS", "PROPRIETARY"],
        "Hyperparameters": [
            {
                "Name": "peft_type",
                "Type": "text",
                "Default": "lora",
                "Options": ["lora", "None"],
                "Scope": "algorithm",
            },
            {
                "Name": "instruction_tuned",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "chat_dataset",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "epoch",
                "Type": "int",
                "Default": 1,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {
                "Name": "learning_rate",
                "Type": "float",
                "Default": 0.0001,
                "Min": 1e-08,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "lora_r",
                "Type": "int",
                "Default": 64,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {"Name": "lora_alpha", "Type": "int", "Default": 16, "Min": 0, "Scope": "algorithm"},
            {
                "Name": "lora_dropout",
                "Type": "float",
                "Default": 0,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            {"Name": "bits", "Type": "int", "Default": 4, "Scope": "algorithm"},
            {
                "Name": "double_quant",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "quant_Type",
                "Type": "text",
                "Default": "nf4",
                "Options": ["fp4", "nf4"],
                "Scope": "algorithm",
            },
            {
                "Name": "per_device_train_batch_size",
                "Type": "int",
                "Default": 1,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {
                "Name": "per_device_eval_batch_size",
                "Type": "int",
                "Default": 2,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {
                "Name": "warmup_ratio",
                "Type": "float",
                "Default": 0.1,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "train_from_scratch",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "fp16",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "bf16",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "evaluation_strategy",
                "Type": "text",
                "Default": "steps",
                "Options": ["steps", "epoch", "no"],
                "Scope": "algorithm",
            },
            {
                "Name": "eval_steps",
                "Type": "int",
                "Default": 20,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {
                "Name": "gradient_accumulation_steps",
                "Type": "int",
                "Default": 4,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {
                "Name": "logging_steps",
                "Type": "int",
                "Default": 8,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {
                "Name": "weight_decay",
                "Type": "float",
                "Default": 0.2,
                "Min": 1e-08,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "load_best_model_at_end",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "max_train_samples",
                "Type": "int",
                "Default": -1,
                "Min": -1,
                "Scope": "algorithm",
            },
            {
                "Name": "max_val_samples",
                "Type": "int",
                "Default": -1,
                "Min": -1,
                "Scope": "algorithm",
            },
            {
                "Name": "seed",
                "Type": "int",
                "Default": 10,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {
                "Name": "max_input_length",
                "Type": "int",
                "Default": 1024,
                "Min": -1,
                "Scope": "algorithm",
            },
            {
                "Name": "validation_split_ratio",
                "Type": "float",
                "Default": 0.2,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "train_data_split_seed",
                "Type": "int",
                "Default": 0,
                "Min": 0,
                "Scope": "algorithm",
            },
            {
                "Name": "preprocessing_num_workers",
                "Type": "text",
                "Default": "None",
                "Scope": "algorithm",
            },
            {"Name": "max_steps", "Type": "int", "Default": -1, "Scope": "algorithm"},
            {
                "Name": "gradient_checkpointing",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "early_stopping_patience",
                "Type": "int",
                "Default": 3,
                "Min": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "early_stopping_threshold",
                "Type": "float",
                "Default": 0.0,
                "Min": 0,
                "Scope": "algorithm",
            },
            {
                "Name": "adam_beta1",
                "Type": "float",
                "Default": 0.9,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "adam_beta2",
                "Type": "float",
                "Default": 0.999,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "adam_epsilon",
                "Type": "float",
                "Default": 1e-08,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "max_grad_norm",
                "Type": "float",
                "Default": 1.0,
                "Min": 0,
                "Scope": "algorithm",
            },
            {
                "Name": "label_smoothing_factor",
                "Type": "float",
                "Default": 0,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "logging_first_step",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "logging_nan_inf_filter",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "save_strategy",
                "Type": "text",
                "Default": "steps",
                "Options": ["no", "epoch", "steps"],
                "Scope": "algorithm",
            },
            {
                "Name": "save_steps",
                "Type": "int",
                "Default": 500,
                "Min": 1,
                "Scope": "algorithm",
            },  # noqa: E501
            {
                "Name": "save_total_limit",
                "Type": "int",
                "Default": 1,
                "Scope": "algorithm",
            },  # noqa: E501
            {
                "Name": "dataloader_drop_last",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "dataloader_num_workers",
                "Type": "int",
                "Default": 0,
                "Min": 0,
                "Scope": "algorithm",
            },
            {
                "Name": "eval_accumulation_steps",
                "Type": "text",
                "Default": "None",
                "Scope": "algorithm",
            },
            {
                "Name": "auto_find_batch_size",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            {
                "Name": "lr_scheduler_type",
                "Type": "text",
                "Default": "constant_with_warmup",
                "Options": ["constant_with_warmup", "linear"],
                "Scope": "algorithm",
            },
            {
                "Name": "warmup_steps",
                "Type": "int",
                "Default": 0,
                "Min": 0,
                "Scope": "algorithm",
            },  # noqa: E501
            {
                "Name": "deepspeed",
                "Type": "text",
                "Default": "False",
                "Options": ["False"],
                "Scope": "algorithm",
            },
            {
                "Name": "sagemaker_submit_directory",
                "Type": "text",
                "Default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "Scope": "container",
            },
            {
                "Name": "sagemaker_program",
                "Type": "text",
                "Default": "transfer_learning.py",
                "Scope": "container",
            },
            {
                "Name": "sagemaker_container_log_level",
                "Type": "text",
                "Default": "20",
                "Scope": "container",
            },
        ],
        "InferenceEnvironmentVariables": [
            {
                "Name": "SAGEMAKER_PROGRAM",
                "Type": "text",
                "Default": "inference.py",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "Type": "text",
                "Default": "/opt/ml/model/code",
                "Scope": "container",
                "RequiredForModelClass": False,
            },
            {
                "Name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "Type": "text",
                "Default": "20",
                "Scope": "container",
                "RequiredForModelClass": False,
            },
            {
                "Name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "Type": "text",
                "Default": "3600",
                "Scope": "container",
                "RequiredForModelClass": False,
            },
            {
                "Name": "ENDPOINT_SERVER_TIMEOUT",
                "Type": "int",
                "Default": 3600,
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "MODEL_CACHE_ROOT",
                "Type": "text",
                "Default": "/opt/ml/model",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "SAGEMAKER_ENV",
                "Type": "text",
                "Default": "1",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "HF_MODEL_ID",
                "Type": "text",
                "Default": "/opt/ml/model",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "MAX_INPUT_LENGTH",
                "Type": "text",
                "Default": "8191",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "MAX_TOTAL_TOKENS",
                "Type": "text",
                "Default": "8192",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "MAX_BATCH_PREFILL_TOKENS",
                "Type": "text",
                "Default": "8191",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "SM_NUM_GPUS",
                "Type": "text",
                "Default": "1",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            {
                "Name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "Type": "int",
                "Default": 1,
                "Scope": "container",
                "RequiredForModelClass": True,
            },
        ],
        "TrainingMetrics": [
            {
                "Name": "huggingface-textgeneration:eval-loss",
                "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
            },
            {
                "Name": "huggingface-textgeneration:train-loss",
                "Regex": "'loss': ([0-9]+\\.[0-9]+)",
            },  # noqa: E501
        ],
        "InferenceDependencies": [],
        "TrainingDependencies": [
            "accelerate==0.26.1",
            "bitsandbytes==0.42.0",
            "deepspeed==0.10.3",
            "docstring-parser==0.15",
            "flash_attn==2.5.5",
            "ninja==1.11.1",
            "packaging==23.2",
            "peft==0.8.2",
            "py_cpuinfo==9.0.0",
            "rich==13.7.0",
            "safetensors==0.4.2",
            "sagemaker_jumpstart_huggingface_script_utilities==1.2.1",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
            "shtab==1.6.5",
            "tokenizers==0.15.1",
            "transformers==4.38.1",
            "trl==0.7.10",
            "tyro==0.7.2",
        ],
        "DefaultInferenceInstanceType": "ml.g5.xlarge",
        "SupportedInferenceInstanceTypes": [
            "ml.g5.xlarge",
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.16xlarge",
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "DefaultTrainingInstanceType": "ml.g5.2xlarge",
        "SupportedTrainingInstanceTypes": [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.16xlarge",
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
        "SageMakerSdkPredictorSpecifications": {
            "SupportedContentTypes": ["application/json"],
            "SupportedAcceptTypes": ["application/json"],
            "DefaultContentType": "application/json",
            "DefaultAcceptType": "application/json",
        },
        "InferenceVolumeSize": 512,
        "TrainingVolumeSize": 512,
        "InferenceEnableNetworkIsolation": True,
        "TrainingEnableNetworkIsolation": True,
        "FineTuningSupported": True,
        "ValidationSupported": True,
        "DefaultTrainingDatasetUri": "s3://jumpstart-cache-prod-us-west-2/training-datasets/oasst_top/train/",  # noqa: E501
        "ResourceNameBase": "hf-llm-gemma-2b-instruct",
        "DefaultPayloads": {
            "HelloWorld": {
                "ContentType": "application/json",
                "PromptKey": "inputs",
                "OutputKeys": {
                    "GeneratedText": "[0].generated_text",
                    "InputLogprobs": "[0].details.prefill[*].logprob",
                },
                "Body": {
                    "Inputs": "<bos><start_of_turn>user\nWrite a hello world program<end_of_turn>\n<start_of_turn>model",  # noqa: E501
                    "Parameters": {
                        "MaxNewTokens": 256,
                        "DecoderInputDetails": True,
                        "Details": True,
                    },
                },
            },
            "MachineLearningPoem": {
                "ContentType": "application/json",
                "PromptKey": "inputs",
                "OutputKeys": {
                    "GeneratedText": "[0].generated_text",
                    "InputLogprobs": "[0].details.prefill[*].logprob",
                },
                "Body": {
                    "Inputs": "Write me a poem about Machine Learning.",
                    "Parameters": {
                        "MaxNewTokens": 256,
                        "DecoderInputDetails": True,
                        "Details": True,
                    },
                },
            },
        },
        "GatedBucket": True,
        "HostingResourceRequirements": {"MinMemoryMb": 8192, "NumAccelerators": 1},
        "HostingInstanceTypeVariants": {
            "Aliases": {
                "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"  # noqa: E501
            },
            "Variants": {
                "g4dn": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "TrainingInstanceTypeVariants": {
            "Aliases": {
                "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"  # noqa: E501
            },
            "Variants": {
                "g4dn": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/g4dn/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz",  # noqa: E501
                    },
                },
                "g5": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/g5/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz",  # noqa: E501
                    },
                },
                "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/p3dn/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz",  # noqa: E501
                    },
                },
                "p4d": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/p4d/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz",  # noqa: E501
                    },
                },
                "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "InferenceConfigRankings": {
            "overall": {"Description": "default", "Rankings": ["variant1"]}
        },
        "InferenceConfigs": {
            "variant1": {
                "ComponentNames": ["variant1"],
                "BenchmarkMetrics": {
                    "ml.g5.12xlarge": [
                        {"Name": "latency", "Unit": "sec", "Value": "0.19", "Concurrency": "1"},
                    ]
                },
            },
        },
        "InferenceConfigComponents": {
            "variant1": {
                "HostingEcrUri": "123456789012.ecr.us-west-2.amazon.com/repository",
                "HostingArtifactUri": "s3://jumpstart-private-cache-prod-us-west-2/meta-textgeneration/meta-textgeneration-llama-2-7b/artifacts/variant1/v1.0.0/",  # noqa: E501
                "HostingScriptUri": "s3://jumpstart-monarch-test-hub-bucket/monarch-curated-hub-1714579993.88695/curated_models/meta-textgeneration-llama-2-7b/4.0.0/source-directory-tarballs/meta/inference/textgeneration/v1.2.3/sourcedir.tar.gz",  # noqa: E501
                "InferenceDependencies": [],
                "InferenceEnvironmentVariables": [
                    {
                        "Name": "SAGEMAKER_PROGRAM",
                        "Type": "text",
                        "Default": "inference.py",
                        "Scope": "container",
                        "RequiredForModelClass": True,
                    }
                ],
                "HostingAdditionalDataSources": {
                    "speculative_decoding": [
                        {
                            "ArtifactVersion": 1,
                            "ChannelName": "speculative_decoding_channel_1",
                            "S3DataSource": {
                                "CompressionType": "None",
                                "S3DataType": "S3Prefix",
                                "S3Uri": "s3://bucket/path/1",
                            },
                        },
                        {
                            "ArtifactVersion": 1,
                            "ChannelName": "speculative_decoding_channel_2",
                            "S3DataSource": {
                                "CompressionType": "None",
                                "S3DataType": "S3Prefix",
                                "S3Uri": "s3://bucket/path/2",
                            },
                        },
                    ]
                },
                "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "ContextualHelp": {
            "HubFormatTrainData": [
                "A train and an optional validation directories. Each directory contains a CSV/JSON/TXT. ",
                "- For CSV/JSON files, the text data is used from the column called 'text' or the first column if no column called 'text' is found",  # noqa: E501
                "- The number of files under train and validation (if provided) should equal to one, respectively.",
                " [Learn how to setup an AWS S3 bucket.](https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html)",  # noqa: E501
            ],
            "HubDefaultTrainData": [
                "Dataset: [SEC](https://www.sec.gov/edgar/searchedgar/companysearch)",
                "SEC filing contains regulatory documents that companies and issuers of securities must submit to the Securities and Exchange Commission (SEC) on a regular basis.",  # noqa: E501
                "License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)",
            ],
        },
        "ModelDataDownloadTimeout": 1200,
        "ContainerStartupHealthCheckTimeout": 1200,
        "EncryptInterContainerTraffic": True,
        "DisableOutputCompression": True,
        "MaxRuntimeInSeconds": 360000,
        "DynamicContainerDeploymentSupported": True,
        "TrainingModelPackageArtifactUri": None,
        "Dependencies": [],
    },
    "meta-textgeneration-llama-2-70b": {
        "Url": "https://ai.meta.com/resources/models-and-libraries/llama-downloads/",
        "MinSdkVersion": "2.198.0",
        "TrainingSupported": True,
        "IncrementalTrainingSupported": False,
        "HostingEcrUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04",  # noqa: E501
        "HostingArtifactUri": "s3://jumpstart-cache-prod-us-west-2/meta-textgeneration/meta-textgeneration-llama-2-70b/artifacts/inference/v1.0.0/",  # noqa: E501
        "HostingScriptUri": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/meta/inference/textgeneration/v1.2.3/sourcedir.tar.gz",  # noqa: E501
        "HostingPrepackedArtifactUri": "s3://jumpstart-cache-prod-us-west-2/meta-textgeneration/meta-textgeneration-llama-2-70b/artifacts/inference-prepack/v1.0.0/",  # noqa: E501
        "HostingPrepackedArtifactVersion": "1.0.0",
        "HostingUseScriptUri": False,
        "HostingEulaUri": "s3://jumpstart-cache-prod-us-west-2/fmhMetadata/eula/llamaEula.txt",
        "InferenceDependencies": [],
        "TrainingDependencies": [
            "accelerate==0.21.0",
            "bitsandbytes==0.39.1",
            "black==23.7.0",
            "brotli==1.0.9",
            "datasets==2.14.1",
            "fire==0.5.0",
            "huggingface-hub==0.20.3",
            "inflate64==0.3.1",
            "loralib==0.1.1",
            "multivolumefile==0.2.3",
            "mypy-extensions==1.0.0",
            "nvidia-cublas-cu12==12.1.3.1",
            "nvidia-cuda-cupti-cu12==12.1.105",
            "nvidia-cuda-nvrtc-cu12==12.1.105",
            "nvidia-cuda-runtime-cu12==12.1.105",
            "nvidia-cudnn-cu12==8.9.2.26",
            "nvidia-cufft-cu12==11.0.2.54",
            "nvidia-curand-cu12==10.3.2.106",
            "nvidia-cusolver-cu12==11.4.5.107",
            "nvidia-cusolver-cu12==11.4.5.107",
            "nvidia-cusparse-cu12==12.1.0.106",
            "nvidia-nccl-cu12==2.19.3",
            "nvidia-nvjitlink-cu12==12.3.101",
            "nvidia-nvtx-cu12==12.1.105",
            "pathspec==0.11.1",
            "peft==0.4.0",
            "py7zr==0.20.5",
            "pybcj==1.0.1",
            "pycryptodomex==3.18.0",
            "pyppmd==1.0.0",
            "pyzstd==0.15.9",
            "safetensors==0.3.1",
            "sagemaker_jumpstart_huggingface_script_utilities==1.1.4",
            "sagemaker_jumpstart_script_utilities==1.1.9",
            "scipy==1.11.1",
            "termcolor==2.3.0",
            "texttable==1.6.7",
            "tokenize-rt==5.1.0",
            "tokenizers==0.13.3",
            "torch==2.2.0",
            "transformers==4.33.3",
            "triton==2.2.0",
            "typing-extensions==4.8.0",
        ],
        "Hyperparameters": [
            {
                "Name": "epoch",
                "Type": "int",
                "Default": 5,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            {
                "Name": "learning_rate",
                "Type": "float",
                "Default": 0.0001,
                "Min": 1e-08,
                "Max": 1,
                "Scope": "algorithm",
            },
            {
                "Name": "instruction_tuned",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
        ],
        "TrainingScriptUri": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/meta/transfer_learning/textgeneration/v1.0.11/sourcedir.tar.gz",  # noqa: E501
        "TrainingPrepackedScriptUri": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/meta/transfer_learning/textgeneration/prepack/v1.0.5/sourcedir.tar.gz",  # noqa: E501
        "TrainingPrepackedScriptVersion": "1.0.5",
        "TrainingEcrUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04",  # TODO: not a training image  # noqa: E501
        "TrainingArtifactUri": "s3://jumpstart-cache-prod-us-west-2/meta-training/train-meta-textgeneration-llama-2-70b.tar.gz",  # noqa: E501
        "InferenceEnvironmentVariables": [
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            },
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            },
        ],
        "TrainingMetrics": [
            {
                "Name": "huggingface-textgeneration:eval-loss",
                "Regex": "eval_epoch_loss=tensor\\(([0-9\\.]+)",
            },
            {
                "Name": "huggingface-textgeneration:eval-ppl",
                "Regex": "eval_ppl=tensor\\(([0-9\\.]+)",
            },
            {
                "Name": "huggingface-textgeneration:train-loss",
                "Regex": "train_epoch_loss=([0-9\\.]+)",
            },
        ],
        "DefaultInferenceInstanceType": "ml.g5.48xlarge",
        "supported_inference_instance_types": ["ml.g5.48xlarge", "ml.p4d.24xlarge"],
        "default_training_instance_type": "ml.g5.48xlarge",
        "SupportedInferenceInstanceTypes": ["ml.g5.48xlarge", "ml.p4d.24xlarge"],
        "ModelDataDownloadTimeout": 1200,
        "ContainerStartupHealthCheckTimeout": 1200,
        "EncryptInterContainerTraffic": True,
        "DisableOutputCompression": True,
        "MaxRuntimeInSeconds": 360000,
        "SageMakerSdkPredictorSpecifications": {
            "SupportedContentTypes": ["application/json"],
            "SupportedAcceptTypes": ["application/json"],
            "DefaultContentType": "application/json",
            "DefaultAcceptType": "application/json",
        },
        "InferenceVolumeSize": 256,
        "TrainingVolumeSize": 256,
        "InferenceEnableNetworkIsolation": True,
        "TrainingEnableNetworkIsolation": True,
        "DefaultTrainingDatasetUri": "s3://jumpstart-cache-prod-us-west-2/training-datasets/sec_amazon/",  # noqa: E501
        "ValidationSupported": True,
        "FineTuningSupported": True,
        "ResourceNameBase": "meta-textgeneration-llama-2-70b",
        "DefaultPayloads": {
            "meaningOfLife": {
                "ContentType": "application/json",
                "PromptKey": "inputs",
                "OutputKeys": {
                    "generated_text": "[0].generated_text",
                    "input_logprobs": "[0].details.prefill[*].logprob",
                },
                "Body": {
                    "inputs": "I believe the meaning of life is",
                    "parameters": {
                        "max_new_tokens": 64,
                        "top_p": 0.9,
                        "temperature": 0.6,
                        "decoder_input_details": True,
                        "details": True,
                    },
                },
            },
            "theoryOfRelativity": {
                "ContentType": "application/json",
                "PromptKey": "inputs",
                "OutputKeys": {"generated_text": "[0].generated_text"},
                "Body": {
                    "inputs": "Simply put, the theory of relativity states that ",
                    "parameters": {"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6},
                },
            },
        },
        "GatedBucket": True,
        "HostingInstanceTypeVariants": {
            "Aliases": {
                "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"  # noqa: E501
            },
            "Variants": {
                "g4dn": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
                "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
                "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            },
        },
        "TrainingInstanceTypeVariants": {
            "Aliases": {
                "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"  # noqa: E501
            },
            "Variants": {
                "g4dn": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "g5": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "meta-training/g5/v1.0.0/train-meta-textgeneration-llama-2-70b.tar.gz",  # noqa: E501
                    },
                },
                "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p4d": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "meta-training/p4d/v1.0.0/train-meta-textgeneration-llama-2-70b.tar.gz",  # noqa: E501
                    },
                },
                "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        "HostingArtifactS3DataType": "S3Prefix",
        "HostingArtifactCompressionType": "None",
        "HostingResourceRequirements": {"MinMemoryMb": 393216, "NumAccelerators": 8},
        "DynamicContainerDeploymentSupported": True,
        "TrainingModelPackageArtifactUri": None,
        "Task": "text generation",
        "DataType": "text",
        "Framework": "meta",
        "Dependencies": [],
    },
    "huggingface-textembedding-bloom-7b1": {
        "Url": "https://huggingface.co/bigscience/bloom-7b1",
        "MinSdkVersion": "2.144.0",
        "TrainingSupported": False,
        "IncrementalTrainingSupported": False,
        "HostingEcrUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04",  # noqa: E501
        "HostingArtifactUri": "s3://jumpstart-cache-prod-us-west-2/huggingface-infer/infer-huggingface-textembedding-bloom-7b1.tar.gz",  # noqa: E501
        "HostingScriptUri": "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/inference/textembedding/v1.0.1/sourcedir.tar.gz",  # noqa: E501
        "HostingPrepackedArtifactUri": "s3://jumpstart-cache-prod-us-west-2/huggingface-infer/prepack/v1.0.1/infer-prepack-huggingface-textembedding-bloom-7b1.tar.gz",  # noqa: E501
        "HostingPrepackedArtifactVersion": "1.0.1",
        "InferenceDependencies": [
            "accelerate==0.16.0",
            "bitsandbytes==0.37.0",
            "filelock==3.9.0",
            "huggingface_hub==0.12.0",
            "regex==2022.7.9",
            "tokenizers==0.13.2",
            "transformers==4.26.0",
        ],
        "TrainingDependencies": [],
        "InferenceEnvironmentVariables": [
            {
                "Name": "SAGEMAKER_PROGRAM",
                "Type": "text",
                "Default": "inference.py",
                "Scope": "container",
                "RequiredForModelClass": True,
            }
        ],
        "TrainingMetrics": [],
        "DefaultInferenceInstanceType": "ml.g5.12xlarge",
        "SupportedInferenceInstanceTypes": [
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.g4dn.12xlarge",
        ],
        "deploy_kwargs": {
            "ModelDataDownloadTimeout": 3600,
            "ContainerStartupHealthCheckTimeout": 3600,
        },
        "SageMakerSdkPredictorSpecifications": {
            "SupportedContentTypes": ["application/json", "application/x-text"],
            "SupportedAcceptTypes": ["application/json;verbose", "application/json"],
            "DefaultContentType": "application/json",
            "DefaultAcceptType": "application/json",
        },
        "InferenceVolumeSize": 256,
        "InferenceEnableNetworkIsolation": True,
        "ValidationSupported": False,
        "FineTuningSupported": False,
        "ResourceNameBase": "hf-textembedding-bloom-7b1",
        "HostingInstanceTypeVariants": {
            "Aliases": {
                "alias_ecr_uri_3": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04",  # noqa: E501
                "cpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-cpu-py38",
                "gpu_ecr_uri_2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38",
            },
            "Variants": {
                "c4": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5d": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c5n": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c6i": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "c7i": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "g4dn": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "g5": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "local": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "m4": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m5d": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m6i": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "m7i": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "p2": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p3dn": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4d": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "p5": {"properties": {"image_uri": "$gpu_ecr_uri_2"}},
                "r5": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r5d": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "r7i": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t2": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "t3": {"properties": {"image_uri": "$cpu_ecr_uri_1"}},
                "trn1": {"properties": {"image_uri": "$alias_ecr_uri_3"}},
                "trn1n": {"properties": {"image_uri": "$alias_ecr_uri_3"}},
            },
        },
        "TrainingModelPackageArtifactUri": None,
        "DynamicContainerDeploymentSupported": False,
        "License": "BigScience RAIL",
        "Dependencies": [],
    },
}
