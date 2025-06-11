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

MOCK_IMAGE_CONFIG = {"RepositoryAccessMode": "Vpc"}
MOCK_VPC_CONFIG = {"Subnets": ["subnet-1234"], "SecurityGroupIds": ["sg123"]}
DEPLOYMENT_CONFIGS = [
    {
        "ConfigName": "neuron-inference",
        "BenchmarkMetrics": [
            {"name": "Latency", "value": "100", "unit": "Tokens/S"},
            {"name": "Throughput", "value": "1867", "unit": "Tokens/S"},
        ],
        "DeploymentArgs": {
            "ModelDataDownloadTimeout": 1200,
            "ContainerStartupHealthCheckTimeout": 1200,
            "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
            ".0-gpu-py310-cu121-ubuntu20.04",
            "ModelData": {
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-private-cache-prod-us-west-2/meta-textgeneration/meta-textgeneration"
                    "-llama-2-7b/artifacts/inference-prepack/v1.0.0/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "InstanceType": "ml.p2.xlarge",
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "MAX_INPUT_LENGTH": "4095",
                "MAX_TOTAL_TOKENS": "4096",
                "SM_NUM_GPUS": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": 16384,
                "NumberOfAcceleratorDevicesRequired": 1,
            },
        },
    },
    {
        "ConfigName": "neuron-inference-budget",
        "BenchmarkMetrics": [
            {"name": "Latency", "value": "100", "unit": "Tokens/S"},
            {"name": "Throughput", "value": "1867", "unit": "Tokens/S"},
        ],
        "DeploymentArgs": {
            "ModelDataDownloadTimeout": 1200,
            "ContainerStartupHealthCheckTimeout": 1200,
            "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
            ".0-gpu-py310-cu121-ubuntu20.04",
            "ModelData": {
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-private-cache-prod-us-west-2/meta-textgeneration/meta-textgeneration"
                    "-llama-2-7b/artifacts/inference-prepack/v1.0.0/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "InstanceType": "ml.p2.xlarge",
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "MAX_INPUT_LENGTH": "4095",
                "MAX_TOTAL_TOKENS": "4096",
                "SM_NUM_GPUS": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": 16384,
                "NumberOfAcceleratorDevicesRequired": 1,
            },
        },
    },
    {
        "ConfigName": "gpu-inference-budget",
        "BenchmarkMetrics": [
            {"name": "Latency", "value": "100", "unit": "Tokens/S"},
            {"name": "Throughput", "value": "1867", "unit": "Tokens/S"},
        ],
        "DeploymentArgs": {
            "ModelDataDownloadTimeout": 1200,
            "ContainerStartupHealthCheckTimeout": 1200,
            "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
            ".0-gpu-py310-cu121-ubuntu20.04",
            "ModelData": {
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-private-cache-prod-us-west-2/meta-textgeneration/meta-textgeneration"
                    "-llama-2-7b/artifacts/inference-prepack/v1.0.0/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "InstanceType": "ml.p2.xlarge",
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "MAX_INPUT_LENGTH": "4095",
                "MAX_TOTAL_TOKENS": "4096",
                "SM_NUM_GPUS": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": 16384,
                "NumberOfAcceleratorDevicesRequired": 1,
            },
        },
    },
    {
        "ConfigName": "gpu-inference",
        "BenchmarkMetrics": [
            {"name": "Latency", "value": "100", "unit": "Tokens/S"},
            {"name": "Throughput", "value": "1867", "unit": "Tokens/S"},
        ],
        "DeploymentArgs": {
            "ModelDataDownloadTimeout": 1200,
            "ContainerStartupHealthCheckTimeout": 1200,
            "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
            ".0-gpu-py310-cu121-ubuntu20.04",
            "ModelData": {
                "S3DataSource": {
                    "S3Uri": "s3://jumpstart-private-cache-prod-us-west-2/meta-textgeneration/meta-textgeneration"
                    "-llama-2-7b/artifacts/inference-prepack/v1.0.0/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "InstanceType": "ml.p2.xlarge",
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "ENDPOINT_SERVER_TIMEOUT": "3600",
                "MODEL_CACHE_ROOT": "/opt/ml/model",
                "SAGEMAKER_ENV": "1",
                "HF_MODEL_ID": "/opt/ml/model",
                "MAX_INPUT_LENGTH": "4095",
                "MAX_TOTAL_TOKENS": "4096",
                "SM_NUM_GPUS": "1",
                "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
            },
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": 16384,
                "NumberOfAcceleratorDevicesRequired": 1,
            },
        },
    },
]
NON_OPTIMIZED_DEPLOYMENT_CONFIG = {
    "ConfigName": "neuron-inference",
    "BenchmarkMetrics": [
        {"name": "Latency", "value": "100", "unit": "Tokens/S"},
        {"name": "Throughput", "value": "1867", "unit": "Tokens/S"},
    ],
    "DeploymentArgs": {
        "ModelDataDownloadTimeout": 1200,
        "ContainerStartupHealthCheckTimeout": 1200,
        "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4"
        ".0-gpu-py310-cu121-ubuntu20.04",
        "ModelData": {
            "S3DataSource": {
                "S3Uri": "s3://jumpstart-private-cache-prod-us-west-2/meta-textgeneration/meta-textgeneration"
                "-llama-2-7b/artifacts/inference-prepack/v1.0.0/",
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        },
        "InstanceType": "ml.p2.xlarge",
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "ENDPOINT_SERVER_TIMEOUT": "3600",
            "MODEL_CACHE_ROOT": "/opt/ml/model",
            "SAGEMAKER_ENV": "1",
            "HF_MODEL_ID": "/opt/ml/model",
            "MAX_INPUT_LENGTH": "4095",
            "MAX_TOTAL_TOKENS": "4096",
            "SM_NUM_GPUS": "1",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        },
        "ComputeResourceRequirements": {
            "MinMemoryRequiredInMb": 16384,
            "NumberOfAcceleratorDevicesRequired": 1,
        },
    },
}
OPTIMIZED_DEPLOYMENT_CONFIG_WITH_GATED_DRAFT_MODEL = {
    "DeploymentConfigName": "lmi-optimized",
    "DeploymentArgs": {
        "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
        "djl-inference:0.29.0-lmi11.0.0-cu124",
        "ModelData": {
            "S3DataSource": {
                "S3Uri": "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/"
                "meta-textgeneration-llama-3-1-70b/artifacts/inference-prepack/v2.0.0/",
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        },
        "ModelPackageArn": None,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "ENDPOINT_SERVER_TIMEOUT": "3600",
            "MODEL_CACHE_ROOT": "/opt/ml/model",
            "SAGEMAKER_ENV": "1",
            "HF_MODEL_ID": "/opt/ml/model",
            "OPTION_SPECULATIVE_DRAFT_MODEL": "/opt/ml/additional-model-data-sources/draft_model",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        },
        "InstanceType": "ml.g6.2xlarge",
        "ComputeResourceRequirements": {
            "MinMemoryRequiredInMb": 131072,
            "NumberOfAcceleratorDevicesRequired": 1,
        },
        "ModelDataDownloadTimeout": 1200,
        "ContainerStartupHealthCheckTimeout": 1200,
        "AdditionalDataSources": {
            "speculative_decoding": [
                {
                    "channel_name": "draft_model",
                    "provider": {"name": "JumpStart", "classification": "gated"},
                    "artifact_version": "v1",
                    "hosting_eula_key": "fmhMetadata/eula/llama3_2Eula.txt",
                    "s3_data_source": {
                        "s3_uri": "meta-textgeneration/meta-textgeneration-llama-3-2-1b/artifacts/"
                        "inference-prepack/v1.0.0/",
                        "compression_type": "None",
                        "s3_data_type": "S3Prefix",
                    },
                }
            ]
        },
    },
    "AccelerationConfigs": [
        {
            "type": "Compilation",
            "enabled": False,
            "diy_workflow_overrides": {
                "gpu-lmi-trt": {
                    "enabled": False,
                    "reason": "TRT-LLM 0.11.0 in LMI v11 does not support llama 3.1",
                }
            },
        },
        {
            "type": "Speculative-Decoding",
            "enabled": True,
            "diy_workflow_overrides": {
                "gpu-lmi-trt": {
                    "enabled": False,
                    "reason": "LMI v11 does not support Speculative Decoding for TRT",
                }
            },
        },
        {
            "type": "Quantization",
            "enabled": False,
            "diy_workflow_overrides": {
                "gpu-lmi-trt": {
                    "enabled": False,
                    "reason": "TRT-LLM 0.11.0 in LMI v11 does not support llama 3.1",
                }
            },
        },
    ],
    "BenchmarkMetrics": {"ml.g6.2xlarge": None},
}
GATED_DRAFT_MODEL_CONFIG = {
    "channel_name": "draft_model",
    "provider": {"name": "JumpStart", "classification": "gated"},
    "artifact_version": "v1",
    "hosting_eula_key": "fmhMetadata/eula/llama3_2Eula.txt",
    "s3_data_source": {
        "s3_uri": "meta-textgeneration/meta-textgeneration-llama-3-2-1b/artifacts/"
        "inference-prepack/v1.0.0/",
        "compression_type": "None",
        "s3_data_type": "S3Prefix",
    },
}
NON_GATED_DRAFT_MODEL_CONFIG = {
    "channel_name": "draft_model",
    "s3_data_source": {
        "compression_type": "None",
        "s3_data_type": "S3Prefix",
        "s3_uri": "s3://sagemaker-sd-models-beta-us-west-2/"
        "sagemaker-speculative-decoding-llama3-small-v3/",
    },
}
CAMEL_CASE_ADDTL_DRAFT_MODEL_DATA_SOURCES = [
    {
        "ChannelName": "draft_model",
        "S3DataSource": {
            "S3Uri": "meta-textgeneration/meta-textgeneration-llama-3-2-1b/artifacts/"
            "inference-prepack/v1.0.0/",
            "CompressionType": "None",
            "S3DataType": "S3Prefix",
        },
    }
]
