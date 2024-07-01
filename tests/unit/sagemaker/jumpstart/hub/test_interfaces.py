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

import pytest
import numpy as np
from sagemaker.jumpstart.types import (
    JumpStartHyperparameter,
    JumpStartInstanceTypeVariants,
    JumpStartEnvironmentVariable,
    JumpStartPredictorSpecs,
    JumpStartSerializablePayload,
)
from sagemaker.jumpstart.hub.interfaces import HubModelDocument
from tests.unit.sagemaker.jumpstart.constants import (
    SPECIAL_MODEL_SPECS_DICT,
    HUB_MODEL_DOCUMENT_DICTS,
)

gemma_model_spec = SPECIAL_MODEL_SPECS_DICT["gemma-model-2b-v1_1_0"]


def test_hub_content_document_from_json_obj():
    region = "us-west-2"
    gemma_model_document = HubModelDocument(
        json_obj=HUB_MODEL_DOCUMENT_DICTS["huggingface-llm-gemma-2b-instruct"], region=region
    )
    assert gemma_model_document.url == "https://huggingface.co/google/gemma-2b-it"
    assert gemma_model_document.min_sdk_version == "2.189.0"
    assert gemma_model_document.training_supported is True
    assert gemma_model_document.incremental_training_supported is False
    assert (
        gemma_model_document.hosting_ecr_uri
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:"
        "2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
    )
    with pytest.raises(AttributeError) as excinfo:
        gemma_model_document.hosting_ecr_specs
    assert str(excinfo.value) == "'HubModelDocument' object has no attribute 'hosting_ecr_specs'"
    assert gemma_model_document.hosting_artifact_s3_data_type == "S3Prefix"
    assert gemma_model_document.hosting_artifact_compression_type == "None"
    assert (
        gemma_model_document.hosting_artifact_uri
        == "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/huggingface-llm-gemma-2b-instruct"
        "/artifacts/inference/v1.0.0/"
    )
    assert (
        gemma_model_document.hosting_script_uri
        == "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/inference/"
        "llm/v1.0.1/sourcedir.tar.gz"
    )
    assert gemma_model_document.inference_dependencies == []
    assert gemma_model_document.training_dependencies == [
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
    ]
    assert (
        gemma_model_document.hosting_prepacked_artifact_uri
        == "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/huggingface-llm-gemma-2b-instruct/"
        "artifacts/inference-prepack/v1.0.0/"
    )
    assert gemma_model_document.hosting_prepacked_artifact_version == "1.0.0"
    assert gemma_model_document.hosting_use_script_uri is False
    assert (
        gemma_model_document.hosting_eula_uri
        == "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/fmhMetadata/terms/gemmaTerms.txt"
    )
    assert (
        gemma_model_document.training_ecr_uri
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers"
        "4.28.1-gpu-py310-cu118-ubuntu20.04"
    )
    with pytest.raises(AttributeError) as excinfo:
        gemma_model_document.training_ecr_specs
    assert str(excinfo.value) == "'HubModelDocument' object has no attribute 'training_ecr_specs'"
    assert (
        gemma_model_document.training_prepacked_script_uri
        == "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/transfer_learning/"
        "llm/prepack/v1.1.1/sourcedir.tar.gz"
    )
    assert gemma_model_document.training_prepacked_script_version == "1.1.1"
    assert (
        gemma_model_document.training_script_uri
        == "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/transfer_learning/"
        "llm/v1.1.1/sourcedir.tar.gz"
    )
    assert gemma_model_document.training_artifact_s3_data_type == "S3Prefix"
    assert gemma_model_document.training_artifact_compression_type == "None"
    assert (
        gemma_model_document.training_artifact_uri
        == "s3://jumpstart-cache-prod-us-west-2/huggingface-training/train-huggingface-llm-gemma-2b-instruct"
        ".tar.gz"
    )
    assert gemma_model_document.hyperparameters == [
        JumpStartHyperparameter(
            {
                "Name": "peft_type",
                "Type": "text",
                "Default": "lora",
                "Options": ["lora", "None"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "instruction_tuned",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "chat_dataset",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "epoch",
                "Type": "int",
                "Default": 1,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "learning_rate",
                "Type": "float",
                "Default": 0.0001,
                "Min": 1e-08,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "lora_r",
                "Type": "int",
                "Default": 64,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {"Name": "lora_alpha", "Type": "int", "Default": 16, "Min": 0, "Scope": "algorithm"},
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "lora_dropout",
                "Type": "float",
                "Default": 0,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {"Name": "bits", "Type": "int", "Default": 4, "Scope": "algorithm"}, is_hub_content=True
        ),
        JumpStartHyperparameter(
            {
                "Name": "double_quant",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "quant_Type",
                "Type": "text",
                "Default": "nf4",
                "Options": ["fp4", "nf4"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "per_device_train_batch_size",
                "Type": "int",
                "Default": 1,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "per_device_eval_batch_size",
                "Type": "int",
                "Default": 2,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "warmup_ratio",
                "Type": "float",
                "Default": 0.1,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "train_from_scratch",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "fp16",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "bf16",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "evaluation_strategy",
                "Type": "text",
                "Default": "steps",
                "Options": ["steps", "epoch", "no"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "eval_steps",
                "Type": "int",
                "Default": 20,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "gradient_accumulation_steps",
                "Type": "int",
                "Default": 4,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "logging_steps",
                "Type": "int",
                "Default": 8,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "weight_decay",
                "Type": "float",
                "Default": 0.2,
                "Min": 1e-08,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "load_best_model_at_end",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "max_train_samples",
                "Type": "int",
                "Default": -1,
                "Min": -1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "max_val_samples",
                "Type": "int",
                "Default": -1,
                "Min": -1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "seed",
                "Type": "int",
                "Default": 10,
                "Min": 1,
                "Max": 1000,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "max_input_length",
                "Type": "int",
                "Default": 1024,
                "Min": -1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "validation_split_ratio",
                "Type": "float",
                "Default": 0.2,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "train_data_split_seed",
                "Type": "int",
                "Default": 0,
                "Min": 0,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "preprocessing_num_workers",
                "Type": "text",
                "Default": "None",
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {"Name": "max_steps", "Type": "int", "Default": -1, "Scope": "algorithm"},
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "gradient_checkpointing",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "early_stopping_patience",
                "Type": "int",
                "Default": 3,
                "Min": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "early_stopping_threshold",
                "Type": "float",
                "Default": 0.0,
                "Min": 0,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "adam_beta1",
                "Type": "float",
                "Default": 0.9,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "adam_beta2",
                "Type": "float",
                "Default": 0.999,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "adam_epsilon",
                "Type": "float",
                "Default": 1e-08,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "max_grad_norm",
                "Type": "float",
                "Default": 1.0,
                "Min": 0,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "label_smoothing_factor",
                "Type": "float",
                "Default": 0,
                "Min": 0,
                "Max": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "logging_first_step",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "logging_nan_inf_filter",
                "Type": "text",
                "Default": "True",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "save_strategy",
                "Type": "text",
                "Default": "steps",
                "Options": ["no", "epoch", "steps"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "save_steps",
                "Type": "int",
                "Default": 500,
                "Min": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "save_total_limit",
                "Type": "int",
                "Default": 1,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "dataloader_drop_last",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "dataloader_num_workers",
                "Type": "int",
                "Default": 0,
                "Min": 0,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "eval_accumulation_steps",
                "Type": "text",
                "Default": "None",
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "auto_find_batch_size",
                "Type": "text",
                "Default": "False",
                "Options": ["True", "False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "lr_scheduler_type",
                "Type": "text",
                "Default": "constant_with_warmup",
                "Options": ["constant_with_warmup", "linear"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "warmup_steps",
                "Type": "int",
                "Default": 0,
                "Min": 0,
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "deepspeed",
                "Type": "text",
                "Default": "False",
                "Options": ["False"],
                "Scope": "algorithm",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "sagemaker_submit_directory",
                "Type": "text",
                "Default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "Scope": "container",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "sagemaker_program",
                "Type": "text",
                "Default": "transfer_learning.py",
                "Scope": "container",
            },
            is_hub_content=True,
        ),
        JumpStartHyperparameter(
            {
                "Name": "sagemaker_container_log_level",
                "Type": "text",
                "Default": "20",
                "Scope": "container",
            },
            is_hub_content=True,
        ),
    ]
    assert gemma_model_document.inference_environment_variables == [
        JumpStartEnvironmentVariable(
            {
                "Name": "SAGEMAKER_PROGRAM",
                "Type": "text",
                "Default": "inference.py",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "Type": "text",
                "Default": "/opt/ml/model/code",
                "Scope": "container",
                "RequiredForModelClass": False,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "Type": "text",
                "Default": "20",
                "Scope": "container",
                "RequiredForModelClass": False,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "Type": "text",
                "Default": "3600",
                "Scope": "container",
                "RequiredForModelClass": False,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "ENDPOINT_SERVER_TIMEOUT",
                "Type": "int",
                "Default": 3600,
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "MODEL_CACHE_ROOT",
                "Type": "text",
                "Default": "/opt/ml/model",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "SAGEMAKER_ENV",
                "Type": "text",
                "Default": "1",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "HF_MODEL_ID",
                "Type": "text",
                "Default": "/opt/ml/model",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "MAX_INPUT_LENGTH",
                "Type": "text",
                "Default": "8191",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "MAX_TOTAL_TOKENS",
                "Type": "text",
                "Default": "8192",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "MAX_BATCH_PREFILL_TOKENS",
                "Type": "text",
                "Default": "8191",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "SM_NUM_GPUS",
                "Type": "text",
                "Default": "1",
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
        JumpStartEnvironmentVariable(
            {
                "Name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "Type": "int",
                "Default": 1,
                "Scope": "container",
                "RequiredForModelClass": True,
            },
            is_hub_content=True,
        ),
    ]
    assert gemma_model_document.training_metrics == [
        {
            "Name": "huggingface-textgeneration:eval-loss",
            "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)",
        },
        {
            "Name": "huggingface-textgeneration:train-loss",
            "Regex": "'loss': ([0-9]+\\.[0-9]+)",
        },
    ]
    assert gemma_model_document.default_inference_instance_type == "ml.g5.xlarge"
    assert gemma_model_document.supported_inference_instance_types == [
        "ml.g5.xlarge",
        "ml.g5.2xlarge",
        "ml.g5.4xlarge",
        "ml.g5.8xlarge",
        "ml.g5.16xlarge",
        "ml.g5.12xlarge",
        "ml.g5.24xlarge",
        "ml.g5.48xlarge",
        "ml.p4d.24xlarge",
    ]
    assert gemma_model_document.default_training_instance_type == "ml.g5.2xlarge"
    assert np.array_equal(
        gemma_model_document.supported_training_instance_types,
        [
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.16xlarge",
            "ml.g5.12xlarge",
            "ml.g5.24xlarge",
            "ml.g5.48xlarge",
            "ml.p4d.24xlarge",
        ],
    )
    assert gemma_model_document.sage_maker_sdk_predictor_specifications == JumpStartPredictorSpecs(
        {
            "SupportedContentTypes": ["application/json"],
            "SupportedAcceptTypes": ["application/json"],
            "DefaultContentType": "application/json",
            "DefaultAcceptType": "application/json",
        },
        is_hub_content=True,
    )
    assert gemma_model_document.inference_volume_size == 512
    assert gemma_model_document.training_volume_size == 512
    assert gemma_model_document.inference_enable_network_isolation is True
    assert gemma_model_document.training_enable_network_isolation is True
    assert gemma_model_document.fine_tuning_supported is True
    assert gemma_model_document.validation_supported is True
    assert (
        gemma_model_document.default_training_dataset_uri
        == "s3://jumpstart-cache-prod-us-west-2/training-datasets/oasst_top/train/"
    )
    assert gemma_model_document.resource_name_base == "hf-llm-gemma-2b-instruct"
    assert gemma_model_document.default_payloads == {
        "HelloWorld": JumpStartSerializablePayload(
            {
                "ContentType": "application/json",
                "PromptKey": "inputs",
                "OutputKeys": {
                    "GeneratedText": "[0].generated_text",
                    "InputLogprobs": "[0].details.prefill[*].logprob",
                },
                "Body": {
                    "Inputs": "<bos><start_of_turn>user\nWrite a hello world program<end_of_turn>"
                    "\n<start_of_turn>model",
                    "Parameters": {
                        "MaxNewTokens": 256,
                        "DecoderInputDetails": True,
                        "Details": True,
                    },
                },
            },
            is_hub_content=True,
        ),
        "MachineLearningPoem": JumpStartSerializablePayload(
            {
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
            is_hub_content=True,
        ),
    }
    assert gemma_model_document.gated_bucket is True
    assert gemma_model_document.hosting_resource_requirements == {
        "MinMemoryMb": 8192,
        "NumAccelerators": 1,
    }
    assert gemma_model_document.hosting_instance_type_variants == JumpStartInstanceTypeVariants(
        {
            "Aliases": {
                "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch"
                "-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
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
        is_hub_content=True,
    )
    assert gemma_model_document.training_instance_type_variants == JumpStartInstanceTypeVariants(
        {
            "Aliases": {
                "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-"
                "training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
            },
            "Variants": {
                "g4dn": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/g4dn/v1.0.0/train-"
                        "huggingface-llm-gemma-2b-instruct.tar.gz",
                    },
                },
                "g5": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/g5/v1.0.0/train-"
                        "huggingface-llm-gemma-2b-instruct.tar.gz",
                    },
                },
                "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/p3dn/v1.0.0/train-"
                        "huggingface-llm-gemma-2b-instruct.tar.gz",
                    },
                },
                "p4d": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/p4d/v1.0.0/train-"
                        "huggingface-llm-gemma-2b-instruct.tar.gz",
                    },
                },
                "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            },
        },
        is_hub_content=True,
    )
    assert gemma_model_document.contextual_help == {
        "HubFormatTrainData": [
            "A train and an optional validation directories. Each directory contains a CSV/JSON/TXT. ",
            "- For CSV/JSON files, the text data is used from the column called 'text' or the "
            "first column if no column called 'text' is found",
            "- The number of files under train and validation (if provided) should equal to one,"
            " respectively.",
            " [Learn how to setup an AWS S3 bucket.]"
            "(https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html)",
        ],
        "HubDefaultTrainData": [
            "Dataset: [SEC](https://www.sec.gov/edgar/searchedgar/companysearch)",
            "SEC filing contains regulatory documents that companies and issuers of securities must "
            "submit to the Securities and Exchange Commission (SEC) on a regular basis.",
            "License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode)",
        ],
    }
    assert gemma_model_document.model_data_download_timeout == 1200
    assert gemma_model_document.container_startup_health_check_timeout == 1200
    assert gemma_model_document.encrypt_inter_container_traffic is True
    assert gemma_model_document.disable_output_compression is True
    assert gemma_model_document.max_runtime_in_seconds == 360000
    assert gemma_model_document.dynamic_container_deployment_supported is True
    assert gemma_model_document.training_model_package_artifact_uri is None
    assert gemma_model_document.dependencies == []
