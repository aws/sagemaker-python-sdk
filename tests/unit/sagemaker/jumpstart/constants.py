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


SPECIAL_MODEL_SPECS_DICT = {
    "env-var-variant-model": {
        "model_id": "huggingface-llm-falcon-180b-bf16",
        "url": "https://huggingface.co/tiiuae/falcon-180B",
        "version": "1.0.0",
        "min_sdk_version": "2.175.0",
        "training_supported": False,
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
        "hosting_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                    "huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                    "cpu_image_uri": "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah",
                }
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
        "version": "1.0.0",
        "min_sdk_version": "2.49.0",
        "training_supported": True,
        "incremental_training_supported": True,
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
                }
            },
            "variants": {
                "p2": {
                    "regional_properties": {"image_uri": "$gpu_image_uri"},
                    "properties": {"prepacked_artifact_key": "hello-world-1"},
                },
                "p3": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
                "p4": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
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
        "training_instance_type_variants": {
            "regional_aliases": {
                "us-west-2": {
                    "gpu_image_uri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
                    "huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04",
                    "cpu_image_uri": "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah",
                }
            },
            "variants": {
                "p2": {
                    "regional_properties": {"image_uri": "$gpu_image_uri"},
                    "properties": {"artifact_key": "hello-mars-1"},
                },
                "p3": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
                "p4": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
                "g4dn": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
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
        "hosting_artifact_key": "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz",
        "training_artifact_key": "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz",
        "hosting_script_key": None,
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        "training_prepacked_script_key": None,
        "hosting_prepacked_artifact_key": "basfsdfssf",
        "training_model_package_artifact_uris": None,
        "deprecate_warn_message": None,
        "deprecated_message": None,
        "hosting_model_package_arns": None,
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
        "model_kwargs": {},
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
        "version": "1.0.0",
        "min_sdk_version": "2.173.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.12.0",
            "py_version": "py38",
        },
        "hosting_artifact_key": "meta-infer/infer-meta-textgeneration-llama-2-7b-f.tar.gz",
        "hosting_script_key": "source-directory-tarballs/meta/inference/textgeneration/v1.0.0/sourcedir.tar.gz",
        "hosting_eula_key": "fmhMetadata/eula/llamaEula.txt",
        "hosting_model_package_arns": {
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/"
            "llama2-7b-f-e46eb8a833643ed58aaccd81498972c3",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/"
            "llama2-7b-f-e46eb8a833643ed58aaccd81498972c3",
        },
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "inference_environment_variables": [],
        "metrics": [],
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
        "inference_volume_size": 256,
        "inference_enable_network_isolation": True,
        "validation_supported": False,
        "fine_tuning_supported": False,
        "resource_name_base": "meta-textgeneration-llama-2-7b-f",
    },
    "js-trainable-model-prepacked": {
        "model_id": "huggingface-text2text-flan-t5-base",
        "url": "https://huggingface.co/google/flan-t5-base",
        "version": "1.2.0",
        "min_sdk_version": "2.130.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.10.2",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-text2text-flan-t5-base.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v1.0.4/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.0.0/infer-prepack-"
        "huggingface-text2text-flan-t5-base.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.0",
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
        "training_dependencies": [
            "Brotli==1.0.9",
            "absl-py==1.4.0",
            "accelerate==0.16.0",
            "datasets==2.9.0",
            "deepspeed==0.8.0",
            "evaluate==0.4.0",
            "hjson==3.1.0",
            "huggingface_hub==0.13.3",
            "inflate64==0.3.1",
            "multivolumefile==0.2.3",
            "ninja==1.11.1",
            "nltk==3.8.1",
            "psutil==5.9.4",
            "py-cpuinfo==9.0.0",
            "py7zr==0.20.4",
            "pybcj==1.0.1",
            "pycryptodomex==3.17",
            "pydantic==1.10.2",
            "pyppmd==1.0.0",
            "pyzstd==0.15.4",
            "rouge-score==0.1.2",
            "sagemaker_jumpstart_script_utilities==1.1.4",
            "sagemaker_jumpstart_tabular_script_utilities==1.0.0",
            "tensorboardX==2.6",
            "texttable==1.6.7",
            "transformers==4.26.0",
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
                "name": "validation_split_ratio",
                "type": "float",
                "default": 0.05,
                "min": 0,
                "max": 1,
                "scope": "algorithm",
            },
            {"name": "train_data_split_seed", "type": "int", "default": 0, "scope": "algorithm"},
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
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/"
        "v1.1.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/"
        "text2text/prepack/v1.0.1/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.0.1",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.10.2",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
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
        "default_inference_instance_type": "ml.g5.xlarge",
        "supported_inference_instance_types": [
            "ml.g5.xlarge",
            "ml.p2.xlarge",
            "ml.g4dn.xlarge",
            "ml.p3.2xlarge",
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
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": False},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/x-text"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-text",
            "default_accept_type": "application/json",
        },
    },
    "js-model-class-model-prepacked": {
        "model_id": "huggingface-txt2img-conflictx-complex-lineart",
        "url": "https://huggingface.co/Conflictx/Complex-Lineart",
        "version": "1.1.0",
        "min_sdk_version": "2.81.0",
        "training_supported": False,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.10.2",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-txt2img-conflictx-complex-lineart.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/txt2img/v1.1.0/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.0.0/infer-prepack-huggingface-txt2img-"
        "conflictx-complex-lineart.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.0",
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
        "supported_inference_instance_types": ["ml.p2.xlarge", "ml.p3.2xlarge", "ml.g4dn.xlarge"],
        "model_kwargs": {},
        "deploy_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/json"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/json",
            "default_accept_type": "application/json",
        },
    },
    "deprecated_model": {
        "model_id": "huggingface-text2text-flan-t5-base",
        "url": "https://huggingface.co/google/flan-t5-base",
        "version": "1.2.0",
        "min_sdk_version": "2.130.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.10.2",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-text2text-flan-t5-base.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v1.0.4/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.0.0/infer-prepack-"
        "huggingface-text2text-flan-t5-base.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.0",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": True,
        "hyperparameters": [],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/"
        "v1.1.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/"
        "text2text/prepack/v1.0.1/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.0.1",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.10.2",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
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
        "default_inference_instance_type": "ml.g5.xlarge",
        "supported_inference_instance_types": [
            "ml.g5.xlarge",
            "ml.p2.xlarge",
            "ml.g4dn.xlarge",
            "ml.p3.2xlarge",
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
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": False},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/x-text"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-text",
            "default_accept_type": "application/json",
        },
    },
    "vulnerable_model": {
        "model_id": "huggingface-text2text-flan-t5-base",
        "url": "https://huggingface.co/google/flan-t5-base",
        "version": "1.2.0",
        "min_sdk_version": "2.130.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.10.2",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-text2text-flan-t5-base.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v1.0.4/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.0.0/infer-prepack-"
        "huggingface-text2text-flan-t5-base.tar.gz",
        "hosting_prepacked_artifact_version": "1.0.0",
        "inference_vulnerable": True,
        "inference_dependencies": ["blah"],
        "inference_vulnerabilities": ["blah"],
        "training_vulnerable": True,
        "training_dependencies": ["blah"],
        "training_vulnerabilities": ["blah"],
        "deprecated": False,
        "hyperparameters": [],
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/text2text/"
        "v1.1.0/sourcedir.tar.gz",
        "training_prepacked_script_key": "source-directory-tarballs/huggingface/transfer_learning/"
        "text2text/prepack/v1.0.1/sourcedir.tar.gz",
        "training_prepacked_script_version": "1.0.1",
        "training_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.10.2",
            "py_version": "py38",
            "huggingface_transformers_version": "4.17.0",
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
        "default_inference_instance_type": "ml.g5.xlarge",
        "supported_inference_instance_types": [
            "ml.g5.xlarge",
            "ml.p2.xlarge",
            "ml.g4dn.xlarge",
            "ml.p3.2xlarge",
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
        "deploy_kwargs": {},
        "estimator_kwargs": {"encrypt_inter_container_traffic": False},
        "fit_kwargs": {},
        "predictor_specs": {
            "supported_content_types": ["application/x-text"],
            "supported_accept_types": ["application/json;verbose", "application/json"],
            "default_content_type": "application/x-text",
            "default_accept_type": "application/json",
        },
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
        "version": "2.0.0",
        "min_sdk_version": "2.173.0",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "djl-deepspeed",
            "framework_version": "0.21.0",
            "py_version": "py39",
        },
        "hosting_artifact_key": "meta-infer/infer-meta-textgeneration-llama-2-7b-f.tar.gz",
        "hosting_use_script_uri": False,
        "hosting_script_key": "source-directory-tarballs/meta/inference/textgeneration/v1.0.0/sourcedir.tar.gz",
        "hosting_eula_key": "fmhMetadata/eula/llamaEula.txt",
        "hosting_model_package_arns": {
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/"
            "llama2-7b-f-e46eb8a833643ed58aaccd81498972c3",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/"
            "llama2-7b-f-e46eb8a833643ed58aaccd81498972c3",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/"
            "llama2-7b-f-e46eb8a833643ed58aaccd81498972c3",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/"
            "llama2-7b-f-e46eb8a833643ed58aaccd81498972c3",
        },
        "training_model_package_artifact_uris": {
            "us-west-2": "s3://jumpstart-cache-alpha-us-west-2/dummy.tar.gz",
            "us-east-1": "s3://jumpstart-cache-alpha-us-west-2/dummy.tar.gz",
            "eu-west-1": "s3://jumpstart-cache-alpha-us-west-2/dummy.tar.gz",
            "ap-southeast-1": "s3://jumpstart-cache-alpha-us-west-2/dummy.tar.gz",
        },
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
        "training_vulnerabilities": [],
        "deprecated": False,
        "hyperparameters": [
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
        "training_script_key": "source-directory-tarballs/meta/transfer_learning/"
        "textgeneration/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework": "djl-deepspeed",
            "framework_version": "0.21.0",
            "py_version": "py39",
        },
        "training_artifact_key": "meta-training/train-meta-textgeneration-llama-2-7b-f.tar.gz",
        "inference_environment_variables": [],
        "metrics": [],
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
        "default_training_instance_type": "ml.p3.2xlarge",
        "supported_training_instance_types": ["ml.p3.2xlarge", "ml.p2.8xlarge", "ml.g4dn.xlarge"],
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
        "inference_enable_network_isolation": True,
        "training_enable_network_isolation": True,
        "default_training_dataset_key": "training-datasets/wikitext/",
        "validation_supported": False,
        "fine_tuning_supported": True,
        "resource_name_base": "meta-textgeneration-llama-2-7b-f",
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
        "hosting_artifact_key": "huggingface-infer/",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/text2text/v1.0.3/sourcedir.tar.gz",
        "hosting_prepacked_artifact_key": "huggingface-infer/prepack/v1.0.1/",
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
    "pytorch-eqa-bert-base-cased": {
        "model_id": "pytorch-eqa-bert-base-cased",
        "url": "https://pytorch.org/hub/huggingface_pytorch-transformers/",
        "version": "1.0.0",
        "min_sdk_version": "2.68.1",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        },
        "hosting_artifact_key": "pytorch-infer/infer-pytorch-eqa-bert-base-cased.tar.gz",
        "hosting_script_key": "source-directory-tarballs/pytorch/inference/eqa/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "transformers==3.5.1",
            "dataclasses==0.8",
            "filelock==3.0.12",
            "packaging==20.8",
            "pyparsing==2.4.7",
            "regex==2020.11.13",
            "sacremoses==0.0.43",
            "sentencepiece==0.1.91",
            "tokenizers==0.9.3",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "transformers==3.5.1",
            "dataclasses==0.8",
            "filelock==3.0.12",
            "packaging==20.8",
            "pyparsing==2.4.7",
            "regex==2020.11.13",
            "sacremoses==0.0.43",
            "sentencepiece==0.1.91",
            "tokenizers==0.9.3",
        ],
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
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/eqa/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.5.0",
            "framework": "pytorch",
            "py_version": "py3",
        },
        "training_artifact_key": "pytorch-training/train-pytorch-eqa-bert-base-cased.tar.gz",
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
    "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1": {
        "model_id": "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1",
        "url": "https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1",
        "version": "1.0.0",
        "min_sdk_version": "2.68.1",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "tensorflow",
            "framework_version": "2.3",
            "py_version": "py37",
        },
        "hosting_artifact_key": "tensorflow-infer/infer-tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1.tar.gz",
        "hosting_script_key": "source-directory-tarballs/tensorflow/inference/ic/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
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
        "training_script_key": "source-directory-tarballs/tensorflow/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "2.3",
            "framework": "tensorflow",
            "py_version": "py37",
        },
        "training_artifact_key": "tensorflow-training/train-tensorflow-ic-bit-"
        "m-r101x1-ilsvrc2012-classification-1.tar.gz",
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
    "mxnet-semseg-fcn-resnet50-ade": {
        "model_id": "mxnet-semseg-fcn-resnet50-ade",
        "url": "https://cv.gluon.ai/model_zoo/segmentation.html",
        "version": "1.0.0",
        "min_sdk_version": "2.68.1",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "mxnet",
            "framework_version": "1.7.0",
            "py_version": "py3",
        },
        "hosting_artifact_key": "mxnet-infer/infer-mxnet-semseg-fcn-resnet50-ade.tar.gz",
        "hosting_script_key": "source-directory-tarballs/mxnet/inference/semseg/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": ["numpy==1.19.5", "opencv_python==4.0.1.23"],
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
        "training_script_key": "source-directory-tarballs/mxnet/transfer_learning/semseg/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.7.0",
            "framework": "mxnet",
            "py_version": "py3",
        },
        "training_artifact_key": "mxnet-training/train-mxnet-semseg-fcn-resnet50-ade.tar.gz",
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
    "huggingface-spc-bert-base-cased": {
        "model_id": "huggingface-spc-bert-base-cased",
        "url": "https://huggingface.co/bert-base-cased",
        "version": "1.0.0",
        "min_sdk_version": "2.68.1",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "huggingface",
            "framework_version": "1.7.1",
            "py_version": "py36",
            "huggingface_transformers_version": "4.6.1",
        },
        "hosting_artifact_key": "huggingface-infer/infer-huggingface-spc-bert-base-cased.tar.gz",
        "hosting_script_key": "source-directory-tarballs/huggingface/inference/spc/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
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
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/spc/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.6.0",
            "framework": "huggingface",
            "huggingface_transformers_version": "4.4.2",
            "py_version": "py36",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-spc-bert-base-cased.tar.gz",
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
    "lightgbm-classification-model": {
        "model_id": "lightgbm-classification-model",
        "url": "https://lightgbm.readthedocs.io/en/latest/",
        "version": "1.0.0",
        "min_sdk_version": "2.68.1",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.9.0",
            "py_version": "py38",
        },
        "hosting_artifact_key": "lightgbm-infer/infer-lightgbm-classification-model.tar.gz",
        "hosting_script_key": "source-directory-tarballs/lightgbm/inference/classification/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "plotly==5.1.0",
            "joblib==1.0.1",
            "scikit_learn==1.0.1",
            "tenacity==8.0.1",
            "lightgbm==3.2.1",
            "threadpoolctl==2.2.0",
            "graphviz==0.17",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "tenacity==8.0.1",
            "plotly==5.1.0",
            "graphviz==0.17",
            "glibc==0.6.1",
            "lightgbm==3.2.1",
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
        "training_script_key": "source-directory-tarballs/lightgbm/transfer_learning/classification/"
        "v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.9.0",
            "framework": "pytorch",
            "py_version": "py38",
        },
        "training_artifact_key": "lightgbm-training/train-lightgbm-classification-model.tar.gz",
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
    "catboost-classification-model": {
        "model_id": "catboost-classification-model",
        "url": "https://catboost.ai/",
        "version": "1.0.0",
        "min_sdk_version": "2.68.1",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "pytorch",
            "framework_version": "1.9.0",
            "py_version": "py38",
        },
        "hosting_artifact_key": "catboost-infer/infer-catboost-classification-model.tar.gz",
        "hosting_script_key": "source-directory-tarballs/catboost/inference/classification/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [
            "tenacity==8.0.1",
            "plotly==5.1.0",
            "graphviz==0.17",
            "pyparsing==2.4.7",
            "cycler==0.10.0",
            "kiwisolver==1.3.2",
            "matplotlib==3.4.3",
            "catboost==1.0.1",
            "scikit_learn==1.0.1",
            "threadpoolctl==2.2.0",
        ],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [
            "tenacity==8.0.1",
            "plotly==5.1.0",
            "graphviz==0.17",
            "catboost==1.0.1",
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
        "training_script_key": "source-directory-tarballs/catboost/transfer_learning/"
        "classification/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.9.0",
            "framework": "pytorch",
            "py_version": "py38",
        },
        "training_artifact_key": "catboost-training/train-catboost-classification-model.tar.gz",
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
    "xgboost-classification-model": {
        "model_id": "xgboost-classification-model",
        "url": "https://xgboost.readthedocs.io/en/latest/",
        "version": "1.0.0",
        "min_sdk_version": "2.68.1",
        "training_supported": True,
        "incremental_training_supported": False,
        "hosting_ecr_specs": {
            "framework": "xgboost",
            "framework_version": "1.3-1",
            "py_version": "py3",
        },
        "hosting_artifact_key": "xgboost-infer/infer-xgboost-classification-model.tar.gz",
        "hosting_script_key": "source-directory-tarballs/xgboost/inference/classification/v1.0.0/sourcedir.tar.gz",
        "inference_vulnerable": False,
        "inference_dependencies": [],
        "inference_vulnerabilities": [],
        "training_vulnerable": False,
        "training_dependencies": [],
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
        "training_script_key": "source-directory-tarballs/xgboost/transfer_learning/classification/"
        "v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.3-1",
            "framework": "xgboost",
            "py_version": "py3",
        },
        "training_artifact_key": "xgboost-training/train-xgboost-classification-model.tar.gz",
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
    "sklearn-classification-linear": {
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

BASE_SPEC = {
    "model_id": "pytorch-ic-mobilenet-v2",
    "url": "https://pytorch.org/hub/pytorch_vision_mobilenet_v2/",
    "version": "1.0.0",
    "min_sdk_version": "2.49.0",
    "training_supported": True,
    "incremental_training_supported": True,
    "gated_bucket": False,
    "default_payloads": None,
    "hosting_ecr_specs": {
        "framework": "pytorch",
        "framework_version": "1.5.0",
        "py_version": "py3",
    },
    "hosting_instance_type_variants": None,
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
    "hosting_model_package_arns": None,
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
    "usage_info_message": None,
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
    "hosting_resource_requirements": {"num_accelerators": 1, "min_memory_mb": 34360},
    "dynamic_container_deployment_supported": True,
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
