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
import copy
import numpy as np
from sagemaker.jumpstart.types import (
    JumpStartECRSpecs,
    JumpStartHyperparameter,
    JumpStartInstanceTypeVariants,
    JumpStartEnvironmentVariable,
    JumpStartPredictorSpecs,
    JumpStartSerializablePayload,
    JumpStartModelSpecs,
    JumpStartModelHeader,
    HubModelDocument,
)
from tests.unit.sagemaker.jumpstart.constants import (
    BASE_SPEC,
    HUB_MODEL_DOCUMENT_DICTS,
    SPECIAL_MODEL_SPECS_DICT,
    BASE_HUB_NOTEBOOK_DOCUMENT,
)

llama_model_document = HUB_MODEL_DOCUMENT_DICTS["meta-textgeneration-llama-2-70b"]
gemma_model_spec = SPECIAL_MODEL_SPECS_DICT["gemma-model-2b-v1_1_0"]

INSTANCE_TYPE_VARIANT = JumpStartInstanceTypeVariants(
    {
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
                    "resource_requirements": {"req1": 1, "req2": {"1": 2, "2": 3}, "req3": 9},
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
                    "resource_requirements": {
                        "req2": {"2": 5, "9": 999},
                        "req3": 999,
                        "req4": "blah",
                    },
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
    }
)


def test_jumpstart_model_header():

    header_dict = {
        "model_id": "tensorflow-ic-imagenet-inception-v3-classification-4",
        "version": "1.0.0",
        "min_version": "2.49.0",
        "spec_key": "community_models_specs/tensorflow-ic-imagenet-inception-v3-classification-4/specs_v1.0.0.json",
    }

    header1 = JumpStartModelHeader(header_dict)

    assert header1.model_id == "tensorflow-ic-imagenet-inception-v3-classification-4"
    assert header1.version == "1.0.0"
    assert header1.min_version == "2.49.0"
    assert (
        header1.spec_key
        == "community_models_specs/tensorflow-ic-imagenet-inception-v3-classification-4/specs_v1.0.0.json"
    )

    assert header1.to_json() == header_dict

    header2 = JumpStartModelHeader(
        {
            "model_id": "pytorch-ic-imagenet-inception-v3-classification-4",
            "version": "1.0.0",
            "min_version": "2.49.0",
            "spec_key": "community_models_specs/tensorflow-ic-imagenet-inception-v3-classification-4/specs_v1.0.0.json",
        }
    )

    assert header1 != header2

    header3 = copy.deepcopy(header1)
    assert header1 == header3


def test_use_training_model_artifact():
    specs1 = JumpStartModelSpecs(BASE_SPEC)
    assert specs1.use_training_model_artifact()
    specs1.gated_bucket = True
    assert not specs1.use_training_model_artifact()
    specs1.gated_bucket = False
    specs1.training_model_package_artifact_uris = {"region1": "blah", "region2": "blah2"}
    assert not specs1.use_training_model_artifact()


def test_jumpstart_model_specs():

    specs1 = JumpStartModelSpecs(BASE_SPEC)

    assert specs1.model_id == "pytorch-ic-mobilenet-v2"
    assert specs1.version == "1.0.0"
    assert specs1.min_sdk_version == "2.49.0"
    assert specs1.training_supported
    assert specs1.incremental_training_supported
    assert specs1.hosting_ecr_specs == JumpStartECRSpecs(
        {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        }
    )
    assert specs1.training_ecr_specs == JumpStartECRSpecs(
        {
            "framework": "pytorch",
            "framework_version": "1.5.0",
            "py_version": "py3",
        }
    )
    assert specs1.hosting_artifact_key == "pytorch-infer/infer-pytorch-ic-mobilenet-v2.tar.gz"
    assert specs1.training_artifact_key == "pytorch-training/train-pytorch-ic-mobilenet-v2.tar.gz"
    assert (
        specs1.hosting_script_key
        == "source-directory-tarballs/pytorch/inference/ic/v1.0.0/sourcedir.tar.gz"
    )
    assert (
        specs1.training_script_key
        == "source-directory-tarballs/pytorch/transfer_learning/ic/v1.0.0/sourcedir.tar.gz"
    )
    assert specs1.hyperparameters == [
        JumpStartHyperparameter(
            {
                "name": "epochs",
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 1000,
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "adam-learning-rate",
                "type": "float",
                "default": 0.05,
                "min": 1e-08,
                "max": 1,
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "batch-size",
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 1024,
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            }
        ),
    ]

    assert specs1.to_json() == BASE_SPEC

    diff_specs = copy.deepcopy(BASE_SPEC)
    diff_specs["model_id"] = "diff model ID"
    specs2 = JumpStartModelSpecs(diff_specs)
    assert specs1 != specs2

    specs3 = copy.deepcopy(specs1)
    assert specs3 == specs1


def test_jumpstart_model_specs_from_describe_hub_content_response():
    # TODO: Implement
    pass


def test_jumpstart_image_uri_instance_variants():

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.p3.200xlarge", region="us-west-2")
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/stud-gpu"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.g9.12xlarge", region="us-west-2")
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:"
        "1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.p3.2xlarge", region="us-west-2")
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:"
        "1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.g4dn.2xlarge", region="us-west-2")
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:"
        "1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.c2.xlarge", region="us-west-2")
        == "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="local", region="us-west-2")
        == "867930986793.dkr.us-west-2.amazonaws.com/cpu-blah"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="local_gpu", region="us-west-2") is None
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.g5.12xlarge", region="us-west-2")
        is None
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.c3.xlarge", region="us-west-2")
        is None
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.c2.xlarge", region="us-east-2000")
        is None
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_image_uri(instance_type="ml.c3.xlarge", region="us-east-2000")
        is None
    )


def test_jumpstart_hyperparameter_instance_variants():

    hyperparams = INSTANCE_TYPE_VARIANT.get_instance_specific_hyperparameters(
        instance_type="ml.g9.2xlarge"
    )
    assert hyperparams == [
        JumpStartHyperparameter(
            {"name": "num_bag_sets", "type": "int", "default": 5, "min": 5, "scope": "algorithm"}
        ),
        JumpStartHyperparameter(
            {
                "name": "num_stack_levels",
                "type": "int",
                "default": 6,
                "min": 7,
                "max": 3,
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "refit_full",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "set_best_to_refit_full",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "save_space",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "verbosity",
                "type": "int",
                "default": 2,
                "min": 0,
                "max": 4,
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            }
        ),
    ]

    hyperparams = INSTANCE_TYPE_VARIANT.get_instance_specific_hyperparameters(
        instance_type="ml.g9.12xlarge"
    )
    assert hyperparams == [
        JumpStartHyperparameter(
            {"name": "eval_metric", "type": "text", "default": "auto", "scope": "algorithm"}
        ),
        JumpStartHyperparameter(
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
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "auto_stack",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "num_bag_folds",
                "type": "text",
                "default": "0",
                "options": ["0", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {"name": "num_bag_sets", "type": "int", "default": 1, "min": 1, "scope": "algorithm"}
        ),
        JumpStartHyperparameter(
            {
                "name": "num_stack_levels",
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 3,
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "refit_full",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "set_best_to_refit_full",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "save_space",
                "type": "text",
                "default": "False",
                "options": ["True", "False"],
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "verbosity",
                "type": "int",
                "default": 2,
                "min": 0,
                "max": 4,
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_submit_directory",
                "type": "text",
                "default": "/opt/ml/input/data/code/sourcedir.tar.gz",
                "scope": "container",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_program",
                "type": "text",
                "default": "transfer_learning.py",
                "scope": "container",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "sagemaker_container_log_level",
                "type": "text",
                "default": "20",
                "scope": "container",
            }
        ),
    ]

    hyperparams = INSTANCE_TYPE_VARIANT.get_instance_specific_hyperparameters(
        instance_type="ml.g77.2xlarge"
    )
    assert hyperparams == []

    hyperparams = INSTANCE_TYPE_VARIANT.get_instance_specific_hyperparameters(
        instance_type="ml.p2.2xlarge"
    )
    assert hyperparams == []


def test_jumpstart_inference_instance_type_variants():
    assert INSTANCE_TYPE_VARIANT.get_instance_specific_supported_inference_instance_types(
        "ml.p2.xlarge"
    ) == ["ml.p2.xlarge", "ml.p3.xlarge"]
    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_default_inference_instance_type("ml.p2.2xlarge")
        == "ml.p2.xlarge"
    )

    assert INSTANCE_TYPE_VARIANT.get_instance_specific_supported_inference_instance_types(
        "ml.p2.12xlarge"
    ) == ["ml.p2.xlarge", "ml.p3.xlarge", "ml.p5.xlarge"]
    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_default_inference_instance_type(
            "ml.p2.12xlarge"
        )
        == "ml.p5.xlarge"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_supported_inference_instance_types(
            "ml.sdfsad.12xlarge"
        )
        == []
    )
    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_default_inference_instance_type(
            "ml.adfas.12xlarge"
        )
        is None
    )

    assert INSTANCE_TYPE_VARIANT.get_instance_specific_supported_inference_instance_types(
        "ml.trn1.12xlarge"
    ) == ["ml.inf1.2xlarge", "ml.inf1.xlarge"]
    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_default_inference_instance_type(
            "ml.trn1.12xlarge"
        )
        == "ml.inf1.xlarge"
    )


def test_jumpstart_environment_variables_instance_variants():
    assert INSTANCE_TYPE_VARIANT.get_instance_specific_environment_variables(
        instance_type="ml.g9.12xlarge"
    ) == {"TENSOR_PARALLEL_DEGREE": "4"}

    assert INSTANCE_TYPE_VARIANT.get_instance_specific_environment_variables(
        instance_type="ml.g5.48xlarge"
    ) == {"TENSOR_PARALLEL_DEGREE": "8", "JOHN": "DOE"}

    assert INSTANCE_TYPE_VARIANT.get_instance_specific_environment_variables(
        instance_type="ml.m2.48xlarge"
    ) == {"TENSOR_PARALLEL_DEGREE": "400"}

    assert INSTANCE_TYPE_VARIANT.get_instance_specific_environment_variables(
        instance_type="ml.g6.48xlarge"
    ) == {"BLAH": "4"}

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_environment_variables(
            instance_type="ml.p2.xlarge"
        )
        == {}
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_environment_variables(
            instance_type="safh8ads9fhsad89fh"
        )
        == {}
    )


def test_jumpstart_metric_definitions_instance_variants():

    metric_definitions = INSTANCE_TYPE_VARIANT.get_instance_specific_metric_definitions(
        instance_type="ml.p2.2xlarge"
    )
    assert metric_definitions == [
        {
            "Name": "huggingface-textgeneration:wtafigo",
            "Regex": "'evasadfasdl_loss': ([0-9]+\\.[0-9]+)",
        },
        {"Name": "huggingface-textgeneration:eval-loss", "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)"},
        {
            "Name": "huggingface-textgeneration:train-loss",
            "Regex": "'instance family specific': ([0-9]+\\.[0-9]+)",
        },
        {
            "Name": "huggingface-textgeneration:noneyourbusiness-loss",
            "Regex": "'loss-noyb': ([0-9]+\\.[0-9]+)",
        },
    ]

    metric_definitions = INSTANCE_TYPE_VARIANT.get_instance_specific_metric_definitions(
        instance_type="ml.p2.12xlarge"
    )
    assert metric_definitions == [
        {"Name": "huggingface-textgeneration:eval-loss", "Regex": "'eval_loss': ([0-9]+\\.[0-9]+)"},
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
        {
            "Name": "huggingface-textgeneration:wtafigo",
            "Regex": "'evasadfasdl_loss': ([0-9]+\\.[0-9]+)",
        },
    ]

    metric_definitions = INSTANCE_TYPE_VARIANT.get_instance_specific_metric_definitions(
        instance_type="ml.g77.2xlarge"
    )
    assert metric_definitions == []

    metric_definitions = INSTANCE_TYPE_VARIANT.get_instance_specific_metric_definitions(
        instance_type="ml.p3.2xlarge"
    )
    assert metric_definitions == []


def test_jumpstart_hosting_prepacked_artifact_key_instance_variants():
    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_prepacked_artifact_key(
            instance_type="ml.g6.xlarge"
        )
        == "path/to/prepacked/inference/artifact/prefix/"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_prepacked_artifact_key(
            instance_type="ml.p4.9xlarge"
        )
        == "path/to/prepacked/inference/artifact/prefix/number2/"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_prepacked_artifact_key(
            instance_type="ml.g9.9xlarge"
        )
        == "asfs/adsf/sda/f"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_prepacked_artifact_key(
            instance_type="ml.g9.12xlarge"
        )
        == "nlahdasf/asdf/asd/f"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_prepacked_artifact_key(
            instance_type="ml.g9dsfsdfs.12xlarge"
        )
        is None
    )


def test_jumpstart_training_artifact_key_instance_variants():
    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_artifact_key(instance_type="ml.g6.xlarge")
        == "path/to/training/artifact.tar.gz"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_artifact_key(instance_type="ml.g4.9xlarge")
        == "path/to/prepacked/training/artifact/prefix/number2/"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_artifact_key(instance_type="ml.p9.9xlarge")
        == "do/re/mi"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_artifact_key(instance_type="ml.p9.12xlarge")
        == "you/not/entertained"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_artifact_key(
            instance_type="ml.g9dsfsdfs.12xlarge"
        )
        is None
    )


def test_jumpstart_resource_requirements_instance_variants():
    assert INSTANCE_TYPE_VARIANT.get_instance_specific_resource_requirements(
        instance_type="ml.p2.xlarge"
    ) == {"req2": {"2": 5, "9": 999}, "req3": 999, "req4": "blah"}

    assert INSTANCE_TYPE_VARIANT.get_instance_specific_resource_requirements(
        instance_type="ml.p2.12xlarge"
    ) == {"req1": 1, "req2": {"1": 2, "2": 3}, "req3": 9, "req4": "blah"}

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_resource_requirements(
            instance_type="ml.p99.12xlarge"
        )
        == {}
    )


def test_hub_instance_varaints():
    instance_variant = JumpStartInstanceTypeVariants(
        llama_model_document.get("HostingInstanceTypeVariants"), is_hub_content=True
    )

    assert instance_variant.get_instance_specific_environment_variables("ml.g5.12xlarge") == {
        "SM_NUM_GPUS": "4"
    }
    assert instance_variant.get_instance_specific_environment_variables("ml.p4d.24xlarge") == {
        "SM_NUM_GPUS": "8"
    }

    assert instance_variant.get_image_uri("ml.g5.2xlarge") == (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1"
        "-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
    )

    instance_variant = JumpStartInstanceTypeVariants(
        llama_model_document.get("TrainingInstanceTypeVariants"), is_hub_content=True
    )

    assert (
        instance_variant.get_instance_specific_gated_model_key_env_var_value("ml.p4d.2xlarge")
        == "meta-training/p4d/v1.0.0/train-meta-textgeneration-llama-2-70b.tar.gz"
    )
    assert (
        instance_variant.get_instance_specific_gated_model_key_env_var_value("ml.g5.24xlarge")
        == "meta-training/g5/v1.0.0/train-meta-textgeneration-llama-2-70b.tar.gz"
    )

    assert instance_variant.get_image_uri("ml.p3.xlarge") == (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training"
        ":2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
    )


def test_extract_task_value_with_match():
    region = "us-west-2"
    gemma_model_document = HubModelDocument(
        json_obj=HUB_MODEL_DOCUMENT_DICTS["huggingface-llm-gemma-2b-instruct"], region=region
    )
    input_string = "| | |\n|---|---|\n||\n| Task: | Text to image|\n| Fine-tunable: | No|\n| Source: | Stability AI"
    expected_output = "Text to image"
    assert gemma_model_document._extract_task_value(input_string) == expected_output


def test_extract_task_value_without_match():
    region = "us-west-2"
    gemma_model_document = HubModelDocument(
        json_obj=HUB_MODEL_DOCUMENT_DICTS["huggingface-llm-gemma-2b-instruct"], region=region
    )
    input_string = "| | |\n|---|---|\n||\n| Fine-tunable: | No|\n| Source: | Stability AI"
    assert gemma_model_document._extract_task_value(input_string) is None


def test_extract_task_value_with_none_input():
    region = "us-west-2"
    gemma_model_document = HubModelDocument(
        json_obj=HUB_MODEL_DOCUMENT_DICTS["huggingface-llm-gemma-2b-instruct"], region=region
    )
    input_string = None
    assert gemma_model_document._extract_task_value(input_string) is None


def test_extract_task_value_with_empty_string():
    region = "us-west-2"
    gemma_model_document = HubModelDocument(
        json_obj=HUB_MODEL_DOCUMENT_DICTS["huggingface-llm-gemma-2b-instruct"], region=region
    )
    input_string = ""
    assert gemma_model_document._extract_task_value(input_string) is None


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
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
    )
    with pytest.raises(AttributeError) as excinfo:
        gemma_model_document.hosting_ecr_specs
    assert str(excinfo.value) == "'HubModelDocument' object has no attribute 'hosting_ecr_specs'"
    assert gemma_model_document.hosting_artifact_s3_data_type == "S3Prefix"
    assert gemma_model_document.hosting_artifact_compression_type == "None"
    assert (
        gemma_model_document.hosting_artifact_uri
        == "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/huggingface-llm-gemma-2b-instruct/artifacts/inference/v1.0.0/"
    )
    assert (
        gemma_model_document.hosting_script_uri
        == "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/inference/llm/v1.0.1/sourcedir.tar.gz"
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
        == "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/huggingface-llm-gemma-2b-instruct/artifacts/inference-prepack/v1.0.0/"
    )
    assert gemma_model_document.hosting_prepacked_artifact_version == "1.0.0"
    assert gemma_model_document.hosting_use_script_uri is False
    assert (
        gemma_model_document.hosting_eula_uri
        == "s3://jumpstart-cache-prod-us-west-2/huggingface-llm/fmhMetadata/terms/gemmaTerms.txt"
    )
    assert (
        gemma_model_document.training_ecr_uri
        == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
    )
    with pytest.raises(AttributeError) as excinfo:
        gemma_model_document.training_ecr_specs
    assert str(excinfo.value) == "'HubModelDocument' object has no attribute 'training_ecr_specs'"
    assert (
        gemma_model_document.training_prepacked_script_uri
        == "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/transfer_learning/llm/prepack/v1.1.1/sourcedir.tar.gz"
    )
    assert gemma_model_document.training_prepacked_script_version == "1.1.1"
    assert (
        gemma_model_document.training_script_uri
        == "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/huggingface/transfer_learning/llm/v1.1.1/sourcedir.tar.gz"
    )
    assert gemma_model_document.training_artifact_s3_data_type == "S3Prefix"
    assert gemma_model_document.training_artifact_compression_type == "None"
    assert (
        gemma_model_document.training_artifact_uri
        == "s3://jumpstart-cache-prod-us-west-2/huggingface-training/train-huggingface-llm-gemma-2b-instruct.tar.gz"
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
                    "Inputs": "<bos><start_of_turn>user\nWrite a hello world program<end_of_turn>\n<start_of_turn>model",
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
                "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
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
                "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
            },
            "Variants": {
                "g4dn": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/g4dn/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz",
                    },
                },
                "g5": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/g5/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz",
                    },
                },
                "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p2": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
                "p3dn": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/p3dn/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz",
                    },
                },
                "p4d": {
                    "properties": {
                        "image_uri": "$gpu_ecr_uri_1",
                        "gated_model_key_env_var_value": "huggingface-training/p4d/v1.0.0/train-huggingface-llm-gemma-2b-instruct.tar.gz",
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
            "- For CSV/JSON files, the text data is used from the column called 'text' or the first column if no column called 'text' is found",
            "- The number of files under train and validation (if provided) should equal to one, respectively.",
            " [Learn how to setup an AWS S3 bucket.](https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingBucket.html)",
        ],
        "HubDefaultTrainData": [
            "Dataset: [SEC](https://www.sec.gov/edgar/searchedgar/companysearch)",
            "SEC filing contains regulatory documents that companies and issuers of securities must submit to the Securities and Exchange Commission (SEC) on a regular basis.",
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


def test_hub_content_document_from_model_specs():
    model_specs = JumpStartModelSpecs(gemma_model_spec)
    region = "us-west-2"
    # model_document_from_model_specs = HubModelDocument(
    #     model_specs=model_specs,
    #     studio_specs=
    #     region=region)
    # model_document_from_json = HubModelDocument(
    #     json_obj=HUB_MODEL_DOCUMENT_DICTS["huggingface-llm-gemma-2b-instruct"],
    #     region=region
    # )
    # assert model_document_from_json.to_json() == model_document_from_model_specs.to_json()
    pass
