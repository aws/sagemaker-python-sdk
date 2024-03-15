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
import copy
from sagemaker.jumpstart.types import (
    JumpStartECRSpecs,
    JumpStartHyperparameter,
    JumpStartInstanceTypeVariants,
    JumpStartModelSpecs,
    JumpStartModelHeader,
)
from tests.unit.sagemaker.jumpstart.constants import BASE_SPEC

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

    BASE_SPEC["model_id"] = "diff model ID"
    specs2 = JumpStartModelSpecs(BASE_SPEC)
    assert specs1 != specs2

    specs3 = copy.deepcopy(specs1)
    assert specs3 == specs1


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
