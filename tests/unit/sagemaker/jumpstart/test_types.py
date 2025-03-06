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
from unittest import TestCase
import pytest
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.types import (
    JumpStartBenchmarkStat,
    JumpStartECRSpecs,
    JumpStartEnvironmentVariable,
    JumpStartHyperparameter,
    JumpStartInstanceTypeVariants,
    JumpStartModelSpecs,
    JumpStartModelHeader,
    JumpStartConfigComponent,
    DeploymentConfigMetadata,
    JumpStartModelInitKwargs,
    S3DataSource,
)
from sagemaker.utils import S3_PREFIX
from tests.unit.sagemaker.jumpstart.constants import (
    BASE_SPEC,
    BASE_HOSTING_ADDITIONAL_DATA_SOURCES,
    INFERENCE_CONFIG_RANKINGS,
    INFERENCE_CONFIGS,
    TRAINING_CONFIG_RANKINGS,
    TRAINING_CONFIGS,
    INIT_KWARGS,
)

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
                    "training_artifact_key": "path/to/prepacked/training/artifact/prefix/number2/"
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
                "properties": {"training_artifact_key": "do/re/mi"},
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
                    "training_artifact_key": "you/not/entertained",
                }
            },
            "g6": {
                "properties": {
                    "environment_variables": {"BLAH": "4"},
                    "training_artifact_key": "path/to/training/artifact.tar.gz",
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
    specs1.training_model_package_artifact_uris = {"region1": "blah", "region2": "blah2"}
    assert not specs1.use_training_model_artifact()


def test_jumpstart_model_specs():

    specs1 = JumpStartModelSpecs(BASE_SPEC)

    assert specs1.model_id == "pytorch-ic-mobilenet-v2"
    assert specs1.version == "3.0.6"
    assert specs1.min_sdk_version == "2.189.0"
    assert specs1.training_supported
    assert specs1.incremental_training_supported
    assert specs1.hosting_ecr_specs == JumpStartECRSpecs(
        {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        }
    )
    assert specs1.training_ecr_specs == JumpStartECRSpecs(
        {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        }
    )
    assert (
        specs1.hosting_artifact_key
        == "pytorch-ic/pytorch-ic-mobilenet-v2/artifacts/inference/v2.0.0/"
    )
    assert (
        specs1.training_artifact_key
        == "pytorch-training/v2.0.0/train-pytorch-ic-mobilenet-v2.tar.gz"
    )
    assert (
        specs1.hosting_script_key
        == "source-directory-tarballs/pytorch/inference/ic/v2.0.0/sourcedir.tar.gz"
    )
    assert (
        specs1.training_script_key
        == "source-directory-tarballs/pytorch/transfer_learning/ic/v2.3.0/sourcedir.tar.gz"
    )
    assert specs1.default_training_dataset_key == "training-datasets/tf_flowers/"
    assert specs1.hyperparameters == [
        JumpStartHyperparameter(
            {
                "name": "train_only_top_layer",
                "type": "text",
                "options": ["True", "False"],
                "default": "True",
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "epochs",
                "type": "int",
                "default": 5,
                "scope": "algorithm",
                "min": 1,
                "max": 1000,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.001,
                "scope": "algorithm",
                "min": 1e-08,
                "max": 1,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "batch_size",
                "type": "int",
                "default": 4,
                "scope": "algorithm",
                "min": 1,
                "max": 1024,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "reinitialize_top_layer",
                "type": "text",
                "options": ["Auto", "True", "False"],
                "default": "Auto",
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

    print(specs1.to_json())
    assert specs1.to_json() == BASE_SPEC

    BASE_SPEC_COPY = copy.deepcopy(BASE_SPEC)
    BASE_SPEC_COPY["model_id"] = "diff model ID"
    specs2 = JumpStartModelSpecs(BASE_SPEC_COPY)
    assert specs1 != specs2

    specs3 = copy.deepcopy(specs1)
    assert specs3 == specs1


class TestS3DataSource(TestCase):
    def setUp(self):
        self.s3_data_source = S3DataSource(
            {
                "compression_type": "None",
                "s3_data_type": "S3Prefix",
                "s3_uri": "key/to/model/artifact/",
                "model_access_config": {"accept_eula": False},
            }
        )

    def test_set_bucket_with_valid_s3_uri(self):
        self.s3_data_source.set_bucket("my-bucket")
        self.assertEqual(self.s3_data_source.s3_uri, f"{S3_PREFIX}my-bucket/key/to/model/artifact/")

    def test_set_bucket_with_existing_s3_uri(self):
        self.s3_data_source.s3_uri = "s3://my-bucket/key/to/model/artifact/"
        self.s3_data_source.set_bucket("random-new-bucket")
        assert self.s3_data_source.s3_uri == "s3://random-new-bucket/key/to/model/artifact/"

    def test_set_bucket_with_existing_s3_uri_empty_bucket(self):
        self.s3_data_source.s3_uri = "s3://my-bucket"
        self.s3_data_source.set_bucket("random-new-bucket")
        assert self.s3_data_source.s3_uri == "s3://random-new-bucket"

    def test_set_bucket_with_existing_s3_uri_empty(self):
        self.s3_data_source.s3_uri = "s3://"
        self.s3_data_source.set_bucket("random-new-bucket")
        assert self.s3_data_source.s3_uri == "s3://random-new-bucket"


def test_get_speculative_decoding_s3_data_sources():
    specs = JumpStartModelSpecs({**BASE_SPEC, **BASE_HOSTING_ADDITIONAL_DATA_SOURCES})
    assert (
        specs.get_speculative_decoding_s3_data_sources()
        == specs.hosting_additional_data_sources.speculative_decoding
    )


def test_get_additional_s3_data_sources():
    specs = JumpStartModelSpecs({**BASE_SPEC, **BASE_HOSTING_ADDITIONAL_DATA_SOURCES})
    data_sources = [
        *specs.hosting_additional_data_sources.speculative_decoding,
        *specs.hosting_additional_data_sources.scripts,
    ]
    assert specs.get_additional_s3_data_sources() == data_sources


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
        INSTANCE_TYPE_VARIANT.get_instance_specific_training_artifact_key(
            instance_type="ml.g6.xlarge"
        )
        == "path/to/training/artifact.tar.gz"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_training_artifact_key(
            instance_type="ml.g4.9xlarge"
        )
        == "path/to/prepacked/training/artifact/prefix/number2/"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_training_artifact_key(
            instance_type="ml.p9.9xlarge"
        )
        == "do/re/mi"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_training_artifact_key(
            instance_type="ml.p9.12xlarge"
        )
        == "you/not/entertained"
    )

    assert (
        INSTANCE_TYPE_VARIANT.get_instance_specific_training_artifact_key(
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


def test_inference_configs_parsing():
    spec = {**BASE_SPEC, **INFERENCE_CONFIGS, **INFERENCE_CONFIG_RANKINGS}
    specs1 = JumpStartModelSpecs(spec)

    assert list(specs1.inference_config_components.keys()) == [
        "neuron-base",
        "neuron-inference",
        "neuron-budget",
        "gpu-inference",
        "gpu-inference-model-package",
        "gpu-inference-budget",
        "gpu-accelerated",
    ]

    # Non-overrided fields in top config
    assert specs1.model_id == "pytorch-ic-mobilenet-v2"
    assert specs1.version == "3.0.6"
    assert specs1.min_sdk_version == "2.189.0"
    assert specs1.training_supported
    assert specs1.incremental_training_supported
    assert specs1.hosting_ecr_specs == JumpStartECRSpecs(
        {
            "framework": "huggingface-llm-neuronx",
            "framework_version": "0.0.17",
            "py_version": "py310",
        }
    )
    assert specs1.training_ecr_specs == JumpStartECRSpecs(
        {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        }
    )
    assert (
        specs1.hosting_artifact_key
        == "artifacts/meta-textgeneration-llama-2-7b/neuron-inference/model/"
    )
    assert (
        specs1.training_artifact_key
        == "pytorch-training/v2.0.0/train-pytorch-ic-mobilenet-v2.tar.gz"
    )
    assert (
        specs1.hosting_script_key
        == "source-directory-tarballs/pytorch/inference/ic/v2.0.0/sourcedir.tar.gz"
    )
    assert (
        specs1.training_script_key
        == "source-directory-tarballs/pytorch/transfer_learning/ic/v2.3.0/sourcedir.tar.gz"
    )
    assert specs1.hyperparameters == [
        JumpStartHyperparameter(
            {
                "name": "train_only_top_layer",
                "type": "text",
                "options": ["True", "False"],
                "default": "True",
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "epochs",
                "type": "int",
                "default": 5,
                "scope": "algorithm",
                "min": 1,
                "max": 1000,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.001,
                "scope": "algorithm",
                "min": 1e-08,
                "max": 1,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "batch_size",
                "type": "int",
                "default": 4,
                "scope": "algorithm",
                "min": 1,
                "max": 1024,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "reinitialize_top_layer",
                "type": "text",
                "options": ["Auto", "True", "False"],
                "default": "Auto",
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
    assert specs1.inference_environment_variables == [
        JumpStartEnvironmentVariable(
            {
                "name": "SAGEMAKER_PROGRAM",
                "type": "text",
                "default": "inference.py",
                "scope": "container",
                "required_for_model_class": True,
            }
        ),
        JumpStartEnvironmentVariable(
            {
                "name": "SAGEMAKER_SUBMIT_DIRECTORY",
                "type": "text",
                "default": "/opt/ml/model/code",
                "scope": "container",
                "required_for_model_class": False,
            }
        ),
        JumpStartEnvironmentVariable(
            {
                "name": "SAGEMAKER_CONTAINER_LOG_LEVEL",
                "type": "text",
                "default": "20",
                "scope": "container",
                "required_for_model_class": False,
            }
        ),
        JumpStartEnvironmentVariable(
            {
                "name": "SAGEMAKER_MODEL_SERVER_TIMEOUT",
                "type": "text",
                "default": "3600",
                "scope": "container",
                "required_for_model_class": False,
            }
        ),
        JumpStartEnvironmentVariable(
            {
                "name": "ENDPOINT_SERVER_TIMEOUT",
                "type": "int",
                "default": 3600,
                "scope": "container",
                "required_for_model_class": True,
            }
        ),
        JumpStartEnvironmentVariable(
            {
                "name": "MODEL_CACHE_ROOT",
                "type": "text",
                "default": "/opt/ml/model",
                "scope": "container",
                "required_for_model_class": True,
            }
        ),
        JumpStartEnvironmentVariable(
            {
                "name": "SAGEMAKER_ENV",
                "type": "text",
                "default": "1",
                "scope": "container",
                "required_for_model_class": True,
            }
        ),
        JumpStartEnvironmentVariable(
            {
                "name": "SAGEMAKER_MODEL_SERVER_WORKERS",
                "type": "int",
                "default": 1,
                "scope": "container",
                "required_for_model_class": True,
            }
        ),
    ]

    # Overrided fields in top config
    assert specs1.supported_inference_instance_types == ["ml.inf2.xlarge", "ml.inf2.2xlarge"]

    config = specs1.inference_configs.get_top_config_from_ranking()

    assert config.benchmark_metrics == {
        "ml.inf2.2xlarge": [
            JumpStartBenchmarkStat(
                {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
            ),
        ]
    }
    assert len(config.config_components) == 1
    assert config.config_components["neuron-inference"] == JumpStartConfigComponent(
        "neuron-inference",
        {
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
    )
    assert list(config.config_components.keys()) == ["neuron-inference"]

    config = specs1.inference_configs.configs["gpu-inference-model-package"]
    assert config.config_components["gpu-inference-model-package"] == JumpStartConfigComponent(
        "gpu-inference-model-package",
        {
            "default_inference_instance_type": "ml.p2.xlarge",
            "supported_inference_instance_types": ["ml.p2.xlarge", "ml.p3.2xlarge"],
            "hosting_model_package_arns": {
                "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/"
                "llama2-7b-v3-740347e540da35b4ab9f6fc0ab3fed2c"
            },
        },
    )
    assert config.resolved_config.get("inference_environment_variables") == []

    spec = {
        **BASE_SPEC,
        **INFERENCE_CONFIGS,
        **INFERENCE_CONFIG_RANKINGS,
        "unrecognized-field": "blah",  # New fields in base metadata fields should be ignored
    }
    specs1 = JumpStartModelSpecs(spec)


def test_set_inference_configs():
    spec = {**BASE_SPEC, **INFERENCE_CONFIGS, **INFERENCE_CONFIG_RANKINGS}
    specs1 = JumpStartModelSpecs(spec)

    assert list(specs1.inference_config_components.keys()) == [
        "neuron-base",
        "neuron-inference",
        "neuron-budget",
        "gpu-inference",
        "gpu-inference-model-package",
        "gpu-inference-budget",
        "gpu-accelerated",
    ]

    with pytest.raises(ValueError) as error:
        specs1.set_config("invalid_name")
    assert "Cannot find Jumpstart config name invalid_name."
    "List of config names that is supported by the model: "
    "['neuron-inference', 'neuron-inference-budget', "
    "'gpu-inference-budget', 'gpu-inference', 'gpu-inference-model-package']" in str(error.value)

    assert specs1.supported_inference_instance_types == ["ml.inf2.xlarge", "ml.inf2.2xlarge"]
    specs1.set_config("gpu-inference")
    assert specs1.supported_inference_instance_types == ["ml.p2.xlarge", "ml.p3.2xlarge"]


def test_training_configs_parsing():
    spec = {**BASE_SPEC, **TRAINING_CONFIGS, **TRAINING_CONFIG_RANKINGS}
    specs1 = JumpStartModelSpecs(spec)

    assert list(specs1.training_config_components.keys()) == [
        "neuron-training",
        "gpu-training",
        "neuron-training-budget",
        "gpu-training-budget",
    ]

    # Non-overrided fields in top config
    # By default training config is not applied to model spec
    assert specs1.model_id == "pytorch-ic-mobilenet-v2"
    assert specs1.version == "3.0.6"
    assert specs1.min_sdk_version == "2.189.0"
    assert specs1.training_supported
    assert specs1.incremental_training_supported
    assert specs1.hosting_ecr_specs == JumpStartECRSpecs(
        {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        }
    )
    assert specs1.training_ecr_specs == JumpStartECRSpecs(
        {
            "framework": "pytorch",
            "framework_version": "1.10.0",
            "py_version": "py38",
        }
    )
    assert (
        specs1.hosting_artifact_key
        == "pytorch-ic/pytorch-ic-mobilenet-v2/artifacts/inference/v2.0.0/"
    )
    assert (
        specs1.training_artifact_key
        == "pytorch-training/v2.0.0/train-pytorch-ic-mobilenet-v2.tar.gz"
    )
    assert (
        specs1.hosting_script_key
        == "source-directory-tarballs/pytorch/inference/ic/v2.0.0/sourcedir.tar.gz"
    )
    assert (
        specs1.training_script_key
        == "source-directory-tarballs/pytorch/transfer_learning/ic/v2.3.0/sourcedir.tar.gz"
    )
    assert specs1.hyperparameters == [
        JumpStartHyperparameter(
            {
                "name": "train_only_top_layer",
                "type": "text",
                "options": ["True", "False"],
                "default": "True",
                "scope": "algorithm",
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "epochs",
                "type": "int",
                "default": 5,
                "scope": "algorithm",
                "min": 1,
                "max": 1000,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "learning_rate",
                "type": "float",
                "default": 0.001,
                "scope": "algorithm",
                "min": 1e-08,
                "max": 1,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "batch_size",
                "type": "int",
                "default": 4,
                "scope": "algorithm",
                "min": 1,
                "max": 1024,
            }
        ),
        JumpStartHyperparameter(
            {
                "name": "reinitialize_top_layer",
                "type": "text",
                "options": ["Auto", "True", "False"],
                "default": "Auto",
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

    config = specs1.training_configs.get_top_config_from_ranking()

    assert config.benchmark_metrics == {
        "ml.tr1n1.2xlarge": [
            JumpStartBenchmarkStat(
                {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
            ),
        ],
        "ml.tr1n1.4xlarge": [
            JumpStartBenchmarkStat(
                {"name": "Latency", "value": "50", "unit": "Tokens/S", "concurrency": 1}
            ),
        ],
    }
    assert len(config.config_components) == 1
    assert config.config_components["neuron-training"] == JumpStartConfigComponent(
        "neuron-training",
        {
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
    )
    assert list(config.config_components.keys()) == ["neuron-training"]


def test_additional_model_data_source_parsing():
    accelerated_first_rankings = {
        "inference_config_rankings": {
            "overall": {
                "description": "Overall rankings of configs",
                "rankings": [
                    "gpu-accelerated",
                    "neuron-inference",
                    "neuron-inference-budget",
                    "gpu-inference",
                    "gpu-inference-budget",
                ],
            }
        }
    }
    spec = {**BASE_SPEC, **INFERENCE_CONFIGS, **accelerated_first_rankings}
    specs1 = JumpStartModelSpecs(spec)

    config = specs1.inference_configs.get_top_config_from_ranking()

    assert config.benchmark_metrics == {
        "ml.p3.2xlarge": [
            JumpStartBenchmarkStat(
                {"name": "Latency", "value": "100", "unit": "Tokens/S", "concurrency": 1}
            ),
        ]
    }
    assert len(config.config_components) == 1
    assert config.config_components["gpu-accelerated"] == JumpStartConfigComponent(
        "gpu-accelerated",
        {
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
    )
    assert list(config.config_components.keys()) == ["gpu-accelerated"]
    assert config.resolved_config["hosting_additional_data_sources"] == {
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
    }


def test_set_inference_config():
    spec = {**BASE_SPEC, **INFERENCE_CONFIGS, **INFERENCE_CONFIG_RANKINGS}
    specs1 = JumpStartModelSpecs(spec)

    assert specs1.supported_inference_instance_types == ["ml.inf2.xlarge", "ml.inf2.2xlarge"]
    specs1.set_config("gpu-inference-budget")

    assert specs1.supported_inference_instance_types == ["ml.p2.xlarge", "ml.p3.2xlarge"]
    assert (
        specs1.hosting_artifact_key
        == "artifacts/meta-textgeneration-llama-2-7b/gpu-inference-budget/model/"
    )


def test_set_training_config():
    spec = {**BASE_SPEC, **TRAINING_CONFIGS, **TRAINING_CONFIG_RANKINGS}
    specs1 = JumpStartModelSpecs(spec)

    assert specs1.supported_training_instance_types == [
        "ml.m5.xlarge",
        "ml.c5.2xlarge",
        "ml.m4.xlarge",
    ]
    specs1.set_config("gpu-training-budget", scope=JumpStartScriptScope.TRAINING)

    assert specs1.supported_training_instance_types == ["ml.p2.xlarge", "ml.p3.2xlarge"]
    assert (
        specs1.training_artifact_key
        == "artifacts/meta-textgeneration-llama-2-7b/gpu-training-budget/model/"
    )

    with pytest.raises(ValueError) as error:
        specs1.set_config("invalid_name", scope=JumpStartScriptScope.TRAINING)
    assert "Cannot find Jumpstart config name invalid_name."
    "List of config names that is supported by the model: "
    "['neuron-training', 'neuron-training-budget', "
    "'gpu-training-budget', 'gpu-training']" in str(error.value)

    with pytest.raises(ValueError) as error:
        specs1.set_config("invalid_name", scope="unknown scope")


def test_deployment_config_metadata():
    spec = {**BASE_SPEC, **INFERENCE_CONFIGS, **INFERENCE_CONFIG_RANKINGS}
    specs = JumpStartModelSpecs(spec)
    jumpstart_config = specs.inference_configs.get_top_config_from_ranking()

    deployment_config_metadata = DeploymentConfigMetadata(
        jumpstart_config.config_name,
        jumpstart_config,
        JumpStartModelInitKwargs(
            model_id=specs.model_id,
            model_data=INIT_KWARGS.get("model_data"),
            image_uri=INIT_KWARGS.get("image_uri"),
            instance_type=INIT_KWARGS.get("instance_type"),
            env=INIT_KWARGS.get("env"),
            config_name=jumpstart_config.config_name,
        ),
    )

    json_obj = deployment_config_metadata.to_json()

    assert isinstance(json_obj, dict)
    assert json_obj["DeploymentConfigName"] == jumpstart_config.config_name
    for key in json_obj["BenchmarkMetrics"]:
        assert len(json_obj["BenchmarkMetrics"][key]) == len(
            jumpstart_config.benchmark_metrics.get(key)
        )
    assert json_obj["AccelerationConfigs"] == jumpstart_config.resolved_config.get(
        "acceleration_configs"
    )
    assert json_obj["DeploymentArgs"]["ImageUri"] == INIT_KWARGS.get("image_uri")
    assert json_obj["DeploymentArgs"]["ModelData"] == INIT_KWARGS.get("model_data")
    assert json_obj["DeploymentArgs"]["Environment"] == INIT_KWARGS.get("env")
    assert json_obj["DeploymentArgs"]["InstanceType"] == INIT_KWARGS.get("instance_type")
