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
            "p2": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
            "p3": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
            "ml.p3.200xlarge": {"regional_properties": {"image_uri": "$gpu_image_uri_2"}},
            "p4": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
            "g4dn": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
            "g9": {"regional_properties": {"image_uri": "$gpu_image_uri"}},
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
                "properties": {"environment_variables": {"TENSOR_PARALLEL_DEGREE": "4"}}
            },
            "g6": {"properties": {"environment_variables": {"BLAH": "4"}}},
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
