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

PROTOTYPICAL_MODEL_SPECS_DICT = {
    "pytorch-eqa-bert-base-cased": {
        "model_id": "pytorch-eqa-bert-base-cased",
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
        "training_script_key": "source-directory-tarballs/pytorch/transfer_learning/eqa/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.5.0",
            "framework": "pytorch",
            "py_version": "py3",
        },
        "training_artifact_key": "pytorch-training/train-pytorch-eqa-bert-base-cased.tar.gz",
    },
    "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1": {
        "model_id": "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1",
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
        "training_script_key": "source-directory-tarballs/tensorflow/transfer_learning/ic/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "2.3",
            "framework": "tensorflow",
            "py_version": "py37",
        },
        "training_artifact_key": "tensorflow-training/"
        "train-tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1.tar.gz",
    },
    "mxnet-semseg-fcn-resnet50-ade": {
        "model_id": "mxnet-semseg-fcn-resnet50-ade",
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
        "training_script_key": "source-directory-tarballs/mxnet/transfer_learning/semseg/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.7.0",
            "framework": "mxnet",
            "py_version": "py3",
        },
        "training_artifact_key": "mxnet-training/train-mxnet-semseg-fcn-resnet50-ade.tar.gz",
    },
    "huggingface-spc-bert-base-cased": {
        "model_id": "huggingface-spc-bert-base-cased",
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
        "training_script_key": "source-directory-tarballs/huggingface/transfer_learning/spc/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.6.0",
            "framework": "huggingface",
            "huggingface_transformers_version": "4.4.2",
            "py_version": "py36",
        },
        "training_artifact_key": "huggingface-training/train-huggingface-spc-bert-base-cased.tar.gz",
    },
    "lightgbm-classification-model": {
        "model_id": "lightgbm-classification-model",
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
        "hosting_script_key": "source-directory-tarballs/lightgbm/inference/classification/"
        "v1.0.0/sourcedir.tar.gz",
        "training_script_key": "source-directory-tarballs/lightgbm/transfer_learning/"
        "classification/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.9.0",
            "framework": "pytorch",
            "py_version": "py38",
        },
        "training_artifact_key": "lightgbm-training/train-lightgbm-classification-model.tar.gz",
    },
    "catboost-classification-model": {
        "model_id": "catboost-classification-model",
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
        "training_script_key": "source-directory-tarballs/catboost/transfer_learning/"
        "classification/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.9.0",
            "framework": "pytorch",
            "py_version": "py38",
        },
        "training_artifact_key": "catboost-training/train-catboost-classification-model.tar.gz",
    },
    "xgboost-classification-model": {
        "model_id": "xgboost-classification-model",
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
        "training_script_key": "source-directory-tarballs/xgboost/transfer_learning/"
        "classification/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "1.3-1",
            "framework": "xgboost",
            "py_version": "py3",
        },
        "training_artifact_key": "xgboost-training/train-xgboost-classification-model.tar.gz",
    },
    "sklearn-classification-linear": {
        "model_id": "sklearn-classification-linear",
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
        "training_script_key": "source-directory-tarballs/sklearn/transfer_learning/"
        "classification/v1.0.0/sourcedir.tar.gz",
        "training_ecr_specs": {
            "framework_version": "0.23-1",
            "framework": "sklearn",
            "py_version": "py3",
        },
        "training_artifact_key": "sklearn-training/train-sklearn-classification-linear.tar.gz",
    },
}

BASE_SPEC = {
    "model_id": "pytorch-ic-mobilenet-v2",
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
    "hyperparameters": {
        "adam-learning-rate": {"type": "float", "default": 0.05, "min": 1e-08, "max": 1},
        "epochs": {"type": "int", "default": 3, "min": 1, "max": 1000},
        "batch-size": {"type": "int", "default": 4, "min": 1, "max": 1024},
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
