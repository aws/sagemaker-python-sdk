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

from enum import Enum
from typing import Dict
from typing import Optional
from typing import Union
import os

"""
This module has support for multiple input data types supported by all the JumpStart
model offerings.
"""


def _to_s3_path(filename: str, s3_prefix: Optional[str]) -> str:
    return filename if not s3_prefix else f"{s3_prefix}/{filename}"


_NB_ASSETS_S3_FOLDER = "inference-notebook-assets"
_TF_FLOWERS_S3_FOLDER = "training-datasets/tf_flowers"

TMP_DIRECTORY_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir)), "tmp"
)

ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID = "JUMPSTART_SDK_TEST_SUITE_ID"

ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME = "JUMPSTART_SDK_TEST_HUB_NAME"

JUMPSTART_TAG = "JumpStart-SDK-Integ-Test-Suite-Id"

HUB_NAME_PREFIX = "PySDK-HubTest-"

TRAINING_DATASET_MODEL_DICT = {
    ("huggingface-spc-bert-base-cased", "1.0.0"): ("training-datasets/QNLI-tiny/"),
    ("huggingface-spc-bert-base-cased", "1.2.3"): ("training-datasets/QNLI-tiny/"),
    ("huggingface-spc-bert-base-cased", "*"): ("training-datasets/QNLI-tiny/"),
    ("js-trainable-model", "*"): ("training-datasets/QNLI-tiny/"),
    ("meta-textgeneration-llama-2-7b", "*"): ("training-datasets/sec_amazon/"),
    ("meta-textgeneration-llama-2-7b", "2.*"): ("training-datasets/sec_amazon/"),
    ("meta-textgeneration-llama-2-7b", "3.*"): ("training-datasets/sec_amazon/"),
    ("meta-textgeneration-llama-2-7b", "4.*"): ("training-datasets/sec_amazon/"),
    ("meta-textgenerationneuron-llama-2-7b", "*"): ("training-datasets/sec_amazon/"),
}


class ContentType(str, Enum):
    """Possible value for content type argument of SageMakerRuntime.invokeEndpoint."""

    X_IMAGE = "application/x-image"
    LIST_TEXT = "application/list-text"
    X_TEXT = "application/x-text"
    TEXT_CSV = "text/csv"


class InferenceImageFilename(str, Enum):
    """Filename of the inference asset in JumpStart distribution buckets."""

    DOG = "dog.jpg"
    CAT = "cat.jpg"
    DAISY = "100080576_f52e8ee070_n.jpg"
    DAISY_2 = "10140303196_b88d3d6cec.jpg"
    ROSE = "102501987_3cdb8e5394_n.jpg"
    NAXOS_TAVERNA = "Naxos_Taverna.jpg"
    PEDESTRIAN = "img_pedestrian.png"


class InferenceTabularDataname(str, Enum):
    """Filename of the tabular data example in JumpStart distribution buckets."""

    REGRESSION_ONEHOT = "regressonehot_data.csv"
    REGRESSION = "regress_data.csv"
    MULTICLASS = "multiclass_data.csv"


class ClassLabelFile(str, Enum):
    """Filename in JumpStart distribution buckets for the map of the class index to human readable labels."""

    IMAGE_NET = "ImageNetLabels.txt"


TEST_ASSETS_SPECS: Dict[
    Union[InferenceImageFilename, InferenceTabularDataname, ClassLabelFile], str
] = {
    InferenceImageFilename.DOG: _to_s3_path(InferenceImageFilename.DOG, _NB_ASSETS_S3_FOLDER),
    InferenceImageFilename.CAT: _to_s3_path(InferenceImageFilename.CAT, _NB_ASSETS_S3_FOLDER),
    InferenceImageFilename.DAISY: _to_s3_path(
        InferenceImageFilename.DAISY, f"{_TF_FLOWERS_S3_FOLDER}/daisy"
    ),
    InferenceImageFilename.DAISY_2: _to_s3_path(
        InferenceImageFilename.DAISY_2, f"{_TF_FLOWERS_S3_FOLDER}/daisy"
    ),
    InferenceImageFilename.ROSE: _to_s3_path(
        InferenceImageFilename.ROSE, f"{_TF_FLOWERS_S3_FOLDER}/roses"
    ),
    InferenceImageFilename.NAXOS_TAVERNA: _to_s3_path(
        InferenceImageFilename.NAXOS_TAVERNA, _NB_ASSETS_S3_FOLDER
    ),
    InferenceImageFilename.PEDESTRIAN: _to_s3_path(
        InferenceImageFilename.PEDESTRIAN, _NB_ASSETS_S3_FOLDER
    ),
    ClassLabelFile.IMAGE_NET: _to_s3_path(ClassLabelFile.IMAGE_NET, _NB_ASSETS_S3_FOLDER),
    InferenceTabularDataname.REGRESSION_ONEHOT: _to_s3_path(
        InferenceTabularDataname.REGRESSION_ONEHOT, _NB_ASSETS_S3_FOLDER
    ),
    InferenceTabularDataname.REGRESSION: _to_s3_path(
        InferenceTabularDataname.REGRESSION, _NB_ASSETS_S3_FOLDER
    ),
    InferenceTabularDataname.MULTICLASS: _to_s3_path(
        InferenceTabularDataname.MULTICLASS, _NB_ASSETS_S3_FOLDER
    ),
}
