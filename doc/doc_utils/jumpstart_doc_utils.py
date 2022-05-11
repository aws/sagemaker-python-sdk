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
from urllib import request
import json
from packaging.version import Version
from enum import Enum


class Tasks(str, Enum):
    """The ML task name as referenced in the infix of the model ID."""

    IC = "ic"
    OD = "od"
    OD1 = "od1"
    SEMSEG = "semseg"
    IS = "is"
    TC = "tc"
    SPC = "spc"
    EQA = "eqa"
    TEXT_GENERATION = "textgeneration"
    IC_EMBEDDING = "icembedding"
    TC_EMBEDDING = "tcembedding"
    NER = "ner"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    TABULAR_REGRESSION = "regression"
    TABULAR_CLASSIFICATION = "classification"


class ProblemTypes(str, Enum):
    """Possible problem types for JumpStart models."""

    IMAGE_CLASSIFICATION = "Image Classification"
    IMAGE_EMBEDDING = "Image Embedding"
    OBJECT_DETECTION = "Object Detection"
    SEMANTIC_SEGMENTATION = "Semantic Segmentation"
    INSTANCE_SEGMENTATION = "Instance Segmentation"
    TEXT_CLASSIFICATION = "Text Classification"
    TEXT_EMBEDDING = "Text Embedding"
    QUESTION_ANSWERING = "Question Answering"
    SENTENCE_PAIR_CLASSIFICATION = "Sentence Pair Classification"
    TEXT_GENERATION = "Text Generation"
    TEXT_SUMMARIZATION = "Text Summarization"
    MACHINE_TRANSLATION = "Machine Translation"
    NAMED_ENTITY_RECOGNITION = "Named Entity Recognition"
    TABULAR_REGRESSION = "Regression"
    TABULAR_CLASSIFICATION = "Classification"


JUMPSTART_REGION = "eu-west-2"
SDK_MANIFEST_FILE = "models_manifest.json"
JUMPSTART_BUCKET_BASE_URL = "https://jumpstart-cache-prod-{}.s3.{}.amazonaws.com".format(
    JUMPSTART_REGION, JUMPSTART_REGION
)
TASK_MAP = {
    Tasks.IC: ProblemTypes.IMAGE_CLASSIFICATION,
    Tasks.IC_EMBEDDING: ProblemTypes.IMAGE_EMBEDDING,
    Tasks.OD: ProblemTypes.OBJECT_DETECTION,
    Tasks.OD1: ProblemTypes.OBJECT_DETECTION,
    Tasks.SEMSEG: ProblemTypes.SEMANTIC_SEGMENTATION,
    Tasks.IS: ProblemTypes.INSTANCE_SEGMENTATION,
    Tasks.TC: ProblemTypes.TEXT_CLASSIFICATION,
    Tasks.TC_EMBEDDING: ProblemTypes.TEXT_EMBEDDING,
    Tasks.EQA: ProblemTypes.QUESTION_ANSWERING,
    Tasks.SPC: ProblemTypes.SENTENCE_PAIR_CLASSIFICATION,
    Tasks.TEXT_GENERATION: ProblemTypes.TEXT_GENERATION,
    Tasks.SUMMARIZATION: ProblemTypes.TEXT_SUMMARIZATION,
    Tasks.TRANSLATION: ProblemTypes.MACHINE_TRANSLATION,
    Tasks.NER: ProblemTypes.NAMED_ENTITY_RECOGNITION,
    Tasks.TABULAR_REGRESSION: ProblemTypes.TABULAR_REGRESSION,
    Tasks.TABULAR_CLASSIFICATION: ProblemTypes.TABULAR_CLASSIFICATION,
}


def get_jumpstart_sdk_manifest():
    url = "{}/{}".format(JUMPSTART_BUCKET_BASE_URL, SDK_MANIFEST_FILE)
    with request.urlopen(url) as f:
        models_manifest = f.read().decode("utf-8")
    return json.loads(models_manifest)


def get_jumpstart_sdk_spec(key):
    url = "{}/{}".format(JUMPSTART_BUCKET_BASE_URL, key)
    with request.urlopen(url) as f:
        model_spec = f.read().decode("utf-8")
    return json.loads(model_spec)


def get_model_task(id):
    task_short = id.split("-")[1]
    return TASK_MAP[task_short] if task_short in TASK_MAP else "Source"


def create_jumpstart_model_table():
    sdk_manifest = get_jumpstart_sdk_manifest()
    sdk_manifest_top_versions_for_models = {}

    for model in sdk_manifest:
        if model["model_id"] not in sdk_manifest_top_versions_for_models:
            sdk_manifest_top_versions_for_models[model["model_id"]] = model
        else:
            if Version(
                sdk_manifest_top_versions_for_models[model["model_id"]]["version"]
            ) < Version(model["version"]):
                sdk_manifest_top_versions_for_models[model["model_id"]] = model

    file_content = []

    file_content.append("==================================\n")
    file_content.append("JumpStart Available Model Table\n")
    file_content.append("==================================\n")
    file_content.append(
        """
    JumpStart for the SageMaker Python SDK uses model ids and model versions to access the necessary
    utilities. This table serves to provide the core material plus some extra information that can be useful
    in selecting the correct model id and corresponding parameters.\n
    """
    )
    file_content.append(
        """
    If you want to automatically use the latest version of the model, use "*" for the `model_version` attribute.
    We highly suggest pinning an exact model version however.\n
    """
    )
    file_content.append(
        """
    Click on the Problem Type to navigate to the source of the model.\n
    """
    )
    file_content.append("\n")
    file_content.append(".. list-table:: Available Models\n")
    file_content.append("   :widths: 50 20 20 20 30\n")
    file_content.append("   :header-rows: 1\n")
    file_content.append("   :class: datatable\n")
    file_content.append("\n")
    file_content.append("   * - Model ID\n")
    file_content.append("     - Fine Tunable?\n")
    file_content.append("     - Latest Version\n")
    file_content.append("     - Min SDK Version\n")
    file_content.append("     - Problem Type/Source\n")

    for model in sdk_manifest_top_versions_for_models.values():
        model_spec = get_jumpstart_sdk_spec(model["spec_key"])
        model_task = get_model_task(model_spec["model_id"])
        file_content.append("   * - {}\n".format(model_spec["model_id"]))
        file_content.append("     - {}\n".format(model_spec["training_supported"]))
        file_content.append("     - {}\n".format(model["version"]))
        file_content.append("     - {}\n".format(model["min_version"]))
        file_content.append("     - `{} <{}>`__\n".format(model_task, model_spec["url"]))

    f = open("doc_utils/jumpstart.rst", "w")
    f.writelines(file_content)
