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


class Frameworks(str, Enum):
    """Possible frameworks for JumpStart models"""

    TENSORFLOW = "Tensorflow Hub"
    PYTORCH = "Pytorch Hub"
    HUGGINGFACE = "HuggingFace"
    CATBOOST = "Catboost"
    GLUONCV = "GluonCV"
    LIGHTGBM = "LightGBM"
    XGBOOST = "XGBoost"
    SCIKIT_LEARN = "ScikitLearn"
    SOURCE = "Source"


JUMPSTART_REGION = "eu-west-2"
SDK_MANIFEST_FILE = "models_manifest.json"
PROPRIETARY_SDK_MANIFEST_FILE = "proprietary-sdk-manifest.json"
JUMPSTART_BUCKET_BASE_URL = "https://jumpstart-cache-prod-{}.s3.{}.amazonaws.com".format(
    JUMPSTART_REGION, JUMPSTART_REGION
)
PROPRIETARY_DOC_BUCKET = "https://jumpstart-cache-prod-us-west-2.s3.us-west-2.amazonaws.com"

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

TO_FRAMEWORK = {
    "Tensorflow Hub": Frameworks.TENSORFLOW,
    "Pytorch Hub": Frameworks.PYTORCH,
    "HuggingFace": Frameworks.HUGGINGFACE,
    "Catboost": Frameworks.CATBOOST,
    "GluonCV": Frameworks.GLUONCV,
    "LightGBM": Frameworks.LIGHTGBM,
    "XGBoost": Frameworks.XGBOOST,
    "ScikitLearn": Frameworks.SCIKIT_LEARN,
    "Source": Frameworks.SOURCE,
}


MODALITY_MAP = {
    (Tasks.IC, Frameworks.PYTORCH): "algorithms/vision/image_classification_pytorch.rst",
    (Tasks.IC, Frameworks.TENSORFLOW): "algorithms/vision/image_classification_tensorflow.rst",
    (Tasks.IC_EMBEDDING, Frameworks.TENSORFLOW): "algorithms/vision/image_embedding_tensorflow.rst",
    (Tasks.IS, Frameworks.GLUONCV): "algorithms/vision/instance_segmentation_mxnet.rst",
    (Tasks.OD, Frameworks.GLUONCV): "algorithms/vision/object_detection_mxnet.rst",
    (Tasks.OD, Frameworks.PYTORCH): "algorithms/vision/object_detection_pytorch.rst",
    (Tasks.OD, Frameworks.TENSORFLOW): "algorithms/vision/object_detection_tensorflow.rst",
    (Tasks.SEMSEG, Frameworks.GLUONCV): "algorithms/vision/semantic_segmentation_mxnet.rst",
    (
        Tasks.TRANSLATION,
        Frameworks.HUGGINGFACE,
    ): "algorithms/text/machine_translation_hugging_face.rst",
    (Tasks.NER, Frameworks.GLUONCV): "algorithms/text/named_entity_recognition_hugging_face.rst",
    (Tasks.EQA, Frameworks.PYTORCH): "algorithms/text/question_answering_pytorch.rst",
    (
        Tasks.SPC,
        Frameworks.HUGGINGFACE,
    ): "algorithms/text/sentence_pair_classification_hugging_face.rst",
    (
        Tasks.SPC,
        Frameworks.TENSORFLOW,
    ): "algorithms/text/sentence_pair_classification_tensorflow.rst",
    (Tasks.TC, Frameworks.TENSORFLOW): "algorithms/text/text_classification_tensorflow.rst",
    (
        Tasks.TC_EMBEDDING,
        Frameworks.GLUONCV,
    ): "algorithms/vision/text_embedding_tensorflow_mxnet.rst",
    (
        Tasks.TC_EMBEDDING,
        Frameworks.TENSORFLOW,
    ): "algorithms/vision/text_embedding_tensorflow_mxnet.rst",
    (
        Tasks.TEXT_GENERATION,
        Frameworks.HUGGINGFACE,
    ): "algorithms/text/text_generation_hugging_face.rst",
    (
        Tasks.SUMMARIZATION,
        Frameworks.HUGGINGFACE,
    ): "algorithms/text/text_summarization_hugging_face.rst",
}


def get_public_s3_json_object(url):
    with request.urlopen(url) as f:
        models_manifest = f.read().decode("utf-8")
    return json.loads(models_manifest)


def get_jumpstart_sdk_manifest():
    return get_public_s3_json_object(f"{JUMPSTART_BUCKET_BASE_URL}/{SDK_MANIFEST_FILE}")


def get_proprietary_sdk_manifest():
    return get_public_s3_json_object(f"{PROPRIETARY_DOC_BUCKET}/{PROPRIETARY_SDK_MANIFEST_FILE}")


def get_jumpstart_sdk_spec(s3_key: str):
    return get_public_s3_json_object(f"{JUMPSTART_BUCKET_BASE_URL}/{s3_key}")


def get_proprietary_sdk_spec(s3_key: str):
    return get_public_s3_json_object(f"{PROPRIETARY_DOC_BUCKET}/{s3_key}")


def get_model_task(id):
    task_short = id.split("-")[1]
    return TASK_MAP[task_short] if task_short in TASK_MAP else "Source"


def get_string_model_task(id):
    return id.split("-")[1]


def get_model_source(url):
    if "tfhub" in url:
        return "Tensorflow Hub"
    if "pytorch" in url:
        return "Pytorch Hub"
    if "huggingface" in url:
        return "HuggingFace"
    if "catboost" in url:
        return "Catboost"
    if "gluon" in url:
        return "GluonCV"
    if "lightgbm" in url:
        return "LightGBM"
    if "xgboost" in url:
        return "XGBoost"
    if "scikit" in url:
        return "ScikitLearn"
    else:
        return "Source"


def create_proprietary_model_table():
    proprietary_content_intro = []
    proprietary_content_intro.append("\n")
    proprietary_content_intro.append(".. list-table:: Available Proprietary Models\n")
    proprietary_content_intro.append("   :widths: 50 20 20 20 20\n")
    proprietary_content_intro.append("   :header-rows: 1\n")
    proprietary_content_intro.append("   :class: datatable\n")
    proprietary_content_intro.append("\n")
    proprietary_content_intro.append("   * - Model ID\n")
    proprietary_content_intro.append("     - Fine Tunable?\n")
    proprietary_content_intro.append("     - Supported Version\n")
    proprietary_content_intro.append("     - Min SDK Version\n")
    proprietary_content_intro.append("     - Source\n")

    sdk_manifest = get_proprietary_sdk_manifest()
    sdk_manifest_top_versions_for_models = {}

    for model in sdk_manifest:
        if model["model_id"] not in sdk_manifest_top_versions_for_models:
            sdk_manifest_top_versions_for_models[model["model_id"]] = model
        else:
            if str(sdk_manifest_top_versions_for_models[model["model_id"]]["version"]) < str(
                model["version"]
            ):
                sdk_manifest_top_versions_for_models[model["model_id"]] = model

    proprietary_content_entries = []
    for model in sdk_manifest_top_versions_for_models.values():
        model_spec = get_proprietary_sdk_spec(model["spec_key"])
        proprietary_content_entries.append("   * - {}\n".format(model_spec["model_id"]))
        proprietary_content_entries.append("     - {}\n".format(False))  # TODO: support training
        proprietary_content_entries.append("     - {}\n".format(model["version"]))
        proprietary_content_entries.append("     - {}\n".format(model["min_version"]))
        proprietary_content_entries.append(
            "     - `{} <{}>`__ |external-link|\n".format("Source", model_spec.get("url"))
        )
    return proprietary_content_intro + proprietary_content_entries + ["\n"]


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

    file_content_intro = []

    file_content_intro.append(".. _all-pretrained-models:\n\n")
    file_content_intro.append(".. |external-link| raw:: html\n\n")
    file_content_intro.append('   <i class="fa fa-external-link"></i>\n\n')

    file_content_intro.append("================================================\n")
    file_content_intro.append("Built-in Algorithms with pre-trained Model Table\n")
    file_content_intro.append("================================================\n")
    file_content_intro.append(
        """
    The SageMaker Python SDK uses model IDs and model versions to access the necessary
    utilities for pre-trained models. This table serves to provide the core material plus
    some extra information that can be useful in selecting the correct model ID and
    corresponding parameters.\n"""
    )
    file_content_intro.append(
        """
    If you want to automatically use the latest version of the model, use "*" for the `model_version` attribute.
    We highly suggest pinning an exact model version however.\n"""
    )
    file_content_intro.append(
        """
    These models are also available through the
    `JumpStart UI in SageMaker Studio <https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html>`__\n"""
    )
    file_content_intro.append("\n")
    file_content_intro.append(".. list-table:: Available Models\n")
    file_content_intro.append("   :widths: 50 20 20 20 30 20\n")
    file_content_intro.append("   :header-rows: 1\n")
    file_content_intro.append("   :class: datatable\n")
    file_content_intro.append("\n")
    file_content_intro.append("   * - Model ID\n")
    file_content_intro.append("     - Fine Tunable?\n")
    file_content_intro.append("     - Latest Version\n")
    file_content_intro.append("     - Min SDK Version\n")
    file_content_intro.append("     - Problem Type\n")
    file_content_intro.append("     - Source\n")

    dynamic_table_files = []
    open_weight_content_entries = []

    for model in sdk_manifest_top_versions_for_models.values():
        model_spec = get_jumpstart_sdk_spec(model["spec_key"])
        model_task = get_model_task(model_spec["model_id"])
        string_model_task = get_string_model_task(model_spec["model_id"])
        model_source = get_model_source(model_spec["url"])
        open_weight_content_entries.append("   * - {}\n".format(model_spec["model_id"]))
        open_weight_content_entries.append("     - {}\n".format(model_spec["training_supported"]))
        open_weight_content_entries.append("     - {}\n".format(model["version"]))
        open_weight_content_entries.append("     - {}\n".format(model["min_version"]))
        open_weight_content_entries.append("     - {}\n".format(model_task))
        open_weight_content_entries.append(
            "     - `{} <{}>`__ |external-link|\n".format(model_source, model_spec["url"])
        )

        if (string_model_task, TO_FRAMEWORK[model_source]) in MODALITY_MAP:
            file_content_single_entry = []

            if (
                MODALITY_MAP[(string_model_task, TO_FRAMEWORK[model_source])]
                not in dynamic_table_files
            ):
                file_content_single_entry.append("\n")
                file_content_single_entry.append(".. list-table:: Available Models\n")
                file_content_single_entry.append("   :widths: 50 20 20 20 20\n")
                file_content_single_entry.append("   :header-rows: 1\n")
                file_content_single_entry.append("   :class: datatable\n")
                file_content_single_entry.append("\n")
                file_content_single_entry.append("   * - Model ID\n")
                file_content_single_entry.append("     - Fine Tunable?\n")
                file_content_single_entry.append("     - Latest Version\n")
                file_content_single_entry.append("     - Min SDK Version\n")
                file_content_single_entry.append("     - Source\n")

                dynamic_table_files.append(
                    MODALITY_MAP[(string_model_task, TO_FRAMEWORK[model_source])]
                )

            file_content_single_entry.append("   * - {}\n".format(model_spec["model_id"]))
            file_content_single_entry.append("     - {}\n".format(model_spec["training_supported"]))
            file_content_single_entry.append("     - {}\n".format(model["version"]))
            file_content_single_entry.append("     - {}\n".format(model["min_version"]))
            file_content_single_entry.append(
                "     - `{} <{}>`__\n".format(model_source, model_spec["url"])
            )
            f = open(MODALITY_MAP[(string_model_task, TO_FRAMEWORK[model_source])], "a")
            f.writelines(file_content_single_entry)
            f.close()

    proprietary_content_entries = create_proprietary_model_table()

    f = open("doc_utils/pretrainedmodels.rst", "a")
    f.writelines(file_content_intro)
    f.writelines(open_weight_content_entries)
    f.writelines(proprietary_content_entries)
    f.close()
