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
"""Accessors to retrieve task fallback input/output schema"""
from __future__ import absolute_import

import json
import os
from enum import Enum
from typing import Any, Tuple


class TASK(str, Enum):
    """Enum class for tasks"""

    AUDIO_CLASSIFICATION = "audio-classification"
    AUDIO_TO_AUDIO = "audio-to-audio"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    CONVERSATIONAL = "conversational"
    DEPTH_ESTIMATION = "depth-estimation"
    DOCUMENT_QUESTION_ANSWERING = "document-question-answering"
    FEATURE_EXTRACTION = "feature-extraction"
    FILL_MASK = "fill-mask"
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_SEGMENTATION = "image-segmentation"
    IMAGE_TO_IMAGE = "image-to-image"
    IMAGE_TO_TEXT = "image-to-text"
    MASK_GENERATION = "mask-generation"
    OBJECT_DETECTION = "object-detection"
    PLACEHOLDER = "placeholder"
    QUESTION_ANSWERING = "question-answering"
    REINFORCEMENT_LEARNING = "reinforcement-learning"
    SENTENCE_SIMILARITY = "sentence-similarity"
    SUMMARIZATION = "summarization"
    TABLE_QUESTION_ANSWERING = "table-question-answering"
    TABULAR_CLASSIFICATION = "tabular-classification"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_GENERATION = "text-generation"
    TEXT_TO_AUDIO = "text-to-audio"
    TEXT_TO_SPEECH = "text-to-speech"
    TEXT_TO_VIDEO = "text-to-video"
    TEXT_2_TEXT_GENERATION = "text2text-generation"
    TOKEN_CLASSIFICATION = "token-classification"
    TRANSLATION = "translation"
    UNCONDITIONAL_IMAGE_GENERATION = "unconditional-image-generation"
    VIDEO_CLASSIFICATION = "video-classification"
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    ZERO_SHOT_OBJECT_DETECTION = "zero-shot-object-detection"


def retrieve_local_schemas(task: str) -> Tuple[Any, Any]:
    """Retrieves task sample inputs and outputs locally.

    Args:
        task (str): Required, the task name

    Returns:
        Tuple[Any, Any]: A tuple that contains the sample input,
        at index 0, and output schema, at index 1.

    Raises:
        ValueError: If no tasks config found or the task does not exist in the local config.
    """
    task_path = os.path.join(os.path.dirname(__file__), "image_uri_config", "tasks.json")
    try:
        with open(task_path) as f:
            task_config = json.load(f)
            task_schema = task_config.get(task, None)

            if task_schema is None:
                raise ValueError(f"Could not find {task} task schema.")

            sample_schema = (
                task_schema["inputs"]["properties"],
                task_schema["outputs"]["properties"],
            )
        return sample_schema

    except FileNotFoundError:
        raise ValueError("Could not find tasks config file.")
