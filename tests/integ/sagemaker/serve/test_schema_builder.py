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

from sagemaker import task
from sagemaker.serve.builder.model_builder import ModelBuilder

import logging

logger = logging.getLogger(__name__)


def test_model_builder_happy_path_with_only_model_id_fill_mask(sagemaker_session):
    model_builder = ModelBuilder(model="bert-base-uncased")

    model = model_builder.build(sagemaker_session=sagemaker_session)

    assert model is not None
    assert model_builder.schema_builder is not None

    inputs, outputs = task.retrieve_local_schemas("fill-mask")
    assert model_builder.schema_builder.sample_input == inputs
    assert model_builder.schema_builder.sample_output == outputs


def test_model_builder_happy_path_with_only_model_id_question_answering(sagemaker_session):
    model_builder = ModelBuilder(model="bert-large-uncased-whole-word-masking-finetuned-squad")

    model = model_builder.build(sagemaker_session=sagemaker_session)

    assert model is not None
    assert model_builder.schema_builder is not None

    inputs, outputs = task.retrieve_local_schemas("question-answering")
    assert model_builder.schema_builder.sample_input == inputs
    assert model_builder.schema_builder.sample_output == outputs
