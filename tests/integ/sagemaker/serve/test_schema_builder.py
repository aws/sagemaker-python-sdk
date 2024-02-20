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

from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.utils import task

import pytest

from sagemaker.serve.utils.exceptions import TaskNotFoundException
from tests.integ.sagemaker.serve.constants import (
    PYTHON_VERSION_IS_NOT_310,
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
)

from tests.integ.timeout import timeout
from tests.integ.utils import cleanup_model_resources

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


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    reason="Testing Schema Builder Simplification feature",
)
def test_model_builder_happy_path_with_only_model_id_question_answering(
    sagemaker_session, gpu_instance_type
):
    model_builder = ModelBuilder(model="bert-large-uncased-whole-word-masking-finetuned-squad")

    model = model_builder.build(sagemaker_session=sagemaker_session)

    assert model is not None
    assert model_builder.schema_builder is not None

    inputs, outputs = task.retrieve_local_schemas("question-answering")
    assert model_builder.schema_builder.sample_input == inputs
    assert model_builder.schema_builder.sample_output == outputs

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        caught_ex = None
        try:
            iam_client = sagemaker_session.boto_session.client("iam")
            role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            predictor = model.deploy(
                role=role_arn, instance_count=1, instance_type=gpu_instance_type
            )

            predicted_outputs = predictor.predict(inputs)
            assert predicted_outputs is not None

        except Exception as e:
            caught_ex = e
        finally:
            cleanup_model_resources(
                sagemaker_session=model_builder.sagemaker_session,
                model_name=model.name,
                endpoint_name=model.endpoint_name,
            )
            if caught_ex:
                logger.exception(caught_ex)
                assert (
                    False
                ), f"{caught_ex} was thrown when running transformers sagemaker endpoint test"


def test_model_builder_negative_path(sagemaker_session):
    model_builder = ModelBuilder(model="CompVis/stable-diffusion-v1-4")

    with pytest.raises(
        TaskNotFoundException,
        match="Error Message: Schema builder for text-to-image could not be found.",
    ):
        model_builder.build(sagemaker_session=sagemaker_session)
