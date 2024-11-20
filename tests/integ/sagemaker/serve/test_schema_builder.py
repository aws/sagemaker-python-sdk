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
from sagemaker_schema_inference_artifacts.huggingface import remote_schema_retriever
from tests.integ.sagemaker.serve.constants import (
    PYTHON_VERSION_IS_NOT_310,
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
)

from tests.integ.timeout import timeout
from tests.integ.utils import cleanup_model_resources

import logging

logger = logging.getLogger(__name__)


def test_model_builder_happy_path_with_only_model_id_text_generation(sagemaker_session):
    model_builder = ModelBuilder(model="HuggingFaceH4/zephyr-7b-beta")

    model = model_builder.build(sagemaker_session=sagemaker_session)

    assert model is not None
    assert model_builder.schema_builder is not None

    inputs, outputs = task.retrieve_local_schemas("text-generation")
    assert model_builder.schema_builder.sample_input["inputs"] == inputs["inputs"]
    assert model_builder.schema_builder.sample_output == outputs


def test_model_builder_negative_path(sagemaker_session):
    # A model-task combo unsupported by both the local and remote schema fallback options. (eg: text-to-video)
    model_builder = ModelBuilder(model="ByteDance/AnimateDiff-Lightning")
    with pytest.raises(
        TaskNotFoundException,
        match="Error Message: HuggingFace Schema builder samples for text-to-video could not be found locally or "
        "via remote.",
    ):
        model_builder.build(sagemaker_session=sagemaker_session)


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    reason="Testing Schema Builder Simplification feature - Local Schema",
)
@pytest.mark.parametrize(
    "model_id, task_provided, instance_type_provided, container_startup_timeout",
    [
        (
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "text-classification",
            "ml.m5.xlarge",
            None,
        ),
        (
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "text-classification",
            "ml.m5.xlarge",
            None,
        ),
        ("HuggingFaceH4/zephyr-7b-beta", "text-generation", "ml.g5.2xlarge", 900),
        ("HuggingFaceH4/zephyr-7b-alpha", "text-generation", "ml.g5.2xlarge", 900),
    ],
)
def test_model_builder_happy_path_with_task_provided_local_schema_mode(
    model_id, task_provided, sagemaker_session, instance_type_provided, container_startup_timeout
):
    model_builder = ModelBuilder(
        model=model_id,
        model_metadata={"HF_TASK": task_provided},
        instance_type=instance_type_provided,
    )

    model = model_builder.build(sagemaker_session=sagemaker_session)

    assert model is not None
    assert model_builder.schema_builder is not None

    inputs, outputs = task.retrieve_local_schemas(task_provided)
    if task_provided == "text-generation":
        # ignore 'tokens' and other metadata in this case
        assert model_builder.schema_builder.sample_input["inputs"] == inputs["inputs"]
    else:
        assert model_builder.schema_builder.sample_input == inputs
    assert model_builder.schema_builder.sample_output == outputs

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        caught_ex = None
        try:
            iam_client = sagemaker_session.boto_session.client("iam")
            role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            if container_startup_timeout:
                predictor = model.deploy(
                    role=role_arn,
                    initial_instance_count=1,
                    instance_type=instance_type_provided,
                    container_startup_health_check_timeout=container_startup_timeout,
                )
            else:
                predictor = model.deploy(
                    role=role_arn, initial_instance_count=1, instance_type=instance_type_provided
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


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    reason="Testing Schema Builder Simplification feature - Remote Schema",
)
@pytest.mark.parametrize(
    "model_id, task_provided, instance_type_provided",
    [
        ("google-bert/bert-base-uncased", "fill-mask", "ml.m5.xlarge"),
        ("google-bert/bert-base-cased", "fill-mask", "ml.m5.xlarge"),
        (
            "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
            "question-answering",
            "ml.m5.xlarge",
        ),
        ("deepset/roberta-base-squad2", "question-answering", "ml.m5.xlarge"),
    ],
)
def test_model_builder_happy_path_with_task_provided_remote_schema_mode(
    model_id, task_provided, sagemaker_session, instance_type_provided
):
    model_builder = ModelBuilder(
        model=model_id,
        model_metadata={"HF_TASK": task_provided},
        instance_type=instance_type_provided,
    )
    model = model_builder.build(sagemaker_session=sagemaker_session)

    assert model is not None
    assert model_builder.schema_builder is not None

    remote_hf_schema_helper = remote_schema_retriever.RemoteSchemaRetriever()
    inputs, outputs = remote_hf_schema_helper.get_resolved_hf_schema_for_task(task_provided)
    assert model_builder.schema_builder.sample_input == inputs
    assert model_builder.schema_builder.sample_output == outputs

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        caught_ex = None
        try:
            iam_client = sagemaker_session.boto_session.client("iam")
            role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            predictor = model.deploy(
                role=role_arn, initial_instance_count=1, instance_type=instance_type_provided
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


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    reason="Testing Schema Builder Simplification feature - Remote Schema",
)
@pytest.mark.parametrize(
    "model_id, task_provided, instance_type_provided",
    [("openai/whisper-tiny.en", "automatic-speech-recognition", "ml.m5.4xlarge")],
)
def test_model_builder_with_task_provided_remote_schema_mode_asr(
    model_id, task_provided, sagemaker_session, instance_type_provided
):
    model_builder = ModelBuilder(
        model=model_id,
        model_metadata={"HF_TASK": task_provided},
        instance_type=instance_type_provided,
    )
    model = model_builder.build(sagemaker_session=sagemaker_session)

    assert model is not None
    assert model_builder.schema_builder is not None

    remote_hf_schema_helper = remote_schema_retriever.RemoteSchemaRetriever()
    inputs, outputs = remote_hf_schema_helper.get_resolved_hf_schema_for_task(task_provided)
    assert model_builder.schema_builder.sample_input == inputs
    assert model_builder.schema_builder.sample_output == outputs


def test_model_builder_negative_path_with_invalid_task(sagemaker_session):
    model_builder = ModelBuilder(
        model="bert-base-uncased", model_metadata={"HF_TASK": "invalid-task"}
    )

    with pytest.raises(
        TaskNotFoundException,
        match="Error Message: HuggingFace Schema builder samples for invalid-task could not be found locally or "
        "via remote.",
    ):
        model_builder.build(sagemaker_session=sagemaker_session)
