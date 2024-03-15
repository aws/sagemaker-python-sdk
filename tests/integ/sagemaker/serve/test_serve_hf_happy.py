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
# from __future__ import absolute_import

# import os
# import pytest
# import platform
# from sagemaker.serve.builder.model_builder import ModelBuilder, Mode, ModelServer, InferenceSpec
# from sagemaker.serve.builder.schema_builder import SchemaBuilder
# from sagemaker import image_uris
# from tests.integ.sagemaker.serve.constants import (
#     SERVE_LOCAL_CONTAINER_TIMEOUT,
#     SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
#     NOT_RUNNING_ON_PY310,
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
#     HF_DIR,
# )

# from tests.integ.timeout import timeout
# from tests.integ.utils import cleanup_model_resources
# import logging

# if platform.python_version_tuple()[1] == "10":
#     from transformers import T5Tokenizer, T5ForConditionalGeneration

# logger = logging.getLogger(__name__)


# HF_SAMPLE_ENG_INPUT = "translate English to French: hello world"
# HF_SAMPLE_FR_OUTPUT = "bonjour monde"
# ROLE_NAME = "SageMakerRole"

# GH_USER_NAME = os.getenv("GH_USER_NAME")
# GH_ACCESS_TOKEN = os.getenv("GH_ACCESS_TOKEN")


# @pytest.fixture
# def huggingface_inference_spec():
#     class HuggingFaceModel(InferenceSpec):
#         def load(self, model_dir: str):
#             logger.info("Loading model!")
#             tokenizer = T5Tokenizer.from_pretrained("t5-small")
#             model = T5ForConditionalGeneration.from_pretrained("t5-small")
#             logger.info(f"model loaded {model}, {tokenizer}")
#             return {"model": model, "tokenizer": tokenizer}

#         def invoke(self, input_data: object, model: object):
#             tokenizer = model["tokenizer"]
#             t5_model = model["model"]

#             input_ids = tokenizer(input_data, return_tensors="pt").input_ids
#             outputs = t5_model.generate(input_ids)

#             return tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return HuggingFaceModel()


# @pytest.fixture
# def pt_dlc(sagemaker_session, cpu_instance_type):
#     return image_uris.retrieve(
#         framework="pytorch",
#         region=sagemaker_session.boto_region_name,
#         version="2.0.0",
#         image_scope="inference",
#         py_version="py310",
#         instance_type=cpu_instance_type,
#     )


# @pytest.fixture
# def model_builder_inference_spec_schema_builder(huggingface_inference_spec, pt_dlc):
#     return ModelBuilder(
#         model_path=HF_DIR,
#         inference_spec=huggingface_inference_spec,
#         schema_builder=SchemaBuilder(HF_SAMPLE_ENG_INPUT, HF_SAMPLE_FR_OUTPUT),
#         image_uri=pt_dlc,
#         model_server=ModelServer.TORCHSERVE,
#         env_vars={
#             "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
#         },
#         dependencies={
#             "auto": False,
#             "custom": [
#                 "transformers==4.35.*",
#                 "sentencepiece==0.1.*",
#                 (
#                     f"git+https://{GH_USER_NAME}:{GH_ACCESS_TOKEN}@github.com"
#                     "/aws/sagemaker-python-sdk-staging.git@inference-experience-dev"
#                 ),
#             ],
#         },
#     )


# @pytest.fixture
# def model_builder(request):
#     return request.getfixturevalue(request.param)


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
# )
# def test_happy_hugging_local_container(sagemaker_session, model_builder):
#     logger.info("Running in LOCAL_CONTAINER mode...")
#     caught_ex = None
#     model = model_builder.build(mode=Mode.LOCAL_CONTAINER, sagemaker_session=sagemaker_session)

#     with timeout(minutes=SERVE_LOCAL_CONTAINER_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
#             predictor = model.deploy()
#             logger.info("Local container successfully deployed.")
#             predictor.predict(HF_SAMPLE_ENG_INPUT)
#         except Exception as e:
#             caught_ex = e
#         finally:
#             if model.modes[str(Mode.LOCAL_CONTAINER)].container:
#                 model.modes[str(Mode.LOCAL_CONTAINER)].container.kill()
#             if caught_ex:
#                 logger.exception(caught_ex)
#                 assert (
#                     False
#                 ), f"{caught_ex} was thrown when running huggingface local container test"


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
# )
# def test_happy_hugging_sagemaker_endpoint(sagemaker_session, model_builder, cpu_instance_type):
#     logger.info("Running in SAGEMAKER_ENDPOINT mode...")
#     caught_ex = None

#     iam_client = sagemaker_session.boto_session.client("iam")
#     role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

#     model = model_builder.build(
#         mode=Mode.SAGEMAKER_ENDPOINT, role_arn=role_arn, sagemaker_session=sagemaker_session
#     )

#     with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
#             predictor = model.deploy(instance_type=cpu_instance_type, initial_instance_count=1)
#             logger.info("Endpoint successfully deployed.")
#             predictor.predict(HF_SAMPLE_ENG_INPUT)
#         except Exception as e:
#             caught_ex = e
#         finally:
#             cleanup_model_resources(
#                 sagemaker_session=model_builder.sagemaker_session,
#                 model_name=model.name,
#                 endpoint_name=model.endpoint_name,
#             )
#             if caught_ex:
#                 logger.exception(caught_ex)
#                 assert (
#                     False
#                 ), f"{caught_ex} was thrown when running huggingface sagemaker endpoint test"
