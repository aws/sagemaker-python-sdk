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

# import pytest
# import os

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image

# from sagemaker.serve.builder.schema_builder import SchemaBuilder
# from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
# from sagemaker.serve.utils.types import ModelServer

# from tests.integ.sagemaker.serve.constants import (
#     TF_EFFICIENT_RESOURCE_DIR,
#     NOT_RUNNING_ON_PY310,
#     SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
# )

# from tests.integ.timeout import timeout
# from tests.integ.utils import cleanup_model_resources
# import logging

# logger = logging.getLogger(__name__)

# ROLE_NAME = "SageMakerRole"


# @pytest.fixture
# def input_array():
#     img = image.load_img(
#         str(os.path.join(TF_EFFICIENT_RESOURCE_DIR, "zidane.jpeg")), target_size=(224, 224)
#     )
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     return x


# @pytest.fixture
# def efficientnet_schema(input_array):
#     # Dummy output
#     output_array = np.random.rand(1, 1000)
#     return SchemaBuilder(sample_input=input_array, sample_output=output_array)


# @pytest.fixture
# def efficientnet_model():
#     return tf.keras.models.load_model(str(os.path.join(TF_EFFICIENT_RESOURCE_DIR, "model.keras")))


# @pytest.fixture
# def dependency_config():
#     return {"auto": False, "custom": ["protobuf=3.20.*", "dill>=0.3.7"]}


# @pytest.fixture
# def model_builder_model_schema_builder(efficientnet_model, efficientnet_schema, dependency_config):
#     return ModelBuilder(
#         model_path=TF_EFFICIENT_RESOURCE_DIR,
#         model=efficientnet_model,
#         schema_builder=efficientnet_schema,
#         image_uri="301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:23.02-py3",
#         model_server=ModelServer.TRITON,
#         dependencies=dependency_config,
#     )


# @pytest.fixture
# def model_builder(request):
#     return request.getfixturevalue(request.param)


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize("model_builder", ["model_builder_model_schema_builder"], indirect=True)
# def test_happy_pytorch_triton_sagemaker_endpoint(
#     sagemaker_session, model_builder, gpu_instance_type, input_array
# ):
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
#             predictor = model.deploy(instance_type=gpu_instance_type, initial_instance_count=1)
#             logger.info("Endpoint successfully deployed.")
#             predictor.predict(input_array)
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
#                 ), f"{caught_ex} was thrown when running pytorch squeezenet sagemaker endpoint test"
