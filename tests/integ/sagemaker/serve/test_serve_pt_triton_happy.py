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
# import torch
# from PIL import Image

# from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
# from sagemaker.serve.builder.schema_builder import SchemaBuilder
# from sagemaker.serve.utils.types import ModelServer
# from torchvision.transforms import transforms
# from torchvision.models.squeezenet import squeezenet1_1

# from tests.integ.sagemaker.serve.constants import (
#     PYTORCH_SQUEEZENET_RESOURCE_DIR,
#     SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
#     NOT_RUNNING_ON_PY310,
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
#     SERVE_LOCAL_CONTAINER_TIMEOUT,
# )

# from tests.integ.timeout import timeout
# from tests.integ.utils import cleanup_model_resources
# import logging

# logger = logging.getLogger(__name__)

# ROLE_NAME = "SageMakerRole"


# @pytest.fixture
# def image_transformation():
#     return transforms.Compose(
#         [
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )


# @pytest.fixture
# def input_tensor(image_transformation):
#     image = Image.open(str(os.path.join(PYTORCH_SQUEEZENET_RESOURCE_DIR, "zidane.jpeg")))

#     return image_transformation(image).unsqueeze(0)


# @pytest.fixture
# def squeezenet_schema(input_tensor):
#     # Dummy output
#     output_tensor = torch.rand(1, 1000)
#     return SchemaBuilder(sample_input=input_tensor, sample_output=output_tensor)


# @pytest.fixture
# def squeezenet_model():
#     model = squeezenet1_1()
#     model.load_state_dict(torch.load(PYTORCH_SQUEEZENET_RESOURCE_DIR + "/model.pth"))
#     model.eval()
#     return model


# @pytest.fixture
# def model_builder_model_schema_builder(squeezenet_model, squeezenet_schema):
#     return ModelBuilder(
#         model_path=PYTORCH_SQUEEZENET_RESOURCE_DIR,
#         model=squeezenet_model,
#         schema_builder=squeezenet_schema,
#         model_server=ModelServer.TRITON,
#     )


# @pytest.fixture
# def model_builder(request):
#     return request.getfixturevalue(request.param)


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize("model_builder", ["model_builder_model_schema_builder"], indirect=True)
# def test_happy_pytorch_triton_local_container(sagemaker_session, model_builder, input_tensor):
#     logger.info("Running in LOCAL_CONTAINER mode...")
#     caught_ex = None

#     model = model_builder.build(mode=Mode.LOCAL_CONTAINER, sagemaker_session=sagemaker_session)

#     with timeout(minutes=SERVE_LOCAL_CONTAINER_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
#             predictor = model.deploy()
#             logger.info("Local container successfully deployed.")
#             predictor.predict(input_tensor)
#         except Exception as e:
#             logger.exception("test failed")
#             caught_ex = e
#         finally:
#             if model.modes[str(Mode.LOCAL_CONTAINER)].container:
#                 model.modes[str(Mode.LOCAL_CONTAINER)].container.kill()
#             if caught_ex:
#                 assert (
#                     False
#                 ), f"{caught_ex} was thrown when running pytorch squeezenet local container test"


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize("model_builder", ["model_builder_model_schema_builder"], indirect=True)
# def test_happy_pytorch_triton_sagemaker_endpoint(
#     sagemaker_session, model_builder, cpu_instance_type, input_tensor
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
#             predictor = model.deploy(instance_type=cpu_instance_type, initial_instance_count=1)
#             logger.info("Endpoint successfully deployed.")
#             predictor.predict(input_tensor)
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
