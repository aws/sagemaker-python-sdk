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
# import torch
# from PIL import Image
# import os

# from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
# from sagemaker.serve.builder.schema_builder import SchemaBuilder
# from sagemaker.serve.spec.inference_spec import InferenceSpec
# from torchvision.transforms import transforms
# from torchvision.models.squeezenet import squeezenet1_1

# from tests.integ.sagemaker.serve.constants import (
#     PYTORCH_SQUEEZENET_RESOURCE_DIR,
#     SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
#     NOT_RUNNING_ON_PY310,
# )
# from tests.integ.timeout import timeout
# from tests.integ.utils import cleanup_model_resources
# import logging

# logger = logging.getLogger(__name__)

# ROLE_NAME = "Admin"

# GH_USER_NAME = os.getenv("GH_USER_NAME")
# GH_ACCESS_TOKEN = os.getenv("GH_ACCESS_TOKEN")


# @pytest.fixture
# def pt_dependencies():
#     return {
#         "auto": True,
#         "custom": [
#             "boto3==1.26.*",
#             "botocore==1.29.*",
#             "s3transfer==0.6.*",
#             (
#                 f"git+https://{GH_USER_NAME}:{GH_ACCESS_TOKEN}@github.com"
#                 "/aws/sagemaker-python-sdk-staging.git@inference-experience-dev"
#             ),
#         ],
#     }


# @pytest.fixture
# def test_image():
#     return Image.open(str(os.path.join(PYTORCH_SQUEEZENET_RESOURCE_DIR, "zidane.jpeg")))


# @pytest.fixture
# def squeezenet_inference_spec():
#     class MySqueezeNetModel(InferenceSpec):
#         def __init__(self) -> None:
#             super().__init__()
#             self.transform = transforms.Compose(
#                 [
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ]
#             )

#         def invoke(self, input_object: object, model: object):
#             # transform
#             image_tensor = self.transform(input_object)
#             input_batch = image_tensor.unsqueeze(0)
#             # invoke
#             with torch.no_grad():
#                 output = model(input_batch)
#             return output

#         def load(self, model_dir: str):
#             model = squeezenet1_1()
#             model.load_state_dict(torch.load(model_dir + "/model.pth"))
#             model.eval()
#             return model

#     return MySqueezeNetModel()


# @pytest.fixture
# def squeezenet_schema():
#     input_image = Image.open(os.path.join(PYTORCH_SQUEEZENET_RESOURCE_DIR, "zidane.jpeg"))
#     output_tensor = torch.rand(3, 4)
#     return SchemaBuilder(sample_input=input_image, sample_output=output_tensor)


# @pytest.fixture
# def model_builder_inference_spec_schema_builder(
#     squeezenet_inference_spec, squeezenet_schema, pt_dependencies
# ):
#     return ModelBuilder(
#         model_path=PYTORCH_SQUEEZENET_RESOURCE_DIR,
#         inference_spec=squeezenet_inference_spec,
#         schema_builder=squeezenet_schema,
#         dependencies=pt_dependencies,
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
# def test_happy_pytorch_local_container(sagemaker_session, model_builder, test_image):
#     logger.info("Running in LOCAL_CONTAINER mode...")
#     caught_ex = None

#     model = model_builder.build(mode=Mode.LOCAL_CONTAINER, sagemaker_session=sagemaker_session)

#     with timeout(minutes=SERVE_LOCAL_CONTAINER_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
#             predictor = model.deploy()
#             logger.info("Local container successfully deployed.")
#             predictor.predict(test_image)
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
# @pytest.mark.parametrize(
#     "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
# )
# def test_happy_pytorch_sagemaker_endpoint(
#     sagemaker_session, model_builder, cpu_instance_type, test_image
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
#             predictor.predict(test_image)
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


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
# )
# def test_happy_pytorch_local_container_overwrite_to_sagemaker_endpoint(
#     sagemaker_session, model_builder, cpu_instance_type, test_image
# ):
#     logger.info("Building model in LOCAL_CONTAINER mode...")
#     caught_ex = None

#     iam_client = sagemaker_session.boto_session.client("iam")
#     role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
#     logger.debug("Role arn: %s", role_arn)

#     model = model_builder.build(
#         mode=Mode.LOCAL_CONTAINER, role_arn=role_arn, sagemaker_session=sagemaker_session
#     )

#     with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
#             predictor = model.deploy(
#                 instance_type=cpu_instance_type,
#                 initial_instance_count=1,
#                 mode=Mode.SAGEMAKER_ENDPOINT,
#             )
#             logger.info("Endpoint successfully deployed.")
#             predictor.predict(test_image)
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


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
# )
# def test_happy_pytorch_sagemaker_endpoint_overwrite_to_local_container(
#     sagemaker_session, model_builder, test_image
# ):
#     logger.info("Building model in SAGEMAKER_ENDPOINT mode...")
#     caught_ex = None

#     iam_client = sagemaker_session.boto_session.client("iam")
#     role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

#     model = model_builder.build(
#         mode=Mode.SAGEMAKER_ENDPOINT, role_arn=role_arn, sagemaker_session=sagemaker_session
#     )

#     with timeout(minutes=SERVE_LOCAL_CONTAINER_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
#             predictor = model.deploy(mode=Mode.LOCAL_CONTAINER)
#             logger.info("Local container successfully deployed.")
#             predictor.predict(test_image)
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
