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
# import time

# from sagemaker.serve.builder.model_builder import ModelBuilder, Mode, ModelServer
# from sagemaker.serve.builder.schema_builder import SchemaBuilder
# from sagemaker.serve.spec.inference_spec import InferenceSpec
# from sagemaker.serve.utils.exceptions import LocalModelLoadException

# from tests.integ.sagemaker.serve.constants import (
#     PYTORCH_SQUEEZENET_RESOURCE_DIR,
#     SERVE_LOCAL_CONTAINER_TIMEOUT,
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
#     NOT_RUNNING_ON_PY310,
# )
# from tests.integ.timeout import timeout
# import logging

# logger = logging.getLogger(__name__)

# ROLE_NAME = "SageMakerRole"

# GH_USER_NAME = os.getenv("GH_USER_NAME")
# GH_ACCESS_TOKEN = os.getenv("GH_ACCESS_TOKEN")


# @pytest.fixture
# def test_image():
#     return Image.open(str(os.path.join(PYTORCH_SQUEEZENET_RESOURCE_DIR, "zidane.jpeg")))


# @pytest.fixture
# def squeezenet_inference_spec():
#     class MySqueezeNetModel(InferenceSpec):
#         def __init__(self) -> None:
#             super().__init__()

#         def invoke(self, input_object: object, model: object):
#             pass

#         def load(self, model_dir: str):
#             # simulate failure
#             time.sleep(300)

#     return MySqueezeNetModel()


# @pytest.fixture
# def squeezenet_schema():
#     input_image = Image.open(os.path.join(PYTORCH_SQUEEZENET_RESOURCE_DIR, "zidane.jpeg"))
#     output_tensor = torch.rand(3, 4)
#     return SchemaBuilder(sample_input=input_image, sample_output=output_tensor)


# @pytest.fixture
# def dependency_config():
#     return {
#         "auto": False,
#         "custom": [
#             (
#                 f"git+https://{GH_USER_NAME}:{GH_ACCESS_TOKEN}@github.com"
#                 "/aws/sagemaker-python-sdk-staging.git@inference-experience-dev"
#             )
#         ],
#     }


# @pytest.fixture
# def model_builder_inference_spec_schema_builder(squeezenet_inference_spec, squeezenet_schema):
#     return ModelBuilder(
#         image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310",
#         model_server=ModelServer.TORCHSERVE,
#         model_path=PYTORCH_SQUEEZENET_RESOURCE_DIR,
#         inference_spec=squeezenet_inference_spec,
#         schema_builder=squeezenet_schema,
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
# def test_timeout_pytorch_local_container(sagemaker_session, model_builder, test_image):
#     logger.info("Running in LOCAL_CONTAINER mode with simulated error...")

#     model = model_builder.build(mode=Mode.LOCAL_CONTAINER, sagemaker_session=sagemaker_session)

#     with timeout(minutes=SERVE_LOCAL_CONTAINER_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
#             model.deploy()
#             logger.info("Local container successfully deployed.")
#             assert False, "do not expect this deployment to pass"
#         except LocalModelLoadException as e:
#             assert "Number or consecutive unsuccessful inference" in str(e)
#         except Exception:
#             assert False, "Exception was not of LocalModelLoadException"
#         finally:
#             if model.modes[str(Mode.LOCAL_CONTAINER)].container:
#                 model.modes[str(Mode.LOCAL_CONTAINER)].container.kill()
