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
# from sagemaker.serve.builder.model_builder import ModelBuilder, Mode, ModelServer
# from sagemaker.serve.builder.schema_builder import SchemaBuilder
# from tests.integ.sagemaker.serve.constants import (
#     SERVE_LOCAL_CONTAINER_TIMEOUT,
#     SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
#     BYOC_IMAGE_URI_TEMPLATE,
#     NOT_RUNNING_ON_PY38,
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
# )
# from tests.integ.timeout import timeout
# from tests.integ.utils import cleanup_model_resources
# import logging

# logger = logging.getLogger(__name__)

# ROLE_NAME = "SageMakerRole"

# GH_USER_NAME = os.getenv("GH_USER_NAME")
# GH_ACCESS_TOKEN = os.getenv("GH_ACCESS_TOKEN")


# @pytest.fixture
# def xgb_dependencies():
#     return {
#         "auto": True,
#         "custom": [
#             "protobuf==3.20.2",
#             (
#                 f"git+https://{GH_USER_NAME}:{GH_ACCESS_TOKEN}@github.com"
#                 "/aws/sagemaker-python-sdk-staging.git@inference-experience-dev"
#             ),
#         ],
#     }


# @pytest.fixture
# def model_server():
#     return ModelServer.TORCHSERVE


# @pytest.fixture
# def xgb_dlc(sagemaker_session):
#     image_tag = "xgb-1.7-1"
#     image_uri = BYOC_IMAGE_URI_TEMPLATE.format(sagemaker_session.boto_region_name, image_tag)
#     print("##############" + image_uri)
#     return image_uri


# @pytest.fixture
# def model_builder_schema_builder(
#     loaded_xgb_model, xgb_test_sets, model_server, xgb_dlc, xgb_dependencies
# ):
#     loaded_xgb_model.fit(xgb_test_sets.x_test, xgb_test_sets.y_test)
#     return ModelBuilder(
#         model=loaded_xgb_model,
#         schema_builder=SchemaBuilder(xgb_test_sets.x_test, xgb_test_sets.y_test),
#         model_server=model_server,
#         image_uri=xgb_dlc,
#         dependencies=xgb_dependencies,
#     )


# @pytest.fixture
# def model_builder(request):
#     return request.getfixturevalue(request.param)


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY38,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder",
#     ["model_builder_schema_builder"],
#     indirect=True,
# )
# def test_happy_byoc_xgb_local_container(sagemaker_session, xgb_test_sets, model_builder):
#     logger.info("Running in LOCAL_CONTAINER mode...")
#     caught_ex = None

#     iam_client = sagemaker_session.boto_session.client("iam")
#     role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

#     model = model_builder.build(
#         mode=Mode.LOCAL_CONTAINER, role_arn=role_arn, sagemaker_session=sagemaker_session
#     )

#     with timeout(minutes=SERVE_LOCAL_CONTAINER_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
#             predictor = model.deploy()
#             logger.info("Local container successfully deployed.")
#             predictor.predict(xgb_test_sets.x_test)
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
#                 assert False, f"{caught_ex} was thrown when running byoc xgb local container test"


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY38,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder",
#     ["model_builder_schema_builder"],
#     indirect=True,
# )
# def test_happy_byoc_xgb_sagemaker_endpoint(sagemaker_session, xgb_test_sets, model_builder):
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
#             predictor = model.deploy(instance_type="ml.c5.xlarge", initial_instance_count=1)
#             logger.info("Endpoint successfully deployed.")
#             predictor.predict(xgb_test_sets.x_test)

#             predictor.delete_endpoint()
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
#                 ), f"{caught_ex} was thrown when running byoc xgb sagemaker endpoint test"
