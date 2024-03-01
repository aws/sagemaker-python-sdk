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

# import docker
# from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
# from sagemaker.serve.builder.schema_builder import SchemaBuilder
# from tests.integ.sagemaker.serve.constants import (
#     SERVE_IN_PROCESS_TIMEOUT,
#     SERVE_LOCAL_CONTAINER_TIMEOUT,
#     SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
#     XGB_RESOURCE_DIR,
#     SERVE_MODEL_PACKAGE_TIMEOUT,
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
#     NOT_RUNNING_ON_PY38,
# )
# from tests.integ.timeout import timeout
# from tests.integ.utils import cleanup_model_resources
# import logging

# logger = logging.getLogger(__name__)

# ROLE_NAME = "SageMakerRole"

# GH_USER_NAME = os.getenv("GH_USER_NAME")
# GH_ACCESS_TOKEN = os.getenv("GH_ACCESS_TOKEN")


# def terminate_all_container():
#     client = docker.from_env()
#     # List all container IDs (including stopped containers)
#     container_ids = [container.id for container in client.containers.list(all=True)]

#     for container_id in container_ids:
#         container = client.containers.get(container_id=container_id)
#         container.stop()


# @pytest.fixture(autouse=True)
# def setup():
#     # This code runs before each test
#     terminate_all_container()
#     yield


# @pytest.fixture
# def xgb_dependencies():
#     return {
#         "auto": True,
#         "custom": [
#             "protobuf==3.20.2",
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
# def model_builder_model_schema_builder(loaded_xgb_model, xgb_test_sets, xgb_dependencies):
#     return ModelBuilder(
#         model_path=XGB_RESOURCE_DIR,
#         model=loaded_xgb_model,
#         schema_builder=SchemaBuilder(xgb_test_sets.x_test, xgb_test_sets.y_test),
#         dependencies=xgb_dependencies,
#     )


# @pytest.fixture
# def model_builder_inference_spec_schema_builder(
#     xgb_inference_spec, xgb_test_sets, xgb_dependencies
# ):
#     return ModelBuilder(
#         model_path=XGB_RESOURCE_DIR,
#         inference_spec=xgb_inference_spec,
#         schema_builder=SchemaBuilder(xgb_test_sets.x_test, xgb_test_sets.y_test),
#         dependencies=xgb_dependencies,
#     )


# @pytest.fixture
# def model_builder(request):
#     return request.getfixturevalue(request.param)


# # XGB Integ Tests
# @pytest.mark.skipif(True, reason="Not supported yet")
# def test_happy_xgb_in_process(xgb_test_sets, model_builder):
#     logger.info("Running in IN_PROCESS mode...")
#     model = model_builder.build(mode=Mode.IN_PROCESS)

#     with timeout(minutes=SERVE_IN_PROCESS_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in IN_PROCESS mode...")
#             predictor = model.deploy()
#             predictor.predict(xgb_test_sets.x_test)
#         except Exception as e:
#             assert False, f"{e} was thrown when running xgb in process test"


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY38,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder",
#     ["model_builder_model_schema_builder", "model_builder_inference_spec_schema_builder"],
#     indirect=True,
# )
# def test_happy_xgb_local_container(sagemaker_session, xgb_test_sets, model_builder):
#     logger.info("Running in LOCAL_CONTAINER mode...")
#     caught_ex = None
#     predictor = None
#     model = model_builder.build(mode=Mode.LOCAL_CONTAINER, sagemaker_session=sagemaker_session)

#     with timeout(minutes=SERVE_LOCAL_CONTAINER_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
#             predictor = model.deploy()
#             logger.info("Local container successfully deployed.")
#             predictor.predict(xgb_test_sets.x_test)
#         except Exception as e:
#             caught_ex = e
#         finally:
#             if predictor:
#                 predictor.delete_predictor()
#             if caught_ex:
#                 logger.exception(caught_ex)
#                 assert False, f"{caught_ex} was thrown when running xgb local container test"


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY38,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder",
#     ["model_builder_model_schema_builder", "model_builder_inference_spec_schema_builder"],
#     indirect=True,
# )
# def test_happy_xgb_sagemaker_endpoint(
#     sagemaker_session, xgb_test_sets, model_builder, cpu_instance_type
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
#                 assert False, f"{caught_ex} was thrown when running xgb sagemaker endpoint test"


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY38,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder",
#     ["model_builder_model_schema_builder", "model_builder_inference_spec_schema_builder"],
#     indirect=True,
# )
# def test_happy_xgb_model_register_local_mode(
#     sagemaker_session, model_builder, cpu_instance_type, xgb_test_sets
# ):
#     logger.info("Running in ModelPackage mode...")
#     caught_ex = None

#     iam_client = sagemaker_session.boto_session.client("iam")
#     role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

#     model = model_builder.build(
#         role_arn=role_arn, sagemaker_session=sagemaker_session, mode=Mode.LOCAL_CONTAINER
#     )
#     model_package = model.register()
#     logger.info("Model package successfully created.")

#     with timeout(minutes=SERVE_MODEL_PACKAGE_TIMEOUT):
#         try:
#             logger.info("Deploying model package")
#             predictor = model_package.deploy(
#                 instance_type=cpu_instance_type, initial_instance_count=1
#             )
#             predictor.predict(xgb_test_sets.x_test)
#             logger.info("Model Package successfully deployed.")
#         except Exception as e:
#             caught_ex = e
#         finally:
#             sagemaker_session.sagemaker_client.delete_model_package(
#                 ModelPackageName=model_package.model_package_arn
#             )
#             if model.modes[str(Mode.LOCAL_CONTAINER)].container:
#                 model.modes[str(Mode.LOCAL_CONTAINER)].container.kill()
#             if caught_ex:
#                 logger.exception(caught_ex)
#                 assert False, f"{caught_ex} was thrown when running xgb local container test"


# @pytest.mark.skipif(
#     NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE or NOT_RUNNING_ON_PY38,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder",
#     ["model_builder_model_schema_builder", "model_builder_inference_spec_schema_builder"],
#     indirect=True,
# )
# def test_happy_xgb_model_register_endpoint_mode(
#     sagemaker_session, model_builder, cpu_instance_type, xgb_test_sets
# ):

#     logger.info("Running in ModelPackage mode...")
#     caught_ex = None

#     iam_client = sagemaker_session.boto_session.client("iam")
#     role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

#     model = model_builder.build(
#         role_arn=role_arn, sagemaker_session=sagemaker_session, mode=Mode.SAGEMAKER_ENDPOINT
#     )
#     model_package = model.register()
#     logger.info("Model package successfully created.")

#     with timeout(minutes=SERVE_MODEL_PACKAGE_TIMEOUT):
#         try:
#             logger.info("Deploying model package")
#             predictor = model_package.deploy(
#                 instance_type=cpu_instance_type, initial_instance_count=1
#             )
#             predictor.predict(xgb_test_sets.x_test)
#             logger.info("Model Package successfully deployed.")
#         except Exception as e:
#             caught_ex = e
#         finally:
#             sagemaker_session.sagemaker_client.delete_model_package(
#                 ModelPackageName=model_package.model_package_arn
#             )
#             cleanup_model_resources(
#                 sagemaker_session=model_builder.sagemaker_session,
#                 model_name=model.name,
#                 endpoint_name=model.endpoint_name,
#             )
#             if caught_ex:
#                 logger.exception(caught_ex)
#                 assert False, f"{caught_ex} was thrown when running xgb endpoint model package test"
