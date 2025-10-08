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

import pytest
import io
import numpy as np

from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association
from sagemaker.s3 import S3Uploader
from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
from sagemaker.serve.builder.schema_builder import SchemaBuilder, CustomPayloadTranslator
from sklearn.datasets import load_diabetes


from tests.integ.sagemaker.serve.constants import (
    XGBOOST_MLFLOW_RESOURCE_DIR,
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
    # SERVE_LOCAL_CONTAINER_TIMEOUT,
    # PYTHON_VERSION_IS_NOT_310,
)
from tests.integ.timeout import timeout
from tests.integ.utils import cleanup_model_resources
import logging

from sagemaker.serve.utils.lineage_constants import (
    MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE,
)

logger = logging.getLogger(__name__)

ROLE_NAME = "SageMakerRole"


@pytest.fixture
def test_data():
    return load_diabetes(return_X_y=True, as_frame=True)


@pytest.fixture
def custom_request_translator():
    # request translator
    class MyRequestTranslator(CustomPayloadTranslator):

        # This function converts the payload to bytes - happens on client side
        def serialize_payload_to_bytes(self, payload: object) -> bytes:
            return self._convert_numpy_to_bytes(payload)

        # This function converts the bytes to payload - happens on server side
        def deserialize_payload_from_stream(self, stream) -> object:
            np_array = np.load(io.BytesIO(stream.read()))
            return np_array

        def _convert_numpy_to_bytes(self, np_array: np.ndarray) -> bytes:
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()

    return MyRequestTranslator()


@pytest.fixture
def custom_response_translator():
    # response translator
    class MyResponseTranslator(CustomPayloadTranslator):
        # This function converts the payload to bytes - happens on server side
        def serialize_payload_to_bytes(self, payload: object) -> bytes:
            return self._convert_numpy_to_bytes(payload)

        # This function converts the bytes to payload - happens on client side
        def deserialize_payload_from_stream(self, stream) -> object:
            return np.load(io.BytesIO(stream.read()))

        def _convert_numpy_to_bytes(self, np_array: np.ndarray) -> bytes:
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()

    return MyResponseTranslator()


@pytest.fixture
def xgboost_schema(custom_request_translator, custom_response_translator, test_data):
    test_x, test_y = test_data
    return SchemaBuilder(
        sample_input=test_x,
        sample_output=test_y,
        input_translator=custom_request_translator,
        output_translator=custom_response_translator,
    )


@pytest.fixture
def model_builder_local_builder(xgboost_schema):
    return ModelBuilder(
        schema_builder=xgboost_schema,
        model_metadata={"MLFLOW_MODEL_PATH": XGBOOST_MLFLOW_RESOURCE_DIR},
    )


@pytest.fixture
def model_builder(request):
    return request.getfixturevalue(request.param)


# @pytest.mark.skipif(
#     PYTHON_VERSION_IS_NOT_310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.flaky(reruns=3, reruns_delay=2)
# @pytest.mark.parametrize("model_builder", ["model_builder_local_builder"], indirect=True)
# def test_happy_mlflow_xgboost_local_container_with_torch_serve(
#     sagemaker_session, model_builder, test_data
# ):
#     logger.info("Running in LOCAL_CONTAINER mode...")
#     caught_ex = None
#
#     model = model_builder.build(mode=Mode.LOCAL_CONTAINER, sagemaker_session=sagemaker_session)
#     test_x, _ = test_data
#
#     with timeout(minutes=SERVE_LOCAL_CONTAINER_TIMEOUT):
#         try:
#             logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
#             predictor = model.deploy()
#             logger.info("Local container successfully deployed.")
#             predictor.predict(test_x)
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


@pytest.mark.skip(
    reason="Skipping it temporarily as we have bug with latest version of XGBoost image \
           that is numpy 2.0 compatible.",
)
def test_happy_xgboost_sagemaker_endpoint_with_torch_serve(
    sagemaker_session,
    xgboost_schema,
    cpu_instance_type,
    test_data,
):
    # TODO: Enable this test once the issue with latest XGBoost image is fixed.
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    caught_ex = None

    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    test_x, _ = test_data

    model_artifacts_uri = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "model_builder_integ_test",
        "mlflow",
        "xgboost",
    )

    model_path = S3Uploader.upload(
        local_path=XGBOOST_MLFLOW_RESOURCE_DIR,
        desired_s3_uri=model_artifacts_uri,
        sagemaker_session=sagemaker_session,
    )

    model_builder = ModelBuilder(
        mode=Mode.SAGEMAKER_ENDPOINT,
        schema_builder=xgboost_schema,
        role_arn=role_arn,
        sagemaker_session=sagemaker_session,
        model_metadata={"MLFLOW_MODEL_PATH": model_path},
    )

    model = model_builder.build(sagemaker_session=sagemaker_session)

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            predictor = model.deploy(instance_type=cpu_instance_type, initial_instance_count=1)
            logger.info("Endpoint successfully deployed.")
            predictor.predict(test_x)
            model_data_artifact = None
            for artifact in Artifact.list(
                source_uri=model_builder.s3_upload_path, sagemaker_session=sagemaker_session
            ):
                model_data_artifact = artifact
            for association in Association.list(
                destination_arn=model_data_artifact.artifact_arn,
                sagemaker_session=sagemaker_session,
            ):
                assert (
                    association.source_type == MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE
                )
                break
        except Exception as e:
            caught_ex = e
        finally:
            cleanup_model_resources(
                sagemaker_session=model_builder.sagemaker_session,
                model_name=model.name,
                endpoint_name=model.endpoint_name,
            )
            if caught_ex:
                raise caught_ex
