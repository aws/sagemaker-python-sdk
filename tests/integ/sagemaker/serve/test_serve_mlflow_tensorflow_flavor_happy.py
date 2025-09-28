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
import tensorflow as tf
from sklearn.datasets import fetch_california_housing


from tests.integ.sagemaker.serve.constants import (
    TENSORFLOW_MLFLOW_RESOURCE_DIR,
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
    PYTHON_VERSION_IS_NOT_310,
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
    dataset = fetch_california_housing(as_frame=True)["frame"]
    dataset = dataset.dropna()
    dataset_tf = tf.convert_to_tensor(dataset, dtype=tf.float32)
    dataset_tf = dataset_tf[:50]
    x_test, y_test = dataset_tf[:, :-1], dataset_tf[:, -1]
    return x_test, y_test


@pytest.fixture
def custom_request_translator():
    class MyRequestTranslator(CustomPayloadTranslator):
        def serialize_payload_to_bytes(self, payload: object) -> bytes:
            return self._convert_numpy_to_bytes(payload)

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
    class MyResponseTranslator(CustomPayloadTranslator):
        def serialize_payload_to_bytes(self, payload: object) -> bytes:
            import numpy as np

            return self._convert_numpy_to_bytes(np.array(payload))

        def deserialize_payload_from_stream(self, stream) -> object:
            import tensorflow as tf

            return tf.convert_to_tensor(np.load(io.BytesIO(stream.read())))

        def _convert_numpy_to_bytes(self, np_array: np.ndarray) -> bytes:
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()

    return MyResponseTranslator()


@pytest.fixture
def tensorflow_schema_builder(custom_request_translator, custom_response_translator, test_data):
    input_data, output_data = test_data
    return SchemaBuilder(
        sample_input=input_data,
        sample_output=output_data,
        input_translator=custom_request_translator,
        output_translator=custom_response_translator,
    )


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    reason="The goal of these test are to test the serving components of our feature",
)

def test_happy_tensorflow_sagemaker_endpoint_with_tensorflow_serving(
    sagemaker_session,
    tensorflow_schema_builder,
    cpu_instance_type,
    test_data,
):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    caught_ex = None

    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

    model_artifacts_uri = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "model_builder_integ_test",
        "mlflow",
        "tensorflow",
    )

    model_path = S3Uploader.upload(
        local_path=TENSORFLOW_MLFLOW_RESOURCE_DIR,
        desired_s3_uri=model_artifacts_uri,
        sagemaker_session=sagemaker_session,
    )

    model_builder = ModelBuilder(
        mode=Mode.SAGEMAKER_ENDPOINT,
        schema_builder=tensorflow_schema_builder,
        role_arn=role_arn,
        sagemaker_session=sagemaker_session,
        model_metadata={"MLFLOW_MODEL_PATH": model_path},
    )

    model = model_builder.build(sagemaker_session=sagemaker_session)

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            test_x, _ = test_data
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
