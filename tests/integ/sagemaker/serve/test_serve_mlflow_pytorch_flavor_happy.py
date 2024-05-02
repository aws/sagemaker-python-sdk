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
import torch
from PIL import Image
import os
import io
import numpy as np

from sagemaker.s3 import S3Uploader
from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
from sagemaker.serve.builder.schema_builder import SchemaBuilder, CustomPayloadTranslator
from torchvision.transforms import transforms

from tests.integ.sagemaker.serve.constants import (
    PYTORCH_SQUEEZENET_RESOURCE_DIR,
    PYTORCH_SQUEEZENET_MLFLOW_RESOURCE_DIR,
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
    # SERVE_LOCAL_CONTAINER_TIMEOUT,
    PYTHON_VERSION_IS_NOT_310,
)
from tests.integ.timeout import timeout
from tests.integ.utils import cleanup_model_resources
import logging

logger = logging.getLogger(__name__)

ROLE_NAME = "SageMakerRole"


@pytest.fixture
def test_image():
    return Image.open(str(os.path.join(PYTORCH_SQUEEZENET_RESOURCE_DIR, "zidane.jpeg")))


@pytest.fixture
def custom_request_translator():
    # request translator
    class MyRequestTranslator(CustomPayloadTranslator):
        def __init__(self):
            super().__init__()
            # Define image transformation
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        # This function converts the payload to bytes - happens on client side
        def serialize_payload_to_bytes(self, payload: object) -> bytes:
            # converts an image to bytes
            image_tensor = self.transform(payload)
            input_batch = image_tensor.unsqueeze(0)
            input_ndarray = input_batch.detach().numpy()
            return self._convert_numpy_to_bytes(input_ndarray)

        # This function converts the bytes to payload - happens on server side
        def deserialize_payload_from_stream(self, stream) -> torch.Tensor:
            # convert payload back to torch.Tensor
            np_array = np.load(io.BytesIO(stream.read()))
            return torch.from_numpy(np_array)

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
        def serialize_payload_to_bytes(self, payload: torch.Tensor) -> bytes:
            return self._convert_numpy_to_bytes(payload.detach().numpy())

        # This function converts the bytes to payload - happens on client side
        def deserialize_payload_from_stream(self, stream) -> object:
            return torch.from_numpy(np.load(io.BytesIO(stream.read())))

        def _convert_numpy_to_bytes(self, np_array: np.ndarray) -> bytes:
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()

    return MyResponseTranslator()


@pytest.fixture
def squeezenet_schema(custom_request_translator, custom_response_translator):
    input_image = Image.open(os.path.join(PYTORCH_SQUEEZENET_RESOURCE_DIR, "zidane.jpeg"))
    output_tensor = torch.rand(3, 4)
    return SchemaBuilder(
        sample_input=input_image,
        sample_output=output_tensor,
        input_translator=custom_request_translator,
        output_translator=custom_response_translator,
    )


@pytest.fixture
def model_builder_local_builder(squeezenet_schema):
    return ModelBuilder(
        schema_builder=squeezenet_schema,
        model_metadata={"MLFLOW_MODEL_PATH": PYTORCH_SQUEEZENET_MLFLOW_RESOURCE_DIR},
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
# def test_happy_mlflow_pytorch_local_container_with_torch_serve(
#     sagemaker_session, model_builder, test_image
# ):
#     logger.info("Running in LOCAL_CONTAINER mode...")
#     caught_ex = None
#
#     model = model_builder.build(mode=Mode.LOCAL_CONTAINER, sagemaker_session=sagemaker_session)
#
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


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,  # or NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
    reason="The goal of these test are to test the serving components of our feature",
)
def test_happy_pytorch_sagemaker_endpoint_with_torch_serve(
    sagemaker_session,
    squeezenet_schema,
    cpu_instance_type,
    test_image,
):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    caught_ex = None

    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

    model_artifacts_uri = "s3://{}/{}/{}/{}".format(
        sagemaker_session.default_bucket(),
        "model_builder_integ_test",
        "mlflow",
        "pytorch",
    )

    model_path = S3Uploader.upload(
        local_path=PYTORCH_SQUEEZENET_MLFLOW_RESOURCE_DIR,
        desired_s3_uri=model_artifacts_uri,
        sagemaker_session=sagemaker_session,
    )

    model_builder = ModelBuilder(
        mode=Mode.SAGEMAKER_ENDPOINT,
        schema_builder=squeezenet_schema,
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
            predictor.predict(test_image)
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
                ignore_if_worker_dies = "Worker died." in str(caught_ex)
                # https://github.com/pytorch/serve/issues/3032
                assert (
                    ignore_if_worker_dies
                ), f"{caught_ex} was thrown when running pytorch squeezenet sagemaker endpoint test"
