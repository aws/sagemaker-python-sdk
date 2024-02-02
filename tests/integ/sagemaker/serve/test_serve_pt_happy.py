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

from unittest.mock import patch, Mock
from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
from sagemaker.serve.builder.schema_builder import SchemaBuilder, CustomPayloadTranslator
from sagemaker.serve.spec.inference_spec import InferenceSpec
from torchvision.transforms import transforms
from torchvision.models.squeezenet import squeezenet1_1

from tests.integ.sagemaker.serve.constants import (
    PYTORCH_SQUEEZENET_RESOURCE_DIR,
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
    PYTHON_VERSION_IS_NOT_310,
)
from tests.integ.timeout import timeout
from tests.integ.utils import cleanup_model_resources
import logging

logger = logging.getLogger(__name__)

ROLE_NAME = "SageMakerRole"
MOCK_HF_MODEL_METADATA_JSON = {"mock_key": "mock_value"}


@pytest.fixture
def test_image():
    return Image.open(str(os.path.join(PYTORCH_SQUEEZENET_RESOURCE_DIR, "zidane.jpeg")))


@pytest.fixture
def squeezenet_inference_spec():
    class MySqueezeNetModel(InferenceSpec):
        def __init__(self) -> None:
            super().__init__()

        def invoke(self, input_object: object, model: object):
            with torch.no_grad():
                output = model(input_object)
            return output

        def load(self, model_dir: str):
            model = squeezenet1_1()
            model.load_state_dict(torch.load(model_dir + "/model.pth"))
            model.eval()
            return model

    return MySqueezeNetModel()


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
            input_ndarray = input_batch.numpy()
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
            return self._convert_numpy_to_bytes(payload.numpy())

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
def model_builder_inference_spec_schema_builder(squeezenet_inference_spec, squeezenet_schema):
    return ModelBuilder(
        model_path=PYTORCH_SQUEEZENET_RESOURCE_DIR,
        inference_spec=squeezenet_inference_spec,
        schema_builder=squeezenet_schema,
    )


@pytest.fixture
def model_builder(request):
    return request.getfixturevalue(request.param)


# @pytest.mark.skipif(
#     PYTHON_VERSION_IS_NOT_310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
# )
# @pytest.mark.slow_test
# @pytest.mark.flaky(reruns=5, reruns_delay=2)
# def test_happy_pytorch_local_container(sagemaker_session, model_builder, test_image):
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


@patch("sagemaker.huggingface.llm_utils.urllib")
@patch("sagemaker.huggingface.llm_utils.json")
@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,  # or NOT_RUNNING_ON_INF_EXP_DEV_PIPELINE,
    reason="The goal of these test are to test the serving components of our feature",
)
@pytest.mark.parametrize(
    "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
)
@pytest.mark.slow_test
def test_happy_pytorch_sagemaker_endpoint(
    mock_urllib,
    mock_json,
    sagemaker_session,
    model_builder,
    cpu_instance_type,
    test_image,
):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    mock_json.load.return_value = MOCK_HF_MODEL_METADATA_JSON
    mock_hf_model_metadata_url = Mock()
    mock_urllib.request.Request.side_effect = mock_hf_model_metadata_url
    caught_ex = None

    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

    model = model_builder.build(
        mode=Mode.SAGEMAKER_ENDPOINT, role_arn=role_arn, sagemaker_session=sagemaker_session
    )

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
                assert (
                    False
                ), f"{caught_ex} was thrown when running pytorch squeezenet sagemaker endpoint test"


# @pytest.mark.skipif(
#     PYTHON_VERSION_IS_NOT_310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
# )
# @pytest.mark.slow_test
# def test_happy_pytorch_local_container_overwrite_to_sagemaker_endpoint(
#     sagemaker_session, model_builder, cpu_instance_type, test_image
# ):
#     logger.info("Building model in LOCAL_CONTAINER mode...")
#     caught_ex = None
#
#     iam_client = sagemaker_session.boto_session.client("iam")
#     role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
#     logger.debug("Role arn: %s", role_arn)
#
#     model = model_builder.build(
#         mode=Mode.LOCAL_CONTAINER, role_arn=role_arn, sagemaker_session=sagemaker_session
#     )
#
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
#     PYTHON_VERSION_IS_NOT_310,
#     reason="The goal of these test are to test the serving components of our feature",
# )
# @pytest.mark.parametrize(
#     "model_builder", ["model_builder_inference_spec_schema_builder"], indirect=True
# )
# @pytest.mark.slow_test
# def test_happy_pytorch_sagemaker_endpoint_overwrite_to_local_container(
#     sagemaker_session, model_builder, test_image
# ):
#     logger.info("Building model in SAGEMAKER_ENDPOINT mode...")
#     caught_ex = None
#
#     iam_client = sagemaker_session.boto_session.client("iam")
#     role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
#
#     model = model_builder.build(
#         mode=Mode.SAGEMAKER_ENDPOINT, role_arn=role_arn, sagemaker_session=sagemaker_session
#     )
#
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
