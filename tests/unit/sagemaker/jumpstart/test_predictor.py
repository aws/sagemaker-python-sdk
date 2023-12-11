from __future__ import absolute_import
import base64
from unittest import mock

from unittest.mock import patch
from sagemaker.deserializers import JSONDeserializer
from sagemaker.jumpstart.enums import MIMEType

from sagemaker import predictor
from sagemaker.jumpstart.model import JumpStartModel


from sagemaker.jumpstart.utils import verify_model_region_and_return_specs
from sagemaker.serializers import IdentitySerializer
from tests.unit.sagemaker.jumpstart.utils import (
    get_special_model_spec,
)


@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_predictor_support(
    patched_get_model_specs, patched_verify_model_region_and_return_specs
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec

    # version not needed for JumpStart predictor
    model_id, model_version = "predictor-specs-model", "*"

    js_predictor = predictor.retrieve_default(
        endpoint_name="blah", model_id=model_id, model_version=model_version
    )

    assert js_predictor.content_type == MIMEType.X_TEXT
    assert isinstance(js_predictor.serializer, IdentitySerializer)

    assert isinstance(js_predictor.deserializer, JSONDeserializer)
    assert js_predictor.accept == MIMEType.JSON


@patch("sagemaker.jumpstart.payload_utils.JumpStartS3PayloadAccessor.get_object_cached")
@patch("sagemaker.jumpstart.model.is_valid_model_id")
@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_serializable_payload_with_predictor(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_is_valid_model_id,
    patched_get_object_cached,
):

    patched_get_object_cached.return_value = base64.b64decode("encodedimage")
    patched_is_valid_model_id.return_value = True

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec

    model_id, model_version = "default_payloads", "*"

    js_predictor = predictor.retrieve_default(
        endpoint_name="blah", model_id=model_id, model_version=model_version
    )

    default_payload = JumpStartModel(
        model_id=model_id, model_version=model_version
    ).retrieve_example_payload()

    invoke_endpoint_mock = mock.Mock()

    js_predictor.sagemaker_session.sagemaker_runtime_client.invoke_endpoint = invoke_endpoint_mock
    js_predictor._handle_response = mock.Mock()

    assert str(default_payload) == (
        "JumpStartSerializablePayload: {'content_type': 'application/json', 'accept': 'application/json'"
        ", 'body': {'prompt': 'a dog', 'num_images_per_prompt': 2, 'num_inference_steps':"
        " 20, 'guidance_scale': 7.5, 'seed': 43, 'eta': 0.7, 'image':"
        " '$s3_b64<inference-notebook-assets/inpainting_cow.jpg>'}}"
    )

    js_predictor.predict(default_payload)

    invoke_endpoint_mock.assert_called_once_with(
        EndpointName="blah",
        ContentType="application/json",
        Accept="application/json",
        Body='{"prompt": "a dog", "num_images_per_prompt": 2, "num_inference_steps": 20, '
        '"guidance_scale": 7.5, "seed": 43, "eta": 0.7, "image": "encodedimage"}',
    )
