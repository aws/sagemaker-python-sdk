from __future__ import absolute_import
import base64
from unittest import mock

from unittest.mock import Mock, patch

import pytest
from sagemaker.deserializers import JSONDeserializer
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.enums import MIMEType, JumpStartModelType

from sagemaker import predictor
from sagemaker.jumpstart.model import JumpStartModel


from sagemaker.jumpstart.utils import verify_model_region_and_return_specs
from sagemaker.serializers import IdentitySerializer, JSONSerializer
from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec, get_spec_from_base_spec


@patch("sagemaker.predictor.get_model_id_version_from_endpoint")
@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_predictor_support(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_get_jumpstart_model_id_version_from_endpoint,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec

    # version not needed for JumpStart predictor
    model_id, model_version = "predictor-specs-model", "*"

    patched_get_jumpstart_model_id_version_from_endpoint.return_value = (
        model_id,
        model_version,
        None,
    )

    js_predictor = predictor.retrieve_default(
        endpoint_name="blah", model_id=model_id, model_version=model_version
    )

    patched_get_jumpstart_model_id_version_from_endpoint.assert_not_called()

    assert js_predictor.content_type == MIMEType.X_TEXT
    assert isinstance(js_predictor.serializer, IdentitySerializer)

    assert isinstance(js_predictor.deserializer, JSONDeserializer)
    assert js_predictor.accept == MIMEType.JSON


@patch("sagemaker.predictor.get_model_id_version_from_endpoint")
@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_proprietary_predictor_support(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_get_jumpstart_model_id_version_from_endpoint,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_spec_from_base_spec

    # version not needed for JumpStart predictor
    model_id, model_version = "ai21-summarization", "*"

    patched_get_jumpstart_model_id_version_from_endpoint.return_value = (
        model_id,
        model_version,
        None,
    )

    js_predictor = predictor.retrieve_default(
        endpoint_name="blah",
        model_id=model_id,
        model_version=model_version,
        model_type=JumpStartModelType.PROPRIETARY,
    )

    patched_get_jumpstart_model_id_version_from_endpoint.assert_not_called()

    assert js_predictor.content_type == MIMEType.JSON
    assert isinstance(js_predictor.serializer, JSONSerializer)

    assert isinstance(js_predictor.deserializer, JSONDeserializer)
    assert js_predictor.accept == MIMEType.JSON


@patch("sagemaker.predictor.Predictor")
@patch("sagemaker.predictor.get_default_predictor")
@patch("sagemaker.predictor.get_model_id_version_from_endpoint")
@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_predictor_support_no_model_id_supplied_happy_case(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_get_jumpstart_model_id_version_from_endpoint,
    patched_get_default_predictor,
    patched_predictor,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec

    patched_get_jumpstart_model_id_version_from_endpoint.return_value = (
        "predictor-specs-model",
        "1.2.3",
        None,
    )

    mock_session = Mock()

    predictor.retrieve_default(endpoint_name="blah", sagemaker_session=mock_session)

    patched_get_jumpstart_model_id_version_from_endpoint.assert_called_once_with(
        "blah", None, mock_session
    )

    patched_get_default_predictor.assert_called_once_with(
        predictor=patched_predictor.return_value,
        model_id="predictor-specs-model",
        model_version="1.2.3",
        region=None,
        tolerate_deprecated_model=False,
        tolerate_vulnerable_model=False,
        sagemaker_session=mock_session,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
    )


@patch("sagemaker.predictor.get_default_predictor")
@patch("sagemaker.predictor.get_model_id_version_from_endpoint")
@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_predictor_support_no_model_id_supplied_sad_case(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_get_jumpstart_model_id_version_from_endpoint,
    patched_get_default_predictor,
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec

    # no JS tags attached to endpoint
    patched_get_jumpstart_model_id_version_from_endpoint.return_value = (None, None, None)

    with pytest.raises(ValueError):
        predictor.retrieve_default(
            endpoint_name="blah",
        )

    patched_get_jumpstart_model_id_version_from_endpoint.assert_called_once_with(
        "blah", None, DEFAULT_JUMPSTART_SAGEMAKER_SESSION
    )
    patched_get_default_predictor.assert_not_called()


@patch("sagemaker.predictor.get_model_id_version_from_endpoint")
@patch("sagemaker.jumpstart.payload_utils.JumpStartS3PayloadAccessor.get_object_cached")
@patch("sagemaker.jumpstart.model.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.utils.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_serializable_payload_with_predictor(
    patched_get_model_specs,
    patched_verify_model_region_and_return_specs,
    patched_validate_model_id_and_get_type,
    patched_get_object_cached,
    patched_get_model_id_version_from_endpoint,
):

    patched_get_object_cached.return_value = base64.b64decode("encodedimage")
    patched_validate_model_id_and_get_type.return_value = True

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec

    model_id, model_version = "default_payloads", "*"
    patched_get_model_id_version_from_endpoint.return_value = model_id, model_version, None

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
