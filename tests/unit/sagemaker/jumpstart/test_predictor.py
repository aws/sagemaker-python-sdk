from __future__ import absolute_import

from unittest.mock import patch
from sagemaker.deserializers import JSONDeserializer
from sagemaker.jumpstart.enums import MIMEType

from sagemaker import predictor


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
