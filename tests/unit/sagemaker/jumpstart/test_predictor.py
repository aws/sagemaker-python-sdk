from __future__ import absolute_import

from unittest.mock import patch
from sagemaker.deserializers import JSONDeserializer
from sagemaker.jumpstart.enums import MIMEType
from sagemaker.jumpstart.predictor import JumpStartPredictor


from sagemaker.jumpstart.utils import verify_model_region_and_return_specs
from sagemaker.serializers import SimpleBaseSerializer
from tests.unit.sagemaker.jumpstart.utils import (
    get_special_model_spec,
)


@patch("sagemaker.jumpstart.artifacts.verify_model_region_and_return_specs")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_list_jumpstart_scripts(
    patched_get_model_specs, patched_verify_model_region_and_return_specs
):

    patched_verify_model_region_and_return_specs.side_effect = verify_model_region_and_return_specs
    patched_get_model_specs.side_effect = get_special_model_spec

    # version not needed for JumpStart predictor
    model_id = "predictor-specs-model"

    predictor = JumpStartPredictor(endpoint_name="blah", model_id=model_id)

    assert predictor.content_type == MIMEType.X_TEXT
    assert predictor.serializer == SimpleBaseSerializer

    assert predictor.deserializer == JSONDeserializer
    assert predictor.accept == MIMEType.JSON
