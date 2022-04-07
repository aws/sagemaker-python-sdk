from __future__ import absolute_import
from unittest.mock import Mock, patch
from sagemaker.jumpstart import notebook_utils
from tests.unit.sagemaker.jumpstart.utils import (
    get_prototype_model_spec,
)


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_get_model_url(
    patched_get_model_specs: Mock,
):

    patched_get_model_specs.side_effect = get_prototype_model_spec

    model_id, version = "xgboost-classification-model", "1.0.0"
    assert "https://xgboost.readthedocs.io/en/latest/" == notebook_utils.get_model_url(
        model_id, version
    )

    model_id, version = "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0"
    assert (
        "https://tfhub.dev/google/bit/m-r101x1/ilsvrc2012_classification/1"
        == notebook_utils.get_model_url(model_id, version)
    )

    model_id, version = "tensorflow-ic-bit-m-r101x1-ilsvrc2012-classification-1", "1.0.0"
    region = "fake-region"

    patched_get_model_specs.reset_mock()
    patched_get_model_specs.side_effect = lambda *largs, **kwargs: get_prototype_model_spec(
        *largs,
        region="us-west-2",
        **{key: value for key, value in kwargs.items() if key != "region"}
    )

    notebook_utils.get_model_url(model_id, version, region=region)

    patched_get_model_specs.assert_called_once_with(
        model_id=model_id, version=version, region=region
    )
