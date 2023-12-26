from __future__ import absolute_import
from unittest.mock import Mock, patch

import pytest

from sagemaker.jumpstart.session_utils import (
    _get_model_id_version_from_ic_endpoint_with_ic_name,
    _get_model_id_version_from_ic_endpoint_without_ic_name,
    _get_model_id_version_from_non_ic_endpoint,
    get_model_id_version_from_endpoint,
)


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_id_version_from_resource_arn")
def test_get_model_id_version_from_non_ic_endpoint_happy_case(
    mock_get_jumpstart_model_id_version_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_id_version_from_resource_arn.return_value = (
        "model_id",
        "model_version",
    )

    retval = _get_model_id_version_from_non_ic_endpoint(
        "bLaH", inference_component_name=None, sagemaker_session=mock_sm_session
    )

    assert retval == ("model_id", "model_version")

    mock_get_jumpstart_model_id_version_from_resource_arn.assert_called_once_with(
        "arn:aws:sagemaker:us-west-2:123456789012:endpoint/blah", mock_sm_session
    )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_id_version_from_resource_arn")
def test_get_model_id_version_from_non_ic_endpoint_inference_component_supplied(
    mock_get_jumpstart_model_id_version_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_id_version_from_resource_arn.return_value = (
        "model_id",
        "model_version",
    )

    with pytest.raises(ValueError):
        _get_model_id_version_from_non_ic_endpoint(
            "blah", inference_component_name="some-name", sagemaker_session=mock_sm_session
        )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_id_version_from_resource_arn")
def test_get_model_id_version_from_non_ic_endpoint_no_model_id_inferred(
    mock_get_jumpstart_model_id_version_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_id_version_from_resource_arn.return_value = (
        None,
        None,
    )

    with pytest.raises(ValueError):
        _get_model_id_version_from_non_ic_endpoint(
            "blah", inference_component_name="some-name", sagemaker_session=mock_sm_session
        )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_id_version_from_resource_arn")
def test_get_model_id_version_from_ic_endpoint_with_ic_name_happy_case(
    mock_get_jumpstart_model_id_version_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_id_version_from_resource_arn.return_value = (
        "model_id",
        "model_version",
    )

    retval = _get_model_id_version_from_ic_endpoint_with_ic_name(
        "bLaH", sagemaker_session=mock_sm_session
    )

    assert retval == ("model_id", "model_version")

    mock_get_jumpstart_model_id_version_from_resource_arn.assert_called_once_with(
        "arn:aws:sagemaker:us-west-2:123456789012:inference-component/bLaH", mock_sm_session
    )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_id_version_from_resource_arn")
def test_get_model_id_version_from_ic_endpoint_with_ic_name_no_model_id_inferred(
    mock_get_jumpstart_model_id_version_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_id_version_from_resource_arn.return_value = (
        None,
        None,
    )

    with pytest.raises(ValueError):
        _get_model_id_version_from_ic_endpoint_with_ic_name(
            "blah", sagemaker_session=mock_sm_session
        )


@patch("sagemaker.jumpstart.session_utils._get_model_id_version_from_ic_endpoint_with_ic_name")
def test_get_model_id_version_from_ic_endpoint_without_ic_name_happy_case(
    mock_get_model_id_version_from_ic_endpoint_with_ic_name,
):
    mock_sm_session = Mock()
    mock_get_model_id_version_from_ic_endpoint_with_ic_name.return_value = (
        "model_id",
        "model_version",
    )
    mock_sm_session.list_inference_components = Mock(
        return_value={"InferenceComponents": [{"InferenceComponentName": "icname"}]}
    )

    retval = _get_model_id_version_from_ic_endpoint_without_ic_name("blahblah", mock_sm_session)

    assert retval == ("model_id", "model_version", "icname")
    mock_sm_session.list_inference_components.assert_called_once_with(
        endpoint_name_equals="blahblah"
    )


@patch("sagemaker.jumpstart.session_utils._get_model_id_version_from_ic_endpoint_with_ic_name")
def test_get_model_id_version_from_ic_endpoint_without_ic_name_no_ic_for_endpoint(
    mock_get_model_id_version_from_ic_endpoint_with_ic_name,
):
    mock_sm_session = Mock()
    mock_get_model_id_version_from_ic_endpoint_with_ic_name.return_value = (
        "model_id",
        "model_version",
    )
    mock_sm_session.list_inference_components = Mock(return_value={"InferenceComponents": []})

    with pytest.raises(ValueError):
        _get_model_id_version_from_ic_endpoint_without_ic_name("blahblah", mock_sm_session)

    mock_sm_session.list_inference_components.assert_called_once_with(
        endpoint_name_equals="blahblah"
    )


@patch("sagemaker.jumpstart.session_utils._get_model_id_version_from_ic_endpoint_with_ic_name")
def test_get_model_id_version_from_ic_endpoint_without_ic_name_multiple_ics_for_endpoint(
    mock_get_model_id_version_from_ic_endpoint_with_ic_name,
):
    mock_sm_session = Mock()
    mock_get_model_id_version_from_ic_endpoint_with_ic_name.return_value = (
        "model_id",
        "model_version",
    )
    mock_sm_session.list_inference_components = Mock(
        return_value={
            "InferenceComponents": [
                {"InferenceComponentName": "icname1"},
                {"InferenceComponentName": "icname2"},
            ]
        }
    )

    with pytest.raises(ValueError):
        _get_model_id_version_from_ic_endpoint_without_ic_name("blahblah", mock_sm_session)

    mock_sm_session.list_inference_components.assert_called_once_with(
        endpoint_name_equals="blahblah"
    )


@patch("sagemaker.jumpstart.session_utils._get_model_id_version_from_non_ic_endpoint")
def test_get_model_id_version_from_endpoint_non_ic_endpoint(
    mock_get_model_id_version_from_non_ic_endpoint,
):
    mock_sm_session = Mock()
    mock_sm_session.is_ic_based_endpoint.return_value = False
    mock_get_model_id_version_from_non_ic_endpoint.return_value = "model_id", "model_version"

    retval = get_model_id_version_from_endpoint("blah", sagemaker_session=mock_sm_session)

    assert retval == ("model_id", "model_version", None)
    mock_get_model_id_version_from_non_ic_endpoint.assert_called_once_with(
        "blah", None, mock_sm_session
    )
    mock_sm_session.is_ic_based_endpoint.assert_called_once_with("blah")


@patch("sagemaker.jumpstart.session_utils._get_model_id_version_from_ic_endpoint_with_ic_name")
def test_get_model_id_version_from_endpoint_ic_endpoint_with_ic_name(
    mock_get_model_id_version_from_ic_endpoint_with_ic_name,
):
    mock_sm_session = Mock()
    mock_sm_session.is_ic_based_endpoint.return_value = True
    mock_get_model_id_version_from_ic_endpoint_with_ic_name.return_value = (
        "model_id",
        "model_version",
    )

    retval = get_model_id_version_from_endpoint(
        "blah", inference_component_name="icname", sagemaker_session=mock_sm_session
    )

    assert retval == ("model_id", "model_version", "icname")
    mock_get_model_id_version_from_ic_endpoint_with_ic_name.assert_called_once_with(
        "icname", mock_sm_session
    )
    mock_sm_session.is_ic_based_endpoint.assert_not_called()


@patch("sagemaker.jumpstart.session_utils._get_model_id_version_from_ic_endpoint_without_ic_name")
def test_get_model_id_version_from_endpoint_ic_endpoint_without_ic_name(
    mock_get_model_id_version_from_ic_endpoint_without_ic_name,
):
    mock_sm_session = Mock()
    mock_sm_session.is_ic_based_endpoint.return_value = True
    mock_get_model_id_version_from_ic_endpoint_without_ic_name.return_value = (
        "model_id",
        "model_version",
        "inferred-icname",
    )

    retval = get_model_id_version_from_endpoint("blah", sagemaker_session=mock_sm_session)

    assert retval == ("model_id", "model_version", "inferred-icname")
    mock_get_model_id_version_from_ic_endpoint_without_ic_name.assert_called_once()
