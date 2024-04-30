from __future__ import absolute_import
from unittest.mock import Mock, patch

import pytest

from sagemaker.jumpstart.session_utils import (
    _get_model_info_from_inference_component_endpoint_with_inference_component_name,
    _get_model_info_from_inference_component_endpoint_without_inference_component_name,
    _get_model_info_from_model_based_endpoint,
    get_model_info_from_endpoint,
    get_model_info_from_training_job,
)


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_info_from_resource_arn")
def test_get_model_info_from_training_job_happy_case(
    mock_get_jumpstart_model_info_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_info_from_resource_arn.return_value = (
        "model_id",
        "model_version",
        None,
        None,
    )

    retval = get_model_info_from_training_job("bLaH", sagemaker_session=mock_sm_session)

    assert retval == ("model_id", "model_version", None, None)

    mock_get_jumpstart_model_info_from_resource_arn.assert_called_once_with(
        "arn:aws:sagemaker:us-west-2:123456789012:training-job/bLaH", mock_sm_session
    )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_info_from_resource_arn")
def test_get_model_info_from_training_job_config_name(
    mock_get_jumpstart_model_info_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_info_from_resource_arn.return_value = (
        "model_id",
        "model_version",
        None,
        "training_config_name",
    )

    retval = get_model_info_from_training_job("bLaH", sagemaker_session=mock_sm_session)

    assert retval == ("model_id", "model_version", None, "training_config_name")

    mock_get_jumpstart_model_info_from_resource_arn.assert_called_once_with(
        "arn:aws:sagemaker:us-west-2:123456789012:training-job/bLaH", mock_sm_session
    )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_info_from_resource_arn")
def test_get_model_info_from_training_job_no_model_id_inferred(
    mock_get_jumpstart_model_info_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_info_from_resource_arn.return_value = (
        None,
        None,
    )

    with pytest.raises(ValueError):
        get_model_info_from_training_job("blah", sagemaker_session=mock_sm_session)


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_info_from_resource_arn")
def test_get_model_info_from_model_based_endpoint_happy_case(
    mock_get_jumpstart_model_info_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_info_from_resource_arn.return_value = (
        "model_id",
        "model_version",
        None,
        None,
    )

    retval = _get_model_info_from_model_based_endpoint(
        "bLaH", inference_component_name=None, sagemaker_session=mock_sm_session
    )

    assert retval == ("model_id", "model_version", None, None)

    mock_get_jumpstart_model_info_from_resource_arn.assert_called_once_with(
        "arn:aws:sagemaker:us-west-2:123456789012:endpoint/blah", mock_sm_session
    )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_info_from_resource_arn")
def test_get_model_info_from_model_based_endpoint_inference_component_supplied(
    mock_get_jumpstart_model_info_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_info_from_resource_arn.return_value = (
        "model_id",
        "model_version",
        None,
        None,
    )

    with pytest.raises(ValueError):
        _get_model_info_from_model_based_endpoint(
            "blah", inference_component_name="some-name", sagemaker_session=mock_sm_session
        )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_info_from_resource_arn")
def test_get_model_info_from_model_based_endpoint_no_model_id_inferred(
    mock_get_jumpstart_model_info_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_info_from_resource_arn.return_value = (
        None,
        None,
        None,
    )

    with pytest.raises(ValueError):
        _get_model_info_from_model_based_endpoint(
            "blah", inference_component_name="some-name", sagemaker_session=mock_sm_session
        )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_info_from_resource_arn")
def test_get_model_info_from_inference_component_endpoint_with_inference_component_name_happy_case(
    mock_get_jumpstart_model_info_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_info_from_resource_arn.return_value = (
        "model_id",
        "model_version",
        None,
        None,
    )

    retval = _get_model_info_from_inference_component_endpoint_with_inference_component_name(
        "bLaH", sagemaker_session=mock_sm_session
    )

    assert retval == ("model_id", "model_version", None, None)

    mock_get_jumpstart_model_info_from_resource_arn.assert_called_once_with(
        "arn:aws:sagemaker:us-west-2:123456789012:inference-component/bLaH", mock_sm_session
    )


@patch("sagemaker.jumpstart.session_utils.get_jumpstart_model_info_from_resource_arn")
def test_get_model_info_from_inference_component_endpoint_with_inference_component_name_no_model_id_inferred(
    mock_get_jumpstart_model_info_from_resource_arn,
):
    mock_sm_session = Mock()
    mock_sm_session.boto_region_name = "us-west-2"
    mock_sm_session.account_id = Mock(return_value="123456789012")

    mock_get_jumpstart_model_info_from_resource_arn.return_value = (
        None,
        None,
        None,
        None,
    )

    with pytest.raises(ValueError):
        _get_model_info_from_inference_component_endpoint_with_inference_component_name(
            "blah", sagemaker_session=mock_sm_session
        )


@patch(
    "sagemaker.jumpstart.session_utils._get_model_info_from_inference_"
    "component_endpoint_with_inference_component_name"
)
def test_get_model_info_from_inference_component_endpoint_without_inference_component_name_happy_case(
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name,
):
    mock_sm_session = Mock()
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name.return_value = (
        "model_id",
        "model_version",
    )
    mock_sm_session.list_and_paginate_inference_component_names_associated_with_endpoint = Mock(
        return_value=["icname"]
    )

    retval = _get_model_info_from_inference_component_endpoint_without_inference_component_name(
        "blahblah", mock_sm_session
    )

    assert retval == ("model_id", "model_version", "icname")
    mock_sm_session.list_and_paginate_inference_component_names_associated_with_endpoint.assert_called_once_with(
        endpoint_name="blahblah"
    )


@patch(
    "sagemaker.jumpstart.session_utils._get_model_info_from_inference_"
    "component_endpoint_with_inference_component_name"
)
def test_get_model_info_from_inference_component_endpoint_without_ic_name_no_ic_for_endpoint(
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name,
):
    mock_sm_session = Mock()
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name.return_value = (
        "model_id",
        "model_version",
    )
    mock_sm_session.list_and_paginate_inference_component_names_associated_with_endpoint = Mock(
        return_value=[]
    )
    with pytest.raises(ValueError):
        _get_model_info_from_inference_component_endpoint_without_inference_component_name(
            "blahblah", mock_sm_session
        )

    mock_sm_session.list_and_paginate_inference_component_names_associated_with_endpoint.assert_called_once_with(
        endpoint_name="blahblah"
    )


@patch(
    "sagemaker.jumpstart.session_utils._get_model"
    "_info_from_inference_component_endpoint_with_inference_component_name"
)
def test_get_model_id_version_from_ic_endpoint_without_inference_component_name_multiple_ics_for_endpoint(
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name,
):
    mock_sm_session = Mock()
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name.return_value = (
        "model_id",
        "model_version",
    )

    mock_sm_session.list_and_paginate_inference_component_names_associated_with_endpoint = Mock(
        return_value=["icname1", "icname2"]
    )

    with pytest.raises(ValueError):
        _get_model_info_from_inference_component_endpoint_without_inference_component_name(
            "blahblah", mock_sm_session
        )

    mock_sm_session.list_and_paginate_inference_component_names_associated_with_endpoint.assert_called_once_with(
        endpoint_name="blahblah"
    )


@patch("sagemaker.jumpstart.session_utils._get_model_info_from_model_based_endpoint")
def test_get_model_info_from_endpoint_non_inference_component_endpoint(
    mock_get_model_info_from_model_based_endpoint,
):
    mock_sm_session = Mock()
    mock_sm_session.is_inference_component_based_endpoint.return_value = False
    mock_get_model_info_from_model_based_endpoint.return_value = (
        "model_id",
        "model_version",
        None,
        None,
    )

    retval = get_model_info_from_endpoint("blah", sagemaker_session=mock_sm_session)

    assert retval == ("model_id", "model_version", None, None, None)
    mock_get_model_info_from_model_based_endpoint.assert_called_once_with(
        "blah", None, mock_sm_session
    )
    mock_sm_session.is_inference_component_based_endpoint.assert_called_once_with("blah")


@patch(
    "sagemaker.jumpstart.session_utils._get_model_info_from_inference_"
    "component_endpoint_with_inference_component_name"
)
def test_get_model_info_from_endpoint_inference_component_endpoint_with_inference_component_name(
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name,
):
    mock_sm_session = Mock()
    mock_sm_session.is_inference_component_based_endpoint.return_value = True
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name.return_value = (
        "model_id",
        "model_version",
        None,
        None,
    )

    retval = get_model_info_from_endpoint(
        "blah", inference_component_name="icname", sagemaker_session=mock_sm_session
    )

    assert retval == ("model_id", "model_version", "icname", None, None)
    mock_get_model_info_from_inference_component_endpoint_with_inference_component_name.assert_called_once_with(
        "icname", mock_sm_session
    )
    mock_sm_session.is_inference_component_based_endpoint.assert_not_called()


@patch(
    "sagemaker.jumpstart.session_utils._get_model_info_from_inference_component_"
    "endpoint_without_inference_component_name"
)
def test_get_model_info_from_endpoint_inference_component_endpoint_without_inference_component_name(
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name,
):
    mock_sm_session = Mock()
    mock_sm_session.is_inference_component_based_endpoint.return_value = True
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name.return_value = (
        "model_id",
        "model_version",
        None,
        None,
        "inferred-icname",
    )

    retval = get_model_info_from_endpoint("blah", sagemaker_session=mock_sm_session)

    assert retval == ("model_id", "model_version", "inferred-icname", None, None)
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name.assert_called_once()


@patch(
    "sagemaker.jumpstart.session_utils._get_model_info_from_inference_component_"
    "endpoint_without_inference_component_name"
)
def test_get_model_info_from_endpoint_inference_component_endpoint_with_inference_config_name(
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name,
):
    mock_sm_session = Mock()
    mock_sm_session.is_inference_component_based_endpoint.return_value = True
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name.return_value = (
        "model_id",
        "model_version",
        "inference_config_name",
        None,
        "inferred-icname",
    )

    retval = get_model_info_from_endpoint("blah", sagemaker_session=mock_sm_session)

    assert retval == ("model_id", "model_version", "inferred-icname", "inference_config_name", None)
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name.assert_called_once()


@patch(
    "sagemaker.jumpstart.session_utils._get_model_info_from_inference_component_"
    "endpoint_without_inference_component_name"
)
def test_get_model_info_from_endpoint_inference_component_endpoint_with_training_config_name(
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name,
):
    mock_sm_session = Mock()
    mock_sm_session.is_inference_component_based_endpoint.return_value = True
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name.return_value = (
        "model_id",
        "model_version",
        None,
        "training_config_name",
        "inferred-icname",
    )

    retval = get_model_info_from_endpoint("blah", sagemaker_session=mock_sm_session)

    assert retval == ("model_id", "model_version", "inferred-icname", None, "training_config_name")
    mock_get_model_info_from_inference_component_endpoint_without_inference_component_name.assert_called_once()
