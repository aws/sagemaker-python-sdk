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
"""Tests related to TensorBoardApp"""
from __future__ import absolute_import

import json
from unittest.mock import patch, Mock, mock_open, PropertyMock

import boto3
import botocore
import pytest

from sagemaker.interactive_apps.tensorboard import TensorBoardApp


TEST_DOMAIN = "testdomain"
TEST_USER_PROFILE = "testuser"
TEST_REGION = "testregion"
TEST_NOTEBOOK_METADATA = json.dumps({"DomainId": TEST_DOMAIN, "UserProfileName": TEST_USER_PROFILE})
TEST_PRESIGNED_URL = (
    f"https://{TEST_DOMAIN}.studio.{TEST_REGION}.sagemaker.aws/auth?token=FAKETOKEN"
)
TEST_TRAINING_JOB = "testjob"

BASE_URL_STUDIO_FORMAT = "https://{}.studio.{}.sagemaker.aws/tensorboard/default"
REDIRECT_STUDIO_FORMAT = (
    "/data/plugin/sagemaker_data_manager/add_folder_or_job?Redirect=True&Name={}"
)
BASE_URL_NON_STUDIO_FORMAT = (
    "https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/tensor-board-landing"
)
REDIRECT_NON_STUDIO_FORMAT = "/{}"


@patch("boto3.client")
@patch("os.path.isfile")
def test_tb_init_and_url_non_studio_user(mock_in_studio, mock_client):
    """
    Test TensorBoardApp for non Studio users.
    """
    mock_in_studio.return_value = False
    mock_client.return_value = boto3.client("sagemaker")
    tb_app = TensorBoardApp(TEST_REGION)
    assert tb_app.region == TEST_REGION
    assert tb_app._domain_id is None
    assert tb_app._user_profile_name is None
    assert tb_app._validate_domain_and_user() is False

    # test url without job redirect
    assert tb_app.get_app_url(
        open_in_default_web_browser=False
    ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    # test url with valid job redirect
    assert tb_app.get_app_url(
        TEST_TRAINING_JOB, open_in_default_web_browser=False
    ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION) + REDIRECT_NON_STUDIO_FORMAT.format(
        TEST_TRAINING_JOB
    )

    # test url with invalid job redirect
    with pytest.raises(ValueError):
        tb_app.get_app_url("invald_job_name!")


@patch("boto3.client")
@patch("sagemaker.interactive_apps.base_interactive_app.BaseInteractiveApp._is_in_studio")
def test_tb_init_and_url_studio_user_valid_medatada(mock_in_studio, mock_client):
    """
    Test TensorBoardApp for Studio user when the notebook metadata file provided by Studio is valid.
    """
    mock_in_studio.return_value = True
    mock_client.return_value = boto3.client("sagemaker")
    with patch("builtins.open", mock_open(read_data=TEST_NOTEBOOK_METADATA)):
        tb_app = TensorBoardApp(TEST_REGION)
        assert tb_app.region == TEST_REGION
        assert tb_app._domain_id == TEST_DOMAIN
        assert tb_app._user_profile_name == TEST_USER_PROFILE
        assert tb_app._validate_domain_and_user() is True

        # test url without job redirect
        assert (
            tb_app.get_app_url(open_in_default_web_browser=False)
            == BASE_URL_STUDIO_FORMAT.format(TEST_DOMAIN, TEST_REGION) + "/#sagemaker_data_manager"
        )

        # test url with valid job redirect
        assert tb_app.get_app_url(
            TEST_TRAINING_JOB, open_in_default_web_browser=False
        ) == BASE_URL_STUDIO_FORMAT.format(
            TEST_DOMAIN, TEST_REGION
        ) + REDIRECT_STUDIO_FORMAT.format(
            TEST_TRAINING_JOB
        )

        # test url with invalid job redirect
        with pytest.raises(ValueError):
            tb_app.get_app_url("invald_job_name!")


@patch("boto3.client")
@patch("sagemaker.interactive_apps.base_interactive_app.BaseInteractiveApp._is_in_studio")
def test_tb_init_and_url_studio_user_invalid_medatada(mock_in_studio, mock_client):
    """
    Test TensorBoardApp for Amazon SageMaker Studio user when the notebook metadata file provided
    by Studio is invalid.
    """
    mock_in_studio.return_value = True
    mock_client.return_value = boto3.client("sagemaker")

    # test file does not contain domain and user profle
    with patch("builtins.open", mock_open(read_data=json.dumps({"Fake": "Fake"}))):
        assert TensorBoardApp(TEST_REGION).get_app_url(
            open_in_default_web_browser=False
        ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    # test invalid user profile name
    with patch(
        "builtins.open",
        mock_open(read_data=json.dumps({"DomainId": TEST_DOMAIN, "UserProfileName": "u" * 64})),
    ):
        assert TensorBoardApp(TEST_REGION).get_app_url(
            open_in_default_web_browser=False
        ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    # test invalid domain id
    with patch(
        "builtins.open",
        mock_open(
            read_data=json.dumps({"DomainId": "d" * 64, "UserProfileName": TEST_USER_PROFILE})
        ),
    ):
        assert TensorBoardApp(TEST_REGION).get_app_url(
            open_in_default_web_browser=False
        ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
@patch("sagemaker.interactive_apps.base_interactive_app.BaseInteractiveApp._is_in_studio")
def test_tb_presigned_url_success(mock_in_studio, mock_client):
    mock_in_studio.return_value = False
    resp = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "AuthorizedUrl": TEST_PRESIGNED_URL,
    }
    attrs = {"create_presigned_domain_url.return_value": resp}
    mock_client.return_value = Mock(**attrs)

    url = TensorBoardApp(TEST_REGION).get_app_url(
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        create_presigned_domain_url=True,
        open_in_default_web_browser=False,
    )
    assert url == f"{TEST_PRESIGNED_URL}&redirect=TensorBoard"

    url = TensorBoardApp(TEST_REGION).get_app_url(
        training_job_name=TEST_TRAINING_JOB,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        create_presigned_domain_url=True,
        open_in_default_web_browser=False,
    )
    assert url.startswith(f"{TEST_PRESIGNED_URL}&redirect=TensorBoard&state=")
    assert url.endswith("==")


@patch("boto3.client")
@patch("sagemaker.interactive_apps.base_interactive_app.BaseInteractiveApp._is_in_studio")
def test_tb_presigned_url_success_open_in_web_browser(mock_in_studio, mock_client):
    mock_in_studio.return_value = False
    resp = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "AuthorizedUrl": TEST_PRESIGNED_URL,
    }
    attrs = {"create_presigned_domain_url.return_value": resp}
    mock_client.return_value = Mock(**attrs)

    with patch("webbrowser.open") as mock_web_browser_open:
        url = TensorBoardApp(TEST_REGION).get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=True,
        )
        mock_web_browser_open.assert_called_with(f"{TEST_PRESIGNED_URL}&redirect=TensorBoard")
        assert url == ""


@patch("boto3.client")
@patch("sagemaker.interactive_apps.base_interactive_app.BaseInteractiveApp._is_in_studio")
def test_tb_presigned_url_not_returned_without_presigned_flag(mock_in_studio, mock_client):
    mock_in_studio.return_value = False
    mock_client.return_value = boto3.client("sagemaker")

    url = TensorBoardApp(TEST_REGION).get_app_url(
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        create_presigned_domain_url=False,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
@patch("sagemaker.interactive_apps.base_interactive_app.BaseInteractiveApp._is_in_studio")
def test_tb_presigned_url_failure(mock_in_studio, mock_client):
    mock_in_studio.return_value = False
    resp = {"ResponseMetadata": {"HTTPStatusCode": 400}}
    attrs = {"create_presigned_domain_url.return_value": resp}
    mock_client.return_value = Mock(**attrs)

    with pytest.raises(ValueError):
        TensorBoardApp(TEST_REGION).get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=False,
        )


@patch("sagemaker.interactive_apps.base_interactive_app.BaseInteractiveApp._is_in_studio")
def test_tb_invalid_presigned_kwargs(mock_in_studio):
    mock_in_studio.return_value = False
    invalid_kwargs = {
        "fake-parameter": True,
        "DomainId": TEST_DOMAIN,
        "UserProfileName": TEST_USER_PROFILE,
    }

    with pytest.raises(botocore.exceptions.ParamValidationError):
        TensorBoardApp(TEST_REGION).get_app_url(
            optional_create_presigned_url_kwargs=invalid_kwargs,
            create_presigned_domain_url=True,
        )


@patch("boto3.client")
@patch("sagemaker.interactive_apps.base_interactive_app.BaseInteractiveApp._is_in_studio")
def test_tb_valid_presigned_kwargs(mock_in_studio, mock_client):
    mock_in_studio.return_value = False

    rsp = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "AuthorizedUrl": TEST_PRESIGNED_URL,
    }
    mock_client = boto3.client("sagemaker")
    mock_client.create_presigned_domain_url = Mock(name="create_presigned_domain_url")
    mock_client.create_presigned_domain_url.return_value = rsp

    valid_kwargs = {"DomainId": TEST_DOMAIN, "UserProfileName": TEST_USER_PROFILE}

    url = TensorBoardApp(TEST_REGION).get_app_url(
        optional_create_presigned_url_kwargs=valid_kwargs,
        create_presigned_domain_url=True,
        open_in_default_web_browser=False,
    )

    assert url == f"{TEST_PRESIGNED_URL}&redirect=TensorBoard"
    mock_client.create_presigned_domain_url.assert_called_once_with(**valid_kwargs)


def test_tb_init_with_default_region():
    """
    Test TensorBoardApp init when user does not provide region.
    """
    # happy case
    with patch("sagemaker.Session.boto_region_name", new_callable=PropertyMock) as region_mock:
        region_mock.return_value = TEST_REGION
        tb_app = TensorBoardApp()
        assert tb_app.region == TEST_REGION

    # no default region configured
    with patch("sagemaker.Session.boto_region_name", new_callable=PropertyMock) as region_mock:
        region_mock.side_effect = [ValueError()]
        with pytest.raises(ValueError):
            tb_app = TensorBoardApp()
