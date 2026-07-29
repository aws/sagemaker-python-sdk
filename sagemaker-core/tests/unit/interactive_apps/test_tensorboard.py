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

from sagemaker.core.interactive_apps.tensorboard import TensorBoardApp


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
def test_tb_init_and_url_no_metadata_file(mock_metadata_file_present, mock_client):
    """
    Test TensorBoardApp URL for the case when no metadata file is present.
    """
    mock_metadata_file_present.return_value = False
    mock_client.return_value = boto3.client("sagemaker")
    tb_app = TensorBoardApp(TEST_REGION)
    assert tb_app.region == TEST_REGION
    assert tb_app._domain_id is None
    assert tb_app._user_profile_name is None
    assert tb_app._in_studio_env is False

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

    # test url when opened in web browser
    with patch("webbrowser.open") as mock_web_browser_open:
        url = tb_app.get_app_url(
            open_in_default_web_browser=True,
        )
        mock_web_browser_open.assert_called_with(
            BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)
        )
        assert url == ""


@patch("boto3.client")
@patch("os.path.isfile")
def test_tb_init_non_studio_metadata_file(mock_metadata_file_present, mock_client):
    """
    Test TensorBoardApp URL for the case when metadata file is present,
    but domain id and/or user profile name are not present in it.
    """
    mock_metadata_file_present.return_value = True
    mock_client.return_value = boto3.client("sagemaker")
    with patch("builtins.open", mock_open(read_data=json.dumps({"Fake": "Fake"}))):
        tb_app = TensorBoardApp(TEST_REGION)
        assert tb_app.region == TEST_REGION
        assert tb_app._domain_id is None
        assert tb_app._user_profile_name is None
        assert tb_app._in_studio_env is False
        assert tb_app.get_app_url(
            open_in_default_web_browser=False
        ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    with patch("builtins.open", mock_open(read_data=json.dumps({"DomainId": TEST_DOMAIN}))):
        tb_app = TensorBoardApp(TEST_REGION)
        assert tb_app.region == TEST_REGION
        assert tb_app._domain_id is None
        assert tb_app._user_profile_name is None
        assert tb_app._in_studio_env is False
        assert tb_app.get_app_url(
            open_in_default_web_browser=False
        ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    with patch(
        "builtins.open", mock_open(read_data=json.dumps({"UserProfileName": TEST_USER_PROFILE}))
    ):
        tb_app = TensorBoardApp(TEST_REGION)
        assert tb_app.region == TEST_REGION
        assert tb_app._domain_id is None
        assert tb_app._user_profile_name is None
        assert tb_app._in_studio_env is False
        assert tb_app.get_app_url(
            open_in_default_web_browser=False
        ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
@patch("os.path.isfile")
def test_tb_init_and_url_valid_studio_medatada_file(mock_metadata_file_present, mock_client):
    """
    Test TensorBoardApp URL for the case when metadata file is present,
    and contains valid domain id and user profile name.
    """
    mock_metadata_file_present.return_value = True
    mock_client.return_value = boto3.client("sagemaker")
    with patch("builtins.open", mock_open(read_data=TEST_NOTEBOOK_METADATA)):
        tb_app = TensorBoardApp(TEST_REGION)
        assert tb_app.region == TEST_REGION
        assert tb_app._domain_id == TEST_DOMAIN
        assert tb_app._user_profile_name == TEST_USER_PROFILE
        assert tb_app._in_studio_env is True

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

        # test url when opened in web browser
        with patch("webbrowser.open") as mock_web_browser_open:
            url = tb_app.get_app_url(
                open_in_default_web_browser=True,
            )
            mock_web_browser_open.assert_called_with(
                BASE_URL_STUDIO_FORMAT.format(TEST_DOMAIN, TEST_REGION) + "/#sagemaker_data_manager"
            )
            assert url == ""


@patch("boto3.client")
@patch("os.path.isfile")
def test_tb_init_and_url_invalid_studio_medatada_file(mock_metadata_file_present, mock_client):
    """
    Test TensorBoardApp URL for the case when metadata file is present,
    and contains invalid domain id and/or user profile name.
    """
    mock_metadata_file_present.return_value = True
    mock_client.return_value = boto3.client("sagemaker")

    # test invalid user profile name
    with patch(
        "builtins.open",
        mock_open(read_data=json.dumps({"DomainId": TEST_DOMAIN, "UserProfileName": "u" * 64})),
    ):
        tb_app = TensorBoardApp(TEST_REGION)
        assert tb_app.region == TEST_REGION
        assert tb_app._domain_id == TEST_DOMAIN
        assert tb_app._in_studio_env is True
        assert tb_app.get_app_url(
            open_in_default_web_browser=False
        ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    # test invalid domain id
    with patch(
        "builtins.open",
        mock_open(
            read_data=json.dumps({"DomainId": "d" * 64, "UserProfileName": TEST_USER_PROFILE})
        ),
    ):
        tb_app = TensorBoardApp(TEST_REGION)
        assert tb_app.region == TEST_REGION
        assert tb_app._user_profile_name == TEST_USER_PROFILE
        assert tb_app._in_studio_env is True
        assert tb_app.get_app_url(
            open_in_default_web_browser=False
        ) == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_success(mock_init, mock_client):
    mock_init.return_value = None
    resp = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "AuthorizedUrl": TEST_PRESIGNED_URL,
    }
    attrs = {"create_presigned_domain_url.return_value": resp}
    mock_client.return_value = Mock(**attrs)

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    # test url without job redirect
    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == f"{TEST_PRESIGNED_URL}&redirect=TensorBoard"

    # test url with valid job redirect
    url = tb_app.get_app_url(
        training_job_name=TEST_TRAINING_JOB,
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url.startswith(f"{TEST_PRESIGNED_URL}&redirect=TensorBoard&state=")
    assert url.endswith("==")

    # test url with invalid job redirect
    with pytest.raises(ValueError):
        tb_app.get_app_url(
            training_job_name="invald_job_name!",
            create_presigned_domain_url=True,
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            open_in_default_web_browser=False,
        )
        mock_web_browser_open.assert_called_with(f"{TEST_PRESIGNED_URL}&redirect=TensorBoard")
        assert url == ""


@patch("boto3.client")
def test_tb_presigned_url_not_returned_without_presigned_flag(mock_client):
    mock_client.return_value = boto3.client("sagemaker")

    url = TensorBoardApp(TEST_REGION).get_app_url(
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        create_presigned_domain_url=False,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
def test_tb_presigned_url_failure(mock_client):
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


def test_tb_invalid_presigned_kwargs():
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
def test_tb_valid_presigned_kwargs(mock_client):

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

    # test url when opened in web browser
    with patch("webbrowser.open") as mock_web_browser_open:
        url = tb_app.get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=True,
        )
        mock_web_browser_open.assert_called_with(f"{TEST_PRESIGNED_URL}&redirect=TensorBoard")
        assert url == ""


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_invalid_params(mock_init, mock_client):
    mock_init.return_value = None
    mock_client.return_value = boto3.client("sagemaker")

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False

    url = tb_app.get_app_url(
        create_presigned_domain_url=False,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id="d" * 64,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name="u" * 64,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_failure(mock_init, mock_client):
    mock_init.return_value = None
    resp = {"ResponseMetadata": {"HTTPStatusCode": 400}}
    attrs = {"create_presigned_domain_url.return_value": resp}
    mock_client.return_value = Mock(**attrs)

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    with pytest.raises(ValueError):
        tb_app.get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=False,
        )


@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_invalid_presigned_kwargs(mock_init):
    mock_init.return_value = None
    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    with pytest.raises(botocore.exceptions.ParamValidationError):
        invalid_kwargs = {"fake-parameter": True}
        tb_app.get_app_url(
            create_presigned_domain_url=True,
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            open_in_default_web_browser=False,
            optional_create_presigned_url_kwargs=invalid_kwargs,
        )


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_valid_presigned_kwargs(mock_init, mock_client):
    mock_init.return_value = None
    resp = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "AuthorizedUrl": TEST_PRESIGNED_URL,
    }
    mock_client = boto3.client("sagemaker")
    mock_client.create_presigned_domain_url = Mock(name="create_presigned_domain_url")
    mock_client.create_presigned_domain_url.return_value = resp

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    valid_kwargs = {"ExpiresInSeconds": 1500}
    tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
        optional_create_presigned_url_kwargs=valid_kwargs,
    )
    mock_client.create_presigned_domain_url.assert_called_with(**valid_kwargs)

    # test url when opened in web browser
    with patch("webbrowser.open") as mock_web_browser_open:
        url = tb_app.get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=True,
        )
        mock_web_browser_open.assert_called_with(f"{TEST_PRESIGNED_URL}&redirect=TensorBoard")
        assert url == ""


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_invalid_params(mock_init, mock_client):
    mock_init.return_value = None
    mock_client.return_value = boto3.client("sagemaker")

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False

    url = tb_app.get_app_url(
        create_presigned_domain_url=False,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id="d" * 64,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name="u" * 64,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_failure(mock_init, mock_client):
    mock_init.return_value = None
    resp = {"ResponseMetadata": {"HTTPStatusCode": 400}}
    attrs = {"create_presigned_domain_url.return_value": resp}
    mock_client.return_value = Mock(**attrs)

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    with pytest.raises(ValueError):
        tb_app.get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=False,
        )


@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_invalid_presigned_kwargs(mock_init):
    mock_init.return_value = None
    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    with pytest.raises(botocore.exceptions.ParamValidationError):
        invalid_kwargs = {"fake-parameter": True}
        tb_app.get_app_url(
            create_presigned_domain_url=True,
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            open_in_default_web_browser=False,
            optional_create_presigned_url_kwargs=invalid_kwargs,
        )


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_valid_presigned_kwargs(mock_init, mock_client):
    mock_init.return_value = None
    resp = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "AuthorizedUrl": TEST_PRESIGNED_URL,
    }
    mock_client = boto3.client("sagemaker")
    mock_client.create_presigned_domain_url = Mock(name="create_presigned_domain_url")
    mock_client.create_presigned_domain_url.return_value = resp

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    valid_kwargs = {"ExpiresInSeconds": 1500}
    tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
        optional_create_presigned_url_kwargs=valid_kwargs,
    )
    mock_client.create_presigned_domain_url.assert_called_with(**valid_kwargs)

    # test url when opened in web browser
    with patch("webbrowser.open") as mock_web_browser_open:
        url = tb_app.get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=True,
        )
        mock_web_browser_open.assert_called_with(f"{TEST_PRESIGNED_URL}&redirect=TensorBoard")
        assert url == ""


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_invalid_params(mock_init, mock_client):
    mock_init.return_value = None
    mock_client.return_value = boto3.client("sagemaker")

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False

    url = tb_app.get_app_url(
        create_presigned_domain_url=False,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id="d" * 64,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name="u" * 64,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_failure(mock_init, mock_client):
    mock_init.return_value = None
    resp = {"ResponseMetadata": {"HTTPStatusCode": 400}}
    attrs = {"create_presigned_domain_url.return_value": resp}
    mock_client.return_value = Mock(**attrs)

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    with pytest.raises(ValueError):
        tb_app.get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=False,
        )


@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_invalid_presigned_kwargs(mock_init):
    mock_init.return_value = None
    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    with pytest.raises(botocore.exceptions.ParamValidationError):
        invalid_kwargs = {"fake-parameter": True}
        tb_app.get_app_url(
            create_presigned_domain_url=True,
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            open_in_default_web_browser=False,
            optional_create_presigned_url_kwargs=invalid_kwargs,
        )


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_valid_presigned_kwargs(mock_init, mock_client):
    mock_init.return_value = None
    resp = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "AuthorizedUrl": TEST_PRESIGNED_URL,
    }
    mock_client = boto3.client("sagemaker")
    mock_client.create_presigned_domain_url = Mock(name="create_presigned_domain_url")
    mock_client.create_presigned_domain_url.return_value = resp

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    valid_kwargs = {"ExpiresInSeconds": 1500}
    tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
        optional_create_presigned_url_kwargs=valid_kwargs,
    )
    mock_client.create_presigned_domain_url.assert_called_with(**valid_kwargs)

    # test url when opened in web browser
    with patch("webbrowser.open") as mock_web_browser_open:
        url = tb_app.get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=True,
        )
        mock_web_browser_open.assert_called_with(f"{TEST_PRESIGNED_URL}&redirect=TensorBoard")
        assert url == ""


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_invalid_params(mock_init, mock_client):
    mock_init.return_value = None
    mock_client.return_value = boto3.client("sagemaker")

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False

    url = tb_app.get_app_url(
        create_presigned_domain_url=False,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id="d" * 64,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    url = tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name="u" * 64,
        open_in_default_web_browser=False,
    )
    assert url == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_presigned_url_failure(mock_init, mock_client):
    mock_init.return_value = None
    resp = {"ResponseMetadata": {"HTTPStatusCode": 400}}
    attrs = {"create_presigned_domain_url.return_value": resp}
    mock_client.return_value = Mock(**attrs)

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    with pytest.raises(ValueError):
        tb_app.get_app_url(
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            create_presigned_domain_url=True,
            open_in_default_web_browser=False,
        )


@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_invalid_presigned_kwargs(mock_init):
    mock_init.return_value = None
    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    with pytest.raises(botocore.exceptions.ParamValidationError):
        invalid_kwargs = {"fake-parameter": True}
        tb_app.get_app_url(
            create_presigned_domain_url=True,
            domain_id=TEST_DOMAIN,
            user_profile_name=TEST_USER_PROFILE,
            open_in_default_web_browser=False,
            optional_create_presigned_url_kwargs=invalid_kwargs,
        )


@patch("boto3.client")
@patch("sagemaker.core.interactive_apps.base_interactive_app.BaseInteractiveApp.__init__")
def test_tb_valid_presigned_kwargs(mock_init, mock_client):
    mock_init.return_value = None
    resp = {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "AuthorizedUrl": TEST_PRESIGNED_URL,
    }
    mock_client = boto3.client("sagemaker")
    mock_client.create_presigned_domain_url = Mock(name="create_presigned_domain_url")
    mock_client.create_presigned_domain_url.return_value = resp

    tb_app = TensorBoardApp(TEST_REGION)
    tb_app.region = TEST_REGION
    tb_app._domain_id = None
    tb_app._user_profile_name = None
    tb_app._in_studio_env = False
    tb_app._sagemaker_client = boto3.client("sagemaker", region_name=TEST_REGION)

    valid_kwargs = {"ExpiresInSeconds": 1500}
    tb_app.get_app_url(
        create_presigned_domain_url=True,
        domain_id=TEST_DOMAIN,
        user_profile_name=TEST_USER_PROFILE,
        open_in_default_web_browser=False,
        optional_create_presigned_url_kwargs=valid_kwargs,
    )
    mock_client.create_presigned_domain_url.assert_called_with(**valid_kwargs)


def test_tb_init_with_default_region():
    """
    Test TensorBoardApp init when user does not provide region.
    """
    # happy case
    with patch(
        "sagemaker.core.helper.session_helper.Session.boto_region_name", new_callable=PropertyMock
    ) as region_mock:
        region_mock.return_value = TEST_REGION
        tb_app = TensorBoardApp()
        assert tb_app.region == TEST_REGION

    # no default region configured
    with patch(
        "sagemaker.core.helper.session_helper.Session.boto_region_name", new_callable=PropertyMock
    ) as region_mock:
        region_mock.side_effect = [ValueError()]
        with pytest.raises(ValueError):
            tb_app = TensorBoardApp()
