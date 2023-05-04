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

from sagemaker.interactive_apps.tensorboard import TensorBoardApp
from unittest.mock import patch, mock_open, PropertyMock

import json
import pytest

TEST_DOMAIN = "testdomain"
TEST_USER_PROFILE = "testuser"
TEST_REGION = "testregion"
TEST_NOTEBOOK_METADATA = json.dumps({"DomainId": TEST_DOMAIN, "UserProfileName": TEST_USER_PROFILE})
TEST_TRAINING_JOB = "testjob"

BASE_URL_STUDIO_FORMAT = "https://{}.studio.{}.sagemaker.aws/tensorboard/default"
REDIRECT_STUDIO_FORMAT = (
    "/data/plugin/sagemaker_data_manager/add_folder_or_job?Redirect=True&Name={}"
)
BASE_URL_NON_STUDIO_FORMAT = (
    "https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/tensor-board-landing"
)
REDIRECT_NON_STUDIO_FORMAT = "/{}"


@patch("os.path.isfile")
def test_tb_init_and_url_non_studio_user(mock_file_exists):
    """
    Test TensorBoardApp for non Studio users.
    """
    mock_file_exists.return_value = False
    tb_app = TensorBoardApp(TEST_REGION)
    assert tb_app.region == TEST_REGION
    assert tb_app._domain_id is None
    assert tb_app._user_profile_name is None
    assert tb_app._valid_domain_and_user is False

    # test url without job redirect
    assert tb_app.get_app_url() == BASE_URL_NON_STUDIO_FORMAT.format(region=TEST_REGION)

    # test url with valid job redirect
    assert tb_app.get_app_url(TEST_TRAINING_JOB) == BASE_URL_NON_STUDIO_FORMAT.format(
        region=TEST_REGION
    ) + REDIRECT_NON_STUDIO_FORMAT.format(TEST_TRAINING_JOB)

    # test url with invalid job redirect
    with pytest.raises(ValueError):
        tb_app.get_app_url("invald_job_name!")


@patch("os.path.isfile")
def test_tb_init_and_url_studio_user_valid_medatada(mock_file_exists):
    """
    Test TensorBoardApp for Studio user when the notebook metadata file provided by Studio is valid.
    """
    mock_file_exists.return_value = True
    with patch("builtins.open", mock_open(read_data=TEST_NOTEBOOK_METADATA)):
        tb_app = TensorBoardApp(TEST_REGION)
        assert tb_app.region == TEST_REGION
        assert tb_app._domain_id == TEST_DOMAIN
        assert tb_app._user_profile_name == TEST_USER_PROFILE
        assert tb_app._valid_domain_and_user is True

        # test url without job redirect
        assert (
            tb_app.get_app_url()
            == BASE_URL_STUDIO_FORMAT.format(TEST_DOMAIN, TEST_REGION) + "/#sagemaker_data_manager"
        )

        # test url with valid job redirect
        assert tb_app.get_app_url(TEST_TRAINING_JOB) == BASE_URL_STUDIO_FORMAT.format(
            TEST_DOMAIN, TEST_REGION
        ) + REDIRECT_STUDIO_FORMAT.format(TEST_TRAINING_JOB)

        # test url with invalid job redirect
        with pytest.raises(ValueError):
            tb_app.get_app_url("invald_job_name!")


@patch("os.path.isfile")
def test_tb_init_and_url_studio_user_invalid_medatada(mock_file_exists):
    """
    Test TensorBoardApp for Studio user when the notebook metadata file provided by Studio is invalid.
    """
    mock_file_exists.return_value = True

    # test file does not contain domain and user profle
    with patch("builtins.open", mock_open(read_data=json.dumps({"Fake": "Fake"}))):
        assert TensorBoardApp(TEST_REGION).get_app_url() == BASE_URL_NON_STUDIO_FORMAT.format(
            region=TEST_REGION
        )

    # test invalid user profile name
    with patch(
        "builtins.open",
        mock_open(read_data=json.dumps({"DomainId": TEST_DOMAIN, "UserProfileName": "u" * 64})),
    ):
        assert TensorBoardApp(TEST_REGION).get_app_url() == BASE_URL_NON_STUDIO_FORMAT.format(
            region=TEST_REGION
        )

    # test invalid domain id
    with patch(
        "builtins.open",
        mock_open(
            read_data=json.dumps({"DomainId": "d" * 64, "UserProfileName": TEST_USER_PROFILE})
        ),
    ):
        assert TensorBoardApp(TEST_REGION).get_app_url() == BASE_URL_NON_STUDIO_FORMAT.format(
            region=TEST_REGION
        )


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
