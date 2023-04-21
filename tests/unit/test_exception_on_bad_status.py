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

import pytest
from mock import Mock, MagicMock
import sagemaker
from sagemaker.session import _check_job_status

EXPANDED_ROLE = "arn:aws:iam::111111111111:role/ExpandedRole"
REGION = "us-west-2"
MODEL_PACKAGE_NAME = "my_model_package"
JOB_NAME = "my_job_name"
ENDPOINT_NAME = "the_point_of_end"


def get_sagemaker_session(returns_status):
    boto_mock = MagicMock(name="boto_session", region_name=REGION)
    client_mock = MagicMock()
    client_mock.describe_model_package = MagicMock(
        return_value={"ModelPackageStatus": returns_status}
    )
    client_mock.describe_endpoint = MagicMock(return_value={"EndpointStatus": returns_status})
    ims = sagemaker.Session(boto_session=boto_mock, sagemaker_client=client_mock)
    ims.expand_role = Mock(return_value=EXPANDED_ROLE)
    return ims


def test_does_not_raise_when_successfully_created_package():
    try:
        sagemaker_session = get_sagemaker_session(returns_status="Completed")
        sagemaker_session.wait_for_model_package(MODEL_PACKAGE_NAME)
    except sagemaker.exceptions.UnexpectedStatusException:
        pytest.fail("UnexpectedStatusException was thrown while it should not")


def test_raise_when_failed_created_package():
    try:
        sagemaker_session = get_sagemaker_session(returns_status="EnRoute")
        sagemaker_session.wait_for_model_package(MODEL_PACKAGE_NAME)
        assert (
            False
        ), "sagemaker.exceptions.UnexpectedStatusException should have been raised but was not"
    except Exception as e:
        assert type(e) == sagemaker.exceptions.UnexpectedStatusException
        assert e.actual_status == "EnRoute"
        assert "Completed" in e.allowed_statuses


def test_does_not_raise_when_correct_job_status():
    try:
        job = Mock()
        _check_job_status(job, {"TransformationJobStatus": "Stopped"}, "TransformationJobStatus")
    except sagemaker.exceptions.UnexpectedStatusException:
        pytest.fail("UnexpectedStatusException was thrown while it should not")


def test_does_raise_when_incorrect_job_status():
    try:
        job = Mock()
        _check_job_status(job, {"TransformationJobStatus": "Failed"}, "TransformationJobStatus")
        assert (
            False
        ), "sagemaker.exceptions.UnexpectedStatusException should have been raised but was not"
    except Exception as e:
        assert type(e) == sagemaker.exceptions.UnexpectedStatusException
        assert e.actual_status == "Failed"
        assert "Completed" in e.allowed_statuses
        assert "Stopped" in e.allowed_statuses


def test_does_raise_capacity_error_when_incorrect_job_status():
    try:
        job = Mock()
        _check_job_status(
            job,
            {
                "TransformationJobStatus": "Failed",
                "FailureReason": "CapacityError: Unable to provision requested ML compute capacity",
            },
            "TransformationJobStatus",
        )
        assert False, "sagemaker.exceptions.CapacityError should have been raised but was not"
    except Exception as e:
        assert type(e) == sagemaker.exceptions.CapacityError
        assert e.actual_status == "Failed"
        assert "Completed" in e.allowed_statuses
        assert "Stopped" in e.allowed_statuses


def test_does_not_raise_when_successfully_deployed_endpoint():
    try:
        sagemaker_session = get_sagemaker_session(returns_status="InService")
        sagemaker_session.wait_for_endpoint(ENDPOINT_NAME)
    except sagemaker.exceptions.UnexpectedStatusException:
        pytest.fail("UnexpectedStatusException was thrown while it should not")


def test_raise_when_failed_to_deploy_endpoint():
    try:
        sagemaker_session = get_sagemaker_session(returns_status="Failed")
        assert sagemaker_session.wait_for_endpoint(ENDPOINT_NAME)
        assert (
            False
        ), "sagemaker.exceptions.UnexpectedStatusException should have been raised but was not"
    except Exception as e:
        assert type(e) == sagemaker.exceptions.UnexpectedStatusException
        assert e.actual_status == "Failed"
        assert "InService" in e.allowed_statuses
