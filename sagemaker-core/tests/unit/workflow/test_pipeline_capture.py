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
"""Tests for the shared ``PipelineSession`` capture helpers."""
from __future__ import absolute_import

import json
from unittest.mock import MagicMock

from sagemaker.core.shapes import Tag
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_capture import (
    capture_create_job_request,
    capture_training_request,
)


def _captured(session):
    (request, _, func_name), _ = session._intercept_create_request.call_args
    return request, func_name


def _capture_job(**overrides):
    session = MagicMock()
    kwargs = dict(
        job_name="job",
        role_arn="arn:aws:iam::123456789012:role/r",
        job_category="AgentRFT",
        schema_version="1.0",
        job_config={"A": "x"},
    )
    kwargs.update(overrides)
    capture_create_job_request(session, **kwargs)
    return _captured(session)


def test_create_job_envelope_is_serialized():
    request, func_name = _capture_job(tags=[Tag(key="k", value="v")])
    assert func_name == "create_job"
    assert request["JobName"] == "job"
    assert request["JobCategory"] == "AgentRFT"
    assert request["Tags"] == [{"Key": "k", "Value": "v"}]
    assert "CustomerDetails" not in request


def test_create_job_document_keeps_none_like_the_direct_path():
    """``serialize`` drops ``None``; ``json.dumps`` on the direct path keeps ``null``."""
    config = {"A": "x", "Nested": {"B": None}, "C": None}
    request, _ = _capture_job(job_config=config)
    assert request["JobConfigDocument"] == config
    assert json.loads(json.dumps(request["JobConfigDocument"])) == json.loads(json.dumps(config))


def test_create_job_document_is_the_caller_dict_with_variables_intact():
    path = ParameterString(name="OutputPath")
    config = {"OutputDataConfig": {"S3OutputPath": path}}
    request, _ = _capture_job(job_config=config)
    assert request["JobConfigDocument"] is config
    assert request["JobConfigDocument"]["OutputDataConfig"]["S3OutputPath"] is path


def test_create_job_customer_details_are_carried():
    request, _ = _capture_job(customer_details={"Env": "e"})
    assert request["CustomerDetails"] == {"Env": "e"}


def test_training_request_drops_client_members_and_job_name():
    session = MagicMock()
    capture_training_request(
        session,
        {
            "training_job_name": "n",
            "session": object(),
            "region": "us-west-2",
            "role_arn": "r",
            "tags": [{"key": "a", "value": "1"}, {"Key": "b", "Value": "2"}],
        },
    )
    request, func_name = _captured(session)
    assert func_name == "train"
    assert request == {
        "RoleArn": "r",
        "Tags": [{"Key": "a", "Value": "1"}, {"Key": "b", "Value": "2"}],
    }
