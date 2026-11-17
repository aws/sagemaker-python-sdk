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

import json
from unittest.mock import Mock

import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession, _JobStepArguments
from sagemaker.mlops.workflow.hub_content_step import HubContentStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.retry import StepExceptionTypeEnum, StepRetryPolicy
from sagemaker.mlops.workflow.steps import CacheConfig, StepTypeEnum

DOCUMENT = {"DatasetS3Prefix": "s3://bucket/data.jsonl", "DatasetFormat": "jsonl"}


@pytest.fixture
def pipeline_session():
    return PipelineSession(boto_session=Mock(region_name="us-west-2"), sagemaker_client=Mock())


@pytest.fixture
def step_args(pipeline_session):
    return pipeline_session.import_hub_content(
        hub_name="my-hub",
        hub_content_name="my-dataset",
        hub_content_type="DataSet",
        document_schema_version="2.0.0",
        hub_content_document=DOCUMENT,
        hub_content_display_name="My dataset",
        hub_content_search_keywords=["pipelines"],
        tags=[{"Key": "team", "Value": "ml"}],
    )


def test_capture_does_not_call_service(pipeline_session, step_args):
    assert isinstance(step_args, _JobStepArguments)
    assert not pipeline_session.sagemaker_client.import_hub_content.called


def test_captured_request(step_args):
    assert step_args.args == {
        "HubName": "my-hub",
        "HubContentName": "my-dataset",
        "HubContentType": "DataSet",
        "DocumentSchemaVersion": "2.0.0",
        "HubContentDocument": DOCUMENT,
        "HubContentDisplayName": "My dataset",
        "HubContentSearchKeywords": ["pipelines"],
        "Tags": [{"Key": "team", "Value": "ml"}],
    }


def test_unset_optional_fields_are_omitted(pipeline_session):
    args = pipeline_session.import_hub_content(
        hub_name="h",
        hub_content_name="n",
        hub_content_type="Model",
        document_schema_version="2.4.0",
        hub_content_document="{}",
    ).args
    assert set(args) == {
        "HubName",
        "HubContentName",
        "HubContentType",
        "DocumentSchemaVersion",
        "HubContentDocument",
    }
    assert args["HubContentDocument"] == "{}"


def test_step_encodes_dict_document(step_args):
    step = HubContentStep(name="Import", step_args=step_args)
    assert step.arguments["HubContentDocument"] == json.dumps(DOCUMENT)
    # The captured request is left untouched, so repeated reads stay stable.
    assert step_args.args["HubContentDocument"] == DOCUMENT
    assert step.arguments["HubContentDocument"] == json.dumps(DOCUMENT)


def test_step_passes_string_document_through(pipeline_session):
    step_args = pipeline_session.import_hub_content(
        hub_name="h",
        hub_content_name="n",
        hub_content_type="Model",
        document_schema_version="2.4.0",
        hub_content_document='{"Url": "u"}',
    )
    step = HubContentStep(name="Import", step_args=step_args)
    assert step.arguments["HubContentDocument"] == '{"Url": "u"}'


def test_document_with_pipeline_variable_becomes_join(pipeline_session):
    prefix = ParameterString(name="Prefix")
    step_args = pipeline_session.import_hub_content(
        hub_name="h",
        hub_content_name="n",
        hub_content_type="DataSet",
        document_schema_version="2.0.0",
        hub_content_document={"DatasetS3Prefix": prefix},
    )
    args = HubContentStep(name="Import", step_args=step_args).arguments
    assert isinstance(args["HubContentDocument"], Join)
    assert args["HubContentDocument"].expr == {
        "Std:Join": {
            "On": "",
            "Values": ['{"DatasetS3Prefix": "', {"Get": "Parameters.Prefix"}, '"}'],
        }
    }


def test_step_to_request(step_args):
    step = HubContentStep(
        name="Import",
        step_args=step_args,
        cache_config=CacheConfig(enable_caching=True, expire_after="P30D"),
        retry_policies=[
            StepRetryPolicy(exception_types=[StepExceptionTypeEnum.THROTTLING], max_attempts=3)
        ],
    )
    assert step.step_type == StepTypeEnum.HUB_CONTENT
    request = step.to_request()
    assert request["Type"] == "HubContent"
    assert request["Name"] == "Import"
    assert request["Arguments"]["HubContentName"] == "my-dataset"
    assert request["CacheConfig"]["Enabled"] is True
    assert request["RetryPolicies"][0]["MaxAttempts"] == 3


def test_step_rejects_wrong_producer(pipeline_session):
    endpoint_args = pipeline_session.create_endpoint(endpoint_name="ep", config_name="cfg")
    with pytest.raises(ValueError, match="import_hub_content"):
        HubContentStep(name="Import", step_args=endpoint_args)


def test_step_rejects_raw_dict():
    with pytest.raises(TypeError):
        HubContentStep(name="Import", step_args={"HubName": "h"})


def test_step_properties(step_args):
    step = HubContentStep(name="Import", step_args=step_args)
    assert step.properties.HubContentArn.expr == {"Get": "Steps.Import.HubContentArn"}
    assert step.properties.HubContentStatus.expr == {"Get": "Steps.Import.HubContentStatus"}


def test_pipeline_definition(pipeline_session, step_args):
    step = HubContentStep(name="Import", step_args=step_args)
    pipeline = Pipeline(name="p", steps=[step], sagemaker_session=pipeline_session)
    steps = json.loads(pipeline.definition())["Steps"]
    assert steps[0]["Type"] == "HubContent"
    assert steps[0]["Arguments"]["HubContentDocument"] == json.dumps(DOCUMENT)


def test_plain_session_calls_service():
    session = Session(boto_session=Mock(region_name="us-west-2"), sagemaker_client=Mock())
    session.sagemaker_client.import_hub_content.return_value = {"HubContentArn": "arn:x"}
    response = session.import_hub_content(
        hub_name="h",
        hub_content_name="n",
        hub_content_type="Model",
        document_schema_version="2.4.0",
        hub_content_document="{}",
    )
    assert response == {"HubContentArn": "arn:x"}
    session.sagemaker_client.import_hub_content.assert_called_once_with(
        HubName="h",
        HubContentName="n",
        HubContentType="Model",
        DocumentSchemaVersion="2.4.0",
        HubContentDocument="{}",
    )
