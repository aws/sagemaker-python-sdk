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
"""Unit tests for AthenaStep -- pure passthrough of the StartQueryExecution request."""

from __future__ import absolute_import

import json
from unittest.mock import MagicMock

import boto3
import pytest

from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import Properties
from sagemaker.mlops.workflow.athena_step import AthenaStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.retry import StepExceptionTypeEnum, StepRetryPolicy
from sagemaker.mlops.workflow.step_dependency import StepDependency, TriggerCondition
from sagemaker.mlops.workflow.steps import CacheConfig, StepTypeEnum

# A full-shape Athena StartQueryExecution request, as a customer would write it.
FULL_ARGUMENTS = {
    "QueryString": "SELECT * FROM daily_txn WHERE txn_date >= '2026-01-01'",
    "QueryExecutionContext": {"Catalog": "unity-catalog", "Database": "prod_catalog$transactions"},
    "ResultConfiguration": {
        "OutputLocation": "s3://bucket/sagemaker-athena-staging/",
        "EncryptionConfiguration": {"EncryptionOption": "SSE_S3"},
    },
    "WorkGroup": "primary",
    "ResultReuseConfiguration": {
        "ResultReuseByAgeConfiguration": {"Enabled": True, "MaxAgeInMinutes": 60}
    },
    "ExecutionParameters": ["us-east-1", "2026-01-01"],
}


# ---------------------------------------------------------------------------
# Passthrough contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        FULL_ARGUMENTS,
        # The service accepts QueryString alone; everything else is optional.
        {"QueryString": "SELECT 1"},
        # The SDK holds no opinion on required fields -- the service decides.
        {},
    ],
    ids=["full", "minimal", "empty"],
)
def test_arguments_passed_through_unchanged(arguments):
    step = AthenaStep(name="RunQuery", arguments=arguments)
    assert step.arguments == arguments
    assert step.to_request()["Arguments"] == arguments


def test_arguments_is_required():
    with pytest.raises(TypeError):
        AthenaStep(name="RunQuery")  # pylint: disable=no-value-for-parameter


def test_caller_mutations_do_not_leak_into_step_state():
    arguments = {
        "QueryString": "SELECT 1",
        "ResultConfiguration": {"EncryptionConfiguration": {"EncryptionOption": "SSE_S3"}},
        "ExecutionParameters": ["1"],
    }
    step = AthenaStep(name="RunQuery", arguments=arguments)

    arguments["QueryString"] = "MUTATED"
    arguments["ResultConfiguration"]["EncryptionConfiguration"]["EncryptionOption"] = "MUTATED"
    arguments["ExecutionParameters"].append("2")

    assert step.arguments == {
        "QueryString": "SELECT 1",
        "ResultConfiguration": {"EncryptionConfiguration": {"EncryptionOption": "SSE_S3"}},
        "ExecutionParameters": ["1"],
    }


def test_pipeline_variable_inside_arguments_compiles_to_get_expression():
    """A pipeline variable nested anywhere in ``arguments`` serializes to a
    ``{"Get": ...}`` expression in the compiled definition, and the step
    compiles unchanged under a ``PipelineSession``."""
    pipeline_session = PipelineSession(
        boto_session=boto3.Session(region_name="us-west-2"),
        sagemaker_client=MagicMock(name="sagemaker_client"),
        sagemaker_config={},
    )
    output_location = ParameterString(name="OutputLocation")
    step = AthenaStep(
        name="RunQuery",
        arguments={
            "QueryString": "SELECT 1",
            "ResultConfiguration": {"OutputLocation": output_location},
        },
    )
    pipeline = Pipeline(
        name="athena-unit",
        parameters=[output_location],
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    definition = json.loads(pipeline.definition())

    compiled_step = definition["Steps"][0]
    assert compiled_step["Type"] == "Athena"
    assert compiled_step["Arguments"]["ResultConfiguration"]["OutputLocation"] == {
        "Get": "Parameters.OutputLocation"
    }
    # Nothing was intercepted: AthenaStep never calls a Session helper.
    assert pipeline_session.context is None


def test_run_conditions_compile_to_depends_on_triggers():
    query = AthenaStep(name="Query", arguments={"QueryString": "SELECT 1"})
    on_failure = AthenaStep(
        name="OnFailure",
        arguments={"QueryString": "SELECT 1"},
        depends_on=[StepDependency(query, TriggerCondition.FAILED)],
    )
    on_any = AthenaStep(
        name="OnAny",
        arguments={"QueryString": "SELECT 1"},
        depends_on=[StepDependency("Query", TriggerCondition.ANY)],
    )
    on_success = AthenaStep(
        name="OnSuccess", arguments={"QueryString": "SELECT 1"}, depends_on=[query]
    )
    pipeline = Pipeline(
        name="athena-run-conditions",
        steps=[query, on_failure, on_any, on_success],
        sagemaker_session=MagicMock(),
    )

    depends_on = {
        step["Name"]: step.get("DependsOn") for step in json.loads(pipeline.definition())["Steps"]
    }

    assert depends_on == {
        "Query": None,
        "OnFailure": [{"Step": "Query", "Trigger": "Failed"}],
        "OnAny": [{"Step": "Query", "Trigger": "Any"}],
        "OnSuccess": ["Query"],
    }


# ---------------------------------------------------------------------------
# Properties shape
# ---------------------------------------------------------------------------


def test_properties_shape_and_reference_paths():
    step = AthenaStep(name="RunQuery", arguments={"QueryString": "SELECT 1"})

    assert isinstance(step.properties, Properties)
    # Loaded from botocore's GetQueryExecutionOutput shape (service_name="athena");
    # the QueryExecution envelope matches the service-side property paths.
    assert (
        step.properties.QueryExecution.QueryExecutionId.expr["Get"]
        == "Steps.RunQuery.QueryExecution.QueryExecutionId"
    )
    assert (
        step.properties.QueryExecution.Status.State.expr["Get"]
        == "Steps.RunQuery.QueryExecution.Status.State"
    )
    assert (
        step.properties.QueryExecution.ResultConfiguration.OutputLocation.expr["Get"]
        == "Steps.RunQuery.QueryExecution.ResultConfiguration.OutputLocation"
    )


# ---------------------------------------------------------------------------
# to_request()
# ---------------------------------------------------------------------------


def test_to_request_includes_step_fields_cache_config_and_retry_policies():
    step = AthenaStep(
        name="QueryUCData",
        arguments=FULL_ARGUMENTS,
        display_name="Query UC Transactions",
        description="Runs Athena query against Unity Catalog",
        depends_on=["Upstream"],
        cache_config=CacheConfig(enable_caching=True, expire_after="PT1H"),
        retry_policies=[
            StepRetryPolicy(
                exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
                max_attempts=3,
                interval_seconds=30,
                backoff_rate=2.0,
            )
        ],
    )

    assert step.to_request() == {
        "Name": "QueryUCData",
        "Type": "Athena",
        "DisplayName": "Query UC Transactions",
        "Description": "Runs Athena query against Unity Catalog",
        "DependsOn": ["Upstream"],
        "Arguments": FULL_ARGUMENTS,
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
        "RetryPolicies": [
            {
                "ExceptionType": ["Step.SERVICE_FAULT"],
                "MaxAttempts": 3,
                "IntervalSeconds": 30,
                "BackoffRate": 2.0,
            }
        ],
    }


def test_to_request_omits_optional_sections_when_not_set():
    request_dict = AthenaStep(name="RunQuery", arguments={"QueryString": "SELECT 1"}).to_request()
    assert "CacheConfig" not in request_dict
    assert "RetryPolicies" not in request_dict
    assert "DependsOn" not in request_dict


# ---------------------------------------------------------------------------
# Package wiring
# ---------------------------------------------------------------------------


def test_athena_step_exported_from_workflow_package():
    from sagemaker.mlops.workflow import AthenaStep as ExportedAthenaStep

    assert ExportedAthenaStep is AthenaStep


def test_step_type_enum_has_athena_value():
    assert StepTypeEnum.ATHENA.value == "Athena"
