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
"""Unit tests for AthenaStep and AthenaQueryConfig."""

from __future__ import absolute_import

import json
from unittest.mock import MagicMock

import boto3
import pytest

from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import Properties
from sagemaker.mlops.workflow.athena_step import AthenaQueryConfig, AthenaStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.retry import StepExceptionTypeEnum, StepRetryPolicy
from sagemaker.mlops.workflow.step_dependency import StepDependency, TriggerCondition
from sagemaker.mlops.workflow.steps import CacheConfig, StepTypeEnum

FULL_QUERY_CONFIG = AthenaQueryConfig(
    query_string="SELECT * FROM daily_txn WHERE region = ? AND txn_date >= ?",
    database="prod_catalog$transactions",
    catalog="unity-catalog",
    work_group="primary",
    execution_parameters=["'us-east-1'", "'2026-01-01'"],
    output_location="s3://bucket/sagemaker-athena-staging/",
    encryption_option="SSE_KMS",
    kms_key="arn:aws:kms:us-west-2:111122223333:key/example",
    expected_bucket_owner="111122223333",
    s3_acl_option="BUCKET_OWNER_FULL_CONTROL",
    result_reuse_max_age_in_minutes=60,
)

FULL_ARGUMENTS = {
    "QueryString": "SELECT * FROM daily_txn WHERE region = ? AND txn_date >= ?",
    "QueryExecutionContext": {"Database": "prod_catalog$transactions", "Catalog": "unity-catalog"},
    "WorkGroup": "primary",
    "ExecutionParameters": ["'us-east-1'", "'2026-01-01'"],
    "ResultConfiguration": {
        "OutputLocation": "s3://bucket/sagemaker-athena-staging/",
        "EncryptionConfiguration": {
            "EncryptionOption": "SSE_KMS",
            "KmsKey": "arn:aws:kms:us-west-2:111122223333:key/example",
        },
        "ExpectedBucketOwner": "111122223333",
        "AclConfiguration": {"S3AclOption": "BUCKET_OWNER_FULL_CONTROL"},
    },
    "ResultReuseConfiguration": {
        "ResultReuseByAgeConfiguration": {"Enabled": True, "MaxAgeInMinutes": 60}
    },
}


# ---------------------------------------------------------------------------
# AthenaQueryConfig
# ---------------------------------------------------------------------------


def test_query_config_builds_full_request():
    assert FULL_QUERY_CONFIG.to_request() == FULL_ARGUMENTS


def test_query_config_minimal_emits_only_query_string():
    assert AthenaQueryConfig(query_string="SELECT 1").to_request() == {"QueryString": "SELECT 1"}


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"database": "db"}, {"QueryExecutionContext": {"Database": "db"}}),
        (
            {"output_location": "s3://b/p/"},
            {"ResultConfiguration": {"OutputLocation": "s3://b/p/"}},
        ),
        (
            {"encryption_option": "SSE_S3"},
            {"ResultConfiguration": {"EncryptionConfiguration": {"EncryptionOption": "SSE_S3"}}},
        ),
        (
            {"s3_acl_option": "BUCKET_OWNER_FULL_CONTROL"},
            {
                "ResultConfiguration": {
                    "AclConfiguration": {"S3AclOption": "BUCKET_OWNER_FULL_CONTROL"}
                }
            },
        ),
    ],
    ids=["context", "output-location", "encryption", "acl"],
)
def test_query_config_nests_only_the_fields_that_are_set(kwargs, expected):
    request = AthenaQueryConfig(query_string="SELECT 1", **kwargs).to_request()
    assert request == {"QueryString": "SELECT 1", **expected}


def test_step_with_query_config_emits_built_request():
    step = AthenaStep(name="RunQuery", query_config=FULL_QUERY_CONFIG)
    assert step.arguments == FULL_ARGUMENTS
    assert step.to_request()["Arguments"] == FULL_ARGUMENTS


# ---------------------------------------------------------------------------
# Raw arguments escape hatch
# ---------------------------------------------------------------------------


def test_raw_arguments_passed_through_unchanged():
    arguments = {
        "QueryString": "SELECT 1",
        "EngineConfiguration": {"MaxConcurrentDpus": 4},
    }
    step = AthenaStep(name="RunQuery", arguments=arguments)
    assert step.arguments == arguments


def test_requires_query_config_or_arguments():
    with pytest.raises(ValueError, match="must have either query_config or arguments"):
        AthenaStep(name="RunQuery")


def test_rejects_both_query_config_and_arguments():
    with pytest.raises(ValueError, match="cannot have both query_config and arguments"):
        AthenaStep(
            name="RunQuery",
            query_config=AthenaQueryConfig(query_string="SELECT 1"),
            arguments={"QueryString": "SELECT 1"},
        )


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


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def test_pipeline_variable_in_query_config_compiles_to_get_expression():
    """A pipeline variable in a config field serializes to a ``{"Get": ...}``
    expression, and the step compiles unchanged under a ``PipelineSession``."""
    pipeline_session = PipelineSession(
        boto_session=boto3.Session(region_name="us-west-2"),
        sagemaker_client=MagicMock(name="sagemaker_client"),
        sagemaker_config={},
    )
    output_location = ParameterString(name="OutputLocation")
    step = AthenaStep(
        name="RunQuery",
        query_config=AthenaQueryConfig(query_string="SELECT 1", output_location=output_location),
    )
    pipeline = Pipeline(
        name="athena-unit",
        parameters=[output_location],
        steps=[step],
        sagemaker_session=pipeline_session,
    )

    compiled_step = json.loads(pipeline.definition())["Steps"][0]

    assert compiled_step["Type"] == "Athena"
    assert compiled_step["Arguments"]["ResultConfiguration"]["OutputLocation"] == {
        "Get": "Parameters.OutputLocation"
    }
    # Nothing was intercepted: AthenaStep never calls a Session helper.
    assert pipeline_session.context is None


def test_run_conditions_compile_to_depends_on_triggers():
    config = AthenaQueryConfig(query_string="SELECT 1")
    query = AthenaStep(name="Query", query_config=config)
    on_failure = AthenaStep(
        name="OnFailure",
        query_config=config,
        depends_on=[StepDependency(query, TriggerCondition.FAILED)],
    )
    on_any = AthenaStep(
        name="OnAny",
        query_config=config,
        depends_on=[StepDependency("Query", TriggerCondition.ANY)],
    )
    on_success = AthenaStep(name="OnSuccess", query_config=config, depends_on=[query])
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
    step = AthenaStep(name="RunQuery", query_config=AthenaQueryConfig(query_string="SELECT 1"))

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
        query_config=FULL_QUERY_CONFIG,
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
    request_dict = AthenaStep(
        name="RunQuery", query_config=AthenaQueryConfig(query_string="SELECT 1")
    ).to_request()
    assert "CacheConfig" not in request_dict
    assert "RetryPolicies" not in request_dict
    assert "DependsOn" not in request_dict


# ---------------------------------------------------------------------------
# Package wiring
# ---------------------------------------------------------------------------


def test_athena_classes_exported_from_workflow_package():
    from sagemaker.mlops.workflow import AthenaQueryConfig as ExportedConfig
    from sagemaker.mlops.workflow import AthenaStep as ExportedAthenaStep

    assert ExportedAthenaStep is AthenaStep
    assert ExportedConfig is AthenaQueryConfig


def test_step_type_enum_has_athena_value():
    assert StepTypeEnum.ATHENA.value == "Athena"
