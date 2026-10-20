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
"""Integration tests for AthenaStep.

Runs Athena queries end-to-end through SageMaker Pipelines and checks that:

* a valid query drives the execution to ``Succeeded`` and the result CSV is
  written to the configured S3 output location,
* an invalid query drives the execution to ``Failed`` with a ``FailureReason``,
* a full-shape ``arguments`` dict, together with ``CacheConfig`` and
  ``RetryPolicies``, is accepted by ``CreatePipeline`` and round-trips
  through ``DescribePipeline`` unchanged.

Requirements on top of the shared workflow integ fixtures:

* The pipeline execution role (``role`` fixture) needs
  ``athena:StartQueryExecution``, ``athena:GetQueryExecution`` and
  ``athena:StopQueryExecution``, plus ``s3:GetBucketLocation`` and
  ``s3:PutObject`` on the session default bucket, where Athena writes the
  query results (under ``<default_bucket_prefix>/<pipeline name>/``).
* No Athena workgroup, database or table setup: the queries run in the
  account's default ``primary`` workgroup and do not read any table.
* The Athena step type must be available to the account and Region;
  otherwise ``CreatePipeline`` rejects the pipeline definition.
"""

from __future__ import absolute_import

import json

from sagemaker.core.common_utils import unique_name_from_base
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.mlops.workflow.athena_step import AthenaStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.retry import StepExceptionTypeEnum, StepRetryPolicy
from sagemaker.mlops.workflow.steps import CacheConfig

from tests.integ.workflow.helpers import wait_pipeline_execution

_OUTPUT_LOCATION_PARAMETER = "OutputLocation"


def _result_location(sagemaker_session, pipeline_name):
    """Return ``(bucket, key_prefix, s3_uri)`` for this pipeline's Athena results."""
    bucket = sagemaker_session.default_bucket()
    key_prefix = "/".join(
        part for part in (sagemaker_session.default_bucket_prefix, pipeline_name) if part
    )
    return bucket, key_prefix, "s3://{}/{}/".format(bucket, key_prefix)


def _create_and_execute(pipeline, role, output_location):
    """Create ``pipeline``, start one execution and wait for it to finish.

    Returns ``(execution, execution_status, step)`` for the pipeline's single step.
    """
    pipeline.create(role_arn=role)
    execution = pipeline.start(parameters={_OUTPUT_LOCATION_PARAMETER: output_location})
    wait_pipeline_execution(execution=execution, delay=20, max_attempts=30)

    status = execution.describe()["PipelineExecutionStatus"]
    steps = execution.list_steps()
    assert len(steps) == 1, "Expected exactly one step, got {}".format(len(steps))
    return execution, status, steps[0]


def _cleanup(pipeline, execution, sagemaker_session, bucket, key_prefix):
    """Best-effort teardown: stop a still-running execution, delete the pipeline
    and delete the Athena result files."""
    if execution is not None:
        try:
            execution.stop()
        except Exception:
            pass
    try:
        pipeline.delete()
    except Exception:
        pass
    if bucket and key_prefix:
        try:
            s3_resource = sagemaker_session.boto_session.resource(
                "s3", region_name=sagemaker_session.boto_region_name
            )
            s3_resource.Bucket(bucket).objects.filter(Prefix=key_prefix).delete()
        except Exception:
            pass


def test_athena_step_query_succeeds_and_writes_result(sagemaker_session_for_pipeline, role):
    """A well-formed ``SELECT 1`` succeeds and its result CSV lands in S3."""
    pipeline_name = unique_name_from_base("athena-step-integ")
    bucket, key_prefix, output_location = _result_location(
        sagemaker_session_for_pipeline, pipeline_name
    )
    output_location_parameter = ParameterString(name=_OUTPUT_LOCATION_PARAMETER)

    step = AthenaStep(
        name="AthenaQuery",
        arguments={
            "QueryString": "SELECT 1 AS test_result",
            "ResultConfiguration": {"OutputLocation": output_location_parameter},
        },
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[output_location_parameter],
        steps=[step],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    execution = None
    try:
        execution, status, athena_step = _create_and_execute(pipeline, role, output_location)
        assert status == "Succeeded", "Execution status was {}".format(status)
        assert athena_step["StepName"] == "AthenaQuery"
        assert athena_step["StepStatus"] == "Succeeded", athena_step.get("FailureReason")

        query_execution_id = athena_step["Metadata"]["Athena"]["QueryExecutionId"]
        result_key = "{}/{}.csv".format(key_prefix, query_execution_id)
        result_files = sagemaker_session_for_pipeline.list_s3_files(bucket, key_prefix)
        assert result_key in result_files, "Athena result file not found: {}".format(result_files)
    finally:
        _cleanup(pipeline, execution, sagemaker_session_for_pipeline, bucket, key_prefix)


def test_athena_step_invalid_query_fails(sagemaker_session_for_pipeline, role):
    """An invalid SQL statement drives the step, and the execution, to ``Failed``."""
    pipeline_name = unique_name_from_base("athena-step-integ")
    bucket, key_prefix, output_location = _result_location(
        sagemaker_session_for_pipeline, pipeline_name
    )
    output_location_parameter = ParameterString(name=_OUTPUT_LOCATION_PARAMETER)

    step = AthenaStep(
        name="AthenaInvalidQuery",
        arguments={
            "QueryString": "SELCT INVALID SYNTAX FROM nonexistent_table_xyz",
            "ResultConfiguration": {"OutputLocation": output_location_parameter},
        },
    )
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[output_location_parameter],
        steps=[step],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    execution = None
    try:
        execution, status, athena_step = _create_and_execute(pipeline, role, output_location)
        assert status == "Failed", "Execution status was {}".format(status)
        assert athena_step["StepStatus"] == "Failed"
        assert athena_step.get("FailureReason"), "Step is Failed but no FailureReason was set"
    finally:
        _cleanup(pipeline, execution, sagemaker_session_for_pipeline, bucket, key_prefix)


def test_athena_step_full_arguments_roundtrip(sagemaker_session_for_pipeline, role):
    """Full-shape ``arguments`` plus ``CacheConfig``/``RetryPolicies`` are accepted at
    ``CreatePipeline`` and come back unchanged from ``DescribePipeline``."""
    pipeline_name = unique_name_from_base("athena-step-integ")
    output_location_parameter = ParameterString(name=_OUTPUT_LOCATION_PARAMETER)

    step = AthenaStep(
        name="AthenaFullQuery",
        arguments={
            "QueryString": "SELECT 1",
            "QueryExecutionContext": {"Database": "default", "Catalog": "AwsDataCatalog"},
            "ResultConfiguration": {
                "OutputLocation": output_location_parameter,
                "EncryptionConfiguration": {"EncryptionOption": "SSE_S3"},
            },
            "WorkGroup": "primary",
        },
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
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[output_location_parameter],
        steps=[step],
        sagemaker_session=sagemaker_session_for_pipeline,
    )

    try:
        pipeline.create(role_arn=role)

        described_definition = json.loads(pipeline.describe()["PipelineDefinition"])
        sent_definition = json.loads(pipeline.definition())
        assert described_definition["Steps"] == sent_definition["Steps"]

        (athena_step,) = described_definition["Steps"]
        assert athena_step["Type"] == "Athena"
        assert athena_step["Arguments"]["WorkGroup"] == "primary"
        assert athena_step["Arguments"]["ResultConfiguration"]["OutputLocation"] == {
            "Get": "Parameters.{}".format(_OUTPUT_LOCATION_PARAMETER)
        }
        assert athena_step["CacheConfig"] == {"Enabled": True, "ExpireAfter": "PT1H"}
        assert athena_step["RetryPolicies"][0]["ExceptionType"] == ["Step.SERVICE_FAULT"]
    finally:
        _cleanup(pipeline, None, sagemaker_session_for_pipeline, None, None)
