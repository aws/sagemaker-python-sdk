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
"""Integration tests: DataPreparer.attach() — re-attach to running/completed jobs.

Tests the F16/F51 fix: customers can reconnect to a long-running job after a
kernel restart using only the job ARN.
"""

from __future__ import absolute_import

from urllib.parse import urlparse

import boto3
import pytest

from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.client import (
    _sdg_output_field,
)

# Submits real jobs and waits for them: scheduled CI only, not PR checks
# (see the gpu_intensive marker in sagemaker-train/pyproject.toml).
pytestmark = pytest.mark.gpu_intensive


def _count_jsonl_lines(s3_uri: str, region: str) -> int:
    """Count non-blank lines in a JSONL file at *s3_uri*."""
    parsed = urlparse(s3_uri)
    s3 = boto3.client("s3", region_name=region)
    body = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
    return sum(1 for line in body.splitlines() if line.strip())


@pytest.mark.integ
class TestAttachSDG:
    """Submit an SDG job with wait=False, then re-attach and wait for results."""

    def test_attach_to_inflight_sdg_job(self, sagemaker_session, test_role, s3_fixtures):
        """Submit SDG fire-and-forget, then attach by ARN and wait for completion.

        Validates:
        1. wait=False returns a Job with a public job_arn
        2. DataPreparer.attach(job_arn) re-connects and waits
        3. The returned DataSet has a public job_arn matching the original
        4. The returned DataSet has a valid ARN (rehydrated from Hub)
        """
        dataset = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="integ-attach-sdg",
            generation_config={
                "DatasetSpecification": {"SpecS3Uri": s3_fixtures["task_description.txt"]},
                "NumSamples": 5,
            },
            output_data_config={
                _sdg_output_field(
                    sagemaker_session
                ): f"s3://{sagemaker_session.default_bucket()}/data-preparer-integ-attach/",
            },
        )

        # Step 1: Fire-and-forget — simulates the first cell before kernel restart
        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=test_role,
        )
        job = preparer.generate(dataset, wait=False)

        # Validate F51: job_arn is publicly accessible
        assert job.job_arn is not None
        assert "SyntheticDataGeneration" in job.job_arn
        assert job.job_name is not None

        saved_arn = job.job_arn

        # Step 2: Re-attach — simulates the second cell after kernel restart
        # (We're in the same process, but the DataPreparer.attach path is
        # identical to what a fresh process would do — it only uses the ARN.)
        result = DataPreparer.attach(
            saved_arn,
            sagemaker_session=sagemaker_session,
        )

        # Step 3: Validate the result
        assert result is not None
        assert result.arn is not None  # DataSet ARN from Hub
        assert result.job_arn == saved_arn  # F51: public job_arn preserved

    def test_attach_to_already_completed_job(self, sagemaker_session, test_role, s3_fixtures):
        """Submit SDG with wait=True (blocking), then attach to the completed job.

        Validates that attach() on an already-completed job returns immediately
        with the same DataSet — no redundant wait.
        """
        dataset = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="integ-attach-completed",
            generation_config={
                "DatasetSpecification": {"SpecS3Uri": s3_fixtures["task_description.txt"]},
                "NumSamples": 5,
            },
            output_data_config={
                _sdg_output_field(
                    sagemaker_session
                ): f"s3://{sagemaker_session.default_bucket()}/data-preparer-integ-attach-done/",
            },
        )

        # Step 1: Submit and wait to completion
        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=test_role,
        )
        original_result = preparer.generate(dataset)
        assert original_result is not None
        saved_arn = original_result.job_arn
        assert saved_arn is not None

        # Step 2: Re-attach to the already-completed job
        reattached_result = DataPreparer.attach(
            saved_arn,
            sagemaker_session=sagemaker_session,
        )

        # Should return immediately with the same dataset
        assert reattached_result is not None
        assert reattached_result.arn == original_result.arn
        assert reattached_result.job_arn == saved_arn

        # B2: the re-attached dataset exposes the same record count as the
        # original.  Both DataSets share the same S3 source, so counts should
        # be identical; counting them independently exercises that the source
        # attribute is correctly populated in both code paths.
        region = sagemaker_session.boto_region_name
        orig_jsonl = original_result.source.rstrip("/") + "/output_samples.jsonl"
        reattach_jsonl = reattached_result.source.rstrip("/") + "/output_samples.jsonl"
        orig_count = _count_jsonl_lines(orig_jsonl, region)
        reattach_count = _count_jsonl_lines(reattach_jsonl, region)
        assert (
            orig_count >= 1
        ), f"B2: expected >= 1 record in SDG output at {orig_jsonl!r}, got {orig_count}"
        assert reattach_count == orig_count, (
            f"B2: re-attached record count {reattach_count} != "
            f"original {orig_count} at {reattach_jsonl!r}"
        )


@pytest.mark.integ
class TestAttachDQ:
    """Attach to a DQ job to ensure it works across primitives."""

    _sdg_result = None

    def test_01_setup_generate_dataset(self, sagemaker_session, test_role, s3_fixtures):
        """Generate a small dataset to evaluate."""
        dataset = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="integ-attach-dq-source",
            generation_config={
                "DatasetSpecification": {"SpecS3Uri": s3_fixtures["task_description.txt"]},
                "NumSamples": 5,
            },
            output_data_config={
                "S3OutputPath": f"s3://{sagemaker_session.default_bucket()}/data-preparer-integ-attach-dq-src/",
            },
        )
        preparer = DataPreparer(sagemaker_session=sagemaker_session, role=test_role)
        result = preparer.generate(dataset)
        assert result is not None
        TestAttachDQ._sdg_result = result

    def test_02_attach_to_inflight_dq_job(self, sagemaker_session, test_role):
        """Submit DQ with wait=False, then attach and wait."""
        if TestAttachDQ._sdg_result is None:
            pytest.skip("No source dataset from setup")

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_QUALITY_EVALUATION,
            input_data_config=[{"DatasetArn": TestAttachDQ._sdg_result.arn}],
            evaluation_config={"Operations": [{"Operator": "default_metrics"}]},
            output_data_config={
                "S3OutputPath": f"s3://{sagemaker_session.default_bucket()}/data-preparer-integ-attach-dq/",
            },
        )
        preparer = DataPreparer(sagemaker_session=sagemaker_session, role=test_role)
        job = preparer.evaluate(dataset, wait=False)

        assert job.job_arn is not None
        assert "DataQualityEvaluation" in job.job_arn

        # Re-attach
        result = DataPreparer.attach(
            job.job_arn,
            sagemaker_session=sagemaker_session,
        )

        assert result is not None
        assert result.arn is not None
        assert result.job_arn == job.job_arn
