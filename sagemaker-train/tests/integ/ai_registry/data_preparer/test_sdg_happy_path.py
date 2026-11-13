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
"""Integration tests: SDG happy path then chain into DQ."""

from __future__ import absolute_import

import json
import re
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


@pytest.mark.integ
class TestSDGThenDQ:
    """SDG generates a dataset in HF_Prompt_Completion format, then DQ evaluates it."""

    _sdg_result = None

    def test_01_sdg_generates_dataset(self, sagemaker_session, test_role, s3_fixtures):
        """Submit SDG job and assert Completed with a valid DataSet."""
        dataset = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="integ-sdg-for-dq",
            generation_config={
                "DatasetSpecification": {"SpecS3Uri": s3_fixtures["task_description.txt"]},
                "NumSamples": 5,
            },
            output_data_config={
                _sdg_output_field(sagemaker_session): (
                    f"s3://{sagemaker_session.default_bucket()}/data-preparer-integ-output/"
                ),
            },
        )
        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=test_role,
        )
        result = preparer.generate(dataset)

        assert result is not None
        assert result.arn is not None
        TestSDGThenDQ._sdg_result = result

        # B1: SDG output JSONL has content — at least one non-blank line, every
        # non-blank line is valid JSON.  The SDK writes the output to
        # {source}/output_samples.jsonl (same derivation as _preview_output).
        output_jsonl = result.source.rstrip("/") + "/output_samples.jsonl"
        parsed = urlparse(output_jsonl)
        s3 = boto3.client("s3", region_name=sagemaker_session.boto_region_name)
        body = (
            s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
            .read()
            .decode("utf-8")
        )
        non_blank = [ln for ln in body.splitlines() if ln.strip()]
        assert (
            len(non_blank) >= 1
        ), f"B1: SDG output JSONL has no non-blank lines at {output_jsonl!r}"
        for ln in non_blank:
            json.loads(ln)  # raises ValueError if the line is not valid JSON

    def test_02_dq_evaluates_sdg_output(self, sagemaker_session, test_role, capsys):
        """Evaluate the SDG-generated dataset with DQ.

        Also covers A8: the SDK previews the DQ job's scored records after
        completion. For v2+ DQ jobs the scored records live in the job's own
        folder under ``OutputDataConfig.S3OutputPath``, in ``record_level/``;
        before the A8 fix the preview looked next to the input dataset instead
        and was always silently empty.
        """
        if TestSDGThenDQ._sdg_result is None:
            pytest.skip("SDG test did not produce a dataset")

        dataset = DatasetJobConfig(
            job_category=JobCategory.DATA_QUALITY_EVALUATION,
            input_data_config=[{"DatasetArn": TestSDGThenDQ._sdg_result.arn}],
            evaluation_config={"Operations": [{"Operator": "default_metrics"}]},
            output_data_config={
                "S3OutputPath": (
                    f"s3://{sagemaker_session.default_bucket()}" "/data-preparer-integ-dq-output/"
                ),
            },
        )
        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=test_role,
        )
        result = preparer.evaluate(dataset)

        assert result is not None
        assert result.arn is not None

        # A8: _preview_output calls print() for both the v1 fallback path and
        # the v2 S3-listing path; both paths resolve to
        # record_level/evaluated_data.jsonl.  capsys captures that print().
        captured = capsys.readouterr()
        assert "Preview (" in captured.out, (
            "A8: DQ _preview_output emitted no 'Preview (...)' line; "
            f"full stdout: {captured.out!r}"
        )
        m = re.search(r"Preview \((\d+) records", captured.out)
        assert (
            m is not None and int(m.group(1)) > 0
        ), f"A8: DQ preview record count is 0 or missing; stdout: {captured.out!r}"
        assert "record_level/evaluated_data.jsonl" in captured.out, (
            "A8: DQ preview path does not contain 'record_level/evaluated_data.jsonl'; "
            f"stdout: {captured.out!r}"
        )
