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
"""Integration tests: DQ 3.0.0 exploratory_analysis (free-form) operator.

exploratory_analysis is a v3 reserved Operator that runs an agent analysis of
the dataset and writes a findings document rather than scoring individual
records.  The rest of the suite covers only default_metrics; this test covers
the free-form operator whose aggregate-only output contract is invisible to
default_metrics tests.

Note: today there is no record_level/ output for this operator; that is
documented in the assertions below rather than asserted, in case the service
adds per-record output in a future release.
"""

from __future__ import absolute_import

import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pytest

from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory

# Submits a real job and waits for it: scheduled CI only, not PR checks
# (see the gpu_intensive marker in sagemaker-train/pyproject.toml).
pytestmark = pytest.mark.gpu_intensive

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def freeform_corpus_uri(sagemaker_session, integ_bucket):
    """One JSONL alone under its own S3 prefix, uploaded and removed per test.

    Deliberately not the shared ``s3_fixtures`` prefix, which holds every
    fixture file -- .py, .md, .json, .txt and several .jsonl -- in one
    directory. That is hygiene for an operator that analyses a corpus, not a
    fix for the intermittent failure described on the test class: the failure
    reproduces from a single-file prefix too.
    """
    s3 = boto3.client("s3", region_name=sagemaker_session.boto_region_name)
    key = f"data-preparer-integ-freeform-{uuid.uuid4().hex[:8]}/byod_training_sft.jsonl"
    s3.upload_file(str(FIXTURES_DIR / "byod_training_sft.jsonl"), integ_bucket, key)
    yield f"s3://{integ_bucket}/{key}"
    try:
        s3.delete_object(Bucket=integ_bucket, Key=key)
    except Exception:
        pass


@pytest.mark.integ
class TestDQFreeformExploratory:
    """DQ exploratory_analysis operator — aggregate-only free-form output.

    UNRESOLVED INTERMITTENT, 2026-11-12 us-west-2. This job fails roughly half
    the time with "ClientError: Customer resource access denied; check provided
    role has access to customer resources passed to job input", on a role and
    dataset that are provably fine -- the same role and account complete
    default_metrics DQ, SDG and attach tests in the same session, and the same
    operator completed six times with correct output.

    Ruled out by measurement, so do not re-test these:
      - DocumentSchemaVersion: fails and passes at both 4.0.0 and 5.0.0.
      - Missing DatasetType: completes with and without ``DatasetType: BYOD``.
      - Readiness race after import: HubContentStatus is Available at t+0 with
        ValidationManifest JobStatus SKIPPED.
      - SDK vs raw CreateJob: both paths complete, and both have failed.
      - Shared vs dedicated S3 prefix: fails from a single-file prefix too.

    Until the cause is found this test is not fit to gate a pipeline. Do NOT
    make it pass by skipping on that message -- that hides a real regression in
    an operator we would then be shipping untested.
    """

    def test_dq_freeform_exploratory_analysis(
        self, sagemaker_session, test_role, freeform_corpus_uri, import_dataset, integ_hub_name
    ):
        """Submit a DQ job with the exploratory_analysis operator and assert the output contract.

        Asserts:
        1. Terminal JobStatus == Completed.
        2. JobConfigSchemaVersion has major version 3 (exploratory_analysis is v3-only).
        3. aggregate/quality_report.json parses and "exploratory_analysis" is a top-level key.
        4. aggregate/exploratory/findings.json parses as a non-empty list whose every element
           carries non-empty string "title" and "evidence".

        Does NOT assert:
        - findings text or count (agent-authored, varies run to run).
        - absence of record_level/ (today's behaviour; asserting a negative would
          break the test if the service adds per-record output).
        """
        # DocumentSchemaVersion 5.0.0 is verified to work for this operator.
        dataset_arn = import_dataset(
            freeform_corpus_uri,
            file_format="HF_Prompt_Completion",
            document_schema_version="5.0.0",
        )
        # Extract name and version for cleanup.
        # ARN: arn:aws:sagemaker:{region}:{account}:hub-content/{hub}/{type}/{name}/{version}
        arn_parts = dataset_arn.split("/")
        hub_content_name = arn_parts[3]
        hub_content_version = arn_parts[4]
        passed = False

        try:
            dataset_config = DatasetJobConfig(
                job_category=JobCategory.DATA_QUALITY_EVALUATION,
                input_data_config=[{"DatasetArn": dataset_arn}],
                evaluation_config={"Operations": [{"Operator": "exploratory_analysis"}]},
                output_data_config={
                    "S3OutputPath": (
                        f"s3://{sagemaker_session.default_bucket()}"
                        "/data-preparer-integ-dq-freeform/"
                    ),
                },
            )
            preparer = DataPreparer(
                sagemaker_session=sagemaker_session,
                role=test_role,
            )

            job = preparer.evaluate(dataset_config, wait=False)
            job_name = job.job_arn.rsplit("/", 1)[-1]
            job.wait()  # returns None for DQ

            desc = sagemaker_session.sagemaker_client.describe_job(
                JobName=job_name, JobCategory="DataQualityEvaluation"
            )

            # 1. Terminal status.
            assert desc["JobStatus"] == "Completed", (
                f"Job {job_name} ended with status {desc['JobStatus']!r}; "
                f"FailureReason: {desc.get('FailureReason')!r}"
            )

            # 2. Schema major version == 3.
            schema_version = desc["JobConfigSchemaVersion"]
            assert schema_version.split(".")[0] == "3", (
                f"Expected schema major version 3, got {schema_version!r}; "
                "exploratory_analysis requires DQ schema v3"
            )

            # Derive the job output folder from MetadataS3Uri.
            job_config_doc = json.loads(desc["JobConfigDocument"])
            metadata_s3_uri = job_config_doc["MetadataS3Uri"]
            output_folder = metadata_s3_uri.rsplit("/metadata.json", 1)[0]

            s3 = boto3.client("s3", region_name=sagemaker_session.boto_region_name)

            def _read_s3_json(s3_uri):
                parsed = urlparse(s3_uri)
                body = (
                    s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
                    .read()
                    .decode("utf-8")
                )
                return json.loads(body)

            # 3. quality_report.json has "exploratory_analysis" as a top-level key.
            quality_report = _read_s3_json(f"{output_folder}/aggregate/quality_report.json")
            assert "exploratory_analysis" in quality_report, (
                f"'exploratory_analysis' not found in quality_report.json top-level keys: "
                f"{list(quality_report.keys())}"
            )

            # 4. findings.json is a non-empty list of dicts with "title" and "evidence".
            findings = _read_s3_json(f"{output_folder}/aggregate/exploratory/findings.json")
            assert isinstance(
                findings, list
            ), f"aggregate/exploratory/findings.json should be a list, got {type(findings).__name__}"
            assert len(findings) > 0, "aggregate/exploratory/findings.json is an empty list"
            for i, finding in enumerate(findings):
                assert isinstance(
                    finding, dict
                ), f"findings[{i}] is not a dict: {type(finding).__name__}"
                assert (
                    isinstance(finding.get("title"), str) and finding["title"]
                ), f"findings[{i}] missing non-empty string 'title': {finding!r}"
                assert (
                    isinstance(finding.get("evidence"), str) and finding["evidence"]
                ), f"findings[{i}] missing non-empty string 'evidence': {finding!r}"

            passed = True

        finally:
            # Keep the dataset on failure: deleting it destroys the only
            # evidence needed to diagnose the transient described above.
            if passed:
                try:
                    sagemaker_session.sagemaker_client.delete_hub_content(
                        HubName=integ_hub_name,
                        HubContentType="DataSet",
                        HubContentName=hub_content_name,
                        HubContentVersion=hub_content_version,
                    )
                except Exception:
                    pass
            else:
                print(f"test failed; leaving {dataset_arn} in place for diagnosis")
