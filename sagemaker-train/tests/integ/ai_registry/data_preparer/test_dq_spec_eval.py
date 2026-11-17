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
"""Integration tests: DQ 3.0.0 spec_eval operator — customer-supplied requirements document.

spec_eval is the third reserved Operator added in DQ schema v3.  Unlike
default_metrics (canned checks) and exploratory_analysis (agent free-form
analysis), spec_eval accepts a customer-supplied markdown document
(``Config.SpecS3Uri``), extracts requirements verbatim from it, generates
judge code, and produces per-record verdicts.  That untrusted-input path has
no other coverage in the suite, making this the highest-value integration test
for the operator.

The test submits a real job against the ``byod_training_sft.jsonl`` sentiment
fixture (5 records) with the ``dq_spec_requirements.md`` fixture as the
requirements document, then asserts the full output contract: terminal status,
schema version, quality report shape, title round-trip from the supplied
document, criteria extraction with source-in-fixture verification, and
per-record verdicts.
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
def spec_eval_inputs(sagemaker_session, integ_bucket):
    """Upload corpus JSONL and requirements document to a per-test S3 prefix.

    Both files land under the same prefix so cleanup is a single iteration.
    The corpus sub-key is kept as a single file (no other fixtures mixed in)
    to match the isolation pattern established by test_dq_freeform.py.
    """
    s3 = boto3.client("s3", region_name=sagemaker_session.boto_region_name)
    prefix = f"data-preparer-integ-spec-eval-{uuid.uuid4().hex[:8]}"
    corpus_key = f"{prefix}/byod_training_sft.jsonl"
    spec_key = f"{prefix}/dq_spec_requirements.md"

    s3.upload_file(str(FIXTURES_DIR / "byod_training_sft.jsonl"), integ_bucket, corpus_key)
    s3.upload_file(str(FIXTURES_DIR / "dq_spec_requirements.md"), integ_bucket, spec_key)

    yield {
        "corpus_uri": f"s3://{integ_bucket}/{corpus_key}",
        "spec_uri": f"s3://{integ_bucket}/{spec_key}",
    }

    for key in [corpus_key, spec_key]:
        try:
            s3.delete_object(Bucket=integ_bucket, Key=key)
        except Exception:
            pass


@pytest.mark.integ
class TestDQSpecEval:
    """DQ spec_eval operator — customer requirements document drives judge-code generation.

    The key assertion unique to this test is the title round-trip
    (``result["title"] == fixture H1``) which proves the supplied document was
    actually read rather than defaulted, and the source-in-fixture check which
    pins the extraction end to end through the untrusted-input path.
    """

    def test_dq_spec_eval_requirements_document(
        self,
        sagemaker_session,
        test_role,
        spec_eval_inputs,
        import_dataset,
        integ_hub_name,
    ):
        """Submit a DQ job with the spec_eval operator and assert the full output contract.

        Asserts:
         1. Terminal JobStatus == Completed.
         2. JobConfigSchemaVersion has major version 3 (spec_eval is v3-only).
         3. aggregate/quality_report.json parses and "spec_eval" is a top-level key.
         4. quality_report["spec_eval"] is a non-empty list; first entry has a "result" dict.
         5. result["title"] equals the H1 of the fixture document (proves the doc was read).
         6. result["grading_criteria_declared"] > 0.
         7. Every row got a verdict: rows_judged == rows_total and rows_missing_verdicts == 0.
         8. spec_eval/extracted_criteria.json is a non-empty list; every element has
            non-empty string "id", "description", and "source".
         9. Each criterion's "source" text appears in the fixture document, normalising
            whitespace (pins the untrusted-input extraction path end to end).
        10. record_level/evaluated_data.jsonl is non-empty; every line carries dq_spec_eval_pass.
        """
        # DocumentSchemaVersion 5.0.0 is verified to work for this operator.
        dataset_arn = import_dataset(
            spec_eval_inputs["corpus_uri"],
            file_format="HF_Prompt_Completion",
            document_schema_version="5.0.0",
        )
        # ARN: arn:aws:sagemaker:{region}:{account}:hub-content/{hub}/{type}/{name}/{version}
        arn_parts = dataset_arn.split("/")
        hub_content_name = arn_parts[3]
        hub_content_version = arn_parts[4]
        passed = False

        # Read fixture text once; used for title assertion (5) and source-in-fixture check (9).
        fixture_text = (FIXTURES_DIR / "dq_spec_requirements.md").read_text()
        h1_line = next(line for line in fixture_text.splitlines() if line.startswith("# "))
        expected_title = h1_line.lstrip("# ").strip()

        try:
            dataset_config = DatasetJobConfig(
                job_category=JobCategory.DATA_QUALITY_EVALUATION,
                input_data_config=[{"DatasetArn": dataset_arn}],
                evaluation_config={
                    "Operations": [
                        {
                            "Operator": "spec_eval",
                            "Config": {"SpecS3Uri": spec_eval_inputs["spec_uri"]},
                        }
                    ]
                },
                output_data_config={
                    "S3OutputPath": (
                        f"s3://{sagemaker_session.default_bucket()}"
                        "/data-preparer-integ-dq-spec-eval/"
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

            # 2. Schema major version == 3 (spec_eval is v3-only).
            schema_version = desc["JobConfigSchemaVersion"]
            assert schema_version.split(".")[0] == "3", (
                f"Expected schema major version 3, got {schema_version!r}; "
                "spec_eval requires DQ schema v3"
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

            def _read_s3_text(s3_uri):
                parsed = urlparse(s3_uri)
                return (
                    s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
                    .read()
                    .decode("utf-8")
                )

            # 3. quality_report.json has "spec_eval" as a top-level key.
            quality_report = _read_s3_json(f"{output_folder}/aggregate/quality_report.json")
            assert "spec_eval" in quality_report, (
                f"'spec_eval' not found in quality_report.json top-level keys: "
                f"{list(quality_report.keys())}"
            )

            # 4. quality_report["spec_eval"] is a non-empty list with a "result" dict.
            spec_eval_list = quality_report["spec_eval"]
            assert isinstance(spec_eval_list, list) and len(spec_eval_list) > 0, (
                f"quality_report['spec_eval'] should be a non-empty list, "
                f"got: {spec_eval_list!r}"
            )
            result = spec_eval_list[0].get("result")
            assert isinstance(result, dict), (
                f"quality_report['spec_eval'][0] missing 'result' dict: " f"{spec_eval_list[0]!r}"
            )

            # 5. Title round-trip: H1 of the fixture must appear verbatim as result["title"].
            assert result.get("title") == expected_title, (
                f"result['title']={result.get('title')!r} does not match "
                f"fixture H1={expected_title!r}; the supplied document may not have been read"
            )

            # 6. At least one grading criterion declared.
            grading_declared = result.get("grading_criteria_declared", 0)
            assert (
                grading_declared > 0
            ), f"grading_criteria_declared={grading_declared}; expected > 0"

            # 7. Every row got a verdict.
            rows_total = result.get("rows_total", 0)
            rows_judged = result.get("rows_judged", 0)
            rows_missing = result.get("rows_missing_verdicts", -1)
            assert (
                rows_judged == rows_total
            ), f"rows_judged={rows_judged} != rows_total={rows_total}"
            assert rows_missing == 0, f"rows_missing_verdicts={rows_missing}; expected 0"

            # 8. extracted_criteria.json is a non-empty list; every element has id/description/source.
            criteria = _read_s3_json(f"{output_folder}/spec_eval/extracted_criteria.json")
            assert isinstance(criteria, list) and len(criteria) > 0, (
                f"spec_eval/extracted_criteria.json should be a non-empty list, "
                f"got: {criteria!r}"
            )
            for i, crit in enumerate(criteria):
                for field in ("id", "description", "source"):
                    val = crit.get(field)
                    assert (
                        isinstance(val, str) and val
                    ), f"criteria[{i}] missing non-empty string '{field}': {crit!r}"

            # 9. Every criterion's "source" text appears in the fixture document.
            # Normalise whitespace on both sides to tolerate line-ending differences.
            def _norm(text):
                return " ".join(text.split())

            fixture_norm = _norm(fixture_text)
            for i, crit in enumerate(criteria):
                source_norm = _norm(crit["source"])
                assert source_norm in fixture_norm, (
                    f"criteria[{i}] source not found in fixture document.\n"
                    f"  source (normalised): {source_norm!r}\n"
                    f"  fixture snippet: {fixture_norm[:300]!r}"
                )

            # 10. record_level output is non-empty; every line carries dq_spec_eval_pass.
            record_text = _read_s3_text(f"{output_folder}/record_level/evaluated_data.jsonl")
            record_lines = [ln for ln in record_text.splitlines() if ln.strip()]
            assert len(record_lines) > 0, (
                "record_level/evaluated_data.jsonl is empty; "
                "spec_eval must write per-record verdicts"
            )
            for idx, line in enumerate(record_lines):
                record = json.loads(line)
                assert "dq_spec_eval_pass" in record, (
                    f"record_level/evaluated_data.jsonl line {idx} "
                    f"missing 'dq_spec_eval_pass': {list(record.keys())}"
                )

            passed = True

        finally:
            # Keep the dataset on failure so the job logs remain discoverable.
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
