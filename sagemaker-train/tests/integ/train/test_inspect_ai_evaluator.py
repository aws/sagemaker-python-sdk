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
"""Integration tests for InspectAIEvaluator.

Prerequisites:
    - Active AWS credentials in us-east-1 with permissions to launch SageMaker
      Pipelines and invoke Bedrock Nova Lite.

The test is self-sufficient: it derives the execution role from the active
credentials (via BaseEvaluator's caller-identity fallback) and uses the
account's default SageMaker bucket (``sagemaker-{region}-{account_id}``,
auto-created by ``Session.default_bucket()``). The BoolQ benchmark files in
``tests/data/inspectai/boolq/`` are uploaded under the ``inspectai-integ/``
prefix if not already present.

Note: the active credentials must be (or be able to assume) a SageMaker
execution role -- i.e. a role whose trust policy allows
``sagemaker.amazonaws.com`` to assume it. CI runners authenticate as such a
role. Credentials whose identity cannot be assumed by SageMaker (e.g. an SSO
admin role) will fail when the pipeline launches.

Run with:
    export AWS_DEFAULT_REGION=us-east-1
    pytest tests/integ/train/test_inspect_ai_evaluator.py -v -s
"""
from __future__ import absolute_import

import logging
import os

import pytest
from sagemaker.train.evaluate import InspectAIEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test configuration
REGION = "us-east-1"
BENCHMARKS_PREFIX = "inspectai-integ/benchmarks/boolq/"
OUTPUT_PREFIX = "inspectai-integ/eval-output/"

# Local BoolQ benchmark fixture files (boolq_pt.py, boolq_data.json, pyproject.toml)
DATA_DIR = os.path.join(os.path.dirname(__file__), "../..", "data")
BENCHMARKS_DATA_DIR = os.path.join(DATA_DIR, "inspectai", "boolq")

EVALUATION_TIMEOUT_SECONDS = 7200  # 2 hours
POLL_INTERVAL_SECONDS = 30


def _prefix_has_content(s3_client, bucket_name: str, prefix: str) -> bool:
    """Check if an S3 prefix has any objects."""
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def _ensure_benchmarks(s3_client, bucket_name: str) -> None:
    """Upload local benchmark files to S3 if not already present."""
    if _prefix_has_content(s3_client, bucket_name, BENCHMARKS_PREFIX):
        logger.info(f"Benchmarks already exist at s3://{bucket_name}/{BENCHMARKS_PREFIX}")
        return

    logger.info(
        f"Uploading benchmarks from {BENCHMARKS_DATA_DIR} "
        f"to s3://{bucket_name}/{BENCHMARKS_PREFIX}"
    )
    for filename in sorted(os.listdir(BENCHMARKS_DATA_DIR)):
        local_path = os.path.join(BENCHMARKS_DATA_DIR, filename)
        if not os.path.isfile(local_path):
            continue
        s3_client.upload_file(local_path, bucket_name, f"{BENCHMARKS_PREFIX}{filename}")
        logger.info(f"Uploaded {filename}")

    logger.info("Benchmarks uploaded successfully.")


@pytest.fixture(scope="module")
def inspect_ai_resources(sagemaker_session_us_east_1):
    """Ensure benchmark content exists in the account's default SageMaker bucket.

    Uses ``Session.default_bucket()`` (creates ``sagemaker-{region}-{account_id}``
    if needed), then uploads the local BoolQ benchmark files under the
    ``inspectai-integ/`` prefix if they are not already present.

    Returns a dict with:
        - benchmarks_path: S3 URI to the benchmark files
        - s3_output_path: S3 URI for evaluation output
    """
    bucket_name = sagemaker_session_us_east_1.default_bucket()
    s3_client = sagemaker_session_us_east_1.boto_session.client("s3", region_name=REGION)
    _ensure_benchmarks(s3_client, bucket_name)

    return {
        "benchmarks_path": f"s3://{bucket_name}/{BENCHMARKS_PREFIX}",
        "s3_output_path": f"s3://{bucket_name}/{OUTPUT_PREFIX}",
        "bucket_name": bucket_name,
    }


@pytest.mark.us_east_1
class TestInspectAIEvaluatorIntegration:
    """Integration tests for InspectAI evaluation with Bedrock inference."""

    def test_inspect_ai_bedrock_evaluation(
        self, sagemaker_session_us_east_1, inspect_ai_resources
    ):
        """Test InspectAI evaluation with Bedrock inference mode.

        Runs a BoolQ benchmark with Nova Lite via Bedrock inference.
        Validates the full lifecycle: create -> wait -> show_results.

        The execution role is resolved from the active credentials by
        BaseEvaluator (no ``role`` argument passed).
        """
        evaluator = InspectAIEvaluator(
            model="nova-textgeneration-lite",
            bedrock_model_id="us.amazon.nova-lite-v1:0",
            benchmarks_path=inspect_ai_resources["benchmarks_path"],
            tasks=[{"name": "boolq_pt", "limit": 10}],
            s3_output_path=inspect_ai_resources["s3_output_path"],
            instance_type="ml.m5.large",
            region=REGION,
            sagemaker_session=sagemaker_session_us_east_1,
        )

        logger.info("Starting InspectAI evaluation with Bedrock inference...")
        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        logger.info(f"Execution ARN: {execution.arn}")

        logger.info(
            f"Waiting for execution (timeout={EVALUATION_TIMEOUT_SECONDS}s, "
            f"poll={POLL_INTERVAL_SECONDS}s)..."
        )
        execution.wait(
            target_status="Succeeded",
            poll=POLL_INTERVAL_SECONDS,
            timeout=EVALUATION_TIMEOUT_SECONDS,
        )

        execution.refresh()
        assert execution.status.overall_status == "Succeeded", (
            f"Execution did not succeed. Status: {execution.status.overall_status}, "
            f"Failure: {execution.status.failure_reason}"
        )

        execution.show_results()
        logger.info("InspectAI Bedrock evaluation completed successfully.")

    def test_inspect_ai_upload_benchmarks(
        self, sagemaker_session_us_east_1, inspect_ai_resources
    ):
        """Test uploading benchmarks to S3 via upload_benchmarks().

        Validates that local benchmark files are successfully uploaded and
        the returned S3 URI is accessible to the training job.
        """
        evaluator = InspectAIEvaluator(
            model="nova-textgeneration-lite",
            bedrock_model_id="us.amazon.nova-lite-v1:0",
            benchmarks_path=inspect_ai_resources["benchmarks_path"],
            s3_output_path=inspect_ai_resources["s3_output_path"],
            instance_type="ml.m5.large",
            region=REGION,
            sagemaker_session=sagemaker_session_us_east_1,
        )

        logger.info(f"Uploading benchmarks from {BENCHMARKS_DATA_DIR}...")
        uploaded_path = evaluator.upload_benchmarks(BENCHMARKS_DATA_DIR)
        logger.info(f"Benchmarks uploaded to: {uploaded_path}")

        assert uploaded_path.startswith("s3://")
        assert "/benchmarks/" in uploaded_path

        # Run an evaluation using the pre-existing benchmarks to confirm
        # the pipeline infrastructure works end-to-end
        evaluator2 = InspectAIEvaluator(
            model="nova-textgeneration-lite",
            bedrock_model_id="us.amazon.nova-lite-v1:0",
            benchmarks_path=inspect_ai_resources["benchmarks_path"],
            tasks=[{"name": "boolq_pt", "limit": 5}],
            s3_output_path=inspect_ai_resources["s3_output_path"],
            instance_type="ml.m5.large",
            region=REGION,
            sagemaker_session=sagemaker_session_us_east_1,
        )

        logger.info("Starting evaluation with pre-existing benchmarks...")
        execution = evaluator2.evaluate()

        assert execution is not None
        assert execution.arn is not None

        execution.wait(
            target_status="Succeeded",
            poll=POLL_INTERVAL_SECONDS,
            timeout=EVALUATION_TIMEOUT_SECONDS,
        )

        execution.refresh()
        assert execution.status.overall_status == "Succeeded", (
            f"Execution did not succeed. Status: {execution.status.overall_status}, "
            f"Failure: {execution.status.failure_reason}"
        )
        execution.show_results()
        logger.info("InspectAI upload + evaluate flow completed successfully.")
