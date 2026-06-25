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
"""Integration test for KMS key propagation in check steps.

This test constructs real QualityCheckStep and ClarifyCheckStep objects using
the actual SDK classes with a real SageMaker Session, then inspects the compiled
step arguments to verify KmsKeyId and VolumeKmsKeyId are present.

No SageMaker compute resources are launched. The only AWS interaction is a small
S3 put_object for the Clarify analysis config (cleaned up in teardown).

Prerequisites:
    - AWS credentials with S3 read/write access to the default SageMaker bucket.

Related ticket: V2184920638
"""
import json
import pytest
import boto3

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.workflow.quality_check_step import (
    QualityCheckStep,
    DataQualityCheckConfig,
)
from sagemaker.mlops.workflow.clarify_check_step import (
    ClarifyCheckStep,
    DataBiasCheckConfig,
)
from sagemaker.mlops.workflow.check_job_config import CheckJobConfig


# Use a fake KMS key ARN — we never actually encrypt anything, we just verify
# the key appears in the compiled request dict.
_TEST_OUTPUT_KMS_KEY = "arn:aws:kms:us-west-2:123456789012:key/test-output-key-id"
_TEST_VOLUME_KMS_KEY = "arn:aws:kms:us-west-2:123456789012:key/test-volume-key-id"

_S3_PREFIX = "integ-test-kms-check-step"


@pytest.fixture(scope="module")
def sagemaker_session():
    """Real SageMaker session with AWS credentials."""
    return Session()


@pytest.fixture(scope="module")
def role():
    return get_execution_role()


@pytest.fixture(scope="module")
def bucket(sagemaker_session):
    return sagemaker_session.default_bucket()


@pytest.fixture(scope="module")
def s3_client(sagemaker_session):
    return boto3.client("s3", region_name=sagemaker_session.boto_region_name)


@pytest.fixture
def check_job_config_with_kms(role, sagemaker_session):
    """CheckJobConfig with both output and volume KMS keys."""
    return CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=30,
        volume_kms_key=_TEST_VOLUME_KMS_KEY,
        output_kms_key=_TEST_OUTPUT_KMS_KEY,
        max_runtime_in_seconds=3600,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture
def check_job_config_no_kms(role, sagemaker_session):
    """CheckJobConfig without KMS keys."""
    return CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=30,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture(autouse=True, scope="module")
def cleanup_s3(bucket, s3_client):
    """Clean up any S3 objects created during the test."""
    yield
    # Teardown: delete all objects under our test prefix
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=_S3_PREFIX)
        if "Contents" in response:
            objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
            s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
    except Exception:
        pass  # Best-effort cleanup


class TestDataQualityCheckStepKms:
    """Verify KMS key propagation in DataQualityCheckStep using real SDK objects."""

    def _build_step(self, check_job_config, bucket):
        """Construct a real DataQualityCheckStep."""
        quality_check_config = DataQualityCheckConfig(
            baseline_dataset=f"s3://{bucket}/{_S3_PREFIX}/input/data.csv",
            dataset_format={"csv": {"header": True}},
            output_s3_uri=f"s3://{bucket}/{_S3_PREFIX}/output/quality-results",
        )
        return QualityCheckStep(
            name="TestDataQualityCheck",
            quality_check_config=quality_check_config,
            check_job_config=check_job_config,
            skip_check=False,
            fail_on_violation=True,
            register_new_baseline=False,
        )

    def test_output_kms_key_in_arguments(self, check_job_config_with_kms, bucket):
        """output_kms_key from CheckJobConfig appears as KmsKeyId in S3Output."""
        step = self._build_step(check_job_config_with_kms, bucket)
        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert "KmsKeyId" in s3_output, (
            f"Expected KmsKeyId in S3Output but got: {s3_output}"
        )
        assert s3_output["KmsKeyId"] == _TEST_OUTPUT_KMS_KEY

    def test_volume_kms_key_in_arguments(self, check_job_config_with_kms, bucket):
        """volume_kms_key from CheckJobConfig appears as VolumeKmsKeyId in ClusterConfig."""
        step = self._build_step(check_job_config_with_kms, bucket)
        args = step.arguments

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert "VolumeKmsKeyId" in cluster_config, (
            f"Expected VolumeKmsKeyId in ClusterConfig but got: {cluster_config}"
        )
        assert cluster_config["VolumeKmsKeyId"] == _TEST_VOLUME_KMS_KEY

    def test_no_kms_keys_when_not_configured(self, check_job_config_no_kms, bucket):
        """KMS keys are absent from arguments when not set in CheckJobConfig."""
        step = self._build_step(check_job_config_no_kms, bucket)
        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert "KmsKeyId" not in s3_output

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert "VolumeKmsKeyId" not in cluster_config

    def test_arguments_are_json_serializable(self, check_job_config_with_kms, bucket):
        """The compiled arguments dict is valid JSON (required for pipeline definitions)."""
        step = self._build_step(check_job_config_with_kms, bucket)
        args = step.arguments

        json_str = json.dumps(args, default=str)
        parsed = json.loads(json_str)
        assert parsed["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["KmsKeyId"] == _TEST_OUTPUT_KMS_KEY
        assert parsed["ProcessingResources"]["ClusterConfig"]["VolumeKmsKeyId"] == _TEST_VOLUME_KMS_KEY


class TestDataBiasCheckStepKms:
    """Verify KMS key propagation in DataBiasCheckStep (ClarifyCheckStep) using real SDK objects."""

    def _build_step(self, check_job_config, bucket):
        """Construct a real DataBiasCheckStep."""
        from sagemaker.core.clarify import DataConfig, BiasConfig

        data_config = DataConfig(
            s3_data_input_path=f"s3://{bucket}/{_S3_PREFIX}/input/bias-data.csv",
            s3_output_path=f"s3://{bucket}/{_S3_PREFIX}/output/bias-results",
            label="target",
            dataset_type="text/csv",
        )
        bias_config = BiasConfig(
            label_values_or_threshold=[1],
            facet_name="gender",
            facet_values_or_threshold=[0],
        )
        clarify_check_config = DataBiasCheckConfig(
            data_config=data_config,
            data_bias_config=bias_config,
        )
        return ClarifyCheckStep(
            name="TestDataBiasCheck",
            clarify_check_config=clarify_check_config,
            check_job_config=check_job_config,
            skip_check=False,
            fail_on_violation=True,
            register_new_baseline=False,
        )

    def test_output_kms_key_in_arguments(self, check_job_config_with_kms, bucket):
        """output_kms_key from CheckJobConfig appears as KmsKeyId in S3Output."""
        step = self._build_step(check_job_config_with_kms, bucket)
        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert "KmsKeyId" in s3_output, (
            f"Expected KmsKeyId in S3Output but got: {s3_output}"
        )
        assert s3_output["KmsKeyId"] == _TEST_OUTPUT_KMS_KEY

    def test_volume_kms_key_in_arguments(self, check_job_config_with_kms, bucket):
        """volume_kms_key from CheckJobConfig appears as VolumeKmsKeyId in ClusterConfig."""
        step = self._build_step(check_job_config_with_kms, bucket)
        args = step.arguments

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert "VolumeKmsKeyId" in cluster_config, (
            f"Expected VolumeKmsKeyId in ClusterConfig but got: {cluster_config}"
        )
        assert cluster_config["VolumeKmsKeyId"] == _TEST_VOLUME_KMS_KEY

    def test_no_kms_keys_when_not_configured(self, check_job_config_no_kms, bucket):
        """KMS keys are absent from arguments when not set in CheckJobConfig."""
        step = self._build_step(check_job_config_no_kms, bucket)
        args = step.arguments

        s3_output = args["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]
        assert "KmsKeyId" not in s3_output

        cluster_config = args["ProcessingResources"]["ClusterConfig"]
        assert "VolumeKmsKeyId" not in cluster_config

    def test_arguments_are_json_serializable(self, check_job_config_with_kms, bucket):
        """The compiled arguments dict is valid JSON (required for pipeline definitions)."""
        step = self._build_step(check_job_config_with_kms, bucket)
        args = step.arguments

        json_str = json.dumps(args, default=str)
        parsed = json.loads(json_str)
        assert parsed["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["KmsKeyId"] == _TEST_OUTPUT_KMS_KEY
        assert parsed["ProcessingResources"]["ClusterConfig"]["VolumeKmsKeyId"] == _TEST_VOLUME_KMS_KEY
