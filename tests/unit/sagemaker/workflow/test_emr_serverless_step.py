"""Unit tests for EMR Serverless step."""

from __future__ import absolute_import

import pytest
from sagemaker.workflow.emr_serverless_step import EMRServerlessStep
from sagemaker.workflow.emr_serverless_step import EMRServerlessJobConfig


class TestEMRServerlessJobConfig:
    """Test EMRServerlessJobConfig class."""

    def test_job_config_structure(self):
        job_config = EMRServerlessJobConfig(
            job_driver={"sparkSubmit": {"entryPoint": "s3://bucket/script.py"}},
            execution_role_arn="arn:aws:iam::123456789012:role/EMRServerlessRole",
            configuration_overrides={
                "applicationConfiguration": [
                    {
                        "classification": "spark-defaults",
                        "properties": {"spark.sql.adaptive.enabled": "true"},
                    }
                ]
            },
        )

        expected = {
            "executionRoleArn": "arn:aws:iam::123456789012:role/EMRServerlessRole",
            "jobDriver": {"sparkSubmit": {"entryPoint": "s3://bucket/script.py"}},
            "configurationOverrides": {
                "applicationConfiguration": [
                    {
                        "classification": "spark-defaults",
                        "properties": {"spark.sql.adaptive.enabled": "true"},
                    }
                ]
            },
        }

        assert job_config.to_request() == expected


class TestEMRServerlessStep:
    """Test EMRServerlessStep class."""

    def test_existing_application_step(self):
        job_config = EMRServerlessJobConfig(
            job_driver={"sparkSubmit": {"entryPoint": "s3://bucket/script.py"}},
            execution_role_arn="arn:aws:iam::123456789012:role/EMRServerlessRole",
        )

        step = EMRServerlessStep(
            name="test-step",
            display_name="Test Step",
            description="Test Description",
            job_config=job_config,
            application_id="app-123",
        )

        expected_args = {
            "ExecutionRoleArn": "arn:aws:iam::123456789012:role/EMRServerlessRole",
            "ApplicationId": "app-123",
            "JobConfig": {
                "applicationId": "app-123",
                "executionRoleArn": "arn:aws:iam::123456789012:role/EMRServerlessRole",
                "jobDriver": {"sparkSubmit": {"entryPoint": "s3://bucket/script.py"}},
            },
        }

        assert step.arguments == expected_args

    def test_new_application_step(self):
        job_config = EMRServerlessJobConfig(
            job_driver={"sparkSubmit": {"entryPoint": "s3://bucket/script.py"}},
            execution_role_arn="arn:aws:iam::123456789012:role/EMRServerlessRole",
        )

        step = EMRServerlessStep(
            name="test-step",
            display_name="Test Step",
            description="Test Description",
            job_config=job_config,
            application_config={
                "name": "test-application",
                "releaseLabel": "emr-6.15.0",
                "type": "SPARK",
            },
        )

        expected_args = {
            "ExecutionRoleArn": "arn:aws:iam::123456789012:role/EMRServerlessRole",
            "ApplicationConfig": {
                "name": "test-application",
                "releaseLabel": "emr-6.15.0",
                "type": "SPARK",
            },
            "JobConfig": {
                "executionRoleArn": "arn:aws:iam::123456789012:role/EMRServerlessRole",
                "jobDriver": {"sparkSubmit": {"entryPoint": "s3://bucket/script.py"}},
            },
        }

        assert step.arguments == expected_args

    def test_validation_errors(self):
        job_config = EMRServerlessJobConfig(
            job_driver={"sparkSubmit": {"entryPoint": "s3://bucket/script.py"}},
            execution_role_arn="arn:aws:iam::123456789012:role/EMRServerlessRole",
        )

        # Should raise error when neither provided
        with pytest.raises(
            ValueError, match="must have either application_id or application_config"
        ):
            EMRServerlessStep(
                name="test-step",
                display_name="Test Step",
                description="Test Description",
                job_config=job_config,
            )

        # Should raise error when both provided
        with pytest.raises(
            ValueError, match="cannot have both application_id and application_config"
        ):
            EMRServerlessStep(
                name="test-step",
                display_name="Test Step",
                description="Test Description",
                job_config=job_config,
                application_id="app-123",
                application_config={"name": "test-app"},
            )

    def test_to_request(self):
        job_config = EMRServerlessJobConfig(
            job_driver={"sparkSubmit": {"entryPoint": "s3://bucket/script.py"}},
            execution_role_arn="arn:aws:iam::123456789012:role/EMRServerlessRole",
        )

        step = EMRServerlessStep(
            name="test-step",
            display_name="Test Step",
            description="Test Description",
            job_config=job_config,
            application_id="app-123",
        )

        request = step.to_request()
        assert request["Name"] == "test-step"
        assert request["Type"] == "EMRServerless"
        assert "Arguments" in request
        assert request["DisplayName"] == "Test Step"
        assert request["Description"] == "Test Description"
