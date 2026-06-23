# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.
"""Unit tests for the shared resolve_mlflow_tracking_fields utility."""

import pytest

from sagemaker.train.common_utils.mlflow_config_utils import resolve_mlflow_tracking_fields


DEFAULT_MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"


class TestResolveMlflowTrackingFields:
    """Tests for the shared resolve_mlflow_tracking_fields function."""

    def test_defaults_experiment_and_run_name_when_uri_set(self):
        """When tracking URI is set but names are empty, default both to base_job_name."""
        uri, exp, run = resolve_mlflow_tracking_fields(
            mlflow_tracking_uri=DEFAULT_MLFLOW_ARN,
            mlflow_experiment_name=None,
            mlflow_run_name=None,
            base_job_name="my-training-job",
        )
        assert uri == DEFAULT_MLFLOW_ARN
        assert exp == "my-training-job"
        assert run == "my-training-job"

    def test_defaults_only_experiment_name_when_run_name_set(self):
        """If only run_name is set, experiment_name still defaults."""
        uri, exp, run = resolve_mlflow_tracking_fields(
            mlflow_tracking_uri=DEFAULT_MLFLOW_ARN,
            mlflow_experiment_name=None,
            mlflow_run_name="user-run",
            base_job_name="my-training-job",
        )
        assert uri == DEFAULT_MLFLOW_ARN
        assert exp == "my-training-job"
        assert run == "user-run"

    def test_defaults_only_run_name_when_experiment_name_set(self):
        """If only experiment_name is set, run_name still defaults."""
        uri, exp, run = resolve_mlflow_tracking_fields(
            mlflow_tracking_uri=DEFAULT_MLFLOW_ARN,
            mlflow_experiment_name="user-experiment",
            mlflow_run_name=None,
            base_job_name="my-training-job",
        )
        assert uri == DEFAULT_MLFLOW_ARN
        assert exp == "user-experiment"
        assert run == "my-training-job"

    def test_preserves_user_provided_names(self):
        """User-provided experiment and run names are never overridden."""
        uri, exp, run = resolve_mlflow_tracking_fields(
            mlflow_tracking_uri=DEFAULT_MLFLOW_ARN,
            mlflow_experiment_name="my-experiment",
            mlflow_run_name="my-run",
            base_job_name="fallback-name",
        )
        assert uri == DEFAULT_MLFLOW_ARN
        assert exp == "my-experiment"
        assert run == "my-run"

    def test_no_defaults_when_uri_not_set(self):
        """Without a tracking URI, all fields stay empty."""
        uri, exp, run = resolve_mlflow_tracking_fields(
            mlflow_tracking_uri=None,
            mlflow_experiment_name=None,
            mlflow_run_name=None,
            base_job_name="my-training-job",
        )
        assert uri == ""
        assert exp == ""
        assert run == ""

    def test_empty_string_uri_treated_as_not_set(self):
        """An empty string tracking URI is treated as disabled."""
        uri, exp, run = resolve_mlflow_tracking_fields(
            mlflow_tracking_uri="",
            mlflow_experiment_name=None,
            mlflow_run_name=None,
            base_job_name="my-training-job",
        )
        assert uri == ""
        assert exp == ""
        assert run == ""

    def test_empty_string_names_treated_as_not_set(self):
        """Empty string names are treated as not set and defaulted."""
        uri, exp, run = resolve_mlflow_tracking_fields(
            mlflow_tracking_uri=DEFAULT_MLFLOW_ARN,
            mlflow_experiment_name="",
            mlflow_run_name="",
            base_job_name="my-job",
        )
        assert uri == DEFAULT_MLFLOW_ARN
        assert exp == "my-job"
        assert run == "my-job"
