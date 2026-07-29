# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.
"""Unit tests for MTRL eval MLflow URL deep-linking (run-level)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from sagemaker.train.common_utils.mlflow_url_utils import get_presigned_mlflow_url
from sagemaker.train.evaluate.execution import (
    MTRLEvaluationExecution,
    PipelineExecutionStatus,
    StepDetail,
)
from sagemaker.train.evaluate.constants import EvalType


MOCK_MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-test123"
MOCK_PRESIGNED_URL = "https://app-test123.mlflow.sagemaker.us-west-2.app.aws/auth?authToken=eyJtoken123"


class TestGetPresignedMlflowUrl:
    """Tests for the unified get_presigned_mlflow_url function."""

    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_with_experiment_id_and_run_id(self, mock_sm_class):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": MOCK_PRESIGNED_URL
        }

        result = get_presigned_mlflow_url(
            MOCK_MLFLOW_ARN,
            experiment_id="23",
            run_id="65fedc9db0a4491e927dc2766e35ad7a",
        )
        assert "#/experiments/23/runs/65fedc9db0a4491e927dc2766e35ad7a?workspace=default" in result
        assert "authToken=eyJtoken123" in result

    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_with_experiment_id_no_run_id(self, mock_sm_class):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": MOCK_PRESIGNED_URL
        }

        result = get_presigned_mlflow_url(MOCK_MLFLOW_ARN, experiment_id="23")
        assert result.endswith("#/experiments/23?workspace=default")

    @patch("sagemaker.train.common_utils.mlflow_url_utils._resolve_experiment_id")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_fallback_to_experiment_name(self, mock_sm_class, mock_resolve):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": MOCK_PRESIGNED_URL
        }
        mock_resolve.return_value = "42"

        result = get_presigned_mlflow_url(MOCK_MLFLOW_ARN, experiment_name="my-eval-exp")
        assert result.endswith("#/experiments/42?workspace=default")

    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_base_url_when_no_ids(self, mock_sm_class):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": MOCK_PRESIGNED_URL
        }

        result = get_presigned_mlflow_url(MOCK_MLFLOW_ARN)
        assert result == MOCK_PRESIGNED_URL

    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_returns_none_on_failure(self, mock_sm_class):
        mock_sm_class.side_effect = Exception("service error")
        result = get_presigned_mlflow_url(MOCK_MLFLOW_ARN, experiment_id="1")
        assert result is None


class TestMTRLEvaluationExecutionMlflowDetails:
    """Tests for MTRLEvaluationExecution.get_mlflow_details()."""

    def _make_execution(self, steps=None):
        return MTRLEvaluationExecution(
            name="test-eval",
            arn="arn:aws:sagemaker:us-west-2:123:pipeline/pipe/execution/exec-123",
            eval_type=EvalType.MTRL,
            status=PipelineExecutionStatus(
                overall_status="Succeeded",
                step_details=steps or [],
            ),
            mlflow_resource_arn=MOCK_MLFLOW_ARN,
            mlflow_experiment_name="mtrl-eval-test",
        )

    def test_returns_none_when_no_steps(self):
        execution = self._make_execution(steps=[])
        assert execution.get_mlflow_details() is None

    def test_returns_none_when_no_completed_eval_steps(self):
        steps = [
            StepDetail(name="CreateEvaluationAction", status="Succeeded", job_arn=None),
            StepDetail(name="EvaluateBaseModel", status="Executing", job_arn=None),
        ]
        execution = self._make_execution(steps=steps)
        assert execution.get_mlflow_details() is None

    @patch("sagemaker.core.resources.Job")
    def test_extracts_details_from_completed_step(self, mock_job_cls):
        config_doc = json.dumps({
            "ServiceOutput": {
                "MlflowDetails": {
                    "ExperimentName": "mtrl-eval-test",
                    "RunName": "base-model-eval",
                    "ExperimentId": "23",
                    "RunId": "65fedc9db0a4491e927dc2766e35ad7a",
                }
            }
        })
        mock_job = MagicMock()
        mock_job.job_config_document = config_doc
        mock_job_cls.get.return_value = mock_job

        steps = [
            StepDetail(name="CreateEvaluationAction", status="Succeeded", job_arn=None),
            StepDetail(
                name="EvaluateBaseModel",
                status="Succeeded",
                job_arn="arn:aws:sagemaker:us-west-2:123:job/eval-base-123",
            ),
        ]
        execution = self._make_execution(steps=steps)

        details = execution.get_mlflow_details()
        assert details is not None
        assert details["ExperimentId"] == "23"
        assert details["RunId"] == "65fedc9db0a4491e927dc2766e35ad7a"
        assert details["RunName"] == "base-model-eval"
        mock_job_cls.get.assert_called_once_with(
            job_name="eval-base-123", job_category="AgentRFTEvaluation", region="us-west-2"
        )

    @patch("sagemaker.core.resources.Job")
    def test_caches_result(self, mock_job_cls):
        config_doc = json.dumps({
            "ServiceOutput": {
                "MlflowDetails": {
                    "ExperimentId": "23",
                    "RunId": "run-abc",
                    "ExperimentName": "exp",
                    "RunName": "base-model-eval",
                }
            }
        })
        mock_job = MagicMock()
        mock_job.job_config_document = config_doc
        mock_job_cls.get.return_value = mock_job

        steps = [
            StepDetail(
                name="EvaluateBaseModel",
                status="Succeeded",
                job_arn="arn:aws:sagemaker:us-west-2:123:job/eval-123",
            ),
        ]
        execution = self._make_execution(steps=steps)

        execution.get_mlflow_details()
        execution.get_mlflow_details()
        assert mock_job_cls.get.call_count == 1

    def test_skips_non_eval_steps(self):
        steps = [
            StepDetail(
                name="AssociateLineage",
                status="Succeeded",
                job_arn="arn:aws:sagemaker:us-west-2:123:job/lineage-123",
            ),
        ]
        execution = self._make_execution(steps=steps)
        assert execution.get_mlflow_details() is None


class TestMTRLEvaluationExecutionGetMlflowUrl:
    """Tests for MTRLEvaluationExecution.get_mlflow_url()."""

    @patch("sagemaker.core.resources.Job")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_deep_links_to_run_when_details_available(self, mock_sm_class, mock_job_cls):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": MOCK_PRESIGNED_URL
        }

        config_doc = json.dumps({
            "ServiceOutput": {
                "MlflowDetails": {
                    "ExperimentId": "23",
                    "RunId": "run-xyz",
                    "ExperimentName": "exp",
                    "RunName": "base-model-eval",
                }
            }
        })
        mock_job = MagicMock()
        mock_job.job_config_document = config_doc
        mock_job_cls.get.return_value = mock_job

        execution = MTRLEvaluationExecution(
            name="test",
            arn="arn:aws:sagemaker:us-west-2:123:pipeline/p/execution/e",
            eval_type=EvalType.MTRL,
            status=PipelineExecutionStatus(
                overall_status="Succeeded",
                step_details=[
                    StepDetail(
                        name="EvaluateBaseModel",
                        status="Succeeded",
                        job_arn="arn:aws:sagemaker:us-west-2:123:job/j",
                    ),
                ],
            ),
            mlflow_resource_arn=MOCK_MLFLOW_ARN,
            mlflow_experiment_name="my-exp",
        )

        url = execution.get_mlflow_url()
        assert "#/experiments/23/runs/run-xyz?workspace=default" in url
        assert "authToken=eyJtoken123" in url

    @patch("sagemaker.train.common_utils.mlflow_url_utils._resolve_run_id")
    @patch("sagemaker.train.common_utils.mlflow_url_utils._resolve_experiment_id")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_resolves_via_rest_api_when_no_job_details(self, mock_sm_class, mock_resolve_exp, mock_resolve_run):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": MOCK_PRESIGNED_URL
        }
        mock_resolve_exp.return_value = "42"
        mock_resolve_run.return_value = "run-456"

        execution = MTRLEvaluationExecution(
            name="test",
            arn="arn:aws:sagemaker:us-west-2:123:pipeline/p/execution/e",
            eval_type=EvalType.MTRL,
            status=PipelineExecutionStatus(
                overall_status="Executing",
                step_details=[],
            ),
            mlflow_resource_arn=MOCK_MLFLOW_ARN,
            mlflow_experiment_name="my-exp",
        )

        url = execution.get_mlflow_url()
        assert "#/experiments/42/runs/run-456?workspace=default" in url

    @patch("sagemaker.train.common_utils.mlflow_url_utils._resolve_experiment_id")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_raises_when_experiment_not_found(self, mock_sm_class, mock_resolve_exp):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": MOCK_PRESIGNED_URL
        }
        mock_resolve_exp.return_value = None

        execution = MTRLEvaluationExecution(
            name="test",
            arn="arn:aws:sagemaker:us-west-2:123:pipeline/p/execution/e",
            eval_type=EvalType.MTRL,
            status=PipelineExecutionStatus(
                overall_status="Executing",
                step_details=[],
            ),
            mlflow_resource_arn=MOCK_MLFLOW_ARN,
            mlflow_experiment_name="my-exp",
        )

        with pytest.raises(RuntimeError, match="Failed to resolve MLflow experiment ID"):
            execution.get_mlflow_url()

    def test_raises_when_no_arn(self):
        execution = MTRLEvaluationExecution(
            name="test",
            arn="arn:aws:sagemaker:us-west-2:123:pipeline/p/execution/e",
            eval_type=EvalType.MTRL,
            status=PipelineExecutionStatus(overall_status="Executing"),
        )
        with pytest.raises(RuntimeError, match="no mlflow_resource_arn"):
            execution.get_mlflow_url()
