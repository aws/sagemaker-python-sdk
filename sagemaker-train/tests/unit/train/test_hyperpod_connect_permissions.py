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
"""Unit tests for the HyperPod caller-side connect-permission integration.

The HyperPod CLI (connect-cluster + start-job) runs locally under the caller's
own credentials, so the SDK cannot auto-create a role for it. Instead the trainer
and evaluator HyperPod paths call ``TrainDefaults.verify_hyperpod_caller_permissions``
up front. These tests confirm that wiring without hitting AWS — both at the
trainer seam (patching the helper) and end-to-end (real helper + resolver,
mocking only the boto IAM/STS clients).
"""
from __future__ import absolute_import

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _Compute:
    """Minimal stand-in for HyperPodCompute (avoids importing the real config)."""

    def __init__(self, cluster_name="my-cluster", namespace="kubeflow"):
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.instance_type = "ml.p5.48xlarge"
        self.node_count = 1


def _make_base_trainer():
    """Build a bare trainer instance without running heavy __init__ work.

    Uses the concrete ``SFTTrainer`` (BaseTrainer is abstract) but bypasses
    ``__init__`` so no model/Hub resolution happens; ``_train_hyperpod`` itself
    is defined on BaseTrainer and shared across trainers.
    """
    from sagemaker.train.sft_trainer import SFTTrainer

    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.sagemaker_session = MagicMock()
    trainer.compute = _Compute()
    trainer._model_name = "amazon.nova-lite-v2"
    trainer._customization_technique = "SFT"
    trainer.training_type = "LORA"
    trainer.training_image = "123.dkr.ecr.us-west-2.amazonaws.com/img:latest"
    trainer.base_job_name = None
    trainer.training_dataset = None
    trainer.validation_dataset = None
    trainer.s3_output_path = "s3://bucket/out/"
    trainer.hyperparameters = None
    trainer._recipe_path = None
    trainer._overrides = None
    return trainer


class TestTrainHyperPodVerifiesConnectPermissions:
    """BaseTrainer._train_hyperpod must verify caller connect perms before submit.

    The permission step is routed through
    ``TrainDefaults.verify_hyperpod_caller_permissions``, so these tests patch that
    single seam and ``validate_hyperpod_compute`` (imported at module top).
    """

    @patch("sagemaker.train.base_trainer.subprocess")
    @patch("sagemaker.train.base_trainer.TrainDefaults.verify_hyperpod_caller_permissions")
    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    def test_verify_called_with_cluster_name(
        self, mock_get_session, mock_validate, mock_verify, mock_subprocess
    ):
        mock_get_session.return_value = MagicMock()
        # start-job output the parser expects.
        mock_subprocess.run.return_value = SimpleNamespace(
            stdout="NAME: my-job-123\n", stderr=""
        )

        trainer = _make_base_trainer()
        # Avoid Hub/image lookups by pre-setting the recipe + image.
        with patch(
            "sagemaker.train.common_utils.finetune_utils.get_training_image",
            return_value=None,
        ), patch(
            "sagemaker.train.base_trainer.get_hyperpod_recipe_path",
            return_value="fine-tuning/nova/test-recipe",
        ):
            trainer._train_hyperpod(wait=False)

        mock_verify.assert_called_once()
        # The cluster name is forwarded so the warning can name the cluster.
        assert mock_verify.call_args.kwargs.get("cluster_name") == "my-cluster"

    @patch("sagemaker.train.base_trainer.subprocess")
    @patch("sagemaker.train.base_trainer.TrainDefaults.verify_hyperpod_caller_permissions")
    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    def test_missing_cluster_name_raises_before_verify(
        self, mock_get_session, mock_validate, mock_verify, mock_subprocess
    ):
        mock_get_session.return_value = MagicMock()
        trainer = _make_base_trainer()
        trainer.compute = _Compute(cluster_name="")

        with pytest.raises(ValueError, match="cluster_name is required"):
            trainer._train_hyperpod(wait=False)

        # The cluster_name guard fires before the permission step.
        mock_verify.assert_not_called()

    @patch("sagemaker.train.base_trainer.subprocess")
    @patch("sagemaker.train.base_trainer.TrainDefaults.verify_hyperpod_caller_permissions")
    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    def test_verify_failure_does_not_block_submit(
        self, mock_get_session, mock_validate, mock_verify, mock_subprocess
    ):
        """A non-blocking verdict (None/False) still lets submission proceed."""
        mock_get_session.return_value = MagicMock()
        mock_verify.return_value = None  # caller perms unverifiable → warn-only
        mock_subprocess.run.return_value = SimpleNamespace(
            stdout="NAME: my-job-123\n", stderr=""
        )

        trainer = _make_base_trainer()
        with patch(
            "sagemaker.train.common_utils.finetune_utils.get_training_image",
            return_value=None,
        ), patch(
            "sagemaker.train.base_trainer.get_hyperpod_recipe_path",
            return_value="fine-tuning/nova/test-recipe",
        ):
            job_name = trainer._train_hyperpod(wait=False)

        assert job_name == "my-job-123"
        mock_verify.assert_called_once()


class TestTrainHyperPodConnectPermissionsEndToEnd:
    """End-to-end: base_trainer -> real verify helper -> real resolver internals.

    Nothing in the verification chain is mocked except the boto IAM/STS clients,
    so this exercises that _train_hyperpod actually reaches
    verify_hyperpod_connect_permissions and that a denied connect action surfaces
    a WARNING without blocking submission.
    """

    def _iam_sts_session(self, decisions):
        """A SageMaker session whose IAM simulate paginator yields `decisions`."""
        mock_session = MagicMock()
        mock_iam = MagicMock()
        mock_sts = MagicMock()

        def client_factory(service, **kwargs):
            return mock_iam if service == "iam" else mock_sts

        mock_session.boto_session.client.side_effect = client_factory
        mock_sts.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/CallerRole/session",
            "Account": "123456789012",
        }
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/CallerRole"}
        }
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "EvaluationResults": [
                    {"EvalActionName": a, "EvalDecision": d} for a, d in decisions
                ]
            }
        ]
        mock_iam.get_paginator.return_value = paginator
        return mock_session

    @patch("sagemaker.train.base_trainer.subprocess")
    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    def test_denied_connect_action_warns_but_submits(
        self, mock_get_session, mock_validate, mock_subprocess, caplog
    ):
        import logging

        # Caller is missing one of the HyperPod connect actions.
        session = self._iam_sts_session(
            [
                ("sagemaker:DescribeCluster", "allowed"),
                ("eks:DescribeCluster", "implicitDeny"),
                ("eks:AccessKubernetesApi", "allowed"),
            ]
        )
        mock_get_session.return_value = session
        mock_subprocess.run.return_value = SimpleNamespace(
            stdout="NAME: my-job-123\n", stderr=""
        )

        trainer = _make_base_trainer()
        trainer.sagemaker_session = session
        with patch(
            "sagemaker.train.common_utils.finetune_utils.get_training_image",
            return_value=None,
        ), patch(
            "sagemaker.train.base_trainer.get_hyperpod_recipe_path",
            return_value="fine-tuning/nova/test-recipe",
        ), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            job_name = trainer._train_hyperpod(wait=False)

        # Submission still proceeds despite the missing permission.
        assert job_name == "my-job-123"
        # The real resolver simulated the caller and warned about the gap.
        assert any(
            "eks:DescribeCluster" in r.getMessage() and "my-cluster" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    @patch("sagemaker.train.base_trainer.subprocess")
    @patch("sagemaker.train.base_trainer.validate_hyperpod_compute")
    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    def test_all_connect_actions_allowed_no_warning(
        self, mock_get_session, mock_validate, mock_subprocess, caplog
    ):
        import logging

        session = self._iam_sts_session(
            [
                ("sagemaker:DescribeCluster", "allowed"),
                ("eks:DescribeCluster", "allowed"),
                ("eks:AccessKubernetesApi", "allowed"),
            ]
        )
        mock_get_session.return_value = session
        mock_subprocess.run.return_value = SimpleNamespace(
            stdout="NAME: my-job-123\n", stderr=""
        )

        trainer = _make_base_trainer()
        trainer.sagemaker_session = session
        with patch(
            "sagemaker.train.common_utils.finetune_utils.get_training_image",
            return_value=None,
        ), patch(
            "sagemaker.train.base_trainer.get_hyperpod_recipe_path",
            return_value="fine-tuning/nova/test-recipe",
        ), caplog.at_level(
            logging.WARNING, logger="sagemaker.core.helper.iam_role_resolver"
        ):
            job_name = trainer._train_hyperpod(wait=False)

        assert job_name == "my-job-123"
        assert not [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "missing IAM permissions" in r.getMessage()
        ]
