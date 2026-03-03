"""RolloutFeedbackClient — reports rewards and completion status back to the training service."""

import logging
from typing import Optional

import boto3

logger = logging.getLogger(__name__)


class RolloutFeedbackClient:
    """Client for reporting rollout feedback (rewards) to the RFT training service.

    Typically used inside an @rft_handler-decorated entrypoint to report
    the reward computed from the agent's rollout.

    Example::

        from aws_rft_sdk import RolloutFeedbackClient

        client = RolloutFeedbackClient(payload.get("metadata"))
        client.report_complete(reward=0.85)

    Args:
        metadata: The ``metadata`` dict from the rollout payload. Contains
            training_job_arn, rollout_id, feedback_endpoint, etc.
    """

    def __init__(self, metadata: dict):
        self._metadata = metadata or {}
        self._training_job_arn = self._metadata.get("training_job_arn")
        self._rollout_id = self._metadata.get("rollout_id")
        self._feedback_endpoint = self._metadata.get("feedback_endpoint")
        self._client = None

    def _get_client(self):
        if self._client is None:
            kwargs = {}
            if self._feedback_endpoint:
                kwargs["endpoint_url"] = self._feedback_endpoint
            self._client = boto3.client("sagemaker", **kwargs)
        return self._client

    def report_complete(self, reward: float):
        """Report successful rollout completion with a reward score.

        Args:
            reward: The computed reward for this rollout (typically 0.0–1.0).
        """
        logger.info(
            "Reporting rollout complete: training_job=%s rollout=%s reward=%s",
            self._training_job_arn,
            self._rollout_id,
            reward,
        )
        # TODO: Replace with actual RFT feedback API call when available.
        # The service API will accept:
        #   - TrainingJobArn
        #   - RolloutId
        #   - Reward (float)
        #   - Status (COMPLETED)
        client = self._get_client()
        # Placeholder — actual API TBD
        # client.send_rollout_feedback(
        #     TrainingJobArn=self._training_job_arn,
        #     RolloutId=self._rollout_id,
        #     Reward=reward,
        #     Status="COMPLETED",
        # )

    def report_error(self, error: str, reward: Optional[float] = None):
        """Report a rollout error.

        Args:
            error: Error description.
            reward: Optional partial reward (defaults to 0.0).
        """
        logger.error(
            "Reporting rollout error: training_job=%s rollout=%s error=%s",
            self._training_job_arn,
            self._rollout_id,
            error,
        )
        # TODO: Replace with actual RFT feedback API call when available.
        # client.send_rollout_feedback(
        #     TrainingJobArn=self._training_job_arn,
        #     RolloutId=self._rollout_id,
        #     Reward=reward or 0.0,
        #     Status="FAILED",
        #     ErrorMessage=error,
        # )
