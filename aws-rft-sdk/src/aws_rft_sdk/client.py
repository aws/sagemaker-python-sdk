"""RolloutFeedbackClient — reports rewards and trajectory completion to AgenticRFTRuntimeService."""

import json
import logging
from typing import List, Optional

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

logger = logging.getLogger(__name__)

# Alpha endpoint; override via metadata["endpoint"] or AGENTIC_RFT_ENDPOINT env var.
_DEFAULT_ENDPOINT = "https://finetuning-job-runtime.alpha.sagemaker.us-west-2.api.aws"
_SIGNING_SERVICE = "sagemaker"


class RolloutFeedbackClient:
    """Client for reporting rollout feedback to the AgenticRFTRuntimeService.

    Calls the real CompleteTrajectory and UpdateReward APIs using SigV4 auth.

    Example::

        from aws_rft_sdk import RolloutFeedbackClient

        client = RolloutFeedbackClient(payload["metadata"])
        client.complete_trajectory()
        client.update_reward([0.8, 0.9, 1.0])

    Args:
        metadata: The ``metadata`` dict from the rollout payload.  Expected keys:
            - ``job_arn``: the RFT job ARN
            - ``trajectory_id``: trajectory to act on
            - ``endpoint`` (optional): override the runtime service URL
            - ``region`` (optional): AWS region (default us-west-2)
    """

    def __init__(self, metadata: dict):
        self._metadata = metadata or {}
        self._job_arn = self._metadata.get("job_arn")
        self._trajectory_id = self._metadata.get("trajectory_id")
        self._endpoint = (
            self._metadata.get("endpoint")
            or _DEFAULT_ENDPOINT
        )
        self._region = self._metadata.get("region", "us-west-2")
        self._credentials = None

    def _get_credentials(self):
        if self._credentials is None:
            session = boto3.Session(region_name=self._region)
            self._credentials = session.get_credentials().get_frozen_credentials()
        return self._credentials

    def _signed_request(self, method: str, path: str, body: dict) -> dict:
        """Send a SigV4-signed request to the runtime service."""
        import requests as http_requests

        url = f"{self._endpoint}{path}"
        data = json.dumps(body)
        headers = {"Content-Type": "application/json"}

        aws_request = AWSRequest(method=method, url=url, data=data, headers=headers)
        SigV4Auth(self._get_credentials(), _SIGNING_SERVICE, self._region).add_auth(aws_request)

        resp = http_requests.request(
            method=method,
            url=url,
            headers=dict(aws_request.headers),
            data=data,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    def complete_trajectory(self):
        """Mark the trajectory as complete (PENDING -> READY).

        Calls POST /CompleteTrajectory with the trajectory ID.
        """
        if not self._trajectory_id:
            logger.warning("No trajectory_id in metadata; skipping complete_trajectory")
            return

        logger.info(
            "CompleteTrajectory: trajectory_id=%s",
            self._trajectory_id,
        )
        self._signed_request("POST", "/CompleteTrajectory", {
            "TrajectoryId": self._trajectory_id,
        })

    def update_reward(self, rewards: List[float]):
        """Submit reward scores for the trajectory (READY -> REWARD_RECEIVED).

        Calls POST /UpdateReward with per-transition rewards.

        Args:
            rewards: List of reward values, one per transition in the trajectory.
        """
        if not self._trajectory_id:
            logger.warning("No trajectory_id in metadata; skipping update_reward")
            return

        logger.info(
            "UpdateReward: trajectory_id=%s rewards=%s",
            self._trajectory_id,
            rewards,
        )
        self._signed_request("POST", "/UpdateReward", {
            "TrajectoryId": self._trajectory_id,
            "Rewards": rewards,
        })

    # Convenience wrappers (backward-compatible names)

    def report_complete(self, reward: float):
        """Complete the trajectory and report a single reward.

        This is a convenience method that calls complete_trajectory()
        then update_reward() with a single reward value.

        Args:
            reward: The computed reward for this rollout.
        """
        self.complete_trajectory()
        self.update_reward([reward])

    def report_error(self, error: str, reward: Optional[float] = None):
        """Log a rollout error.

        Args:
            error: Error description.
            reward: Optional partial reward.
        """
        logger.error(
            "Rollout error: trajectory_id=%s error=%s",
            self._trajectory_id,
            error,
        )
        # Still try to complete + report zero reward so the trajectory isn't stuck
        try:
            self.complete_trajectory()
            self.update_reward([reward or 0.0])
        except Exception:
            logger.exception("Failed to report error reward")
