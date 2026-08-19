"""Rollout feedback client for reporting completion and rewards to the RFT Runtime Service."""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSPreparedRequest

from sagemaker.rft.models import RolloutMetadata

logger = logging.getLogger(__name__)


class RolloutFeedbackClient:
    """Client for reporting rollout completion to the RFT Runtime Service.

    Calls the runtime service's ``/CompleteTrajectory`` and ``/UpdateReward``
    APIs using SigV4-signed requests.

    Example::

        feedback = RolloutFeedbackClient(metadata)
        try:
            result = run_agent(...)
            feedback.complete_trajectory()
            feedback.update_reward(reward=0.95)
        except Exception:
            logger.exception("Rollout failed")
            raise
    """

    def __init__(self, metadata: dict[str, Any] | RolloutMetadata) -> None:
        """Initialize the feedback client.

        Args:
            metadata: Rollout metadata dict or RolloutMetadata instance.
                Expected keys: ``endpoint``, ``job_arn``, ``trajectory_id``, ``region``.
        """
        if isinstance(metadata, RolloutMetadata):
            metadata = metadata.model_dump()
        elif not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict or RolloutMetadata, got {type(metadata).__name__}."
            )

        self._endpoint = metadata.get("endpoint", "").rstrip("/")
        self._job_arn = metadata.get("job_arn", "")
        self._trajectory_id = metadata.get("trajectory_id", "")
        self._region = metadata.get("region", "us-west-2")
        self._metadata = metadata

    def complete_trajectory(self) -> None:
        """Report trajectory completion to the runtime service."""
        logger.info("CompleteTrajectory: trajectory_id=%s", self._trajectory_id)
        body = json.dumps({
            "trajectoryId": self._trajectory_id,
            "jobArn": self._job_arn,
        })
        self._signed_post("/CompleteTrajectory", body)

    def update_reward(self, reward: float | list[float]) -> None:
        """Report reward(s) to the runtime service.

        Args:
            reward: A single float or list of floats for multi-step rewards.
        """
        rewards = [reward] if isinstance(reward, (int, float)) else reward
        body = json.dumps({
            "trajectoryId": self._trajectory_id,
            "jobArn": self._job_arn,
            "rewards": rewards,
        })
        self._signed_post("/UpdateReward", body)

    def _signed_post(self, path: str, body: str) -> None:
        """Send a SigV4-signed POST to the runtime service."""
        import requests as req_lib

        url = f"{self._endpoint}{path}"
        try:
            session = boto3.Session()
            credentials = session.get_credentials().get_frozen_credentials()
            request = AWSPreparedRequest(
                method="POST",
                url=url,
                headers={"Content-Type": "application/json"},
                body=body,
            )
            SigV4Auth(credentials, "sagemaker", self._region).add_auth(request)
            response = req_lib.post(
                url,
                headers=dict(request.headers),
                data=body,
                timeout=10,
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning("Failed %s: %s", path, e)
