"""Rollout feedback client for reporting completion and rewards to the RFT Runtime Service."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional, Union

import requests as req_lib

from sagemaker.core.token_generator import generate_token
from sagemaker.train.rft.models import RolloutMetadata

logger = logging.getLogger(__name__)

_TRAJECTORY_ALREADY_PROCESSED_MARKERS = (
    "not in valid status",
    "Cannot transition trajectory",
)


def _is_trajectory_already_processed(error: str) -> bool:
    """Return True if the error indicates the trajectory was already completed."""
    return any(marker in error for marker in _TRAJECTORY_ALREADY_PROCESSED_MARKERS)


_DEFAULT_ENDPOINT = os.environ.get(
    "RFT_RUNTIME_ENDPOINT",
    "https://job-runtime.sagemaker.us-east-1.api.aws"
)


def _build_endpoint(region: str, stage: str = "") -> str:
    """Build the RFT Runtime Service endpoint URL for a given region and stage."""
    if stage and stage != "prod":
        prefix = f"job-runtime.{stage}."
    else:
        prefix = "job-runtime."
    return f"https://{prefix}sagemaker.{region}.api.aws"


class RolloutFeedbackClient:
    """Client for reporting rollout completion to the RFT Runtime Service.

    Calls the runtime service's ``/complete-rollout`` and ``/update-reward``
    APIs using bearer token auth.

    Example::

        feedback = RolloutFeedbackClient(metadata)
        feedback.report_complete(reward=0.95)
    """

    def __init__(self, metadata: dict[str, Any] | RolloutMetadata) -> None:
        if isinstance(metadata, RolloutMetadata):
            metadata = metadata.model_dump()
        elif not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict or RolloutMetadata, got {type(metadata).__name__}."
            )

        self._region = (
            metadata.get("region")
            or os.environ.get("AWS_REGION")
            or "us-west-2"
        )
        self._endpoint = (
            metadata.get("endpoint")
            or os.environ.get("RFT_RUNTIME_ENDPOINT")
            or _build_endpoint(self._region, os.environ.get("RFT_STAGE", ""))
        ).rstrip("/")
        self._job_arn = (
            metadata.get("job_arn")
            or metadata.get("jobArn")
            or ""
        )
        self._trajectory_id = (
            metadata.get("trajectory_id")
            or metadata.get("trajectoryId")
            or metadata.get("rolloutId")
            or ""
        )
        self._metadata = metadata

    def complete_rollout(self, status: str = "ready") -> None:
        """Report trajectory completion to the runtime service.

        Args:
            status: Target status - "ready" for success, "failed" for errors.
        """
        if not self._trajectory_id:
            logger.warning("No trajectory_id in metadata; skipping complete_rollout")
            return

        logger.info(
            "CompleteRollout: trajectory_id=%s status=%s",
            self._trajectory_id, status,
        )
        try:
            self._bearer_post("/complete-rollout", json.dumps({
                "JobArn": self._job_arn,
                "TrajectoryId": self._trajectory_id,
                "Status": status,
            }))
        except Exception as e:
            err_str = str(e)
            if "404" in err_str:
                logger.warning(
                    "CompleteRollout 404: trajectory %s not found",
                    self._trajectory_id,
                )
            elif _is_trajectory_already_processed(err_str):
                logger.warning(
                    "CompleteRollout: trajectory %s already in terminal status, skipping",
                    self._trajectory_id,
                )
            else:
                raise

    def update_reward(self, reward: Union[float, List[float]]) -> None:
        """Report reward(s) to the runtime service.

        Args:
            reward: A single float or list of floats for per-turn rewards.
        """
        rewards = [reward] if isinstance(reward, (int, float)) else list(reward)

        if not self._trajectory_id:
            logger.warning("No trajectory_id in metadata; skipping update_reward")
            return

        logger.info(
            "UpdateReward: trajectory_id=%s rewards=%s",
            self._trajectory_id, rewards,
        )
        try:
            self._bearer_post("/update-reward", json.dumps({
                "JobArn": self._job_arn,
                "TrajectoryId": self._trajectory_id,
                "Rewards": rewards,
            }))
        except Exception as e:
            err_str = str(e)
            if "404" in err_str:
                logger.warning(
                    "UpdateReward 404: trajectory %s not found",
                    self._trajectory_id,
                )
            elif _is_trajectory_already_processed(err_str):
                logger.warning(
                    "UpdateReward: trajectory %s already in terminal status, skipping",
                    self._trajectory_id,
                )
            else:
                raise

    def report_complete(self, reward: Union[float, List[float]]) -> None:
        """Complete the trajectory and report reward(s).

        Convenience method that calls complete_rollout() then update_reward().

        Args:
            reward: The computed reward(s) for this rollout.
        """
        self.complete_rollout(status="ready")
        self.update_reward(reward)

    def report_error(self, error: str, reward: Optional[float] = None) -> None:
        """Report a rollout error, marking the trajectory as failed.

        Args:
            error: Error description.
            reward: Optional partial reward (defaults to 0.0).
        """
        logger.error("Rollout error: trajectory_id=%s error=%s", self._trajectory_id, error)
        self.complete_rollout(status="failed")
        self.update_reward(reward if reward is not None else 0.0)

    def _bearer_post(self, path: str, body: str) -> None:
        """Send a bearer-token-authenticated POST to the runtime service."""
        url = f"{self._endpoint}{path}"
        try:
            token = generate_token(region=self._region)
            response = req_lib.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                data=body,
                timeout=120,
            )
            if response.status_code != 200:
                logger.warning("Failed %s: status=%s body=%s", path, response.status_code, response.text[:500])
            response.raise_for_status()
        except Exception as e:
            logger.warning("Failed %s: %s", path, e)
            raise
