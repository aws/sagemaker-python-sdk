"""Rollout feedback client for reporting completion status to the trainer."""

from __future__ import annotations

import logging
from typing import Any

import requests

from sagemaker.rft.models import RolloutMetadata

logger = logging.getLogger(__name__)


class RolloutFeedbackClient:
    """Client for reporting rollout completion to the trainer.

    Create one instance per rollout and call either report_complete()
    or report_error() when the rollout finishes.

    Example::

        feedback = RolloutFeedbackClient(request.metadata)
        try:
            result = run_agent(...)
            feedback.report_complete(reward=compute_reward(result))
        except Exception:
            feedback.report_error()
            raise
    """

    def __init__(self, metadata: dict[str, Any] | RolloutMetadata) -> None:
        """Initialize the feedback client.

        Args:
            metadata: Rollout metadata - either a dict or RolloutMetadata instance.
                Must contain reporting_address.

        Raises:
            TypeError: If metadata is not a dict or RolloutMetadata.
            ValueError: If metadata is missing required 'reporting_address' field.
        """
        if isinstance(metadata, RolloutMetadata):
            metadata = metadata.model_dump()
        elif not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict or RolloutMetadata, got {type(metadata).__name__}."
            )

        if "reporting_address" not in metadata:
            raise ValueError(
                "metadata missing required field 'reporting_address'. "
                "Ensure you're passing the metadata from the rollout request."
            )

        self._server_address = metadata["reporting_address"].rstrip("/")
        self._metadata = metadata

    def report_complete(self, reward: float | list[float] | None = None) -> None:
        """Report successful rollout completion, optionally with reward.

        Args:
            reward: Optional reward value(s). Can be a single float or a list
                of floats for multi-step rewards.
        """
        if reward is not None:
            self._report_reward(reward)
        self._report_status("finished")

    def report_error(self) -> None:
        """Report rollout error."""
        self._report_status("error")

    def _report_reward(self, reward: float | list[float]) -> None:
        """Report reward to the trainer."""
        try:
            response = requests.post(
                f"{self._server_address}/rollout/reward",
                headers={"Content-Type": "application/json"},
                json={"metadata": self._metadata, "reward": reward},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to report reward for {self._metadata}: {e}")

    def _report_status(self, status: str) -> None:
        """Report rollout status to the trainer."""
        try:
            response = requests.post(
                f"{self._server_address}/rollout/status",
                headers={"Content-Type": "application/json"},
                json={"metadata": self._metadata, "status": status},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to report status for {self._metadata}: {e}")
