"""Thread-local context for RFT rollout metadata.

The rft_handler decorator populates this context from the payload metadata.
The Strands model wrapper reads it to inject per-request headers.
"""

import threading
import uuid
from typing import Optional

_context = threading.local()


class RFTContext:
    """Access the current RFT rollout context.

    Set by @rft_handler, read by wrap_model adapters to inject headers.

    The injected headers match the AgenticRFTRuntimeService API:
      - ``X-Rft-Job-Arn``: job ARN that identifies the Lego session
      - ``X-Trajectory-Id``: groups turns into a single trajectory
      - ``X-Span-Id``: unique ID for each turn within the trajectory
    """

    @staticmethod
    def get_headers() -> dict:
        """Return HTTP headers for the current rollout context.

        A new ``X-Span-Id`` is generated on every call so each inference
        turn gets a unique span within the trajectory.
        """
        metadata = getattr(_context, "metadata", None)
        if metadata is None:
            return {}
        headers = {}
        if metadata.get("job_arn"):
            headers["X-Rft-Job-Arn"] = metadata["job_arn"]
        if metadata.get("trajectory_id"):
            headers["X-Trajectory-Id"] = metadata["trajectory_id"]
            # Auto-generate a span ID for each inference call
            headers["X-Span-Id"] = str(uuid.uuid4())
        return headers

    @staticmethod
    def get_metadata() -> Optional[dict]:
        """Return the raw metadata dict, or None if not in an RFT context."""
        return getattr(_context, "metadata", None)


def _set_metadata(metadata: dict):
    _context.metadata = metadata


def _clear_metadata():
    _context.metadata = None
