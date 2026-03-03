"""Thread-local context for RFT rollout metadata.

The rft_handler decorator populates this context from the payload metadata.
The Strands model wrapper reads it to inject per-request headers.
"""

import threading
from typing import Optional

_context = threading.local()


class RFTContext:
    """Access the current RFT rollout context.

    Set by @rft_handler, read by wrap_model adapters to inject headers.
    """

    @staticmethod
    def get_headers() -> dict:
        """Return HTTP headers for the current rollout context."""
        metadata = getattr(_context, "metadata", None)
        if metadata is None:
            return {}
        headers = {}
        if metadata.get("training_job_arn"):
            headers["X-RFT-Training-Job-Arn"] = metadata["training_job_arn"]
        if metadata.get("rollout_id"):
            headers["X-RFT-Rollout-Id"] = metadata["rollout_id"]
        if metadata.get("episode_id"):
            headers["X-RFT-Episode-Id"] = metadata["episode_id"]
        return headers

    @staticmethod
    def get_metadata() -> Optional[dict]:
        """Return the raw metadata dict, or None if not in an RFT context."""
        return getattr(_context, "metadata", None)


def _set_metadata(metadata: dict):
    _context.metadata = metadata


def _clear_metadata():
    _context.metadata = None
