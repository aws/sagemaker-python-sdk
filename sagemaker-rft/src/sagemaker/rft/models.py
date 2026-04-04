"""Contract models for the rollout server API.

These models define the enforced contract between the platform trainer
and customer rollout servers.

Customer server requirements:
    POST /rollout - Accept RolloutRequest
    GET /health   - Return {"status": "healthy"} when ready
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RolloutMetadata(BaseModel):
    """Metadata sent by the trainer with each rollout request.

    Pass this entire object (or its dict form) to RolloutFeedbackClient
    and make_inference_headers.
    """

    job_arn: str = Field(description="Training job ARN")
    trajectory_id: str = Field(description="Unique trajectory identifier")
    endpoint: str = Field(description="RFT Runtime Service endpoint URL")
    region: str = Field(default="us-west-2", description="AWS region")


class InferenceParams(BaseModel):
    """Inference parameters for rollout sampling.

    All fields are optional - if not provided, model defaults are used.
    """

    temperature: float | None = Field(default=None, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    top_p: float | None = Field(default=None, description="Top-p (nucleus) sampling")


class RolloutRequest(BaseModel):
    """Request format sent by the trainer to your /rollout endpoint.

    This is the enforced contract. Your server must accept this exact format.
    """

    instance: Dict[str, Any] = Field(
        description="Problem instance from customer's data file"
    )
    metadata: RolloutMetadata = Field(description="Platform-provided rollout context")
    inference_params: InferenceParams | None = Field(
        default=None,
        description="Optional inference parameters (temperature, max_tokens, top_p)",
    )
    model_name: str | None = Field(
        default=None, description="Optional model name override from trainer"
    )
    model_endpoint: str | None = Field(
        default=None, description="Optional model endpoint override from trainer"
    )
