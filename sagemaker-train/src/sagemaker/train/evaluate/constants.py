"""Constants for SageMaker Evaluation Module.

This module contains shared constants used across the evaluation module.
"""

import uuid
from enum import Enum
from sagemaker.core.image_uris import _registry_from_region, config_for_framework
from typing import Optional

class EvalType(Enum):
    """Enumeration of supported evaluation types."""

    BENCHMARK = "benchmark"
    CUSTOM_SCORER = "customscorer"
    LLM_AS_JUDGE = "llmasjudge"
    MTRL = "mtrl"
    INSPECT_AI = "inspectai"


# Pipeline naming convention
_PIPELINE_NAME_PREFIX = "SagemakerEvaluation"


def _get_eval_type_display_name(eval_type: EvalType) -> str:
    """Map eval type to display name for pipeline naming.

    Args:
        eval_type (EvalType): The evaluation type.

    Returns:
        str: The display name for the evaluation type.
            - BENCHMARK → "BenchmarkEvaluation"
            - CUSTOM_SCORER → "CustomScorerEvaluation"
            - LLM_AS_JUDGE → "LLMAJEvaluation"
            - MTRL → "MTRLEvaluation"
            - INSPECT_AI → "InspectAIEvaluation"
    """
    mapping = {
        EvalType.BENCHMARK: "BenchmarkEvaluation",
        EvalType.CUSTOM_SCORER: "CustomScorerEvaluation",
        EvalType.LLM_AS_JUDGE: "LLMAJEvaluation",
        EvalType.MTRL: "MTRLEvaluation",
        EvalType.INSPECT_AI: "InspectAIEvaluation",
    }
    return mapping[eval_type]


def _get_pipeline_name(eval_type: EvalType, unique_id: Optional[str] = None) -> str:
    """Generate pipeline name with pattern: SagemakerEvaluation-[evaluationType]-[uuid]

    The generated name follows the pattern: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}

    Args:
        eval_type (EvalType): The evaluation type.
        unique_id (Optional[str]): Optional UUID. If not provided, generates a new one.

    Returns:
        str: The pipeline name (e.g., 'SagemakerEvaluation-BenchmarkEvaluation-abc123').
    """
    eval_type_name = _get_eval_type_display_name(eval_type)
    if unique_id is None:
        unique_id = str(uuid.uuid4())
    return f"{_PIPELINE_NAME_PREFIX}-{eval_type_name}-{unique_id}"


def _get_pipeline_name_prefix(eval_type: EvalType) -> str:
    """Get pipeline name prefix for searching existing pipelines.

    Args:
        eval_type (EvalType): The evaluation type.

    Returns:
        str: The pipeline name prefix (e.g., 'SagemakerEvaluation-BenchmarkEvaluation').
    """
    eval_type_name = _get_eval_type_display_name(eval_type)
    return f"{_PIPELINE_NAME_PREFIX}-{eval_type_name}"


# Tag keys for pipeline execution classification
_TAG_EVAL_TYPE_PREFIX = "sagemaker-pysdk"
_TAG_EVALUATION = "sagemaker-pysdk-evaluation"
_TAG_SAGEMAKER_MODEL_EVALUATION = "SagemakerModelEvaluation"


_INSPECT_AI_FRAMEWORK = "sagemaker-inspect-ai"


def _get_inspect_ai_default_image_uri(region: str) -> str:
    """Get the default InspectAI container image URI for a given region.

    Resolves the ECR registry account from the per-region map in
    ``sagemaker-core/.../image_uri_config/sagemaker-inspect-ai.json``,
    matching the convention used by every other Deep Learning Container
    family in this SDK.

    Args:
        region: AWS region (e.g., 'us-east-1', 'cn-north-1', 'us-gov-west-1').

    Returns:
        ECR image URI for the InspectAI container.

    Raises:
        ValueError: If ``region`` is not present in the registry map.
    """

    config = config_for_framework(_INSPECT_AI_FRAMEWORK)
    version_config = config["versions"]["latest"]
    registry = _registry_from_region(region, version_config["registries"])
    return f"{registry}.dkr.ecr.{region}.amazonaws.com/{version_config['repository']}"


def _get_eval_type_tag_key(eval_type: EvalType) -> str:
    """Get the tag key for a specific evaluation type.

    Args:
        eval_type (EvalType): The evaluation type.

    Returns:
        str: The tag key (e.g., 'sagemaker-pysdk-benchmark').
    """
    return f"{_TAG_EVAL_TYPE_PREFIX}-{eval_type.value}"
