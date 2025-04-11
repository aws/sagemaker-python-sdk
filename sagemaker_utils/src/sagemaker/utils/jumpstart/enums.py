from __future__ import absolute_import

from enum import Enum

class JumpStartModelType(str, Enum):
    """Enum class for JumpStart model type.

    OPEN_WEIGHTS: Publicly available models have open weights
    and are onboarded and maintained by JumpStart.
    PROPRIETARY: Proprietary models from third-party providers do not have open weights.
    You must subscribe to proprietary models in AWS Marketplace before use.
    """

    OPEN_WEIGHTS = "open_weights"
    PROPRIETARY = "proprietary"

class JumpStartScriptScope(str, Enum):
    """Enum class for JumpStart script scopes."""

    INFERENCE = "inference"
    TRAINING = "training"