"""SageMaker Core package for low-level resource management and SDK foundations."""

from sagemaker.core.utils.utils import enable_textual_rich_console_and_traceback
from sagemaker.core.deprecations import register_removed_module_finder


enable_textual_rich_console_and_traceback()

# Install the meta-path finder that gives actionable migration guidance for v2
# modules removed in v3. sagemaker-core is the universal dependency of every v3
# package, so registering here guarantees the finder is active whenever the SDK
# is used (the namespace package __init__ also registers it for the cold
# first-import case).
register_removed_module_finder()

# Job management
from sagemaker.core.job import _Job  # noqa: F401
from sagemaker.core.processing import (  # noqa: F401
    Processor,
    ScriptProcessor,
    FrameworkProcessor,
)
from sagemaker.core.transformer import Transformer  # noqa: F401

# Partner App
from sagemaker.core.partner_app.auth_provider import PartnerAppAuthProvider  # noqa: F401

# Attribution
from sagemaker.core.telemetry.attribution import Attribution, set_attribution  # noqa: F401

# Note: HyperparameterTuner and WarmStartTypes are in sagemaker.train.tuner
# They are not re-exported from core to avoid circular dependencies
# No-op line to trigger CI for a Codecov verification (DO NOT MERGE).
