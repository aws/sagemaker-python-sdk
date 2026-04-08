"""SageMaker Core package for low-level resource management and SDK foundations."""
from sagemaker.core.utils.utils import enable_textual_rich_console_and_traceback


enable_textual_rich_console_and_traceback()

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
