# Early pydantic compatibility check - must happen before any pydantic imports
try:
    from sagemaker.core._pydantic_compat import check_pydantic_compatibility
    check_pydantic_compatibility()
except ImportError as e:
    if "pydantic" in str(e).lower():
        raise
    # If it's a different ImportError (e.g., module not found for _pydantic_compat itself),
    # let it pass and fail later with a more standard error
    pass

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
