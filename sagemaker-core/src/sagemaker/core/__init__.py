"""SageMaker Core package for low-level resource management and SDK foundations."""

import logging as _logging

from sagemaker.core.utils.utils import enable_textual_rich_console_and_traceback
from sagemaker.core.deprecations import register_removed_module_finder

# A library must not hijack stdout or emit to an unconfigured root logger. Attach a
# NullHandler to the top-level "sagemaker" logger so SDK log records are discarded by
# default until the application configures logging (see #4387). sagemaker-core is the
# universal dependency of every v3 package, so installing it here covers all of them.
_sagemaker_root_logger = _logging.getLogger("sagemaker")
if not any(isinstance(_h, _logging.NullHandler) for _h in _sagemaker_root_logger.handlers):
    _sagemaker_root_logger.addHandler(_logging.NullHandler())

enable_textual_rich_console_and_traceback()  # opt-in; no-op unless SAGEMAKER_ENABLE_RICH_LOGGING is set

# Install the meta-path finder that gives actionable migration guidance for v2
# modules removed in v3. sagemaker-core is the universal dependency of every v3
# package, so registering here guarantees the finder is active whenever the SDK
# is used (the namespace package __init__ also registers it for the cold
# first-import case).
register_removed_module_finder()

# Job management
from sagemaker.core.job import _Job  # noqa: F401, E402
from sagemaker.core.processing import (  # noqa: F401, E402
    Processor,
    ScriptProcessor,
    FrameworkProcessor,
)
from sagemaker.core.transformer import Transformer  # noqa: F401, E402

# Partner App
from sagemaker.core.partner_app.auth_provider import PartnerAppAuthProvider  # noqa: F401, E402

# Attribution
from sagemaker.core.telemetry.attribution import Attribution, set_attribution  # noqa: F401, E402

# Note: HyperparameterTuner and WarmStartTypes are in sagemaker.train.tuner
# They are not re-exported from core to avoid circular dependencies
