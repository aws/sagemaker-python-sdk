# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Module for deprecation abstractions."""
from __future__ import absolute_import

import importlib.abc
import logging
import sys
import warnings

logger = logging.getLogger(__name__)

V2_URL = "https://sagemaker.readthedocs.io/en/stable/v2.html"

# Migration guide for users moving from the v2 SDK to v3.
V3_MIGRATION_URL = "https://github.com/aws/sagemaker-python-sdk/blob/master/migration.md"

# Real top-level ``sagemaker.*`` names that ship in v3. The fallback finder must
# never intercept these -- even when a package (e.g. sagemaker-train) is simply
# not installed, its absence should surface the normal error, not a bogus
# "removed in v3" message.
_KNOWN_V3_TOPLEVEL = frozenset({"core", "train", "serve", "mlops", "lineage", "ai_registry"})


def _emit_removed_telemetry(module):
    """Best-effort telemetry that a removed v2 module was imported.

    Imported lazily and fully guarded so that telemetry can never delay or break
    the import/error path. See
    ``sagemaker.core.telemetry.telemetry_logging.emit_removed_interface_telemetry``
    for the time-bounded, opt-out-able implementation.
    """
    try:
        from sagemaker.core.telemetry.telemetry_logging import (
            emit_removed_interface_telemetry,
        )

        emit_removed_interface_telemetry(module)
    except Exception:  # pylint: disable=W0703
        logger.debug("Removed-interface telemetry hook failed; continuing.")


def raise_removed_in_v3(module, replacement=None, v3_import=None):
    """Warn and then raise an actionable error for a v2 module removed in v3.

    The v2 SDK exposed top-level modules (e.g. ``sagemaker.estimator``) that no
    longer exist in v3. Importing one would otherwise fail with a bare
    ``ModuleNotFoundError: No module named 'sagemaker.estimator'`` that gives the
    caller no path forward. This helper is called from lightweight "tombstone"
    modules that stand in for those removed names: it emits a
    ``DeprecationWarning`` and then raises a ``ModuleNotFoundError`` whose message
    names the v3 replacement and links the migration guide.

    Args:
        module (str): The removed v2 module path, e.g. ``"sagemaker.estimator"``.
        replacement (str): Human readable v3 replacement, e.g.
            ``"ModelTrainer in the sagemaker-train package"``. Optional.
        v3_import (str): The exact v3 import statement, e.g.
            ``"from sagemaker.train import ModelTrainer"``. Quoted verbatim so the
            caller can copy-paste it. Optional.

    Raises:
        ModuleNotFoundError: always, after emitting the deprecation warning.
    """
    msg = f"`{module}` was removed in the SageMaker Python SDK v3."
    if replacement:
        msg += f" Use {replacement}."
    if v3_import:
        msg += f" ({v3_import})"
    msg += f"\nSee {V3_MIGRATION_URL} for the migration guide."

    _emit_removed_telemetry(module)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    logger.warning(msg)
    raise ModuleNotFoundError(msg, name=module)


class _RemovedV2ModuleFinder(importlib.abc.MetaPathFinder):
    """Fallback finder that gives actionable guidance for removed v2 modules.

    Curated removals ship as explicit "tombstone" modules (e.g.
    ``sagemaker/estimator.py``) that raise a precise, per-module message. This
    finder is the *catch-all* for every other ``sagemaker.<name>`` that existed
    in v2 but was not individually tombstoned: instead of a bare
    ``ModuleNotFoundError: No module named 'sagemaker.foo'``, the caller gets a
    message that says the module is not available in v3 and points to the
    migration guide.

    It is registered by appending to ``sys.meta_path``, so it only runs *after*
    the normal import machinery has failed to locate the module. That ordering
    guarantees it never shadows:
      - real v3 packages (``sagemaker.core``, ``sagemaker.train``, ...), and
      - the curated tombstone modules, which resolve as ordinary files first.
    """

    def find_spec(self, fullname, path=None, target=None):
        """Intercept only genuinely-missing top-level ``sagemaker.<name>`` imports."""
        if not fullname.startswith("sagemaker."):
            return None
        leaf = fullname[len("sagemaker.") :]
        # Only guard top-level names; never touch real v3 subpackages.
        if "." in leaf or leaf in _KNOWN_V3_TOPLEVEL:
            return None

        msg = (
            f"`{fullname}` is not available in the SageMaker Python SDK v3. "
            "It may have been removed or moved to a new location."
            f"\nSee {V3_MIGRATION_URL} for the migration guide."
        )
        _emit_removed_telemetry(fullname)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        logger.warning(msg)
        raise ModuleNotFoundError(msg, name=fullname)


def register_removed_module_finder():
    """Install the fallback finder for removed v2 modules (idempotent).

    Appends a single ``_RemovedV2ModuleFinder`` to ``sys.meta_path`` so it acts
    as a last resort. Safe to call multiple times -- it installs at most one
    instance per process.
    """
    if any(isinstance(f, _RemovedV2ModuleFinder) for f in sys.meta_path):
        return
    sys.meta_path.append(_RemovedV2ModuleFinder())


def _warn(msg, sdk_version=None):
    """Generic warning raiser referencing V2

    Args:
        phrase: The phrase to include in the warning.
        sdk_version: the sdk version of removal of support.
    """
    _sdk_version = sdk_version if sdk_version is not None else "2"
    full_msg = f"{msg} in sagemaker>={_sdk_version}.\nSee: {V2_URL} for details."
    warnings.warn(full_msg, DeprecationWarning, stacklevel=2)
    logger.warning(full_msg)


def removed_warning(phrase, sdk_version=None):
    """Raise a warning for a no-op in sagemaker>=2

    Args:
        phrase: the prefix phrase of the warning message.
        sdk_version: the sdk version of removal of support.
    """
    _warn(f"{phrase} is a no-op", sdk_version)


def renamed_warning(phrase):
    """Raise a warning for a rename in sagemaker>=2

    Args:
        phrase: the prefix phrase of the warning message.
    """
    _warn(f"{phrase} has been renamed")


def deprecation_warn(name, date, msg=None):
    """Raise a warning for soon to be deprecated feature in sagemaker>=2

    Args:
        name (str): Name of the feature
        date (str): the date when the feature will be deprecated
        msg (str): the prefix phrase of the warning message.
    """
    _warn(f"{name} will be deprecated on {date}.{msg}")


def deprecation_warn_base(msg):
    """Raise a warning for soon to be deprecated feature in sagemaker>=2

    Args:
        msg (str): the warning message.
    """
    _warn(msg)


def deprecation_warning(date, msg=None):
    """Decorator for raising deprecation warning for a feature in sagemaker>=2

    Args:
        date (str): the date when the feature will be deprecated
        msg (str): the prefix phrase of the warning message.

    Usage:
        @deprecation_warning(msg="message", date="date")
        def sample_function():
            print("xxxx....")

        @deprecation_warning(msg="message", date="date")
        class SampleClass():
            def __init__(self):
                print("xxxx....")

    """

    def deprecate(obj):
        def wrapper(*args, **kwargs):
            deprecation_warn(obj.__name__, date, msg)
            return obj(*args, **kwargs)

        return wrapper

    return deprecate


def renamed_kwargs(old_name, new_name, value, kwargs):
    """Checks if the deprecated argument is in kwargs

    Raises warning, if present.

    Args:
        old_name: name of deprecated argument
        new_name: name of the new argument
        value: value associated with new name, if supplied
        kwargs: keyword arguments dict

    Returns:
        value of the keyword argument, if present
    """
    if old_name in kwargs:
        value = kwargs.get(old_name, value)
        kwargs[new_name] = value
        renamed_warning(old_name)
    return value


def removed_arg(name, arg):
    """Checks if the deprecated argument is populated.

    Raises warning, if not None.

    Args:
        name: name of deprecated argument
        arg: the argument to check
    """
    if arg is not None:
        removed_warning(name)


def removed_kwargs(name, kwargs):
    """Checks if the deprecated argument is in kwargs

    Raises warning, if present.

    Args:
        name: name of deprecated argument
        kwargs: keyword arguments dict
    """
    if name in kwargs:
        removed_warning(name)


def removed_function(name):
    """A no-op deprecated function factory."""

    def func(*args, **kwargs):  # pylint: disable=W0613
        removed_warning(f"The function {name}")

    return func


def deprecated(sdk_version=None):
    """Decorator for raising deprecated warning for a feature in sagemaker>=2

    Args:
        sdk_version (str): the sdk version of removal of support.

    Usage:
        @deprecated()
        def sample_function():
            print("xxxx....")

        @deprecated(sdk_version="2.66")
        class SampleClass():
            def __init__(self):
                print("xxxx....")

    """

    def deprecate(obj):
        def wrapper(*args, **kwargs):
            removed_warning(obj.__name__, sdk_version)
            return obj(*args, **kwargs)

        return wrapper

    return deprecate


def deprecated_function(func, name):
    """Wrap a function with a deprecation warning.

    Args:
        func: Function to wrap in a deprecation warning.
        name: The name that has been deprecated.

    Returns:
        The modified function
    """

    def deprecate(*args, **kwargs):
        renamed_warning(f"The {name}")
        return func(*args, **kwargs)

    return deprecate


def deprecated_serialize(instance, name):
    """Modifies a serializer instance serialize method.

    Args:
        instance: Instance to modify serialize method.
        name: The name that has been deprecated.

    Returns:
        The modified instance
    """
    instance.serialize = deprecated_function(instance.serialize, name)
    return instance


def deprecated_deserialize(instance, name):
    """Modifies a deserializer instance deserialize method.

    Args:
        instance: Instance to modify deserialize method.
        name: The name that has been deprecated.

    Returns:
        The modified instance
    """
    instance.deserialize = deprecated_function(instance.deserialize, name)
    return instance


def deprecated_class(cls, name):
    """Returns a class based on super class with a deprecation warning.

    Args:
        cls: The class to derive with a deprecation warning on __init__
        name: The name of the class.

    Returns:
        The modified class.
    """

    class DeprecatedClass(cls):
        """Provides a warning for the class name."""

        def __init__(self, *args, **kwargs):
            """Provides a warning for the class name."""
            renamed_warning(f"The class {name}")
            super(DeprecatedClass, self).__init__(*args, **kwargs)

    return DeprecatedClass
