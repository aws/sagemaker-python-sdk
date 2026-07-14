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

# Top-level ``sagemaker.<name>`` modules that existed in v2 but were removed in
# v3 (some relocated under ``sagemaker.core.*``). Derived from the v2 top-level
# module surface (the ``master-v2`` branch), minus names that still exist in v3.
# The fallback finder only emits migration guidance for THESE names, so a typo
# or hallucinated import (e.g. ``sagemaker.foobar``) gets a plain
# ``ModuleNotFoundError`` rather than a misleading "was removed" message. V2 is
# in maintenance, so this surface is effectively frozen.
_REMOVED_V2_MODULES = frozenset(
    {
        "_studio",
        "accept_types",
        "algorithm",
        "amazon",
        "amtviz",
        "analytics",
        "apiutils",
        "async_inference",
        "automl",
        "aws_batch",
        "base_deserializers",
        "base_predictor",
        "base_serializers",
        "batch_inference",
        "chainer",
        "clarify",
        "cli",
        "collection",
        "compute_resource_requirements",
        "config",
        "container_base_model",
        "content_types",
        "dataset_definition",
        "debugger",
        "deprecations",
        "deserializers",
        "djl_inference",
        "drift_check_baselines",
        "enums",
        "environment_variables",
        "estimator",
        "exceptions",
        "experiments",
        "explainer",
        "feature_store",
        "fw_utils",
        "git_utils",
        "huggingface",
        "hyperparameters",
        "image_uri_config",
        "image_uris",
        "inference_recommender",
        "inputs",
        "instance_group",
        "instance_types",
        "instance_types_gpu_info",
        "interactive_apps",
        "iterators",
        "job",
        "jumpstart",
        "lambda_helper",
        "local",
        "logs",
        "metadata_properties",
        "metric_definitions",
        "mlflow",
        "model",
        "model_card",
        "model_life_cycle",
        "model_metrics",
        "model_monitor",
        "model_uris",
        "modules",
        "multidatamodel",
        "mxnet",
        "network",
        "parameter",
        "partner_app",
        "payloads",
        "pipeline",
        "predictor",
        "predictor_async",
        "processing",
        "pytorch",
        "remote_function",
        "resource_requirements",
        "rl",
        "s3",
        "s3_utils",
        "script_uris",
        "serializer_utils",
        "serializers",
        "serverless",
        "session",
        "session_settings",
        "sklearn",
        "spark",
        "sparkml",
        "stabilityai",
        "telemetry",
        "tensorflow",
        "training_compiler",
        "transformer",
        "tuner",
        "user_agent",
        "utilities",
        "utils",
        "vpc_utils",
        "workflow",
        "wrangler",
        "xgboost",
    }
)

_DOCS_BASE = "https://sagemaker.readthedocs.io/en/stable/api/generated/"

# Curated, high-traffic removed v2 modules -> precise v3 guidance:
# (replacement, exact import, docs module path). The fallback finder uses this
# to emit a specific message (exact class + copy-pasteable import + docs link)
# for these names, and a generic "was removed" message for every other name in
# _REMOVED_V2_MODULES. All targets are verified importable in v3.
_TRAIN = ("from sagemaker.train import ModelTrainer", "sagemaker.train.model_trainer")
_SERVE = ("from sagemaker.serve import ModelBuilder", "sagemaker.serve.model_builder")
_CORE_RES = "sagemaker.core.resources"
_V3_REPLACEMENTS = {
    "estimator": ("`ModelTrainer`", _TRAIN[0], _TRAIN[1]),
    "algorithm": ("`ModelTrainer`", _TRAIN[0], _TRAIN[1]),
    "model": ("`ModelBuilder`", _SERVE[0], _SERVE[1]),
    "multidatamodel": ("`ModelBuilder`", _SERVE[0], _SERVE[1]),
    "predictor_async": ("`ModelBuilder` (async deploy)", _SERVE[0], _SERVE[1]),
    "predictor": ("the `Endpoint` resource", f"from {_CORE_RES} import Endpoint", _CORE_RES),
    "base_predictor": ("the `Endpoint` resource", f"from {_CORE_RES} import Endpoint", _CORE_RES),
    "transformer": (
        "the `TransformJob` resource",
        f"from {_CORE_RES} import TransformJob",
        _CORE_RES,
    ),
    "tuner": (
        "the `HyperParameterTuningJob` resource",
        f"from {_CORE_RES} import HyperParameterTuningJob",
        _CORE_RES,
    ),
    "processing": (
        "the `ProcessingJob` resource",
        f"from {_CORE_RES} import ProcessingJob",
        _CORE_RES,
    ),
    "automl": ("the `AutoMLJob` resource", f"from {_CORE_RES} import AutoMLJob", _CORE_RES),
    "pipeline": (
        "`Pipeline`",
        "from sagemaker.mlops.workflow.pipeline import Pipeline",
        "sagemaker.mlops.workflow.pipeline",
    ),
}


def raise_removed_in_v3(module, replacement=None, v3_import=None, v3_docs=None):
    """Warn and then raise an actionable error for a v2 module removed in v3.

    The v2 SDK exposed top-level modules (e.g. ``sagemaker.estimator``) that no
    longer exist in v3. Importing one would otherwise fail with a bare
    ``ModuleNotFoundError: No module named 'sagemaker.estimator'`` that gives the
    caller no path forward. This helper is called from ``_RemovedV2ModuleFinder``
    for removed names: it emits a ``DeprecationWarning`` and then raises a
    ``ModuleNotFoundError`` whose message names the exact v3 replacement, the
    import to copy-paste, and a direct link to that replacement's API docs (plus
    the migration guide).

    Args:
        module (str): The removed v2 module path, e.g. ``"sagemaker.estimator"``.
        replacement (str): Human readable v3 replacement, e.g. ``"ModelTrainer"``.
            Optional.
        v3_import (str): The exact v3 import statement, e.g.
            ``"from sagemaker.train import ModelTrainer"``. Quoted verbatim so the
            caller can copy-paste it. Optional.
        v3_docs (str): Direct URL to the v3 replacement's API documentation, e.g.
            the generated ``sagemaker.train.model_trainer`` page. Optional.

    Raises:
        ModuleNotFoundError: always, after emitting the deprecation warning.
    """
    msg = f"`{module}` was removed in the SageMaker Python SDK v3."
    if replacement:
        msg += f" Use {replacement}."
    if v3_import:
        msg += f" ({v3_import})"
    if v3_docs:
        msg += f"\nDocs: {v3_docs}"
    msg += f"\nSee {V3_MIGRATION_URL} for the migration guide."

    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # The raised ModuleNotFoundError below is the loud, authoritative signal
    # (it stops execution and carries the full message). Log at debug only, to
    # leave a breadcrumb for log-captured environments without duplicating the
    # message at WARNING level.
    logger.debug(msg)
    raise ModuleNotFoundError(msg, name=module)


class _RemovedV2ModuleFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that gives actionable guidance for removed v2 modules.

    A single hook handles all removed top-level ``sagemaker.<name>`` modules:
      - names in ``_V3_REPLACEMENTS`` get a **precise** message (exact v3 class,
        copy-pasteable import, and API-docs link),
      - other names in ``_REMOVED_V2_MODULES`` get a **generic** "was removed"
        message, and
      - any other name (typo, hallucinated import) falls through to Python's
        plain ``ModuleNotFoundError`` -- we never claim something "was removed"
        when it never existed.

    It is registered by *appending* to ``sys.meta_path``, so it only runs after
    the normal import machinery fails to locate the module. That ordering
    guarantees it never shadows a real module: if v3 ever ships a top-level
    module whose name matches a removed v2 one, the real module resolves first
    and this finder is never consulted for it.
    """

    def find_spec(self, fullname, path=None, target=None):
        """Emit guidance only for known removed v2 top-level ``sagemaker`` modules."""
        if not fullname.startswith("sagemaker."):
            return None
        leaf = fullname[len("sagemaker.") :]
        # Only guard top-level names; never touch real v3 subpackages.
        if "." in leaf or leaf in _KNOWN_V3_TOPLEVEL:
            return None
        # Only guard names that were actually v2 modules. Unknown names (typos,
        # hallucinated imports) fall through to a plain ModuleNotFoundError so we
        # never claim something "was removed" when it never existed.
        if leaf not in _REMOVED_V2_MODULES:
            return None

        if leaf in _V3_REPLACEMENTS:
            # Curated, high-traffic module -> precise guidance.
            replacement, v3_import, docs_module = _V3_REPLACEMENTS[leaf]
            raise_removed_in_v3(
                module=fullname,
                replacement=replacement,
                v3_import=v3_import,
                v3_docs=f"{_DOCS_BASE}{docs_module}.html",
            )

        # Other removed v2 module -> generic guidance.
        msg = (
            f"`{fullname}` was removed in the SageMaker Python SDK v3. "
            "It may have moved to a new location."
            f"\nSee {V3_MIGRATION_URL} for the migration guide."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        # See raise_removed_in_v3: the raised error is the loud signal; log at
        # debug to avoid duplicating the message at WARNING level.
        logger.debug(msg)
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
    logger.debug("Registered SageMaker v2 removed-module guidance finder on sys.meta_path.")


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
