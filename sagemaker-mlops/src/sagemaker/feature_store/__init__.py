# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Backward compatibility shim for sagemaker.feature_store.

This module re-exports everything from sagemaker.mlops.feature_store
and emits a DeprecationWarning to guide users to the new import path.
"""
import warnings

warnings.warn(
    "sagemaker.feature_store is deprecated, use sagemaker.mlops.feature_store instead",
    DeprecationWarning,
    stacklevel=2,
)

from sagemaker.mlops.feature_store import *  # noqa: F401, F403, E402
from sagemaker.mlops.feature_store import __all__  # noqa: F401, E402
