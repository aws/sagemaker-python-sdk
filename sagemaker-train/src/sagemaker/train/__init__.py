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
"""SageMaker Python SDK Train Module."""
from __future__ import absolute_import

# Lazy imports to avoid circular dependencies
# Session and get_execution_role are available from sagemaker.core.helper.session_helper
# Import them directly from there if needed, or use lazy import pattern

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "Session":
        from sagemaker.core.helper.session_helper import Session
        return Session
    elif name == "get_execution_role":
        from sagemaker.core.helper.session_helper import get_execution_role
        return get_execution_role
    elif name == "ModelTrainer":
        from sagemaker.train.model_trainer import ModelTrainer
        return ModelTrainer
    elif name == "logger":
        from sagemaker.core.utils.utils import logger
        return logger
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
