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
"""SageMaker modules directory."""
from __future__ import absolute_import

from sagemaker.core.utils.utils import logger as sagemaker_core_logger
from sagemaker.core.helper.session_helper import Session, get_execution_role  # noqa: F401

logger = sagemaker_core_logger
