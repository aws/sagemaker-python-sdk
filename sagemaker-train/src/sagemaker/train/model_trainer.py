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
"""ModelTrainer module.

This module provides the ModelTrainer class for configuring and launching
training jobs on SageMaker.

Note: This module re-exports public symbols from _model_trainer.
Additional exports should be added here as the full implementation requires.
"""
from __future__ import absolute_import

from sagemaker.train._model_trainer import ModelTrainer  # noqa: F401
from sagemaker.train._model_trainer import Mode  # noqa: F401
