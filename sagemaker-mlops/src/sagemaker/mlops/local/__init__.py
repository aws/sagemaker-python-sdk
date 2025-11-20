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
"""Local pipeline execution for SageMaker MLOps."""
from __future__ import absolute_import

from sagemaker.mlops.local.local_pipeline_session import LocalPipelineSession  # noqa: F401

# Pipeline execution is now in MLOps
# For backward compatibility, users should update imports:
# OLD: from sagemaker.core.local import LocalSession
# NEW: from sagemaker.core.local import LocalSession (for jobs)
#      from sagemaker.mlops.local import LocalPipelineSession (for pipelines)

__all__ = ["LocalPipelineSession"]
