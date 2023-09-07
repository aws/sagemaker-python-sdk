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
from __future__ import absolute_import

from botocore.exceptions import WaiterError

from sagemaker.workflow.pipeline import _PipelineExecution


def wait_pipeline_execution(execution: _PipelineExecution, delay: int = 30, max_attempts: int = 60):
    try:
        execution.wait(delay=delay, max_attempts=max_attempts)
    except WaiterError:
        pass
