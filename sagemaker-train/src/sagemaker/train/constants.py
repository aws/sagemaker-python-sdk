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
"""
DEPRECATED: This module has been moved to sagemaker.core.training.constants

This is a backward compatibility shim. Please update your imports to:
    from sagemaker.core.training.constants import ...
"""
from __future__ import absolute_import

import os

SM_CODE = "code"
SM_CODE_CONTAINER_PATH = "/opt/ml/input/data/code"

SM_DRIVERS = "sm_drivers"
SM_DRIVERS_CONTAINER_PATH = "/opt/ml/input/data/sm_drivers"
SM_DRIVERS_LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "container_drivers"
)

SOURCE_CODE_JSON = "sourcecode.json"
DISTRIBUTED_JSON = "distributed.json"
TRAIN_SCRIPT = "sm_train.sh"

DEFAULT_CONTAINER_ENTRYPOINT = ["/bin/bash"]
DEFAULT_CONTAINER_ARGUMENTS = [
    "-c",
    f"chmod +x {SM_DRIVERS_CONTAINER_PATH}/{TRAIN_SCRIPT} "
    + f"&& {SM_DRIVERS_CONTAINER_PATH}/{TRAIN_SCRIPT}",
]

HUB_NAME = "SageMakerPublicHub"

# Allowed reward model IDs for RLAIF trainer
ALLOWED_REWARD_MODEL_IDS = [
    "openai.gpt-oss-120b-1:0",
    "openai.gpt-oss-20b-1:0", 
    "qwen.qwen3-32b-v1:0",
    "qwen.qwen3-coder-30b-a3b-v1:0"
]
