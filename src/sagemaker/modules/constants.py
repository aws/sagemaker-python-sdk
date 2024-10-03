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
"""Constants module."""
from __future__ import absolute_import
import os

DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"

SOURCE_CODE_CONTAINER_PATH = "/opt/ml/input/data/code"

SM_CODE_CONTAINER_PATH = "/opt/ml/input/data/sm_code"
SM_CODE_LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
TRAIN_SCRIPT = "train.sh"

DEFAULT_CONTAINER_ENTRYPOINT = ["/bin/bash"]
DEFAULT_CONTAINER_ARGUMENTS = [
    "-c",
    f"chmod +x {SM_CODE_CONTAINER_PATH}/{TRAIN_SCRIPT} "
    + f"&& {SM_CODE_CONTAINER_PATH}/{TRAIN_SCRIPT}",
]
