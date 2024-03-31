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

import os
import platform

from tests.integ import DATA_DIR

SERVE_IN_PROCESS_TIMEOUT = 5
SERVE_MODEL_PACKAGE_TIMEOUT = 10
SERVE_LOCAL_CONTAINER_TIMEOUT = 10
SERVE_SAGEMAKER_ENDPOINT_TIMEOUT = 15
SERVE_SAVE_TIMEOUT = 2

PYTHON_VERSION_IS_NOT_38 = platform.python_version_tuple()[1] != "8"
PYTHON_VERSION_IS_NOT_310 = platform.python_version_tuple()[1] != "10"

XGB_RESOURCE_DIR = os.path.join(DATA_DIR, "serve_resources", "xgboost")
PYTORCH_SQUEEZENET_RESOURCE_DIR = os.path.join(DATA_DIR, "serve_resources", "pytorch")
PYTORCH_SQUEEZENET_MLFLOW_RESOURCE_DIR = os.path.join(
    DATA_DIR, "serve_resources", "mlflow", "pytorch"
)
XGBOOST_MLFLOW_RESOURCE_DIR = os.path.join(DATA_DIR, "serve_resources", "mlflow", "xgboost")
TF_EFFICIENT_RESOURCE_DIR = os.path.join(DATA_DIR, "serve_resources", "tensorflow")
HF_DIR = os.path.join(DATA_DIR, "serve_resources", "hf")

BYOC_IMAGE_URI_TEMPLATE = "661407751302.dkr.ecr.{}.amazonaws.com/byoc-integ-test-images:{}"
