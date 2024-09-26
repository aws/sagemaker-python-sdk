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
"""Holds constants used for interpreting MLflow models."""
from __future__ import absolute_import

DEFAULT_FW_USED_FOR_DEFAULT_IMAGE = "pytorch"
DEFAULT_PYTORCH_VERSION = {
    "py38": "1.12.1",
    "py39": "1.13.1",
    "py310": "2.2.0",
}
MODEL_PACKAGE_ARN_REGEX = (
    r"^arn:aws:sagemaker:[a-z0-9\-]+:[0-9]{12}:model-package\/(.*?)(?:/(\d+))?$"
)
MLFLOW_RUN_ID_REGEX = r"^runs:/[a-zA-Z0-9]+(/[a-zA-Z0-9\-_\.]*)+$"
MLFLOW_REGISTRY_PATH_REGEX = r"^models:/[a-zA-Z0-9\-_\.]+[@/]?[a-zA-Z0-9\-_\.][/a-zA-Z0-9\-_\.]*$"
S3_PATH_REGEX = r"^s3:\/\/[a-zA-Z0-9\-_\.]+(?:\/[a-zA-Z0-9\-_\/\.]*)?$"
MLFLOW_TRACKING_ARN = "MLFLOW_TRACKING_ARN"
MLFLOW_MODEL_PATH = "MLFLOW_MODEL_PATH"
MLFLOW_METADATA_FILE = "MLmodel"
MLFLOW_PIP_DEPENDENCY_FILE = "requirements.txt"
MLFLOW_PYFUNC = "python_function"
MLFLOW_FLAVOR_TO_PYTHON_PACKAGE_MAP = {
    "sklearn": "scikit-learn",
    "pytorch": "torch",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "h2o": "h2o",
    "spark": "pyspark",
    "onnx": "onnxruntime",
}
TENSORFLOW_SAVED_MODEL_NAME = "saved_model.pb"
FLAVORS_WITH_FRAMEWORK_SPECIFIC_DLC_SUPPORT = {
    "sklearn": "sklearn",
    "pytorch": "pytorch",
    "xgboost": "xgboost",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
}
FLAVORS_DEFAULT_WITH_TF_SERVING = ["keras", "tensorflow"]
