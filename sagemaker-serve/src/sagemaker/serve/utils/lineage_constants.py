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
"""Holds constants used for lineage support"""
from __future__ import absolute_import


LINEAGE_POLLER_INTERVAL_SECS = 15
LINEAGE_POLLER_MAX_TIMEOUT_SECS = 120
TRACKING_SERVER_ARN_REGEX = r"arn:(.*?):sagemaker:(.*?):(.*?):mlflow-tracking-server/(.*?)$"
TRACKING_SERVER_CREATION_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
MODEL_BUILDER_MLFLOW_MODEL_PATH_LINEAGE_ARTIFACT_TYPE = "ModelBuilderInputModelData"
MLFLOW_S3_PATH = "S3"
MLFLOW_MODEL_PACKAGE_PATH = "ModelPackage"
MLFLOW_RUN_ID = "MLflowRunId"
MLFLOW_LOCAL_PATH = "Local"
MLFLOW_REGISTRY_PATH = "MLflowRegistry"
ERROR = "Error"
CODE = "Code"
CONTRIBUTED_TO = "ContributedTo"
VALIDATION_EXCEPTION = "ValidationException"
