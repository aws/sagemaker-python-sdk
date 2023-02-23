# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying athis file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module configures the default values for SageMaker Python SDK."""

from __future__ import absolute_import
from sagemaker.config.config import SageMakerConfig  # noqa: F401
from sagemaker.config.config_schema import (  # noqa: F401
    SECURITY_GROUP_IDS,
    SUBNETS,
    ENABLE_NETWORK_ISOLATION,
    VOLUME_KMS_KEY_ID,
    KMS_KEY_ID,
    ROLE_ARN,
    EXECUTION_ROLE_ARN,
    CLUSTER_ROLE_ARN,
    VPC_CONFIG,
    OUTPUT_DATA_CONFIG,
    AUTO_ML_JOB_CONFIG,
    ASYNC_INFERENCE_CONFIG,
    OUTPUT_CONFIG,
    PROCESSING_OUTPUT_CONFIG,
    CLUSTER_CONFIG,
    NETWORK_CONFIG,
    CORE_DUMP_CONFIG,
    DATA_CAPTURE_CONFIG,
    MONITORING_OUTPUT_CONFIG,
    RESOURCE_CONFIG,
    SCHEMA_VERSION,
    DATASET_DEFINITION,
    ATHENA_DATASET_DEFINITION,
    REDSHIFT_DATASET_DEFINITION,
    MONITORING_JOB_DEFINITION,
    SAGEMAKER,
    PYTHON_SDK,
    MODULES,
    OFFLINE_STORE_CONFIG,
    ONLINE_STORE_CONFIG,
    S3_STORAGE_CONFIG,
    SECURITY_CONFIG,
    TRANSFORM_JOB_DEFINITION,
    MONITORING_SCHEDULE_CONFIG,
    MONITORING_RESOURCES,
    PROCESSING_RESOURCES,
    PRODUCTION_VARIANTS,
    SHADOW_PRODUCTION_VARIANTS,
    TRANSFORM_OUTPUT,
    TRANSFORM_RESOURCES,
    VALIDATION_ROLE,
    VALIDATION_SPECIFICATION,
    VALIDATION_PROFILES,
    PROCESSING_INPUTS,
    FEATURE_GROUP,
    EDGE_PACKAGING_JOB,
    TRAINING_JOB,
    PROCESSING_JOB,
    MODEL_PACKAGE,
    MODEL,
    MONITORING_SCHEDULE,
    ENDPOINT_CONFIG,
    AUTO_ML,
    COMPILATION_JOB,
    PIPELINE,
    TRANSFORM_JOB,
    ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION,
    TAGS,
    KEY,
    VALUE,
)
