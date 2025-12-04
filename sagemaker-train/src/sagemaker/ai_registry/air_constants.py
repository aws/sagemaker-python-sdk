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
"""Constants for AI Registry Hub operations."""

from enum import Enum

# Hub configuration
AI_REGISTRY_HUB_NAME = "sdk-test-hub"
AIR_DEFAULT_PAGE_SIZE = 10
AIR_HUB_CONTENT_DEFAULT_VERSION = "1.0.0"

# Dataset constants
DATASET_HUB_CONTENT_TYPE = "DataSet"
DATASET_HUB_CONTENT_SUBTYPE = "AWS/DataSets"
DATASET_DOCUMENT_SCHEMA_VERSION = "2.0.0"
DATASET_DEFAULT_METHOD = "generated"

# TODO: Fetch these from intelligent defaults rather than hardcoding
DATASET_DEFAULT_TYPE = "CUSTOMER_PROVIDED"
DATASET_DEFAULT_CONVERSATION_ID = "default-conversation"
DATASET_DEFAULT_CHECKPOINT_ID = "default-checkpoint"

# Evaluator constants
EVALUATOR_HUB_CONTENT_TYPE = "JsonDoc"
EVALUATOR_HUB_CONTENT_SUBTYPE = "AWS/Evaluator"
EVALUATOR_SPEC_HUB_CONTENT_SUBTYPE = "AWS/Specification"
EVALUATOR_DOCUMENT_SCHEMA_VERSION = "2.0.0"
EVALUATOR_DEFAULT_METHOD = "lambda"
EVALUATOR_DEFAULT_RUNTIME = "python3.9"
EVALUATOR_BYOCODE = "BYOCode"
EVALUATOR_BYOLAMBDA = "BYOLambda"

EVALUATOR_DEFAULT_S3_PREFIX = "evaluators"

# Dataset file validation constants
DATASET_MAX_FILE_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB in bytes
DATASET_SUPPORTED_EXTENSIONS = ['.jsonl']

# Evaluator types
REWARD_FUNCTION = "RewardFunction"
REWARD_PROMPT = "RewardPrompt"

# AWS Lambda constants
LAMBDA_ARN_PREFIX = "arn:aws:lambda:"

# Tag keys
TAG_KEY_METHOD = "method"
TAG_KEY_CUSTOMIZATION_TECHNIQUE = "customization_technique"
TAG_KEY_DOMAIN_ID = "@domain"

# Response keys
RESPONSE_KEY_HUB_CONTENT_NAME = "HubContentName"
RESPONSE_KEY_HUB_CONTENT_VERSION = "HubContentVersion"
RESPONSE_KEY_HUB_CONTENT_ARN = "HubContentArn"
RESPONSE_KEY_HUB_CONTENT_STATUS = "HubContentStatus"
RESPONSE_KEY_HUB_CONTENT_DOCUMENT = "HubContentDocument"
RESPONSE_KEY_HUB_CONTENT_DESCRIPTION = "HubContentDescription"
RESPONSE_KEY_HUB_CONTENT_SEARCH_KEYWORDS = "HubContentSearchKeywords"
RESPONSE_KEY_CREATION_TIME = "CreationTime"
RESPONSE_KEY_LAST_MODIFIED_TIME = "LastModifiedTime"
RESPONSE_KEY_FUNCTION_ARN = "FunctionArn"
RESPONSE_KEY_ITEMS = "items"
RESPONSE_KEY_NEXT_TOKEN = "next_token"

# Document keys
DOC_KEY_SUB_TYPE = "EvaluatorType"
DOC_KEY_JSON_CONTENT = "JsonContent"
DOC_KEY_REFERENCE = "Reference"
DOC_KEY_DATASET_S3_BUCKET = "DatasetS3Bucket"
DOC_KEY_DATASET_S3_PREFIX = "DatasetS3Prefix"

class HubContentStatus(Enum):
    """HubContent status enum."""

    AVAILABLE = "Available"
    IMPORTING = "Importing"
    DELETING = "Deleting"
    IMPORT_FAILED = "ImportFailed"
    DELETE_FAILED = "DeleteFailed"
