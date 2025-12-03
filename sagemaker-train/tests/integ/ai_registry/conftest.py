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

"""Fixtures for AI Registry integration tests."""

import os
import tempfile
import uuid
import zipfile
import pytest
import boto3

from sagemaker.ai_registry.air_utils import _get_default_bucket
from sagemaker.train.defaults import TrainDefaults


@pytest.fixture
def unique_name():
    """Generate unique name for testing."""
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_bucket():
    """Get test S3 bucket name."""
    return _get_default_bucket()


@pytest.fixture
def test_role():
    """Get test IAM role ARN."""
    return TrainDefaults.get_role()


@pytest.fixture
def sample_jsonl_file():
    """Create sample JSONL dataset file."""
    content = """{"prompt": "What is AI?", "completion": "AI is artificial intelligence."}
{"prompt": "What is ML?", "completion": "ML is machine learning."}
{"prompt": "What is DL?", "completion": "DL is deep learning."}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(content)
        f.flush()  # Ensure content is written to disk
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_lambda_code():
    """Create sample Lambda function code as zip."""
    code = '''import json
def lambda_handler(event, context):
    return {"statusCode": 200, "body": json.dumps({"score": 0.8})}
'''
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as zip_f:
        with zipfile.ZipFile(zip_f.name, 'w') as zf:
            zf.writestr('lambda_function.py', code)
        yield zip_f.name
    os.unlink(zip_f.name)


@pytest.fixture
def sample_prompt_file():
    """Create sample prompt file."""
    content = "Evaluate the response: {response}"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_hub_content_document():
    """Create sample hub content document."""
    from sagemaker.ai_registry.dataset_utils import DataSetHubContentDocument
    from sagemaker.ai_registry.air_constants import (
        DATASET_DEFAULT_TYPE, DATASET_DEFAULT_CONVERSATION_ID, DATASET_DEFAULT_CHECKPOINT_ID
    )
    
    document = DataSetHubContentDocument(
        dataset_s3_bucket=_get_default_bucket(),
        dataset_s3_prefix="test",
        dataset_context_s3_uri="\"\"",
        dataset_type=DATASET_DEFAULT_TYPE,
        dataset_role_arn=TrainDefaults.get_role(),
        conversation_id=DATASET_DEFAULT_CONVERSATION_ID,
        conversation_checkpoint_id=DATASET_DEFAULT_CHECKPOINT_ID,
        dependencies=[],
    )
    return document.to_json()


@pytest.fixture
def cleanup_list():
    """Track resources for cleanup."""
    resources = []
    yield resources
    for evaluator in resources:
        try:
            from sagemaker.ai_registry.air_hub import AIRHub
            AIRHub.delete_hub_content(
                hub_content_type=evaluator.hub_content_type,
                hub_content_name=evaluator.name,
                hub_content_version=evaluator.version
            )
        except Exception:
            pass
