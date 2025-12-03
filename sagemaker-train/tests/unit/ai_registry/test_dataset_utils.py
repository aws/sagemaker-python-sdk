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

"""Tests for dataset utilities."""

import json
import pytest

from sagemaker.ai_registry.dataset_utils import (
    CustomizationTechnique,
    DataSetMethod,
    DataSetHubContentDocument
)


class TestCustomizationTechnique:
    def test_customization_technique_values(self):
        """Test that CustomizationTechnique enum has expected values."""
        assert CustomizationTechnique.SFT == "sft"
        assert CustomizationTechnique.DPO == "dpo"
        assert CustomizationTechnique.RLVR == "rlvr"

    def test_customization_technique_string_conversion(self):
        """Test string conversion of CustomizationTechnique."""
        assert CustomizationTechnique.SFT.value == "sft"
        assert CustomizationTechnique.DPO.value == "dpo"
        assert CustomizationTechnique.RLVR.value == "rlvr"


class TestDataSetMethod:
    def test_dataset_method_values(self):
        """Test that DataSetMethod enum has expected values."""
        assert DataSetMethod.UPLOADED.value == "uploaded"
        assert DataSetMethod.GENERATED.value == "generated"


class TestDataSetHubContentDocument:
    def test_create_minimal_document(self):
        """Test creating document with minimal parameters."""
        doc = DataSetHubContentDocument()
        
        assert doc.dataset_type == "AGENT_GENERATED"
        assert doc.dataset_role_arn is None
        assert doc.dependencies == []

    def test_create_full_document(self):
        """Test creating document with all parameters."""
        doc = DataSetHubContentDocument(
            dataset_type="CUSTOMER_PROVIDED",
            dataset_role_arn="arn:aws:iam::123456789012:role/TestRole",
            dataset_s3_bucket="test-bucket",
            dataset_s3_prefix="datasets/test",
            dataset_context_s3_uri="s3://test-bucket/context",
            specification_arn="arn:aws:sagemaker:us-west-2:123456789012:specification/test",
            conversation_id="conv-123",
            conversation_checkpoint_id="checkpoint-456",
            dependencies=["dep1", "dep2"]
        )
        
        assert doc.dataset_type == "CUSTOMER_PROVIDED"
        assert doc.dataset_role_arn == "arn:aws:iam::123456789012:role/TestRole"
        assert doc.dataset_s3_bucket == "test-bucket"
        assert doc.dataset_s3_prefix == "datasets/test"
        assert doc.dataset_context_s3_uri == "s3://test-bucket/context"
        assert doc.specification_arn == "arn:aws:sagemaker:us-west-2:123456789012:specification/test"
        assert doc.conversation_id == "conv-123"
        assert doc.conversation_checkpoint_id == "checkpoint-456"
        assert doc.dependencies == ["dep1", "dep2"]

    def test_to_json_minimal(self):
        """Test JSON serialization with minimal parameters."""
        doc = DataSetHubContentDocument()
        json_str = doc.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["DatasetType"] == "AGENT_GENERATED"
        assert parsed["Dependencies"] == []
        assert "DatasetRoleArn" not in parsed

    def test_to_json_full(self):
        """Test JSON serialization with all parameters."""
        doc = DataSetHubContentDocument(
            dataset_type="CUSTOMER_PROVIDED",
            dataset_role_arn="arn:aws:iam::123456789012:role/TestRole",
            dataset_s3_bucket="test-bucket",
            dataset_s3_prefix="datasets/test",
            dataset_context_s3_uri="s3://test-bucket/context",
            specification_arn="arn:aws:sagemaker:us-west-2:123456789012:specification/test",
            conversation_id="conv-123",
            conversation_checkpoint_id="checkpoint-456",
            dependencies=["dep1", "dep2"]
        )
        
        json_str = doc.to_json()
        parsed = json.loads(json_str)
        
        expected_keys = {
            "DatasetType", "DatasetRoleArn", "DatasetS3Bucket", "DatasetS3Prefix",
            "DatasetContextS3Uri", "SpecificationArn", "ConversationId",
            "ConversationCheckpointId", "Dependencies"
        }
        
        assert set(parsed.keys()) == expected_keys
        assert parsed["DatasetType"] == "CUSTOMER_PROVIDED"
        assert parsed["DatasetRoleArn"] == "arn:aws:iam::123456789012:role/TestRole"
        assert parsed["DatasetS3Bucket"] == "test-bucket"
        assert parsed["DatasetS3Prefix"] == "datasets/test"
        assert parsed["DatasetContextS3Uri"] == "s3://test-bucket/context"
        assert parsed["SpecificationArn"] == "arn:aws:sagemaker:us-west-2:123456789012:specification/test"
        assert parsed["ConversationId"] == "conv-123"
        assert parsed["ConversationCheckpointId"] == "checkpoint-456"
        assert parsed["Dependencies"] == ["dep1", "dep2"]

    def test_to_json_partial(self):
        """Test JSON serialization with some parameters."""
        doc = DataSetHubContentDocument(
            dataset_type="CUSTOMER_PROVIDED",
            dataset_s3_bucket="test-bucket",
            dependencies=["dep1"]
        )
        
        json_str = doc.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["DatasetType"] == "CUSTOMER_PROVIDED"
        assert parsed["DatasetS3Bucket"] == "test-bucket"
        assert parsed["Dependencies"] == ["dep1"]
        
        # These should not be present since they were None
        excluded_keys = {
            "DatasetRoleArn", "DatasetS3Prefix", "DatasetContextS3Uri",
            "SpecificationArn", "ConversationId", "ConversationCheckpointId"
        }
        
        for key in excluded_keys:
            assert key not in parsed

    def test_to_json_empty_dependencies(self):
        """Test JSON serialization with empty dependencies list."""
        doc = DataSetHubContentDocument(dependencies=[])
        json_str = doc.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["Dependencies"] == []

    def test_to_json_none_dependencies(self):
        """Test JSON serialization with None dependencies (should default to empty list)."""
        doc = DataSetHubContentDocument(dependencies=None)
        json_str = doc.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["Dependencies"] == []
