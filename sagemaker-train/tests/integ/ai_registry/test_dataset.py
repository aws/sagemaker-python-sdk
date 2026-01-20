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

"""Integration tests for DataSet."""
import os
import time

import pytest
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique
from sagemaker.ai_registry.air_constants import HubContentStatus
from unittest.mock import patch


@pytest.mark.serial
class TestDataSetIntegration:
    """Integration tests for DataSet operations."""

    def test_create_dataset_from_local_file(self, unique_name, sample_jsonl_file, cleanup_list):
        """Test creating dataset from local file."""
        dataset = DataSet.create(
            name=unique_name,
            source=sample_jsonl_file,
            customization_technique=CustomizationTechnique.SFT,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name
        assert dataset.arn is not None
        assert dataset.version is not None
        assert dataset.customization_technique == CustomizationTechnique.SFT

    def test_create_dataset_from_s3_oss_sft(self, unique_name, test_bucket, cleanup_list):
        """Test creating SFT dataset from S3 URI."""
        s3_uri = f"s3://{test_bucket}/test_datasets/OSS/oss_sft_train.jsonl"
        dataset = DataSet.create(
            name=unique_name,
            source=s3_uri,
            customization_technique=CustomizationTechnique.SFT,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name
        assert dataset.customization_technique == CustomizationTechnique.SFT

    def test_create_dataset_from_s3_oss_rlvr(self, unique_name, test_bucket, cleanup_list):
        """Test creating RLVR dataset from S3 URI."""
        s3_uri = f"s3://{test_bucket}/test_datasets/OSS/oss_rlvr_train.jsonl"
        dataset = DataSet.create(
            name=unique_name,
            source=s3_uri,
            customization_technique=CustomizationTechnique.RLVR,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name
        assert dataset.customization_technique == CustomizationTechnique.RLVR

    def test_create_dataset_from_s3_oss_dpo(self, unique_name, test_bucket, cleanup_list):
        """Test creating RLVR dataset from S3 URI."""
        s3_uri = f"s3://{test_bucket}/test_datasets/OSS/oss_dpo_train.jsonl"
        dataset = DataSet.create(
            name=unique_name,
            source=s3_uri,
            customization_technique=CustomizationTechnique.DPO,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name
        assert dataset.customization_technique == CustomizationTechnique.DPO

    def test_create_dataset_from_s3_nova_sft(self, unique_name, test_bucket, cleanup_list):
        """Test creating RLVR dataset from S3 URI."""
        s3_uri = f"s3://{test_bucket}/test_datasets/Nova/nova_sft_train.jsonl"
        dataset = DataSet.create(
            name=unique_name,
            source=s3_uri,
            customization_technique=CustomizationTechnique.SFT,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name
        assert dataset.customization_technique == CustomizationTechnique.SFT

    def test_create_dataset_from_s3_nova_dpo(self, unique_name, test_bucket, cleanup_list):
        """Test creating RLVR dataset from S3 URI."""
        s3_uri = f"s3://{test_bucket}/test_datasets/Nova/nova_dpo_train.jsonl"
        dataset = DataSet.create(
            name=unique_name,
            source=s3_uri,
            customization_technique=CustomizationTechnique.DPO,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name
        assert dataset.customization_technique == CustomizationTechnique.DPO

    def test_create_dataset_from_s3_nova_rft(self, unique_name, test_bucket, cleanup_list):
        """Test creating RLVR dataset from S3 URI."""
        s3_uri = f"s3://{test_bucket}/test_datasets/Nova/nova_rft_train.jsonl"
        dataset = DataSet.create(
            name=unique_name,
            source=s3_uri,
            customization_technique=CustomizationTechnique.RLVR,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name
        assert dataset.customization_technique == CustomizationTechnique.RLVR

    def test_create_dataset_from_s3_nova_eval(self, unique_name, test_bucket, cleanup_list):
        """Test creating RLVR dataset from S3 URI."""
        s3_uri = f"s3://{test_bucket}/test_datasets/Nova/nova_eval.jsonl"
        dataset = DataSet.create(
            name=unique_name,
            source=s3_uri,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name

    def test_get_dataset(self, unique_name, sample_jsonl_file):
        """Test retrieving dataset by name."""
        created = DataSet.create(name=unique_name, source=sample_jsonl_file, wait=False)
        retrieved = DataSet.get(unique_name)
        assert retrieved.name == created.name
        assert retrieved.arn == created.arn

    def test_get_all_datasets(self):
        """Test listing all datasets."""
        datasets = list(DataSet.get_all(max_results=5))
        assert isinstance(datasets, list)

    def test_dataset_refresh(self, unique_name, sample_jsonl_file):
        """Test refreshing dataset status."""
        dataset = DataSet.create(name=unique_name, source=sample_jsonl_file, wait=False)
        dataset.refresh()
        time.sleep(3)
        assert dataset.status in [HubContentStatus.IMPORTING.value, HubContentStatus.AVAILABLE.value]

    def test_dataset_get_versions(self, unique_name, sample_jsonl_file):
        """Test getting dataset versions."""
        dataset = DataSet.create(name=unique_name, source=sample_jsonl_file, wait=False)
        versions = dataset.get_versions()
        assert len(versions) >= 1
        assert all(isinstance(v, DataSet) for v in versions)

    def test_dataset_delete(self, unique_name, sample_jsonl_file):
        """Test deleting dataset."""
        dataset = DataSet.create(name=unique_name, source=sample_jsonl_file, wait=False)
        result = dataset.delete()
        assert result is True

    def test_dataset_delete_by_name(self, unique_name, sample_jsonl_file):
        """Test deleting dataset by name."""
        DataSet.create(name=unique_name, source=sample_jsonl_file, wait=False)
        result = DataSet.delete_by_name(unique_name)
        assert result is True

    def test_dataset_wait(self, unique_name, sample_jsonl_file, cleanup_list):
        """Test waiting for dataset to be available."""
        dataset = DataSet.create(name=unique_name, source=sample_jsonl_file, wait=True)
        time.sleep(3)
        cleanup_list.append(dataset)
        assert dataset.status == HubContentStatus.AVAILABLE.value

    def test_create_dataset_version(self, unique_name, sample_jsonl_file, cleanup_list):
        """Test creating new dataset version."""
        dataset = DataSet.create(name=unique_name, source=sample_jsonl_file, wait=False)
        result = dataset.create_version(sample_jsonl_file)
        cleanup_list.append(cleanup_list)
        assert result is True

    def test_dataset_validation_invalid_extension(self, unique_name):
        """Test dataset validation with invalid file extension."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            DataSet._validate_dataset_file("test.txt")

    def test_create_dataset_with_invalid_format_s3(self, unique_name, test_bucket):
        """Test creating dataset from S3 with invalid format fails."""
        # This would require an actual invalid file in S3, so we'll mock it
        with patch('sagemaker.ai_registry.dataset.AIRHub.download_from_s3'), \
             patch('sagemaker.ai_registry.dataset.DataSet._validate_dataset_format', side_effect=ValueError("Invalid format")):
            with pytest.raises(ValueError, match="Invalid format"):
                DataSet.create(
                    name=unique_name,
                    source=f"s3://{test_bucket}/invalid_file.jsonl",
                    wait=False
                )

    def test_create_dataset_with_invalid_format_local(self, unique_name):
        """Test creating dataset from local file with invalid format fails."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jsonl', mode='w', delete=False) as f:
            f.write("invalid content")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Unable to detect format"):
                    DataSet.create(
                        name=unique_name,
                        source=f.name,
                        wait=False
                    )
            finally:
                os.unlink(f.name)

    def test_dataset_validation_large_file(self, unique_name):
        """Test dataset validation with oversized file."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.write(b'x' * (1024 * 1024 * 1024 + 1))  # > 1GB
            f.flush()
            with pytest.raises(ValueError, match="exceeds maximum allowed size"):
                DataSet._validate_dataset_file(f.name)
            os.unlink(f.name)

    def test_dataset_with_description(self, unique_name, sample_jsonl_file, cleanup_list):
        """Test creating dataset with description."""
        description = "Test dataset description"
        dataset = DataSet.create(
            name=unique_name,
            source=sample_jsonl_file,
            description=description,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.description is not None

    def test_dataset_with_tags(self, unique_name, sample_jsonl_file, cleanup_list):
        """Test creating dataset with custom tags."""
        tags = [("env", "test"), ("team", "ml")]
        dataset = DataSet.create(
            name=unique_name,
            source=sample_jsonl_file,
            tags=tags,
            wait=False
        )
        cleanup_list.append(dataset)
        assert dataset.name == unique_name

    def test_dataset_format_validation_success(self, unique_name, sample_jsonl_file):
        """Test dataset format validation succeeds for valid files."""
        # Should not raise any exception for valid JSONL file
        DataSet._validate_dataset_format(sample_jsonl_file)

    def test_dataset_format_validation_failure_invalid_format(self, unique_name):
        """Test dataset format validation fails for invalid format."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jsonl', mode='w', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            with pytest.raises(ValueError, match="Unable to detect format"):
                DataSet._validate_dataset_format(f.name)
            os.unlink(f.name)

    def test_dataset_format_validation_failure_empty_file(self, unique_name):
        """Test dataset format validation fails for empty files."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.flush()  # Create empty file
            with pytest.raises(ValueError, match="Unable to detect format"):
                DataSet._validate_dataset_format(f.name)
            os.unlink(f.name)

