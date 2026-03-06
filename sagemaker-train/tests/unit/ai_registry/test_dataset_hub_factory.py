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

import os
import pytest
from unittest.mock import Mock, patch

from sagemaker.ai_registry.dataset_hub_factory import DataSetHubFactory
from sagemaker.ai_registry.dataset_transformation import DatasetFormat
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique
from sagemaker.ai_registry.air_constants import HubContentStatus


def _make_dataset(**overrides):
    """Helper to create a mock DataSet with sensible defaults."""
    ds = Mock()
    ds.name = overrides.get("name", "test-ds")
    ds.arn = overrides.get("arn", "arn:aws:sagemaker:us-east-1:123456789012:hub-content/test-ds")
    ds.version = overrides.get("version", "1.0.0")
    ds.status = overrides.get("status", HubContentStatus.AVAILABLE)
    ds.source = overrides.get("source", "s3://bucket/datasets/test-ds/data.jsonl")
    ds.description = overrides.get("description", "test")
    ds.customization_technique = overrides.get("customization_technique", CustomizationTechnique.SFT)
    return ds


class TestResolveDataset:
    def test_resolve_with_dataset_instance(self):
        ds = _make_dataset()
        result = DataSetHubFactory._resolve_dataset(ds)
        assert result is ds

    @patch("sagemaker.ai_registry.dataset.DataSet.get")
    def test_resolve_with_name_string(self, mock_get):
        mock_get.return_value = _make_dataset()
        result = DataSetHubFactory._resolve_dataset("my-dataset")
        mock_get.assert_called_once_with(name="my-dataset", sagemaker_session=None)
        assert result.name == "test-ds"

    @patch("sagemaker.ai_registry.dataset.DataSet.get")
    def test_resolve_with_session(self, mock_get):
        mock_session = Mock()
        mock_get.return_value = _make_dataset()
        DataSetHubFactory._resolve_dataset("my-dataset", sagemaker_session=mock_session)
        mock_get.assert_called_once_with(name="my-dataset", sagemaker_session=mock_session)


class TestDownloadToLocal:
    @patch("sagemaker.ai_registry.air_hub.AIRHub.download_from_s3")
    def test_download_creates_temp_file(self, mock_download):
        path = DataSetHubFactory._download_to_local("s3://bucket/data.jsonl")
        try:
            assert path.endswith(".jsonl")
            mock_download.assert_called_once_with("s3://bucket/data.jsonl", path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @patch("sagemaker.ai_registry.air_hub.AIRHub.download_from_s3")
    def test_download_default_suffix(self, mock_download):
        path = DataSetHubFactory._download_to_local("s3://bucket/data")
        try:
            assert path.endswith(".jsonl")
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestTransformDatasetValidation:
    @patch("sagemaker.ai_registry.dataset_transformation.DatasetTransformation")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_neither_source_nor_dataset_raises(self, mock_ds, mock_transform):
        with pytest.raises(ValueError, match="Exactly one of"):
            DataSetHubFactory.transform_dataset(name="test", target_format=DatasetFormat.GENQA)

    @patch("sagemaker.ai_registry.dataset_transformation.DatasetTransformation")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    def test_both_source_and_dataset_raises(self, mock_ds, mock_transform):
        with pytest.raises(ValueError, match="Exactly one of"):
            DataSetHubFactory.transform_dataset(
                name="test",
                target_format=DatasetFormat.GENQA,
                source="s3://bucket/data.jsonl",
                dataset=_make_dataset(),
            )


class TestTransformDatasetFromSource:
    @patch("sagemaker.ai_registry.dataset.DataSet")
    @patch("sagemaker.ai_registry.dataset_transformation.DatasetTransformation")
    def test_local_file_source(self, mock_transformation_cls, mock_dataset_cls):
        mock_transformation_cls.detect_format.return_value = DatasetFormat.OPENAI_CHAT
        mock_transformation_cls.transform_file.return_value = "/tmp/transformed.jsonl"
        mock_dataset_cls.create.return_value = _make_dataset()

        result = DataSetHubFactory.transform_dataset(
            name="new-ds",
            target_format=DatasetFormat.GENQA,
            source="/local/data.jsonl",
            customization_technique=CustomizationTechnique.SFT,
        )

        mock_transformation_cls.detect_format.assert_called_once_with("/local/data.jsonl")
        mock_transformation_cls.transform_file.assert_called_once_with(
            file_path="/local/data.jsonl",
            source_format=DatasetFormat.OPENAI_CHAT,
            target_format=DatasetFormat.GENQA,
        )
        mock_dataset_cls.create.assert_called_once_with(
            name="new-ds",
            source="/tmp/transformed.jsonl",
            customization_technique=CustomizationTechnique.SFT,
            sagemaker_session=None,
        )
        assert result.name == "test-ds"

    @patch("sagemaker.ai_registry.dataset.DataSet")
    @patch("sagemaker.ai_registry.dataset_transformation.DatasetTransformation")
    @patch.object(DataSetHubFactory, "_download_to_local")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_s3_source_downloads_and_cleans_up(
        self, mock_remove, mock_exists, mock_download, mock_transformation_cls, mock_dataset_cls
    ):
        mock_download.return_value = "/tmp/downloaded.jsonl"
        mock_transformation_cls.detect_format.return_value = DatasetFormat.HF_PROMPT_COMPLETION
        mock_transformation_cls.transform_file.return_value = "/tmp/transformed.jsonl"
        mock_dataset_cls.create.return_value = _make_dataset()

        result = DataSetHubFactory.transform_dataset(
            name="new-ds",
            target_format=DatasetFormat.GENQA,
            source="s3://bucket/data.jsonl",
        )

        mock_download.assert_called_once_with("s3://bucket/data.jsonl")
        mock_transformation_cls.detect_format.assert_called_once_with("/tmp/downloaded.jsonl")
        mock_remove.assert_called_once_with("/tmp/downloaded.jsonl")

    @patch("sagemaker.ai_registry.dataset_transformation.DatasetTransformation")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    @patch.object(DataSetHubFactory, "_download_to_local")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_s3_source_cleans_up_on_error(
        self, mock_remove, mock_exists, mock_download, mock_ds, mock_transformation_cls
    ):
        mock_download.return_value = "/tmp/downloaded.jsonl"
        mock_transformation_cls.detect_format.side_effect = ValueError("bad format")

        with pytest.raises(ValueError, match="bad format"):
            DataSetHubFactory.transform_dataset(
                name="new-ds",
                target_format=DatasetFormat.GENQA,
                source="s3://bucket/data.jsonl",
            )

        mock_remove.assert_called_once_with("/tmp/downloaded.jsonl")


class TestTransformDatasetFromExisting:
    @patch("sagemaker.ai_registry.dataset.DataSet")
    @patch("sagemaker.ai_registry.dataset_transformation.DatasetTransformation")
    @patch.object(DataSetHubFactory, "_download_to_local")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_existing_dataset_instance(
        self, mock_remove, mock_exists, mock_download, mock_transformation_cls, mock_dataset_cls
    ):
        ds = _make_dataset()
        mock_download.return_value = "/tmp/downloaded.jsonl"
        mock_transformation_cls.detect_format.return_value = DatasetFormat.VERL
        mock_transformation_cls.transform_file.return_value = "/tmp/transformed.jsonl"
        mock_dataset_cls.get.return_value = _make_dataset(version="2.0.0")

        result = DataSetHubFactory.transform_dataset(
            name="test-ds",
            target_format=DatasetFormat.GENQA,
            dataset=ds,
            customization_technique=CustomizationTechnique.DPO,
        )

        mock_download.assert_called_once_with(ds.source)
        mock_transformation_cls.transform_file.assert_called_once_with(
            file_path="/tmp/downloaded.jsonl",
            source_format=DatasetFormat.VERL,
            target_format=DatasetFormat.GENQA,
        )
        assert result.version == "2.0.0"

    @patch("sagemaker.ai_registry.dataset.DataSet")
    @patch("sagemaker.ai_registry.dataset_transformation.DatasetTransformation")
    @patch.object(DataSetHubFactory, "_download_to_local")
    @patch.object(DataSetHubFactory, "_resolve_dataset")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_existing_dataset_by_name(
        self,
        mock_remove,
        mock_exists,
        mock_resolve,
        mock_download,
        mock_transformation_cls,
        mock_dataset_cls,
    ):
        resolved_ds = _make_dataset()
        mock_resolve.return_value = resolved_ds
        mock_download.return_value = "/tmp/downloaded.jsonl"
        mock_transformation_cls.detect_format.return_value = DatasetFormat.OPENAI_CHAT
        mock_transformation_cls.transform_file.return_value = "/tmp/transformed.jsonl"
        mock_dataset_cls.get.return_value = _make_dataset(version="2.0.0")

        result = DataSetHubFactory.transform_dataset(
            name="test-ds",
            target_format=DatasetFormat.GENQA,
            dataset="test-ds",
        )

        mock_resolve.assert_called_once_with("test-ds", None)
        mock_download.assert_called_once_with(resolved_ds.source)

    @patch("sagemaker.ai_registry.dataset_transformation.DatasetTransformation")
    @patch("sagemaker.ai_registry.dataset.DataSet")
    @patch.object(DataSetHubFactory, "_download_to_local")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_existing_dataset_cleans_up_on_error(
        self, mock_remove, mock_exists, mock_download, mock_ds, mock_transformation_cls
    ):
        ds = _make_dataset()
        mock_download.return_value = "/tmp/downloaded.jsonl"
        mock_transformation_cls.detect_format.side_effect = ValueError("bad")

        with pytest.raises(ValueError, match="bad"):
            DataSetHubFactory.transform_dataset(
                name="test-ds",
                target_format=DatasetFormat.GENQA,
                dataset=ds,
            )

        mock_remove.assert_called_once_with("/tmp/downloaded.jsonl")


class TestStubMethods:
    def test_split_dataset_not_implemented(self):
        with pytest.raises(NotImplementedError):
            DataSetHubFactory.split_dataset(name="test")

    def test_score_dataset_not_implemented(self):
        with pytest.raises(NotImplementedError):
            DataSetHubFactory.score_dataset(name="test", scoring_params={})

    def test_generate_dataset_not_implemented(self):
        with pytest.raises(NotImplementedError):
            DataSetHubFactory.generate_dataset(name="test", generation_params={})
