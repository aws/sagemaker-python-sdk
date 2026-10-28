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
"""Unit tests for the DatasetFileFormat enum and Custom-format handling."""

from unittest.mock import patch

from sagemaker.ai_registry.air_constants import DatasetFileFormat
from sagemaker.ai_registry.dataset import DataSet


class TestDatasetFileFormatEnum:
    """The DatasetFileFormat enum backs the valid dataset_file_format values."""

    def test_custom_is_a_member(self):
        assert DatasetFileFormat.CUSTOM.value == "Custom"

    def test_str_enum_compares_equal_to_wire_value(self):
        # str-based enum so members serialize/compare as their wire value.
        assert DatasetFileFormat.HF_PROMPT_COMPLETION == "HF_Prompt_Completion"

    def test_all_expected_formats_present(self):
        values = {fmt.value for fmt in DatasetFileFormat}
        assert values == {
            "Bedrock_Converse",
            "HF_Prompt_Completion",
            "HF_Preference",
            "Verl",
            "Nova_PPO",
            "Nova_CPT",
            "OpenAI_Messages",
            "SageMaker_Eval",
            "Nova_RLVR",
            "Custom",
        }

    def test_member_serializes_as_wire_value(self):
        # The hub content document is built with json.dumps; a str-enum member
        # must serialize as its plain wire value, not "DatasetFileFormat.CUSTOM".
        import json

        assert json.dumps(DatasetFileFormat.CUSTOM) == '"Custom"'


class TestValidateDatasetFormatCustom:
    """Custom skips validation entirely; known formats still run the detector."""

    @patch("sagemaker.ai_registry.dataset.DatasetFormatDetector")
    def test_custom_skips_detector(self, mock_detector_class):
        # Custom must not even construct/call the detector.
        DataSet._validate_dataset_format("/path/to/anything.jsonl", "Custom")
        mock_detector_class.assert_not_called()

    @patch("sagemaker.ai_registry.dataset.DatasetFormatDetector")
    def test_custom_enum_member_skips_detector(self, mock_detector_class):
        # The docstring promises a DatasetFileFormat member is accepted too.
        DataSet._validate_dataset_format("/path/to/anything.jsonl", DatasetFileFormat.CUSTOM)
        mock_detector_class.assert_not_called()

    @patch("sagemaker.ai_registry.dataset.DatasetFormatDetector")
    def test_known_format_runs_detector(self, mock_detector_class):
        mock_detector = mock_detector_class.return_value
        mock_detector.validate_dataset.return_value = "genqa"
        DataSet._validate_dataset_format("/path/to/file.jsonl", "SageMaker_Eval")
        mock_detector.validate_dataset.assert_called_once_with("/path/to/file.jsonl")

    @patch("sagemaker.ai_registry.dataset.DatasetFormatDetector")
    def test_none_format_runs_detector(self, mock_detector_class):
        # None (no declared format) still validates against known schemas.
        mock_detector = mock_detector_class.return_value
        mock_detector.validate_dataset.return_value = "genqa"
        DataSet._validate_dataset_format("/path/to/file.jsonl")
        mock_detector.validate_dataset.assert_called_once_with("/path/to/file.jsonl")
