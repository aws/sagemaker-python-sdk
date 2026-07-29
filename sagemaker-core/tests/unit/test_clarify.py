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
"""Unit tests for sagemaker.core.clarify module."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch
from sagemaker.core.clarify import (
    SegmentationConfig,
    TimeSeriesDataConfig,
    TimeSeriesJSONDatasetFormat,
    DatasetType,
)


class TestSegmentationConfig:
    """Test SegmentationConfig class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        config = SegmentationConfig(
            name_or_index="age",
            segments=[["[1, 4]", "(5, 6]"], ["(7, 9)"]],
            config_name="age_segments",
            display_aliases=["Young", "Middle", "Old"],
        )
        assert config.name_or_index == "age"
        assert len(config.segments) == 2
        assert config.config_name == "age_segments"
        assert len(config.display_aliases) == 3

    def test_init_with_integer_index(self):
        """Test initialization with integer index."""
        config = SegmentationConfig(name_or_index=0, segments=[["A", "B"], ["C"]])
        assert config.name_or_index == 0
        assert len(config.segments) == 2

    def test_init_without_optional_params(self):
        """Test initialization without optional parameters."""
        config = SegmentationConfig(name_or_index="category", segments=[["A", "B"]])
        assert config.config_name is None
        assert config.display_aliases is None

    def test_init_with_none_name_raises_error(self):
        """Test that None name_or_index raises ValueError."""
        with pytest.raises(ValueError, match="`name_or_index` cannot be None"):
            SegmentationConfig(name_or_index=None, segments=[["A"]])

    def test_init_with_invalid_segments_raises_error(self):
        """Test that invalid segments raise ValueError."""
        with pytest.raises(ValueError, match="`segments` must be a list of lists"):
            SegmentationConfig(name_or_index="test", segments="invalid")

    def test_init_with_empty_segments_raises_error(self):
        """Test that empty segments raise ValueError."""
        with pytest.raises(ValueError, match="`segments` must be a list of lists"):
            SegmentationConfig(name_or_index="test", segments=[])

    def test_init_with_wrong_display_aliases_count(self):
        """Test that wrong number of display aliases raises ValueError."""
        with pytest.raises(ValueError, match="Number of `display_aliases` must equal"):
            SegmentationConfig(
                name_or_index="test",
                segments=[["A"], ["B"]],
                display_aliases=["One"],  # Should be 2 or 3
            )

    def test_to_dict(self):
        """Test to_dict method."""
        config = SegmentationConfig(
            name_or_index="age",
            segments=[["[1, 4]"]],
            config_name="test_config",
            display_aliases=["Young", "Old"],
        )
        result = config.to_dict()
        assert result["name_or_index"] == "age"
        assert result["segments"] == [["[1, 4]"]]
        assert result["config_name"] == "test_config"
        assert result["display_aliases"] == ["Young", "Old"]


class TestTimeSeriesDataConfig:
    """Test TimeSeriesDataConfig class."""

    def test_init_with_string_params(self):
        """Test initialization with string parameters."""
        config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            related_time_series=["related1", "related2"],
            static_covariates=["static1"],
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        )
        data = config.get_time_series_data_config()
        assert data["target_time_series"] == "target"
        assert data["item_id"] == "id"
        assert data["timestamp"] == "time"
        assert data["related_time_series"] == ["related1", "related2"]
        assert data["static_covariates"] == ["static1"]
        assert data["dataset_format"] == "columns"

    def test_init_with_int_params(self):
        """Test initialization with integer parameters."""
        config = TimeSeriesDataConfig(
            target_time_series=1,
            item_id=2,
            timestamp=3,
            related_time_series=[4, 5],
            static_covariates=[6],
        )
        data = config.get_time_series_data_config()
        assert data["target_time_series"] == 1
        assert data["item_id"] == 2
        assert data["timestamp"] == 3
        assert data["related_time_series"] == [4, 5]

    def test_init_without_target_raises_error(self):
        """Test that missing target_time_series raises ValueError."""
        with pytest.raises(ValueError, match="Please provide a target time series"):
            TimeSeriesDataConfig(target_time_series=None, item_id="id", timestamp="time")

    def test_init_without_item_id_raises_error(self):
        """Test that missing item_id raises ValueError."""
        with pytest.raises(ValueError, match="Please provide an item id"):
            TimeSeriesDataConfig(target_time_series="target", item_id=None, timestamp="time")

    def test_init_without_timestamp_raises_error(self):
        """Test that missing timestamp raises ValueError."""
        with pytest.raises(ValueError, match="Please provide a timestamp"):
            TimeSeriesDataConfig(target_time_series="target", item_id="id", timestamp=None)

    def test_init_with_mixed_types_raises_error(self):
        """Test that mixed types raise ValueError."""
        with pytest.raises(ValueError, match="Please provide"):
            TimeSeriesDataConfig(
                target_time_series="target", item_id=1, timestamp="time"  # int instead of str
            )

    def test_init_with_invalid_related_time_series(self):
        """Test that invalid related_time_series raises ValueError."""
        with pytest.raises(ValueError, match="Please provide a list"):
            TimeSeriesDataConfig(
                target_time_series="target",
                item_id="id",
                timestamp="time",
                related_time_series="invalid",  # Should be list
            )

    def test_init_with_empty_strings_in_related_raises_error(self):
        """Test that empty strings in related_time_series raise ValueError."""
        with pytest.raises(ValueError, match="Please do not provide empty strings"):
            TimeSeriesDataConfig(
                target_time_series="target",
                item_id="id",
                timestamp="time",
                related_time_series=["valid", ""],
            )

    def test_init_with_dataset_format_for_int_raises_error(self):
        """Test that dataset_format with int params raises ValueError."""
        with pytest.raises(ValueError, match="Dataset format should only be provided"):
            TimeSeriesDataConfig(
                target_time_series=1,
                item_id=2,
                timestamp=3,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
            )

    def test_init_without_dataset_format_for_string_raises_error(self):
        """Test that missing dataset_format with string params raises ValueError."""
        with pytest.raises(ValueError, match="Please provide a valid dataset format"):
            TimeSeriesDataConfig(target_time_series="target", item_id="id", timestamp="time")

    def test_get_time_series_data_config_returns_copy(self):
        """Test that get_time_series_data_config returns a copy."""
        config = TimeSeriesDataConfig(target_time_series=1, item_id=2, timestamp=3)
        data1 = config.get_time_series_data_config()
        data2 = config.get_time_series_data_config()
        assert data1 is not data2
        assert data1 == data2


class TestDatasetType:
    """Test DatasetType enum."""

    def test_dataset_type_values(self):
        """Test DatasetType enum values."""
        assert DatasetType.TEXTCSV.value == "text/csv"
        assert DatasetType.JSONLINES.value == "application/jsonlines"
        assert DatasetType.JSON.value == "application/json"
        assert DatasetType.PARQUET.value == "application/x-parquet"
        assert DatasetType.IMAGE.value == "application/x-image"


class TestTimeSeriesJSONDatasetFormat:
    """Test TimeSeriesJSONDatasetFormat enum."""

    def test_format_values(self):
        """Test TimeSeriesJSONDatasetFormat enum values."""
        assert TimeSeriesJSONDatasetFormat.COLUMNS.value == "columns"
        assert TimeSeriesJSONDatasetFormat.ITEM_RECORDS.value == "item_records"
        assert TimeSeriesJSONDatasetFormat.TIMESTAMP_RECORDS.value == "timestamp_records"


class TestSegmentationConfigExtended:
    """Extended test cases for SegmentationConfig."""

    def test_to_dict_without_optional_fields(self):
        """Test to_dict without optional fields."""
        config = SegmentationConfig(name_or_index="category", segments=[["A", "B"]])
        result = config.to_dict()

        assert "name_or_index" in result
        assert "segments" in result
        assert "config_name" not in result
        assert "display_aliases" not in result

    def test_segments_with_intervals(self):
        """Test segments with interval notation."""
        config = SegmentationConfig(
            name_or_index="age", segments=[["[0, 18]"], ["(18, 65]"], ["(65, 100]"]]
        )

        assert len(config.segments) == 3
        assert config.segments[0] == ["[0, 18]"]

    def test_segments_with_multiple_intervals(self):
        """Test segments with multiple intervals."""
        config = SegmentationConfig(
            name_or_index="score", segments=[["[0, 50]", "(50, 75]"], ["(75, 100]"]]
        )

        assert len(config.segments) == 2
        assert len(config.segments[0]) == 2

    def test_display_aliases_equal_to_segments(self):
        """Test display aliases equal to number of segments."""
        config = SegmentationConfig(
            name_or_index="category",
            segments=[["A"], ["B"]],
            display_aliases=["Group A", "Group B"],
        )

        assert len(config.display_aliases) == 2

    def test_display_aliases_with_default_segment(self):
        """Test display aliases including default segment."""
        config = SegmentationConfig(
            name_or_index="category",
            segments=[["A"], ["B"]],
            display_aliases=["Group A", "Group B", "Others"],
        )

        assert len(config.display_aliases) == 3


class TestTimeSeriesDataConfigExtended:
    """Extended test cases for TimeSeriesDataConfig."""

    def test_with_all_optional_params_string(self):
        """Test with all optional parameters as strings."""
        config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            related_time_series=["related1", "related2", "related3"],
            static_covariates=["static1", "static2"],
            dataset_format=TimeSeriesJSONDatasetFormat.ITEM_RECORDS,
        )

        data = config.get_time_series_data_config()
        assert len(data["related_time_series"]) == 3
        assert len(data["static_covariates"]) == 2
        assert data["dataset_format"] == "item_records"

    def test_with_all_optional_params_int(self):
        """Test with all optional parameters as integers."""
        config = TimeSeriesDataConfig(
            target_time_series=1,
            item_id=2,
            timestamp=3,
            related_time_series=[4, 5],
            static_covariates=[6, 7],
        )

        data = config.get_time_series_data_config()
        assert data["related_time_series"] == [4, 5]
        assert data["static_covariates"] == [6, 7]

    def test_timestamp_records_format(self):
        """Test with TIMESTAMP_RECORDS format."""
        config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            dataset_format=TimeSeriesJSONDatasetFormat.TIMESTAMP_RECORDS,
        )

        data = config.get_time_series_data_config()
        assert data["dataset_format"] == "timestamp_records"

    def test_invalid_related_time_series_type_mismatch(self):
        """Test with type mismatch in related_time_series."""
        with pytest.raises(ValueError, match="Please provide a list"):
            TimeSeriesDataConfig(
                target_time_series="target",
                item_id="id",
                timestamp="time",
                related_time_series=["valid", 123],  # Mixed types
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
            )

    def test_invalid_static_covariates_type_mismatch(self):
        """Test with type mismatch in static_covariates."""
        with pytest.raises(ValueError, match="Please provide a list"):
            TimeSeriesDataConfig(
                target_time_series=1,
                item_id=2,
                timestamp=3,
                static_covariates=[4, "invalid"],  # Mixed types
            )

    def test_empty_string_in_static_covariates(self):
        """Test with empty string in static_covariates."""
        with pytest.raises(ValueError, match="Please do not provide empty strings"):
            TimeSeriesDataConfig(
                target_time_series="target",
                item_id="id",
                timestamp="time",
                static_covariates=["valid", ""],
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
            )

    def test_config_immutability(self):
        """Test that returned config is a copy."""
        config = TimeSeriesDataConfig(target_time_series=1, item_id=2, timestamp=3)

        data1 = config.get_time_series_data_config()
        data1["target_time_series"] = 999
        data2 = config.get_time_series_data_config()

        assert data2["target_time_series"] == 1  # Original unchanged


class TestDatasetTypeExtended:
    """Extended test cases for DatasetType enum."""

    def test_all_dataset_types(self):
        """Test all dataset type values."""
        assert DatasetType.TEXTCSV.value == "text/csv"
        assert DatasetType.JSONLINES.value == "application/jsonlines"
        assert DatasetType.JSON.value == "application/json"
        assert DatasetType.PARQUET.value == "application/x-parquet"
        assert DatasetType.IMAGE.value == "application/x-image"

    def test_dataset_type_membership(self):
        """Test dataset type membership."""
        assert DatasetType.TEXTCSV in DatasetType
        assert DatasetType.JSONLINES in DatasetType
        assert DatasetType.JSON in DatasetType
        assert DatasetType.PARQUET in DatasetType
        assert DatasetType.IMAGE in DatasetType


class TestDataConfig:
    """Test DataConfig class."""

    def test_init_with_csv_dataset(self):
        """Test initialization with CSV dataset."""
        from sagemaker.core.clarify import DataConfig

        config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            label="target",
            headers=["col1", "col2", "target"],
            dataset_type="text/csv",
        )
        assert config.s3_data_input_path == "s3://bucket/input"
        assert config.s3_output_path == "s3://bucket/output"
        assert config.label == "target"

    def test_init_with_json_dataset_without_features_raises_error(self):
        """Test that JSON dataset without features raises ValueError."""
        from sagemaker.core.clarify import DataConfig

        with pytest.raises(ValueError, match="features JMESPath is required"):
            DataConfig(
                s3_data_input_path="s3://bucket/input",
                s3_output_path="s3://bucket/output",
                dataset_type="application/json",
            )

    def test_init_with_invalid_dataset_type_raises_error(self):
        """Test that invalid dataset_type raises ValueError."""
        from sagemaker.core.clarify import DataConfig

        with pytest.raises(ValueError, match="Invalid dataset_type"):
            DataConfig(
                s3_data_input_path="s3://bucket/input",
                s3_output_path="s3://bucket/output",
                dataset_type="invalid/type",
            )

    def test_init_with_predicted_label_for_image_raises_error(self):
        """Test that predicted_label with image dataset raises ValueError."""
        from sagemaker.core.clarify import DataConfig

        with pytest.raises(ValueError, match="not supported"):
            DataConfig(
                s3_data_input_path="s3://bucket/input",
                s3_output_path="s3://bucket/output",
                dataset_type="application/x-image",
                predicted_label="label",
            )

    def test_init_with_facet_dataset_for_non_csv_raises_error(self):
        """Test that facet_dataset_uri with non-CSV raises ValueError."""
        from sagemaker.core.clarify import DataConfig

        with pytest.raises(ValueError, match="not supported"):
            DataConfig(
                s3_data_input_path="s3://bucket/input",
                s3_output_path="s3://bucket/output",
                dataset_type="application/json",
                features="data",
                facet_dataset_uri="s3://bucket/facet",
            )

    def test_init_with_time_series_non_json_raises_error(self):
        """Test that time series with non-JSON raises ValueError."""
        from sagemaker.core.clarify import (
            DataConfig,
            TimeSeriesDataConfig,
            TimeSeriesJSONDatasetFormat,
        )

        ts_config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        )
        with pytest.raises(ValueError, match="only supports JSON format"):
            DataConfig(
                s3_data_input_path="s3://bucket/input",
                s3_output_path="s3://bucket/output",
                dataset_type="text/csv",
                time_series_data_config=ts_config,
            )

    def test_get_config(self):
        """Test get_config returns copy."""
        from sagemaker.core.clarify import DataConfig

        config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            dataset_type="text/csv",
        )
        config1 = config.get_config()
        config2 = config.get_config()
        assert config1 is not config2


class TestBiasConfig:
    """Test BiasConfig class."""

    def test_init_with_single_facet(self):
        """Test initialization with single facet."""
        from sagemaker.core.clarify import BiasConfig

        config = BiasConfig(
            label_values_or_threshold=[1], facet_name="gender", facet_values_or_threshold=[0]
        )
        assert config.analysis_config["label_values_or_threshold"] == [1]
        assert len(config.analysis_config["facet"]) == 1

    def test_init_with_multiple_facets(self):
        """Test initialization with multiple facets."""
        from sagemaker.core.clarify import BiasConfig

        config = BiasConfig(
            label_values_or_threshold=[1],
            facet_name=["gender", "age"],
            facet_values_or_threshold=[[0], [18]],
        )
        assert len(config.analysis_config["facet"]) == 2

    def test_init_with_mismatched_facets_raises_error(self):
        """Test that mismatched facet counts raise ValueError."""
        from sagemaker.core.clarify import BiasConfig

        with pytest.raises(ValueError, match="number of facet names doesn't match"):
            BiasConfig(
                label_values_or_threshold=[1],
                facet_name=["gender", "age"],
                facet_values_or_threshold=[[0]],
            )

    def test_get_config(self):
        """Test get_config returns copy."""
        from sagemaker.core.clarify import BiasConfig

        config = BiasConfig(label_values_or_threshold=[1], facet_name="gender")
        config1 = config.get_config()
        config2 = config.get_config()
        assert config1 is not config2


class TestTimeSeriesModelConfig:
    """Test TimeSeriesModelConfig class."""

    def test_init_with_valid_forecast(self):
        """Test initialization with valid forecast."""
        from sagemaker.core.clarify import TimeSeriesModelConfig

        config = TimeSeriesModelConfig(forecast="predictions.mean")
        assert config.time_series_model_config["forecast"] == "predictions.mean"

    def test_init_with_non_string_raises_error(self):
        """Test that non-string forecast raises ValueError."""
        from sagemaker.core.clarify import TimeSeriesModelConfig

        with pytest.raises(ValueError, match="Please provide a string"):
            TimeSeriesModelConfig(forecast=123)

    def test_get_time_series_model_config(self):
        """Test get_time_series_model_config returns copy."""
        from sagemaker.core.clarify import TimeSeriesModelConfig

        config = TimeSeriesModelConfig(forecast="predictions")
        config1 = config.get_time_series_model_config()
        config2 = config.get_time_series_model_config()
        assert config1 is not config2


class TestModelConfig:
    """Test ModelConfig class."""

    def test_init_with_model_params(self):
        """Test initialization with model parameters."""
        from sagemaker.core.clarify import ModelConfig

        config = ModelConfig(model_name="my-model", instance_count=1, instance_type="ml.m5.xlarge")
        assert config.predictor_config["model_name"] == "my-model"
        assert config.predictor_config["initial_instance_count"] == 1

    def test_init_with_endpoint_name(self):
        """Test initialization with endpoint name."""
        from sagemaker.core.clarify import ModelConfig

        config = ModelConfig(endpoint_name="my-endpoint")
        assert config.predictor_config["endpoint_name"] == "my-endpoint"

    def test_init_with_invalid_endpoint_prefix_raises_error(self):
        """Test that invalid endpoint_name_prefix raises ValueError."""
        from sagemaker.core.clarify import ModelConfig

        with pytest.raises(ValueError, match="Invalid endpoint_name_prefix"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                endpoint_name_prefix="!invalid",
            )

    def test_init_with_invalid_accept_type_raises_error(self):
        """Test that invalid accept_type raises ValueError."""
        from sagemaker.core.clarify import ModelConfig

        with pytest.raises(ValueError, match="Invalid accept_type"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                accept_type="invalid/type",
            )

    def test_init_with_invalid_content_type_raises_error(self):
        """Test that invalid content_type raises ValueError."""
        from sagemaker.core.clarify import ModelConfig

        with pytest.raises(ValueError, match="Invalid content_type"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                content_type="invalid/type",
            )

    def test_init_with_jsonlines_without_content_template_raises_error(self):
        """Test that JSONLines without content_template raises ValueError."""
        from sagemaker.core.clarify import ModelConfig

        with pytest.raises(ValueError, match="content_template field is required"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                content_type="application/jsonlines",
            )

    def test_init_with_jsonlines_without_features_placeholder_raises_error(self):
        """Test that JSONLines without $features raises ValueError."""
        from sagemaker.core.clarify import ModelConfig

        with pytest.raises(ValueError, match="Please include a placeholder"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                content_type="application/jsonlines",
                content_template='{"data": $invalid}',
            )

    def test_init_with_json_without_templates_raises_error(self):
        """Test that JSON without templates raises ValueError."""
        from sagemaker.core.clarify import ModelConfig

        with pytest.raises(ValueError, match="content_template and record_template are required"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                content_type="application/json",
            )

    def test_init_with_json_without_record_placeholder_raises_error(self):
        """Test that JSON without $record raises ValueError."""
        from sagemaker.core.clarify import ModelConfig

        with pytest.raises(ValueError, match="Please include either placeholder"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                content_type="application/json",
                content_template='{"data": $invalid}',
                record_template="$features",
            )

    def test_init_with_time_series_csv_accept_raises_error(self):
        """Test that time series with CSV accept_type raises ValueError."""
        from sagemaker.core.clarify import ModelConfig, TimeSeriesModelConfig

        ts_config = TimeSeriesModelConfig(forecast="predictions")
        with pytest.raises(ValueError, match="must be JSON or JSONLines"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                accept_type="text/csv",
                time_series_model_config=ts_config,
            )

    def test_get_predictor_config(self):
        """Test get_predictor_config returns copy."""
        from sagemaker.core.clarify import ModelConfig

        config = ModelConfig(model_name="model", instance_count=1, instance_type="ml.m5.xlarge")
        config1 = config.get_predictor_config()
        config2 = config.get_predictor_config()
        assert config1 is not config2


class TestModelPredictedLabelConfig:
    """Test ModelPredictedLabelConfig class."""

    def test_init_with_label(self):
        """Test initialization with label."""
        from sagemaker.core.clarify import ModelPredictedLabelConfig

        config = ModelPredictedLabelConfig(label="predicted_label")
        assert config.label == "predicted_label"

    def test_init_with_probability_threshold(self):
        """Test initialization with probability threshold."""
        from sagemaker.core.clarify import ModelPredictedLabelConfig

        config = ModelPredictedLabelConfig(probability_threshold=0.7)
        assert config.probability_threshold == 0.7

    def test_init_with_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises TypeError."""
        from sagemaker.core.clarify import ModelPredictedLabelConfig

        with pytest.raises(TypeError, match="Invalid probability_threshold"):
            ModelPredictedLabelConfig(probability_threshold="invalid")

    def test_get_predictor_config(self):
        """Test get_predictor_config returns tuple."""
        from sagemaker.core.clarify import ModelPredictedLabelConfig

        config = ModelPredictedLabelConfig(label="label", probability_threshold=0.5)
        threshold, pred_config = config.get_predictor_config()
        assert threshold == 0.5
        assert pred_config["label"] == "label"


class TestPDPConfig:
    """Test PDPConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        from sagemaker.core.clarify import PDPConfig

        config = PDPConfig()
        result = config.get_explainability_config()
        assert result["pdp"]["grid_resolution"] == 15
        assert result["pdp"]["top_k_features"] == 10

    def test_init_with_features(self):
        """Test initialization with features."""
        from sagemaker.core.clarify import PDPConfig

        config = PDPConfig(features=["feature1", "feature2"])
        result = config.get_explainability_config()
        assert result["pdp"]["features"] == ["feature1", "feature2"]

    def test_get_explainability_config(self):
        """Test get_explainability_config returns copy."""
        from sagemaker.core.clarify import PDPConfig

        config = PDPConfig()
        config1 = config.get_explainability_config()
        config2 = config.get_explainability_config()
        assert config1 is not config2


class TestTextConfig:
    """Test TextConfig class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        from sagemaker.core.clarify import TextConfig

        config = TextConfig(granularity="token", language="english")
        result = config.get_text_config()
        assert result["granularity"] == "token"
        assert result["language"] == "english"

    def test_init_with_invalid_granularity_raises_error(self):
        """Test that invalid granularity raises ValueError."""
        from sagemaker.core.clarify import TextConfig

        with pytest.raises(ValueError, match="Invalid granularity"):
            TextConfig(granularity="invalid", language="english")

    def test_init_with_invalid_language_raises_error(self):
        """Test that invalid language raises ValueError."""
        from sagemaker.core.clarify import TextConfig

        with pytest.raises(ValueError, match="Invalid language"):
            TextConfig(granularity="token", language="invalid")

    def test_get_text_config(self):
        """Test get_text_config returns copy."""
        from sagemaker.core.clarify import TextConfig

        config = TextConfig(granularity="sentence", language="french")
        config1 = config.get_text_config()
        config2 = config.get_text_config()
        assert config1 is not config2


class TestImageConfig:
    """Test ImageConfig class."""

    def test_init_with_image_classification(self):
        """Test initialization with image classification."""
        from sagemaker.core.clarify import ImageConfig

        config = ImageConfig(model_type="IMAGE_CLASSIFICATION")
        result = config.get_image_config()
        assert result["model_type"] == "IMAGE_CLASSIFICATION"

    def test_init_with_object_detection(self):
        """Test initialization with object detection."""
        from sagemaker.core.clarify import ImageConfig

        config = ImageConfig(model_type="OBJECT_DETECTION", max_objects=5, iou_threshold=0.6)
        result = config.get_image_config()
        assert result["model_type"] == "OBJECT_DETECTION"
        assert result["max_objects"] == 5

    def test_init_with_invalid_model_type_raises_error(self):
        """Test that invalid model_type raises ValueError."""
        from sagemaker.core.clarify import ImageConfig

        with pytest.raises(ValueError, match="only supports object detection"):
            ImageConfig(model_type="INVALID_TYPE")

    def test_get_image_config(self):
        """Test get_image_config returns copy."""
        from sagemaker.core.clarify import ImageConfig

        config = ImageConfig(model_type="IMAGE_CLASSIFICATION")
        config1 = config.get_image_config()
        config2 = config.get_image_config()
        assert config1 is not config2


class TestSHAPConfig:
    """Test SHAPConfig class."""

    def test_init_with_baseline(self):
        """Test initialization with baseline."""
        from sagemaker.core.clarify import SHAPConfig

        config = SHAPConfig(baseline=[[1, 2, 3]])
        result = config.get_explainability_config()
        assert result["shap"]["baseline"] == [[1, 2, 3]]

    def test_init_with_invalid_agg_method_raises_error(self):
        """Test that invalid agg_method raises ValueError."""
        from sagemaker.core.clarify import SHAPConfig

        with pytest.raises(ValueError, match="Invalid agg_method"):
            SHAPConfig(agg_method="invalid")

    def test_init_with_baseline_and_num_clusters_raises_error(self):
        """Test that baseline and num_clusters together raise ValueError."""
        from sagemaker.core.clarify import SHAPConfig

        with pytest.raises(ValueError, match="cannot be provided together"):
            SHAPConfig(baseline=[[1, 2]], num_clusters=5)

    def test_init_with_text_config(self):
        """Test initialization with text config."""
        from sagemaker.core.clarify import SHAPConfig, TextConfig

        text_config = TextConfig(granularity="token", language="english")
        config = SHAPConfig(text_config=text_config)
        result = config.get_explainability_config()
        assert "text_config" in result["shap"]

    def test_init_with_features_to_explain_and_text_raises_error(self):
        """Test that features_to_explain with text raises ValueError."""
        from sagemaker.core.clarify import SHAPConfig, TextConfig

        text_config = TextConfig(granularity="token", language="english")
        with pytest.raises(ValueError, match="not supported for datasets containing text"):
            SHAPConfig(text_config=text_config, features_to_explain=["feature1"])

    def test_get_explainability_config(self):
        """Test get_explainability_config returns copy."""
        from sagemaker.core.clarify import SHAPConfig

        config = SHAPConfig()
        config1 = config.get_explainability_config()
        config2 = config.get_explainability_config()
        assert config1 is not config2


class TestAsymmetricShapleyValueConfig:
    """Test AsymmetricShapleyValueConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        config = AsymmetricShapleyValueConfig()
        result = config.get_explainability_config()
        assert result["asymmetric_shapley_value"]["direction"] == "chronological"
        assert result["asymmetric_shapley_value"]["granularity"] == "timewise"

    def test_init_with_invalid_direction_raises_error(self):
        """Test that invalid direction raises ValueError."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        with pytest.raises(ValueError, match="Please provide a valid explanation direction"):
            AsymmetricShapleyValueConfig(direction="invalid")

    def test_init_with_invalid_granularity_raises_error(self):
        """Test that invalid granularity raises ValueError."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        with pytest.raises(ValueError, match="Please provide a valid granularity"):
            AsymmetricShapleyValueConfig(granularity="invalid")

    def test_init_with_fine_grained_without_num_samples_raises_error(self):
        """Test that fine_grained without num_samples raises ValueError."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        with pytest.raises(ValueError, match="Please provide an integer"):
            AsymmetricShapleyValueConfig(granularity="fine_grained")

    def test_init_with_fine_grained_non_chronological_raises_error(self):
        """Test that fine_grained with non-chronological raises ValueError."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        with pytest.raises(ValueError, match="not supported together"):
            AsymmetricShapleyValueConfig(
                direction="anti_chronological", granularity="fine_grained", num_samples=100
            )

    def test_init_with_num_samples_for_timewise_raises_error(self):
        """Test that num_samples for timewise raises ValueError."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        with pytest.raises(ValueError, match="only used for fine-grained"):
            AsymmetricShapleyValueConfig(granularity="timewise", num_samples=100)

    def test_init_with_invalid_target_baseline_raises_error(self):
        """Test that invalid target baseline raises ValueError."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        with pytest.raises(ValueError, match="invalid"):
            AsymmetricShapleyValueConfig(baseline={"target_time_series": "invalid"})

    def test_init_with_invalid_related_baseline_raises_error(self):
        """Test that invalid related baseline raises ValueError."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        with pytest.raises(ValueError, match="invalid"):
            AsymmetricShapleyValueConfig(baseline={"related_time_series": "invalid"})

    def test_get_explainability_config(self):
        """Test get_explainability_config returns copy."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        config = AsymmetricShapleyValueConfig()
        config1 = config.get_explainability_config()
        config2 = config.get_explainability_config()
        assert config1 is not config2


class TestSageMakerClarifyProcessor:
    """Test SageMakerClarifyProcessor class."""

    @patch("sagemaker.core.clarify.image_uris.retrieve")
    def test_init(self, mock_retrieve):
        """Test initialization."""
        from sagemaker.core.clarify import SageMakerClarifyProcessor
        from sagemaker.core.helper.session_helper import Session

        mock_retrieve.return_value = "clarify-image-uri"
        mock_session = Mock(spec=Session)
        mock_session.boto_region_name = "us-west-2"

        processor = SageMakerClarifyProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        assert processor.job_name_prefix is None
        assert processor.skip_early_validation is False

    @patch("sagemaker.core.clarify.image_uris.retrieve")
    def test_run_raises_not_implemented(self, mock_retrieve):
        """Test that run method raises NotImplementedError."""
        from sagemaker.core.clarify import SageMakerClarifyProcessor
        from sagemaker.core.helper.session_helper import Session

        mock_retrieve.return_value = "clarify-image-uri"
        mock_session = Mock(spec=Session)
        mock_session.boto_region_name = "us-west-2"

        processor = SageMakerClarifyProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with pytest.raises(NotImplementedError, match="Please choose a method"):
            processor.run()


class TestAnalysisConfigGenerator:
    """Test _AnalysisConfigGenerator class."""

    def test_bias_pre_training(self):
        """Test bias_pre_training method."""
        from sagemaker.core.clarify import DataConfig, BiasConfig
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            label="target",
            dataset_type="text/csv",
        )
        bias_config = BiasConfig(label_values_or_threshold=[1], facet_name="gender")

        result = _AnalysisConfigGenerator.bias_pre_training(data_config, bias_config, "all")

        assert "methods" in result
        assert "pre_training_bias" in result["methods"]

    def test_bias_post_training(self):
        """Test bias_post_training method."""
        from sagemaker.core.clarify import DataConfig, BiasConfig, ModelConfig
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            label="target",
            dataset_type="text/csv",
        )
        bias_config = BiasConfig(label_values_or_threshold=[1], facet_name="gender")
        model_config = ModelConfig(
            model_name="model", instance_count=1, instance_type="ml.m5.xlarge"
        )

        result = _AnalysisConfigGenerator.bias_post_training(
            data_config, bias_config, None, "all", model_config
        )

        assert "methods" in result
        assert "post_training_bias" in result["methods"]

    def test_explainability_with_time_series_without_data_config_raises_error(self):
        """Test explainability with AsymmetricShapley without TimeSeriesDataConfig raises error."""
        from sagemaker.core.clarify import DataConfig, ModelConfig, AsymmetricShapleyValueConfig
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            dataset_type="application/json",
            features="data",
        )
        model_config = ModelConfig(
            model_name="model", instance_count=1, instance_type="ml.m5.xlarge"
        )
        explainability_config = AsymmetricShapleyValueConfig()

        with pytest.raises(ValueError, match="Please provide a TimeSeriesDataConfig"):
            _AnalysisConfigGenerator.explainability(
                data_config, model_config, None, explainability_config
            )

    def test_add_predictor_without_model_config_for_shap_raises_error(self):
        """Test _add_predictor without model_config for SHAP raises error."""
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        analysis_config = {"methods": {"shap": {}}}

        with pytest.raises(ValueError, match="model_config must be provided"):
            _AnalysisConfigGenerator._add_predictor(analysis_config, None, None)

    def test_add_methods_without_any_method_raises_error(self):
        """Test _add_methods without any method raises error."""
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        with pytest.raises(AttributeError, match="must have at least one working method"):
            _AnalysisConfigGenerator._add_methods({})

    def test_merge_explainability_configs_with_asymmetric_raises_error(self):
        """Test _merge_explainability_configs with AsymmetricShapley raises error."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        config = AsymmetricShapleyValueConfig()

        with pytest.raises(ValueError, match="do not provide Asymmetric"):
            _AnalysisConfigGenerator._merge_explainability_configs(config)

    def test_merge_explainability_configs_with_pdp_without_features_raises_error(self):
        """Test _merge_explainability_configs with PDP without features raises error."""
        from sagemaker.core.clarify import PDPConfig
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        config = PDPConfig()

        with pytest.raises(ValueError, match="PDP features must be provided"):
            _AnalysisConfigGenerator._merge_explainability_configs(config)

    def test_validate_time_series_static_covariates_baseline_mismatch_raises_error(self):
        """Test validation of static covariates baseline with mismatch raises error."""
        from sagemaker.core.clarify import (
            AsymmetricShapleyValueConfig,
            DataConfig,
            TimeSeriesDataConfig,
            TimeSeriesJSONDatasetFormat,
        )
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        ts_data_config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            static_covariates=["cov1", "cov2"],
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        )
        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            dataset_type="application/json",
            time_series_data_config=ts_data_config,
        )
        explainability_config = AsymmetricShapleyValueConfig(
            baseline={"static_covariates": {"item1": [1]}}
        )

        with pytest.raises(ValueError, match="does not match number"):
            _AnalysisConfigGenerator._validate_time_series_static_covariates_baseline(
                explainability_config, data_config
            )


class TestProcessingOutputHandler:
    """Test ProcessingOutputHandler class."""

    def test_get_s3_upload_mode_for_image(self):
        """Test get_s3_upload_mode for image dataset."""
        from sagemaker.core.clarify import ProcessingOutputHandler

        analysis_config = {"dataset_type": "application/x-image"}
        result = ProcessingOutputHandler.get_s3_upload_mode(analysis_config)

        assert result == "Continuous"

    def test_get_s3_upload_mode_for_csv(self):
        """Test get_s3_upload_mode for CSV dataset."""
        from sagemaker.core.clarify import ProcessingOutputHandler

        analysis_config = {"dataset_type": "text/csv"}
        result = ProcessingOutputHandler.get_s3_upload_mode(analysis_config)

        assert result == "EndOfJob"


class TestDataConfigExtended:
    """Extended tests for DataConfig."""

    def test_init_with_all_optional_params(self):
        """Test initialization with all optional parameters."""
        from sagemaker.core.clarify import DataConfig, SegmentationConfig

        seg_config = SegmentationConfig(name_or_index="age", segments=[[18]])
        config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            s3_analysis_config_output_path="s3://bucket/analysis",
            label="target",
            headers=["col1", "col2", "target"],
            dataset_type="text/csv",
            s3_compression_type="Gzip",
            joinsource="id",
            facet_dataset_uri="s3://bucket/facet",
            facet_headers=["facet1"],
            predicted_label_dataset_uri="s3://bucket/predicted",
            predicted_label_headers=["pred"],
            predicted_label="prediction",
            excluded_columns=["col3"],
            segmentation_config=[seg_config],
        )
        assert config.s3_analysis_config_output_path == "s3://bucket/analysis"
        assert config.s3_compression_type == "Gzip"
        assert config.analysis_config["joinsource_name_or_index"] == "id"

    def test_init_with_excluded_columns_for_image_raises_error(self):
        """Test that excluded_columns with image raises ValueError."""
        from sagemaker.core.clarify import DataConfig

        with pytest.raises(ValueError, match="not supported"):
            DataConfig(
                s3_data_input_path="s3://bucket/input",
                s3_output_path="s3://bucket/output",
                dataset_type="application/x-image",
                excluded_columns=["col1"],
            )

    def test_init_with_predicted_label_dataset_for_non_csv_raises_error(self):
        """Test that predicted_label_dataset_uri with non-CSV raises ValueError."""
        from sagemaker.core.clarify import DataConfig

        with pytest.raises(ValueError, match="not supported"):
            DataConfig(
                s3_data_input_path="s3://bucket/input",
                s3_output_path="s3://bucket/output",
                dataset_type="application/json",
                features="data",
                predicted_label_dataset_uri="s3://bucket/predicted",
            )


class TestBiasConfigExtended:
    """Extended tests for BiasConfig."""

    def test_init_with_group_name(self):
        """Test initialization with group_name."""
        from sagemaker.core.clarify import BiasConfig

        config = BiasConfig(
            label_values_or_threshold=[1], facet_name="gender", group_name="group_id"
        )
        assert config.analysis_config["group_variable"] == "group_id"

    def test_init_with_multiple_facets_no_threshold(self):
        """Test initialization with multiple facets without threshold."""
        from sagemaker.core.clarify import BiasConfig

        config = BiasConfig(label_values_or_threshold=[1], facet_name=["gender", "age"])
        assert len(config.analysis_config["facet"]) == 2
        assert "value_or_threshold" not in config.analysis_config["facet"][0]


class TestModelConfigExtended:
    """Extended tests for ModelConfig."""

    def test_init_with_all_optional_params(self):
        """Test initialization with all optional parameters."""
        from sagemaker.core.clarify import ModelConfig

        config = ModelConfig(
            model_name="model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            accept_type="application/json",
            content_type="application/json",
            content_template='{"data": $record}',
            record_template="$features",
            custom_attributes="attr1=value1",
            accelerator_type="ml.eia2.medium",
            endpoint_name_prefix="my-endpoint",
            target_model="target-model",
        )
        assert config.predictor_config["custom_attributes"] == "attr1=value1"
        assert config.predictor_config["accelerator_type"] == "ml.eia2.medium"
        assert config.predictor_config["target_model"] == "target-model"

    def test_init_with_time_series_invalid_content_type_raises_error(self):
        """Test that time series with invalid content_type raises ValueError."""
        from sagemaker.core.clarify import ModelConfig, TimeSeriesModelConfig

        ts_config = TimeSeriesModelConfig(forecast="predictions")
        with pytest.raises(ValueError, match="must be JSON or JSONLines"):
            ModelConfig(
                model_name="model",
                instance_count=1,
                instance_type="ml.m5.xlarge",
                content_type="text/csv",
                time_series_model_config=ts_config,
            )


class TestSHAPConfigExtended:
    """Extended tests for SHAPConfig."""

    def test_init_with_image_config(self):
        """Test initialization with image config."""
        from sagemaker.core.clarify import SHAPConfig, ImageConfig

        image_config = ImageConfig(model_type="IMAGE_CLASSIFICATION")
        config = SHAPConfig(image_config=image_config)
        result = config.get_explainability_config()
        assert "image_config" in result["shap"]

    def test_init_with_features_to_explain_and_image_raises_error(self):
        """Test that features_to_explain with image raises ValueError."""
        from sagemaker.core.clarify import SHAPConfig, ImageConfig

        image_config = ImageConfig(model_type="IMAGE_CLASSIFICATION")
        with pytest.raises(ValueError, match="not supported for datasets containing"):
            SHAPConfig(image_config=image_config, features_to_explain=["feature1"])

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        from sagemaker.core.clarify import SHAPConfig

        config = SHAPConfig(
            baseline=[[1, 2, 3]],
            num_samples=100,
            agg_method="mean_abs",
            use_logit=True,
            save_local_shap_values=False,
            seed=42,
            features_to_explain=["feature1", "feature2"],
        )
        result = config.get_explainability_config()
        assert result["shap"]["use_logit"] is True
        assert result["shap"]["save_local_shap_values"] is False
        assert result["shap"]["seed"] == 42


class TestAsymmetricShapleyValueConfigExtended:
    """Extended tests for AsymmetricShapleyValueConfig."""

    def test_init_with_all_directions(self):
        """Test initialization with all direction options."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        for direction in ["chronological", "anti_chronological", "bidirectional"]:
            config = AsymmetricShapleyValueConfig(direction=direction)
            result = config.get_explainability_config()
            assert result["asymmetric_shapley_value"]["direction"] == direction

    def test_init_with_baseline_string(self):
        """Test initialization with baseline as string."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        config = AsymmetricShapleyValueConfig(baseline="s3://bucket/baseline")
        result = config.get_explainability_config()
        assert result["asymmetric_shapley_value"]["baseline"] == "s3://bucket/baseline"

    def test_init_with_baseline_dict(self):
        """Test initialization with baseline as dict."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        baseline = {
            "target_time_series": "zero",
            "related_time_series": "mean",
            "static_covariates": {"item1": [1, 2]},
        }
        config = AsymmetricShapleyValueConfig(baseline=baseline)
        result = config.get_explainability_config()
        assert result["asymmetric_shapley_value"]["baseline"] == baseline

    def test_init_with_fine_grained_chronological(self):
        """Test initialization with fine_grained and chronological."""
        from sagemaker.core.clarify import AsymmetricShapleyValueConfig

        config = AsymmetricShapleyValueConfig(
            direction="chronological", granularity="fine_grained", num_samples=50
        )
        result = config.get_explainability_config()
        assert result["asymmetric_shapley_value"]["num_samples"] == 50


class TestAnalysisConfigGeneratorExtended:
    """Extended tests for _AnalysisConfigGenerator."""

    def test_bias(self):
        """Test bias method."""
        from sagemaker.core.clarify import (
            DataConfig,
            BiasConfig,
            ModelConfig,
            _AnalysisConfigGenerator,
        )

        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            label="target",
            dataset_type="text/csv",
        )
        bias_config = BiasConfig(label_values_or_threshold=[1], facet_name="gender")
        model_config = ModelConfig(
            model_name="model", instance_count=1, instance_type="ml.m5.xlarge"
        )

        result = _AnalysisConfigGenerator.bias(
            data_config, bias_config, model_config, None, "all", "all"
        )

        assert "methods" in result
        assert "pre_training_bias" in result["methods"]
        assert "post_training_bias" in result["methods"]

    def test_explainability_with_shap(self):
        """Test explainability with SHAP config."""
        from sagemaker.core.clarify import (
            DataConfig,
            ModelConfig,
            SHAPConfig,
            _AnalysisConfigGenerator,
        )

        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            dataset_type="text/csv",
        )
        model_config = ModelConfig(
            model_name="model", instance_count=1, instance_type="ml.m5.xlarge"
        )
        shap_config = SHAPConfig()

        result = _AnalysisConfigGenerator.explainability(
            data_config, model_config, None, shap_config
        )

        assert "methods" in result
        assert "shap" in result["methods"]

    def test_bias_and_explainability(self):
        """Test bias_and_explainability method."""
        from sagemaker.core.clarify import (
            DataConfig,
            BiasConfig,
            ModelConfig,
            SHAPConfig,
            _AnalysisConfigGenerator,
        )

        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            label="target",
            dataset_type="text/csv",
        )
        bias_config = BiasConfig(label_values_or_threshold=[1], facet_name="gender")
        model_config = ModelConfig(
            model_name="model", instance_count=1, instance_type="ml.m5.xlarge"
        )
        shap_config = SHAPConfig()

        result = _AnalysisConfigGenerator.bias_and_explainability(
            data_config, model_config, None, shap_config, bias_config, "all", "all"
        )

        assert "methods" in result
        assert "shap" in result["methods"]
        assert "pre_training_bias" in result["methods"]

    def test_bias_and_explainability_with_time_series_raises_error(self):
        """Test that bias_and_explainability with time series raises error."""
        from sagemaker.core.clarify import (
            DataConfig,
            BiasConfig,
            ModelConfig,
            AsymmetricShapleyValueConfig,
            TimeSeriesDataConfig,
            TimeSeriesJSONDatasetFormat,
            _AnalysisConfigGenerator,
        )

        ts_data_config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        )
        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            dataset_type="application/json",
            time_series_data_config=ts_data_config,
        )
        bias_config = BiasConfig(label_values_or_threshold=[1], facet_name="gender")
        model_config = ModelConfig(
            model_name="model", instance_count=1, instance_type="ml.m5.xlarge"
        )
        explainability_config = AsymmetricShapleyValueConfig()

        with pytest.raises(ValueError, match="Bias metrics are unsupported"):
            _AnalysisConfigGenerator.bias_and_explainability(
                data_config, model_config, None, explainability_config, bias_config
            )

    def test_add_predictor_with_model_predicted_label_config(self):
        """Test _add_predictor with ModelPredictedLabelConfig."""
        from sagemaker.core.clarify import (
            ModelConfig,
            ModelPredictedLabelConfig,
            _AnalysisConfigGenerator,
        )

        model_config = ModelConfig(
            model_name="model", instance_count=1, instance_type="ml.m5.xlarge"
        )
        model_predicted_label_config = ModelPredictedLabelConfig(
            label="predicted_label", probability_threshold=0.5
        )
        analysis_config = {"methods": {}}

        result = _AnalysisConfigGenerator._add_predictor(
            analysis_config, model_config, model_predicted_label_config
        )

        assert "predictor" in result
        assert result["probability_threshold"] == 0.5

    def test_merge_explainability_configs_with_list(self):
        """Test _merge_explainability_configs with list of configs."""
        from sagemaker.core.clarify import SHAPConfig, PDPConfig, _AnalysisConfigGenerator

        shap_config = SHAPConfig()
        pdp_config = PDPConfig(features=["feature1"])

        result = _AnalysisConfigGenerator._merge_explainability_configs([shap_config, pdp_config])

        assert "shap" in result
        assert "pdp" in result

    def test_merge_explainability_configs_with_duplicate_raises_error(self):
        """Test _merge_explainability_configs with duplicates raises error."""
        from sagemaker.core.clarify import SHAPConfig, _AnalysisConfigGenerator

        shap_config1 = SHAPConfig()
        shap_config2 = SHAPConfig()

        with pytest.raises(ValueError, match="Duplicate explainability configs"):
            _AnalysisConfigGenerator._merge_explainability_configs([shap_config1, shap_config2])

    def test_validate_time_series_static_covariates_baseline_no_covariates_raises_error(self):
        """Test validation when baseline provided but no covariates in data config."""
        from sagemaker.core.clarify import (
            AsymmetricShapleyValueConfig,
            DataConfig,
            TimeSeriesDataConfig,
            TimeSeriesJSONDatasetFormat,
            _AnalysisConfigGenerator,
        )

        ts_data_config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        )
        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            dataset_type="application/json",
            time_series_data_config=ts_data_config,
        )
        explainability_config = AsymmetricShapleyValueConfig(
            baseline={"static_covariates": {"item1": [1, 2]}}
        )

        with pytest.raises(ValueError, match="no static covariate columns"):
            _AnalysisConfigGenerator._validate_time_series_static_covariates_baseline(
                explainability_config, data_config
            )

    def test_validate_time_series_static_covariates_baseline_not_list_raises_error(self):
        """Test validation when baseline entry is not a list."""
        from sagemaker.core.clarify import (
            AsymmetricShapleyValueConfig,
            DataConfig,
            TimeSeriesDataConfig,
            TimeSeriesJSONDatasetFormat,
            _AnalysisConfigGenerator,
        )

        ts_data_config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            static_covariates=["cov1"],
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        )
        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            dataset_type="application/json",
            time_series_data_config=ts_data_config,
        )
        explainability_config = AsymmetricShapleyValueConfig(
            baseline={"static_covariates": {"item1": "not_a_list"}}
        )

        with pytest.raises(ValueError, match="must be a list"):
            _AnalysisConfigGenerator._validate_time_series_static_covariates_baseline(
                explainability_config, data_config
            )

    def test_explainability_with_shap_without_time_series_data_config_raises_error(self):
        """Test explainability with SHAP when TimeSeriesDataConfig is provided raises error."""
        from sagemaker.core.clarify import (
            DataConfig,
            ModelConfig,
            SHAPConfig,
            TimeSeriesDataConfig,
            TimeSeriesJSONDatasetFormat,
            _AnalysisConfigGenerator,
        )

        ts_data_config = TimeSeriesDataConfig(
            target_time_series="target",
            item_id="id",
            timestamp="time",
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        )
        data_config = DataConfig(
            s3_data_input_path="s3://bucket/input",
            s3_output_path="s3://bucket/output",
            dataset_type="application/json",
            time_series_data_config=ts_data_config,
        )
        model_config = ModelConfig(
            model_name="model", instance_count=1, instance_type="ml.m5.xlarge"
        )
        shap_config = SHAPConfig()

        with pytest.raises(ValueError, match="Please provide an AsymmetricShapleyValueConfig"):
            _AnalysisConfigGenerator.explainability(data_config, model_config, None, shap_config)

    def test_add_predictor_without_model_config_and_predicted_label_raises_error(self):
        """Test _add_predictor without model_config and predicted_label raises error."""
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        analysis_config = {"methods": {"post_training_bias": {}}}

        with pytest.raises(ValueError, match="model_config must be provided"):
            _AnalysisConfigGenerator._add_predictor(analysis_config, None, None)

    def test_merge_explainability_configs_empty_list_raises_error(self):
        """Test _merge_explainability_configs with empty list raises error."""
        from sagemaker.core.clarify import _AnalysisConfigGenerator

        with pytest.raises(ValueError, match="Please provide at least one"):
            _AnalysisConfigGenerator._merge_explainability_configs([])

    def test_merge_explainability_configs_list_with_pdp_no_shap_no_features_raises_error(self):
        """Test _merge_explainability_configs with PDP without SHAP and no features raises error."""
        from sagemaker.core.clarify import PDPConfig, _AnalysisConfigGenerator

        pdp_config = PDPConfig()

        with pytest.raises(ValueError, match="PDP features must be provided"):
            _AnalysisConfigGenerator._merge_explainability_configs([pdp_config])
