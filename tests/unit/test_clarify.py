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

from __future__ import absolute_import, print_function

import copy

import pytest
from dataclasses import dataclass
from mock import ANY, MagicMock, Mock, patch
from typing import Any, Dict, List, NamedTuple, Optional, Union

from sagemaker import Processor, image_uris
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    TimeSeriesDataConfig,
    ModelConfig,
    TimeSeriesModelConfig,
    ModelPredictedLabelConfig,
    PDPConfig,
    SageMakerClarifyProcessor,
    SHAPConfig,
    AsymmetricShapleyValueConfig,
    TextConfig,
    ImageConfig,
    _AnalysisConfigGenerator,
    DatasetType,
    ProcessingOutputHandler,
    SegmentationConfig,
    ASYM_SHAP_VAL_EXPLANATION_DIRECTIONS,
    TimeSeriesJSONDatasetFormat,
)

JOB_NAME_PREFIX = "my-prefix"
TIMESTAMP = "2021-06-17-22-29-54-685"
JOB_NAME = "{}-{}".format(JOB_NAME_PREFIX, TIMESTAMP)


def test_uri():
    uri = image_uris.retrieve("clarify", "us-west-2")
    assert "306415355426.dkr.ecr.us-west-2.amazonaws.com/sagemaker-clarify-processing:1.0" == uri


@pytest.mark.parametrize(
    ("dataset_type", "features", "excluded_columns", "predicted_label"),
    [
        ("text/csv", None, ["F4"], "Predicted Label"),
        ("application/jsonlines", None, ["F4"], "Predicted Label"),
        ("application/json", "[*].[F1,F2,F3]", ["F4"], "Predicted Label"),
        ("application/x-parquet", None, ["F4"], "Predicted Label"),
    ],
)
def test_data_config(dataset_type, features, excluded_columns, predicted_label):
    # facets in input dataset
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    label_name = "Label"
    headers = ["Label", "F1", "F2", "F3", "F4", "Predicted Label"]
    segment_config = [
        SegmentationConfig(
            name_or_index="F1",
            segments=[[0]],
            config_name="c1",
            display_aliases=["a1"],
        )
    ]

    data_config = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=s3_output_path,
        features=features,
        label=label_name,
        headers=headers,
        dataset_type=dataset_type,
        excluded_columns=excluded_columns,
        predicted_label=predicted_label,
        segmentation_config=segment_config,
    )

    expected_config = {
        "dataset_type": dataset_type,
        "headers": headers,
        "label": "Label",
        "segment_config": [
            {
                "config_name": "c1",
                "display_aliases": ["a1"],
                "name_or_index": "F1",
                "segments": [[0]],
            }
        ],
    }
    if features:
        expected_config["features"] = features
    if excluded_columns:
        expected_config["excluded_columns"] = excluded_columns
    if predicted_label:
        expected_config["predicted_label"] = predicted_label

    assert expected_config == data_config.get_config()
    assert s3_data_input_path == data_config.s3_data_input_path
    assert s3_output_path == data_config.s3_output_path
    assert "None" == data_config.s3_compression_type
    assert "FullyReplicated" == data_config.s3_data_distribution_type


def test_data_config_with_separate_facet_dataset():
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    label_name = "Label"
    headers = ["Label", "F1", "F2", "F3", "F4"]

    # facets NOT in input dataset
    joinsource = 5
    facet_dataset_uri = "s3://path/to/facet.csv"
    facet_headers = ["Age"]
    predicted_label_dataset_uri = "s3://path/to/pred.csv"
    predicted_label_headers = ["Label", "F1", "F2", "F3", "F4", "Age"]
    predicted_label = "predicted_label"
    excluded_columns = "F4"

    data_config_no_facet = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=s3_output_path,
        label=label_name,
        headers=headers,
        dataset_type="text/csv",
        joinsource=joinsource,
        facet_dataset_uri=facet_dataset_uri,
        facet_headers=facet_headers,
        predicted_label_dataset_uri=predicted_label_dataset_uri,
        predicted_label_headers=predicted_label_headers,
        predicted_label=predicted_label,
        excluded_columns=excluded_columns,
    )

    expected_config_no_facet = {
        "dataset_type": "text/csv",
        "headers": headers,
        "label": label_name,
        "joinsource_name_or_index": joinsource,
        "facet_dataset_uri": facet_dataset_uri,
        "facet_headers": facet_headers,
        "predicted_label_dataset_uri": predicted_label_dataset_uri,
        "predicted_label_headers": predicted_label_headers,
        "predicted_label": predicted_label,
        "excluded_columns": excluded_columns,
    }

    assert expected_config_no_facet == data_config_no_facet.get_config()
    assert joinsource == data_config_no_facet.analysis_config["joinsource_name_or_index"]
    assert facet_dataset_uri == data_config_no_facet.facet_dataset_uri
    assert facet_headers == data_config_no_facet.facet_headers
    assert predicted_label_dataset_uri == data_config_no_facet.predicted_label_dataset_uri
    assert predicted_label_headers == data_config_no_facet.predicted_label_headers
    assert predicted_label == data_config_no_facet.predicted_label

    excluded_columns = "F4"
    data_config_excluded_cols = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=s3_output_path,
        label=label_name,
        headers=headers,
        dataset_type="text/csv",
        joinsource=joinsource,
        excluded_columns=excluded_columns,
    )

    expected_config_excluded_cols = {
        "dataset_type": "text/csv",
        "headers": headers,
        "label": label_name,
        "joinsource_name_or_index": joinsource,
        "excluded_columns": excluded_columns,
    }

    assert expected_config_excluded_cols == data_config_excluded_cols.get_config()
    assert joinsource == data_config_excluded_cols.analysis_config["joinsource_name_or_index"]
    assert excluded_columns == data_config_excluded_cols.excluded_columns


def test_invalid_data_config():
    # facets included in input dataset
    with pytest.raises(ValueError, match=r"^Invalid dataset_type"):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="whatnot_type",
        )
    # facets NOT included in input dataset
    error_msg = r"^The parameter 'predicted_label' is not supported for dataset_type"
    with pytest.raises(ValueError, match=error_msg):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="application/x-image",
            predicted_label="label",
        )
    error_msg = r"^The parameter 'excluded_columns' is not supported for dataset_type"
    with pytest.raises(ValueError, match=error_msg):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="application/x-image",
            excluded_columns="excluded",
        )
    error_msg = r"^The parameters 'facet_dataset_uri' and 'facet_headers' are not supported for dataset_type"  # noqa E501  # pylint: disable=c0301
    with pytest.raises(ValueError, match=error_msg):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="application/x-image",
            facet_dataset_uri="facet_dataset/URI",
            facet_headers="facet",
        )
    error_msg = r"^The parameters 'predicted_label_dataset_uri' and 'predicted_label_headers' are not supported for dataset_type"  # noqa E501  # pylint: disable=c0301
    with pytest.raises(ValueError, match=error_msg):
        DataConfig(
            s3_data_input_path="s3://bucket/inputpath",
            s3_output_path="s3://bucket/outputpath",
            dataset_type="application/jsonlines",
            predicted_label_dataset_uri="pred_dataset/URI",
            predicted_label_headers="prediction",
        )


@pytest.mark.parametrize(
    ("name_or_index", "segments", "config_name", "display_aliases"),
    [
        ("feature1", [[0]], None, None),
        ("feature1", [[0], ["[1, 3)", "(5, 10]"]], None, None),
        ("feature1", [[0], ["[1, 3)", "(5, 10]"]], "config1", None),
        ("feature1", [["A", "B"]], "config1", ["seg1"]),
        ("feature1", [["A", "B"]], "config1", ["seg1", "default_seg"]),
    ],
)
def test_segmentation_config(name_or_index, segments, config_name, display_aliases):
    segmentation_config = SegmentationConfig(
        name_or_index=name_or_index,
        segments=segments,
        config_name=config_name,
        display_aliases=display_aliases,
    )

    assert segmentation_config.name_or_index == name_or_index
    assert segmentation_config.segments == segments
    if segmentation_config.config_name:
        assert segmentation_config.config_name == config_name
    if segmentation_config.display_aliases:
        assert segmentation_config.display_aliases == display_aliases


@pytest.mark.parametrize(
    ("name_or_index", "segments", "config_name", "display_aliases", "error_msg"),
    [
        (None, [[0]], "config1", None, "`name_or_index` cannot be None"),
        (
            "feature1",
            "0",
            "config1",
            ["seg1"],
            "`segments` must be a list of lists of values or intervals.",
        ),
        (
            "feature1",
            [[0]],
            "config1",
            ["seg1", "seg2", "seg3"],
            "Number of `display_aliases` must equal the number of segments specified or with one "
            "additional default segment display alias.",
        ),
    ],
)
def test_invalid_segmentation_config(
    name_or_index, segments, config_name, display_aliases, error_msg
):
    with pytest.raises(ValueError, match=error_msg):
        SegmentationConfig(
            name_or_index=name_or_index,
            segments=segments,
            config_name=config_name,
            display_aliases=display_aliases,
        )


# features JMESPath is required for JSON dataset types
def test_json_type_data_config_missing_features():
    # facets in input dataset
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    label_name = "Label"
    headers = ["Label", "F1", "F2", "F3", "F4", "Predicted Label"]
    with pytest.raises(
        ValueError, match="features JMESPath is required for application/json dataset_type"
    ):
        DataConfig(
            s3_data_input_path=s3_data_input_path,
            s3_output_path=s3_output_path,
            features=None,
            label=label_name,
            headers=headers,
            dataset_type="application/json",
            excluded_columns=["F4"],
            predicted_label="Predicted Label",
        )


def test_s3_data_distribution_type_ignorance():
    data_config = DataConfig(
        s3_data_input_path="s3://input/train.csv",
        s3_output_path="s3://output/analysis_test_result",
        label="Label",
        headers=["Label", "F1", "F2", "F3", "F4"],
        dataset_type="text/csv",
        joinsource="F4",
    )
    assert data_config.s3_data_distribution_type == "FullyReplicated"


@dataclass
class TimeSeriesDataConfigCase:
    target_time_series: Union[str, int]
    item_id: Union[str, int]
    timestamp: Union[str, int]
    related_time_series: Optional[List[Union[str, int]]]
    static_covariates: Optional[List[Union[str, int]]]
    dataset_format: Optional[TimeSeriesJSONDatasetFormat] = None
    error: Optional[Exception] = None
    error_message: Optional[str] = None


class TestTimeSeriesDataConfig:
    valid_ts_data_config_case_list = [
        TimeSeriesDataConfigCase(  # no optional args provided str case
            target_time_series="target_time_series",
            item_id="item_id",
            timestamp="timestamp",
            related_time_series=None,
            static_covariates=None,
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        ),
        TimeSeriesDataConfigCase(  # related_time_series provided str case
            target_time_series="target_time_series",
            item_id="item_id",
            timestamp="timestamp",
            related_time_series=["ts1", "ts2", "ts3"],
            static_covariates=None,
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        ),
        TimeSeriesDataConfigCase(  # static_covariates provided str case
            target_time_series="target_time_series",
            item_id="item_id",
            timestamp="timestamp",
            related_time_series=None,
            static_covariates=["a", "b", "c", "d"],
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        ),
        TimeSeriesDataConfigCase(  # both related_time_series and static_covariates provided str case
            target_time_series="target_time_series",
            item_id="item_id",
            timestamp="timestamp",
            related_time_series=["ts1", "ts2", "ts3"],
            static_covariates=["a", "b", "c", "d"],
            dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
        ),
        TimeSeriesDataConfigCase(  # no optional args provided int case
            target_time_series=1,
            item_id=2,
            timestamp=3,
            related_time_series=None,
            static_covariates=None,
        ),
        TimeSeriesDataConfigCase(  # related_time_series provided int case
            target_time_series=1,
            item_id=2,
            timestamp=3,
            related_time_series=[4, 5, 6],
            static_covariates=None,
        ),
        TimeSeriesDataConfigCase(  # static_covariates provided int case
            target_time_series=1,
            item_id=2,
            timestamp=3,
            related_time_series=None,
            static_covariates=[7, 8, 9, 10],
        ),
        TimeSeriesDataConfigCase(  # both related_time_series and static_covariates provided int case
            target_time_series=1,
            item_id=2,
            timestamp=3,
            related_time_series=[4, 5, 6],
            static_covariates=[7, 8, 9, 10],
        ),
    ]

    @pytest.mark.parametrize("test_case", valid_ts_data_config_case_list)
    def test_time_series_data_config(self, test_case: TimeSeriesDataConfigCase):
        """
        GIVEN A set of valid parameters are given
        WHEN A TimeSeriesDataConfig object is instantiated
        THEN the returned config dictionary matches what's expected
        """
        # construct expected output
        expected_output = {
            "target_time_series": test_case.target_time_series,
            "item_id": test_case.item_id,
            "timestamp": test_case.timestamp,
        }
        if isinstance(test_case.target_time_series, str):
            expected_output["dataset_format"] = test_case.dataset_format.value
        if test_case.related_time_series:
            expected_output["related_time_series"] = test_case.related_time_series
        if test_case.static_covariates:
            expected_output["static_covariates"] = test_case.static_covariates
        # GIVEN, WHEN
        ts_data_config = TimeSeriesDataConfig(
            target_time_series=test_case.target_time_series,
            item_id=test_case.item_id,
            timestamp=test_case.timestamp,
            related_time_series=test_case.related_time_series,
            static_covariates=test_case.static_covariates,
            dataset_format=test_case.dataset_format,
        )
        # THEN
        assert ts_data_config.time_series_data_config == expected_output

    @pytest.mark.parametrize(
        "test_case",
        [
            TimeSeriesDataConfigCase(  # no target_time_series provided
                target_time_series=None,
                item_id="item_id",
                timestamp="timestamp",
                related_time_series=None,
                static_covariates=None,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message="Please provide a target time series.",
            ),
            TimeSeriesDataConfigCase(  # no item_id provided
                target_time_series="target_time_series",
                item_id=None,
                timestamp="timestamp",
                related_time_series=None,
                static_covariates=None,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message="Please provide an item id.",
            ),
            TimeSeriesDataConfigCase(  # no timestamp provided
                target_time_series="target_time_series",
                item_id="item_id",
                timestamp=None,
                related_time_series=None,
                static_covariates=None,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message="Please provide a timestamp.",
            ),
            TimeSeriesDataConfigCase(  # target_time_series not int or str
                target_time_series=["target_time_series"],
                item_id="item_id",
                timestamp="timestamp",
                related_time_series=None,
                static_covariates=None,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message="Please provide a string or an int for ``target_time_series``",
            ),
            TimeSeriesDataConfigCase(  # item_id differing type from str target_time_series
                target_time_series="target_time_series",
                item_id=5,
                timestamp="timestamp",
                related_time_series=None,
                static_covariates=None,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message=f"Please provide {str} for ``item_id``",
            ),
            TimeSeriesDataConfigCase(  # timestamp differing type from str target_time_series
                target_time_series="target_time_series",
                item_id="item_id",
                timestamp=10,
                related_time_series=None,
                static_covariates=None,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message=f"Please provide {str} for ``timestamp``",
            ),
            TimeSeriesDataConfigCase(  # related_time_series not str list if str target_time_series
                target_time_series="target_time_series",
                item_id="item_id",
                timestamp="timestamp",
                related_time_series=["ts1", "ts2", "ts3", 4],
                static_covariates=None,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message=f"Please provide a list of {str} for ``related_time_series``",
            ),
            TimeSeriesDataConfigCase(  # static_covariates not str list if str target_time_series
                target_time_series="target_time_series",
                item_id="item_id",
                timestamp="timestamp",
                related_time_series=None,
                static_covariates=[4, 5, 6.0],
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message=f"Please provide a list of {str} for ``static_covariates``",
            ),
            TimeSeriesDataConfigCase(  # item_id differing type from int target_time_series
                target_time_series=1,
                item_id="item_id",
                timestamp=3,
                related_time_series=None,
                static_covariates=None,
                dataset_format=None,
                error=ValueError,
                error_message=f"Please provide {int} for ``item_id``",
            ),
            TimeSeriesDataConfigCase(  # timestamp differing type from int target_time_series
                target_time_series=1,
                item_id=2,
                timestamp="timestamp",
                related_time_series=None,
                static_covariates=None,
                dataset_format=None,
                error=ValueError,
                error_message=f"Please provide {int} for ``timestamp``",
            ),
            TimeSeriesDataConfigCase(  # related_time_series not int list if int target_time_series
                target_time_series=1,
                item_id=2,
                timestamp=3,
                related_time_series=[4, 5, 6, "ts7"],
                static_covariates=None,
                dataset_format=None,
                error=ValueError,
                error_message=f"Please provide a list of {int} for ``related_time_series``",
            ),
            TimeSeriesDataConfigCase(  # static_covariates not int list if int target_time_series
                target_time_series=1,
                item_id=2,
                timestamp=3,
                related_time_series=[4, 5, 6, 7],
                static_covariates=[8, 9, "10"],
                dataset_format=None,
                error=ValueError,
                error_message=f"Please provide a list of {int} for ``static_covariates``",
            ),
            TimeSeriesDataConfigCase(  # related_time_series contains blank string
                target_time_series="target_time_series",
                item_id="item_id",
                timestamp="timestamp",
                related_time_series=["ts1", "ts2", "ts3", ""],
                static_covariates=None,
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message="Please do not provide empty strings in ``related_time_series``",
            ),
            TimeSeriesDataConfigCase(  # static_covariates contains blank string
                target_time_series="target_time_series",
                item_id="item_id",
                timestamp="timestamp",
                related_time_series=None,
                static_covariates=["scv4", "scv5", "scv6", ""],
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message="Please do not provide empty strings in ``static_covariates``",
            ),
            TimeSeriesDataConfigCase(  # dataset_format provided int case
                target_time_series=1,
                item_id=2,
                timestamp=3,
                related_time_series=[4, 5, 6],
                static_covariates=[7, 8, 9, 10],
                dataset_format=TimeSeriesJSONDatasetFormat.COLUMNS,
                error=ValueError,
                error_message="Dataset format should only be provided when data files are JSONs.",
            ),
            TimeSeriesDataConfigCase(  # dataset_format not provided str case
                target_time_series="target_time_series",
                item_id="item_id",
                timestamp="timestamp",
                related_time_series=["ts1", "ts2", "ts3"],
                static_covariates=["a", "b", "c", "d"],
                dataset_format=None,
                error=ValueError,
                error_message="Please provide a valid dataset format.",
            ),
            TimeSeriesDataConfigCase(  # dataset_format wrong type str case
                target_time_series="target_time_series",
                item_id="item_id",
                timestamp="timestamp",
                related_time_series=["ts1", "ts2", "ts3"],
                static_covariates=["a", "b", "c", "d"],
                dataset_format="made_up_format",
                error=ValueError,
                error_message="Please provide a valid dataset format.",
            ),
        ],
    )
    def test_time_series_data_config_invalid(self, test_case: TimeSeriesDataConfigCase):
        """
        GIVEN required parameters are incomplete or invalid
        WHEN TimeSeriesDataConfig constructor is called
        THEN the expected error and message are raised
        """
        with pytest.raises(test_case.error, match=test_case.error_message):
            TimeSeriesDataConfig(
                target_time_series=test_case.target_time_series,
                item_id=test_case.item_id,
                timestamp=test_case.timestamp,
                related_time_series=test_case.related_time_series,
                static_covariates=test_case.static_covariates,
                dataset_format=test_case.dataset_format,
            )

    @pytest.mark.parametrize("test_case", valid_ts_data_config_case_list)
    def test_data_config_with_time_series(self, test_case: TimeSeriesDataConfigCase):
        """
        GIVEN a TimeSeriesDataConfig object is created
        WHEN a DataConfig object is created and given valid params + the TimeSeriesDataConfig
        THEN the internal config dictionary matches what's expected
        """
        # currently TSX only supports json so skip non-json tests
        if isinstance(test_case.target_time_series, int):
            return
        # setup
        headers = ["item_id", "timestamp", "target_ts", "rts1", "scv1"]
        # construct expected output
        mock_ts_data_config_dict = {
            "target_time_series": test_case.target_time_series,
            "item_id": test_case.item_id,
            "timestamp": test_case.timestamp,
        }
        if isinstance(test_case.target_time_series, str):
            dataset_type = "application/json"
            mock_ts_data_config_dict["dataset_format"] = test_case.dataset_format.value
        else:
            dataset_type = "text/csv"
        if test_case.related_time_series:
            mock_ts_data_config_dict["related_time_series"] = test_case.related_time_series
        if test_case.static_covariates:
            mock_ts_data_config_dict["static_covariates"] = test_case.static_covariates
        expected_config = {
            "dataset_type": dataset_type,
            "headers": headers,
            "time_series_data_config": mock_ts_data_config_dict,
        }
        # GIVEN
        ts_data_config = Mock()
        ts_data_config.get_time_series_data_config.return_value = copy.deepcopy(
            mock_ts_data_config_dict
        )
        # WHEN
        data_config = DataConfig(
            s3_data_input_path="s3://path/to/input.csv",
            s3_output_path="s3://path/to/output",
            headers=headers,
            dataset_type=dataset_type,
            time_series_data_config=ts_data_config,
        )
        # THEN
        assert expected_config == data_config.get_config()


def test_bias_config():
    label_values = [1]
    facet_name = "F1"
    facet_threshold = 0.3
    group_name = "A151"

    data_bias_config = BiasConfig(
        label_values_or_threshold=label_values,
        facet_name=facet_name,
        facet_values_or_threshold=facet_threshold,
        group_name=group_name,
    )

    expected_config = {
        "label_values_or_threshold": label_values,
        "facet": [{"name_or_index": facet_name, "value_or_threshold": facet_threshold}],
        "group_variable": group_name,
    }
    assert expected_config == data_bias_config.get_config()


def test_invalid_bias_config():
    # Empty facet list,
    with pytest.raises(AssertionError, match="Please provide at least one facet"):
        BiasConfig(
            label_values_or_threshold=[1],
            facet_name=[],
        )

    # Two facets but only one value
    with pytest.raises(
        ValueError,
        match="The number of facet names doesn't match the number of facet values",
    ):
        BiasConfig(
            label_values_or_threshold=[1],
            facet_name=["Feature1", "Feature2"],
            facet_values_or_threshold=[[1]],
        )


@pytest.mark.parametrize(
    "facet_name,facet_values_or_threshold,expected_result",
    [
        # One facet, assume that it is binary and value 1 indicates the sensitive group
        [
            "Feature1",
            [1],
            {
                "facet": [{"name_or_index": "Feature1", "value_or_threshold": [1]}],
            },
        ],
        # The same facet as above, facet value is not specified. (Clarify will compute bias metrics
        # for each binary value).
        [
            "Feature1",
            None,
            {
                "facet": [{"name_or_index": "Feature1"}],
            },
        ],
        # Assume that the 2nd column (index 1, zero-based) of the dataset as facet, it has
        # four categories and two of them indicate the sensitive group.
        [
            1,
            ["category1, category2"],
            {
                "facet": [{"name_or_index": 1, "value_or_threshold": ["category1, category2"]}],
            },
        ],
        # The same facet as above, facet values are not specified. (Clarify will iterate
        # the categories and compute bias metrics for each category).
        [
            1,
            None,
            {
                "facet": [{"name_or_index": 1}],
            },
        ],
        # Assume that the facet is numeric value in range [0.0, 1.0]. Given facet threshold 0.5,
        # interval (0.5, 1.0] indicates the sensitive group.
        [
            "Feature3",
            [0.5],
            {
                "facet": [{"name_or_index": "Feature3", "value_or_threshold": [0.5]}],
            },
        ],
        # Multiple facets
        [
            ["Feature1", 1, "Feature3"],
            [[1], ["category1, category2"], [0.5]],
            {
                "facet": [
                    {"name_or_index": "Feature1", "value_or_threshold": [1]},
                    {
                        "name_or_index": 1,
                        "value_or_threshold": ["category1, category2"],
                    },
                    {"name_or_index": "Feature3", "value_or_threshold": [0.5]},
                ],
            },
        ],
        # Multiple facets, no value or threshold
        [
            ["Feature1", 1, "Feature3"],
            None,
            {
                "facet": [
                    {"name_or_index": "Feature1"},
                    {"name_or_index": 1},
                    {"name_or_index": "Feature3"},
                ],
            },
        ],
        # Multiple facets, specify values or threshold for some of them
        [
            ["Feature1", 1, "Feature3"],
            [[1], None, [0.5]],
            {
                "facet": [
                    {"name_or_index": "Feature1", "value_or_threshold": [1]},
                    {"name_or_index": 1},
                    {"name_or_index": "Feature3", "value_or_threshold": [0.5]},
                ],
            },
        ],
    ],
)
def test_facet_of_bias_config(facet_name, facet_values_or_threshold, expected_result):
    label_values = [1]
    bias_config = BiasConfig(
        label_values_or_threshold=label_values,
        facet_name=facet_name,
        facet_values_or_threshold=facet_values_or_threshold,
    )
    expected_config = {
        "label_values_or_threshold": label_values,
        **expected_result,
    }
    assert bias_config.get_config() == expected_config


@pytest.mark.parametrize(
    ("content_type", "accept_type"),
    [
        # All the combinations of content_type and accept_type should be acceptable
        ("text/csv", "text/csv"),
        ("application/jsonlines", "application/jsonlines"),
        ("text/csv", "application/json"),
        ("application/jsonlines", "application/json"),
        ("application/jsonlines", "text/csv"),
        ("application/json", "application/json"),
        ("application/json", "application/jsonlines"),
        ("application/json", "text/csv"),
        ("image/jpeg", "text/csv"),
        ("image/jpg", "text/csv"),
        ("image/png", "text/csv"),
        ("application/x-npy", "text/csv"),
    ],
)
def test_valid_model_config(content_type, accept_type):
    model_name = "xgboost-model"
    instance_type = "ml.c5.xlarge"
    instance_count = 1
    custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"
    target_model = "target_model_name"
    accelerator_type = "ml.eia1.medium"
    content_template = (
        '{"instances":$features}'
        if content_type == "application/jsonlines"
        else "$records" if content_type == "application/json" else None
    )
    record_template = "$features_kvp" if content_type == "application/json" else None
    model_config = ModelConfig(
        model_name=model_name,
        instance_type=instance_type,
        instance_count=instance_count,
        accept_type=accept_type,
        content_type=content_type,
        content_template=content_template,
        record_template=record_template,
        custom_attributes=custom_attributes,
        accelerator_type=accelerator_type,
        target_model=target_model,
    )
    expected_config = {
        "model_name": model_name,
        "instance_type": instance_type,
        "initial_instance_count": instance_count,
        "accept_type": accept_type,
        "content_type": content_type,
        "custom_attributes": custom_attributes,
        "accelerator_type": accelerator_type,
        "target_model": target_model,
    }
    if content_template is not None:
        expected_config["content_template"] = content_template
    if record_template is not None:
        expected_config["record_template"] = record_template
    assert expected_config == model_config.get_predictor_config()


@pytest.mark.parametrize(
    ("error", "content_type", "accept_type", "content_template", "record_template"),
    [
        (
            "Invalid accept_type invalid_accept_type. Please choose text/csv or application/jsonlines.",
            "text/csv",
            "invalid_accept_type",
            None,
            None,
        ),
        (
            "Invalid content_type invalid_content_type. Please choose text/csv or application/jsonlines.",
            "invalid_content_type",
            "text/csv",
            None,
            None,
        ),
        (
            "content_template field is required for content_type",
            "application/jsonlines",
            "text/csv",
            None,
            None,
        ),
        (
            "content_template and record_template are required for content_type",
            "application/json",
            "text/csv",
            None,
            None,
        ),
        (
            "content_template and record_template are required for content_type",
            "application/json",
            "text/csv",
            "$records",
            None,
        ),
        (
            r"Invalid content_template invalid_content_template. Please include a placeholder \$features.",
            "application/jsonlines",
            "text/csv",
            "invalid_content_template",
            None,
        ),
        (
            r"Invalid content_template invalid_content_template. Please include either placeholder "
            r"\$records or \$record.",
            "application/json",
            "text/csv",
            "invalid_content_template",
            "$features",
        ),
    ],
)
def test_invalid_model_config(error, content_type, accept_type, content_template, record_template):
    with pytest.raises(ValueError, match=error):
        ModelConfig(
            model_name="xgboost-model",
            instance_type="ml.c5.xlarge",
            instance_count=1,
            content_type=content_type,
            accept_type=accept_type,
            content_template=content_template,
            record_template=record_template,
        )


def test_invalid_model_config_with_bad_endpoint_name_prefix():
    with pytest.raises(ValueError) as error:
        ModelConfig(
            model_name="xgboost-model",
            instance_type="ml.c5.xlarge",
            instance_count=1,
            accept_type="invalid_accept_type",
            endpoint_name_prefix="~invalid_endpoint_prefix",
        )
    assert (
        "Invalid endpoint_name_prefix. Please follow pattern ^[a-zA-Z0-9](-*[a-zA-Z0-9])."
        in str(error.value)
    )


def test_model_predicted_label_config():
    label = "label"
    probability = "pr"
    probability_threshold = 0.2
    label_headers = ["success"]
    model_config = ModelPredictedLabelConfig(
        label=label,
        probability=probability,
        probability_threshold=probability_threshold,
        label_headers=label_headers,
    )
    pr_threshold, config = model_config.get_predictor_config()
    expected_config = {
        "label": label,
        "probability": probability,
        "label_headers": label_headers,
    }
    assert probability_threshold == pr_threshold
    assert expected_config == config


def test_invalid_model_predicted_label_config():
    with pytest.raises(TypeError) as error:
        ModelPredictedLabelConfig(
            probability_threshold="invalid",
        )
    assert (
        "Invalid probability_threshold invalid. Please choose one that can be cast to float."
        in str(error.value)
    )


class TestTimeSeriesModelConfig:
    def test_time_series_model_config(self):
        """
        GIVEN a valid forecast expression
        WHEN a TimeSeriesModelConfig is constructed with it
        THEN the predictor_config dictionary matches the expected
        """
        # GIVEN
        forecast = "results.[forecast]"  # mock JMESPath expression for forecast
        # create expected output
        expected_config = {
            "forecast": forecast,
        }
        # WHEN
        ts_model_config = TimeSeriesModelConfig(
            forecast,
        )
        # THEN
        assert ts_model_config.time_series_model_config == expected_config

    @pytest.mark.parametrize(
        ("forecast", "error", "error_message"),
        [
            (
                None,
                ValueError,
                "Please provide a string JMESPath expression for ``forecast``",
            ),
            (
                123,
                ValueError,
                "Please provide a string JMESPath expression for ``forecast``",
            ),
        ],
    )
    def test_time_series_model_config_invalid(
        self,
        forecast,
        error,
        error_message,
    ):
        """
        GIVEN invalid args for a TimeSeriesModelConfig
        WHEN TimeSeriesModelConfig constructor is called
        THEN The appropriate error is raised
        """
        with pytest.raises(error, match=error_message):
            TimeSeriesModelConfig(
                forecast=forecast,
            )

    @pytest.mark.parametrize(
        ("content_type", "accept_type"),
        [
            ("application/json", "application/json"),
            ("application/json", "application/jsonlines"),
            ("application/jsonlines", "application/json"),
            ("application/jsonlines", "application/jsonlines"),
        ],
    )
    def test_model_config_with_time_series(self, content_type, accept_type):
        """
        GIVEN valid fields for a ModelConfig and a TimeSeriesModelConfig
        WHEN a ModelConfig is constructed with them
        THEN actual predictor_config matches expected
        """
        # setup
        model_name = "xgboost-model"
        instance_type = "ml.c5.xlarge"
        instance_count = 1
        custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"
        target_model = "target_model_name"
        accelerator_type = "ml.eia1.medium"
        content_template = (
            '{"instances":$features}'
            if content_type == "application/jsonlines"
            else "$records" if content_type == "application/json" else None
        )
        record_template = "$features_kvp" if content_type == "application/json" else None
        # create mock config for TimeSeriesModelConfig
        forecast = "results.[forecast]"  # mock JMESPath expression for forecast
        mock_ts_model_config_dict = {
            "forecast": forecast,
        }
        # create expected config
        expected_config = {
            "model_name": model_name,
            "instance_type": instance_type,
            "initial_instance_count": instance_count,
            "accept_type": accept_type,
            "content_type": content_type,
            "custom_attributes": custom_attributes,
            "accelerator_type": accelerator_type,
            "target_model": target_model,
            "time_series_predictor_config": mock_ts_model_config_dict,
        }
        if content_template is not None:
            expected_config["content_template"] = content_template
        if record_template is not None:
            expected_config["record_template"] = record_template
        # GIVEN
        mock_ts_model_config = Mock()  # create mock TimeSeriesModelConfig object
        mock_ts_model_config.get_time_series_model_config.return_value = copy.deepcopy(
            mock_ts_model_config_dict
        )  # set the mock's get_config return value
        # WHEN
        model_config = ModelConfig(
            model_name=model_name,
            instance_type=instance_type,
            instance_count=instance_count,
            accept_type=accept_type,
            content_type=content_type,
            content_template=content_template,
            record_template=record_template,
            custom_attributes=custom_attributes,
            accelerator_type=accelerator_type,
            target_model=target_model,
            time_series_model_config=mock_ts_model_config,
        )
        # THEN
        assert expected_config == model_config.get_predictor_config()


@pytest.mark.parametrize(
    "baseline",
    [
        ([[0.26124998927116394, 0.2824999988079071, 0.06875000149011612]]),
        (
            {
                "instances": [
                    {"features": [0.26124998927116394, 0.2824999988079071, 0.06875000149011612]}
                ]
            }
        ),
    ],
)
def test_valid_shap_config(baseline):
    num_samples = 100
    agg_method = "mean_sq"
    use_logit = True
    seed = 123
    granularity = "sentence"
    language = "german"
    model_type = "IMAGE_CLASSIFICATION"
    num_segments = 2
    feature_extraction_method = "segmentation"
    segment_compactness = 10
    max_objects = 4
    iou_threshold = 0.5
    context = 1.0
    text_config = TextConfig(
        granularity=granularity,
        language=language,
    )
    image_config = ImageConfig(
        model_type=model_type,
        num_segments=num_segments,
        feature_extraction_method=feature_extraction_method,
        segment_compactness=segment_compactness,
        max_objects=max_objects,
        iou_threshold=iou_threshold,
        context=context,
    )
    shap_config = SHAPConfig(
        baseline=baseline,
        num_samples=num_samples,
        agg_method=agg_method,
        use_logit=use_logit,
        seed=seed,
        text_config=text_config,
        image_config=image_config,
    )
    expected_config = {
        "shap": {
            "baseline": baseline,
            "num_samples": num_samples,
            "agg_method": agg_method,
            "use_logit": use_logit,
            "save_local_shap_values": True,
            "seed": seed,
            "text_config": {
                "granularity": granularity,
                "language": language,
            },
            "image_config": {
                "model_type": model_type,
                "num_segments": num_segments,
                "feature_extraction_method": feature_extraction_method,
                "segment_compactness": segment_compactness,
                "max_objects": max_objects,
                "iou_threshold": iou_threshold,
                "context": context,
            },
        }
    }
    assert expected_config == shap_config.get_explainability_config()


def test_shap_config_features_to_explain():
    baseline = [1, 2, 3]
    num_samples = 100
    agg_method = "mean_sq"
    use_logit = True
    save_local_shap_values = True
    seed = 123
    features_to_explain = [0, 1]
    shap_config = SHAPConfig(
        baseline=baseline,
        num_samples=num_samples,
        agg_method=agg_method,
        use_logit=use_logit,
        save_local_shap_values=save_local_shap_values,
        seed=seed,
        features_to_explain=features_to_explain,
    )
    expected_config = {
        "shap": {
            "baseline": baseline,
            "num_samples": num_samples,
            "agg_method": agg_method,
            "use_logit": use_logit,
            "save_local_shap_values": save_local_shap_values,
            "seed": seed,
            "features_to_explain": features_to_explain,
        }
    }
    assert expected_config == shap_config.get_explainability_config()


def test_shap_config_no_baseline():
    num_samples = 100
    agg_method = "mean_sq"
    use_logit = True
    seed = 123
    shap_config = SHAPConfig(
        num_samples=num_samples,
        agg_method=agg_method,
        num_clusters=2,
        use_logit=use_logit,
        seed=seed,
    )
    expected_config = {
        "shap": {
            "num_samples": num_samples,
            "agg_method": agg_method,
            "num_clusters": 2,
            "use_logit": use_logit,
            "save_local_shap_values": True,
            "seed": seed,
        }
    }
    assert expected_config == shap_config.get_explainability_config()


def test_shap_config_no_parameters():
    shap_config = SHAPConfig()
    expected_config = {
        "shap": {
            "use_logit": False,
            "save_local_shap_values": True,
        }
    }
    assert expected_config == shap_config.get_explainability_config()


class AsymmetricShapleyValueConfigCase(NamedTuple):
    direction: str
    granularity: str
    num_samples: Optional[int] = None
    baseline: Optional[Union[str, Dict[str, Any]]] = None
    error: Exception = None
    error_message: str = None


class TestAsymmetricShapleyValueConfig:
    @pytest.mark.parametrize(
        "test_case",
        [
            AsymmetricShapleyValueConfigCase(  # cases for timewise granularity
                direction=direction,
                granularity="timewise",
                num_samples=None,
            )
            for direction in ASYM_SHAP_VAL_EXPLANATION_DIRECTIONS
        ]
        + [
            AsymmetricShapleyValueConfigCase(  # case for fine_grained granularity
                direction="chronological",
                granularity="fine_grained",
                num_samples=1,
            ),
            AsymmetricShapleyValueConfigCase(  # case for target time series baseline
                direction="chronological",
                granularity="timewise",
                baseline={"target_time_series": "mean"},
            ),
            AsymmetricShapleyValueConfigCase(  # case for related time series baseline
                direction="chronological",
                granularity="timewise",
                baseline={"related_time_series": "zero"},
            ),
        ],
    )
    def test_asymmetric_shapley_value_config(self, test_case: AsymmetricShapleyValueConfigCase):
        """
        GIVEN valid arguments for an AsymmetricShapleyValueConfig object
        WHEN AsymmetricShapleyValueConfig object is instantiated with those arguments
        THEN the asymmetric_shapley_value_config dictionary matches expected
        """
        # test case is GIVEN
        # construct expected config
        expected_config = {
            "direction": test_case.direction,
            "granularity": test_case.granularity,
        }
        if test_case.granularity == "fine_grained":
            expected_config["num_samples"] = test_case.num_samples
        if test_case.baseline:
            expected_config["baseline"] = test_case.baseline
        # WHEN
        asym_shap_val_config = AsymmetricShapleyValueConfig(
            direction=test_case.direction,
            granularity=test_case.granularity,
            num_samples=test_case.num_samples,
            baseline=test_case.baseline,
        )
        # THEN
        assert asym_shap_val_config.asymmetric_shapley_value_config == expected_config

    @pytest.mark.parametrize(
        "test_case",
        [
            AsymmetricShapleyValueConfigCase(  # case for invalid direction
                direction="non-directional",
                granularity="timewise",
                num_samples=None,
                error=ValueError,
                error_message="Please provide a valid explanation direction from: "
                + ", ".join(ASYM_SHAP_VAL_EXPLANATION_DIRECTIONS),
            ),
            AsymmetricShapleyValueConfigCase(  # case for fine_grained and no num_samples
                direction="chronological",
                granularity="fine_grained",
                num_samples=None,
                error=ValueError,
                error_message="Please provide an integer for ``num_samples``.",
            ),
            AsymmetricShapleyValueConfigCase(  # case for fine_grained and non-integer num_samples
                direction="chronological",
                granularity="fine_grained",
                num_samples="5",
                error=ValueError,
                error_message="Please provide an integer for ``num_samples``.",
            ),
            AsymmetricShapleyValueConfigCase(  # case for num_samples when non fine-grained explanation
                direction="chronological",
                granularity="timewise",
                num_samples=5,
                error=ValueError,
                error_message="``num_samples`` is only used for fine-grained explanations.",
            ),
            AsymmetricShapleyValueConfigCase(  # case for anti_chronological and fine_grained
                direction="anti_chronological",
                granularity="fine_grained",
                num_samples=5,
                error=ValueError,
                error_message="not supported together.",
            ),
            AsymmetricShapleyValueConfigCase(  # case for bidirectional and fine_grained
                direction="bidirectional",
                granularity="fine_grained",
                num_samples=5,
                error=ValueError,
                error_message="not supported together.",
            ),
            AsymmetricShapleyValueConfigCase(  # case for unsupported target time series baseline value
                direction="chronological",
                granularity="timewise",
                baseline={"target_time_series": "median"},
                error=ValueError,
                error_message="for ``target_time_series`` is invalid.",
            ),
            AsymmetricShapleyValueConfigCase(  # case for unsupported related time series baseline value
                direction="chronological",
                granularity="timewise",
                baseline={"related_time_series": "mode"},
                error=ValueError,
                error_message="for ``related_time_series`` is invalid.",
            ),
        ],
    )
    def test_asymmetric_shapley_value_config_invalid(self, test_case):
        """
        GIVEN invalid parameters for AsymmetricShapleyValue
        WHEN AsymmetricShapleyValueConfig constructor is called with them
        THEN the expected error and message are raised
        """
        # test case is GIVEN
        with pytest.raises(test_case.error, match=test_case.error_message):  # THEN
            AsymmetricShapleyValueConfig(  # WHEN
                direction=test_case.direction,
                granularity=test_case.granularity,
                num_samples=test_case.num_samples,
                baseline=test_case.baseline,
            )


def test_pdp_config():
    pdp_config = PDPConfig(features=["f1", "f2"], grid_resolution=20)
    expected_config = {
        "pdp": {"features": ["f1", "f2"], "grid_resolution": 20, "top_k_features": 10}
    }
    assert expected_config == pdp_config.get_explainability_config()


def test_text_config():
    granularity = "sentence"
    language = "german"
    text_config = TextConfig(
        granularity=granularity,
        language=language,
    )
    expected_config = {
        "granularity": granularity,
        "language": language,
    }
    assert expected_config == text_config.get_text_config()


def test_invalid_text_config():
    with pytest.raises(ValueError) as error:
        TextConfig(
            granularity="invalid",
            language="english",
        )
    assert (
        "Invalid granularity invalid. Please choose among ['token', 'sentence', 'paragraph']"
        in str(error.value)
    )
    with pytest.raises(ValueError) as error:
        TextConfig(
            granularity="token",
            language="invalid",
        )
    assert "Invalid language invalid. Please choose among ['chinese'," in str(error.value)


def test_image_config():
    model_type = "IMAGE_CLASSIFICATION"
    num_segments = 2
    feature_extraction_method = "segmentation"
    segment_compactness = 10
    max_objects = 4
    iou_threshold = 0.5
    context = 1.0
    image_config = ImageConfig(
        model_type=model_type,
        num_segments=num_segments,
        feature_extraction_method=feature_extraction_method,
        segment_compactness=segment_compactness,
        max_objects=max_objects,
        iou_threshold=iou_threshold,
        context=context,
    )
    expected_config = {
        "model_type": model_type,
        "num_segments": num_segments,
        "feature_extraction_method": feature_extraction_method,
        "segment_compactness": segment_compactness,
        "max_objects": max_objects,
        "iou_threshold": iou_threshold,
        "context": context,
    }

    assert expected_config == image_config.get_image_config()


def test_invalid_image_config():
    model_type = "OBJECT_SEGMENTATION"
    num_segments = 2
    with pytest.raises(ValueError) as error:
        ImageConfig(
            model_type=model_type,
            num_segments=num_segments,
        )
    assert (
        "Clarify SHAP only supports object detection and image classification methods. "
        "Please set model_type to OBJECT_DETECTION or IMAGE_CLASSIFICATION." in str(error.value)
    )


def test_invalid_shap_config():
    with pytest.raises(ValueError) as error:
        SHAPConfig(
            baseline=[[1]],
            num_samples=1,
            agg_method="invalid",
        )
    assert "Invalid agg_method invalid. Please choose mean_abs, median, or mean_sq." in str(
        error.value
    )
    with pytest.raises(ValueError) as error:
        SHAPConfig(baseline=[[1]], num_samples=1, agg_method="mean_abs", num_clusters=2)
    assert (
        "Baseline and num_clusters cannot be provided together. Please specify one of the two."
        in str(error.value)
    )
    with pytest.raises(ValueError) as error:
        SHAPConfig(
            baseline=[[1, 2]],
            num_samples=1,
            text_config=TextConfig(granularity="token", language="english"),
            features_to_explain=[0],
        )
    assert (
        "`features_to_explain` is not supported for datasets containing text features or images."
        in str(error.value)
    )


@pytest.fixture(scope="module")
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name="us-west-2")
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name="us-west-2",
        config=None,
        local_mode=False,
        default_bucket_prefix=None,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value="mybucket")
    session_mock.upload_data = Mock(
        name="upload_data", return_value="mocked_s3_uri_from_upload_data"
    )
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = "arn:aws:iam::012345678901:role/SageMakerRole"
    session_mock.sagemaker_config = {}
    return session_mock


@pytest.fixture(scope="module")
def clarify_processor(sagemaker_session):
    return SageMakerClarifyProcessor(
        role="AmazonSageMaker-ExecutionRole",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture(scope="module")
def clarify_processor_with_job_name_prefix(sagemaker_session):
    return SageMakerClarifyProcessor(
        role="AmazonSageMaker-ExecutionRole",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
        job_name_prefix=JOB_NAME_PREFIX,
    )


@pytest.fixture(scope="module")
def data_config():
    return DataConfig(
        s3_data_input_path="s3://input/train.csv",
        s3_output_path="s3://output/analysis_test_result",
        label="Label",
        headers=["Label", "F1", "F2", "F3", "F4"],
        dataset_type="text/csv",
        joinsource="F4",
    )


@pytest.fixture(scope="module")
def data_bias_config():
    return BiasConfig(
        label_values_or_threshold=[1],
        facet_name="F1",
        group_name="F2",
    )


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(
        model_name="xgboost-model",
        instance_type="ml.c5.xlarge",
        instance_count=1,
    )


@pytest.fixture(scope="module")
def model_predicted_label_config():
    return ModelPredictedLabelConfig()


@pytest.fixture(scope="module")
def shap_config():
    return SHAPConfig(
        baseline=[
            [
                0.26124998927116394,
                0.2824999988079071,
                0.06875000149011612,
            ],
        ],
        num_samples=100,
        agg_method="mean_sq",
    )


@pytest.fixture(scope="module")
def pdp_config():
    return PDPConfig(features=["F1", "F2"], grid_resolution=20)


def test_model_config_validations():
    new_model_endpoint_definition = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "instance_count": 1,
    }
    existing_endpoint_definition = {"endpoint_name": "existing_endpoint"}

    with pytest.raises(AssertionError):
        # should be one of them
        ModelConfig(
            **new_model_endpoint_definition,
            **existing_endpoint_definition,
        )

    with pytest.raises(AssertionError):
        # should be one of them
        ModelConfig(
            endpoint_name_prefix="prefix",
            **existing_endpoint_definition,
        )

    # success path for new model
    assert ModelConfig(**new_model_endpoint_definition).predictor_config == {
        "initial_instance_count": 1,
        "instance_type": "ml.c5.xlarge",
        "model_name": "xgboost-model",
    }

    # success path for existing endpoint
    assert (
        ModelConfig(**existing_endpoint_definition).predictor_config == existing_endpoint_definition
    )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_pre_training_bias(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    data_bias_config,
):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        clarify_processor.run_pre_training_bias(
            data_config,
            data_bias_config,
            wait=True,
            job_name="test",
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
                "F4",
            ],
            "joinsource_name_or_index": "F4",
            "label": "Label",
            "label_values_or_threshold": [1],
            "facet": [{"name_or_index": "F1"}],
            "group_variable": "F2",
            "methods": {
                "report": {"name": "report", "title": "Analysis Report"},
                "pre_training_bias": {"methods": "all"},
            },
        }
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        clarify_processor_with_job_name_prefix.run_pre_training_bias(
            data_config,
            data_bias_config,
            wait=True,
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        name_from_base.assert_called_with(JOB_NAME_PREFIX)
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            JOB_NAME,
            None,
            {"ExperimentName": "AnExperiment"},
        )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_post_training_bias(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    data_bias_config,
    model_config,
    model_predicted_label_config,
):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        clarify_processor.run_post_training_bias(
            data_config,
            data_bias_config,
            model_config,
            model_predicted_label_config,
            wait=True,
            job_name="test",
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
                "F4",
            ],
            "label": "Label",
            "label_values_or_threshold": [1],
            "joinsource_name_or_index": "F4",
            "facet": [{"name_or_index": "F1"}],
            "group_variable": "F2",
            "methods": {
                "report": {"name": "report", "title": "Analysis Report"},
                "post_training_bias": {"methods": "all"},
            },
            "predictor": {
                "model_name": "xgboost-model",
                "instance_type": "ml.c5.xlarge",
                "initial_instance_count": 1,
            },
        }
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        clarify_processor_with_job_name_prefix.run_post_training_bias(
            data_config,
            data_bias_config,
            model_config,
            model_predicted_label_config,
            wait=True,
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        name_from_base.assert_called_with(JOB_NAME_PREFIX)
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            JOB_NAME,
            None,
            {"ExperimentName": "AnExperiment"},
        )


@patch.object(Processor, "run")
def test_run_on_s3_analysis_config_file(
    processor_run, sagemaker_session, clarify_processor, data_config
):
    analysis_config = {
        "dataset_type": "text/csv",
        "methods": {"post_training_bias": {"methods": "all"}},
    }
    with patch("sagemaker.clarify._upload_analysis_config", return_value=None) as mock_method:
        clarify_processor._run(
            data_config,
            analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        analysis_config_file = mock_method.call_args[0][0]
        mock_method.assert_called_with(
            analysis_config_file, data_config.s3_output_path, sagemaker_session, None
        )

        data_config_with_analysis_config_output = DataConfig(
            s3_data_input_path="s3://input/train.csv",
            s3_output_path="s3://output/analysis_test_result",
            s3_analysis_config_output_path="s3://analysis_config_output",
            label="Label",
            headers=[
                "Label",
                "F1",
                "F2",
                "F3",
            ],
            dataset_type="text/csv",
        )
        clarify_processor._run(
            data_config_with_analysis_config_output,
            analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        analysis_config_file = mock_method.call_args[0][0]
        mock_method.assert_called_with(
            analysis_config_file,
            data_config_with_analysis_config_output.s3_analysis_config_output_path,
            sagemaker_session,
            None,
        )


def _run_test_explain(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
    pdp_config,
    model_scores,
    expected_predictor_config,
    expected_text_config=None,
    expected_image_config=None,
):
    with patch.object(SageMakerClarifyProcessor, "_run", return_value=None) as mock_method:
        explanation_configs = None
        if shap_config and pdp_config:
            explanation_configs = [shap_config, pdp_config]
        elif shap_config:
            explanation_configs = shap_config
        elif pdp_config:
            explanation_configs = pdp_config

        clarify_processor.run_explainability(
            data_config,
            model_config,
            explanation_configs,
            model_scores=model_scores,
            wait=True,
            job_name="test",
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        expected_analysis_config = {
            "dataset_type": "text/csv",
            "headers": [
                "Label",
                "F1",
                "F2",
                "F3",
                "F4",
            ],
            "label": "Label",
            "joinsource_name_or_index": "F4",
            "predictor": expected_predictor_config,
        }
        expected_explanation_configs = {}
        if shap_config:
            expected_explanation_configs["shap"] = {
                "baseline": [
                    [
                        0.26124998927116394,
                        0.2824999988079071,
                        0.06875000149011612,
                    ]
                ],
                "num_samples": 100,
                "agg_method": "mean_sq",
                "use_logit": False,
                "save_local_shap_values": True,
            }
            if expected_text_config:
                expected_explanation_configs["shap"]["text_config"] = expected_text_config
            if expected_image_config:
                expected_explanation_configs["shap"]["image_config"] = expected_image_config
        if pdp_config:
            expected_explanation_configs["pdp"] = {
                "features": ["F1", "F2"],
                "grid_resolution": 20,
                "top_k_features": 10,
            }
        expected_analysis_config["methods"] = {
            "report": {"name": "report", "title": "Analysis Report"},
            **expected_explanation_configs,
        }
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            "test",
            None,
            {"ExperimentName": "AnExperiment"},
        )
        clarify_processor_with_job_name_prefix.run_explainability(
            data_config,
            model_config,
            explanation_configs,
            model_scores=model_scores,
            wait=True,
            experiment_config={"ExperimentName": "AnExperiment"},
        )
        name_from_base.assert_called_with(JOB_NAME_PREFIX)
        mock_method.assert_called_with(
            data_config,
            expected_analysis_config,
            True,
            True,
            JOB_NAME,
            None,
            {"ExperimentName": "AnExperiment"},
        )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_pdp(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
    pdp_config,
):
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    _run_test_explain(
        name_from_base,
        clarify_processor,
        clarify_processor_with_job_name_prefix,
        data_config,
        model_config,
        None,
        pdp_config,
        None,
        expected_predictor_config,
    )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_shap(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
):
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    _run_test_explain(
        name_from_base,
        clarify_processor,
        clarify_processor_with_job_name_prefix,
        data_config,
        model_config,
        shap_config,
        None,
        None,
        expected_predictor_config,
    )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_explainability_with_invalid_config(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
):
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    with pytest.raises(
        AttributeError,
        match="analysis_config must have at least one working method: "
        "One of the `pre_training_methods`, `post_training_methods`, `explainability_config`.",
    ):
        _run_test_explain(
            name_from_base,
            clarify_processor,
            clarify_processor_with_job_name_prefix,
            data_config,
            model_config,
            None,
            None,
            None,
            expected_predictor_config,
        )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_explainability_with_multiple_shap_config(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
):
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    with pytest.raises(ValueError, match="Duplicate explainability configs are provided"):
        second_shap_config = copy.deepcopy(shap_config)
        second_shap_config.shap_config["num_samples"] = 200
        _run_test_explain(
            name_from_base,
            clarify_processor,
            clarify_processor_with_job_name_prefix,
            data_config,
            model_config,
            [shap_config, second_shap_config],
            None,
            None,
            expected_predictor_config,
        )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_shap_with_predicted_label(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
    shap_config,
    pdp_config,
):
    probability = "pr"
    label_headers = ["success"]
    model_scores = ModelPredictedLabelConfig(
        probability=probability,
        label_headers=label_headers,
    )
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
        "probability": probability,
        "label_headers": label_headers,
    }
    _run_test_explain(
        name_from_base,
        clarify_processor,
        clarify_processor_with_job_name_prefix,
        data_config,
        model_config,
        shap_config,
        pdp_config,
        model_scores,
        expected_predictor_config,
    )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_shap_with_text_config(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
):
    granularity = "paragraph"
    language = "ukrainian"

    shap_config = SHAPConfig(
        baseline=[
            [
                0.26124998927116394,
                0.2824999988079071,
                0.06875000149011612,
            ]
        ],
        num_samples=100,
        agg_method="mean_sq",
        text_config=TextConfig(granularity, language),
    )

    expected_text_config = {
        "granularity": granularity,
        "language": language,
    }
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }
    _run_test_explain(
        name_from_base,
        clarify_processor,
        clarify_processor_with_job_name_prefix,
        data_config,
        model_config,
        shap_config,
        None,
        None,
        expected_predictor_config,
        expected_text_config=expected_text_config,
    )


@patch("sagemaker.utils.name_from_base", return_value=JOB_NAME)
def test_shap_with_image_config(
    name_from_base,
    clarify_processor,
    clarify_processor_with_job_name_prefix,
    data_config,
    model_config,
):
    model_type = "IMAGE_CLASSIFICATION"
    num_segments = 2
    feature_extraction_method = "segmentation"
    segment_compactness = 10
    max_objects = 4
    iou_threshold = 0.5
    context = 1.0
    image_config = ImageConfig(
        model_type=model_type,
        num_segments=num_segments,
        feature_extraction_method=feature_extraction_method,
        segment_compactness=segment_compactness,
        max_objects=max_objects,
        iou_threshold=iou_threshold,
        context=context,
    )

    shap_config = SHAPConfig(
        baseline=[
            [
                0.26124998927116394,
                0.2824999988079071,
                0.06875000149011612,
            ]
        ],
        num_samples=100,
        agg_method="mean_sq",
        image_config=image_config,
    )

    expected_image_config = {
        "model_type": model_type,
        "num_segments": num_segments,
        "feature_extraction_method": feature_extraction_method,
        "segment_compactness": segment_compactness,
        "max_objects": max_objects,
        "iou_threshold": iou_threshold,
        "context": context,
    }
    expected_predictor_config = {
        "model_name": "xgboost-model",
        "instance_type": "ml.c5.xlarge",
        "initial_instance_count": 1,
    }

    _run_test_explain(
        name_from_base,
        clarify_processor,
        clarify_processor_with_job_name_prefix,
        data_config,
        model_config,
        shap_config,
        None,
        None,
        expected_predictor_config,
        expected_image_config=expected_image_config,
    )


def test_analysis_config_generator_for_explainability(data_config, model_config):
    model_scores = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.explainability(
        data_config,
        model_config,
        model_scores,
        SHAPConfig(),
    )
    expected = {
        "dataset_type": "text/csv",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "shap": {"save_local_shap_values": True, "use_logit": False},
        },
        "predictor": {
            "initial_instance_count": 1,
            "instance_type": "ml.c5.xlarge",
            "label_headers": ["success"],
            "model_name": "xgboost-model",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_explainability_failing(data_config, model_config):
    model_scores = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    with pytest.raises(
        ValueError,
        match="PDP features must be provided when ShapConfig is not provided",
    ):
        _AnalysisConfigGenerator.explainability(
            data_config,
            model_config,
            model_scores,
            PDPConfig(),
        )

    with pytest.raises(ValueError, match="Duplicate explainability configs are provided"):
        _AnalysisConfigGenerator.explainability(
            data_config,
            model_config,
            model_scores,
            [SHAPConfig(), SHAPConfig()],
        )

    with pytest.raises(
        AttributeError,
        match="analysis_config must have at least one working method: "
        "One of the "
        "`pre_training_methods`, `post_training_methods`, `explainability_config`.",
    ):
        _AnalysisConfigGenerator.explainability(
            data_config,
            model_config,
            model_scores,
            [],
        )


def test_analysis_config_generator_for_bias_explainability(
    data_config, data_bias_config, model_config
):
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias_and_explainability(
        data_config,
        model_config,
        model_predicted_label_config,
        [SHAPConfig(), PDPConfig()],
        data_bias_config,
        pre_training_methods="all",
        post_training_methods="all",
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "pdp": {"grid_resolution": 15, "top_k_features": 10},
            "post_training_bias": {"methods": "all"},
            "pre_training_bias": {"methods": "all"},
            "report": {"name": "report", "title": "Analysis Report"},
            "shap": {"save_local_shap_values": True, "use_logit": False},
        },
        "predictor": {
            "initial_instance_count": 1,
            "instance_type": "ml.c5.xlarge",
            "label_headers": ["success"],
            "model_name": "xgboost-model",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_bias_explainability_with_existing_endpoint(
    data_config, data_bias_config
):
    model_config = ModelConfig(endpoint_name="existing_endpoint_name")
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias_and_explainability(
        data_config,
        model_config,
        model_predicted_label_config,
        [SHAPConfig(), PDPConfig()],
        data_bias_config,
        pre_training_methods="all",
        post_training_methods="all",
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "pdp": {"grid_resolution": 15, "top_k_features": 10},
            "post_training_bias": {"methods": "all"},
            "pre_training_bias": {"methods": "all"},
            "report": {"name": "report", "title": "Analysis Report"},
            "shap": {"save_local_shap_values": True, "use_logit": False},
        },
        "predictor": {
            "label_headers": ["success"],
            "endpoint_name": "existing_endpoint_name",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_bias_pre_training(data_config, data_bias_config):
    actual = _AnalysisConfigGenerator.bias_pre_training(
        data_config, data_bias_config, methods="all"
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "pre_training_bias": {"methods": "all"},
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_bias_post_training(
    data_config, data_bias_config, model_config
):
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias_post_training(
        data_config,
        data_bias_config,
        model_predicted_label_config,
        methods="all",
        model_config=model_config,
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "post_training_bias": {"methods": "all"},
        },
        "predictor": {
            "initial_instance_count": 1,
            "instance_type": "ml.c5.xlarge",
            "label_headers": ["success"],
            "model_name": "xgboost-model",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_generator_for_bias(data_config, data_bias_config, model_config):
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias(
        data_config,
        data_bias_config,
        model_config,
        model_predicted_label_config,
        pre_training_methods="all",
        post_training_methods="all",
    )
    expected = {
        "dataset_type": "text/csv",
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "joinsource_name_or_index": "F4",
        "label": "Label",
        "label_values_or_threshold": [1],
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "post_training_bias": {"methods": "all"},
            "pre_training_bias": {"methods": "all"},
        },
        "predictor": {
            "initial_instance_count": 1,
            "instance_type": "ml.c5.xlarge",
            "label_headers": ["success"],
            "model_name": "xgboost-model",
            "probability": "pr",
        },
    }
    assert actual == expected


def test_analysis_config_for_bias_no_model_config(data_bias_config):
    s3_data_input_path = "s3://path/to/input.csv"
    s3_output_path = "s3://path/to/output"
    predicted_labels_uri = "s3://path/to/predicted_labels.csv"
    label_name = "Label"
    headers = [
        "Label",
        "F1",
        "F2",
        "F3",
        "F4",
    ]
    dataset_type = "text/csv"
    data_config = DataConfig(
        s3_data_input_path=s3_data_input_path,
        s3_output_path=s3_output_path,
        label=label_name,
        headers=headers,
        dataset_type=dataset_type,
        predicted_label_dataset_uri=predicted_labels_uri,
        predicted_label_headers=["PredictedLabel"],
        predicted_label="PredictedLabel",
    )
    model_config = None
    model_predicted_label_config = ModelPredictedLabelConfig(
        probability="pr",
        probability_threshold=0.8,
        label_headers=["success"],
    )
    actual = _AnalysisConfigGenerator.bias(
        data_config,
        data_bias_config,
        model_config,
        model_predicted_label_config,
        pre_training_methods="all",
        post_training_methods="all",
    )
    expected = {
        "dataset_type": "text/csv",
        "headers": ["Label", "F1", "F2", "F3", "F4"],
        "label": "Label",
        "predicted_label_dataset_uri": "s3://path/to/predicted_labels.csv",
        "predicted_label_headers": ["PredictedLabel"],
        "predicted_label": "PredictedLabel",
        "label_values_or_threshold": [1],
        "facet": [{"name_or_index": "F1"}],
        "group_variable": "F2",
        "methods": {
            "report": {"name": "report", "title": "Analysis Report"},
            "pre_training_bias": {"methods": "all"},
            "post_training_bias": {"methods": "all"},
        },
        "probability_threshold": 0.8,
    }
    assert actual == expected


def test_invalid_analysis_config(data_config, data_bias_config, model_config):
    with pytest.raises(
        ValueError, match="model_config must be provided when explainability methods are selected."
    ):
        _AnalysisConfigGenerator.bias_and_explainability(
            data_config=data_config,
            model_config=None,
            model_predicted_label_config=ModelPredictedLabelConfig(),
            explainability_config=SHAPConfig(),
            bias_config=data_bias_config,
            pre_training_methods="all",
            post_training_methods="all",
        )

    with pytest.raises(
        ValueError,
        match="model_config must be provided when `predicted_label_dataset_uri` or "
        "`predicted_label` are not provided in data_config.",
    ):
        _AnalysisConfigGenerator.bias(
            data_config=data_config,
            model_config=None,
            model_predicted_label_config=ModelPredictedLabelConfig(),
            bias_config=data_bias_config,
            pre_training_methods="all",
            post_training_methods="all",
        )


def _build_pdp_config_mock():
    pdp_config_dict = {
        "pdp": {
            "grid_resolution": 15,
            "top_k_features": 10,
            "features": [
                "some",
                "features",
            ],
        }
    }
    pdp_config = Mock(spec=PDPConfig)
    pdp_config.get_explainability_config.return_value = pdp_config_dict
    return pdp_config


def _build_asymmetric_shapley_value_config_mock():
    asym_shap_val_config_dict = {
        "direction": "chronological",
        "granularity": "fine_grained",
        "num_samples": 20,
    }
    asym_shap_val_config = Mock(spec=AsymmetricShapleyValueConfig)
    asym_shap_val_config.get_explainability_config.return_value = {
        "asymmetric_shapley_value": asym_shap_val_config_dict
    }
    return asym_shap_val_config


def _build_data_config_mock():
    """
    Builds a mock DataConfig for the time series _AnalysisConfigGenerator unit tests.
    """
    # setup a time_series_data_config dictionary
    time_series_data_config = {
        "target_time_series": "target_ts",
        "item_id": "id",
        "timestamp": "timestamp",
        "related_time_series": ["rts1", "rts2", "rts3"],
        "static_covariates": ["scv1", "scv2", "scv3"],
        "dataset_format": TimeSeriesJSONDatasetFormat.COLUMNS,
    }
    # setup DataConfig mock
    data_config = Mock(spec=DataConfig)
    data_config.analysis_config = {"time_series_data_config": time_series_data_config}
    return data_config


def _build_model_config_mock():
    """
    Builds a mock ModelConfig for the time series _AnalysisConfigGenerator unit tests.
    """
    time_series_model_config = {"forecast": "mean"}
    model_config = Mock(spec=ModelConfig)
    model_config.predictor_config = {"time_series_predictor_config": time_series_model_config}
    return model_config


@dataclass
class ValidateTSXBaselineCase:
    explainability_config: AsymmetricShapleyValueConfig
    data_config: DataConfig
    error: Optional[Exception] = None
    error_msg: Optional[str] = None


class TestAnalysisConfigGeneratorForTimeSeriesExplainability:
    @patch(
        "sagemaker.clarify._AnalysisConfigGenerator._validate_time_series_static_covariates_baseline"
    )
    @patch("sagemaker.clarify._AnalysisConfigGenerator._add_methods")
    @patch("sagemaker.clarify._AnalysisConfigGenerator._add_predictor")
    def test_explainability_for_time_series(self, _add_predictor, _add_methods, _validate_ts_scv):
        """
        GIVEN a valid DataConfig and ModelConfig that contain time_series_data_config and
            time_series_model_config respectively as well as an AsymmetricShapleyValueConfig
        WHEN _AnalysisConfigGenerator.explainability() is called with those args
        THEN _add_predictor and _add methods calls are as expected
        """
        # GIVEN
        # get DataConfig mock
        data_config_mock = _build_data_config_mock()
        # get ModelConfig mock
        model_config_mock = _build_model_config_mock()
        # get AsymmetricShapleyValueConfig mock for explainability_config
        explainability_config = _build_asymmetric_shapley_value_config_mock()
        # get time_series_data_config dict from mock
        time_series_data_config = copy.deepcopy(
            data_config_mock.analysis_config.get("time_series_data_config")
        )
        # get time_series_predictor_config from mock
        time_series_model_config = copy.deepcopy(
            model_config_mock.predictor_config.get("time_series_model_config")
        )
        # setup _add_predictor call to return what would be expected at that stage
        analysis_config_after_add_predictor = {
            "time_series_data_config": time_series_data_config,
            "time_series_predictor_config": time_series_model_config,
        }
        _add_predictor.return_value = analysis_config_after_add_predictor
        # WHEN
        _AnalysisConfigGenerator.explainability(
            data_config=data_config_mock,
            model_config=model_config_mock,
            model_predicted_label_config=None,
            explainability_config=explainability_config,
        )
        # THEN
        _add_predictor.assert_called_once_with(
            data_config_mock.analysis_config,
            model_config_mock,
            ANY,
        )
        _add_methods.assert_called_once_with(
            ANY,
            explainability_config=explainability_config,
        )
        _validate_ts_scv.assert_called_once_with(
            explainability_config=explainability_config,
            data_config=data_config_mock,
        )

    def test_explainability_for_time_series_invalid(self):
        # data config mocks
        data_config_with_ts = _build_data_config_mock()
        data_config_without_ts = Mock(spec=DataConfig)
        data_config_without_ts.analysis_config = dict()
        # model config mocks
        model_config_with_ts = _build_model_config_mock()
        model_config_without_ts = Mock(spec=ModelConfig)
        model_config_without_ts.predictor_config = dict()
        # asymmetric shapley value config mock (for ts)
        asym_shap_val_config_mock = _build_asymmetric_shapley_value_config_mock()
        # pdp config mock (for non-ts)
        pdp_config_mock = _build_pdp_config_mock()
        # case 1: ASV (ts case) and no timeseries data config given
        with pytest.raises(
            ValueError, match="Please provide a TimeSeriesDataConfig to DataConfig."
        ):
            _AnalysisConfigGenerator.explainability(
                data_config=data_config_without_ts,
                model_config=model_config_with_ts,
                model_predicted_label_config=None,
                explainability_config=asym_shap_val_config_mock,
            )
        # case 2: ASV (ts case) and no timeseries model config given
        with pytest.raises(
            ValueError, match="Please provide a TimeSeriesModelConfig to ModelConfig."
        ):
            _AnalysisConfigGenerator.explainability(
                data_config=data_config_with_ts,
                model_config=model_config_without_ts,
                model_predicted_label_config=None,
                explainability_config=asym_shap_val_config_mock,
            )
        # case 3: pdp (non ts case) and timeseries data config given
        with pytest.raises(ValueError, match="please do not provide a TimeSeriesDataConfig."):
            _AnalysisConfigGenerator.explainability(
                data_config=data_config_with_ts,
                model_config=model_config_without_ts,
                model_predicted_label_config=None,
                explainability_config=pdp_config_mock,
            )
        # case 4: pdp (non ts case) and timeseries model config given
        with pytest.raises(ValueError, match="please do not provide a TimeSeriesModelConfig."):
            _AnalysisConfigGenerator.explainability(
                data_config=data_config_without_ts,
                model_config=model_config_with_ts,
                model_predicted_label_config=None,
                explainability_config=pdp_config_mock,
            )

    def test_bias_and_explainability_invalid_for_time_series(self):
        """
        GIVEN user provides TimeSeriesDataConfig, TimeSeriesModelConfig, and/or
            AsymmetricShapleyValueConfig for DataConfig, ModelConfig, and as explainability_config
            respectively
        WHEN _AnalysisConfigGenerator.bias_and_explainability is called
        THEN the appropriate error is raised
        """
        # data config mocks
        data_config_with_ts = _build_data_config_mock()
        data_config_without_ts = Mock(spec=DataConfig)
        data_config_without_ts.analysis_config = dict()
        # model config mocks
        model_config_with_ts = _build_model_config_mock()
        model_config_without_ts = Mock(spec=ModelConfig)
        model_config_without_ts.predictor_config = dict()
        # asymmetric shap config mock (for ts)
        asym_shap_val_config_mock = _build_asymmetric_shapley_value_config_mock()
        # pdp config mock (for non-ts)
        pdp_config_mock = _build_pdp_config_mock()
        # case 1: asymmetric shap is given as explainability_config
        with pytest.raises(ValueError, match="Bias metrics are unsupported for time series."):
            _AnalysisConfigGenerator.bias_and_explainability(
                data_config=data_config_without_ts,
                model_config=model_config_without_ts,
                model_predicted_label_config=None,
                explainability_config=asym_shap_val_config_mock,
                bias_config=None,
            )
        # case 2: TimeSeriesModelConfig given to ModelConfig
        with pytest.raises(ValueError, match="Bias metrics are unsupported for time series."):
            _AnalysisConfigGenerator.bias_and_explainability(
                data_config=data_config_without_ts,
                model_config=model_config_with_ts,
                model_predicted_label_config=None,
                explainability_config=pdp_config_mock,
                bias_config=None,
            )
        # case 3: TimeSeriesDataConfig given to DataConfig
        with pytest.raises(ValueError, match="Bias metrics are unsupported for time series."):
            _AnalysisConfigGenerator.bias_and_explainability(
                data_config=data_config_with_ts,
                model_config=model_config_without_ts,
                model_predicted_label_config=None,
                explainability_config=pdp_config_mock,
                bias_config=None,
            )

    @pytest.mark.parametrize(
        ("mock_config", "error", "error_message"),
        [
            (  # single asym shap config for non TSX
                _build_asymmetric_shapley_value_config_mock(),
                ValueError,
                "Please do not provide Asymmetric Shapley Value configs for non-TimeSeries uses.",
            ),
            (  # list with asym shap config for non-TSX
                [
                    _build_asymmetric_shapley_value_config_mock(),
                    _build_pdp_config_mock(),
                ],
                ValueError,
                "Please do not provide Asymmetric Shapley Value configs for non-TimeSeries uses.",
            ),
        ],
    )
    def test_merge_explainability_configs_with_timeseries_invalid(
        self,
        mock_config,
        error,
        error_message,
    ):
        """
        GIVEN _merge_explainability_configs is called with a explainability config or list thereof
        WHEN explainability_config is or contains an AsymmetricShapleyValueConfig
        THEN the function will raise the appropriate error
        """
        with pytest.raises(error, match=error_message):
            _AnalysisConfigGenerator._merge_explainability_configs(
                explainability_config=mock_config,
            )

    @pytest.mark.parametrize(
        "case",
        [
            ValidateTSXBaselineCase(
                explainability_config=AsymmetricShapleyValueConfig(
                    direction="chronological",
                    granularity="timewise",
                    baseline={
                        "target_time_series": "zero",
                        "related_time_series": "zero",
                        "static_covariates": {
                            "item1": [0.0, 0.5, 1.0],
                            "item2": [0.3, 0.6, 0.9],
                            "item3": [0.0, 1.0, 1.0],
                            "item4": [0.9, 0.6, 0.3],
                            "item5": [1.0, 0.5, 0.0],
                        },
                    },
                ),
                data_config=DataConfig(
                    s3_data_input_path="s3://data/input",
                    s3_output_path="s3://data/output",
                    headers=["id", "time", "tts", "rts_1", "rts_2", "scv1", "scv2", "scv3"],
                    dataset_type="application/json",
                    time_series_data_config=TimeSeriesDataConfig(
                        item_id="[].id",
                        timestamp="[].temporal[].timestamp",
                        target_time_series="[].temporal[].target",
                        related_time_series=["[].temporal[].rts_1", "[].temporal[].rts_2"],
                        static_covariates=["[].cov_1", "[].cov_2", "[].cov_3"],
                        dataset_format=TimeSeriesJSONDatasetFormat.ITEM_RECORDS,
                    ),
                ),
            ),
        ],
    )
    def test_time_series_baseline_valid_static_covariates(self, case: ValidateTSXBaselineCase):
        """
        GIVEN AsymmetricShapleyValueConfig and TimeSeriesDataConfig are created and a baseline
            is provided
        WHEN AnalysisConfigGenerator._validate_time_series_static_covariates_baseline() is called
        THEN no error is raised
        """
        _AnalysisConfigGenerator._validate_time_series_static_covariates_baseline(
            explainability_config=case.explainability_config,
            data_config=case.data_config,
        )

    @pytest.mark.parametrize(
        "case",
        [
            ValidateTSXBaselineCase(  # some item ids are missing baseline values
                explainability_config=AsymmetricShapleyValueConfig(
                    direction="chronological",
                    granularity="timewise",
                    baseline={
                        "target_time_series": "zero",
                        "related_time_series": "zero",
                        "static_covariates": {
                            "item1": [0.0, 0.5, 1.0],
                            "item2": [0.3, 0.6, 0.9],
                            "item3": [0.0],
                            "item4": [0.9, 0.6, 0.3],
                            "item5": [1.0],
                        },
                    },
                ),
                data_config=DataConfig(
                    s3_data_input_path="s3://data/input",
                    s3_output_path="s3://data/output",
                    headers=["id", "time", "tts", "rts_1", "rts_2", "scv1", "scv2", "scv3"],
                    dataset_type="application/json",
                    time_series_data_config=TimeSeriesDataConfig(
                        item_id="[].id",
                        timestamp="[].temporal[].timestamp",
                        target_time_series="[].temporal[].target",
                        related_time_series=["[].temporal[].rts_1", "[].temporal[].rts_2"],
                        static_covariates=["[].cov_1", "[].cov_2", "[].cov_3"],
                        dataset_format=TimeSeriesJSONDatasetFormat.ITEM_RECORDS,
                    ),
                ),
                error=ValueError,
                error_msg="baseline entry for item3 does not match number",
            ),
            ValidateTSXBaselineCase(  # no static covariates are in data config
                explainability_config=AsymmetricShapleyValueConfig(
                    direction="chronological",
                    granularity="timewise",
                    baseline={
                        "target_time_series": "zero",
                        "related_time_series": "zero",
                        "static_covariates": {
                            "item1": [0.0, 0.5, 1.0],
                            "item2": [0.3, 0.6, 0.9],
                            "item3": [0.0, 1.0, 1.0],
                            "item4": [0.9, 0.6, 0.3],
                            "item5": [1.0, 0.5, 0.0],
                        },
                    },
                ),
                data_config=DataConfig(
                    s3_data_input_path="s3://data/input",
                    s3_output_path="s3://data/output",
                    headers=["id", "time", "tts", "rts_1", "rts_2"],
                    dataset_type="application/json",
                    time_series_data_config=TimeSeriesDataConfig(
                        item_id="[].id",
                        timestamp="[].temporal[].timestamp",
                        target_time_series="[].temporal[].target",
                        related_time_series=["[].temporal[].rts_1", "[].temporal[].rts_2"],
                        dataset_format=TimeSeriesJSONDatasetFormat.ITEM_RECORDS,
                    ),
                ),
                error=ValueError,
                error_msg="no static covariate columns are provided in TimeSeriesDataConfig",
            ),
            ValidateTSXBaselineCase(  # some item ids do not have a list as their baseline
                explainability_config=AsymmetricShapleyValueConfig(
                    direction="chronological",
                    granularity="timewise",
                    baseline={
                        "target_time_series": "zero",
                        "related_time_series": "zero",
                        "static_covariates": {
                            "item1": [0.0, 0.5, 1.0],
                            "item2": [0.3, 0.6, 0.9],
                            "item3": [0.0, 1.0, 1.0],
                            "item4": [0.9, 0.6, 0.3],
                            "item5": {"cov_1": 1.0, "cov_2": 0.5, "cov_3": 0.0},
                        },
                    },
                ),
                data_config=DataConfig(
                    s3_data_input_path="s3://data/input",
                    s3_output_path="s3://data/output",
                    headers=["id", "time", "tts", "rts_1", "rts_2", "scv1", "scv2", "scv3"],
                    dataset_type="application/json",
                    time_series_data_config=TimeSeriesDataConfig(
                        item_id="[].id",
                        timestamp="[].temporal[].timestamp",
                        target_time_series="[].temporal[].target",
                        related_time_series=["[].temporal[].rts_1", "[].temporal[].rts_2"],
                        static_covariates=["[].cov_1", "[].cov_2", "[].cov_3"],
                        dataset_format=TimeSeriesJSONDatasetFormat.ITEM_RECORDS,
                    ),
                ),
                error=ValueError,
                error_msg="Baseline entry for item5 must be a list",
            ),
        ],
    )
    def test_time_series_baseline_invalid_static_covariates(self, case: ValidateTSXBaselineCase):
        """
        GIVEN AsymmetricShapleyValueConfig and TimeSeriesDataConfig are created and a baseline
            is provided where the static covariates baseline values are misconfigured
        WHEN AnalysisConfigGenerator._validate_time_series_static_covariates_baseline() is called
        THEN the appropriate error is raised
        """
        with pytest.raises(case.error, match=case.error_msg):
            _AnalysisConfigGenerator._validate_time_series_static_covariates_baseline(
                explainability_config=case.explainability_config,
                data_config=case.data_config,
            )


class TestProcessingOutputHandler:
    def test_get_s3_upload_mode_image(self):
        analysis_config = {"dataset_type": DatasetType.IMAGE.value}
        s3_upload_mode = ProcessingOutputHandler.get_s3_upload_mode(analysis_config)
        assert s3_upload_mode == ProcessingOutputHandler.S3UploadMode.CONTINUOUS.value

    def test_get_s3_upload_mode_text(self):
        analysis_config = {"dataset_type": DatasetType.TEXTCSV.value}
        s3_upload_mode = ProcessingOutputHandler.get_s3_upload_mode(analysis_config)
        assert s3_upload_mode == ProcessingOutputHandler.S3UploadMode.ENDOFJOB.value
