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
"""Unit tests for to_dataframe() on the recommendation views."""
from __future__ import absolute_import

from types import SimpleNamespace

import pytest

from sagemaker.serve.ai_inference_recommender._recommendation_view import (
    _RecommendationsView,
    _RecommendationView,
)

pd = pytest.importorskip("pandas")


def _perf(metric, stat, value, unit):
    return SimpleNamespace(metric=metric, stat=stat, value=value, unit=unit)


def _rec_row(
    spec_name="A",
    instance_type="ml.g6.12xlarge",
    instance_count=1,
    perf=None,
    image_uri=".../djl-inference:0.36.0-lmi25.0.0-cu130",
):
    return SimpleNamespace(
        model_details=SimpleNamespace(
            model_package_arn="arn:aws:sm:us-west-2:1:model-package/p/1",
            inference_specification_name=spec_name,
        ),
        deployment_configuration=SimpleNamespace(
            instance_type=instance_type,
            instance_count=instance_count,
            copy_count_per_instance=1,
            image_uri=image_uri,
        ),
        expected_performance=perf or [],
    )


class TestRecommendationViewToDataFrame:
    def test_columns_and_rows_match_expected_performance(self):
        perf = [
            _perf("RequestThroughput", "avg", 3.8, "requests/sec"),
            _perf("RequestLatency", "p50", 206.6, "ms"),
        ]
        view = _RecommendationView(_rec_row(perf=perf), index=0)
        df = view.to_dataframe()
        assert list(df.columns) == ["metric", "stat", "value", "unit"]
        assert len(df) == 2
        assert set(df["metric"]) == {"RequestThroughput", "RequestLatency"}
        assert df.iloc[0]["value"] == 3.8  # native float, not preformatted

    def test_empty_expected_performance_yields_empty_frame_with_schema(self):
        df = _RecommendationView(_rec_row(perf=[]), index=0).to_dataframe()
        assert len(df) == 0
        assert list(df.columns) == ["metric", "stat", "value", "unit"]


class TestRecommendationsViewToDataFrame:
    def _view(self):
        rows = [
            _rec_row(
                spec_name="A",
                instance_type="ml.g6.12xlarge",
                perf=[
                    _perf("RequestThroughput", "avg", 3.8, "requests/sec"),
                    _perf("RequestLatency", "p50", 206.6, "ms"),
                    _perf("RequestLatency", "p90", 257.9, "ms"),
                ],
            ),
            _rec_row(spec_name="B", instance_type="ml.g6.2xlarge", perf=[]),
        ]
        return _RecommendationsView(_RecommendationView(r, index=i) for i, r in enumerate(rows))

    def test_one_row_per_recommendation_indexed_by_idx(self):
        df = self._view().to_dataframe()
        assert df.index.name == "idx"
        assert list(df.index) == [0, 1]
        # idx is the index, not a duplicated column
        assert "idx" not in df.columns

    def test_columns_match_printed_table(self):
        df = self._view().to_dataframe()
        assert list(df.columns) == [
            "spec_name",
            "instance_type",
            "instances",
            "copies/inst",
            "container",
            "req/s",
            "tok/s",
            "lat_p50",
            "lat_p90",
            "lat_p99",
            "ttft_p50",
            "itl_p50",
        ]

    def test_numeric_columns_are_native_and_missing_is_nan(self):
        df = self._view().to_dataframe()
        assert df.loc[0, "req/s"] == 3.8
        assert df.loc[0, "lat_p50"] == 206.6
        assert df.loc[0, "instance_type"] == "ml.g6.12xlarge"
        assert df.loc[0, "container"] == "lmi25.0.0"
        # row B has no expected_performance -> numeric metric is NaN, not "-"
        assert pd.isna(df.loc[1, "req/s"])

    def test_empty_view_yields_empty_frame_with_schema(self):
        df = _RecommendationsView().to_dataframe()
        assert len(df) == 0
        assert "instance_type" in df.columns

    def test_missing_container_is_none_not_dash_sentinel(self):
        """A row without an image_uri gets None in the container column — not
        the "-" display sentinel — so .notna()/groupby/value_counts treat it as
        missing rather than a container literally named '-'."""
        rows = [
            _rec_row(spec_name="A", image_uri=None),
            _rec_row(spec_name="B", image_uri=".../djl-inference:0.36.0-lmi25.0.0-cu130"),
        ]
        view = _RecommendationsView(_RecommendationView(r, index=i) for i, r in enumerate(rows))
        df = view.to_dataframe()
        assert pd.isna(df.loc[0, "container"])  # missing -> NaN, not "-"
        assert df.loc[1, "container"] == "lmi25.0.0"
        # The printed table still shows a dash for the missing one.
        assert "-" in str(view)

    def test_metric_absent_from_all_rows_is_float_and_sortable(self):
        # No row has expected_performance -> every metric column is float64 NaN,
        # not object holding None, so nlargest/sort_values do not raise.
        rows = [_rec_row(spec_name="A", perf=[]), _rec_row(spec_name="B", perf=[])]
        view = _RecommendationsView(_RecommendationView(r, index=i) for i, r in enumerate(rows))
        df = view.to_dataframe()
        assert str(df["req/s"].dtype) == "float64"
        df.nlargest(1, "req/s")  # would raise on object dtype

    def test_str_and_dataframe_agree_on_row_order(self):
        """The comparative table and the frame list rows in the same order
        (by idx), driven from the shared _row_records()."""
        view = self._view()
        frame_order = list(view.to_dataframe().index)
        # The printed table shows each row's idx as "[N]" in the first column.
        table_order = [
            int(line.strip()[1:].split("]")[0])
            for line in str(view).splitlines()
            if line.strip().startswith("[")
        ]
        assert table_order == frame_order
