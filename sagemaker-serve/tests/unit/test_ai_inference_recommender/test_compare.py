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
"""Unit tests for compare_benchmarks / BenchmarkComparison."""
from __future__ import absolute_import

import pytest

from sagemaker.serve.ai_inference_recommender import (
    BenchmarkResult,
    compare_benchmarks,
)
from sagemaker.serve.ai_inference_recommender.result import (
    BenchmarkMetrics,
    BenchmarkSearchResult,
)


def _result(throughput=None, latency=None, output_seq_len=None, s3="s3://b/out/"):
    """A single-run BenchmarkResult with request_throughput / request_latency,
    and optionally output_sequence_length (a directionless metric)."""
    profile = {}
    if throughput is not None:
        profile["request_throughput"] = {"unit": "req/s", "avg": throughput, "p50": throughput}
    if latency is not None:
        profile["request_latency"] = {"unit": "ms", "avg": latency, "p50": latency}
    if output_seq_len is not None:
        profile["output_sequence_length"] = {"unit": "tokens", "avg": output_seq_len}
    return BenchmarkResult(
        metrics=BenchmarkMetrics.from_profile_json(profile),
        s3_output_location=s3,
    )


class TestCompareBenchmarks:
    def test_requires_at_least_two(self):
        with pytest.raises(ValueError, match="at least two"):
            compare_benchmarks(_result(throughput=1.0))

    def test_names_length_must_match(self):
        with pytest.raises(ValueError, match="names has"):
            compare_benchmarks(_result(throughput=1.0), _result(throughput=2.0), names=["only-one"])

    def test_unknown_stat_rejected(self):
        with pytest.raises(ValueError, match="stat must be one of"):
            compare_benchmarks(_result(throughput=1.0), _result(throughput=2.0), stat="p42")

    def test_search_result_rejected(self):
        search = BenchmarkResult(
            metrics=BenchmarkMetrics.from_profile_json({}),
            s3_output_location="s3://b/out/",
            search=BenchmarkSearchResult(swept_dim="concurrency", winner=8),
        )
        with pytest.raises(ValueError, match="search/sweep"):
            compare_benchmarks(_result(throughput=1.0), search)

    def test_default_run_names(self):
        cmp = compare_benchmarks(_result(throughput=1.0), _result(throughput=2.0))
        assert cmp.names == ["run1", "run2"]

    def test_custom_names_used(self):
        cmp = compare_benchmarks(
            _result(throughput=1.0), _result(throughput=2.0), names=["before", "after"]
        )
        assert cmp.names == ["before", "after"]

    def test_throughput_increase_is_positive_delta(self):
        # request_throughput is higher-is-better: 10 -> 15 is +50%.
        cmp = compare_benchmarks(_result(throughput=10.0), _result(throughput=15.0))
        text = str(cmp)
        assert "request_throughput" in text
        assert "+50.0%" in text

    def test_latency_decrease_is_positive_delta(self):
        # request_latency is lower-is-better: 100 -> 80 is a 20% improvement,
        # reported as +20.0% (sign flipped so + is always better).
        cmp = compare_benchmarks(_result(latency=100.0), _result(latency=80.0))
        text = str(cmp)
        assert "request_latency" in text
        assert "+20.0%" in text

    def test_latency_increase_is_negative_delta(self):
        # 100 -> 120 latency is a regression: -20.0%.
        cmp = compare_benchmarks(_result(latency=100.0), _result(latency=120.0))
        assert "-20.0%" in str(cmp)

    def test_table_has_a_column_per_run_and_delta(self):
        cmp = compare_benchmarks(
            _result(throughput=10.0),
            _result(throughput=15.0),
            _result(throughput=20.0),
            names=["a", "b", "c"],
        )
        text = str(cmp)
        # value column per run + a delta column for each non-baseline run
        assert "a" in text and "b" in text and "c" in text
        assert "Δ% b" in text and "Δ% c" in text
        # baseline note names the first run
        assert "baseline: a" in text

    def test_stat_selects_percentile(self):
        cmp = compare_benchmarks(_result(throughput=10.0), _result(throughput=15.0), stat="p50")
        assert cmp.stat == "p50"
        assert "+50.0%" in str(cmp)

    def test_missing_metric_renders_dash_not_crash(self):
        # First run has throughput, second doesn't -> value '-' and delta '-'.
        cmp = compare_benchmarks(_result(throughput=10.0), _result(latency=5.0))
        text = str(cmp)
        assert "request_throughput" in text
        assert "request_latency" in text

    def test_directionless_metric_has_no_signed_delta(self):
        # output_sequence_length has no better/worse direction: halving it must
        # NOT read as a +50% improvement under the "+Δ = better" header.
        cmp = compare_benchmarks(_result(output_seq_len=128.0), _result(output_seq_len=64.0))
        text = str(cmp)
        assert "output_sequence_length" in text
        assert "+50.0%" not in text and "-50.0%" not in text

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="names must be unique"):
            compare_benchmarks(_result(throughput=1.0), _result(throughput=2.0), names=["a", "a"])

    def test_name_colliding_with_unit_rejected(self):
        with pytest.raises(ValueError, match="reserved column name 'unit'"):
            compare_benchmarks(
                _result(throughput=1.0), _result(throughput=2.0), names=["unit", "b"]
            )


pd = pytest.importorskip("pandas")


class TestBenchmarkComparisonToDataFrame:
    def test_columns_are_unit_runs_and_deltas(self):
        cmp = compare_benchmarks(
            _result(throughput=10.0),
            _result(throughput=15.0),
            _result(throughput=20.0),
            names=["a", "b", "c"],
        )
        df = cmp.to_dataframe()
        assert list(df.columns) == ["unit", "a", "b", "c", "Δ% b", "Δ% c"]
        assert df.index.name == "metric"
        assert "request_throughput" in df.index

    def test_values_and_units_are_native(self):
        cmp = compare_benchmarks(
            _result(throughput=10.0), _result(throughput=15.0), names=["base", "cand"]
        )
        row = cmp.to_dataframe().loc["request_throughput"]
        assert row["base"] == 10.0
        assert row["cand"] == 15.0
        assert row["unit"] == "req/s"

    def test_delta_is_numeric_and_signed_for_direction(self):
        # throughput up = better (+); latency up = worse (-). Deltas are numbers.
        up = compare_benchmarks(_result(throughput=10.0), _result(throughput=15.0))
        assert up.to_dataframe().loc["request_throughput", "Δ% run2"] == pytest.approx(50.0)

        worse = compare_benchmarks(_result(latency=100.0), _result(latency=120.0))
        assert worse.to_dataframe().loc["request_latency", "Δ% run2"] == pytest.approx(-20.0)

    def test_missing_value_yields_nan_delta(self):
        # Baseline has throughput, candidate does not -> value + delta are NaN.
        cmp = compare_benchmarks(_result(throughput=10.0), _result(latency=5.0))
        df = cmp.to_dataframe()
        assert pd.isna(df.loc["request_throughput", "run2"])
        assert pd.isna(df.loc["request_throughput", "Δ% run2"])

    def test_stat_selects_percentile(self):
        cmp = compare_benchmarks(_result(throughput=10.0), _result(throughput=15.0), stat="p50")
        # p50 was set equal to avg in the builder; delta still +50%.
        assert cmp.to_dataframe().loc["request_throughput", "Δ% run2"] == pytest.approx(50.0)

    def test_all_nan_delta_column_is_float_and_sortable(self):
        # A directionless metric present in both runs yields an all-NaN delta
        # column; it must be float64, not object holding None, so nlargest/
        # sort_values (the reason to want a frame) do not raise.
        a = _result(output_seq_len=128.0)
        b = _result(output_seq_len=64.0)
        df = compare_benchmarks(a, b).to_dataframe()
        assert df["Δ% run2"].isna().all()
        assert str(df["Δ% run2"].dtype) == "float64"
        df.nlargest(1, "Δ% run2")  # would raise on object dtype

    def test_directionless_metric_delta_is_nan_in_frame(self):
        cmp = compare_benchmarks(_result(output_seq_len=128.0), _result(output_seq_len=64.0))
        df = cmp.to_dataframe()
        # value columns are populated, but the oriented delta is NaN (no direction).
        assert df.loc["output_sequence_length", "run1"] == 128.0
        assert pd.isna(df.loc["output_sequence_length", "Δ% run2"])
