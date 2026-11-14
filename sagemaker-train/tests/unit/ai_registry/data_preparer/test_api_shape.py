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
"""Unit tests for the DQ request-shape strategy.

Version dispatch happens once via ``DqShape.for_version(...)``; every downstream
caller consumes a single strategy object and never re-checks the schema version.
Keeps the two field differences that matter to the SDK today — Output S3 URI
field name and Dataset ARN location — in one place.
"""

from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer._api_shape import (
    DqShape,
    V1DqShape,
    V2DqShape,
    V3DqShape,
)


class TestForVersionDispatch:
    """for_version resolves the string once at the call boundary."""

    def test_v0_2_returns_v2_shape(self):
        assert isinstance(DqShape.for_version("0.2.0"), V2DqShape)

    def test_v2_0_0_returns_v2_shape(self):
        # The service plans to rename the 0.2.0 preview schema to 2.0.0
        # "without other changes" when v2 becomes the default. Both strings
        # must select the v2 parser or the rename silently re-routes v2 jobs
        # to the v1 parser (rehydration then fails on the v2 document shape).
        assert isinstance(DqShape.for_version("2.0.0"), V2DqShape)

    def test_v3_0_0_returns_v3_shape(self):
        assert isinstance(DqShape.for_version("3.0.0"), V3DqShape)

    def test_v1_returns_v1_shape(self):
        assert isinstance(DqShape.for_version("1.0.0"), V1DqShape)

    def test_none_returns_v1_shape(self):
        # Legacy jobs / mainline callers never set a version. Fall back to v1.
        assert isinstance(DqShape.for_version(None), V1DqShape)

    def test_empty_string_returns_v1_shape(self):
        assert isinstance(DqShape.for_version(""), V1DqShape)

    def test_unknown_version_returns_v1_shape(self):
        # Unknown/future versions default to v1 rather than crashing.
        assert isinstance(DqShape.for_version("99.99.99"), V1DqShape)


class TestExtractOutputS3UriV1:
    """v1: read OutputDataConfig.OutputS3Uri; anything else is invisible."""

    def test_reads_output_s3_uri_field(self):
        shape = DqShape.for_version("1.0.0")
        doc = {"OutputDataConfig": {"OutputS3Uri": "s3://bucket/out/"}}
        assert shape.extract_output_s3_uri(doc) == "s3://bucket/out/"

    def test_ignores_v2_s3_output_path_field(self):
        # A v1 caller who accidentally sends the v2 field should get None here —
        # not silently fall through to v2's field name.
        shape = DqShape.for_version("1.0.0")
        doc = {"OutputDataConfig": {"S3OutputPath": "s3://bucket/out/"}}
        assert shape.extract_output_s3_uri(doc) is None

    def test_returns_none_when_output_data_config_missing(self):
        shape = DqShape.for_version("1.0.0")
        assert shape.extract_output_s3_uri({}) is None

    def test_returns_none_when_output_data_config_wrong_type(self):
        shape = DqShape.for_version("1.0.0")
        assert shape.extract_output_s3_uri({"OutputDataConfig": "not-a-dict"}) is None

    def test_returns_none_when_uri_wrong_type(self):
        shape = DqShape.for_version("1.0.0")
        doc = {"OutputDataConfig": {"OutputS3Uri": 12345}}
        assert shape.extract_output_s3_uri(doc) is None


class TestExtractOutputS3UriV2:
    """v2 (wire alias 0.2.0): read OutputDataConfig.S3OutputPath."""

    def test_reads_s3_output_path_field(self):
        shape = DqShape.for_version("0.2.0")
        doc = {"OutputDataConfig": {"S3OutputPath": "s3://bucket/out/"}}
        assert shape.extract_output_s3_uri(doc) == "s3://bucket/out/"

    def test_ignores_v1_output_s3_uri_field(self):
        # v2 requests must not silently fall back to the legacy field name —
        # a mixed-shape request would be a validation error, and doing anything
        # softer here would mask that.
        shape = DqShape.for_version("0.2.0")
        doc = {"OutputDataConfig": {"OutputS3Uri": "s3://bucket/out/"}}
        assert shape.extract_output_s3_uri(doc) is None

    def test_returns_none_when_output_data_config_missing(self):
        shape = DqShape.for_version("0.2.0")
        assert shape.extract_output_s3_uri({}) is None


class TestExtractDatasetArnV1:
    """v1: scalar top-level DatasetArn."""

    def test_reads_scalar_dataset_arn(self):
        arn = "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/D/1.0.0"
        shape = DqShape.for_version("1.0.0")
        assert shape.extract_dataset_arn({"DatasetArn": arn}) == arn

    def test_ignores_v2_input_data_config(self):
        arn = "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/D/1.0.0"
        shape = DqShape.for_version("1.0.0")
        doc = {"InputDataConfig": [{"DatasetArn": arn}]}
        assert shape.extract_dataset_arn(doc) is None

    def test_returns_none_when_missing(self):
        shape = DqShape.for_version("1.0.0")
        assert shape.extract_dataset_arn({}) is None


class TestExtractDatasetArnV2:
    """v2 (wire alias 0.2.0): InputDataConfig[0].DatasetArn."""

    def test_reads_from_input_data_config(self):
        arn = "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/D/1.0.0"
        shape = DqShape.for_version("0.2.0")
        doc = {"InputDataConfig": [{"DatasetArn": arn}]}
        assert shape.extract_dataset_arn(doc) == arn

    def test_ignores_scalar_dataset_arn(self):
        # A v2 request that accidentally includes the legacy scalar field must
        # not resolve through it — mixed-shape requests are validation errors.
        arn = "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/D/1.0.0"
        shape = DqShape.for_version("0.2.0")
        assert shape.extract_dataset_arn({"DatasetArn": arn}) is None

    def test_returns_none_when_input_data_config_missing(self):
        shape = DqShape.for_version("0.2.0")
        assert shape.extract_dataset_arn({}) is None

    def test_returns_none_when_input_data_config_empty_list(self):
        shape = DqShape.for_version("0.2.0")
        assert shape.extract_dataset_arn({"InputDataConfig": []}) is None

    def test_returns_none_when_first_entry_missing_dataset_arn(self):
        shape = DqShape.for_version("0.2.0")
        doc = {"InputDataConfig": [{"WrongField": "value"}]}
        assert shape.extract_dataset_arn(doc) is None

    def test_returns_none_when_input_data_config_wrong_type(self):
        # Malformed / stringified — extractor should not crash.
        shape = DqShape.for_version("0.2.0")
        assert shape.extract_dataset_arn({"InputDataConfig": "not-a-list"}) is None


class TestExtractV3:
    """v3 (3.0.0): Operations replaces v2's groups; the input and output fields are v2's."""

    def test_reads_input_data_config_and_s3_output_path(self):
        arn = "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/D/1.0.0"
        shape = DqShape.for_version("3.0.0")
        doc = {
            "InputDataConfig": [{"DatasetArn": arn}],
            "OutputDataConfig": {"S3OutputPath": "s3://bucket/out/"},
            "EvaluationConfig": {"Operations": [{"Operator": "default_metrics"}]},
        }
        assert shape.extract_dataset_arn(doc) == arn
        assert shape.extract_output_s3_uri(doc) == "s3://bucket/out/"


class TestForVersionDispatchA7:
    """A7 (P1): for_version must normalize partial semver strings and dispatch on major.

    The service has returned both two-part ("2.0") and three-part ("2.0.0") version
    strings for the same schema. An exact-string membership check (old code) silently
    routed "2.0" to the v1 parser, causing rehydration to fail with
    "Completed job missing DatasetArn" on any job whose JobConfigSchemaVersion is "2.0".

    All cases that FAIL on the original code (exact-string match) are marked [A7-fail].
    """

    @pytest.mark.parametrize(
        "version, expected_shape_cls",
        [
            # --- Two-part versions: the service returns these, old code mis-routes them ---
            ("2.0", V2DqShape),  # [A7-fail] main regression case from live jobs
            ("3.0", V3DqShape),  # [A7-fail] analogous v3 regression
            # --- Single-component versions ---
            ("2", V2DqShape),  # [A7-fail]
            ("3", V3DqShape),  # [A7-fail]
            # --- Three-part versions (already worked, must still work) ---
            ("2.0.0", V2DqShape),
            ("3.0.0", V3DqShape),
            ("1.0.0", V1DqShape),
            # --- Legacy preview alias (must still work) ---
            ("0.2.0", V2DqShape),
            # --- v1 / legacy / fallback territory ---
            ("1", V1DqShape),
            ("1.5.0", V1DqShape),
            # --- Sentinel / unknown / garbage → v1 ---
            (None, V1DqShape),
            ("", V1DqShape),
            ("abc", V1DqShape),
        ],
    )
    def test_version_dispatch(self, version, expected_shape_cls):
        shape = DqShape.for_version(version)
        assert isinstance(shape, expected_shape_cls), (
            f"for_version({version!r}) returned {type(shape).__name__}, "
            f"expected {expected_shape_cls.__name__}"
        )
