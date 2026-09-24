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
"""Unit tests for workflow quality_check_step."""

from __future__ import absolute_import

from unittest.mock import Mock

from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.mlops.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)


def test_data_quality_check_config_init():
    config = DataQualityCheckConfig(
        baseline_dataset="s3://bucket/data.csv", dataset_format={"csv": {"header": True}}
    )
    assert config.baseline_dataset == "s3://bucket/data.csv"
    assert config.dataset_format == {"csv": {"header": True}}


def test_model_quality_check_config_init():
    config = ModelQualityCheckConfig(
        baseline_dataset="s3://bucket/data.csv",
        dataset_format={"csv": {"header": True}},
        problem_type="BinaryClassification",
    )
    assert config.problem_type == "BinaryClassification"


def test_generate_baseline_job_inputs_with_pipeline_variable_baseline_dataset():
    """Regression for #6206.

    When baseline_dataset is a pipeline variable, the ProcessingInput's s3_input must still
    include the required s3_data_type; omitting it raised a pydantic ValidationError.
    """
    baseline_dataset = Join(
        on="/",
        values=["s3:/", "my-bucket", ParameterString(name="EndpointName"), "baseline/data.parquet"],
    )
    config = DataQualityCheckConfig(
        baseline_dataset=baseline_dataset,
        dataset_format={"parquet": {}},
        output_s3_uri="s3://my-bucket/output/",
    )

    step = object.__new__(QualityCheckStep)
    step.quality_check_config = config
    # Isolate the pipeline-variable baseline branch; the script inputs go through the monitor.
    step._model_monitor = Mock()
    step._model_monitor._upload_and_convert_to_processing_input.return_value = Mock()

    inputs = step._generate_baseline_job_inputs()

    baseline_input = inputs["baseline_dataset_input"]
    s3_input = baseline_input.s3_input
    assert s3_input.s3_data_type == "S3Prefix"
    assert s3_input.s3_uri == baseline_dataset
    # These must be concrete values, not the Unassigned() sentinel: the arguments serializer
    # emits them into the pipeline definition, and json.dumps chokes on Unassigned (#6206).
    import json

    from sagemaker.core.utils.utils import Unassigned

    assert not isinstance(s3_input.s3_input_mode, Unassigned)
    assert not isinstance(s3_input.s3_data_distribution_type, Unassigned)
    serialized = {
        "S3Uri": "s3://resolved/at/runtime",  # a pipeline var resolves to an expr; stub for json
        "LocalPath": s3_input.local_path,
        "S3DataType": getattr(s3_input, "s3_data_type", "S3Prefix"),
        "S3InputMode": getattr(s3_input, "s3_input_mode", "File"),
    }
    json.dumps(serialized)  # must not raise TypeError on an Unassigned sentinel
