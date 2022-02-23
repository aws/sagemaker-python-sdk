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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json

import pytest
import warnings

from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.workflow.steps import TransformStep, TransformInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString

from sagemaker.transformer import Transformer

REGION = "us-west-2"
IMAGE_URI = "fakeimage"
MODEL_NAME = "gisele"
DUMMY_S3_SCRIPT_PATH = "s3://dummy-s3/dummy_script.py"
DUMMY_S3_SOURCE_DIR = "s3://dummy-s3-source-dir/"
INSTANCE_TYPE = "ml.m4.xlarge"


@pytest.fixture
def pipeline_session():
    return PipelineSession()


def test_transform_step_with_transformer(pipeline_session):
    model_name = ParameterString("ModelName")
    transformer = Transformer(
        model_name=model_name,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=f"s3://{pipeline_session.default_bucket()}/Transform",
        sagemaker_session=pipeline_session,
    )

    transform_inputs = TransformInput(
        data=f"s3://{pipeline_session.default_bucket()}/batch-data",
    )

    with warnings.catch_warnings(record=True) as w:
        step_args = transformer.transform(
            data=transform_inputs.data,
            data_type=transform_inputs.data_type,
            content_type=transform_inputs.content_type,
            compression_type=transform_inputs.compression_type,
            split_type=transform_inputs.split_type,
            input_filter=transform_inputs.input_filter,
            output_filter=transform_inputs.output_filter,
            join_source=transform_inputs.join_source,
            model_client_config=transform_inputs.model_client_config,
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Running within a PipelineSession" in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        step = TransformStep(
            name="MyTransformStep",
            step_args=step_args,
        )
        assert len(w) == 0

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step],
        parameters=[model_name],
        sagemaker_session=pipeline_session,
    )
    step_args["ModelName"] = model_name.expr
    assert json.loads(pipeline.definition())["Steps"][0] == {
        "Name": "MyTransformStep",
        "Type": "Transform",
        "Arguments": step_args,
    }
