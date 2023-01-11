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
from __future__ import absolute_import

import pytest
import os

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from tests.integ import DATA_DIR

DATA_DIR = os.path.join(DATA_DIR, "automl", "data")

def test_hashing_behavior(
    pipeline_session,
    role,
    script_dir,
    athena_dataset_definition,
    region_name,
):
    default_bucket = pipeline_session.default_bucket()
    data_path = os.path.join(DATA_DIR, "workflow")
    code_dir = os.path.join(script_dir, "train.py")

    framework_version = "0.20.0"
    instance_type = "ml.m5.xlarge"
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    output_prefix = ParameterString(name="OutputPrefix", default_value="output")

    input_data = f"s3://sagemaker-sample-data-{region_name}/processing/census/census-income.csv"

    # additionally add abalone input, so we can test input s3 file from local upload
    abalone_input = ProcessingInput(
        input_name="abalone_data",
        source=os.path.join(data_path, "abalone-dataset.csv"),
        destination="/opt/ml/processing/input",
    )

    # define processing step
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name="test-sklearn",
        sagemaker_session=pipeline_session,
        role=role,
    )
    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
            ProcessingInput(dataset_definition=athena_dataset_definition),
            abalone_input,
        ],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
            ProcessingOutput(
                output_name="test_data",
                source="/opt/ml/processing/test",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        pipeline_session.default_bucket(),
                        "test-sklearn",
                        output_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            ),
        ],
        code=os.path.join(script_dir, "preprocessing.py"),
    )
    step_process = ProcessingStep(
        name="my-process",
        display_name="ProcessingStep",
        description="description for Processing step",
        step_args=processor_args,
    )