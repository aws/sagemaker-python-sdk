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
"""Unit tests for workflow cleanup_step."""

from __future__ import absolute_import

import json
from unittest.mock import Mock

import pytest

from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.mlops.workflow import CleanupStep, S3CleanupSource
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


class CustomStep(Step):
    def __init__(self, name, step_type=StepTypeEnum.TRAINING):
        super(CustomStep, self).__init__(name=name, step_type=step_type, depends_on=[])

    @property
    def arguments(self):
        return {}

    @property
    def properties(self):
        return None


@pytest.fixture
def mock_session():
    session = Mock()
    session.boto_region_name = "us-west-2"
    session.local_mode = False
    session.sagemaker_config = {}
    return session


def test_cleanup_step_resolved_source():
    training = CustomStep("TrainingJob1")
    step = CleanupStep(name="CleanupStep1", s3_sources=[S3CleanupSource(step=training)])

    assert step.step_type == StepTypeEnum.CLEANUP
    assert step.arguments == {"Resources": {"S3": [{"StepName": "TrainingJob1"}]}}
    assert step.depends_on == [training]


def test_cleanup_step_direct_uri_source_with_step_name():
    step = CleanupStep(
        name="CleanupStep1",
        s3_sources=[S3CleanupSource(step="ProcessingJob1", s3_uris=["s3://bucket/out/data.json"])],
    )

    assert step.arguments == {
        "Resources": {
            "S3": [{"StepName": "ProcessingJob1", "S3Uris": ["s3://bucket/out/data.json"]}]
        }
    }
    assert step.depends_on == ["ProcessingJob1"]


def test_cleanup_step_explicit_depends_on_adds_sources():
    step = CleanupStep(
        name="CleanupStep1",
        s3_sources=[S3CleanupSource(step="TrainingJob1"), S3CleanupSource(step="AthenaStep1")],
        depends_on=["OtherStep", "AthenaStep1"],
    )

    assert step.depends_on == ["OtherStep", "AthenaStep1", "TrainingJob1"]


def test_cleanup_step_requires_sources():
    with pytest.raises(ValueError, match="s3_sources"):
        CleanupStep(name="CleanupStep1", s3_sources=[])


def test_s3_cleanup_source_rejects_empty_uris():
    with pytest.raises(ValueError, match="s3_uris"):
        S3CleanupSource(step="ProcessingJob1", s3_uris=[])


def test_cleanup_step_properties_raises_error():
    step = CleanupStep(name="CleanupStep1", s3_sources=[S3CleanupSource(step="TrainingJob1")])
    with pytest.raises(RuntimeError, match="not available"):
        _ = step.properties


def test_cleanup_step_pipeline_definition(mock_session):
    output_path = ParameterString(name="ProcessingOutputPath", default_value="s3://bucket/out")
    training = CustomStep("TrainingJob1")
    processing = CustomStep("ProcessingJob1", step_type=StepTypeEnum.PROCESSING)
    cleanup = CleanupStep(
        name="CleanupStep1",
        s3_sources=[
            S3CleanupSource(step=training),
            S3CleanupSource(
                step=processing,
                s3_uris=[Join(on="", values=[output_path, "/processing-job-data.json"])],
            ),
        ],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[output_path],
        steps=[training, processing, cleanup],
        sagemaker_session=mock_session,
    )

    steps = json.loads(pipeline.definition())["Steps"]
    cleanup_definition = next(s for s in steps if s["Name"] == "CleanupStep1")

    assert cleanup_definition == {
        "Name": "CleanupStep1",
        "Type": "Cleanup",
        "DependsOn": ["TrainingJob1", "ProcessingJob1"],
        "Arguments": {
            "Resources": {
                "S3": [
                    {"StepName": "TrainingJob1"},
                    {
                        "StepName": "ProcessingJob1",
                        "S3Uris": [
                            {
                                "Std:Join": {
                                    "On": "",
                                    "Values": [
                                        {"Get": "Parameters.ProcessingOutputPath"},
                                        "/processing-job-data.json",
                                    ],
                                }
                            }
                        ],
                    },
                ]
            }
        },
    }
