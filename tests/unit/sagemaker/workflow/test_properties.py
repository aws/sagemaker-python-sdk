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

from sagemaker.workflow.properties import Properties


def test_properties_describe_training_job_response():
    prop = Properties("Steps.MyStep", "DescribeTrainingJobResponse")
    some_prop_names = ["TrainingJobName", "TrainingJobArn", "HyperParameters", "OutputDataConfig"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.CreationTime.expr == {"Get": "Steps.MyStep.CreationTime"}
    assert prop.HyperParameters.expr == {"Get": "Steps.MyStep.HyperParameters"}
    assert prop.OutputDataConfig.S3OutputPath.expr == {
        "Get": "Steps.MyStep.OutputDataConfig.S3OutputPath"
    }


def test_properties_describe_processing_job_response():
    prop = Properties("Steps.MyStep", "DescribeProcessingJobResponse")
    some_prop_names = ["ProcessingInputs", "ProcessingOutputConfig", "ProcessingEndTime"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.ProcessingJobName.expr == {"Get": "Steps.MyStep.ProcessingJobName"}
    assert prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri.expr == {
        "Get": "Steps.MyStep.ProcessingOutputConfig.Outputs['MyOutputName'].S3Output.S3Uri"
    }


def test_properties_tuning_job():
    prop = Properties(
        "Steps.MyStep",
        shape_names=[
            "DescribeHyperParameterTuningJobResponse",
            "ListTrainingJobsForHyperParameterTuningJobResponse",
        ],
    )
    some_prop_names = [
        "BestTrainingJob",
        "HyperParameterTuningJobConfig",
        "ObjectiveStatusCounters",
        "TrainingJobSummaries",
    ]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()

    assert prop.HyperParameterTuningJobName.expr == {
        "Get": "Steps.MyStep.HyperParameterTuningJobName"
    }
    assert prop.HyperParameterTuningJobConfig.HyperParameterTuningJobObjective.Type.expr == {
        "Get": "Steps.MyStep.HyperParameterTuningJobConfig.HyperParameterTuningJobObjective.Type"
    }
    assert prop.ObjectiveStatusCounters.Succeeded.expr == {
        "Get": "Steps.MyStep.ObjectiveStatusCounters.Succeeded"
    }
    assert prop.TrainingJobSummaries[0].TrainingJobName.expr == {
        "Get": "Steps.MyStep.TrainingJobSummaries[0].TrainingJobName"
    }


def test_properties_describe_model_package_output():
    prop = Properties("Steps.MyStep", "DescribeModelPackageOutput")
    some_prop_names = ["ModelPackageName", "ModelPackageGroupName", "ModelPackageArn"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.ModelPackageName.expr == {"Get": "Steps.MyStep.ModelPackageName"}
    assert prop.ValidationSpecification.ValidationRole.expr == {
        "Get": "Steps.MyStep.ValidationSpecification.ValidationRole"
    }
