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
from mock import Mock
from copy import deepcopy

from sagemaker.workflow.properties import Properties
from sagemaker.workflow.step_outputs import get_step


def test_get_step_from_properties():
    step = Mock()
    step.name = "MyStep"

    prop = Properties(step_name="MyStep", step=step, shape_name="DescribeProcessingJobResponse")
    assert prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri._referenced_steps == [
        step
    ]
    assert get_step(prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri) == step
    # step instance is dropped after deepcopy
    assert deepcopy(
        prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri
    )._referenced_steps == ["MyStep"]


def test_properties_describe_training_job_response():
    prop = Properties(step_name="MyStep", shape_name="DescribeTrainingJobResponse")
    some_prop_names = ["TrainingJobName", "TrainingJobArn", "HyperParameters", "OutputDataConfig"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()

    assert prop.CreationTime.expr == {"Get": "Steps.MyStep.CreationTime"}
    assert prop.CreationTime.expr == deepcopy(prop.CreationTime).expr
    assert prop.HyperParameters.expr == {"Get": "Steps.MyStep.HyperParameters"}
    assert prop.HyperParameters.expr == deepcopy(prop.HyperParameters).expr
    assert prop.OutputDataConfig.S3OutputPath.expr == {
        "Get": "Steps.MyStep.OutputDataConfig.S3OutputPath"
    }
    assert (
        prop.OutputDataConfig.S3OutputPath.expr == deepcopy(prop.OutputDataConfig.S3OutputPath).expr
    )


def test_properties_describe_processing_job_response():
    prop = Properties(step_name="MyStep", shape_name="DescribeProcessingJobResponse")
    some_prop_names = ["ProcessingInputs", "ProcessingOutputConfig", "ProcessingEndTime"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.ProcessingJobName.expr == {"Get": "Steps.MyStep.ProcessingJobName"}
    assert prop.ProcessingJobName.expr == deepcopy(prop.ProcessingJobName).expr
    assert prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri.expr == {
        "Get": "Steps.MyStep.ProcessingOutputConfig.Outputs['MyOutputName'].S3Output.S3Uri"
    }
    assert (
        prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri.expr
        == deepcopy(prop.ProcessingOutputConfig.Outputs["MyOutputName"].S3Output.S3Uri).expr
    )


def test_properties_tuning_job():
    prop = Properties(
        step_name="MyStep",
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
    assert (
        prop.TrainingJobSummaries[0].TrainingJobName.expr
        == deepcopy(prop.TrainingJobSummaries[0].TrainingJobName).expr
    )


def test_properties_emr_step():
    prop = Properties("MyStep", shape_name="Step", service_name="emr")
    some_prop_names = ["Id", "Name", "Config", "ActionOnFailure", "Status"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()

    assert prop.Id.expr == {"Get": "Steps.MyStep.Id"}
    assert prop.Name.expr == {"Get": "Steps.MyStep.Name"}
    assert prop.ActionOnFailure.expr == {"Get": "Steps.MyStep.ActionOnFailure"}
    assert prop.Config.Jar.expr == {"Get": "Steps.MyStep.Config.Jar"}
    assert prop.Status.State.expr == {"Get": "Steps.MyStep.Status.State"}


def test_properties_describe_model_package_output():
    prop = Properties(step_name="MyStep", shape_name="DescribeModelPackageOutput")
    some_prop_names = ["ModelPackageName", "ModelPackageGroupName", "ModelPackageArn"]
    for name in some_prop_names:
        assert name in prop.__dict__.keys()
    assert prop.ModelPackageName.expr == {"Get": "Steps.MyStep.ModelPackageName"}
    assert prop.ValidationSpecification.ValidationRole.expr == {
        "Get": "Steps.MyStep.ValidationSpecification.ValidationRole"
    }


def test_to_string():
    prop = Properties("MyStep", shape_name="DescribeTrainingJobResponse")

    assert prop.CreationTime.to_string().expr == {
        "Std:Join": {
            "On": "",
            "Values": [{"Get": "Steps.MyStep.CreationTime"}],
        },
    }


def test_implicit_value():
    prop = Properties("MyStep", shape_name="DescribeTrainingJobResponse")

    with pytest.raises(TypeError) as error:
        str(prop.CreationTime)
    assert "Pipeline variables do not support __str__ operation." in str(error.value)

    with pytest.raises(TypeError) as error:
        int(prop.CreationTime)
    assert str(error.value) == "Pipeline variables do not support __int__ operation."

    with pytest.raises(TypeError) as error:
        float(prop.CreationTime)
    assert str(error.value) == "Pipeline variables do not support __float__ operation."


def test_add_func():
    prop_train = Properties("MyStepTrain", shape_name="DescribeTrainingJobResponse")
    prop_model = Properties("MyStepModel", shape_name="DescribeModelPackageOutput")

    with pytest.raises(TypeError) as error:
        prop_train + prop_model

    assert str(error.value) == "Pipeline variables do not support concatenation."
