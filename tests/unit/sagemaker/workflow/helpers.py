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
"""Helper methods for testing."""
from __future__ import absolute_import

import re

from sagemaker.utils import base_from_name
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import ConfigurableRetryStep, StepTypeEnum
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.utilities import execute_job_functions


def ordered(obj):
    """Helper function for dict comparison.

    Recursively orders a json-like dict or list of dicts.

    Args:
        obj: either a list or a dict

    Returns:
        either a sorted list of elements or sorted list of tuples
    """
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def get_step_args_helper(step_args, step_type, use_custom_job_prefix=False):
    execute_job_functions(step_args)
    request_args = step_args.func_args[0].sagemaker_session.context.args

    if step_type == "Processing":
        if not use_custom_job_prefix:
            request_args.pop("ProcessingJobName", None)
        else:
            request_args["ProcessingJobName"] = strip_timestamp(request_args["ProcessingJobName"])
        request_args.pop("ExperimentConfig", None)
    elif step_type == "Training":
        if "HyperParameters" in request_args:
            request_args["HyperParameters"].pop("sagemaker_job_name", None)
        if not use_custom_job_prefix:
            request_args.pop("TrainingJobName", None)
        else:
            request_args["TrainingJobName"] = strip_timestamp(request_args["TrainingJobName"])
        request_args.pop("ExperimentConfig", None)
    elif step_type == "Transform":
        if not use_custom_job_prefix:
            request_args.pop("TransformJobName", None)
        else:
            request_args["TransformJobName"] = strip_timestamp(request_args["TransformJobName"])
        request_args.pop("ExperimentConfig", None)
    elif step_type == "HyperParameterTuning":
        if not use_custom_job_prefix:
            request_args.pop("HyperParameterTuningJobName", None)
        else:
            request_args["HyperParameterTuningJobName"] = strip_timestamp(
                request_args["HyperParameterTuningJobName"]
            )

    return request_args


def strip_timestamp(job_name):
    return base_from_name(job_name)


class CustomStep(ConfigurableRetryStep):
    def __init__(
        self,
        name,
        input_data=None,
        display_name=None,
        description=None,
        depends_on=None,
        retry_policies=None,
    ):
        self.input_data = input_data
        super(CustomStep, self).__init__(
            name, StepTypeEnum.TRAINING, display_name, description, depends_on, retry_policies
        )
        # for testing property reference, we just use DescribeTrainingJobResponse shape here.
        self._properties = Properties(name, shape_name="DescribeTrainingJobResponse")

    @property
    def arguments(self):
        if self.input_data:
            return {"input_data": self.input_data}
        return dict()

    @property
    def properties(self):
        return self._properties


class CustomStepCollection(StepCollection):
    def __init__(self, name, num_steps=2, depends_on=None):
        steps = []
        previous_step = None
        for i in range(num_steps):
            step_depends_on = depends_on if not previous_step else [previous_step]
            step = CustomStep(name=f"{name}-{i}", depends_on=step_depends_on)
            steps.append(step)
            previous_step = step
        super(CustomStepCollection, self).__init__(name, steps)
