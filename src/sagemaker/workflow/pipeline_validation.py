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
"""The `PipelineValidation` definitions for SageMaker Pipelines Workflows."""
from __future__ import absolute_import

import botocore.loaders
from botocore.exceptions import ValidationError

from sagemaker.workflow.steps import StepTypeEnum
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.step_collections import StepCollection


class PipelineValidation:
    """Validator for SageMaker Pipeline Workflows"""

    def __init__(self, pipeline):
        """Constructs a PipelineValidation obj from Pipeline attributes

        Args:
            pipeline (Pipeline):
                The current pipeline workflow object used for validation
        """
        # Load Botocore for model validation
        loader = botocore.loaders.Loader()
        sagemaker_model = loader.load_service_model("sagemaker", "service-2")
        self._shapes_map = sagemaker_model["shapes"]
        self._pipeline = pipeline

    def validate(self):
        """Validate pipeline obj as a series of method calls"""
        self._validate_processing_inputs()

    def _validate_processing_inputs(self):
        """Validate pipeline definition processingStep/input satisfies max length"""
        max_processing_inputs = self._shapes_map["ProcessingInputs"]["max"]
        for step in self._pipeline.steps:
            if isinstance(step, ProcessingStep) and step is not None:
                if (
                    step.step_type is StepTypeEnum.PROCESSING and step.inputs
                    and len(step.inputs) > max_processing_inputs
                ):
                    raise ValidationError(
                        value=len(step.inputs), param=step.inputs, type_name=type(step.inputs)
                    )
            if isinstance(step, StepCollection) and step is not None:
                for s in step.steps:
                    if isinstance(s, ProcessingStep) and s is not None:
                        if (
                            s.step_type is StepTypeEnum.PROCESSING and s.inputs
                            and len(s.inputs) > max_processing_inputs
                        ):
                            raise ValidationError(
                                value=len(s.inputs),
                                param=s.inputs,
                                type_name=type(s.inputs),
                            )
