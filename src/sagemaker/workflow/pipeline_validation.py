
from __future__ import absolute_import

from typing import Dict, Union, List

import attr

import botocore.loaders
from botocore.exceptions import ClientError, ValidationError
from typing import Any, Dict, List, Sequence, Union, Optional

# from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import StepTypeEnum
from sagemaker.workflow.steps import Step
from sagemaker.workflow.step_collections import StepCollection

class PipelineValidation:
    """

    Extends pipeline for client-side validation

    Attributes:
        _shapes_map dict(): dictionary of botocore model restrictions to check at compile time

    """

    _shapes_map = dict()

    def __init__(self,
                 steps: Sequence[Union[Step, StepCollection]] = attr.ib(factory=list)
                 ):
        """ Loads up the shapes from the botocore service model. """
        loader = botocore.loaders.Loader()
        sagemaker_model = loader.load_service_model("sagemaker", "service-2")
        self._shapes_map = sagemaker_model["shapes"]
        self._steps = steps

    def validate_pipeline_inputs(self):
        """ Validate pipeline definition as a series of method checks """
        try:
            self._validate_processing_steps()
        except ClientError as e:
            raise

    def _validate_processing_steps(self):
        """ Validate that ProcessingInputs size is less than 10 (as defined in botocore) """
        max_processing_inputs = self._shapes_map["ProcessingInputs"]["max"]
        for step in self._steps:
            if step.step_type is StepTypeEnum.PROCESSING and len(step.inputs) > max_processing_inputs:
                raise ValidationError(value=len(step.inputs), param=step.inputs, type_name=StepTypeEnum.PROCESSING)





