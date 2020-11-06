# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""The step definitions for workflow."""
from __future__ import absolute_import

from typing import List

import attr

from sagemaker.estimator import EstimatorBase
from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.steps import Step
from sagemaker.workflow._utils import (
    _RegisterModelStep,
    _RepackModelStep,
)


@attr.s
class StepCollection:
    """A wrapper of pipeline steps for workflow.

    Attributes:
        steps (List[Step]): A list of steps.
    """

    steps: List[Step] = attr.ib(factory=list)

    def request_dicts(self) -> List[RequestType]:
        """Gets the request structure for workflow service calls."""
        return [step.to_request() for step in self.steps]


class RegisterModel(StepCollection):
    """Register Model step collection for workflow.

    Attributes:
        steps (List[Step]): A list of steps.
    """

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        model_data,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_group_name=None,
        image_uri=None,
        compile_model_family=None,
        **kwargs,
    ):
        """Constructs steps `_RepackModelStep` and `_RegisterModelStep` based on the estimator.

        Args:
            name (str): The name of the training step.
            estimator: The estimator instance.
            model_data: the S3 URI to the model data from training.
            content_types (list): The supported MIME types for the input data (default: None).
            response_types (list): The supported MIME types for the output data (default: None).
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time (default: None).
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed (default: None).
            model_package_group_name (str): Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned (default: None).
            image_uri (str): The container image uri for Model Package, if not specified,
                Estimator's training container image will be used (default: None).
            compile_model_family (str): Instance family for compiled model, if specified, a compiled
                model will be used (default: None).
            **kwargs: additional arguments to `create_model`.
        """
        steps: List[Step] = []
        if "entry_point" in kwargs:
            entry_point = kwargs["entry_point"]
            source_dir = kwargs.get("source_dir")
            dependencies = kwargs.get("dependencies")
            repack_model_step = _RepackModelStep(
                name=f"{name}RepackModel",
                estimator=estimator,
                model_data=model_data,
                entry_point=entry_point,
                source_dir=source_dir,
                dependencies=dependencies,
            )
            steps.append(repack_model_step)
            model_data = repack_model_step.properties.ModelArtifacts.S3ModelArtifacts

        register_model_step = _RegisterModelStep(
            name=name,
            estimator=estimator,
            model_data=model_data,
            content_types=content_types,
            response_types=response_types,
            inference_instances=inference_instances,
            transform_instances=transform_instances,
            model_package_group_name=model_package_group_name,
            image_uri=image_uri,
            compile_model_family=compile_model_family,
            **kwargs,
        )
        steps.append(register_model_step)
        self.steps = steps
