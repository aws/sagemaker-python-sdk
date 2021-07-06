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
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.transformer import Transformer
from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.steps import (
    CreateModelStep,
    Step,
    TransformStep,
)
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
        """Get the request structure for workflow service calls."""
        return [step.to_request() for step in self.steps]


class RegisterModel(StepCollection):
    """Register Model step collection for workflow."""

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        model_data,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        depends_on: List[str] = None,
        model_package_group_name=None,
        model_metrics=None,
        approval_status=None,
        image_uri=None,
        compile_model_family=None,
        description=None,
        tags=None,
        **kwargs,
    ):
        """Construct steps `_RepackModelStep` and `_RegisterModelStep` based on the estimator.

        Args:
            name (str): The name of the training step.
            estimator: The estimator instance.
            model_data: The S3 uri to the model data from training.
            content_types (list): The supported MIME types for the input data (default: None).
            response_types (list): The supported MIME types for the output data (default: None).
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time (default: None).
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed (default: None).
            depends_on (List[str]): The list of step names the first step in the collection
                depends on
            model_package_group_name (str): The Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
                or "PendingManualApproval" (default: "PendingManualApproval").
            image_uri (str): The container image uri for Model Package, if not specified,
                Estimator's training container image is used (default: None).
            compile_model_family (str): The instance family for the compiled model. If
                specified, a compiled model is used (default: None).
            description (str): Model Package description (default: None).
            tags (List[dict[str, str]]): The list of tags to attach to the model package group. Note
                that tags will only be applied to newly created model package groups; if the
                name of an existing group is passed to "model_package_group_name",
                tags will not be applied.
            **kwargs: additional arguments to `create_model`.
        """
        steps: List[Step] = []
        repack_model = False
        if "entry_point" in kwargs:
            repack_model = True
            entry_point = kwargs.pop("entry_point", None)
            source_dir = kwargs.get("source_dir")
            dependencies = kwargs.get("dependencies")
            kwargs = dict(**kwargs, output_kms_key=kwargs.pop("model_kms_key", None))

            repack_model_step = _RepackModelStep(
                name=f"{name}RepackModel",
                depends_on=depends_on,
                estimator=estimator,
                model_data=model_data,
                entry_point=entry_point,
                source_dir=source_dir,
                dependencies=dependencies,
                **kwargs,
            )
            steps.append(repack_model_step)
            model_data = repack_model_step.properties.ModelArtifacts.S3ModelArtifacts

        # remove kwargs consumed by model repacking step
        kwargs.pop("entry_point", None)
        kwargs.pop("source_dir", None)
        kwargs.pop("dependencies", None)
        kwargs.pop("output_kms_key", None)

        register_model_step = _RegisterModelStep(
            name=name,
            estimator=estimator,
            model_data=model_data,
            content_types=content_types,
            response_types=response_types,
            inference_instances=inference_instances,
            transform_instances=transform_instances,
            model_package_group_name=model_package_group_name,
            model_metrics=model_metrics,
            approval_status=approval_status,
            image_uri=image_uri,
            compile_model_family=compile_model_family,
            description=description,
            tags=tags,
            **kwargs,
        )
        if not repack_model:
            register_model_step.add_depends_on(depends_on)

        steps.append(register_model_step)
        self.steps = steps


class EstimatorTransformer(StepCollection):
    """Creates a Transformer step collection for workflow."""

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        model_data,
        model_inputs,
        instance_count,
        instance_type,
        transform_inputs,
        # model arguments
        image_uri=None,
        predictor_cls=None,
        env=None,
        # transformer arguments
        strategy=None,
        assemble_with=None,
        output_path=None,
        output_kms_key=None,
        accept=None,
        max_concurrent_transforms=None,
        max_payload=None,
        tags=None,
        volume_kms_key=None,
        depends_on: List[str] = None,
        **kwargs,
    ):
        """Construct steps required for a Transformer step collection:

        An estimator-centric step collection. It models what happens in workflows
        when invoking the `transform()` method on an estimator instance:
        First, if custom
        model artifacts are required, a `_RepackModelStep` is included.
        Second, a
        `CreateModelStep` with the model data passed in from a training step or other
        training job output.
        Finally, a `TransformerStep`.

        If repacking
        the model artifacts is not necessary, only the CreateModelStep and TransformerStep
        are in the step collection.

        Args:
            name (str): The name of the Transform Step.
            estimator: The estimator instance.
            instance_count (int): The number of EC2 instances to use.
            instance_type (str): The type of EC2 instance to use.
            strategy (str): The strategy used to decide how to batch records in
                a single request (default: None). Valid values: 'MultiRecord'
                and 'SingleRecord'.
            assemble_with (str): How the output is assembled (default: None).
                Valid values: 'Line' or 'None'.
            output_path (str): The S3 location for saving the transform result. If
                not specified, results are stored to a default bucket.
            output_kms_key (str): Optional. A KMS key ID for encrypting the
                transform output (default: None).
            accept (str): The accept header passed by the client to
                the inference endpoint. If it is supported by the endpoint,
                it will be the format of the batch transform output.
            env (dict): The Environment variables to be set for use during the
                transform job (default: None).
            depends_on (List[str]): The list of step names the first step in
                the collection depends on
        """
        steps = []
        if "entry_point" in kwargs:
            entry_point = kwargs["entry_point"]
            source_dir = kwargs.get("source_dir")
            dependencies = kwargs.get("dependencies")
            repack_model_step = _RepackModelStep(
                name=f"{name}RepackModel",
                depends_on=depends_on,
                estimator=estimator,
                model_data=model_data,
                entry_point=entry_point,
                source_dir=source_dir,
                dependencies=dependencies,
            )
            steps.append(repack_model_step)
            model_data = repack_model_step.properties.ModelArtifacts.S3ModelArtifacts

        def predict_wrapper(endpoint, session):
            return Predictor(endpoint, session)

        predictor_cls = predictor_cls or predict_wrapper

        model = Model(
            image_uri=image_uri or estimator.training_image_uri(),
            model_data=model_data,
            predictor_cls=predictor_cls,
            vpc_config=None,
            sagemaker_session=estimator.sagemaker_session,
            role=estimator.role,
            **kwargs,
        )
        model_step = CreateModelStep(
            name=f"{name}CreateModelStep",
            model=model,
            inputs=model_inputs,
        )
        if "entry_point" not in kwargs and depends_on:
            # if the CreateModelStep is the first step in the collection
            model_step.add_depends_on(depends_on)
        steps.append(model_step)

        transformer = Transformer(
            model_name=model_step.properties.ModelName,
            instance_count=instance_count,
            instance_type=instance_type,
            strategy=strategy,
            assemble_with=assemble_with,
            output_path=output_path,
            output_kms_key=output_kms_key,
            accept=accept,
            max_concurrent_transforms=max_concurrent_transforms,
            max_payload=max_payload,
            env=env,
            tags=tags,
            base_transform_job_name=name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=estimator.sagemaker_session,
        )
        transform_step = TransformStep(
            name=f"{name}TransformStep",
            transformer=transformer,
            inputs=transform_inputs,
        )
        steps.append(transform_step)

        self.steps = steps
