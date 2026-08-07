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
"""Step definitions for Amazon Bedrock deployment steps in Pipelines.

Design note: mirrors Tioga's Bedrock step definitions in
``IronmanTiogaPipelineDefinitionRepository``. Each step's ``Arguments``
block is an opaque ``StructureArgument`` validated against the AWS
Bedrock SDK request class (``CreateCustomModelRequest``,
``CreateCustomModelDeploymentRequest``, ``CreateModelImportJobRequest``,
``CreateProvisionedModelThroughputRequest``) with no field exclusions.
Any field the AWS Bedrock API accepts, Tioga accepts.

Two fields carry server-side validation that this SDK does not duplicate:
``BedrockCustomModelStep.arguments["ModelName"]`` and
``BedrockCustomModelDeploymentStep.arguments["ModelDeploymentName"]``
must be pipeline parameter references (not hardcoded strings). Tioga
rejects hardcoded values with a clear error at pipeline creation time.

Property references use PascalCase field names (e.g.
``step.properties.ModelArn``) because Tioga's property-path resolver
uses PascalCase, whereas the Bedrock JSON API uses camelCase member
names. Field lists are hand-populated below from each ``Get*Response``.
"""

from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Union

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.properties import Properties

from sagemaker.mlops.workflow._argument_validation import validate_step_arguments
from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


def _validate_bedrock_arguments(
    step_class_name: str, arguments: Dict[str, Any], operation_name: str
) -> None:
    """Validate a Bedrock step's arguments against the botocore input shape.

    Bedrock's JSON API members are camelCase, but pipeline ``Arguments``
    fields are PascalCase (matching Tioga's property-path resolver), so
    shape member names are PascalCase-converted before comparison.
    """
    validate_step_arguments(
        step_class_name,
        arguments,
        service_name="bedrock",
        operation_name=operation_name,
        pascal_case=True,
    )


# PascalCase property paths for each Bedrock step, sourced from each
# ``Get*Response`` shape. Users reference these via
# ``step.properties.<field>``.
_BEDROCK_CUSTOM_MODEL_FIELDS = [
    "ModelArn",
    "ModelName",
    "JobArn",
    "JobName",
    "BaseModelArn",
    "CustomizationType",
    "ModelKmsKeyArn",
    "HyperParameters",
    "TrainingDataConfig",
    "ValidationDataConfig",
    "OutputDataConfig",
    "TrainingMetrics",
    "ValidationMetrics",
    "CreationTime",
    "CustomizationConfig",
    "ModelStatus",
    "FailureMessage",
]

_BEDROCK_CUSTOM_MODEL_DEPLOYMENT_FIELDS = [
    "ModelDeploymentArn",
    "ModelDeploymentName",
    "ModelArn",
    "CreatedAt",
    "Status",
    "FailureMessage",
    "Description",
    "Tags",
]

_BEDROCK_MODEL_IMPORT_FIELDS = [
    "JobArn",
    "JobName",
    "ImportedModelName",
    "ImportedModelArn",
    "RoleArn",
    "ModelDataSource",
    "Status",
    "FailureMessage",
    "CreationTime",
    "LastModifiedTime",
    "EndTime",
    "VpcConfig",
    "ImportedModelKmsKeyArn",
]

_BEDROCK_PROVISIONED_MODEL_THROUGHPUT_FIELDS = [
    "ModelUnits",
    "DesiredModelUnits",
    "ProvisionedModelName",
    "ProvisionedModelArn",
    "ModelArn",
    "DesiredModelArn",
    "FoundationModelArn",
    "Status",
    "CreationTime",
    "LastModifiedTime",
    "FailureMessage",
    "CommitmentDuration",
    "CommitmentExpirationTime",
]


def _bedrock_properties(step_name: str, step, fields: List[str]) -> Properties:
    """Build a bare ``Properties`` root with the given top-level fields."""
    root = Properties(step_name=step_name, step=step)
    for field in fields:
        root.__dict__[field] = Properties(step_name=step_name, path=field)
    return root


class BedrockCustomModelStep(Step):
    """Creates a custom model in Amazon Bedrock.

    Wraps Bedrock's ``CreateCustomModel`` API. The ``arguments`` dict is
    passed through to the service; it accepts any field of
    ``CreateCustomModelRequest`` (no exclusions). Typical fields:
    ``ModelName`` (required, must be pipeline parameter reference),
    ``RoleArn`` (required), ``ModelSourceConfig`` (required),
    ``ClientRequestToken``, ``ModelKmsKeyArn``.

    Server-side validation enforces ``ModelName`` as a pipeline parameter
    reference — hardcoded strings are rejected at pipeline creation time.
    """

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct a ``BedrockCustomModelStep``.

        Args:
            name (str): The name of the step.
            arguments (Dict[str, Any]): The ``Arguments`` block for
                ``CreateCustomModel``. ``ModelName`` must be a pipeline
                parameter reference. ``ClientRequestToken`` is optional
                — the pipeline service auto-generates one if omitted.
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            description=description,
            step_type=StepTypeEnum.BEDROCK_CUSTOM_MODEL,
            depends_on=depends_on,
        )
        if arguments is None:
            raise ValueError("arguments is required for BedrockCustomModelStep.")
        _validate_bedrock_arguments("BedrockCustomModelStep", arguments, "CreateCustomModel")
        self._arguments = arguments
        self._properties = _bedrock_properties(name, self, _BEDROCK_CUSTOM_MODEL_FIELDS)

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateCustomModel`` call."""
        _validate_bedrock_arguments("BedrockCustomModelStep", self._arguments, "CreateCustomModel")
        return self._arguments

    @property
    def properties(self):
        """PascalCase fields from ``GetCustomModelResponse``."""
        return self._properties


class BedrockCustomModelDeploymentStep(Step):
    """Deploys a Bedrock custom model for inference.

    Wraps Bedrock's ``CreateCustomModelDeployment`` API. The ``arguments``
    dict is passed through; it accepts any field of
    ``CreateCustomModelDeploymentRequest`` (no exclusions). Typical
    fields: ``ModelDeploymentName`` (required, must be pipeline parameter
    reference), ``ModelArn`` (required), ``Description``,
    ``ClientRequestToken``, ``Tags``.

    Server-side validation enforces ``ModelDeploymentName`` as a pipeline
    parameter reference.
    """

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct a ``BedrockCustomModelDeploymentStep``.

        Args:
            name (str): The name of the step.
            arguments (Dict[str, Any]): The ``Arguments`` block for
                ``CreateCustomModelDeployment``. ``ModelDeploymentName``
                must be a pipeline parameter reference.
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            description=description,
            step_type=StepTypeEnum.BEDROCK_CUSTOM_MODEL_DEPLOYMENT,
            depends_on=depends_on,
        )
        if arguments is None:
            raise ValueError("arguments is required for BedrockCustomModelDeploymentStep.")
        _validate_bedrock_arguments(
            "BedrockCustomModelDeploymentStep", arguments, "CreateCustomModelDeployment"
        )
        self._arguments = arguments
        self._properties = _bedrock_properties(name, self, _BEDROCK_CUSTOM_MODEL_DEPLOYMENT_FIELDS)

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateCustomModelDeployment`` call."""
        _validate_bedrock_arguments(
            "BedrockCustomModelDeploymentStep", self._arguments, "CreateCustomModelDeployment"
        )
        return self._arguments

    @property
    def properties(self):
        """PascalCase fields from ``GetCustomModelDeploymentResponse``."""
        return self._properties


class BedrockModelImportStep(Step):
    """Imports a SageMaker-trained model into Bedrock.

    Wraps Bedrock's ``CreateModelImportJob`` API. The ``arguments`` dict
    is passed through; it accepts any field of
    ``CreateModelImportJobRequest`` (no exclusions). Typical fields:
    ``ImportedModelName``, ``JobName``, ``RoleArn``, ``ModelDataSource``,
    ``ClientRequestToken``, ``VpcConfig``, ``ImportedModelKmsKeyId``.
    """

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct a ``BedrockModelImportStep``.

        Args:
            name (str): The name of the step.
            arguments (Dict[str, Any]): The ``Arguments`` block for
                ``CreateModelImportJob``.
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            description=description,
            step_type=StepTypeEnum.BEDROCK_MODEL_IMPORT,
            depends_on=depends_on,
        )
        if arguments is None:
            raise ValueError("arguments is required for BedrockModelImportStep.")
        _validate_bedrock_arguments("BedrockModelImportStep", arguments, "CreateModelImportJob")
        self._arguments = arguments
        self._properties = _bedrock_properties(name, self, _BEDROCK_MODEL_IMPORT_FIELDS)

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateModelImportJob`` call."""
        _validate_bedrock_arguments(
            "BedrockModelImportStep", self._arguments, "CreateModelImportJob"
        )
        return self._arguments

    @property
    def properties(self):
        """PascalCase fields from ``GetModelImportJobResponse``."""
        return self._properties


class BedrockProvisionedModelThroughputStep(Step):
    """Creates dedicated provisioned throughput for a Bedrock model.

    Wraps Bedrock's ``CreateProvisionedModelThroughput`` API. The
    ``arguments`` dict is passed through; it accepts any field of
    ``CreateProvisionedModelThroughputRequest`` (no exclusions). Typical
    fields: ``ProvisionedModelName``, ``ModelId``, ``ModelUnits``,
    ``CommitmentDuration`` (``OneMonth``/``SixMonths``/``NoCommitment``),
    ``ClientRequestToken``, ``Tags``.
    """

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct a ``BedrockProvisionedModelThroughputStep``.

        Args:
            name (str): The name of the step.
            arguments (Dict[str, Any]): The ``Arguments`` block for
                ``CreateProvisionedModelThroughput``.
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            description=description,
            step_type=StepTypeEnum.BEDROCK_PROVISIONED_MODEL_THROUGHPUT,
            depends_on=depends_on,
        )
        if arguments is None:
            raise ValueError("arguments is required for BedrockProvisionedModelThroughputStep.")
        _validate_bedrock_arguments(
            "BedrockProvisionedModelThroughputStep", arguments, "CreateProvisionedModelThroughput"
        )
        self._arguments = arguments
        self._properties = _bedrock_properties(
            name, self, _BEDROCK_PROVISIONED_MODEL_THROUGHPUT_FIELDS
        )

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateProvisionedModelThroughput`` call."""
        _validate_bedrock_arguments(
            "BedrockProvisionedModelThroughputStep",
            self._arguments,
            "CreateProvisionedModelThroughput",
        )
        return self._arguments

    @property
    def properties(self):
        """PascalCase fields from ``GetProvisionedModelThroughputResponse``."""
        return self._properties
