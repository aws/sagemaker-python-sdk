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
"""Step definitions for Amazon Bedrock deployment steps in Pipelines."""

from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Union

from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.steps import Step, StepTypeEnum

# Property paths for each Bedrock step, sourced from each ``Get*Response`` shape.
# Users reference these via ``step.properties.<field>``.
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
    forwarded to the service. Typical fields: ``ModelName``, ``RoleArn``,
    ``ModelSourceConfig``, ``ClientRequestToken``, ``ModelKmsKeyArn``.
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
                ``CreateCustomModel``. ``ClientRequestToken`` is optional
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
        self._arguments = arguments
        self._properties = _bedrock_properties(name, self, _BEDROCK_CUSTOM_MODEL_FIELDS)

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateCustomModel`` call."""
        return self._arguments

    @property
    def properties(self):
        """Fields from ``GetCustomModelResponse``."""
        return self._properties


class BedrockCustomModelDeploymentStep(Step):
    """Deploys a Bedrock custom model for inference.

    Wraps Bedrock's ``CreateCustomModelDeployment`` API. The ``arguments``
    dict is forwarded to the service.
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
                ``CreateCustomModelDeployment``.
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
        self._arguments = arguments
        self._properties = _bedrock_properties(name, self, _BEDROCK_CUSTOM_MODEL_DEPLOYMENT_FIELDS)

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateCustomModelDeployment`` call."""
        return self._arguments

    @property
    def properties(self):
        """Fields from ``GetCustomModelDeploymentResponse``."""
        return self._properties


class BedrockModelImportStep(Step):
    """Imports a SageMaker-trained model into Bedrock.

    Wraps Bedrock's ``CreateModelImportJob`` API. The ``arguments`` dict
    is forwarded to the service.
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
        self._arguments = arguments
        self._properties = _bedrock_properties(name, self, _BEDROCK_MODEL_IMPORT_FIELDS)

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateModelImportJob`` call."""
        return self._arguments

    @property
    def properties(self):
        """Fields from ``GetModelImportJobResponse``."""
        return self._properties


class BedrockProvisionedModelThroughputStep(Step):
    """Creates dedicated provisioned throughput for a Bedrock model.

    Wraps Bedrock's ``CreateProvisionedModelThroughput`` API. The
    ``arguments`` dict is forwarded to the service.
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
        self._arguments = arguments
        self._properties = _bedrock_properties(
            name, self, _BEDROCK_PROVISIONED_MODEL_THROUGHPUT_FIELDS
        )

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateProvisionedModelThroughput`` call."""
        return self._arguments

    @property
    def properties(self):
        """Fields from ``GetProvisionedModelThroughputResponse``."""
        return self._properties
