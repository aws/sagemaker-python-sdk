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
"""Step definitions for SageMaker Endpoint deployment in Pipelines.

Design note: these classes mirror Tioga's own step definitions
(``EndpointConfigStep``, ``EndpointStep`` in
``IronmanTiogaPipelineDefinitionRepository``). Tioga models each step's
``Arguments`` block as an opaque ``StructureArgument`` validated against
the underlying SageMaker Coral request model (``CreateEndpointConfigInput``
or ``CreateEndpointInput``) minus a small exclusion set. This SDK
validates the **top-level keys** of the ``arguments`` dict against the
public ``CreateEndpointConfig``/``CreateEndpoint`` API input shape at
construction time (values are not validated -- they may be pipeline
variables) and forwards the dict to the service, which remains the
authority on full schema validation.

Excluded fields (Tioga will reject the pipeline if present):

* ``EndpointConfig``: ``DataCaptureConfig``, ``ExplainerConfig``
* ``Endpoint``: ``DeploymentConfig``
"""

from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Union

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.properties import Properties

from sagemaker.mlops.workflow._argument_validation import validate_step_arguments
from sagemaker.mlops.workflow.retry import RetryPolicy
from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import (
    CacheConfig,
    ConfigurableRetryStep,
    Step,
    StepTypeEnum,
)


class EndpointConfigStep(ConfigurableRetryStep):
    """Creates a SageMaker EndpointConfig within a pipeline.

    Wraps the SageMaker ``CreateEndpointConfig`` API. The ``arguments``
    dict is passed through to the service; it accepts any field of
    ``CreateEndpointConfigInput`` **except** ``DataCaptureConfig`` and
    ``ExplainerConfig``, which are rejected by Tioga.

    Per the Zimmer step contract, ``EndpointConfig`` is structurally
    cacheable (``cache_config``) and retryable (``retry_policies``).
    """

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        cache_config: Optional[CacheConfig] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
    ):
        """Construct an ``EndpointConfigStep``.

        Args:
            name (str): The name of the step.
            arguments (Dict[str, Any]): The ``Arguments`` block for the
                ``CreateEndpointConfig`` call. Required fields:
                ``EndpointConfigName``, ``ProductionVariants``. Optional
                fields include ``KmsKeyId``, ``AsyncInferenceConfig``,
                ``ShadowProductionVariants``, ``ExecutionRoleArn``,
                ``VpcConfig``, ``EnableNetworkIsolation``,
                ``MetricsConfig``. Values may be pipeline variables
                (parameter references, step property references) — the
                pipeline compiler resolves them at definition time.
                Do not include ``DataCaptureConfig`` or ``ExplainerConfig``
                (Tioga rejects them).
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
            cache_config (CacheConfig): Optional cache configuration.
            retry_policies (List[RetryPolicy]): Optional retry policies.
        """
        super().__init__(
            name=name,
            step_type=StepTypeEnum.ENDPOINT_CONFIG,
            display_name=display_name,
            description=description,
            depends_on=depends_on,
            retry_policies=retry_policies,
        )
        if arguments is None:
            raise ValueError("arguments is required for EndpointConfigStep.")
        validate_step_arguments(
            "EndpointConfigStep",
            arguments,
            service_name="sagemaker",
            operation_name="CreateEndpointConfig",
            unsupported_fields=("DataCaptureConfig", "ExplainerConfig"),
        )
        self._arguments = arguments
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeEndpointConfigOutput"
        )

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateEndpointConfig`` call."""
        validate_step_arguments(
            "EndpointConfigStep",
            self._arguments,
            service_name="sagemaker",
            operation_name="CreateEndpointConfig",
            unsupported_fields=("DataCaptureConfig", "ExplainerConfig"),
        )
        return self._arguments

    @property
    def properties(self):
        """A ``Properties`` object shaped like ``DescribeEndpointConfigOutput``."""
        return self._properties

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)
        return request_dict


class EndpointStep(Step):
    """Creates or updates a SageMaker Endpoint within a pipeline.

    Wraps the SageMaker ``CreateEndpoint``/``UpdateEndpoint`` API — the
    pipeline chooses create-vs-update based on endpoint existence. The
    ``arguments`` dict is passed through to the service; it accepts any
    field of ``CreateEndpointInput`` **except** ``DeploymentConfig``,
    which is rejected by Tioga.

    Per the Zimmer step contract, ``Endpoint`` is structurally cacheable
    but not retryable at the pipeline level.
    """

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        """Construct an ``EndpointStep``.

        Args:
            name (str): The name of the step.
            arguments (Dict[str, Any]): The ``Arguments`` block for the
                ``CreateEndpoint`` / ``UpdateEndpoint`` call. Required
                fields: ``EndpointName``, ``EndpointConfigName``. Optional
                fields: ``GraphConfigName``, ``DeletionCondition``.
                Values may be pipeline variables. Do not include
                ``DeploymentConfig`` (Tioga rejects it).
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
            cache_config (CacheConfig): Optional cache configuration.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            description=description,
            step_type=StepTypeEnum.ENDPOINT,
            depends_on=depends_on,
        )
        if arguments is None:
            raise ValueError("arguments is required for EndpointStep.")
        validate_step_arguments(
            "EndpointStep",
            arguments,
            service_name="sagemaker",
            operation_name="CreateEndpoint",
            unsupported_fields=("DeploymentConfig",),
        )
        self._arguments = arguments
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeEndpointOutput"
        )

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateEndpoint``/``UpdateEndpoint`` call."""
        validate_step_arguments(
            "EndpointStep",
            self._arguments,
            service_name="sagemaker",
            operation_name="CreateEndpoint",
            unsupported_fields=("DeploymentConfig",),
        )
        return self._arguments

    @property
    def properties(self):
        """A ``Properties`` object shaped like ``DescribeEndpointOutput``."""
        return self._properties

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)
        return request_dict
