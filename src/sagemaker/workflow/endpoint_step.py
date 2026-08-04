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
"""Step definitions for SageMaker Endpoint deployment in Pipelines."""

from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Union

from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.retry import RetryPolicy
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.steps import (
    CacheConfig,
    ConfigurableRetryStep,
    Step,
    StepTypeEnum,
)


class EndpointConfigStep(ConfigurableRetryStep):
    """Creates a SageMaker EndpointConfig within a pipeline.

    Wraps the SageMaker ``CreateEndpointConfig`` API. The ``arguments``
    dict is forwarded to the service — refer to the
    `CreateEndpointConfig API reference
    <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpointConfig.html>`_
    for accepted fields. Values may be pipeline variables (parameter
    references, step property references).
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
                ``CreateEndpointConfig`` call. Values may be pipeline
                variables.
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
        self._arguments = arguments
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeEndpointConfigOutput"
        )

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateEndpointConfig`` call."""
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
    pipeline chooses create-vs-update based on endpoint existence. Refer
    to the `CreateEndpoint API reference
    <https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpoint.html>`_
    for accepted fields.
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
                ``CreateEndpoint``/``UpdateEndpoint`` call. Values may
                be pipeline variables.
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
        self._arguments = arguments
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeEndpointOutput"
        )

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the ``CreateEndpoint``/``UpdateEndpoint`` call."""
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
