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

These steps follow the ``step_args`` convention used by ``TrainingStep``
and ``ModelStep``: call the corresponding session method under a
:class:`~sagemaker.core.workflow.pipeline_context.PipelineSession` and
pass the returned step arguments to the step. The request is captured at
call time and the service call is deferred to pipeline execution.

Example::

    pipeline_session = PipelineSession()

    config_step_args = pipeline_session.endpoint_from_production_variants(
        name="my-endpoint-config",
        production_variants=[...],
    )
    config_step = EndpointConfigStep(name="CreateConfig", step_args=config_step_args)

    endpoint_step_args = pipeline_session.create_endpoint(
        endpoint_name="my-endpoint",
        config_name="my-endpoint-config",
    )
    endpoint_step = EndpointStep(name="CreateEndpoint", step_args=endpoint_step_args)
"""

from __future__ import absolute_import

from typing import List, Optional, Union

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.pipeline_context import _JobStepArguments
from sagemaker.core.workflow.properties import Properties
from sagemaker.core.workflow.utilities import validate_step_args_input

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

    Wraps the SageMaker ``CreateEndpointConfig`` API. The ``step_args``
    must be obtained by calling
    :meth:`~sagemaker.core.helper.session_helper.Session.endpoint_from_production_variants`
    on a ``PipelineSession``.

    ``EndpointConfig`` is structurally cacheable (``cache_config``) and
    retryable (``retry_policies``).
    """

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        cache_config: Optional[CacheConfig] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
    ):
        """Construct an ``EndpointConfigStep``.

        Args:
            name (str): The name of the step.
            step_args (_JobStepArguments): The arguments for this step,
                obtained from
                ``pipeline_session.endpoint_from_production_variants()``.
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
        validate_step_args_input(
            step_args=step_args,
            expected_caller={"endpoint_from_production_variants"},
            error_message=(
                "The step_args of EndpointConfigStep must be obtained from "
                "pipeline_session.endpoint_from_production_variants()."
            ),
        )
        self.step_args = step_args
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeEndpointConfigOutput"
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call ``create_endpoint_config``."""
        return self.step_args.args

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

    Wraps the SageMaker ``CreateEndpoint``/``UpdateEndpoint`` API -- the
    pipeline chooses create-vs-update based on endpoint existence. The
    ``step_args`` must be obtained by calling
    :meth:`~sagemaker.core.helper.session_helper.Session.create_endpoint`
    on a ``PipelineSession``.

    ``Endpoint`` is structurally cacheable but not retryable at the
    pipeline level.
    """

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        """Construct an ``EndpointStep``.

        Args:
            name (str): The name of the step.
            step_args (_JobStepArguments): The arguments for this step,
                obtained from ``pipeline_session.create_endpoint()``.
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
        validate_step_args_input(
            step_args=step_args,
            expected_caller={"create_endpoint"},
            error_message=(
                "The step_args of EndpointStep must be obtained from "
                "pipeline_session.create_endpoint()."
            ),
        )
        self.step_args = step_args
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeEndpointOutput"
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call ``create_endpoint``."""
        return self.step_args.args

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
