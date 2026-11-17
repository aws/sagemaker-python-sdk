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
"""Step definition for importing SageMaker hub content in Pipelines.

Example::

    pipeline_session = PipelineSession()

    step_args = pipeline_session.import_hub_content(
        hub_name="my-hub",
        hub_content_name="my-dataset",
        hub_content_type="DataSet",
        document_schema_version="2.0.0",
        hub_content_document={...},
    )
    step = HubContentStep(name="ImportDataset", step_args=step_args)
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


class HubContentStep(ConfigurableRetryStep):
    """Imports content into a SageMaker hub within a pipeline.

    Wraps the SageMaker ``ImportHubContent`` API. The ``step_args`` must be
    obtained by calling
    :meth:`~sagemaker.core.helper.session_helper.Session.import_hub_content`
    on a ``PipelineSession``. ``properties`` is shaped like
    ``DescribeHubContentResponse`` for the imported version.
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
        """Construct a ``HubContentStep``.

        Args:
            name (str): The name of the step.
            step_args (_JobStepArguments): The arguments for this step,
                obtained from ``pipeline_session.import_hub_content()``.
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
            cache_config (CacheConfig): Optional cache configuration.
            retry_policies (List[RetryPolicy]): Optional retry policies.
        """
        super().__init__(
            name=name,
            step_type=StepTypeEnum.HUB_CONTENT,
            display_name=display_name,
            description=description,
            depends_on=depends_on,
            retry_policies=retry_policies,
        )
        validate_step_args_input(
            step_args=step_args,
            expected_caller={"import_hub_content"},
            error_message=(
                "The step_args of HubContentStep must be obtained from "
                "pipeline_session.import_hub_content()."
            ),
        )
        self.step_args = step_args
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeHubContentResponse"
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call ``import_hub_content``.

        A dict ``HubContentDocument`` is encoded to its string form here, as
        ``JobStep`` does for ``JobConfigDocument``: pipeline variables inside it
        become a ``Join`` that the service resolves at execution time.
        """
        from sagemaker.core.workflow.job_config_document import (
            convert_job_config_document_to_string,
        )

        request_dict = dict(self.step_args.args)
        document = request_dict.get("HubContentDocument")
        if isinstance(document, dict):
            request_dict["HubContentDocument"] = convert_job_config_document_to_string(document)
        return request_dict

    @property
    def properties(self):
        """A ``Properties`` object shaped like ``DescribeHubContentResponse``."""
        return self._properties

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)
        return request_dict
