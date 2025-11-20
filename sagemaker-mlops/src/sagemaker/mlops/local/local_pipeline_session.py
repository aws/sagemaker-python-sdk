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
"""Local Pipeline Session - extends LocalSession with pipeline execution capabilities."""
from __future__ import absolute_import

import logging
from datetime import datetime
from botocore.exceptions import ClientError

from sagemaker.core.local import LocalSession
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.mlops.local.pipeline_entities import _LocalPipeline

logger = logging.getLogger(__name__)


class LocalPipelineSession(LocalSession):
    """Extends LocalSession with pipeline execution capabilities.
    
    This class provides local pipeline execution functionality that was previously
    in LocalSession. It's now in the MLOps package since pipeline orchestration
    is an MLOps concern.
    
    Usage:
        from sagemaker.mlops.local import LocalPipelineSession
        from sagemaker.mlops.workflow import Pipeline
        
        session = LocalPipelineSession()
        session.create_pipeline(pipeline, "My pipeline")
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize LocalPipelineSession.
        
        Accepts the same arguments as LocalSession.
        """
        super().__init__(*args, **kwargs)
        # Add pipeline storage to the sagemaker_client
        if not hasattr(self.sagemaker_client, '_pipelines'):
            self.sagemaker_client._pipelines = {}
    
    @_telemetry_emitter(Feature.LOCAL_MODE, "local_pipeline_session.create_pipeline")
    def create_pipeline(
        self, pipeline, pipeline_description, **kwargs  # pylint: disable=unused-argument
    ):
        """Create a local pipeline.

        Args:
            pipeline (Pipeline): Pipeline object
            pipeline_description (str): Description of the pipeline

        Returns:
            Pipeline metadata (PipelineArn)
        """
        local_pipeline = _LocalPipeline(
            pipeline=pipeline,
            pipeline_description=pipeline_description,
            local_session=self,
        )
        self.sagemaker_client._pipelines[pipeline.name] = local_pipeline
        return {"PipelineArn": pipeline.name}

    def update_pipeline(
        self, pipeline, pipeline_description, **kwargs  # pylint: disable=unused-argument
    ):
        """Update a local pipeline.

        Args:
            pipeline (Pipeline): Pipeline object
            pipeline_description (str): Description of the pipeline

        Returns:
            Pipeline metadata (PipelineArn)
        """
        if pipeline.name not in self.sagemaker_client._pipelines:
            error_response = {
                "Error": {
                    "Code": "ResourceNotFound",
                    "Message": "Pipeline {} does not exist".format(pipeline.name),
                }
            }
            raise ClientError(error_response, "update_pipeline")
        self.sagemaker_client._pipelines[pipeline.name].pipeline_description = pipeline_description
        self.sagemaker_client._pipelines[pipeline.name].pipeline = pipeline
        self.sagemaker_client._pipelines[pipeline.name].last_modified_time = (
            datetime.now().timestamp()
        )
        return {"PipelineArn": pipeline.name}

    def describe_pipeline(self, PipelineName):
        """Describe the pipeline.

        Args:
          PipelineName (str): Name of the pipeline

        Returns:
            Pipeline metadata (PipelineArn, PipelineDefinition, LastModifiedTime, etc)
        """
        if PipelineName not in self.sagemaker_client._pipelines:
            error_response = {
                "Error": {
                    "Code": "ResourceNotFound",
                    "Message": "Pipeline {} does not exist".format(PipelineName),
                }
            }
            raise ClientError(error_response, "describe_pipeline")
        return self.sagemaker_client._pipelines[PipelineName].describe()

    def delete_pipeline(self, PipelineName):
        """Delete the local pipeline.

        Args:
          PipelineName (str): Name of the pipeline

        Returns:
            Pipeline metadata (PipelineArn)
        """
        if PipelineName in self.sagemaker_client._pipelines:
            del self.sagemaker_client._pipelines[PipelineName]
        return {"PipelineArn": PipelineName}

    def start_pipeline_execution(self, PipelineName, **kwargs):
        """Start the pipeline.

        Args:
          PipelineName (str): Name of the pipeline

        Returns: 
            _LocalPipelineExecution object
        """
        if "ParallelismConfiguration" in kwargs:
            logger.warning("Parallelism configuration is not supported in local mode.")
        if "SelectiveExecutionConfig" in kwargs:
            raise ValueError("SelectiveExecutionConfig is not supported in local mode.")
        if PipelineName not in self.sagemaker_client._pipelines:
            error_response = {
                "Error": {
                    "Code": "ResourceNotFound",
                    "Message": "Pipeline {} does not exist".format(PipelineName),
                }
            }
            raise ClientError(error_response, "start_pipeline_execution")
        return self.sagemaker_client._pipelines[PipelineName].start(**kwargs)
