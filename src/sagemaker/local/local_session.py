# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Placeholder docstring"""
from __future__ import absolute_import

import logging
import platform

import boto3
import urllib3
from botocore.exceptions import ClientError

from sagemaker.local.image import _SageMakerContainer
from sagemaker.local.entities import (
    _LocalEndpointConfig,
    _LocalEndpoint,
    _LocalModel,
    _LocalTrainingJob,
    _LocalTransformJob,
)
from sagemaker.session import Session
from sagemaker.utils import get_config_value

logger = logging.getLogger(__name__)


class LocalSagemakerClient(object):
    """A SageMakerClient that implements the API calls locally.

    Used for doing local training and hosting local endpoints. It still needs access to
    a boto client to interact with S3 but it won't perform any SageMaker call.

    Implements the methods with the same signature as the boto SageMakerClient.

    Args:

    Returns:

    """

    _training_jobs = {}
    _transform_jobs = {}
    _models = {}
    _endpoint_configs = {}
    _endpoints = {}

    def __init__(self, sagemaker_session=None):
        """Initialize a LocalSageMakerClient.

        Args:
            sagemaker_session (sagemaker.session.Session): a session to use to read configurations
                from, and use its boto client.
        """
        self.sagemaker_session = sagemaker_session or LocalSession()

    def create_training_job(
        self,
        TrainingJobName,
        AlgorithmSpecification,
        OutputDataConfig,
        ResourceConfig,
        InputDataConfig=None,
        **kwargs
    ):
        """Create a training job in Local Mode

        Args:
          TrainingJobName(str): local training job name.
          AlgorithmSpecification(dict): Identifies the training algorithm to use.
          InputDataConfig(dict, optional): Describes the training dataset and the location where
            it is stored. (Default value = None)
          OutputDataConfig(dict): Identifies the location where you want to save the results of
            model training.
          ResourceConfig(dict): Identifies the resources to use for local model training.
          HyperParameters(dict) [optional]: Specifies these algorithm-specific parameters to
            influence the quality of the final model.
          **kwargs:

        Returns:

        """
        InputDataConfig = InputDataConfig or {}
        container = _SageMakerContainer(
            ResourceConfig["InstanceType"],
            ResourceConfig["InstanceCount"],
            AlgorithmSpecification["TrainingImage"],
            self.sagemaker_session,
        )
        training_job = _LocalTrainingJob(container)
        hyperparameters = kwargs["HyperParameters"] if "HyperParameters" in kwargs else {}
        training_job.start(InputDataConfig, OutputDataConfig, hyperparameters, TrainingJobName)

        LocalSagemakerClient._training_jobs[TrainingJobName] = training_job

    def describe_training_job(self, TrainingJobName):
        """Describe a local training job.

        Args:
          TrainingJobName(str): Training job name to describe.
        Returns: (dict) DescribeTrainingJob Response.

        Returns:

        """
        if TrainingJobName not in LocalSagemakerClient._training_jobs:
            error_response = {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Could not find local training job",
                }
            }
            raise ClientError(error_response, "describe_training_job")
        return LocalSagemakerClient._training_jobs[TrainingJobName].describe()

    def create_transform_job(
        self,
        TransformJobName,
        ModelName,
        TransformInput,
        TransformOutput,
        TransformResources,
        **kwargs
    ):
        """

        Args:
          TransformJobName:
          ModelName:
          TransformInput:
          TransformOutput:
          TransformResources:
          **kwargs:

        Returns:

        """
        transform_job = _LocalTransformJob(TransformJobName, ModelName, self.sagemaker_session)
        LocalSagemakerClient._transform_jobs[TransformJobName] = transform_job
        transform_job.start(TransformInput, TransformOutput, TransformResources, **kwargs)

    def describe_transform_job(self, TransformJobName):
        """

        Args:
          TransformJobName:

        Returns:

        """
        if TransformJobName not in LocalSagemakerClient._transform_jobs:
            error_response = {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Could not find local transform job",
                }
            }
            raise ClientError(error_response, "describe_transform_job")
        return LocalSagemakerClient._transform_jobs[TransformJobName].describe()

    def create_model(
        self, ModelName, PrimaryContainer, *args, **kwargs
    ):  # pylint: disable=unused-argument
        """Create a Local Model Object


        Args:
          ModelName (str): the Model Name
          PrimaryContainer (dict): a SageMaker primary container definition
          *args:
          **kwargs:

        Returns:
        """
        LocalSagemakerClient._models[ModelName] = _LocalModel(ModelName, PrimaryContainer)

    def describe_model(self, ModelName):
        """

        Args:
          ModelName:

        Returns:

        """
        if ModelName not in LocalSagemakerClient._models:
            error_response = {
                "Error": {"Code": "ValidationException", "Message": "Could not find local model"}
            }
            raise ClientError(error_response, "describe_model")
        return LocalSagemakerClient._models[ModelName].describe()

    def describe_endpoint_config(self, EndpointConfigName):
        """

        Args:
          EndpointConfigName:

        Returns:

        """
        if EndpointConfigName not in LocalSagemakerClient._endpoint_configs:
            error_response = {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Could not find local endpoint config",
                }
            }
            raise ClientError(error_response, "describe_endpoint_config")
        return LocalSagemakerClient._endpoint_configs[EndpointConfigName].describe()

    def create_endpoint_config(self, EndpointConfigName, ProductionVariants, Tags=None):
        """

        Args:
          EndpointConfigName:
          ProductionVariants:
          Tags:  (Default value = None)

        Returns:

        """
        LocalSagemakerClient._endpoint_configs[EndpointConfigName] = _LocalEndpointConfig(
            EndpointConfigName, ProductionVariants, Tags
        )

    def describe_endpoint(self, EndpointName):
        """

        Args:
          EndpointName:

        Returns:

        """
        if EndpointName not in LocalSagemakerClient._endpoints:
            error_response = {
                "Error": {"Code": "ValidationException", "Message": "Could not find local endpoint"}
            }
            raise ClientError(error_response, "describe_endpoint")
        return LocalSagemakerClient._endpoints[EndpointName].describe()

    def create_endpoint(self, EndpointName, EndpointConfigName, Tags=None):
        """

        Args:
          EndpointName:
          EndpointConfigName:
          Tags:  (Default value = None)

        Returns:

        """
        endpoint = _LocalEndpoint(EndpointName, EndpointConfigName, Tags, self.sagemaker_session)
        LocalSagemakerClient._endpoints[EndpointName] = endpoint
        endpoint.serve()

    def update_endpoint(self, EndpointName, EndpointConfigName):  # pylint: disable=unused-argument
        """

        Args:
          EndpointName:
          EndpointConfigName:

        Returns:

        """
        raise NotImplementedError("Update endpoint name is not supported in local session.")

    def delete_endpoint(self, EndpointName):
        """

        Args:
          EndpointName:

        Returns:

        """
        if EndpointName in LocalSagemakerClient._endpoints:
            LocalSagemakerClient._endpoints[EndpointName].stop()

    def delete_endpoint_config(self, EndpointConfigName):
        """

        Args:
          EndpointConfigName:

        Returns:

        """
        if EndpointConfigName in LocalSagemakerClient._endpoint_configs:
            del LocalSagemakerClient._endpoint_configs[EndpointConfigName]

    def delete_model(self, ModelName):
        """

        Args:
          ModelName:

        Returns:

        """
        if ModelName in LocalSagemakerClient._models:
            del LocalSagemakerClient._models[ModelName]


class LocalSagemakerRuntimeClient(object):
    """A SageMaker Runtime client that calls a local endpoint only."""

    def __init__(self, config=None):
        """Initializes a LocalSageMakerRuntimeClient

        Args:
            config (dict): Optional configuration for this client. In particular only
                the local port is read.
        """
        self.http = urllib3.PoolManager()
        self.serving_port = 8080
        self.config = config
        self.serving_port = get_config_value("local.serving_port", config) or 8080

    def invoke_endpoint(
        self,
        Body,
        EndpointName,  # pylint: disable=unused-argument
        ContentType=None,
        Accept=None,
        CustomAttributes=None,
    ):
        """

        Args:
          Body:
          EndpointName:
          Accept:  (Default value = None)
          CustomAttributes:  (Default value = None)

        Returns:

        """
        url = "http://localhost:%s/invocations" % self.serving_port
        headers = {}

        if ContentType is not None:
            headers["Content-type"] = ContentType

        if Accept is not None:
            headers["Accept"] = Accept

        if CustomAttributes is not None:
            headers["X-Amzn-SageMaker-Custom-Attributes"] = CustomAttributes

        r = self.http.request("POST", url, body=Body, preload_content=False, headers=headers)

        return {"Body": r, "ContentType": Accept}


class LocalSession(Session):
    """Placeholder docstring"""

    def __init__(self, boto_session=None):
        super(LocalSession, self).__init__(boto_session)

        if platform.system() == "Windows":
            logger.warning("Windows Support for Local Mode is Experimental")

    def _initialize(self, boto_session, sagemaker_client, sagemaker_runtime_client):
        """Initialize this Local SageMaker Session.

        Args:
          boto_session:
          sagemaker_client:
          sagemaker_runtime_client:

        Returns:

        """

        self.boto_session = boto_session or boto3.Session()
        self._region_name = self.boto_session.region_name

        if self._region_name is None:
            raise ValueError(
                "Must setup local AWS configuration with a region supported by SageMaker."
            )

        self.sagemaker_client = LocalSagemakerClient(self)
        self.sagemaker_runtime_client = LocalSagemakerRuntimeClient(self.config)
        self.local_mode = True

    def logs_for_job(self, job_name, wait=False, poll=5):
        """

        Args:
          job_name:
          wait:  (Default value = False)
          poll:  (Default value = 5)

        Returns:

        """
        # override logs_for_job() as it doesn't need to perform any action
        # on local mode.
        pass  # pylint: disable=unnecessary-pass


class file_input(object):
    """Amazon SageMaker channel configuration for FILE data sources, used in local mode."""

    def __init__(self, fileUri, content_type=None):
        """Create a definition for input data used by an SageMaker training job in local mode.
        """
        self.config = {
            "DataSource": {
                "FileDataSource": {
                    "FileDataDistributionType": "FullyReplicated",
                    "FileUri": fileUri,
                }
            }
        }

        if content_type is not None:
            self.config["ContentType"] = content_type
