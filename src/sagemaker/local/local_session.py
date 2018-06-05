# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import datetime
import logging
import platform
import time

import boto3
import urllib3
from botocore.exceptions import ClientError

from sagemaker.local.image import _SageMakerContainer
from sagemaker.session import Session
from sagemaker.utils import get_config_value

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class LocalSagemakerClient(object):
    """A SageMakerClient that implements the API calls locally.

    Used for doing local training and hosting local endpoints. It still needs access to
    a boto client to interact with S3 but it won't perform any SageMaker call.

    Implements the methods with the same signature as the boto SageMakerClient.
    """
    def __init__(self, sagemaker_session=None):
        """Initialize a LocalSageMakerClient.

        Args:
            sagemaker_session (sagemaker.session.Session): a session to use to read configurations
                from, and use its boto client.
        """
        self.train_container = None
        self.serve_container = None
        self.sagemaker_session = sagemaker_session or LocalSession()
        self.s3_model_artifacts = None
        self.model_name = None
        self.primary_container = None
        self.role_arn = None
        self.created_endpoint = False

    def create_training_job(self, TrainingJobName, AlgorithmSpecification, RoleArn, InputDataConfig, OutputDataConfig,
                            ResourceConfig, StoppingCondition, HyperParameters, Tags=None):

        self.train_container = _SageMakerContainer(ResourceConfig['InstanceType'], ResourceConfig['InstanceCount'],
                                                   AlgorithmSpecification['TrainingImage'], self.sagemaker_session)

        for channel in InputDataConfig:

            if channel['DataSource'] and 'S3DataSource' in channel['DataSource']:
                data_distribution = channel['DataSource']['S3DataSource']['S3DataDistributionType']
            elif channel['DataSource'] and 'FileDataSource' in channel['DataSource']:
                data_distribution = channel['DataSource']['FileDataSource']['FileDataDistributionType']
            else:
                raise ValueError('Need channel[\'DataSource\'] to have [\'S3DataSource\'] or [\'FileDataSource\']')

            if data_distribution != 'FullyReplicated':
                raise RuntimeError("DataDistribution: %s is not currently supported in Local Mode" %
                                   data_distribution)

        self.s3_model_artifacts = self.train_container.train(InputDataConfig, HyperParameters)

    def describe_training_job(self, TrainingJobName):
        """Describe a local training job.

        Args:
            TrainingJobName (str): Not used in this implmentation.

        Returns: (dict) DescribeTrainingJob Response.

        """
        response = {'ResourceConfig': {'InstanceCount': self.train_container.instance_count},
                    'TrainingJobStatus': 'Completed',
                    'TrainingStartTime': datetime.datetime.now(),
                    'TrainingEndTime': datetime.datetime.now(),
                    'ModelArtifacts': {'S3ModelArtifacts': self.s3_model_artifacts}
                    }
        return response

    def create_model(self, ModelName, PrimaryContainer, ExecutionRoleArn):
        self.model_name = ModelName
        self.primary_container = PrimaryContainer
        self.role_arn = ExecutionRoleArn

    def describe_endpoint_config(self, EndpointConfigName):
        if self.created_endpoint:
            return True
        else:
            error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Could not find endpoint'}}
            raise ClientError(error_response, 'describe_endpoint_config')

    def create_endpoint_config(self, EndpointConfigName, ProductionVariants):
        self.variants = ProductionVariants

    def describe_endpoint(self, EndpointName):
        return {'EndpointStatus': 'InService'}

    def create_endpoint(self, EndpointName, EndpointConfigName):
        instance_type = self.variants[0]['InstanceType']
        instance_count = self.variants[0]['InitialInstanceCount']
        self.serve_container = _SageMakerContainer(instance_type, instance_count,
                                                   self.primary_container['Image'], self.sagemaker_session)
        self.serve_container.serve(self.primary_container)
        self.created_endpoint = True

        i = 0
        http = urllib3.PoolManager()
        serving_port = get_config_value('local.serving_port', self.sagemaker_session.config) or 8080
        endpoint_url = "http://localhost:%s/ping" % serving_port
        while True:
            i += 1
            if i >= 10:
                raise RuntimeError("Giving up, endpoint: %s didn't launch correctly" % EndpointName)

            logger.info("Checking if endpoint is up, attempt: %s" % i)
            try:
                r = http.request('GET', endpoint_url)
                if r.status != 200:
                    logger.info("Container still not up, got: %s" % r.status)
                else:
                    return
            except urllib3.exceptions.RequestError:
                logger.info("Container still not up")

            time.sleep(1)

    def delete_endpoint(self, EndpointName):
        self.serve_container.stop_serving()


class LocalSagemakerRuntimeClient(object):
    """A SageMaker Runtime client that calls a local endpoint only.

    """
    def __init__(self, config=None):
        """Initializes a LocalSageMakerRuntimeClient

        Args:
            config (dict): Optional configuration for this client. In particular only
                the local port is read.
        """
        self.http = urllib3.PoolManager()
        self.serving_port = 8080
        self.config = config
        self.serving_port = get_config_value('local.serving_port', config) or 8080

    def invoke_endpoint(self, Body, EndpointName, ContentType, Accept):
        url = "http://localhost:%s/invocations" % self.serving_port
        r = self.http.request('POST', url, body=Body, preload_content=False,
                              headers={'Content-type': ContentType, 'Accept': Accept})

        return {'Body': r, 'ContentType': Accept}


class LocalSession(Session):

    def __init__(self, boto_session=None):
        super(LocalSession, self).__init__(boto_session)

        if platform.system() == 'Windows':
            logger.warning("Windows Support for Local Mode is Experimental")

    def _initialize(self, boto_session, sagemaker_client, sagemaker_runtime_client):
        """Initialize this Local SageMaker Session."""

        self.boto_session = boto_session or boto3.Session()
        self._region_name = self.boto_session.region_name

        if self._region_name is None:
            raise ValueError('Must setup local AWS configuration with a region supported by SageMaker.')

        self.sagemaker_client = LocalSagemakerClient(self)
        self.sagemaker_runtime_client = LocalSagemakerRuntimeClient(self.config)
        self.local_mode = True

    def logs_for_job(self, job_name, wait=False, poll=5):
        # override logs_for_job() as it doesn't need to perform any action
        # on local mode.
        pass


class file_input(object):
    """Amazon SageMaker channel configuration for FILE data sources, used in local mode.

    Attributes:
        config (dict[str, dict]): A SageMaker ``DataSource`` referencing a SageMaker ``FileDataSource``.
    """

    def __init__(self, fileUri, content_type=None):
        """Create a definition for input data used by an SageMaker training job in local mode.
        """
        self.config = {
            'DataSource': {
                'FileDataSource': {
                    'FileDataDistributionType': 'FullyReplicated',
                    'FileUri': fileUri
                }
            }
        }

        if content_type is not None:
            self.config['ContentType'] = content_type
