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
import time
import urllib3

from sagemaker.local.image import _SageMakerContainer
from sagemaker.utils import get_config_value

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

_UNUSED_ARN = 'local:arn-does-not-matter'
HEALTH_CHECK_TIMEOUT_LIMIT = 30


class _LocalTrainingJob(object):

    _STARTING = 'Starting'
    _TRAINING = 'Training'
    _COMPLETED = 'Completed'
    _states = ['Starting', 'Training', 'Completed']

    def __init__(self, container):
        self.container = container
        self.model_artifacts = None
        self.state = 'created'
        self.start_time = None
        self.end_time = None

    def start(self, input_data_config, hyperparameters, job_name):
        for channel in input_data_config:
            if channel['DataSource'] and 'S3DataSource' in channel['DataSource']:
                data_distribution = channel['DataSource']['S3DataSource']['S3DataDistributionType']
            elif channel['DataSource'] and 'FileDataSource' in channel['DataSource']:
                data_distribution = channel['DataSource']['FileDataSource']['FileDataDistributionType']
            else:
                raise ValueError('Need channel[\'DataSource\'] to have [\'S3DataSource\'] or [\'FileDataSource\']')

            if data_distribution != 'FullyReplicated':
                raise RuntimeError('DataDistribution: %s is not currently supported in Local Mode' %
                                   data_distribution)

        self.start = datetime.datetime.now()
        self.state = self._TRAINING

        self.model_artifacts = self.container.train(input_data_config, hyperparameters, job_name)
        self.end = datetime.datetime.now()
        self.state = self._COMPLETED

    def describe(self):
        response = {
            'ResourceConfig': {
                'InstanceCount': self.container.instance_count
            },
            'TrainingJobStatus': self.state,
            'TrainingStartTime': self.start_time,
            'TrainingEndTime': self.end_time,
            'ModelArtifacts': {
                'S3ModelArtifacts': self.model_artifacts
            }
        }
        return response


class _LocalModel(object):

    def __init__(self, model_name, primary_container):
        self.model_name = model_name
        self.primary_container = primary_container
        self.creation_time = datetime.datetime.now()

    def describe(self):
        response = {
            'ModelName': self.model_name,
            'CreationTime': self.creation_time,
            'ExecutionRoleArn': _UNUSED_ARN,
            'ModelArn': _UNUSED_ARN,
            'PrimaryContainer': self.primary_container
        }
        return response


class _LocalEndpointConfig(object):

    def __init__(self, config_name, production_variants):
        self.name = config_name
        self.production_variants = production_variants
        self.creation_time = datetime.datetime.now()

    def describe(self):
        response = {
            'EndpointConfigName': self.name,
            'EndpointConfigArn': _UNUSED_ARN,
            'CreationTime': self.creation_time,
            'ProductionVariants': self.production_variants
        }
        return response


class _LocalEndpoint(object):

    _CREATING = 'Creating'
    _IN_SERVICE = 'InService'
    _FAILED = 'Failed'

    def __init__(self, endpoint_name, endpoint_config_name, local_session=None):
        # runtime import since there is a cyclic dependency between entities and local_session
        from sagemaker.local import LocalSession
        self.local_session = local_session or LocalSession()
        local_client = self.local_session.sagemaker_client

        self.name = endpoint_name
        self.endpoint_config = local_client.describe_endpoint_config(endpoint_config_name)
        self.production_variant = self.endpoint_config['ProductionVariants'][0]

        model_name = self.production_variant['ModelName']
        self.primary_container = local_client.describe_model(model_name)['PrimaryContainer']

        self.container = None
        self.create_time = None
        self.state = _LocalEndpoint._CREATING

    def serve(self):
        image = self.primary_container['Image']
        instance_type = self.production_variant['InstanceType']
        instance_count = self.production_variant['InitialInstanceCount']

        self.create_time = datetime.datetime.now()
        self.container = _SageMakerContainer(instance_type, instance_count, image, self.local_session)
        self.container.serve(self.primary_container['ModelDataUrl'], self.primary_container['Environment'])

        i = 0
        http = urllib3.PoolManager()
        serving_port = get_config_value('local.serving_port', self.local_session.config) or 8080
        endpoint_url = 'http://localhost:%s/ping' % serving_port
        while True:
            i += 1
            if i >= HEALTH_CHECK_TIMEOUT_LIMIT:
                self.state = _LocalEndpoint._FAILED
                raise RuntimeError('Giving up, endpoint: %s didn\'t launch correctly' % self.name)

            logger.info('Checking if endpoint is up, attempt: %s' % i)
            try:
                r = http.request('GET', endpoint_url)
                if r.status != 200:
                    logger.info('Container still not up, got: %s' % r.status)
                else:
                    # the container is running and it passed the healthcheck status is now InService
                    self.state = _LocalEndpoint._IN_SERVICE
                    return
            except urllib3.exceptions.RequestError:
                logger.info('Container still not up')

            time.sleep(1)

    def stop(self):
        if self.container:
            self.container.stop_serving()

    def describe(self):
        response = {
            'EndpointConfigName': self.endpoint_config['EndpointConfigName'],
            'CreationTime': self.create_time,
            'ProductionVariants': self.endpoint_config['ProductionVariants'],
            'EndpointName': self.name,
            'EndpointArn': _UNUSED_ARN,
            'EndpointStatus': self.state
        }
        return response
