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
import json
import logging
import os
import tempfile
import time
import urllib3

from sagemaker.local.data import BatchStrategyFactory, DataSourceFactory, SplitterFactory
from sagemaker.local.image import _SageMakerContainer
from sagemaker.local.utils import copy_directory_structure, move_to_destination
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

    def start(self, input_data_config, hyperparameters):
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

        self.model_artifacts = self.container.train(input_data_config, hyperparameters)
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


class _LocalTransformJob(object):

    _CREATING = 'Creating'

    def __init__(self, transform_job_name, model_name, local_session=None):
        from sagemaker.local import LocalSession
        self.local_session = local_session or LocalSession()
        local_client = self.local_session.sagemaker_client

        self.name = transform_job_name

        # TODO - support SageMaker Models not just local models. This is not
        # ideal but it may be a good thing to do.
        self.primary_container = local_client.describe_model(model_name)['PrimaryContainer']
        self.container = None
        self.create_time = None
        self.state = _LocalTransformJob._CREATING

    def start(self, input_data, output_data, transform_resources, **kwargs):
        image = self.primary_container['Image']
        instance_type = transform_resources['InstanceType']
        instance_count = 1

        self.create_time = datetime.datetime.now()
        environment = self._get_container_environment(**kwargs)

        # Start the container, pass the environment and wait for it to start up
        self.container = _SageMakerContainer(instance_type, instance_count, image, self.local_session)
        self.container.serve(self.primary_container['ModelDataUrl'], environment)

        serving_port = get_config_value('local.serving_port', self.local_session.config) or 8080
        _wait_for_serving_container(serving_port)

        # Get capabilities from Container if needed
        endpoint_url = 'http://localhost:%s/execution-parameters' % serving_port
        response, code = _perform_request(endpoint_url)
        if code == 200:
            execution_parameters = json.load(response.read())
            # MaxConcurrentTransforms is ignored because we currently only support 1
            for setting in ('BatchStrategy', 'MaxPayloadInMB'):
                if setting not in kwargs and setting in execution_parameters:
                    kwargs[setting] = execution_parameters[setting]

        # Apply Defaults if none was provided
        kwargs.update(self._get_required_defaults(**kwargs))

        # run the batch inference requests
        self._batch_inference(input_data, output_data, **kwargs)

    def _get_container_environment(self, **kwargs):
        """Get all the Environment variables that will be passed to the container

        Certain input fields such as BatchStrategy have different values for the API vs the Environment
        variables, su ch as SingleRecord vs SINGLE_RECORD. This method also handles this conversion.

        Args:
            **kwargs: existing transform arguments

        Returns:
            (dict) All the environment variables that should be set in the container

        """
        environment = self.primary_container['Environment']
        environment['SAGEMAKER_BATCH'] = 'True'
        if 'MaxPayloadInMB' in kwargs:
            environment['SAGEMAKER_MAX_PAYLOAD_IN_MB'] = str(kwargs['MaxPayloadInMB'])

        if 'BatchStrategy' in kwargs:
            strategy_env_value = ''
            if kwargs['BatchStrategy'] == 'SingleRecord':
                strategy_env_value = 'SINGLE_RECORD'
            elif kwargs['BatchStrategy'] == 'MultiRecord':
                strategy_env_value = 'MULTI_RECORD'
            else:
                raise ValueError('Invalid BatchStrategy, must be \'SingleRecord\' or \'MultiRecord\'')
            environment['SAGEMAKER_BATCH_STRATEGY'] = strategy_env_value

        # we only do 1 max concurrent transform in Local Mode
        environment['SAGEMAKER_MAX_CONCURRENT_TRANSFORMS'] = '1'

        # if there were environment variables passed to the Transformer we will pass them to the
        # container as well.
        if 'Environment' in kwargs:
            environment.update(kwargs['Environment'])
        return environment

    def _get_required_defaults(self, **kwargs):
        """
        Return the default values for anything that was not provided by either the user or the container
        Args:
            **kwargs: current transform arguments

        Returns:
            (dict) key/values for the default parameters that are missing.

        """
        defaults = {}
        if 'BatchStrategy' not in kwargs:
            defaults['BatchStrategy'] = 'MultiRecord'

        if 'MaxPayloadInMB' not in kwargs:
            defaults['MaxPayloadInMB'] = 6

        return defaults


    def _batch_inference(self, input_data, output_data, **kwargs):
        # TODO - Figure if we should pass FileDataSource here instead. Ideally not but the semantics
        # are just weird.
        input_path = input_data['DataSource']['S3DataSource']['S3Uri']

        # Transform the input data to feed the serving container. We need to first gather the files
        # from S3 or Local FileSystem. Split them as required (Line, RecordIO, None) and finally batch them
        # according to the batch strategy and limit the request size.

        batch_strategy = kwargs['BatchStrategy']
        max_payload = int(kwargs['MaxPayloadInMB'])

        data_source = DataSourceFactory.get_instance(input_path, self.local_session)
        split_type = input_data['SplitType'] if 'SplitType' in input_data else None
        splitter = SplitterFactory.get_instance(split_type)
        final_data = BatchStrategyFactory.get_instance(batch_strategy, splitter)

        # Output settings
        accept = output_data['Accept'] if 'Accept' in output_data else None

        # Root dir to use for intermediate data location. To make things simple we will write here regardless
        # of the final destination. At the end the files will either be moved or uploaded to S3 and deleted.
        root_dir = get_config_value('local.container_root', self.local_session.config)
        if root_dir:
            root_dir = os.path.abspath(root_dir)

        working_dir = tempfile.mkdtemp(dir=root_dir)
        dataset_dir = data_source.get_root_dir()

        for file in data_source.get_file_list():

            relative_path = os.path.dirname(os.path.relpath(file, dataset_dir))
            filename = os.path.basename(file)
            copy_directory_structure(working_dir, relative_path)
            destination_path = os.path.join(working_dir, relative_path, filename + '.out')

            with open(destination_path, 'w') as f:
                for item in final_data.pad(file, max_payload):
                    # call the container and add the result to inference.
                    response = self.local_session.sagemaker_runtime_client.invoke_endpoint(
                        item, '', input_data['ContentType'], accept)

                    response_body = response['Body']
                    data = response_body.read()
                    response_body.close()
                    print('data: %s' % data)
                    f.write(data)
                    if 'AssembleWith' in output_data and output_data['AssembleWith'] == 'Line':
                        f.write('\n')

        print(working_dir)
        move_to_destination(working_dir, output_data['S3OutputPath'], self.local_session)
        self.container.stop_serving()


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

        serving_port = get_config_value('local.serving_port', self.local_session.config) or 8080
        _wait_for_serving_container(serving_port)
        # the container is running and it passed the healthcheck status is now InService
        self.state = _LocalEndpoint._IN_SERVICE

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


def _wait_for_serving_container(serving_port):
    i = 0
    http = urllib3.PoolManager()

    endpoint_url = 'http://localhost:%s/ping' % serving_port
    while True:
        i += 1
        if i >= HEALTH_CHECK_TIMEOUT_LIMIT:
            raise RuntimeError('Giving up, endpoint didn\'t launch correctly')

        logger.info('Checking if serving container is up, attempt: %s' % i)
        response, code = _perform_request(endpoint_url, http)
        if code != 200:
            logger.info('Container still not up, got: %s' % code)
        else:
            return

        time.sleep(1)

def _perform_request(endpoint_url, pool_manager=None):
    http = pool_manager or urllib3.PoolManager()
    try:
        r = http.request('GET', endpoint_url)
        code = r.status
    except urllib3.exceptions.RequestError:
        return None, -1
    return r, code
