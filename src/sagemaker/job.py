# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from abc import abstractmethod
from six import string_types

from sagemaker.local import file_input
from sagemaker.session import s3_input


class _Job(object):
    """Handle creating, starting and waiting for Amazon SageMaker jobs to finish.

    This class shouldn't be directly instantiated.

    Subclasses must define a way to create, start and wait for an Amazon SageMaker job.
    """

    def __init__(self, sagemaker_session, job_name):
        self.sagemaker_session = sagemaker_session
        self.job_name = job_name

    @abstractmethod
    def start_new(cls, estimator, inputs):
        """Create a new Amazon SageMaker job from the estimator.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): Estimator object created by the user.
            inputs (str): Parameters used when called  :meth:`~sagemaker.estimator.EstimatorBase.fit`.

        Returns:
            sagemaker.job: Constructed object that captures all information about the started job.
        """
        pass

    @abstractmethod
    def wait(self):
        """Wait for the Amazon SageMaker job to finish.
        """
        pass

    @staticmethod
    def _load_config(inputs, estimator):
        input_config = _Job._format_inputs_to_input_config(inputs)
        role = estimator.sagemaker_session.expand_role(estimator.role)
        output_config = _Job._prepare_output_config(estimator.output_path, estimator.output_kms_key)
        resource_config = _Job._prepare_resource_config(estimator.train_instance_count,
                                                        estimator.train_instance_type,
                                                        estimator.train_volume_size)
        stop_condition = _Job._prepare_stop_condition(estimator.train_max_run)

        return {'input_config': input_config,
                'role': role,
                'output_config': output_config,
                'resource_config': resource_config,
                'stop_condition': stop_condition}

    @staticmethod
    def _format_inputs_to_input_config(inputs):
        # Deferred import due to circular dependency
        from sagemaker.amazon.amazon_estimator import RecordSet
        if isinstance(inputs, RecordSet):
            inputs = inputs.data_channel()

        input_dict = {}
        if isinstance(inputs, string_types):
            input_dict['training'] = _Job._format_string_uri_input(inputs)
        elif isinstance(inputs, s3_input):
            input_dict['training'] = inputs
        elif isinstance(input, file_input):
            input_dict['training'] = inputs
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                input_dict[k] = _Job._format_string_uri_input(v)
        elif isinstance(inputs, list):
            input_dict = _Job._format_record_set_list_input(inputs)
        else:
            raise ValueError(
                'Cannot format input {}. Expecting one of str, dict or s3_input'.format(inputs))

        channels = []
        for channel_name, channel_s3_input in input_dict.items():
            channel_config = channel_s3_input.config.copy()
            channel_config['ChannelName'] = channel_name
            channels.append(channel_config)
        return channels

    @staticmethod
    def _format_string_uri_input(input):
        if isinstance(input, str):
            if input.startswith('s3://'):
                return s3_input(input)
            elif input.startswith('file://'):
                return file_input(input)
            else:
                raise ValueError(
                    'Training input data must be a valid S3 or FILE URI: must start with "s3://" or '
                    '"file://"')
        elif isinstance(input, s3_input):
            return input
        elif isinstance(input, file_input):
            return input
        else:
            raise ValueError(
                'Cannot format input {}. Expecting one of str, s3_input, or file_input'.format(
                    input))

    @staticmethod
    def _format_record_set_list_input(inputs):
        # Deferred import due to circular dependency
        from sagemaker.amazon.amazon_estimator import RecordSet

        input_dict = {}
        for record in inputs:
            if not isinstance(record, RecordSet):
                raise ValueError('List compatible only with RecordSets.')

            if record.channel in input_dict:
                raise ValueError('Duplicate channels not allowed.')

            input_dict[record.channel] = record.records_s3_input()

        return input_dict

    @staticmethod
    def _prepare_output_config(s3_path, kms_key_id):
        config = {'S3OutputPath': s3_path}
        if kms_key_id is not None:
            config['KmsKeyId'] = kms_key_id
        return config

    @staticmethod
    def _prepare_resource_config(instance_count, instance_type, volume_size):
        return {'InstanceCount': instance_count,
                'InstanceType': instance_type,
                'VolumeSizeInGB': volume_size}

    @staticmethod
    def _prepare_stop_condition(max_run):
        return {'MaxRuntimeInSeconds': max_run}

    @property
    def name(self):
        return self.job_name
