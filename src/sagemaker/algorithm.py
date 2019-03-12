# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sagemaker
import sagemaker.parameter
from sagemaker import vpc_utils
from sagemaker.estimator import EstimatorBase
from sagemaker.transformer import Transformer
from sagemaker.predictor import RealTimePredictor


class AlgorithmEstimator(EstimatorBase):
    """A generic Estimator to train using any algorithm object (with an ``algorithm_arn``).
    The Algorithm can be your own, or any Algorithm from AWS Marketplace that you have a valid
    subscription for. This class will perform client-side validation on all the inputs.
    """

    # These Hyperparameter Types have a range definition.
    _hyperpameters_with_range = ('Integer', 'Continuous', 'Categorical')

    def __init__(
        self,
        algorithm_arn,
        role,
        train_instance_count,
        train_instance_type,
        train_volume_size=30,
        train_volume_kms_key=None,
        train_max_run=24 * 60 * 60,
        input_mode='File',
        output_path=None,
        output_kms_key=None,
        base_job_name=None,
        sagemaker_session=None,
        hyperparameters=None,
        tags=None,
        subnets=None,
        security_group_ids=None,
        model_uri=None,
        model_channel_name='model',
        metric_definitions=None,
        encrypt_inter_container_traffic=False
    ):
        """Initialize an ``AlgorithmEstimator`` instance.

        Args:
           algorithm_arn (str): algorithm arn used for training. Can be just the name if your
                account owns the algorithm.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                After the endpoint is created, the inference code might use the IAM role,
                if it needs to access an AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            train_volume_size (int): Size in GB of the EBS volume to use for storing input data
                during training (default: 30). Must be large enough to store training data if File Mode is used
                (which is the default).
            train_volume_kms_key (str): Optional. KMS key ID for encrypting EBS volume attached to the
                training instance (default: None).
            train_max_run (int): Timeout in seconds for training (default: 24 * 60 * 60).
                After this amount of time Amazon SageMaker terminates the job regardless of its current status.
            input_mode (str): The input mode that the algorithm supports (default: 'File'). Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to a local directory.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.

                This argument can be overriden on a per-channel basis using ``sagemaker.session.s3_input.input_mode``.
            output_path (str): S3 location for saving the training result (model artifacts and output files).
                If not specified, results are stored to a default bucket. If the bucket with the specific name
                does not exist, the estimator creates the bucket during the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method execution.
            output_kms_key (str): Optional. KMS key ID for encrypting the training output (default: None).
            base_job_name (str): Prefix for training job name when the :meth:`~sagemaker.estimator.EstimatorBase.fit`
                method launches. If not specified, the estimator generates a default job name, based on
                the training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            tags (list[dict]): List of tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            subnets (list[str]): List of subnet ids. If not specified training job will be created without VPC config.
            security_group_ids (list[str]): List of security group ids. If not specified training job will be created
                without VPC config.
            model_uri (str): URI where a pre-trained model is stored, either locally or in S3 (default: None). If
                specified, the estimator will create a channel pointing to the model so the training job can download
                it. This model can be a 'model.tar.gz' from a previous training job, or other artifacts coming from a
                different source.
                More information: https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
            model_channel_name (str): Name of the channel where 'model_uri' will be downloaded (default: 'model').
            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s) used to evaluate the
                training jobs. Each dictionary contains two keys: 'Name' for the name of the metric, and 'Regex' for
                the regular expression used to extract the metric from the logs.
            encrypt_inter_container_traffic (bool): Specifies whether traffic between training containers is encrypted
                for the training job (default: ``False``).
        """
        self.algorithm_arn = algorithm_arn
        super(AlgorithmEstimator, self).__init__(
            role,
            train_instance_count,
            train_instance_type,
            train_volume_size,
            train_volume_kms_key,
            train_max_run,
            input_mode,
            output_path,
            output_kms_key,
            base_job_name,
            sagemaker_session,
            tags,
            subnets,
            security_group_ids,
            model_uri=model_uri,
            model_channel_name=model_channel_name,
            metric_definitions=metric_definitions,
            encrypt_inter_container_traffic=encrypt_inter_container_traffic
        )

        self.algorithm_spec = self.sagemaker_session.sagemaker_client.describe_algorithm(
            AlgorithmName=algorithm_arn
        )
        self.validate_train_spec()
        self.hyperparameter_definitions = self._parse_hyperparameters()

        self.hyperparam_dict = {}
        if hyperparameters:
            self.set_hyperparameters(**hyperparameters)

    def validate_train_spec(self):
        train_spec = self.algorithm_spec['TrainingSpecification']
        algorithm_name = self.algorithm_spec['AlgorithmName']

        # Check that the input mode provided is compatible with the training input modes for the
        # algorithm.
        train_input_modes = self._algorithm_training_input_modes(train_spec['TrainingChannels'])
        if self.input_mode not in train_input_modes:
            raise ValueError(
                'Invalid input mode: %s. %s only supports: %s'
                % (self.input_mode, algorithm_name, train_input_modes)
            )

        # Check that the training instance type is compatible with the algorithm.
        supported_instances = train_spec['SupportedTrainingInstanceTypes']
        if self.train_instance_type not in supported_instances:
            raise ValueError(
                'Invalid train_instance_type: %s. %s supports the following instance types: %s'
                % (self.train_instance_type, algorithm_name, supported_instances)
            )

        # Verify if distributed training is supported by the algorithm
        if (
            self.train_instance_count > 1
            and 'SupportsDistributedTraining' in train_spec
            and not train_spec['SupportsDistributedTraining']
        ):
            raise ValueError(
                'Distributed training is not supported by %s. '
                'Please set train_instance_count=1' % algorithm_name
            )

    def set_hyperparameters(self, **kwargs):
        for k, v in kwargs.items():
            value = self._validate_and_cast_hyperparameter(k, v)
            self.hyperparam_dict[k] = value

        self._validate_and_set_default_hyperparameters()

    def hyperparameters(self):
        """Returns the hyperparameters as a dictionary to use for training.

        The fit() method, that does the model training, calls this method to find the hyperparameters you specified.
        """
        return self.hyperparam_dict

    def train_image(self):
        """Returns the docker image to use for training.

        The fit() method, that does the model training, calls this method to find the image to use for model training.
        """
        raise RuntimeError('train_image is never meant to be called on Algorithm Estimators')

    def enable_network_isolation(self):
        """Return True if this Estimator will need network isolation to run.

        On Algorithm Estimators this depends on the algorithm being used. If this is algorithm
        owned by your account it will be False. If this is an an algorithm consumed from Marketplace
        it will be True.

        Returns:
            bool: Whether this Estimator needs network isolation or not.
        """
        return self._is_marketplace()

    def create_model(
        self,
        role=None,
        predictor_cls=None,
        serializer=None,
        deserializer=None,
        content_type=None,
        accept=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        **kwargs
    ):
        """Create a model to deploy.

        The serializer, deserializer, content_type, and accept arguments are only used to define a default
        RealTimePredictor. They are ignored if an explicit predictor class is passed in. Other arguments
        are passed through to the Model class.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
            predictor_cls (RealTimePredictor): The predictor class to use when deploying the model.
            serializer (callable): Should accept a single argument, the input data, and return a sequence
                of bytes. May provide a content_type attribute that defines the endpoint request content type
            deserializer (callable): Should accept two arguments, the result data and the response content type,
                and return a sequence of bytes. May provide a content_type attribute that defines th endpoint
                response Accept content type.
            content_type (str): The invocation ContentType, overriding any content_type from the serializer
            accept (str): The invocation Accept, overriding any accept from the deserializer.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

        Returns:
            a Model ready for deployment.
        """
        if predictor_cls is None:

            def predict_wrapper(endpoint, session):
                return RealTimePredictor(
                    endpoint, session, serializer, deserializer, content_type, accept
                )

            predictor_cls = predict_wrapper

        role = role or self.role

        return sagemaker.ModelPackage(
            role,
            algorithm_arn=self.algorithm_arn,
            model_data=self.model_data,
            vpc_config=self.get_vpc_config(vpc_config_override),
            sagemaker_session=self.sagemaker_session,
            predictor_cls=predictor_cls,
            **kwargs
        )

    def transformer(self, instance_count, instance_type, strategy=None, assemble_with=None, output_path=None,
                    output_kms_key=None, accept=None, env=None, max_concurrent_transforms=None,
                    max_payload=None, tags=None, role=None, volume_kms_key=None):
        """Return a ``Transformer`` that uses a SageMaker Model based on the training job. It reuses the
        SageMaker Session and base job name used by the Estimator.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example, 'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in a single request (default: None).
                Valid values: 'MULTI_RECORD' and 'SINGLE_RECORD'.
            assemble_with (str): How the output is assembled (default: None). Valid values: 'Line' or 'None'.
            output_path (str): S3 location for saving the transform result. If not specified, results are stored to
                a default bucket.
            output_kms_key (str): Optional. KMS key ID for encrypting the transform output (default: None).
            accept (str): The content type accepted by the endpoint deployed during the transform job.
            env (dict): Environment variables to be set for use during the transform job (default: None).
            max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
                each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP request to the container in MB.
            tags (list[dict]): List of tags for labeling a transform job. If none specified, then the tags used for
                the training job are used for the transform job.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
            volume_kms_key (str): Optional. KMS key ID for encrypting the volume attached to the ML
                compute instance (default: None).
        """
        role = role or self.role

        if self.latest_training_job is not None:
            model = self.create_model(role=role)
            model._create_sagemaker_model()
            model_name = model.name
            transform_env = {}
            if env is not None:
                transform_env = model.env.copy()
                transform_env.update(env)
            if self._is_marketplace():
                transform_env = None

            tags = tags or self.tags
        else:
            raise RuntimeError('No finished training job found associated with this estimator')

        return Transformer(model_name, instance_count, instance_type, strategy=strategy,
                           assemble_with=assemble_with, output_path=output_path,
                           output_kms_key=output_kms_key, accept=accept,
                           max_concurrent_transforms=max_concurrent_transforms,
                           max_payload=max_payload, env=transform_env, tags=tags,
                           base_transform_job_name=self.base_job_name,
                           volume_kms_key=volume_kms_key, sagemaker_session=self.sagemaker_session)

    def _is_marketplace(self):
        return 'ProductId' in self.algorithm_spec

    def _prepare_for_training(self, job_name=None):
        # Validate hyperparameters
        # an explicit call to set_hyperparameters() will also validate the hyperparameters
        # but it is possible that the user never called it.
        self._validate_and_set_default_hyperparameters()

        super(AlgorithmEstimator, self)._prepare_for_training(job_name)

    def fit(self, inputs=None, wait=True, logs=True, job_name=None):
        if inputs:
            self._validate_input_channels(inputs)

        super(AlgorithmEstimator, self).fit(inputs, wait, logs, job_name)

    def _validate_input_channels(self, channels):
        train_spec = self.algorithm_spec['TrainingSpecification']
        algorithm_name = self.algorithm_spec['AlgorithmName']
        training_channels = {c['Name']: c for c in train_spec['TrainingChannels']}

        # check for unknown channels that the algorithm does not support
        for c in channels:
            if c not in training_channels:
                raise ValueError(
                    'Unknown input channel: %s is not supported by: %s' % (c, algorithm_name)
                )

        # check for required channels that were not provided
        for name, channel in training_channels.items():
            if name not in channels and 'IsRequired' in channel and channel['IsRequired']:
                raise ValueError('Required input channel: %s Was not provided.' % (name))

    def _validate_and_cast_hyperparameter(self, name, v):
        algorithm_name = self.algorithm_spec['AlgorithmName']

        if name not in self.hyperparameter_definitions:
            raise ValueError(
                'Invalid hyperparameter: %s is not supported by %s' % (name, algorithm_name)
            )

        definition = self.hyperparameter_definitions[name]
        if 'class' in definition:
            value = definition['class'].cast_to_type(v)
        else:
            value = v

        if 'range' in definition and not definition['range'].is_valid(value):
            valid_range = definition['range'].as_tuning_range(name)
            raise ValueError('Invalid value: %s Supported range: %s' % (value, valid_range))
        return value

    def _validate_and_set_default_hyperparameters(self):
        # Check if all the required hyperparameters are set. If there is a default value
        # for one, set it.
        for name, definition in self.hyperparameter_definitions.items():
            if name not in self.hyperparam_dict:
                spec = definition['spec']
                if 'DefaultValue' in spec:
                    self.hyperparam_dict[name] = spec['DefaultValue']
                elif 'IsRequired' in spec and spec['IsRequired']:
                    raise ValueError('Required hyperparameter: %s is not set' % name)

    def _parse_hyperparameters(self):
        definitions = {}

        training_spec = self.algorithm_spec['TrainingSpecification']
        if 'SupportedHyperParameters' in training_spec:
            hyperparameters = training_spec['SupportedHyperParameters']
            for h in hyperparameters:
                parameter_type = h['Type']
                name = h['Name']
                parameter_class, parameter_range = self._hyperparameter_range_and_class(
                    parameter_type, h
                )

                definitions[name] = {'spec': h}
                if parameter_range:
                    definitions[name]['range'] = parameter_range
                if parameter_class:
                    definitions[name]['class'] = parameter_class

        return definitions

    def _hyperparameter_range_and_class(self, parameter_type, hyperparameter):
        if parameter_type in self._hyperpameters_with_range:
            range_name = parameter_type + 'ParameterRangeSpecification'

        parameter_class = None
        parameter_range = None

        if parameter_type in ('Integer', 'Continuous'):
            # Integer and Continuous are handled the same way. We get the min and max values
            # and just create an Instance of Parameter. Note that the range is optional for all
            # the Parameter Types.
            if parameter_type == 'Integer':
                parameter_class = sagemaker.parameter.IntegerParameter
            else:
                parameter_class = sagemaker.parameter.ContinuousParameter

            if 'Range' in hyperparameter:
                min_value = parameter_class.cast_to_type(
                    hyperparameter['Range'][range_name]['MinValue']
                )
                max_value = parameter_class.cast_to_type(
                    hyperparameter['Range'][range_name]['MaxValue']
                )
                parameter_range = parameter_class(min_value, max_value)

        elif parameter_type == 'Categorical':
            parameter_class = sagemaker.parameter.CategoricalParameter
            if 'Range' in hyperparameter:
                values = hyperparameter['Range'][range_name]['Values']
                parameter_range = sagemaker.parameter.CategoricalParameter(values)
        elif parameter_type == 'FreeText':
            pass
        else:
            raise ValueError(
                'Invalid Hyperparameter type: %s. Valid ones are:'
                '(Integer, Continuous, Categorical, FreeText)' % parameter_type
            )

        return parameter_class, parameter_range

    def _algorithm_training_input_modes(self, training_channels):
        current_input_modes = {'File', 'Pipe'}
        for channel in training_channels:
            supported_input_modes = set(channel['SupportedInputModes'])
            current_input_modes = current_input_modes & supported_input_modes

        return current_input_modes
