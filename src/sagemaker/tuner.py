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

import importlib
import inspect
import json

from sagemaker.amazon.amazon_estimator import AmazonAlgorithmEstimatorBase, RecordSet
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from sagemaker.estimator import Framework
from sagemaker.job import _Job
from sagemaker.session import Session
from sagemaker.utils import base_name_from_image, name_from_base, to_str

AMAZON_ESTIMATOR_MODULE = 'sagemaker'
AMAZON_ESTIMATOR_CLS_NAMES = {
    'factorization-machines': 'FactorizationMachines',
    'kmeans': 'KMeans',
    'lda': 'LDA',
    'linear-learner': 'LinearLearner',
    'ntm': 'NTM',
    'pca': 'PCA',
    'randomcutforest': 'RandomCutForest',
}


class _ParameterRange(object):
    __all_types__ = ['Continuous', 'Categorical', 'Integer']

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def as_tuning_range(self, name):
        return {'Name': name,
                'MinValue': to_str(self.min_value),
                'MaxValue': to_str(self.max_value)}


class ContinuousParameter(_ParameterRange):
    __name__ = 'Continuous'

    def __init__(self, min_value, max_value):
        super(ContinuousParameter, self).__init__(min_value, max_value)


class CategoricalParameter(_ParameterRange):
    __name__ = 'Categorical'

    def __init__(self, values):
        if isinstance(values, list):
            self.values = [to_str(v) for v in values]
        else:
            self.values = [to_str(values)]

    def as_tuning_range(self, name):
        return {'Name': name,
                'Values': self.values}

    def as_json_range(self, name):
        return {'Name': name, 'Values': [json.dumps(v) for v in self.values]}


class IntegerParameter(_ParameterRange):
    __name__ = 'Integer'

    def __init__(self, min_value, max_value):
        super(IntegerParameter, self).__init__(min_value, max_value)


class HyperparameterTuner(object):
    TUNING_JOB_NAME_MAX_LENGTH = 32

    SAGEMAKER_ESTIMATOR_MODULE = 'sagemaker_estimator_module'
    SAGEMAKER_ESTIMATOR_CLASS_NAME = 'sagemaker_estimator_class_name'

    DEFAULT_ESTIMATOR_MODULE = 'sagemaker.estimator'
    DEFAULT_ESTIMATOR_CLS_NAME = 'Estimator'

    def __init__(self, estimator, objective_metric_name, hyperparameter_ranges, metric_definitions=None,
                 strategy='Bayesian', objective_type='Maximize', max_jobs=1, max_parallel_jobs=1,
                 tags=None, base_tuning_job_name=None):
        self._hyperparameter_ranges = hyperparameter_ranges
        if self._hyperparameter_ranges is None or len(self._hyperparameter_ranges) == 0:
            raise ValueError('Need to specify hyperparameter ranges')

        self.estimator = estimator
        self.objective_metric_name = objective_metric_name
        self.metric_definitions = metric_definitions
        self._validate_parameter_ranges()

        self.strategy = strategy
        self.objective_type = objective_type
        self.max_jobs = max_jobs
        self.max_parallel_jobs = max_parallel_jobs

        self.tags = tags
        self.base_tuning_job_name = base_tuning_job_name
        self._current_job_name = None
        self.latest_tuning_job = None

    def _prepare_for_training(self, job_name=None):
        if job_name is not None:
            self._current_job_name = job_name
        else:
            base_name = self.base_tuning_job_name or base_name_from_image(self.estimator.train_image())
            self._current_job_name = name_from_base(base_name, max_length=self.TUNING_JOB_NAME_MAX_LENGTH, short=True)

        self.static_hyperparameters = {to_str(k): to_str(v) for (k, v) in self.estimator.hyperparameters().items()}
        for hyperparameter_name in self._hyperparameter_ranges.keys():
            self.static_hyperparameters.pop(hyperparameter_name, None)

        # For attach() to know what estimator to use for non-1P algorithms
        # (1P algorithms don't accept extra hyperparameters)
        if not isinstance(self.estimator, AmazonAlgorithmEstimatorBase):
            self.static_hyperparameters[self.SAGEMAKER_ESTIMATOR_CLASS_NAME] = json.dumps(
                self.estimator.__class__.__name__)
            self.static_hyperparameters[self.SAGEMAKER_ESTIMATOR_MODULE] = json.dumps(self.estimator.__module__)

    def fit(self, inputs, job_name=None, **kwargs):
        """Start a hyperparameter tuning job.

        Args:
            inputs (str): Parameters used when called :meth:`~sagemaker.estimator.EstimatorBase.fit`.
            job_name (str): Tuning job name. If not specified, the tuner generates a default job name,
                based on the training image name and current timestamp.
            **kwargs: Other arguments
        """
        if isinstance(inputs, list) or isinstance(inputs, RecordSet):
            self.estimator._prepare_for_training(inputs, **kwargs)
        else:
            self.estimator._prepare_for_training(job_name)

        self._prepare_for_training(job_name=job_name)
        self.latest_tuning_job = _TuningJob.start_new(self, inputs)

    @classmethod
    def attach(cls, tuning_job_name, sagemaker_session=None, job_details=None, estimator_cls=None):
        sagemaker_session = sagemaker_session or Session()

        if job_details is None:
            job_details = sagemaker_session.sagemaker_client \
                .describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

        estimator_cls = cls._prepare_estimator_cls(estimator_cls, job_details['TrainingJobDefinition'])
        estimator = cls._prepare_estimator_from_job_description(estimator_cls, job_details['TrainingJobDefinition'],
                                                                sagemaker_session)
        init_params = cls._prepare_init_params_from_job_description(job_details)

        tuner = cls(estimator=estimator, **init_params)
        tuner.latest_tuning_job = _TuningJob(sagemaker_session=sagemaker_session, tuning_job_name=tuning_job_name)

        return tuner

    def deploy(self, initial_instance_count, instance_type, endpoint_name=None, **kwargs):
        """Deploy the best trained or user specified model to an Amazon SageMaker endpoint and return a
        ``sagemaker.RealTimePredictor``
        object.

                More information:
                http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html

                Args:
                    initial_instance_count (int): Minimum number of EC2 instances to deploy to an endpoint for
                    prediction.
                    instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
                        for example, 'ml.c4.xlarge'.
                    endpoint_name (str): Name to use for creating an Amazon SageMaker endpoint. If not specified,
                    the name of the training job is used.
                    **kwargs: Passed to invocation of ``create_model()``. Implementations may customize
                        ``create_model()`` to accept ``**kwargs`` to customize model creation during deploy.
                        For more, see the implementation docs.

                Returns:
                    sagemaker.predictor.RealTimePredictor: A predictor that provides a ``predict()`` method,
                        which can be used to send requests to the Amazon SageMaker endpoint and obtain inferences.
                """
        endpoint_name = endpoint_name or self.best_training_job()
        best_estimator = self.estimator.attach(self.best_training_job(),
                                               sagemaker_session=self.estimator.sagemaker_session)
        return best_estimator.deploy(initial_instance_count, instance_type, endpoint_name=endpoint_name, **kwargs)

    def stop_tuning_job(self):
        """Stop latest running tuning job.
        """
        self._ensure_last_tuning_job()
        self.latest_tuning_job.stop()

    def wait(self):
        """Wait for latest tuning job to finish.
        """
        self._ensure_last_tuning_job()
        self.latest_tuning_job.wait()

    def best_training_job(self):
        """Return name of the best training job for the latest tuning job.
        """
        self._ensure_last_tuning_job()

        tuning_job_describe_result = \
            self.estimator.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=self.latest_tuning_job.name)

        try:
            return tuning_job_describe_result['BestTrainingJob']['TrainingJobName']
        except KeyError:
            raise Exception('Best training job not available for tuning job: {}'.format(self.latest_tuning_job.name))

    def delete_endpoint(self, endpoint_name=None):
        """Delete an Amazon SageMaker endpoint.

        If an endpoint name is not specified, this defaults to looking for an endpoint that
        shares a name with the best training job for deletion.

        Args:
            endpoint_name (str): Name of the endpoint to delete
        """
        endpoint_name = endpoint_name or self.best_training_job()
        self.sagemaker_session.delete_endpoint(endpoint_name)

    def _ensure_last_tuning_job(self):
        if self.latest_tuning_job is None:
            raise ValueError('No tuning job available')

    @classmethod
    def _prepare_estimator_cls(cls, estimator_cls, training_details):
        # Check for customer-specified estimator first
        if estimator_cls is not None:
            module, cls_name = estimator_cls.rsplit('.', 1)
            return getattr(importlib.import_module(module), cls_name)

        # Then check for estimator class in hyperparameters
        hyperparameters = training_details['StaticHyperParameters']
        if cls.SAGEMAKER_ESTIMATOR_CLASS_NAME in hyperparameters and cls.SAGEMAKER_ESTIMATOR_MODULE in hyperparameters:
            module = hyperparameters.get(cls.SAGEMAKER_ESTIMATOR_MODULE)
            cls_name = hyperparameters.get(cls.SAGEMAKER_ESTIMATOR_CLASS_NAME)
            return getattr(importlib.import_module(json.loads(module)), json.loads(cls_name))

        # Then try to derive the estimator from the image name for 1P algorithms
        image_name = training_details['AlgorithmSpecification']['TrainingImage']
        algorithm = image_name[image_name.find('/') + 1:image_name.find(':')]
        if algorithm in AMAZON_ESTIMATOR_CLS_NAMES:
            cls_name = AMAZON_ESTIMATOR_CLS_NAMES[algorithm]
            return getattr(importlib.import_module(AMAZON_ESTIMATOR_MODULE), cls_name)

        # Default to the BYO estimator
        return getattr(importlib.import_module(cls.DEFAULT_ESTIMATOR_MODULE), cls.DEFAULT_ESTIMATOR_CLS_NAME)

    @classmethod
    def _prepare_estimator_from_job_description(cls, estimator_cls, training_details, sagemaker_session):
        # Swap name for static hyperparameters to what an estimator would expect
        training_details['HyperParameters'] = training_details['StaticHyperParameters']
        del training_details['StaticHyperParameters']

        # Remove hyperparameter reserved by SageMaker for tuning jobs
        del training_details['HyperParameters']['_tuning_objective_metric']

        # Add items expected by the estimator (but aren't needed otherwise)
        training_details['TrainingJobName'] = ''
        if 'KmsKeyId' not in training_details['OutputDataConfig']:
            training_details['OutputDataConfig']['KmsKeyId'] = ''

        estimator_init_params = estimator_cls._prepare_init_params_from_job_description(training_details)
        return estimator_cls(sagemaker_session=sagemaker_session, **estimator_init_params)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        tuning_config = job_details['HyperParameterTuningJobConfig']
        return {
            'metric_definitions': job_details['TrainingJobDefinition']['AlgorithmSpecification']['MetricDefinitions'],
            'objective_metric_name': tuning_config['HyperParameterTuningJobObjective']['MetricName'],
            'objective_type': tuning_config['HyperParameterTuningJobObjective']['Type'],
            'hyperparameter_ranges': cls._prepare_parameter_ranges(tuning_config['ParameterRanges']),
            'strategy': tuning_config['Strategy'],
            'max_jobs': tuning_config['ResourceLimits']['MaxNumberOfTrainingJobs'],
            'max_parallel_jobs': tuning_config['ResourceLimits']['MaxParallelTrainingJobs'],
        }

    @classmethod
    def _prepare_parameter_ranges(cls, parameter_ranges):
        ranges = {}

        for parameter in parameter_ranges['CategoricalParameterRanges']:
            ranges[parameter['Name']] = CategoricalParameter(parameter['Values'])

        for parameter in parameter_ranges['ContinuousParameterRanges']:
            ranges[parameter['Name']] = ContinuousParameter(float(parameter['MinValue']), float(parameter['MaxValue']))

        for parameter in parameter_ranges['IntegerParameterRanges']:
            ranges[parameter['Name']] = IntegerParameter(int(parameter['MinValue']), int(parameter['MaxValue']))

        return ranges

    def hyperparameter_ranges(self):
        """Return collections of ``ParameterRanges``

        Returns:
            dict: ParameterRanges suitable for a hyperparameter tuning job.
        """
        hyperparameter_ranges = dict()
        for range_type in _ParameterRange.__all_types__:
            parameter_ranges = []
            for parameter_name, parameter in self._hyperparameter_ranges.items():
                if parameter is not None and parameter.__name__ == range_type:
                    # Categorical parameters needed to be serialized as JSON for our framework containers
                    if isinstance(parameter, CategoricalParameter) and isinstance(self.estimator, Framework):
                        tuning_range = parameter.as_json_range(parameter_name)
                    else:
                        tuning_range = parameter.as_tuning_range(parameter_name)
                    parameter_ranges.append(tuning_range)
            hyperparameter_ranges[range_type + 'ParameterRanges'] = parameter_ranges
        return hyperparameter_ranges

    @property
    def sagemaker_session(self):
        """The tuner shares the sagemaker_session object with its estimator.
        Convenience method.
        """
        return self.estimator.sagemaker_session

    def analytics(self):
        """An instance of HyperparameterTuningJobAnalytics for this latest tuning job of this tuner.
        Analytics olbject gives you access to tuning results summarized into a pandas dataframe.
        """
        return HyperparameterTuningJobAnalytics(self.latest_tuning_job.name, self.sagemaker_session)

    def _validate_parameter_ranges(self):
        for kls in inspect.getmro(self.estimator.__class__)[::-1]:
            for attribute, value in kls.__dict__.items():
                if isinstance(value, hp):
                    try:
                        # The hyperparam names may not be the same as the class attribute that holds them,
                        # for instance: local_lloyd_init_method is called local_init_method. We need to map these
                        # and pass the correct name to the constructor.
                        parameter_range = self._hyperparameter_ranges[value.name]

                        if isinstance(parameter_range, _ParameterRange):
                            for parameter_range_attribute, parameter_range_value in parameter_range.__dict__.items():
                                # Categorical ranges
                                if isinstance(parameter_range_value, list):
                                    for categorical_value in parameter_range_value:
                                        value.validate(categorical_value)
                                # Continuous, Integer ranges
                                else:
                                    value.validate(parameter_range_value)
                    except KeyError:
                        pass


class _TuningJob(_Job):
    def __init__(self, sagemaker_session, tuning_job_name):
        super(_TuningJob, self).__init__(sagemaker_session, tuning_job_name)

    @classmethod
    def start_new(cls, tuner, inputs):
        """Create a new Amazon SageMaker hyperparameter tuning job from the HyperparameterTuner.

        Args:
            tuner (sagemaker.tuner.HyperparameterTuner): HyperparameterTuner object created by the user.
            inputs (str): Parameters used when called :meth:`~sagemaker.estimator.EstimatorBase.fit`.

        Returns:
            sagemaker.tuner._TuningJob: Constructed object that captures all information about the started job.
        """
        config = _Job._load_config(inputs, tuner.estimator)

        tuner.estimator.sagemaker_session.tune(job_name=tuner._current_job_name, strategy=tuner.strategy,
                                               objective_type=tuner.objective_type,
                                               objective_metric_name=tuner.objective_metric_name,
                                               max_jobs=tuner.max_jobs, max_parallel_jobs=tuner.max_parallel_jobs,
                                               parameter_ranges=tuner.hyperparameter_ranges(),
                                               static_hyperparameters=tuner.static_hyperparameters,
                                               image=tuner.estimator.train_image(),
                                               input_mode=tuner.estimator.input_mode,
                                               metric_definitions=tuner.metric_definitions,
                                               role=(config['role']), input_config=(config['input_config']),
                                               output_config=(config['output_config']),
                                               resource_config=(config['resource_config']),
                                               stop_condition=(config['stop_condition']), tags=tuner.tags)

        return cls(tuner.sagemaker_session, tuner._current_job_name)

    def stop(self):
        self.sagemaker_session.stop_tuning_job(name=self.name)

    def wait(self):
        self.sagemaker_session.wait_for_tuning_job(self.name)
