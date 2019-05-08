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
import logging
from enum import Enum

import sagemaker
from sagemaker.amazon.amazon_estimator import RecordSet
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from sagemaker.estimator import Framework
from sagemaker.job import _Job
from sagemaker.parameter import (CategoricalParameter, ContinuousParameter,
                                 IntegerParameter, ParameterRange)
from sagemaker.session import Session
from sagemaker.session import s3_input
from sagemaker.utils import base_name_from_image, name_from_base, to_str

AMAZON_ESTIMATOR_MODULE = 'sagemaker'
AMAZON_ESTIMATOR_CLS_NAMES = {
    'factorization-machines': 'FactorizationMachines',
    'kmeans': 'KMeans',
    'lda': 'LDA',
    'linear-learner': 'LinearLearner',
    'ntm': 'NTM',
    'randomcutforest': 'RandomCutForest',
    'knn': 'KNN',
    'object2vec': 'Object2Vec',
}
HYPERPARAMETER_TUNING_JOB_NAME = 'HyperParameterTuningJobName'
PARENT_HYPERPARAMETER_TUNING_JOBS = 'ParentHyperParameterTuningJobs'
WARM_START_TYPE = 'WarmStartType'


class WarmStartTypes(Enum):
    """Warm Start Configuration type. There can be two types of warm start jobs:
        * IdenticalDataAndAlgorithm: Type of warm start that allows users to reuse training results from existing
        tuning jobs that have the same algorithm code and datasets.
        * TransferLearning: Type of warm start that allows users to reuse training results from existing tuning jobs
        that have similar algorithm code and datasets.
    """
    IDENTICAL_DATA_AND_ALGORITHM = "IdenticalDataAndAlgorithm"
    TRANSFER_LEARNING = "TransferLearning"


class WarmStartConfig(object):
    """Warm Start Configuration which defines the nature of the warm start ``HyperparameterTuner``, with type and
    parents for warm start.

    Examples:
        >>> warm_start_config = WarmStartConfig(type=WarmStartTypes.TransferLearning, parents={"p1","p2"})
        >>> warm_start_config.type
        "TransferLearning"
        >>> warm_start_config.parents
        {"p1","p2"}
    """

    def __init__(self, warm_start_type, parents):
        """Initializes the ``WarmStartConfig`` with the provided ``WarmStartTypes`` and parents.

        Args:
            warm_start_type (sagemaker.tuner.WarmStartTypes): This should be one of the supported warm start types
            in WarmStartType
            parents (set{str}): Set of parent tuning jobs which will be used to warm start the new tuning job.
        """

        if warm_start_type not in WarmStartTypes:
            raise ValueError(
                "Invalid type: {}, valid warm start types are: [{}]".format(warm_start_type,
                                                                            [t for t in WarmStartTypes]))

        if not parents:
            raise ValueError("Invalid parents: {}, parents should not be None/empty".format(parents))

        self.type = warm_start_type
        self.parents = set(parents)

    @classmethod
    def from_job_desc(cls, warm_start_config):
        """Creates an instance of ``WarmStartConfig`` class, from warm start configuration response from
        DescribeTrainingJob.

        Args:
            warm_start_config (dict): The expected format of the ``warm_start_config`` contains two first-class
            fields:
                * "type": Type of warm start tuner, currently two supported types - "IdenticalDataAndAlgorithm" and
                "TransferLearning".
                * "parents": List of tuning job names from which the warm start should be done.

        Returns:
            sagemaker.tuner.WarmStartConfig: De-serialized instance of WarmStartConfig containing the type and parents
            provided as part of ``warm_start_config``.

        Examples:
            >>> warm_start_config = WarmStartConfig.from_job_desc(warm_start_config={
            >>>    "WarmStartType":"TransferLearning",
            >>>    "ParentHyperParameterTuningJobs": [
            >>>        {'HyperParameterTuningJobName': "p1"},
            >>>        {'HyperParameterTuningJobName': "p2"},
            >>>    ]
            >>>})
            >>> warm_start_config.type
            "TransferLearning"
            >>> warm_start_config.parents
            ["p1","p2"]
        """
        if not warm_start_config or \
                WARM_START_TYPE not in warm_start_config or \
                PARENT_HYPERPARAMETER_TUNING_JOBS not in warm_start_config:
            return None

        parents = []
        for parent in warm_start_config[PARENT_HYPERPARAMETER_TUNING_JOBS]:
            parents.append(parent[HYPERPARAMETER_TUNING_JOB_NAME])

        return cls(warm_start_type=WarmStartTypes(warm_start_config[WARM_START_TYPE]),
                   parents=parents)

    def to_input_req(self):
        """Converts the ``self`` instance to the desired input request format.

        Returns:
            dict: Containing the "WarmStartType" and "ParentHyperParameterTuningJobs" as the first class fields.

        Examples:
            >>> warm_start_config = WarmStartConfig(warm_start_type=WarmStartTypes.TransferLearning,parents=["p1,p2"])
            >>> warm_start_config.to_input_req()
            {
                "WarmStartType":"TransferLearning",
                "ParentHyperParameterTuningJobs": [
                    {'HyperParameterTuningJobName': "p1"},
                    {'HyperParameterTuningJobName': "p2"},
                ]
            }
        """
        return {
            WARM_START_TYPE: self.type.value,
            PARENT_HYPERPARAMETER_TUNING_JOBS: [{HYPERPARAMETER_TUNING_JOB_NAME: parent} for parent in self.parents]
        }


class HyperparameterTuner(object):
    """A class for creating and interacting with Amazon SageMaker hyperparameter tuning jobs, as well as
    deploying the resulting model(s).
    """
    TUNING_JOB_NAME_MAX_LENGTH = 32

    SAGEMAKER_ESTIMATOR_MODULE = 'sagemaker_estimator_module'
    SAGEMAKER_ESTIMATOR_CLASS_NAME = 'sagemaker_estimator_class_name'

    DEFAULT_ESTIMATOR_MODULE = 'sagemaker.estimator'
    DEFAULT_ESTIMATOR_CLS_NAME = 'Estimator'

    def __init__(self, estimator, objective_metric_name, hyperparameter_ranges, metric_definitions=None,
                 strategy='Bayesian', objective_type='Maximize', max_jobs=1, max_parallel_jobs=1,
                 tags=None, base_tuning_job_name=None, warm_start_config=None, early_stopping_type='Off'):
        """Initialize a ``HyperparameterTuner``. It takes an estimator to obtain configuration information
        for training jobs that are created as the result of a hyperparameter tuning job.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): An estimator object that has been initialized with
                the desired configuration. There does not need to be a training job associated with this instance.
            objective_metric_name (str): Name of the metric for evaluating training jobs.
            hyperparameter_ranges (dict[str, sagemaker.parameter.ParameterRange]): Dictionary of parameter ranges.
                These parameter ranges can be one of three types: Continuous, Integer, or Categorical. The keys of the
                dictionary are the names of the hyperparameter, and the values are the appropriate parameter range class
                to represent the range.
            metric_definitions (list[dict]): A list of dictionaries that defines the metric(s) used to evaluate the
                training jobs (default: None). Each dictionary contains two keys: 'Name' for the name of the metric, and
                'Regex' for the regular expression used to extract the metric from the logs. This should be defined only
                for hyperparameter tuning jobs that don't use an Amazon algorithm.
            strategy (str): Strategy to be used for hyperparameter estimations (default: 'Bayesian').
            objective_type (str): The type of the objective metric for evaluating training jobs. This value can be
                either 'Minimize' or 'Maximize' (default: 'Maximize').
            max_jobs (int): Maximum total number of training jobs to start for the hyperparameter tuning job
                (default: 1).
            max_parallel_jobs (int): Maximum number of parallel training jobs to start (default: 1).
            tags (list[dict]): List of tags for labeling the tuning job (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            base_tuning_job_name (str): Prefix for the hyperparameter tuning job name when the
                :meth:`~sagemaker.tuner.HyperparameterTuner.fit` method launches. If not specified,
                a default job name is generated, based on the training image name and current timestamp.
            warm_start_config (sagemaker.tuner.WarmStartConfig): A ``WarmStartConfig`` object that has been initialized
                with the configuration defining the nature of warm start tuning job.
            early_stopping_type (str): Specifies whether early stopping is enabled for the job.
                Can be either 'Auto' or 'Off' (default: 'Off'). If set to 'Off', early stopping will not be attempted.
                If set to 'Auto', early stopping of some training jobs may happen, but is not guaranteed to.
        """
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
        self.warm_start_config = warm_start_config
        self.early_stopping_type = early_stopping_type

    def _prepare_for_training(self, job_name=None, include_cls_metadata=False):
        if job_name is not None:
            self._current_job_name = job_name
        else:
            base_name = self.base_tuning_job_name or base_name_from_image(self.estimator.train_image())
            self._current_job_name = name_from_base(base_name, max_length=self.TUNING_JOB_NAME_MAX_LENGTH, short=True)

        self.static_hyperparameters = {to_str(k): to_str(v) for (k, v) in self.estimator.hyperparameters().items()}
        for hyperparameter_name in self._hyperparameter_ranges.keys():
            self.static_hyperparameters.pop(hyperparameter_name, None)

        # For attach() to know what estimator to use for frameworks
        # (other algorithms may not accept extra hyperparameters)
        if include_cls_metadata or isinstance(self.estimator, Framework):
            self.static_hyperparameters[self.SAGEMAKER_ESTIMATOR_CLASS_NAME] = json.dumps(
                self.estimator.__class__.__name__)
            self.static_hyperparameters[self.SAGEMAKER_ESTIMATOR_MODULE] = json.dumps(self.estimator.__module__)

    def fit(self, inputs=None, job_name=None, include_cls_metadata=False, **kwargs):
        """Start a hyperparameter tuning job.

        Args:
            inputs: Information about the training data. Please refer to the ``fit()`` method of
                the associated estimator, as this can take any of the following forms:

                * (str) - The S3 location where training data is saved.
                * (dict[str, str] or dict[str, sagemaker.session.s3_input]) - If using multiple channels for
                    training data, you can specify a dict mapping channel names
                    to strings or :func:`~sagemaker.session.s3_input` objects.
                * (sagemaker.session.s3_input) - Channel configuration for S3 data sources that can provide
                    additional information about the training dataset. See :func:`sagemaker.session.s3_input`
                    for full details.
                * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                    Amazon :class:~`Record` objects serialized and stored in S3.
                    For use with an estimator for an Amazon algorithm.
                * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects, where each instance is
                    a different channel of training data.

            job_name (str): Tuning job name. If not specified, the tuner generates a default job name,
                based on the training image name and current timestamp.
            include_cls_metadata (bool): Whether or not the hyperparameter tuning job should include
                information about the estimator class (default: False). This information is passed
                as a hyperparameter, so if the algorithm you are using cannot handle
                unknown hyperparameters (e.g. an Amazon SageMaker built-in algorithm that
                does not have a custom estimator in the Python SDK), then set
                ``include_cls_metadata`` to ``False``.
            **kwargs: Other arguments needed for training. Please refer to the ``fit()`` method of the associated
                estimator to see what other arguments are needed.
        """
        if isinstance(inputs, list) or isinstance(inputs, RecordSet):
            self.estimator._prepare_for_training(inputs, **kwargs)
        else:
            self.estimator._prepare_for_training(job_name)

        self._prepare_for_training(job_name=job_name, include_cls_metadata=include_cls_metadata)
        self.latest_tuning_job = _TuningJob.start_new(self, inputs)

    @classmethod
    def attach(cls, tuning_job_name, sagemaker_session=None, job_details=None, estimator_cls=None):
        """Attach to an existing hyperparameter tuning job.

        Create a HyperparameterTuner bound to an existing hyperparameter tuning job. After attaching, if there exists a
        best training job (or any other completed training job), that can be deployed to create
        an Amazon SageMaker Endpoint and return a ``Predictor``.

        Args:
            tuning_job_name (str): The name of the hyperparameter tuning job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, one is created
                using the default AWS configuration chain.
            job_details (dict): The response to a ``DescribeHyperParameterTuningJob`` call. If not specified,
                the ``HyperparameterTuner`` will perform one such call with the provided hyperparameter tuning job name.
            estimator_cls (str): The estimator class name associated with the training jobs,
                e.g. 'sagemaker.estimator.Estimator'. If not specified, the ``HyperparameterTuner`` will try to derive
                the correct estimator class from training job metadata, defaulting to
                :class:~`sagemaker.estimator.Estimator` if it is unable to determine a more specific class.

        Examples:
            >>> my_tuner.fit()
            >>> job_name = my_tuner.latest_tuning_job.name
            Later on:
            >>> attached_tuner = HyperparameterTuner.attach(job_name)
            >>> attached_tuner.deploy()

        Returns:
            sagemaker.tuner.HyperparameterTuner: A ``HyperparameterTuner`` instance with the attached hyperparameter
                tuning job.
        """
        sagemaker_session = sagemaker_session or Session()

        if job_details is None:
            job_details = sagemaker_session.sagemaker_client \
                .describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

        estimator_cls = cls._prepare_estimator_cls(estimator_cls, job_details['TrainingJobDefinition'])
        estimator = cls._prepare_estimator_from_job_description(estimator_cls, job_details['TrainingJobDefinition'],
                                                                sagemaker_session)
        init_params = cls._prepare_init_params_from_job_description(job_details)

        tuner = cls(estimator=estimator, **init_params)
        tuner.latest_tuning_job = _TuningJob(sagemaker_session=sagemaker_session, job_name=tuning_job_name)

        return tuner

    def deploy(self, initial_instance_count, instance_type, accelerator_type=None, endpoint_name=None, **kwargs):
        """Deploy the best trained or user specified model to an Amazon SageMaker endpoint and return a
        ``sagemaker.RealTimePredictor`` object.

        For more information: http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html

        Args:
            initial_instance_count (int): Minimum number of EC2 instances to deploy to an endpoint for prediction.
            instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
                for example, 'ml.c4.xlarge'.
            accelerator_type (str): Type of Elastic Inference accelerator to attach to an endpoint for model loading
                and inference, for example, 'ml.eia1.medium'. If not specified, no Elastic Inference accelerator
                will be attached to the endpoint.
                For more information: https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            endpoint_name (str): Name to use for creating an Amazon SageMaker endpoint. If not specified,
                the name of the training job is used.
            **kwargs: Other arguments needed for deployment. Please refer to the ``create_model()`` method of
                the associated estimator to see what other arguments are needed.

        Returns:
            sagemaker.predictor.RealTimePredictor: A predictor that provides a ``predict()`` method,
                which can be used to send requests to the Amazon SageMaker endpoint and obtain inferences.
        """
        endpoint_name = endpoint_name or self.best_training_job()
        best_estimator = self.estimator.attach(self.best_training_job(),
                                               sagemaker_session=self.estimator.sagemaker_session)
        return best_estimator.deploy(initial_instance_count, instance_type,
                                     accelerator_type=accelerator_type,
                                     endpoint_name=endpoint_name, **kwargs)

    def stop_tuning_job(self):
        """Stop latest running hyperparameter tuning job.
        """
        self._ensure_last_tuning_job()
        self.latest_tuning_job.stop()

    def wait(self):
        """Wait for latest hyperparameter tuning job to finish.
        """
        self._ensure_last_tuning_job()
        self.latest_tuning_job.wait()

    def best_training_job(self):
        """Return name of the best training job for the latest hyperparameter tuning job.

        Raises:
            Exception: If there is no best training job available for the hyperparameter tuning job.
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
            'warm_start_config': WarmStartConfig.from_job_desc(job_details.get('WarmStartConfig', None)),
            'early_stopping_type': tuning_config['TrainingJobEarlyStoppingType']
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
        """Return the hyperparameter ranges in a dictionary to be used as part of a request for creating a
        hyperparameter tuning job.
        """
        hyperparameter_ranges = dict()
        for range_type in ParameterRange.__all_types__:
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
        """Convenience method for accessing the :class:`~sagemaker.session.Session` object associated
        with the estimator for the ``HyperparameterTuner``.
        """
        return self.estimator.sagemaker_session

    def analytics(self):
        """An instance of HyperparameterTuningJobAnalytics for this latest tuning job of this tuner.
        Analytics olbject gives you access to tuning results summarized into a pandas dataframe.
        """
        return HyperparameterTuningJobAnalytics(self.latest_tuning_job.name, self.sagemaker_session)

    def _validate_parameter_ranges(self):
        for kls in inspect.getmro(self.estimator.__class__)[::-1]:
            for _, value in kls.__dict__.items():
                if isinstance(value, hp):
                    try:
                        # The hyperparam names may not be the same as the class attribute that holds them,
                        # for instance: local_lloyd_init_method is called local_init_method. We need to map these
                        # and pass the correct name to the constructor.
                        parameter_range = self._hyperparameter_ranges[value.name]

                        if isinstance(parameter_range, ParameterRange):
                            self._validate_parameter_range(value, parameter_range)
                    except KeyError:
                        pass

    def _validate_parameter_range(self, value_hp, parameter_range):
        for parameter_range_key, parameter_range_value in parameter_range.__dict__.items():
            if parameter_range_key == 'scaling_type':
                continue

            # Categorical ranges
            if isinstance(parameter_range_value, list):
                for categorical_value in parameter_range_value:
                    value_hp.validate(categorical_value)
            # Continuous, Integer ranges
            else:
                value_hp.validate(parameter_range_value)

    def transfer_learning_tuner(self, additional_parents=None, estimator=None):
        """Creates a new ``HyperparameterTuner`` by copying the request fields from the provided parent to the new
        instance of ``HyperparameterTuner``. Followed by addition of warm start configuration with the type as
        "TransferLearning" and parents as the union of provided list of ``additional_parents`` and the ``self``.
        Also, training image in the new tuner's estimator is updated with the provided ``training_image``.

        Args:
            additional_parents (set{str}): Set of additional parents along with the self to be used in warm starting
            the transfer learning tuner.
            estimator (sagemaker.estimator.EstimatorBase): An estimator object that has been initialized with
                the desired configuration. There does not need to be a training job associated with this instance.

        Returns:
            sagemaker.tuner.HyperparameterTuner: ``HyperparameterTuner`` instance which can be used to launch transfer
            learning tuning job.

        Examples:
            >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
            >>> transfer_learning_tuner = parent_tuner.transfer_learning_tuner(additional_parents={"parent-job-2"})
            Later On:
            >>> transfer_learning_tuner.fit(inputs={})
        """

        return self._create_warm_start_tuner(additional_parents=additional_parents,
                                             warm_start_type=WarmStartTypes.TRANSFER_LEARNING,
                                             estimator=estimator)

    def identical_dataset_and_algorithm_tuner(self, additional_parents=None):
        """Creates a new ``HyperparameterTuner`` by copying the request fields from the provided parent to the new
        instance of ``HyperparameterTuner``. Followed by addition of warm start configuration with the type as
        "IdenticalDataAndAlgorithm" and parents as the union of provided list of ``additional_parents`` and the ``self``

        Args:
            additional_parents (set{str}): Set of additional parents along with the self to be used in warm starting
            the identical dataset and algorithm tuner.

        Returns:
            sagemaker.tuner.HyperparameterTuner: HyperparameterTuner instance which can be used to launch identical
            dataset and algorithm tuning job.

        Examples:
            >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
            >>> identical_dataset_algo_tuner = parent_tuner.identical_dataset_and_algorithm_tuner(
            >>>                                                             additional_parents={"parent-job-2"})
            Later On:
            >>> identical_dataset_algo_tuner.fit(inputs={})
        """

        return self._create_warm_start_tuner(additional_parents=additional_parents,
                                             warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM)

    def _create_warm_start_tuner(self, additional_parents, warm_start_type, estimator=None):
        """Creates a new ``HyperparameterTuner`` with ``WarmStartConfig``, where type will be equal to
        ``warm_start_type`` and``parents`` would be equal to union of ``additional_parents`` and self.

        Args:
            additional_parents (set{str}): Additional parents along with self, to be used for warm starting.
            warm_start_type (sagemaker.tuner.WarmStartTypes): Type of warm start job.

        Returns:
            sagemaker.tuner.HyperparameterTuner: Instance with the request fields copied from self along with the
            warm start configuration
        """
        all_parents = {self.latest_tuning_job.name}
        if additional_parents:
            all_parents = all_parents.union(additional_parents)

        return HyperparameterTuner(estimator=estimator if estimator else self.estimator,
                                   objective_metric_name=self.objective_metric_name,
                                   hyperparameter_ranges=self._hyperparameter_ranges,
                                   objective_type=self.objective_type,
                                   max_jobs=self.max_jobs,
                                   max_parallel_jobs=self.max_parallel_jobs,
                                   warm_start_config=WarmStartConfig(warm_start_type=warm_start_type,
                                                                     parents=all_parents))


class _TuningJob(_Job):
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

        warm_start_config_req = None
        if tuner.warm_start_config:
            warm_start_config_req = tuner.warm_start_config.to_input_req()

        tuner_args = config.copy()

        tuner_args['job_name'] = tuner._current_job_name
        tuner_args['strategy'] = tuner.strategy
        tuner_args['objective_type'] = tuner.objective_type
        tuner_args['objective_metric_name'] = tuner.objective_metric_name
        tuner_args['max_jobs'] = tuner.max_jobs
        tuner_args['max_parallel_jobs'] = tuner.max_parallel_jobs
        tuner_args['parameter_ranges'] = tuner.hyperparameter_ranges()
        tuner_args['static_hyperparameters'] = tuner.static_hyperparameters
        tuner_args['input_mode'] = tuner.estimator.input_mode
        tuner_args['metric_definitions'] = tuner.metric_definitions
        tuner_args['tags'] = tuner.tags
        tuner_args['warm_start_config'] = warm_start_config_req
        tuner_args['early_stopping_type'] = tuner.early_stopping_type

        if isinstance(inputs, s3_input):
            if 'InputMode' in inputs.config:
                logging.debug('Selecting s3_input\'s input_mode ({}) for TrainingInputMode.'
                              .format(inputs.config['InputMode']))
                tuner_args['input_mode'] = inputs.config['InputMode']

        if isinstance(tuner.estimator, sagemaker.algorithm.AlgorithmEstimator):
            tuner_args['algorithm_arn'] = tuner.estimator.algorithm_arn
        else:
            tuner_args['image'] = tuner.estimator.train_image()

        tuner_args['enable_network_isolation'] = tuner.estimator.enable_network_isolation()
        tuner_args['encrypt_inter_container_traffic'] = \
            tuner.estimator.encrypt_inter_container_traffic

        tuner.estimator.sagemaker_session.tune(**tuner_args)

        return cls(tuner.sagemaker_session, tuner._current_job_name)

    def stop(self):
        self.sagemaker_session.stop_tuning_job(name=self.name)

    def wait(self):
        self.sagemaker_session.wait_for_tuning_job(self.name)


def create_identical_dataset_and_algorithm_tuner(parent, additional_parents=None, sagemaker_session=None):
    """Creates a new tuner by copying the request fields from the provided parent to the new instance of
        ``HyperparameterTuner`` followed by addition of warm start configuration with the type as
        "IdenticalDataAndAlgorithm" and ``parents`` as the union of provided list of ``additional_parents`` and the
        ``parent``.

    Args:
        parent (str): Primary parent tuning job's name from which the Tuner and Estimator configuration has to be copied
        additional_parents (set{str}): Set of additional parent tuning job's names along with the primary parent tuning
            job name to be used in warm starting the transfer learning tuner.
        sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

    Returns:
        sagemaker.tuner.HyperparameterTuner: a new ``HyperparameterTuner`` object for the warm-started
        hyperparameter tuning job
    """

    parent_tuner = HyperparameterTuner.attach(tuning_job_name=parent, sagemaker_session=sagemaker_session)
    return parent_tuner.identical_dataset_and_algorithm_tuner(additional_parents=additional_parents)


def create_transfer_learning_tuner(parent, additional_parents=None, estimator=None, sagemaker_session=None):
    """Creates a new ``HyperParameterTuner`` by copying the request fields from the provided parent to the new instance
        of ``HyperparameterTuner`` followed by addition of warm start configuration with the type as "TransferLearning"
        and ``parents`` as the union of provided list of ``additional_parents`` and the ``parent``.

    Args:
        parent (str): Primary parent tuning job's name from which the Tuner and Estimator configuration has to be copied
        additional_parents (set{str}): Set of additional parent tuning job's names along with the primary parent tuning
            job name to be used in warm starting the identical dataset and algorithm tuner.
        estimator (sagemaker.estimator.EstimatorBase): An estimator object that has been initialized with
                the desired configuration. There does not need to be a training job associated with this instance.
        sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, one is created
                using the default AWS configuration chain.

    Returns:
        sagemaker.tuner.HyperparameterTuner: New instance of warm started HyperparameterTuner
    """

    parent_tuner = HyperparameterTuner.attach(tuning_job_name=parent, sagemaker_session=sagemaker_session)
    return parent_tuner.transfer_learning_tuner(additional_parents=additional_parents,
                                                estimator=estimator)
