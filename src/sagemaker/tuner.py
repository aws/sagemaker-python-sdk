# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import importlib
import inspect
import json
import logging
from enum import Enum

import sagemaker
from sagemaker.amazon.amazon_estimator import (
    RecordSet,
    AmazonAlgorithmEstimatorBase,
    FileSystemRecordSet,
)
from sagemaker.amazon.hyperparameter import Hyperparameter as hp  # noqa
from sagemaker.analytics import HyperparameterTuningJobAnalytics
from sagemaker.estimator import Framework
from sagemaker.job import _Job
from sagemaker.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
    ParameterRange,
)
from sagemaker.session import Session
from sagemaker.session import s3_input
from sagemaker.utils import base_name_from_image, name_from_base, to_str

AMAZON_ESTIMATOR_MODULE = "sagemaker"
AMAZON_ESTIMATOR_CLS_NAMES = {
    "factorization-machines": "FactorizationMachines",
    "kmeans": "KMeans",
    "lda": "LDA",
    "linear-learner": "LinearLearner",
    "ntm": "NTM",
    "randomcutforest": "RandomCutForest",
    "knn": "KNN",
    "object2vec": "Object2Vec",
}
HYPERPARAMETER_TUNING_JOB_NAME = "HyperParameterTuningJobName"
PARENT_HYPERPARAMETER_TUNING_JOBS = "ParentHyperParameterTuningJobs"
WARM_START_TYPE = "WarmStartType"


class WarmStartTypes(Enum):
    """Warm Start Configuration type. There can be two types of warm start jobs:
    * IdenticalDataAndAlgorithm: Type of warm start that allows users to reuse
    training results from existing tuning jobs that have the same algorithm code
    and datasets. * TransferLearning: Type of warm start that allows users to
    reuse training results from existing tuning jobs that have similar algorithm
    code and datasets.
    """

    IDENTICAL_DATA_AND_ALGORITHM = "IdenticalDataAndAlgorithm"
    TRANSFER_LEARNING = "TransferLearning"


class WarmStartConfig(object):
    """Warm Start Configuration which defines the nature of the warm start
    ``HyperparameterTuner``, with type and parents for warm start.

    Examples:
        >>> warm_start_config = WarmStartConfig(
        >>>                         type=WarmStartTypes.TransferLearning, parents={"p1","p2"})
        >>> warm_start_config.type
        "TransferLearning"
        >>> warm_start_config.parents
        {"p1","p2"}
    """

    def __init__(self, warm_start_type, parents):
        """Initializes the ``WarmStartConfig`` with the provided
        ``WarmStartTypes`` and parents.

        Args:
            warm_start_type (sagemaker.tuner.WarmStartTypes): This should be one
                of the supported warm start types in WarmStartType
            parents (set{str}): Set of parent tuning jobs which will be used to
                warm start the new tuning job.
        """

        if warm_start_type not in WarmStartTypes:
            raise ValueError(
                "Invalid type: {}, valid warm start types are: [{}]".format(
                    warm_start_type, [t for t in WarmStartTypes]
                )
            )

        if not parents:
            raise ValueError(
                "Invalid parents: {}, parents should not be None/empty".format(parents)
            )

        self.type = warm_start_type
        self.parents = set(parents)

    @classmethod
    def from_job_desc(cls, warm_start_config):
        """Creates an instance of ``WarmStartConfig`` class, from warm start
        configuration response from DescribeTrainingJob.

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

        Args:
            warm_start_config (dict): The expected format of the
                ``warm_start_config`` contains two first-class

        Returns:
            sagemaker.tuner.WarmStartConfig: De-serialized instance of
            WarmStartConfig containing the type and parents provided as part of
            ``warm_start_config``.
        """
        if (
            not warm_start_config
            or WARM_START_TYPE not in warm_start_config
            or PARENT_HYPERPARAMETER_TUNING_JOBS not in warm_start_config
        ):
            return None

        parents = []
        for parent in warm_start_config[PARENT_HYPERPARAMETER_TUNING_JOBS]:
            parents.append(parent[HYPERPARAMETER_TUNING_JOB_NAME])

        return cls(
            warm_start_type=WarmStartTypes(warm_start_config[WARM_START_TYPE]), parents=parents
        )

    def to_input_req(self):
        """Converts the ``self`` instance to the desired input request format.

        Examples:
            >>> warm_start_config = WarmStartConfig
            (
                warm_start_type=WarmStartTypes.TransferLearning,parents=["p1,p2"]
            )
            >>> warm_start_config.to_input_req()
            {
                "WarmStartType":"TransferLearning",
                "ParentHyperParameterTuningJobs": [
                    {'HyperParameterTuningJobName': "p1"},
                    {'HyperParameterTuningJobName': "p2"},
                ]
            }

        Returns:
            dict: Containing the "WarmStartType" and
            "ParentHyperParameterTuningJobs" as the first class fields.
        """
        return {
            WARM_START_TYPE: self.type.value,
            PARENT_HYPERPARAMETER_TUNING_JOBS: [
                {HYPERPARAMETER_TUNING_JOB_NAME: parent} for parent in self.parents
            ],
        }


class HyperparameterTuner(object):
    """A class for creating and interacting with Amazon SageMaker hyperparameter
    tuning jobs, as well as deploying the resulting model(s).
    """

    TUNING_JOB_NAME_MAX_LENGTH = 32

    SAGEMAKER_ESTIMATOR_MODULE = "sagemaker_estimator_module"
    SAGEMAKER_ESTIMATOR_CLASS_NAME = "sagemaker_estimator_class_name"

    DEFAULT_ESTIMATOR_MODULE = "sagemaker.estimator"
    DEFAULT_ESTIMATOR_CLS_NAME = "Estimator"

    def __init__(
        self,
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions=None,
        strategy="Bayesian",
        objective_type="Maximize",
        max_jobs=1,
        max_parallel_jobs=1,
        tags=None,
        base_tuning_job_name=None,
        warm_start_config=None,
        early_stopping_type="Off",
        estimator_name=None,
    ):
        """Initialize a ``HyperparameterTuner``. It takes an estimator to obtain
        configuration information for training jobs that are created as the
        result of a hyperparameter tuning job.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): An estimator object
                that has been initialized with the desired configuration. There
                does not need to be a training job associated with this
                instance.
            objective_metric_name (str): Name of the metric for evaluating
                training jobs.
            hyperparameter_ranges (dict[str, sagemaker.parameter.ParameterRange]): Dictionary of
                parameter ranges. These parameter ranges can be one
                of three types: Continuous, Integer, or Categorical. The keys of
                the dictionary are the names of the hyperparameter, and the
                values are the appropriate parameter range class to represent
                the range.
            metric_definitions (list[dict]): A list of dictionaries that defines
                the metric(s) used to evaluate the training jobs (default:
                None). Each dictionary contains two keys: 'Name' for the name of
                the metric, and 'Regex' for the regular expression used to
                extract the metric from the logs. This should be defined only
                for hyperparameter tuning jobs that don't use an Amazon
                algorithm.
            strategy (str): Strategy to be used for hyperparameter estimations
                (default: 'Bayesian').
            objective_type (str): The type of the objective metric for
                evaluating training jobs. This value can be either 'Minimize' or
                'Maximize' (default: 'Maximize').
            max_jobs (int): Maximum total number of training jobs to start for
                the hyperparameter tuning job (default: 1).
            max_parallel_jobs (int): Maximum number of parallel training jobs to
                start (default: 1).
            tags (list[dict]): List of tags for labeling the tuning job
                (default: None). For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            base_tuning_job_name (str): Prefix for the hyperparameter tuning job
                name when the :meth:`~sagemaker.tuner.HyperparameterTuner.fit`
                method launches. If not specified, a default job name is
                generated, based on the training image name and current
                timestamp.
            warm_start_config (sagemaker.tuner.WarmStartConfig): A
                ``WarmStartConfig`` object that has been initialized with the
                configuration defining the nature of warm start tuning job.
            early_stopping_type (str): Specifies whether early stopping is
                enabled for the job. Can be either 'Auto' or 'Off' (default:
                'Off'). If set to 'Off', early stopping will not be attempted.
                If set to 'Auto', early stopping of some training jobs may
                happen, but is not guaranteed to.
            estimator_name (str): A unique name to identify an estimator within the
                hyperparameter tuning job, when more than one estimator is used with
                the same tuning job (default: None).
        """
        if hyperparameter_ranges is None or len(hyperparameter_ranges) == 0:
            raise ValueError("Need to specify hyperparameter ranges")

        if estimator_name is not None:
            self.estimator = None
            self.objective_metric_name = None
            self._hyperparameter_ranges = None
            self.metric_definitions = None
            self.estimator_dict = {estimator_name: estimator}
            self.objective_metric_name_dict = {estimator_name: objective_metric_name}
            self._hyperparameter_ranges_dict = {estimator_name: hyperparameter_ranges}
            self.metric_definitions_dict = (
                {estimator_name: metric_definitions} if metric_definitions is not None else {}
            )
            self.static_hyperparameters = None
        else:
            self.estimator = estimator
            self.objective_metric_name = objective_metric_name
            self._hyperparameter_ranges = hyperparameter_ranges
            self.metric_definitions = metric_definitions
            self.estimator_dict = None
            self.objective_metric_name_dict = None
            self._hyperparameter_ranges_dict = None
            self.metric_definitions_dict = None
            self.static_hyperparameters_dict = None

        self._validate_parameter_ranges(estimator, hyperparameter_ranges)

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

    def _prepare_for_tuning(self, job_name=None, include_cls_metadata=False):
        """Prepare the tuner instance for tuning (fit)"""
        self._prepare_job_name_for_tuning(job_name=job_name)
        self._prepare_static_hyperparameters_for_tuning(include_cls_metadata=include_cls_metadata)

    def _prepare_job_name_for_tuning(self, job_name=None):
        """Set current job name before starting tuning"""
        if job_name is not None:
            self._current_job_name = job_name
        else:
            base_name = self.base_tuning_job_name
            if base_name is None:
                estimator = (
                    self.estimator or self.estimator_dict[sorted(self.estimator_dict.keys())[0]]
                )
                base_name = base_name_from_image(estimator.train_image())
            self._current_job_name = name_from_base(
                base_name, max_length=self.TUNING_JOB_NAME_MAX_LENGTH, short=True
            )

    def _prepare_static_hyperparameters_for_tuning(self, include_cls_metadata=False):
        """Prepare static hyperparameters for all estimators before tuning"""
        self.static_hyperparameters = None
        if self.estimator is not None:
            self.static_hyperparameters = self._prepare_static_hyperparameters(
                self.estimator, self._hyperparameter_ranges, include_cls_metadata
            )

        self.static_hyperparameters_dict = None
        if self.estimator_dict is not None:
            self.static_hyperparameters_dict = {
                estimator_name: self._prepare_static_hyperparameters(
                    estimator,
                    self._hyperparameter_ranges_dict[estimator_name],
                    include_cls_metadata.get(estimator_name, False),
                )
                for (estimator_name, estimator) in self.estimator_dict.items()
            }

    @classmethod
    def _prepare_static_hyperparameters(
        cls, estimator, hyperparameter_ranges, include_cls_metadata
    ):
        """Prepare static hyperparameters for one estimator before tuning"""
        # Remove any hyperparameter that will be tuned
        static_hyperparameters = {
            to_str(k): to_str(v) for (k, v) in estimator.hyperparameters().items()
        }
        for hyperparameter_name in hyperparameter_ranges.keys():
            static_hyperparameters.pop(hyperparameter_name, None)

        # For attach() to know what estimator to use for frameworks
        # (other algorithms may not accept extra hyperparameters)
        if include_cls_metadata or isinstance(estimator, Framework):
            static_hyperparameters[cls.SAGEMAKER_ESTIMATOR_CLASS_NAME] = json.dumps(
                estimator.__class__.__name__
            )
            static_hyperparameters[cls.SAGEMAKER_ESTIMATOR_MODULE] = json.dumps(
                estimator.__module__
            )

        return static_hyperparameters

    def fit(
        self,
        inputs=None,
        job_name=None,
        include_cls_metadata=False,
        estimator_kwargs=None,
        **kwargs
    ):
        """Start a hyperparameter tuning job.

        Args:
            inputs: Information about the training data. Please refer to the
                ``fit()`` method of the associated estimator, as this can take
                any of the following forms:

                * (str) - The S3 location where training data is saved.
                * (dict[str, str] or dict[str, sagemaker.session.s3_input]) -
                    If using multiple channels for training data, you can specify
                    a dict mapping channel names to strings or
                    :func:`~sagemaker.session.s3_input` objects.
                * (sagemaker.session.s3_input) - Channel configuration for S3 data sources that can
                    provide additional information about the training dataset.
                    See :func:`sagemaker.session.s3_input` for full details.
                * (sagemaker.session.FileSystemInput) - channel configuration for
                    a file system data source that can provide additional information as well as
                    the path to the training dataset.
                * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                    Amazon :class:~`Record` objects serialized and stored in S3.
                    For use with an estimator for an Amazon algorithm.
                * (sagemaker.amazon.amazon_estimator.FileSystemRecordSet) -
                    Amazon SageMaker channel configuration for a file system data source for
                    Amazon algorithms.
                * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
                    where each instance is a different channel of training data.
                * (list[sagemaker.amazon.amazon_estimator.FileSystemRecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.FileSystemRecordSet` objects,
                    where each instance is a different channel of training data.

            job_name (str): Tuning job name. If not specified, the tuner
                generates a default job name, based on the training image name
                and current timestamp.
            include_cls_metadata: It can take one of the following two forms.

                * (bool) - Whether or not the hyperparameter tuning job should include information
                    about the estimator class (default: False). This information is passed as a
                    hyperparameter, so if the algorithm you are using cannot handle unknown
                    hyperparameters (e.g. an Amazon SageMaker built-in algorithm that does not
                    have a custom estimator in the Python SDK), then set ``include_cls_metadata``
                    to ``False``.
                * (dict[str, bool]) - This version should be used for tuners created via the
                    factory method create(), to specify the flag for each estimator provided in
                    the estimator_dict argument of the method. The keys would be the same
                    estimator names as in estimator_dict. If one estimator doesn't need the flag
                    set, then no need to include it in the dictionary.

            estimator_kwargs (dict[str, dict]): Dictionary for other arguments needed for
                training. Should be used only for tuners created via the factory method create().
                The keys are the estimator names for the estimator_dict argument of create()
                method. Each value is a dictionary for the other arguments needed for training
                of the corresponding estimator.
            **kwargs: Other arguments needed for training. Please refer to the
                ``fit()`` method of the associated estimator to see what other
                arguments are needed.
        """
        if self.estimator is not None:
            self._fit_with_estimator(inputs, job_name, include_cls_metadata, **kwargs)
        else:
            self._fit_with_estimator_dict(inputs, job_name, include_cls_metadata, estimator_kwargs)

    def _fit_with_estimator(self, inputs, job_name, include_cls_metadata, **kwargs):
        """Start tuning for tuner instances that have the ``estimator`` field set"""
        self._prepare_estimator_for_tuning(self.estimator, inputs, job_name, **kwargs)
        self._prepare_for_tuning(job_name=job_name, include_cls_metadata=include_cls_metadata)
        self.latest_tuning_job = _TuningJob.start_new(self, inputs)

    def _fit_with_estimator_dict(self, inputs, job_name, include_cls_metadata, estimator_kwargs):
        """Start tuning for tuner instances that have the ``estimator_dict`` field set"""
        estimator_names = sorted(self.estimator_dict.keys())
        self._validate_dict_argument(name="inputs", value=inputs, allowed_keys=estimator_names)
        self._validate_dict_argument(
            name="include_cls_metadata", value=include_cls_metadata, allowed_keys=estimator_names
        )
        self._validate_dict_argument(
            name="estimator_kwargs", value=estimator_kwargs, allowed_keys=estimator_names
        )

        for (estimator_name, estimator) in self.estimator_dict.items():
            ins = inputs.get(estimator_name, None) if inputs is not None else None
            args = estimator_kwargs.get(estimator_name, {}) if estimator_kwargs is not None else {}
            self._prepare_estimator_for_tuning(estimator, ins, job_name, **args)

        inc_cls_metadata = include_cls_metadata if include_cls_metadata is not None else {}
        self._prepare_for_tuning(job_name=job_name, include_cls_metadata=inc_cls_metadata)

        self.latest_tuning_job = _TuningJob.start_new(self, inputs)

    @classmethod
    def _prepare_estimator_for_tuning(cls, estimator, inputs, job_name, **kwargs):
        """Prepare one estimator before starting tuning"""
        if isinstance(inputs, (list, RecordSet, FileSystemRecordSet)):
            estimator._prepare_for_training(inputs, **kwargs)
        else:
            estimator._prepare_for_training(job_name)

    @classmethod
    def attach(cls, tuning_job_name, sagemaker_session=None, job_details=None, estimator_cls=None):
        """Attach to an existing hyperparameter tuning job.

        Create a HyperparameterTuner bound to an existing hyperparameter
        tuning job. After attaching, if there exists a best training job (or any
        other completed training job), that can be deployed to create an Amazon
        SageMaker Endpoint and return a ``Predictor``.

        The ``HyperparameterTuner`` instance could be created in one of the following two forms.

            * If the 'TrainingJobDefinition' field is present in tuning job description, the tuner
                will be created using the default constructor with a single estimator.
            * If the 'TrainingJobDefinitions' field (list) is present in tuning job description,
                the tuner will be created using the factory method ``create()`` with one or
                several estimators. Each estimator corresponds to one item in the
                'TrainingJobDefinitions' field, while the estimator names would come from the
                'DefinitionName' field of items in the 'TrainingJobDefinitions' field. For more
                details on how tuners are created from multiple estimators, see ``create()``
                documentation.

        For more details on 'TrainingJobDefinition' and 'TrainingJobDefinitions' fields in tuning
        job description, see
        https://botocore.readthedocs.io/en/latest/reference/services/sagemaker.html#SageMaker.Client.create_hyper_parameter_tuning_job

        Args:
            tuning_job_name (str): The name of the hyperparameter tuning job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which manages
                interactions with Amazon SageMaker APIs and any other AWS services needed.
                If not specified, one is created using the default AWS configuration chain.
            job_details (dict): The response to a ``DescribeHyperParameterTuningJob`` call.
                If not specified, the ``HyperparameterTuner`` will perform one such call with
                the provided hyperparameter tuning job name.
            estimator_cls: It can take one of the following two forms.

                (str): The estimator class name associated with the training jobs, e.g.
                    'sagemaker.estimator.Estimator'. If not specified, the ``HyperparameterTuner``
                    will try to derive the correct estimator class from training job metadata,
                    defaulting to :class:~`sagemaker.estimator.Estimator` if it is unable to
                    determine a more specific class.
                (dict[str, str]): This form should be used only when the 'TrainingJobDefinitions'
                    field (list) is present in tuning job description. In this scenario training
                    jobs could be created from different training job definitions in the
                    'TrainingJobDefinitions' field, each of which would be mapped to a different
                    estimator after the ``attach()`` call. The ``estimator_cls`` should then be a
                    dictionary to specify estimator class names for individual estimators as
                    needed. The keys should be the 'DefinitionName' value of items in
                    'TrainingJobDefinitions', which would be used as estimator names in the
                    resulting tuner instance.

        Examples:
            Example #1 - assuming we have the following tuning job description, which has the
            'TrainingJobDefinition' field present using a SageMaker built-in algorithm (i.e. PCA),
            and ``attach()`` can derive the estimator class from the training image.
            So ``estimator_cls`` would not be needed.

            .. code:: python

                {
                    'BestTrainingJob': 'best_training_job_name',
                    'TrainingJobDefinition': {
                        'AlgorithmSpecification': {
                            'TrainingImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1,
                        },
                    },
                }

            >>> my_tuner.fit()
            >>> job_name = my_tuner.latest_tuning_job.name
            Later on:
            >>> attached_tuner = HyperparameterTuner.attach(job_name)
            >>> attached_tuner.deploy()

            Example #2 - assuming we have the following tuning job description, which has a 2-item
            list for the 'TrainingJobDefinitions' field. In this case 'estimator_cls' is only
            needed for the 2nd item since the 1st item uses a SageMaker built-in algorithm
            (i.e. PCA).

            .. code:: python

                {
                    'BestTrainingJob': 'best_training_job_name',
                    'TrainingJobDefinitions': [
                        {
                            'DefinitionName': 'estimator_pca',
                            'AlgorithmSpecification': {
                                'TrainingImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/pca:1,
                            },
                        },
                        {
                            'DefinitionName': 'estimator_byoa',
                            'AlgorithmSpecification': {
                                'TrainingImage': '123456789012.dkr.ecr.us-west-2.amazonaws.com/byoa:latest,
                            },
                        }
                    ]
                }

            >>> my_tuner.fit()
            >>> job_name = my_tuner.latest_tuning_job.name
            Later on:
            >>> attached_tuner = HyperparameterTuner.attach(
            >>>     job_name,
            >>>     estimator_cls={
            >>>         'estimator_byoa': 'org.byoa.Estimator'
            >>>     })
            >>> attached_tuner.deploy()


        Returns:
            sagemaker.tuner.HyperparameterTuner: A ``HyperparameterTuner``
            instance with the attached hyperparameter tuning job.
        """
        sagemaker_session = sagemaker_session or Session()

        if job_details is None:
            job_details = sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name
            )

        if "TrainingJobDefinition" in job_details:
            return cls._attach_with_training_details(
                tuning_job_name, sagemaker_session, estimator_cls, job_details
            )

        return cls._attach_with_training_details_list(
            tuning_job_name, sagemaker_session, estimator_cls, job_details
        )

    @classmethod
    def _attach_with_training_details(
        cls, tuning_job_name, sagemaker_session, estimator_cls, job_details
    ):
        """Create a HyperparameterTuner bound to an existing hyperparameter
        tuning job that has the ``TrainingJobDefinition`` field set."""
        estimator = cls._prepare_estimator(
            estimator_cls=estimator_cls,
            training_details=job_details["TrainingJobDefinition"],
            parameter_ranges=job_details["HyperParameterTuningJobConfig"]["ParameterRanges"],
            sagemaker_session=sagemaker_session,
        )
        init_params = cls._prepare_init_params_from_job_description(job_details)

        tuner = cls(estimator=estimator, **init_params)
        tuner.latest_tuning_job = _TuningJob(
            sagemaker_session=sagemaker_session, job_name=tuning_job_name
        )

        return tuner

    @classmethod
    def _attach_with_training_details_list(
        cls, tuning_job_name, sagemaker_session, estimator_cls, job_details
    ):
        """Create a HyperparameterTuner bound to an existing hyperparameter
        tuning job that has the ``TrainingJobDefinitions`` field set."""
        estimator_names = sorted(
            [
                training_details["DefinitionName"]
                for training_details in job_details["TrainingJobDefinitions"]
            ]
        )
        cls._validate_dict_argument(
            name="estimator_cls", value=estimator_cls, allowed_keys=estimator_names
        )

        estimator_dict = {}
        objective_metric_name_dict = {}
        hyperparameter_ranges_dict = {}
        metric_definitions_dict = {}

        for training_details in job_details["TrainingJobDefinitions"]:
            estimator_name = training_details["DefinitionName"]

            estimator_dict[estimator_name] = cls._prepare_estimator(
                estimator_cls=estimator_cls.get(estimator_name) if estimator_cls else None,
                training_details=training_details,
                parameter_ranges=training_details["HyperParameterRanges"],
                sagemaker_session=sagemaker_session,
            )

            objective_metric_name_dict[estimator_name] = training_details["TuningObjective"][
                "MetricName"
            ]
            hyperparameter_ranges_dict[
                estimator_name
            ] = cls._prepare_parameter_ranges_from_job_description(  # noqa: E501 # pylint: disable=line-too-long
                training_details["HyperParameterRanges"]
            )

            metric_definitions = training_details["AlgorithmSpecification"].get(
                "MetricDefinitions", None
            )
            if metric_definitions is not None:
                metric_definitions_dict[estimator_name] = metric_definitions

        init_params = cls._prepare_init_params_from_job_description(job_details)

        tuner = HyperparameterTuner.create(
            estimator_dict=estimator_dict,
            objective_metric_name_dict=objective_metric_name_dict,
            hyperparameter_ranges_dict=hyperparameter_ranges_dict,
            metric_definitions_dict=metric_definitions_dict,
            **init_params
        )
        tuner.latest_tuning_job = _TuningJob(
            sagemaker_session=sagemaker_session, job_name=tuning_job_name
        )

        return tuner

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        accelerator_type=None,
        endpoint_name=None,
        wait=True,
        model_name=None,
        kms_key=None,
        data_capture_config=None,
        **kwargs
    ):
        """Deploy the best trained or user specified model to an Amazon
        SageMaker endpoint and return a ``sagemaker.RealTimePredictor`` object.

        For more information:
        http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html

        Args:
            initial_instance_count (int): Minimum number of EC2 instances to
                deploy to an endpoint for prediction.
            instance_type (str): Type of EC2 instance to deploy to an endpoint
                for prediction, for example, 'ml.c4.xlarge'.
            accelerator_type (str): Type of Elastic Inference accelerator to
                attach to an endpoint for model loading and inference, for
                example, 'ml.eia1.medium'. If not specified, no Elastic
                Inference accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            endpoint_name (str): Name to use for creating an Amazon SageMaker
                endpoint. If not specified, the name of the training job is
                used.
            wait (bool): Whether the call should wait until the deployment of
                model completes (default: True).
            model_name (str): Name to use for creating an Amazon SageMaker
                model. If not specified, the name of the training job is used.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint.
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.
            **kwargs: Other arguments needed for deployment. Please refer to the
                ``create_model()`` method of the associated estimator to see
                what other arguments are needed.

        Returns:
            sagemaker.predictor.RealTimePredictor: A predictor that provides a ``predict()``
                method, which can be used to send requests to the Amazon SageMaker endpoint
                and obtain inferences.
        """
        best_training_job = self._get_best_training_job()
        best_estimator = self.best_estimator(best_training_job)

        return best_estimator.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name or best_training_job["TrainingJobName"],
            wait=wait,
            model_name=model_name,
            kms_key=kms_key,
            data_capture_config=data_capture_config,
            **kwargs
        )

    def stop_tuning_job(self):
        """Stop latest running hyperparameter tuning job."""
        self._ensure_last_tuning_job()
        self.latest_tuning_job.stop()

    def wait(self):
        """Wait for latest hyperparameter tuning job to finish."""
        self._ensure_last_tuning_job()
        self.latest_tuning_job.wait()

    def best_estimator(self, best_training_job=None):
        """Return the estimator that has best training job attached. The trained model can then
        be deployed to an Amazon SageMaker endpoint and return a ``sagemaker.RealTimePredictor``
        object.

        Args:
            best_training_job (dict): Dictionary containing "TrainingJobName" and
                "TrainingJobDefinitionName".

                Example:

                .. code:: python

                    {
                        "TrainingJobName": "my_training_job_name",
                        "TrainingJobDefinitionName": "my_training_job_definition_name"
                    }

        Returns:
            sagemaker.estimator.EstimatorBase: The estimator that has the best training job
                attached.

        Raises:
            Exception: If there is no best training job available for the hyperparameter tuning job.
        """
        if best_training_job is None:
            best_training_job = self._get_best_training_job()

        if self.estimator is not None:
            best_estimator = self.estimator
        else:
            best_estimator_name = best_training_job["TrainingJobDefinitionName"]
            best_estimator = self.estimator_dict[best_estimator_name]

        return best_estimator.attach(
            training_job_name=best_training_job["TrainingJobName"],
            sagemaker_session=self.sagemaker_session,
        )

    def best_training_job(self):
        """Return name of the best training job for the latest hyperparameter
        tuning job.

        Raises:
            Exception: If there is no best training job available for the
                hyperparameter tuning job.
        """
        return self._get_best_training_job()["TrainingJobName"]

    def _get_best_training_job(self):
        """Return the best training job for the latest hyperparameter
        tuning job.

        Raises:
            Exception: If there is no best training job available for the
                hyperparameter tuning job.
        """
        self._ensure_last_tuning_job()

        tuning_job_describe_result = self.sagemaker_session.sagemaker_client.describe_hyper_parameter_tuning_job(  # noqa: E501 # pylint: disable=line-too-long
            HyperParameterTuningJobName=self.latest_tuning_job.name
        )

        try:
            return tuning_job_describe_result["BestTrainingJob"]
        except KeyError:
            raise Exception(
                "Best training job not available for tuning job: {}".format(
                    self.latest_tuning_job.name
                )
            )

    def delete_endpoint(self, endpoint_name=None):
        """Delete an Amazon SageMaker endpoint.

        If an endpoint name is not specified, this defaults to looking for an
        endpoint that shares a name with the best training job for deletion.

        Args:
            endpoint_name (str): Name of the endpoint to delete
        """
        endpoint_name = endpoint_name or self.best_training_job()
        self.sagemaker_session.delete_endpoint(endpoint_name)

    def _ensure_last_tuning_job(self):
        """Placeholder docstring"""
        if self.latest_tuning_job is None:
            raise ValueError("No tuning job available")

    @classmethod
    def _prepare_estimator(
        cls, estimator_cls, training_details, parameter_ranges, sagemaker_session
    ):
        """Attach an estimator from training job details"""
        estimator_cls = cls._prepare_estimator_cls(estimator_cls, training_details)
        return cls._prepare_estimator_from_job_description(
            estimator_cls, training_details, parameter_ranges, sagemaker_session
        )

    @classmethod
    def _prepare_estimator_cls(cls, estimator_cls, training_details):
        # Check for customer-specified estimator first
        """
        Args:
            estimator_cls:
            training_details:
        """
        if estimator_cls is not None:
            module, cls_name = estimator_cls.rsplit(".", 1)
            return getattr(importlib.import_module(module), cls_name)

        # Then check for estimator class in hyperparameters
        hyperparameters = training_details["StaticHyperParameters"]
        if (
            cls.SAGEMAKER_ESTIMATOR_CLASS_NAME in hyperparameters
            and cls.SAGEMAKER_ESTIMATOR_MODULE in hyperparameters
        ):
            module = hyperparameters.get(cls.SAGEMAKER_ESTIMATOR_MODULE)
            cls_name = hyperparameters.get(cls.SAGEMAKER_ESTIMATOR_CLASS_NAME)
            return getattr(importlib.import_module(json.loads(module)), json.loads(cls_name))

        # Then try to derive the estimator from the image name for 1P algorithms
        image_name = training_details["AlgorithmSpecification"]["TrainingImage"]
        algorithm = image_name[image_name.find("/") + 1 : image_name.find(":")]
        if algorithm in AMAZON_ESTIMATOR_CLS_NAMES:
            cls_name = AMAZON_ESTIMATOR_CLS_NAMES[algorithm]
            return getattr(importlib.import_module(AMAZON_ESTIMATOR_MODULE), cls_name)

        # Default to the BYO estimator
        return getattr(
            importlib.import_module(cls.DEFAULT_ESTIMATOR_MODULE), cls.DEFAULT_ESTIMATOR_CLS_NAME
        )

    @classmethod
    def _prepare_estimator_from_job_description(
        cls, estimator_cls, training_details, parameter_ranges, sagemaker_session
    ):
        """
        Args:
            estimator_cls:
            job_details:
            sagemaker_session:
        """
        # Swap name for static hyperparameters to what an estimator would expect
        training_details["HyperParameters"] = training_details["StaticHyperParameters"]
        del training_details["StaticHyperParameters"]

        # Remove hyperparameter reserved by SageMaker for tuning jobs
        del training_details["HyperParameters"]["_tuning_objective_metric"]

        # Add missing hyperparameters defined in the hyperparameter ranges,
        # as potentially required in the Amazon algorithm estimator's constructor
        if issubclass(estimator_cls, AmazonAlgorithmEstimatorBase):
            additional_hyperparameters = cls._extract_hyperparameters_from_parameter_ranges(
                parameter_ranges
            )
            training_details["HyperParameters"].update(additional_hyperparameters)

        # Add items expected by the estimator (but aren't needed otherwise)
        training_details["TrainingJobName"] = ""
        if "KmsKeyId" not in training_details["OutputDataConfig"]:
            training_details["OutputDataConfig"]["KmsKeyId"] = ""

        estimator_init_params = estimator_cls._prepare_init_params_from_job_description(
            training_details
        )
        return estimator_cls(sagemaker_session=sagemaker_session, **estimator_init_params)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """
        Args:
            job_details:
        """
        tuning_config = job_details["HyperParameterTuningJobConfig"]

        params = {
            "strategy": tuning_config["Strategy"],
            "max_jobs": tuning_config["ResourceLimits"]["MaxNumberOfTrainingJobs"],
            "max_parallel_jobs": tuning_config["ResourceLimits"]["MaxParallelTrainingJobs"],
            "warm_start_config": WarmStartConfig.from_job_desc(
                job_details.get("WarmStartConfig", None)
            ),
            "early_stopping_type": tuning_config["TrainingJobEarlyStoppingType"],
        }

        if "HyperParameterTuningJobObjective" in tuning_config:
            params["objective_metric_name"] = tuning_config["HyperParameterTuningJobObjective"][
                "MetricName"
            ]
            params["objective_type"] = tuning_config["HyperParameterTuningJobObjective"]["Type"]

        if "ParameterRanges" in tuning_config:
            params["hyperparameter_ranges"] = cls._prepare_parameter_ranges_from_job_description(
                tuning_config["ParameterRanges"]
            )

        if "TrainingJobDefinition" in job_details:
            params["metric_definitions"] = job_details["TrainingJobDefinition"][
                "AlgorithmSpecification"
            ]["MetricDefinitions"]

        if "TrainingJobDefinitions" in job_details:
            params["objective_type"] = job_details["TrainingJobDefinitions"][0]["TuningObjective"][
                "Type"
            ]

        return params

    @classmethod
    def _prepare_parameter_ranges_from_job_description(cls, parameter_ranges):
        """
        Args:
            parameter_ranges:
        """
        ranges = {}

        for parameter in parameter_ranges["CategoricalParameterRanges"]:
            ranges[parameter["Name"]] = CategoricalParameter(parameter["Values"])

        for parameter in parameter_ranges["ContinuousParameterRanges"]:
            ranges[parameter["Name"]] = ContinuousParameter(
                float(parameter["MinValue"]), float(parameter["MaxValue"])
            )

        for parameter in parameter_ranges["IntegerParameterRanges"]:
            ranges[parameter["Name"]] = IntegerParameter(
                int(parameter["MinValue"]), int(parameter["MaxValue"])
            )

        return ranges

    @classmethod
    def _extract_hyperparameters_from_parameter_ranges(cls, parameter_ranges):
        """
        Args:
            parameter_ranges:
        """
        hyperparameters = {}

        for parameter in parameter_ranges["CategoricalParameterRanges"]:
            hyperparameters[parameter["Name"]] = parameter["Values"][0]

        for parameter in parameter_ranges["ContinuousParameterRanges"]:
            hyperparameters[parameter["Name"]] = float(parameter["MinValue"])

        for parameter in parameter_ranges["IntegerParameterRanges"]:
            hyperparameters[parameter["Name"]] = int(parameter["MinValue"])

        return hyperparameters

    def hyperparameter_ranges(self):
        """Return the hyperparameter ranges in a dictionary to be used as part
        of a request for creating a hyperparameter tuning job.
        """
        if self._hyperparameter_ranges is None:
            return None

        return self._prepare_parameter_ranges_for_tuning(
            self._hyperparameter_ranges, self.estimator
        )

    def hyperparameter_ranges_dict(self):
        """Return a dictionary of hyperparameter ranges for all estimators in ``estimator_dict``
        """
        if self._hyperparameter_ranges_dict is None:
            return None

        return {
            estimator_name: self._prepare_parameter_ranges_for_tuning(
                self._hyperparameter_ranges_dict[estimator_name],
                self.estimator_dict[estimator_name],
            )
            for estimator_name in sorted(self.estimator_dict.keys())
        }

    @classmethod
    def _prepare_parameter_ranges_for_tuning(cls, parameter_ranges, estimator):
        """Prepare hyperparameter ranges for tuning"""
        processed_parameter_ranges = dict()
        for range_type in ParameterRange.__all_types__:
            hp_ranges = []
            for parameter_name, parameter in parameter_ranges.items():
                if parameter is not None and parameter.__name__ == range_type:
                    # Categorical parameters needed to be serialized as JSON for our framework
                    # containers
                    if isinstance(parameter, CategoricalParameter) and isinstance(
                        estimator, Framework
                    ):
                        tuning_range = parameter.as_json_range(parameter_name)
                    else:
                        tuning_range = parameter.as_tuning_range(parameter_name)
                    hp_ranges.append(tuning_range)
            processed_parameter_ranges[range_type + "ParameterRanges"] = hp_ranges
        return processed_parameter_ranges

    @property
    def sagemaker_session(self):
        """Convenience method for accessing the
        :class:`~sagemaker.session.Session` object associated with the estimator
        for the ``HyperparameterTuner``.
        """
        estimator = self.estimator
        if estimator is None:
            first_estimator_name = sorted(self.estimator_dict.keys())[0]
            estimator = self.estimator_dict[first_estimator_name]
        return estimator.sagemaker_session

    def analytics(self):
        """An instance of HyperparameterTuningJobAnalytics for this latest
        tuning job of this tuner. Analytics olbject gives you access to tuning
        results summarized into a pandas dataframe.
        """
        return HyperparameterTuningJobAnalytics(self.latest_tuning_job.name, self.sagemaker_session)

    def _validate_parameter_ranges(self, estimator, hyperparameter_ranges):
        """Validate hyperparameter ranges for an estimator"""
        for kls in inspect.getmro(estimator.__class__)[::-1]:
            for _, value in kls.__dict__.items():
                if isinstance(value, hp):
                    try:
                        # The hyperparam names may not be the same as the class attribute that
                        # holds them, for instance: local_lloyd_init_method is called
                        # local_init_method. We need to map these and pass the correct name to
                        # the constructor.
                        parameter_range = hyperparameter_ranges[value.name]

                        if isinstance(parameter_range, ParameterRange):
                            self._validate_parameter_range(value, parameter_range)
                    except KeyError:
                        pass

    def _validate_parameter_range(self, value_hp, parameter_range):
        """
        Args:
            value_hp:
            parameter_range:
        """
        for (parameter_range_key, parameter_range_value) in parameter_range.__dict__.items():
            if parameter_range_key == "scaling_type":
                continue

            # Categorical ranges
            if isinstance(parameter_range_value, list):
                for categorical_value in parameter_range_value:
                    value_hp.validate(categorical_value)
            # Continuous, Integer ranges
            else:
                value_hp.validate(parameter_range_value)

    def transfer_learning_tuner(self, additional_parents=None, estimator=None):
        """Creates a new ``HyperparameterTuner`` by copying the request fields
        from the provided parent to the new instance of ``HyperparameterTuner``.
        Followed by addition of warm start configuration with the type as
        "TransferLearning" and parents as the union of provided list of
        ``additional_parents`` and the ``self``. Also, training image in the new
        tuner's estimator is updated with the provided ``training_image``.

        Examples:
            >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
            >>> transfer_learning_tuner = parent_tuner.transfer_learning_tuner(
            >>>                                             additional_parents={"parent-job-2"})
            Later On:
            >>> transfer_learning_tuner.fit(inputs={})

        Args:
            additional_parents (set{str}): Set of additional parents along with
                the self to be used in warm starting
            estimator (sagemaker.estimator.EstimatorBase): An estimator object
                that has been initialized with the desired configuration. There
                does not need to be a training job associated with this
                instance.

        Returns:
            sagemaker.tuner.HyperparameterTuner: ``HyperparameterTuner``
            instance which can be used to launch transfer learning tuning job.
        """

        return self._create_warm_start_tuner(
            additional_parents=additional_parents,
            warm_start_type=WarmStartTypes.TRANSFER_LEARNING,
            estimator=estimator,
        )

    def identical_dataset_and_algorithm_tuner(self, additional_parents=None):
        """Creates a new ``HyperparameterTuner`` by copying the request fields
        from the provided parent to the new instance of ``HyperparameterTuner``.
        Followed by addition of warm start configuration with the type as
        "IdenticalDataAndAlgorithm" and parents as the union of provided list of
        ``additional_parents`` and the ``self``

        Examples:
            >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
            >>> identical_dataset_algo_tuner = parent_tuner.identical_dataset_and_algorithm_tuner(
            >>>                                                additional_parents={"parent-job-2"})
            Later On:
            >>> identical_dataset_algo_tuner.fit(inputs={})

        Args:
            additional_parents (set{str}): Set of additional parents along with
                the self to be used in warm starting

        Returns:
            sagemaker.tuner.HyperparameterTuner: HyperparameterTuner instance
            which can be used to launch identical dataset and algorithm tuning
            job.
        """

        return self._create_warm_start_tuner(
            additional_parents=additional_parents,
            warm_start_type=WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
        )

    def _create_warm_start_tuner(self, additional_parents, warm_start_type, estimator=None):
        """Creates a new ``HyperparameterTuner`` with ``WarmStartConfig``, where
        type will be equal to ``warm_start_type`` and``parents`` would be equal
        to union of ``additional_parents`` and self.

        Args:
            additional_parents (set{str}): Additional parents along with self,
                to be used for warm starting.
            warm_start_type (sagemaker.tuner.WarmStartTypes): Type of warm start
                job.
            estimator:

        Returns:
            sagemaker.tuner.HyperparameterTuner: Instance with the request
            fields copied from self along with the warm start configuration
        """
        all_parents = {self.latest_tuning_job.name}
        if additional_parents:
            all_parents = all_parents.union(additional_parents)

        if self.estimator is not None:
            return HyperparameterTuner(
                estimator=estimator if estimator else self.estimator,
                objective_metric_name=self.objective_metric_name,
                hyperparameter_ranges=self._hyperparameter_ranges,
                strategy=self.strategy,
                objective_type=self.objective_type,
                max_jobs=self.max_jobs,
                max_parallel_jobs=self.max_parallel_jobs,
                warm_start_config=WarmStartConfig(
                    warm_start_type=warm_start_type, parents=all_parents
                ),
                early_stopping_type=self.early_stopping_type,
            )

        if len(self.estimator_dict) > 1:
            raise ValueError(
                "Warm start is not supported currently for tuners with multiple estimators"
            )

        if estimator is not None:
            estimator_name = list(self.estimator_dict.keys())[0]
            estimator_dict = {estimator_name: estimator}
        else:
            estimator_dict = self.estimator_dict

        return HyperparameterTuner.create(
            estimator_dict=estimator_dict,
            objective_metric_name_dict=self.objective_metric_name_dict,
            hyperparameter_ranges_dict=self._hyperparameter_ranges_dict,
            metric_definitions_dict=self.metric_definitions_dict,
            strategy=self.strategy,
            objective_type=self.objective_type,
            max_jobs=self.max_jobs,
            max_parallel_jobs=self.max_parallel_jobs,
            warm_start_config=WarmStartConfig(warm_start_type=warm_start_type, parents=all_parents),
            early_stopping_type=self.early_stopping_type,
        )

    @classmethod
    def create(
        cls,
        estimator_dict,
        objective_metric_name_dict,
        hyperparameter_ranges_dict,
        metric_definitions_dict=None,
        base_tuning_job_name=None,
        strategy="Bayesian",
        objective_type="Maximize",
        max_jobs=1,
        max_parallel_jobs=1,
        tags=None,
        warm_start_config=None,
        early_stopping_type="Off",
    ):
        """Factory method to create a ``HyperparameterTuner`` instance. It takes one or more
        estimators to obtain configuration information for training jobs that are created as the
        result of a hyperparameter tuning job. The estimators are provided through a dictionary
        (i.e. ``estimator_dict``) with unique estimator names as the keys. For individual
        estimators separate objective metric names and hyperparameter ranges should be provided in
        two dictionaries, i.e. ``objective_metric_name_dict`` and ``hyperparameter_ranges_dict``,
        with the same estimator names as the keys. Optional metrics definitions could also be
        provided for individual estimators via another dictionary ``metric_definitions_dict``.

        Args:
            estimator_dict (dict[str, sagemaker.estimator.EstimatorBase]): Dictionary of estimator
                instances that have been initialized with the desired configuration. There does not
                need to be a training job associated with the estimator instances. The keys of the
                dictionary would be referred to as "estimator names".
            objective_metric_name_dict (dict[str, str]): Dictionary of names of the objective
                metric for evaluating training jobs. The keys are the same set of estimator names
                as in ``estimator_dict``, and there must be one entry for each estimator in
                ``estimator_dict``.
            hyperparameter_ranges_dict (dict[str, dict[str, sagemaker.parameter.ParameterRange]]):
                Dictionary of tunable hyperparameter ranges. The keys are the same set of estimator
                names as in estimator_dict, and there must be one entry for each estimator in
                estimator_dict. Each value is a dictionary of sagemaker.parameter.ParameterRange
                instance, which can be one of three types: Continuous, Integer, or Categorical.
                The keys of each ParameterRange dictionaries are the names of the hyperparameter,
                and the values are the appropriate parameter range class to represent the range.
            metric_definitions_dict (dict(str, list[dict]]): Dictionary of metric definitions.
                The keys are the same set or a subset of estimator names as in estimator_dict,
                and there must be one entry for each estimator in estimator_dict. Each value is
                a list of dictionaries that defines the metric(s) used to evaluate the training
                jobs (default: None). Each of these dictionaries contains two keys: 'Name' for the
                name of the metric, and 'Regex' for the regular expression used to extract the
                metric from the logs. This should be defined only for hyperparameter tuning jobs
                that don't use an Amazon algorithm.
            base_tuning_job_name (str): Prefix for the hyperparameter tuning job name when the
                :meth:`~sagemaker.tuner.HyperparameterTuner.fit` method launches. If not specified,
                a default job name is generated, based on the training image name and current
                timestamp.
            strategy (str): Strategy to be used for hyperparameter estimations
                (default: 'Bayesian').
            objective_type (str): The type of the objective metric for evaluating training jobs.
                This value can be either 'Minimize' or 'Maximize' (default: 'Maximize').
            max_jobs (int): Maximum total number of training jobs to start for the hyperparameter
                tuning job (default: 1).
            max_parallel_jobs (int): Maximum number of parallel training jobs to start
                (default: 1).
            tags (list[dict]): List of tags for labeling the tuning job (default: None). For more,
                see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            warm_start_config (sagemaker.tuner.WarmStartConfig): A ``WarmStartConfig`` object that
                has been initialized with the configuration defining the nature of warm start
                tuning job.
            early_stopping_type (str): Specifies whether early stopping is enabled for the job.
                Can be either 'Auto' or 'Off' (default: 'Off'). If set to 'Off', early stopping
                will not be attempted. If set to 'Auto', early stopping of some training jobs may
                happen, but is not guaranteed to.

        Returns:
            sagemaker.tuner.HyperparameterTuner: a new ``HyperparameterTuner`` object that can
            start a hyperparameter tuning job with one or more estimators.

        """

        cls._validate_create_tuner_inputs(
            estimator_dict,
            objective_metric_name_dict,
            hyperparameter_ranges_dict,
            metric_definitions_dict,
        )

        estimator_names = sorted(estimator_dict.keys())
        first_estimator_name = estimator_names[0]

        metric_definitions = (
            metric_definitions_dict.get(first_estimator_name, None)
            if metric_definitions_dict is not None
            else None
        )

        tuner = HyperparameterTuner(
            base_tuning_job_name=base_tuning_job_name,
            estimator_name=first_estimator_name,
            estimator=estimator_dict[first_estimator_name],
            objective_metric_name=objective_metric_name_dict[first_estimator_name],
            hyperparameter_ranges=hyperparameter_ranges_dict[first_estimator_name],
            metric_definitions=metric_definitions,
            strategy=strategy,
            objective_type=objective_type,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            tags=tags,
            warm_start_config=warm_start_config,
            early_stopping_type=early_stopping_type,
        )

        for estimator_name in estimator_names[1:]:
            metric_definitions = (
                metric_definitions_dict.get(estimator_name, None)
                if metric_definitions_dict is not None
                else None
            )
            tuner._add_estimator(
                estimator_name=estimator_name,
                estimator=estimator_dict[estimator_name],
                objective_metric_name=objective_metric_name_dict[estimator_name],
                hyperparameter_ranges=hyperparameter_ranges_dict[estimator_name],
                metric_definitions=metric_definitions,
            )
        return tuner

    @classmethod
    def _validate_create_tuner_inputs(
        cls,
        estimator_dict,
        objective_metric_name_dict,
        hyperparameter_ranges_dict,
        metric_definitions_dict=None,
    ):
        """Validate inputs for ``HyperparameterTuner.create()``"""
        cls._validate_estimator_dict(estimator_dict)

        estimator_names = sorted(estimator_dict.keys())

        cls._validate_dict_argument(
            name="objective_metric_name_dict",
            value=objective_metric_name_dict,
            allowed_keys=estimator_names,
            require_same_keys=True,
        )
        cls._validate_dict_argument(
            name="hyperparameter_ranges_dict",
            value=hyperparameter_ranges_dict,
            allowed_keys=estimator_names,
            require_same_keys=True,
        )
        cls._validate_dict_argument(
            name="metric_definitions_dict",
            value=metric_definitions_dict,
            allowed_keys=estimator_names,
        )

    @classmethod
    def _validate_estimator_dict(cls, estimator_dict):
        """Validate ``estimator_dict`` in inputs for ``HyperparameterTuner.create()``"""
        if estimator_dict is None or len(estimator_dict) == 0:
            raise ValueError("At least one estimator should be provided")
        if None in estimator_dict.keys():
            raise ValueError("Estimator names cannot be None")

    @classmethod
    def _validate_dict_argument(cls, name, value, allowed_keys, require_same_keys=False):
        """Check if an argument is an dictionary with correct key set"""
        if value is None:
            return

        if not isinstance(value, dict):
            raise ValueError(
                "Argument '{}' must be a dictionary using {} as keys".format(name, allowed_keys)
            )

        value_keys = sorted(value.keys())

        if require_same_keys:
            if value_keys != allowed_keys:
                raise ValueError(
                    "The keys of argument '{}' must be the same as {}".format(name, allowed_keys)
                )
        else:
            if not set(value_keys).issubset(set(allowed_keys)):
                raise ValueError(
                    "The keys of argument '{}' must be a subset of {}".format(name, allowed_keys)
                )

    def _add_estimator(
        self,
        estimator_name,
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions=None,
    ):
        """Add an estimator with corresponding objective metric name, parameter ranges and metric
        definitions (if applicable)"""
        self.estimator_dict[estimator_name] = estimator
        self.objective_metric_name_dict[estimator_name] = objective_metric_name
        self._hyperparameter_ranges_dict[estimator_name] = hyperparameter_ranges
        if metric_definitions is not None:
            self.metric_definitions_dict[estimator_name] = metric_definitions


class _TuningJob(_Job):
    """Placeholder docstring"""

    @classmethod
    def start_new(cls, tuner, inputs):
        """Create a new Amazon SageMaker hyperparameter tuning job from the
        HyperparameterTuner.

        Args:
            tuner (sagemaker.tuner.HyperparameterTuner): HyperparameterTuner
                object created by the user.
            inputs (str): Parameters used when called
                :meth:`~sagemaker.estimator.EstimatorBase.fit`.

        Returns:
            sagemaker.tuner._TuningJob: Constructed object that captures all
            information about the started job.
        """

        logging.info("_TuningJob.start_new!!!")

        warm_start_config_req = None
        if tuner.warm_start_config:
            warm_start_config_req = tuner.warm_start_config.to_input_req()

        tuning_config = {
            "strategy": tuner.strategy,
            "max_jobs": tuner.max_jobs,
            "max_parallel_jobs": tuner.max_parallel_jobs,
            "early_stopping_type": tuner.early_stopping_type,
        }

        if tuner.objective_metric_name is not None:
            tuning_config["objective_type"] = tuner.objective_type
            tuning_config["objective_metric_name"] = tuner.objective_metric_name

        parameter_ranges = tuner.hyperparameter_ranges()
        if parameter_ranges is not None:
            tuning_config["parameter_ranges"] = parameter_ranges

        tuner_args = {
            "job_name": tuner._current_job_name,
            "tuning_config": tuning_config,
            "tags": tuner.tags,
            "warm_start_config": warm_start_config_req,
        }

        if tuner.estimator is not None:
            tuner_args["training_config"] = cls._prepare_training_config(
                inputs, tuner.estimator, tuner.static_hyperparameters, tuner.metric_definitions
            )

        if tuner.estimator_dict is not None:
            tuner_args["training_config_list"] = [
                cls._prepare_training_config(
                    inputs.get(estimator_name, None) if inputs is not None else None,
                    tuner.estimator_dict[estimator_name],
                    tuner.static_hyperparameters_dict[estimator_name],
                    tuner.metric_definitions_dict.get(estimator_name, None),
                    estimator_name,
                    tuner.objective_type,
                    tuner.objective_metric_name_dict[estimator_name],
                    tuner.hyperparameter_ranges_dict()[estimator_name],
                )
                for estimator_name in sorted(tuner.estimator_dict.keys())
            ]

        tuner.sagemaker_session.create_tuning_job(**tuner_args)
        return cls(tuner.sagemaker_session, tuner._current_job_name)

    @staticmethod
    def _prepare_training_config(
        inputs,
        estimator,
        static_hyperparameters,
        metric_definitions,
        estimator_name=None,
        objective_type=None,
        objective_metric_name=None,
        parameter_ranges=None,
    ):
        """Prepare training config for one estimator"""
        training_config = _Job._load_config(inputs, estimator)

        training_config["input_mode"] = estimator.input_mode
        training_config["metric_definitions"] = metric_definitions

        if isinstance(inputs, s3_input):
            if "InputMode" in inputs.config:
                logging.debug(
                    "Selecting s3_input's input_mode (%s) for TrainingInputMode.",
                    inputs.config["InputMode"],
                )
                training_config["input_mode"] = inputs.config["InputMode"]

        if isinstance(estimator, sagemaker.algorithm.AlgorithmEstimator):
            training_config["algorithm_arn"] = estimator.algorithm_arn
        else:
            training_config["image"] = estimator.train_image()

        training_config["enable_network_isolation"] = estimator.enable_network_isolation()
        training_config[
            "encrypt_inter_container_traffic"
        ] = estimator.encrypt_inter_container_traffic

        training_config["train_use_spot_instances"] = estimator.train_use_spot_instances
        training_config["checkpoint_s3_uri"] = estimator.checkpoint_s3_uri
        training_config["checkpoint_local_path"] = estimator.checkpoint_local_path

        training_config["static_hyperparameters"] = static_hyperparameters

        if estimator_name is not None:
            training_config["estimator_name"] = estimator_name

        if objective_type is not None:
            training_config["objective_type"] = objective_type

        if objective_metric_name is not None:
            training_config["objective_metric_name"] = objective_metric_name

        if parameter_ranges is not None:
            training_config["parameter_ranges"] = parameter_ranges

        return training_config

    def stop(self):
        """Placeholder docstring"""
        self.sagemaker_session.stop_tuning_job(name=self.name)

    def wait(self):
        """Placeholder docstring"""
        self.sagemaker_session.wait_for_tuning_job(self.name)


def create_identical_dataset_and_algorithm_tuner(
    parent, additional_parents=None, sagemaker_session=None
):
    """Creates a new tuner by copying the request fields from the provided parent to the new
        instance of ``HyperparameterTuner`` followed by addition of warm start configuration
        with the type as "IdenticalDataAndAlgorithm" and ``parents`` as the
        union of provided list of ``additional_parents`` and the ``parent``.

    Args:
        parent (str): Primary parent tuning job's name from which the Tuner and
            Estimator configuration has to be copied
        additional_parents (set{str}): Set of additional parent tuning job's
            names along with the primary parent tuning job name to be used in
            warm starting the transfer learning tuner.
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other AWS
            services needed. If not specified, one is created using the default
            AWS configuration chain.

    Returns:
        sagemaker.tuner.HyperparameterTuner: a new ``HyperparameterTuner``
        object for the warm-started hyperparameter tuning job
    """

    parent_tuner = HyperparameterTuner.attach(
        tuning_job_name=parent, sagemaker_session=sagemaker_session
    )
    return parent_tuner.identical_dataset_and_algorithm_tuner(additional_parents=additional_parents)


def create_transfer_learning_tuner(
    parent, additional_parents=None, estimator=None, sagemaker_session=None
):
    """Creates a new ``HyperParameterTuner`` by copying the request fields from the
    provided parent to the new instance
        of ``HyperparameterTuner`` followed by addition of warm start
        configuration with the type as "TransferLearning" and ``parents`` as the
        union of provided list of ``additional_parents`` and the ``parent``.

    Args:
        parent (str): Primary parent tuning job's name from which the Tuner and
            Estimator configuration has to be copied
        additional_parents (set{str}): Set of additional parent tuning job's
            names along with the primary parent tuning job name to be used in
            warm starting the identical dataset and algorithm tuner.
        estimator (sagemaker.estimator.EstimatorBase): An estimator object that
            has been initialized with the desired configuration. There does not
            need to be a training job associated with this instance.
        sagemaker_session (sagemaker.session.Session): Session object which
            manages interactions with Amazon SageMaker APIs and any other AWS
            services needed. If not specified, one is created using the default
            AWS configuration chain.

    Returns:
        sagemaker.tuner.HyperparameterTuner: New instance of warm started
        HyperparameterTuner
    """

    parent_tuner = HyperparameterTuner.attach(
        tuning_job_name=parent, sagemaker_session=sagemaker_session
    )
    return parent_tuner.transfer_learning_tuner(
        additional_parents=additional_parents, estimator=estimator
    )
