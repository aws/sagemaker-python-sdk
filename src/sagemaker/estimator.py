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
from __future__ import print_function, absolute_import

import json
import logging
import os
import uuid

from abc import ABCMeta
from abc import abstractmethod

from six import with_metaclass
from six import string_types
from six.moves.urllib.parse import urlparse
import sagemaker
from sagemaker import git_utils, image_uris
from sagemaker.analytics import TrainingJobAnalytics
from sagemaker.debugger import TensorBoardOutputConfig  # noqa: F401 # pylint: disable=unused-import
from sagemaker.debugger import (
    DebuggerHookConfig,
    FrameworkProfile,
    get_default_profiler_rule,
    get_rule_container_image_uri,
    ProfilerConfig,
    ProfilerRule,
    Rule,
)
from sagemaker.deprecations import (
    removed_kwargs,
    removed_function,
    renamed_kwargs,
)
from sagemaker.s3 import S3Uploader, parse_s3_url

from sagemaker.fw_utils import (
    tar_and_upload_dir,
    UploadedCode,
    validate_source_dir,
    _region_supports_debugger,
    _region_supports_profiler,
    get_mp_parameters,
)
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.entities import Expression
from sagemaker.inputs import TrainingInput
from sagemaker.job import _Job
from sagemaker.local import LocalSession
from sagemaker.model import Model, NEO_ALLOWED_FRAMEWORKS
from sagemaker.model import (
    SCRIPT_PARAM_NAME,
    DIR_PARAM_NAME,
    CONTAINER_LOG_LEVEL_PARAM_NAME,
    JOB_NAME_PARAM_NAME,
    SAGEMAKER_REGION_PARAM_NAME,
)
from sagemaker.predictor import Predictor
from sagemaker.session import Session
from sagemaker.transformer import Transformer
from sagemaker.utils import (
    base_from_name,
    base_name_from_image,
    build_dict,
    get_config_value,
    name_from_base,
)
from sagemaker import vpc_utils

logger = logging.getLogger(__name__)


class EstimatorBase(with_metaclass(ABCMeta, object)):  # pylint: disable=too-many-public-methods
    """Handle end-to-end Amazon SageMaker training and deployment tasks.

    For introduction to model training and deployment, see
    http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html

    Subclasses must define a way to determine what image to use for training,
    what hyperparameters to use, and how to create an appropriate predictor
    instance.
    """

    def __init__(
        self,
        role,
        instance_count=None,
        instance_type=None,
        volume_size=30,
        volume_kms_key=None,
        max_run=24 * 60 * 60,
        input_mode="File",
        output_path=None,
        output_kms_key=None,
        base_job_name=None,
        sagemaker_session=None,
        tags=None,
        subnets=None,
        security_group_ids=None,
        model_uri=None,
        model_channel_name="model",
        metric_definitions=None,
        encrypt_inter_container_traffic=False,
        use_spot_instances=False,
        max_wait=None,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
        rules=None,
        debugger_hook_config=None,
        tensorboard_output_config=None,
        enable_sagemaker_metrics=None,
        enable_network_isolation=False,
        profiler_config=None,
        disable_profiler=False,
        environment=None,
        max_retry_attempts=None,
        **kwargs,
    ):
        """Initialize an ``EstimatorBase`` instance.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            instance_count (int): Number of Amazon EC2 instances to use
                for training.
            instance_type (str): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            volume_size (int): Size in GB of the EBS volume to use for
                storing input data during training (default: 30). Must be large
                enough to store training data if File Mode is used (which is the
                default).
            volume_kms_key (str): Optional. KMS key ID for encrypting EBS
                volume attached to the training instance (default: None).
            max_run (int): Timeout in seconds for training (default: 24 *
                60 * 60). After this amount of time Amazon SageMaker terminates
                the job regardless of its current status.
            input_mode (str): The input mode that the algorithm supports
                (default: 'File'). Valid modes: 'File' - Amazon SageMaker copies
                the training dataset from the S3 location to a local directory.
                'Pipe' - Amazon SageMaker streams data directly from S3 to the
                container via a Unix-named pipe. This argument can be overriden
                on a per-channel basis using
                ``sagemaker.inputs.TrainingInput.input_mode``.
            output_path (str): S3 location for saving the training result (model
                artifacts and output files). If not specified, results are
                stored to a default bucket. If the bucket with the specific name
                does not exist, the estimator creates the bucket during the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method execution.
                file:// urls are used for local mode. For example: 'file://model/'
                will save to the model folder in the current directory.
            output_kms_key (str): Optional. KMS key ID for encrypting the
                training output (default: None).
            base_job_name (str): Prefix for training job name when the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method launches.
                If not specified, the estimator generates a default job name
                based on the training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            tags (list[dict]): List of tags for labeling a training job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            subnets (list[str]): List of subnet ids. If not specified training
                job will be created without VPC config.
            security_group_ids (list[str]): List of security group ids. If not
                specified training job will be created without VPC config.
            model_uri (str): URI where a pre-trained model is stored, either
                locally or in S3 (default: None). If specified, the estimator
                will create a channel pointing to the model so the training job
                can download it. This model can be a 'model.tar.gz' from a
                previous training job, or other artifacts coming from a
                different source.

                In local mode, this should point to the path in which the model
                is located and not the file itself, as local Docker containers
                will try to mount the URI as a volume.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
            model_channel_name (str): Name of the channel where 'model_uri' will
                be downloaded (default: 'model').
            metric_definitions (list[dict]): A list of dictionaries that defines
                the metric(s) used to evaluate the training jobs. Each
                dictionary contains two keys: 'Name' for the name of the metric,
                and 'Regex' for the regular expression used to extract the
                metric from the logs. This should be defined only for jobs that
                don't use an Amazon algorithm.
            encrypt_inter_container_traffic (bool): Specifies whether traffic
                between training containers is encrypted for the training job
                (default: ``False``).
            use_spot_instances (bool): Specifies whether to use SageMaker
                Managed Spot instances for training. If enabled then the
                ``max_wait`` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (default: ``False``).
            max_wait (int): Timeout in seconds waiting for spot training
                job (default: None). After this amount of time Amazon
                SageMaker will stop waiting for managed spot training job to
                complete (default: ``None``).
            checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
                that the algorithm persists (if any) during training. (default:
                ``None``).
            checkpoint_local_path (str): The local path that the algorithm
                writes its checkpoints to. SageMaker will persist all files
                under this path to `checkpoint_s3_uri` continually during
                training. On job startup the reverse happens - data from the
                s3 location is downloaded to this path before the algorithm is
                started. If the path is unset then SageMaker assumes the
                checkpoints will be provided under `/opt/ml/checkpoints/`.
                (default: ``None``).
            rules (list[:class:`~sagemaker.debugger.RuleBase`]): A list of
                :class:`~sagemaker.debugger.RuleBase` objects used to define
                SageMaker Debugger rules for real-time analysis
                (default: ``None``). For more information,
                see `Continuous analyses through rules
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html
                #continuous-analyses-through-rules)>`_.
            debugger_hook_config (:class:`~sagemaker.debugger.DebuggerHookConfig` or bool):
                Configuration for how debugging information is emitted with
                SageMaker Debugger. If not specified, a default one is created using
                the estimator's ``output_path``, unless the region does not
                support SageMaker Debugger. To disable SageMaker Debugger,
                set this parameter to ``False``. For more information, see
                `Capture real-time debugging data during model training in Amazon SageMaker
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#
                capture-real-time-debugging-data-during-model-training-in-amazon-sagemaker>`_.
            tensorboard_output_config (:class:`~sagemaker.debugger.TensorBoardOutputConfig`):
                Configuration for customizing debugging visualization using TensorBoard
                (default: ``None``). For more information,
                see `Capture real time tensorboard data
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#
                capture-real-time-tensorboard-data-from-the-debugging-hook>`_.
            enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
                Series. For more information, see `AlgorithmSpecification API
                <https://docs.aws.amazon.com/sagemaker/latest/dg/
                API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-
                EnableSageMakerMetricsTimeSeries>`_.
                (default: ``None``).
            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode (default: ``False``). Network
                isolation mode restricts the container access to outside networks
                (such as the Internet). The container does not make any inbound or
                outbound network calls. Also known as Internet-free mode.
            profiler_config (:class:`~sagemaker.debugger.ProfilerConfig`):
                Configuration for how SageMaker Debugger collects
                monitoring and profiling information from your training job.
                If not specified, a default configuration is created using
                the estimator's ``output_path``, unless the region does not
                support SageMaker Debugger. To disable SageMaker Debugger
                monitoring and profiling, set the
                ``disable_profiler`` parameter to ``True``.
            disable_profiler (bool): Specifies whether Debugger monitoring and profiling
                will be disabled (default: ``False``).
            environment (dict[str, str]) : Environment variables to be set for
                use during training job (default: ``None``)
             max_retry_attempts (int): The number of times to move a job to the STARTING status.
                You can specify between 1 and 30 attempts.
                If the value of attempts is greater than zero,
                the job is retried on InternalServerFailure
                the same number of attempts as the value.
                You can cap the total duration for your job by setting ``max_wait`` and ``max_run``
                (default: ``None``)

        """
        instance_count = renamed_kwargs(
            "train_instance_count", "instance_count", instance_count, kwargs
        )
        instance_type = renamed_kwargs(
            "train_instance_type", "instance_type", instance_type, kwargs
        )
        max_run = renamed_kwargs("train_max_run", "max_run", max_run, kwargs)
        use_spot_instances = renamed_kwargs(
            "train_use_spot_instances", "use_spot_instances", use_spot_instances, kwargs
        )
        max_wait = renamed_kwargs("train_max_wait", "max_wait", max_wait, kwargs)
        volume_size = renamed_kwargs("train_volume_size", "volume_size", volume_size, kwargs)
        volume_kms_key = renamed_kwargs(
            "train_volume_kms_key", "volume_kms_key", volume_kms_key, kwargs
        )

        if instance_count is None or instance_type is None:
            raise ValueError("Both instance_count and instance_type are required.")

        self.role = role
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.volume_size = volume_size
        self.volume_kms_key = volume_kms_key
        self.max_run = max_run
        self.input_mode = input_mode
        self.tags = tags
        self.metric_definitions = metric_definitions
        self.model_uri = model_uri
        self.model_channel_name = model_channel_name
        self.code_uri = None
        self.code_channel_name = "code"

        if self.instance_type in ("local", "local_gpu"):
            if self.instance_type == "local_gpu" and self.instance_count > 1:
                raise RuntimeError("Distributed Training in Local GPU is not supported")
            self.sagemaker_session = sagemaker_session or LocalSession()
            if not isinstance(self.sagemaker_session, sagemaker.local.LocalSession):
                raise RuntimeError(
                    "instance_type local or local_gpu is only supported with an"
                    "instance of LocalSession"
                )
        else:
            self.sagemaker_session = sagemaker_session or Session()

        self.base_job_name = base_job_name
        self._current_job_name = None
        if (
            not self.sagemaker_session.local_mode
            and output_path
            and output_path.startswith("file://")
        ):
            raise RuntimeError("file:// output paths are only supported in Local Mode")
        self.output_path = output_path
        self.output_kms_key = output_kms_key
        self.latest_training_job = None
        self.jobs = []
        self.deploy_instance_type = None

        self._compiled_models = {}

        # VPC configurations
        self.subnets = subnets
        self.security_group_ids = security_group_ids

        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        self.use_spot_instances = use_spot_instances
        self.max_wait = max_wait
        self.checkpoint_s3_uri = checkpoint_s3_uri
        self.checkpoint_local_path = checkpoint_local_path

        self.rules = rules
        self.debugger_hook_config = debugger_hook_config
        self.tensorboard_output_config = tensorboard_output_config

        self.debugger_rule_configs = None
        self.collection_configs = None

        self.enable_sagemaker_metrics = enable_sagemaker_metrics
        self._enable_network_isolation = enable_network_isolation

        self.profiler_config = profiler_config
        self.disable_profiler = disable_profiler

        self.environment = environment

        self.max_retry_attempts = max_retry_attempts

        if not _region_supports_profiler(self.sagemaker_session.boto_region_name):
            self.disable_profiler = True

        self.profiler_rule_configs = None
        self.profiler_rules = None
        self.debugger_rules = None

    @abstractmethod
    def training_image_uri(self):
        """Return the Docker image to use for training.

        The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does
        the model training, calls this method to find the image to use for model
        training.

        Returns:
            str: The URI of the Docker image.
        """

    @abstractmethod
    def hyperparameters(self):
        """Return the hyperparameters as a dictionary to use for training.

        The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which
        trains the model, calls this method to find the hyperparameters.

        Returns:
            dict[str, str]: The hyperparameters.
        """

    def enable_network_isolation(self):
        """Return True if this Estimator will need network isolation to run.

        Returns:
            bool: Whether this Estimator needs network isolation or not.
        """
        return self._enable_network_isolation

    def prepare_workflow_for_training(self, job_name=None):
        """Calls _prepare_for_training. Used when setting up a workflow.

        Args:
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        self._prepare_for_training(job_name=job_name)

    def _ensure_base_job_name(self):
        """Set ``self.base_job_name`` if it is not set already."""
        # honor supplied base_job_name or generate it
        if self.base_job_name is None:
            self.base_job_name = base_name_from_image(self.training_image_uri())

    def _get_or_create_name(self, name=None):
        """Generate a name based on the base job name or training image if needed.

        Args:
            name (str): User-supplied name. If not specified, a name is generated from
                the base job name or training image.

        Returns:
            str: Either the user-supplied name or a generated name.
        """
        if name:
            return name

        self._ensure_base_job_name()
        return name_from_base(self.base_job_name)

    def _prepare_for_training(self, job_name=None):
        """Set any values in the estimator that need to be set before training.

        Args:
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        self._current_job_name = self._get_or_create_name(job_name)

        # if output_path was specified we use it otherwise initialize here.
        # For Local Mode with local_code=True we don't need an explicit output_path
        if self.output_path is None:
            local_code = get_config_value("local.local_code", self.sagemaker_session.config)
            if self.sagemaker_session.local_mode and local_code:
                self.output_path = ""
            else:
                self.output_path = "s3://{}/".format(self.sagemaker_session.default_bucket())

        self._prepare_rules()
        self._prepare_debugger_for_training()
        self._prepare_profiler_for_training()

    def _prepare_rules(self):
        """Rules list includes both debugger and profiler rules.

        Customer can explicitly disable any rule by setting rules to an empty list.
        """
        self.debugger_rules = []
        self.profiler_rules = []
        if self.rules is not None:
            for rule in self.rules:
                if isinstance(rule, Rule):
                    self.debugger_rules.append(rule)
                elif isinstance(rule, ProfilerRule):
                    self.profiler_rules.append(rule)
                else:
                    raise RuntimeError(
                        "Rules list can only contain sagemaker.debugger.Rule "
                        + "and sagemaker.debugger.ProfilerRule"
                    )

    def _prepare_debugger_for_training(self):
        """Prepare debugger rules and debugger configs for training."""
        if self.debugger_rules and self.debugger_hook_config is None:
            self.debugger_hook_config = DebuggerHookConfig(s3_output_path=self.output_path)
        # If debugger_hook_config was provided without an S3 URI, default it for the customer.
        if self.debugger_hook_config and not self.debugger_hook_config.s3_output_path:
            self.debugger_hook_config.s3_output_path = self.output_path
        self.debugger_rule_configs = self._prepare_debugger_rules()
        self._prepare_collection_configs()

    def _prepare_debugger_rules(self):
        """Set any necessary values in debugger rules, if they are provided."""
        debugger_rule_configs = []
        if self.debugger_rules:
            for rule in self.debugger_rules:
                self._set_default_rule_config(rule)
                self._set_source_s3_uri(rule)
                rule.prepare_actions(self._current_job_name)
                debugger_rule_configs.append(rule.to_debugger_rule_config_dict())
        return debugger_rule_configs

    def _prepare_collection_configs(self):
        """De-duplicate configurations and save them in the debugger hook configuration."""
        # Create a set to de-duplicate CollectionConfigs.
        self.collection_configs = set()
        # Iterate through the debugger rules and add their respective CollectionConfigs to the set.
        if self.debugger_rules:
            for rule in self.debugger_rules:
                self.collection_configs.update(rule.collection_configs)
        # Add the CollectionConfigs from DebuggerHookConfig to the set.
        if self.debugger_hook_config:
            self.collection_configs.update(self.debugger_hook_config.collection_configs or [])

    def _prepare_profiler_for_training(self):
        """Set necessary values and do basic validations in profiler config and profiler rules.

        When user explicitly set rules to an empty list, default profiler rule won't be enabled.
        Default profiler rule will be enabled in supported regions when either:
        1. user doesn't specify any rules, i.e., rules=None; or
        2. user only specify debugger rules, i.e., rules=[Rule.sagemaker(...)]
        """
        if self.disable_profiler:
            if self.profiler_config:
                raise RuntimeError("profiler_config cannot be set when disable_profiler is True.")
            if self.profiler_rules:
                raise RuntimeError("ProfilerRule cannot be set when disable_profiler is True.")
        elif _region_supports_profiler(self.sagemaker_session.boto_region_name):
            if self.profiler_config is None:
                self.profiler_config = ProfilerConfig(s3_output_path=self.output_path)
            if self.rules is None or (self.rules and not self.profiler_rules):
                self.profiler_rules = [get_default_profiler_rule()]

        if self.profiler_config and not self.profiler_config.s3_output_path:
            self.profiler_config.s3_output_path = self.output_path

        self.profiler_rule_configs = self._prepare_profiler_rules()

    def _prepare_profiler_rules(self):
        """Set any necessary values in profiler rules, if they are provided."""
        profiler_rule_configs = []
        if self.profiler_rules:
            for rule in self.profiler_rules:
                self._set_default_rule_config(rule)
                self._set_source_s3_uri(rule)
                profiler_rule_configs.append(rule.to_profiler_rule_config_dict())
        return profiler_rule_configs

    def _set_default_rule_config(self, rule):
        """Set default rule configurations.

        Args:
            rule (:class:`~sagemaker.debugger.RuleBase`): Any rule object that derives from RuleBase
        """
        if rule.image_uri == "DEFAULT_RULE_EVALUATOR_IMAGE":
            rule.image_uri = get_rule_container_image_uri(self.sagemaker_session.boto_region_name)
            rule.instance_type = None
            rule.volume_size_in_gb = None

    def _set_source_s3_uri(self, rule):
        """Set updated source S3 uri when specified.

        Args:
            rule (:class:`~sagemaker.debugger.RuleBase`): Any rule object that derives from RuleBase
        """
        if "source_s3_uri" in (rule.rule_parameters or {}):
            parse_result = urlparse(rule.rule_parameters["source_s3_uri"])
            if parse_result.scheme != "s3":
                desired_s3_uri = os.path.join(
                    "s3://",
                    self.sagemaker_session.default_bucket(),
                    rule.name,
                    str(uuid.uuid4()),
                )
                s3_uri = S3Uploader.upload(
                    local_path=rule.rule_parameters["source_s3_uri"],
                    desired_s3_uri=desired_s3_uri,
                    sagemaker_session=self.sagemaker_session,
                )
                rule.rule_parameters["source_s3_uri"] = s3_uri

    def latest_job_debugger_artifacts_path(self):
        """Gets the path to the DebuggerHookConfig output artifacts.

        Returns:
            str: An S3 path to the output artifacts.
        """
        self._ensure_latest_training_job(
            error_message="""Cannot get the Debugger artifacts path.
        The Estimator is not associated with a training job."""
        )
        if self.debugger_hook_config is not None:
            return os.path.join(
                self.debugger_hook_config.s3_output_path,
                self.latest_training_job.name,
                "debug-output",
            )
        return None

    def latest_job_tensorboard_artifacts_path(self):
        """Gets the path to the TensorBoardOutputConfig output artifacts.

        Returns:
            str: An S3 path to the output artifacts.
        """
        self._ensure_latest_training_job(
            error_message="""Cannot get the TensorBoard artifacts path.
        The Estimator is not associated with a training job."""
        )
        if self.debugger_hook_config is not None:
            return os.path.join(
                self.tensorboard_output_config.s3_output_path,
                self.latest_training_job.name,
                "tensorboard-output",
            )
        return None

    def latest_job_profiler_artifacts_path(self):
        """Gets the path to the profiling output artifacts.

        Returns:
            str: An S3 path to the output artifacts.
        """
        self._ensure_latest_training_job(
            error_message="""Cannot get the profiling output artifacts path.
        The Estimator is not associated with a training job."""
        )
        if self.profiler_config is not None:
            return os.path.join(
                self.profiler_config.s3_output_path,
                self.latest_training_job.name,
                "profiler-output",
            )
        return None

    def fit(self, inputs=None, wait=True, logs="All", job_name=None, experiment_config=None):
        """Train a model using the input training dataset.

        The API calls the Amazon SageMaker CreateTrainingJob API to start
        model training. The API uses configuration you provided to create the
        estimator and the specified input training data to send the
        CreatingTrainingJob request to Amazon SageMaker.

        This is a synchronous operation. After the model training
        successfully completes, you can call the ``deploy()`` method to host the
        model using the Amazon SageMaker hosting services.

        Args:
            inputs (str or dict or sagemaker.inputs.TrainingInput): Information
                about the training data. This can be one of three types:

                * (str) the S3 location where training data is saved, or a file:// path in
                    local mode.
                * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) If using multiple
                    channels for training data, you can specify a dict mapping channel names to
                    strings or :func:`~sagemaker.inputs.TrainingInput` objects.
                * (sagemaker.inputs.TrainingInput) - channel configuration for S3 data sources
                    that can provide additional information as well as the path to the training
                    dataset.
                    See :func:`sagemaker.inputs.TrainingInput` for full details.
                * (sagemaker.session.FileSystemInput) - channel configuration for
                    a file system data source that can provide additional information as well as
                    the path to the training dataset.

            wait (bool): Whether the call should wait until the job completes (default: True).
            logs ([str]): A list of strings specifying which logs to print. Acceptable
                strings are "All", "None", "Training", or "Rules". To maintain backwards
                compatibility, boolean values are also accepted and converted to strings.
                Only meaningful when wait is True.
            job_name (str): Training job name. If not specified, the estimator generates
                a default job name based on the training image name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Dictionary contains three optional keys,
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.

        """
        self._prepare_for_training(job_name=job_name)

        self.latest_training_job = _TrainingJob.start_new(self, inputs, experiment_config)
        self.jobs.append(self.latest_training_job)
        if wait:
            self.latest_training_job.wait(logs=logs)

    def _compilation_job_name(self):
        """Placeholder docstring"""
        base_name = self.base_job_name or base_name_from_image(self.training_image_uri())
        return name_from_base("compilation-" + base_name)

    def compile_model(
        self,
        target_instance_family,
        input_shape,
        output_path,
        framework=None,
        framework_version=None,
        compile_max_run=15 * 60,
        tags=None,
        target_platform_os=None,
        target_platform_arch=None,
        target_platform_accelerator=None,
        compiler_options=None,
        **kwargs,
    ):
        """Compile a Neo model using the input model.

        Args:
            target_instance_family (str): Identifies the device that you want to
                run your model after compilation, for example: ml_c5. For allowed
                strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
            input_shape (dict): Specifies the name and shape of the expected
                inputs for your trained model in json dictionary form, for
                example: {'data':[1,3,1024,1024]}, or {'var1': [1,1,28,28],
                'var2':[1,1,28,28]}
            output_path (str): Specifies where to store the compiled model
            framework (str): The framework that is used to train the original
                model. Allowed values: 'mxnet', 'tensorflow', 'keras', 'pytorch',
                'onnx', 'xgboost'
            framework_version (str): The version of the framework
            compile_max_run (int): Timeout in seconds for compilation (default:
                3 * 60). After this amount of time Amazon SageMaker Neo
                terminates the compilation job regardless of its current status.
            tags (list[dict]): List of tags for labeling a compilation job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            target_platform_os (str): Target Platform OS, for example: 'LINUX'.
                For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family.
            target_platform_arch (str): Target Platform Architecture, for example: 'X86_64'.
                For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family.
            target_platform_accelerator (str, optional): Target Platform Accelerator,
                for example: 'NVIDIA'. For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family.
            compiler_options (dict, optional): Additional parameters for compiler.
                Compiler Options are TargetPlatform / target_instance_family specific. See
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html for details.
            **kwargs: Passed to invocation of ``create_model()``.
                Implementations may customize ``create_model()`` to accept
                ``**kwargs`` to customize model creation during deploy. For
                more, see the implementation docs.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """
        if framework and framework not in NEO_ALLOWED_FRAMEWORKS:
            raise ValueError(
                "Please use valid framework, allowed values: {}".format(NEO_ALLOWED_FRAMEWORKS)
            )

        if (framework is None) != (framework_version is None):
            raise ValueError("You should provide framework and framework_version at the same time.")

        model = self.create_model(**kwargs)

        self._compiled_models[target_instance_family] = model.compile(
            target_instance_family,
            input_shape,
            output_path,
            self.role,
            tags,
            self._compilation_job_name(),
            compile_max_run,
            framework=framework,
            framework_version=framework_version,
            target_platform_os=target_platform_os,
            target_platform_arch=target_platform_arch,
            target_platform_accelerator=target_platform_accelerator,
            compiler_options=compiler_options,
        )
        return self._compiled_models[target_instance_family]

    @classmethod
    def attach(cls, training_job_name, sagemaker_session=None, model_channel_name="model"):
        """Attach to an existing training job.

        Create an Estimator bound to an existing training job, each subclass
        is responsible to implement
        ``_prepare_init_params_from_job_description()`` as this method delegates
        the actual conversion of a training job description to the arguments
        that the class constructor expects. After attaching, if the training job
        has a Complete status, it can be ``deploy()`` ed to create a SageMaker
        Endpoint and return a ``Predictor``.

        If the training job is in progress, attach will block until the training job
        completes, but logs of the training job will not display. To see the logs
        content, please call ``logs()``

        Examples:
            >>> my_estimator.fit(wait=False)
            >>> training_job_name = my_estimator.latest_training_job.name
            Later on:
            >>> attached_estimator = Estimator.attach(training_job_name)
            >>> attached_estimator.logs()
            >>> attached_estimator.deploy()

        Args:
            training_job_name (str): The name of the training job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded (default: 'model'). If no channel
                with the same name exists in the training job, this option will
                be ignored.

        Returns:
            Instance of the calling ``Estimator`` Class with the attached
            training job.
        """
        sagemaker_session = sagemaker_session or Session()

        job_details = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        init_params = cls._prepare_init_params_from_job_description(job_details, model_channel_name)
        tags = sagemaker_session.sagemaker_client.list_tags(
            ResourceArn=job_details["TrainingJobArn"]
        )["Tags"]
        init_params.update(tags=tags)

        estimator = cls(sagemaker_session=sagemaker_session, **init_params)
        estimator.latest_training_job = _TrainingJob(
            sagemaker_session=sagemaker_session, job_name=training_job_name
        )
        estimator._current_job_name = estimator.latest_training_job.name
        estimator.latest_training_job.wait(logs="None")
        return estimator

    def logs(self):
        """Display the logs for Estimator's training job.

        If the output is a tty or a Jupyter cell, it will be color-coded based
        on which instance the log entry is from.
        """
        self.sagemaker_session.logs_for_job(self.latest_training_job.name, wait=True)

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        serializer=None,
        deserializer=None,
        accelerator_type=None,
        endpoint_name=None,
        use_compiled_model=False,
        wait=True,
        model_name=None,
        kms_key=None,
        data_capture_config=None,
        tags=None,
        **kwargs,
    ):
        """Deploy the trained model to an Amazon SageMaker endpoint.

         And then return ``sagemaker.Predictor`` object.

        More information:
        http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html

        Args:
            initial_instance_count (int): Minimum number of EC2 instances to
                deploy to an endpoint for prediction.
            instance_type (str): Type of EC2 instance to deploy to an endpoint
                for prediction, for example, 'ml.c4.xlarge'.
            serializer (:class:`~sagemaker.serializers.BaseSerializer`): A
                serializer object, used to encode data for an inference endpoint
                (default: None). If ``serializer`` is not None, then
                ``serializer`` will override the default serializer. The
                default serializer is set by the ``predictor_cls``.
            deserializer (:class:`~sagemaker.deserializers.BaseDeserializer`): A
                deserializer object, used to decode data from an inference
                endpoint (default: None). If ``deserializer`` is not None, then
                ``deserializer`` will override the default deserializer. The
                default deserializer is set by the ``predictor_cls``.
            accelerator_type (str): Type of Elastic Inference accelerator to
                attach to an endpoint for model loading and inference, for
                example, 'ml.eia1.medium'. If not specified, no Elastic
                Inference accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            endpoint_name (str): Name to use for creating an Amazon SageMaker
                endpoint. If not specified, the name of the training job is
                used.
            use_compiled_model (bool): Flag to select whether to use compiled
                (optimized) model. Default: False.
            wait (bool): Whether the call should wait until the deployment of
                model completes (default: True).
            model_name (str): Name to use for creating an Amazon SageMaker
                model. If not specified, the estimator generates a default job name
                based on the training image name and current timestamp.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint.
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.
            tags(List[dict[str, str]]): Optional. The list of tags to attach to this specific
                endpoint. Example:
                >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
                For more information about tags, see
                https://boto3.amazonaws.com/v1/documentation\
                /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
            **kwargs: Passed to invocation of ``create_model()``.
                Implementations may customize ``create_model()`` to accept
                ``**kwargs`` to customize model creation during deploy.
                For more, see the implementation docs.

        Returns:
            sagemaker.predictor.Predictor: A predictor that provides a ``predict()`` method,
                which can be used to send requests to the Amazon SageMaker
                endpoint and obtain inferences.
        """
        removed_kwargs("update_endpoint", kwargs)
        self._ensure_latest_training_job()
        self._ensure_base_job_name()
        default_name = name_from_base(self.base_job_name)
        endpoint_name = endpoint_name or default_name
        model_name = model_name or default_name

        self.deploy_instance_type = instance_type
        if use_compiled_model:
            family = "_".join(instance_type.split(".")[:-1])
            if family not in self._compiled_models:
                raise ValueError(
                    "No compiled model for {}. "
                    "Please compile one with compile_model before deploying.".format(family)
                )
            model = self._compiled_models[family]
        else:
            kwargs["model_kms_key"] = self.output_kms_key
            model = self.create_model(**kwargs)

        model.name = model_name

        return model.deploy(
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            serializer=serializer,
            deserializer=deserializer,
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
            tags=tags or self.tags,
            wait=wait,
            kms_key=kms_key,
            data_capture_config=data_capture_config,
        )

    def register(
        self,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        image_uri=None,
        model_package_name=None,
        model_package_group_name=None,
        model_metrics=None,
        metadata_properties=None,
        marketplace_cert=False,
        approval_status=None,
        description=None,
        compile_model_family=None,
        model_name=None,
        **kwargs,
    ):
        """Creates a model package for creating SageMaker models or listing on Marketplace.

        Args:
            content_types (list): The supported MIME types for the input data.
            response_types (list): The supported MIME types for the output data.
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time.
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed.
            image_uri (str): The container image uri for Model Package, if not specified,
                Estimator's training container image will be used (default: None).
            model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
                using `model_package_name` makes the Model Package un-versioned (default: None).
            model_package_group_name (str): Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties (default: None).
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace (default: False).
            approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
                or "PendingManualApproval" (default: "PendingManualApproval").
            description (str): Model Package description (default: None).
            compile_model_family (str): Instance family for compiled model, if specified, a compiled
                model will be used (default: None).
            model_name (str): User defined model name (default: None).
            **kwargs: Passed to invocation of ``create_model()``. Implementations may customize
                ``create_model()`` to accept ``**kwargs`` to customize model creation during
                deploy. For more, see the implementation docs.

        Returns:
            str: A string of SageMaker Model Package ARN.
        """
        default_name = name_from_base(self.base_job_name)
        model_name = model_name or default_name
        if compile_model_family is not None:
            model = self._compiled_models[compile_model_family]
        else:
            model = self.create_model(image_uri=image_uri, **kwargs)
        model.name = model_name
        return model.register(
            content_types,
            response_types,
            inference_instances,
            transform_instances,
            model_package_name,
            model_package_group_name,
            image_uri,
            model_metrics,
            metadata_properties,
            marketplace_cert,
            approval_status,
            description,
        )

    @property
    def model_data(self):
        """str: The model location in S3. Only set if Estimator has been ``fit()``."""
        if self.latest_training_job is not None:
            model_uri = self.sagemaker_session.sagemaker_client.describe_training_job(
                TrainingJobName=self.latest_training_job.name
            )["ModelArtifacts"]["S3ModelArtifacts"]
        else:
            logger.warning(
                "No finished training job found associated with this estimator. Please make sure "
                "this estimator is only used for building workflow config"
            )
            model_uri = os.path.join(
                self.output_path, self._current_job_name, "output", "model.tar.gz"
            )

        return model_uri

    @abstractmethod
    def create_model(self, **kwargs):
        """Create a SageMaker ``Model`` object that can be deployed to an ``Endpoint``.

        Args:
            **kwargs: Keyword arguments used by the implemented method for
                creating the ``Model``.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = dict()

        init_params["role"] = job_details["RoleArn"]
        init_params["instance_count"] = job_details["ResourceConfig"]["InstanceCount"]
        init_params["instance_type"] = job_details["ResourceConfig"]["InstanceType"]
        init_params["volume_size"] = job_details["ResourceConfig"]["VolumeSizeInGB"]
        init_params["max_run"] = job_details["StoppingCondition"]["MaxRuntimeInSeconds"]
        init_params["input_mode"] = job_details["AlgorithmSpecification"]["TrainingInputMode"]
        init_params["base_job_name"] = base_from_name(job_details["TrainingJobName"])
        init_params["output_path"] = job_details["OutputDataConfig"]["S3OutputPath"]
        init_params["output_kms_key"] = job_details["OutputDataConfig"]["KmsKeyId"]
        if "EnableNetworkIsolation" in job_details:
            init_params["enable_network_isolation"] = job_details["EnableNetworkIsolation"]

        has_hps = "HyperParameters" in job_details
        init_params["hyperparameters"] = job_details["HyperParameters"] if has_hps else {}

        if "AlgorithmName" in job_details["AlgorithmSpecification"]:
            init_params["algorithm_arn"] = job_details["AlgorithmSpecification"]["AlgorithmName"]
        elif "TrainingImage" in job_details["AlgorithmSpecification"]:
            init_params["image_uri"] = job_details["AlgorithmSpecification"]["TrainingImage"]
        else:
            raise RuntimeError(
                "Invalid AlgorithmSpecification. Either TrainingImage or "
                "AlgorithmName is expected. None was found."
            )

        if "MetricDefinitons" in job_details["AlgorithmSpecification"]:
            init_params["metric_definitions"] = job_details["AlgorithmSpecification"][
                "MetricsDefinition"
            ]

        if "EnableInterContainerTrafficEncryption" in job_details:
            init_params["encrypt_inter_container_traffic"] = job_details[
                "EnableInterContainerTrafficEncryption"
            ]

        subnets, security_group_ids = vpc_utils.from_dict(job_details.get(vpc_utils.VPC_CONFIG_KEY))
        if subnets:
            init_params["subnets"] = subnets
        if security_group_ids:
            init_params["security_group_ids"] = security_group_ids

        if "InputDataConfig" in job_details and model_channel_name:
            for channel in job_details["InputDataConfig"]:
                if channel["ChannelName"] == model_channel_name:
                    init_params["model_channel_name"] = model_channel_name
                    init_params["model_uri"] = channel["DataSource"]["S3DataSource"]["S3Uri"]
                    break

        if job_details.get("EnableManagedSpotTraining", False):
            init_params["use_spot_instances"] = True
            max_wait = job_details.get("StoppingCondition", {}).get("MaxWaitTimeInSeconds")
            if max_wait:
                init_params["max_wait"] = max_wait

        if job_details.get("RetryStrategy", False):
            init_params["max_retry_attempts"] = job_details.get("RetryStrategy", {}).get(
                "MaximumRetryAttempts"
            )
            max_wait = job_details.get("StoppingCondition", {}).get("MaxWaitTimeInSeconds")
            if max_wait:
                init_params["max_wait"] = max_wait
        return init_params

    def transformer(
        self,
        instance_count,
        instance_type,
        strategy=None,
        assemble_with=None,
        output_path=None,
        output_kms_key=None,
        accept=None,
        env=None,
        max_concurrent_transforms=None,
        max_payload=None,
        tags=None,
        role=None,
        volume_kms_key=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        enable_network_isolation=None,
        model_name=None,
    ):
        """Return a ``Transformer`` that uses a SageMaker Model based on the training job.

        It reuses the SageMaker Session and base job name used by
        the Estimator.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in
                a single request (default: None). Valid values: 'MultiRecord'
                and 'SingleRecord'.
            assemble_with (str): How the output is assembled (default: None).
                Valid values: 'Line' or 'None'.
            output_path (str): S3 location for saving the transform result. If
                not specified, results are stored to a default bucket.
            output_kms_key (str): Optional. KMS key ID for encrypting the
                transform output (default: None).
            accept (str): The accept header passed by the client to
                the inference endpoint. If it is supported by the endpoint,
                it will be the format of the batch transform output.
            env (dict): Environment variables to be set for use during the
                transform job (default: None).
            max_concurrent_transforms (int): The maximum number of HTTP requests
                to be made to each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP
                request to the container in MB.
            tags (list[dict]): List of tags for labeling a transform job. If
                none specified, then the tags used for the training job are used
                for the transform job.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            volume_kms_key (str): Optional. KMS key ID for encrypting the volume
                attached to the ML compute instance (default: None).
            vpc_config_override (dict[str, list[str]]): Optional override for the
                VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.

                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode. Network isolation mode restricts
                the container access to outside networks (such as the internet).
                The container does not make any inbound or outbound network
                calls. If True, a channel named "code" will be created for any
                user entry script for inference. Also known as Internet-free mode.
                If not specified, this setting is taken from the estimator's
                current configuration.
            model_name (str): Name to use for creating an Amazon SageMaker
                model. If not specified, the estimator generates a default job name
                based on the training image name and current timestamp.
        """
        tags = tags or self.tags
        model_name = self._get_or_create_name(model_name)

        if self.latest_training_job is None:
            logger.warning(
                "No finished training job found associated with this estimator. Please make sure "
                "this estimator is only used for building workflow config"
            )
        else:
            if enable_network_isolation is None:
                enable_network_isolation = self.enable_network_isolation()

            model = self.create_model(
                vpc_config_override=vpc_config_override,
                model_kms_key=self.output_kms_key,
                enable_network_isolation=enable_network_isolation,
            )

            # not all create_model() implementations have the same kwargs
            model.name = model_name
            if role is not None:
                model.role = role

            model._create_sagemaker_model(instance_type, tags=tags)

        return Transformer(
            model_name,
            instance_count,
            instance_type,
            strategy=strategy,
            assemble_with=assemble_with,
            output_path=output_path,
            output_kms_key=output_kms_key,
            accept=accept,
            max_concurrent_transforms=max_concurrent_transforms,
            max_payload=max_payload,
            env=env,
            tags=tags,
            base_transform_job_name=self.base_job_name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=self.sagemaker_session,
        )

    @property
    def training_job_analytics(self):
        """Return a ``TrainingJobAnalytics`` object for the current training job."""
        if self._current_job_name is None:
            raise ValueError("Estimator is not associated with a TrainingJob")
        return TrainingJobAnalytics(
            self._current_job_name, sagemaker_session=self.sagemaker_session
        )

    def get_vpc_config(self, vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT):
        """Returns VpcConfig dict either from this Estimator's subnets and security groups.

        Or else validate and return an optional override value.

        Args:
            vpc_config_override:
        """
        if vpc_config_override is vpc_utils.VPC_CONFIG_DEFAULT:
            return vpc_utils.to_dict(self.subnets, self.security_group_ids)
        return vpc_utils.sanitize(vpc_config_override)

    def _ensure_latest_training_job(
        self, error_message="Estimator is not associated with a training job"
    ):
        """Placeholder docstring"""
        if self.latest_training_job is None:
            raise ValueError(error_message)

    delete_endpoint = removed_function("delete_endpoint")

    def enable_default_profiling(self):
        """Update training job to enable Debugger monitoring.

        This method enables Debugger monitoring with
        the default ``profiler_config`` parameter to collect system
        metrics and the default built-in ``profiler_report`` rule.
        Framework metrics won't be saved.
        To update training job to emit framework metrics, you can use
        :class:`~sagemaker.estimator.Estimator.update_profiler`
        method and specify the framework metrics you want to enable.

        This method is callable when the training job is in progress while
        Debugger monitoring is disabled.
        """
        self._ensure_latest_training_job()

        training_job_details = self.latest_training_job.describe()

        if training_job_details.get("ProfilingStatus") == "Enabled":
            raise ValueError(
                "Debugger monitoring is already enabled. To update the profiler_config parameter "
                "and the Debugger profiling rules, please use the update_profiler function."
            )

        if "ProfilerConfig" in training_job_details and training_job_details["ProfilerConfig"].get(
            "S3OutputPath"
        ):
            self.profiler_config = ProfilerConfig(
                s3_output_path=training_job_details["ProfilerConfig"]["S3OutputPath"]
            )
        else:
            self.profiler_config = ProfilerConfig(s3_output_path=self.output_path)

        self.profiler_rules = [get_default_profiler_rule()]
        self.profiler_rule_configs = self._prepare_profiler_rules()

        _TrainingJob.update(
            self, self.profiler_rule_configs, self.profiler_config._to_request_dict()
        )

    def disable_profiling(self):
        """Update the current training job in progress to disable profiling.

        Debugger stops collecting the system and framework metrics
        and turns off the Debugger built-in monitoring and profiling rules.

        """
        self._ensure_latest_training_job()

        training_job_details = self.latest_training_job.describe()

        if training_job_details.get("ProfilingStatus") == "Disabled":
            raise ValueError("Profiler is already disabled.")

        _TrainingJob.update(
            self, profiler_config=ProfilerConfig._to_profiler_disabled_request_dict()
        )

    def update_profiler(
        self,
        rules=None,
        system_monitor_interval_millis=None,
        s3_output_path=None,
        framework_profile_params=None,
        disable_framework_metrics=False,
    ):
        """Update training jobs to enable profiling.

        This method updates the ``profiler_config`` parameter
        and initiates Debugger built-in rules for profiling.

        Args:
            rules (list[:class:`~sagemaker.debugger.ProfilerRule`]): A list of
                :class:`~sagemaker.debugger.ProfilerRule` objects to define
                rules for continuous analysis with SageMaker Debugger. Currently, you can
                only add new profiler rules during the training job. (default: ``None``)
            s3_output_path (str): The location in S3 to store the output. If profiler is enabled
                once, s3_output_path cannot be changed. (default: ``None``)
            system_monitor_interval_millis (int): How often profiling system metrics are
                collected; Unit: Milliseconds (default: ``None``)
            framework_profile_params (:class:`~sagemaker.debugger.FrameworkProfile`):
                A parameter object for framework metrics profiling. Configure it using
                the :class:`~sagemaker.debugger.FrameworkProfile` class.
                To use the default framework profile parameters, pass ``FrameworkProfile()``.
                For more information about the default values,
                see :class:`~sagemaker.debugger.FrameworkProfile`. (default: ``None``)
            disable_framework_metrics (bool): Specify whether to disable all the framework metrics.
                This won't update system metrics and the Debugger built-in rules for monitoring.
                To stop both monitoring and profiling,
                use the :class:`~sagemaker.estimator.Estimator.desable_profiling`
                method. (default: ``False``)

        .. attention::

            Updating the profiling configuration for TensorFlow dataloader profiling
            is currently not available. If you started a TensorFlow training job only with
            monitoring and want to enable profiling while the training job is running,
            the dataloader profiling cannot be updated.

        """
        self._ensure_latest_training_job()

        if (
            not rules
            and not system_monitor_interval_millis
            and not s3_output_path
            and not framework_profile_params
            and not disable_framework_metrics
        ):
            raise ValueError("Please provide profiler config or profiler rule to be updated.")

        if disable_framework_metrics and framework_profile_params:
            raise ValueError(
                "framework_profile_params cannot be set when disable_framework_metrics is True"
            )

        profiler_config_request_dict = None
        profiler_rule_configs = None

        if rules:
            for rule in rules:
                if not isinstance(rule, ProfilerRule):
                    raise ValueError("Please provide ProfilerRule to be updated.")
            self.profiler_rules = rules
            profiler_rule_configs = self._prepare_profiler_rules()

        if disable_framework_metrics:
            empty_framework_profile_param = FrameworkProfile()
            empty_framework_profile_param.profiling_parameters = {}
            self.profiler_config = ProfilerConfig(
                s3_output_path=s3_output_path,
                system_monitor_interval_millis=system_monitor_interval_millis,
                framework_profile_params=empty_framework_profile_param,
            )
        else:
            self.profiler_config = ProfilerConfig(
                s3_output_path=s3_output_path,
                system_monitor_interval_millis=system_monitor_interval_millis,
                framework_profile_params=framework_profile_params,
            )

        profiler_config_request_dict = self.profiler_config._to_request_dict()

        _TrainingJob.update(self, profiler_rule_configs, profiler_config_request_dict)


class _TrainingJob(_Job):
    """Placeholder docstring"""

    @classmethod
    def start_new(cls, estimator, inputs, experiment_config):
        """Create a new Amazon SageMaker training job from the estimator.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): Estimator object
                created by the user.
            inputs (str): Parameters used when called
                :meth:`~sagemaker.estimator.EstimatorBase.fit`.
            experiment_config (dict[str, str]): Experiment management configuration used when called
                :meth:`~sagemaker.estimator.EstimatorBase.fit`.  Dictionary contains
                three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.

        Returns:
            sagemaker.estimator._TrainingJob: Constructed object that captures
            all information about the started training job.
        """
        train_args = cls._get_train_args(estimator, inputs, experiment_config)
        estimator.sagemaker_session.train(**train_args)

        return cls(estimator.sagemaker_session, estimator._current_job_name)

    @classmethod
    def _get_train_args(cls, estimator, inputs, experiment_config):
        """Constructs a dict of arguments for an Amazon SageMaker training job from the estimator.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): Estimator object
                created by the user.
            inputs (str): Parameters used when called
                :meth:`~sagemaker.estimator.EstimatorBase.fit`.
            experiment_config (dict[str, str]): Experiment management configuration used when called
                :meth:`~sagemaker.estimator.EstimatorBase.fit`.  Dictionary contains
                three optional keys, 'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.

        Returns:
            Dict: dict for `sagemaker.session.Session.train` method
        """

        local_mode = estimator.sagemaker_session.local_mode
        model_uri = estimator.model_uri

        # Allow file:// input only in local mode
        if cls._is_local_channel(inputs) or cls._is_local_channel(model_uri):
            if not local_mode:
                raise ValueError(
                    "File URIs are supported in local mode only. Please use a S3 URI instead."
                )

        config = _Job._load_config(inputs, estimator)

        current_hyperparameters = estimator.hyperparameters()
        if current_hyperparameters is not None:
            hyperparameters = {
                str(k): (v if isinstance(v, (Parameter, Expression, Properties)) else str(v))
                for (k, v) in current_hyperparameters.items()
            }

        train_args = config.copy()
        train_args["input_mode"] = estimator.input_mode
        train_args["job_name"] = estimator._current_job_name
        train_args["hyperparameters"] = hyperparameters
        train_args["tags"] = estimator.tags
        train_args["metric_definitions"] = estimator.metric_definitions
        train_args["experiment_config"] = experiment_config
        train_args["environment"] = estimator.environment

        if isinstance(inputs, TrainingInput):
            if "InputMode" in inputs.config:
                logger.debug(
                    "Selecting TrainingInput's input_mode (%s) for TrainingInputMode.",
                    inputs.config["InputMode"],
                )
                train_args["input_mode"] = inputs.config["InputMode"]

        if estimator.enable_network_isolation():
            train_args["enable_network_isolation"] = True

        if estimator.max_retry_attempts is not None:
            train_args["retry_strategy"] = {"MaximumRetryAttempts": estimator.max_retry_attempts}
        else:
            train_args["retry_strategy"] = None

        if estimator.encrypt_inter_container_traffic:
            train_args["encrypt_inter_container_traffic"] = True

        if isinstance(estimator, sagemaker.algorithm.AlgorithmEstimator):
            train_args["algorithm_arn"] = estimator.algorithm_arn
        else:
            train_args["image_uri"] = estimator.training_image_uri()

        if estimator.debugger_rule_configs:
            train_args["debugger_rule_configs"] = estimator.debugger_rule_configs

        if estimator.debugger_hook_config:
            estimator.debugger_hook_config.collection_configs = estimator.collection_configs
            train_args["debugger_hook_config"] = estimator.debugger_hook_config._to_request_dict()

        if estimator.tensorboard_output_config:
            train_args[
                "tensorboard_output_config"
            ] = estimator.tensorboard_output_config._to_request_dict()

        cls._add_spot_checkpoint_args(local_mode, estimator, train_args)

        if estimator.enable_sagemaker_metrics is not None:
            train_args["enable_sagemaker_metrics"] = estimator.enable_sagemaker_metrics

        if estimator.profiler_rule_configs:
            train_args["profiler_rule_configs"] = estimator.profiler_rule_configs

        if estimator.profiler_config:
            train_args["profiler_config"] = estimator.profiler_config._to_request_dict()

        return train_args

    @classmethod
    def _add_spot_checkpoint_args(cls, local_mode, estimator, train_args):
        """Placeholder docstring"""
        if estimator.use_spot_instances:
            if local_mode:
                raise ValueError("Spot training is not supported in local mode.")
            train_args["use_spot_instances"] = True

        if estimator.checkpoint_s3_uri:
            if local_mode:
                raise ValueError("Setting checkpoint_s3_uri is not supported in local mode.")
            train_args["checkpoint_s3_uri"] = estimator.checkpoint_s3_uri

        if estimator.checkpoint_local_path:
            if local_mode:
                raise ValueError("Setting checkpoint_local_path is not supported in local mode.")
            train_args["checkpoint_local_path"] = estimator.checkpoint_local_path

    @classmethod
    def _is_local_channel(cls, input_uri):
        """Placeholder docstring"""
        return isinstance(input_uri, string_types) and input_uri.startswith("file://")

    @classmethod
    def update(cls, estimator, profiler_rule_configs=None, profiler_config=None):
        """Update a running Amazon SageMaker training job.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): Estimator object created by the user.
            profiler_rule_configs (list): List of profiler rule configurations to be
                updated in the training job. (default: ``None``).
            profiler_config (dict): Configuration for how profiling information is emitted with
                SageMaker Debugger. (default: ``None``).

        Returns:
            sagemaker.estimator._TrainingJob: Constructed object that captures
            all information about the updated training job.
        """
        update_args = cls._get_update_args(estimator, profiler_rule_configs, profiler_config)
        estimator.sagemaker_session.update_training_job(**update_args)

        return estimator.latest_training_job

    @classmethod
    def _get_update_args(cls, estimator, profiler_rule_configs, profiler_config):
        """Constructs a dict of arguments for updating an Amazon SageMaker training job.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): Estimator object
                created by the user.
            profiler_rule_configs (list): List of profiler rule configurations to be
                updated in the training job. (default: ``None``).
            profiler_config (dict): Configuration for how profiling information is emitted with
                SageMaker Debugger. (default: ``None``).

        Returns:
            Dict: dict for `sagemaker.session.Session.update_training_job` method
        """
        update_args = {"job_name": estimator.latest_training_job.name}
        update_args.update(build_dict("profiler_rule_configs", profiler_rule_configs))
        update_args.update(build_dict("profiler_config", profiler_config))

        return update_args

    def wait(self, logs="All"):
        """Placeholder docstring.

        Args:
            logs ([str]): A list of strings specifying which logs to print. Acceptable
                strings are "All", "None", "Training", or "Rules". To maintain backwards
                compatibility, boolean values are also accepted and converted to strings.
        """
        # Convert boolean values of logs to strings.
        log_string_map = {True: "All", False: "None"}
        if isinstance(logs, bool):
            logs = log_string_map[logs]
        # If logs are requested, call logs_for_jobs.
        if logs != "None":
            self.sagemaker_session.logs_for_job(self.job_name, wait=True, log_type=logs)
        else:
            self.sagemaker_session.wait_for_job(self.job_name)

    def describe(self):
        """Returns a response from the DescribeTrainingJob API call."""
        return self.sagemaker_session.describe_training_job(self.job_name)

    def rule_job_summary(self):
        """Calls describe_training_job and returns two dictionaries.

        Returns:
            list[dict]: A list of DebugRuleEvaluationStatuses and ProfilerRuleEvaluationStatuses
                dictionary.
        """
        job_summary = self.describe()
        rule_eval_statuses = job_summary.get("DebugRuleEvaluationStatuses") or []
        rule_eval_statuses.extend(job_summary.get("ProfilerRuleEvaluationStatuses") or [])

        return rule_eval_statuses

    def stop(self):
        """Stops the training job."""
        self.sagemaker_session.stop_training_job(self.name)


class Estimator(EstimatorBase):
    """A generic Estimator to train using any supplied algorithm.

    This class is designed for use with algorithms that don't have their own, custom class.
    """

    def __init__(
        self,
        image_uri,
        role,
        instance_count=None,
        instance_type=None,
        volume_size=30,
        volume_kms_key=None,
        max_run=24 * 60 * 60,
        input_mode="File",
        output_path=None,
        output_kms_key=None,
        base_job_name=None,
        sagemaker_session=None,
        hyperparameters=None,
        tags=None,
        subnets=None,
        security_group_ids=None,
        model_uri=None,
        model_channel_name="model",
        metric_definitions=None,
        encrypt_inter_container_traffic=False,
        use_spot_instances=False,
        max_wait=None,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
        enable_network_isolation=False,
        rules=None,
        debugger_hook_config=None,
        tensorboard_output_config=None,
        enable_sagemaker_metrics=None,
        profiler_config=None,
        disable_profiler=False,
        environment=None,
        max_retry_attempts=None,
        **kwargs,
    ):
        """Initialize an ``Estimator`` instance.

        Args:
            image_uri (str): The container image to use for training.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            instance_count (int): Number of Amazon EC2 instances to use
                for training.
            instance_type (str): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            volume_size (int): Size in GB of the EBS volume to use for
                storing input data during training (default: 30). Must be large
                enough to store training data if File Mode is used (which is the
                default).
            volume_kms_key (str): Optional. KMS key ID for encrypting EBS
                volume attached to the training instance (default: None).
            max_run (int): Timeout in seconds for training (default: 24 *
                60 * 60). After this amount of time Amazon SageMaker terminates
                the job regardless of its current status.
            input_mode (str): The input mode that the algorithm supports
                (default: 'File'). Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the
                  S3 location to a local directory.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the
                  container via a Unix-named pipe.

                This argument can be overriden on a per-channel basis using
                ``sagemaker.inputs.TrainingInput.input_mode``.
            output_path (str): S3 location for saving the training result (model
                artifacts and output files). If not specified, results are
                stored to a default bucket. If the bucket with the specific name
                does not exist, the estimator creates the bucket during the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method execution.
            output_kms_key (str): Optional. KMS key ID for encrypting the
                training output (default: None).
            base_job_name (str): Prefix for training job name when the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method launches.
                If not specified, the estimator generates a default job name,
                based on the training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            hyperparameters (dict): Dictionary containing the hyperparameters to
                initialize this estimator with.
            tags (list[dict]): List of tags for labeling a training job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            subnets (list[str]): List of subnet ids. If not specified training
                job will be created without VPC config.
            security_group_ids (list[str]): List of security group ids. If not
                specified training job will be created without VPC config.
            model_uri (str): URI where a pre-trained model is stored, either
                locally or in S3 (default: None). If specified, the estimator
                will create a channel pointing to the model so the training job
                can download it. This model can be a 'model.tar.gz' from a
                previous training job, or other artifacts coming from a
                different source.

                In local mode, this should point to the path in which the model
                is located and not the file itself, as local Docker containers
                will try to mount the URI as a volume.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization
            model_channel_name (str): Name of the channel where 'model_uri' will
                be downloaded (default: 'model').
            metric_definitions (list[dict]): A list of dictionaries that defines
                the metric(s) used to evaluate the training jobs. Each
                dictionary contains two keys: 'Name' for the name of the metric,
                and 'Regex' for the regular expression used to extract the
                metric from the logs. This should be defined only for jobs that
                don't use an Amazon algorithm.
            encrypt_inter_container_traffic (bool): Specifies whether traffic
                between training containers is encrypted for the training job
                (default: ``False``).
            use_spot_instances (bool): Specifies whether to use SageMaker
                Managed Spot instances for training. If enabled then the
                ``max_wait`` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (default: ``False``).
            max_wait (int): Timeout in seconds waiting for spot training
                instances (default: None). After this amount of time Amazon
                SageMaker will stop waiting for Spot instances to become
                available (default: ``None``).
            checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
                that the algorithm persists (if any) during training. (default:
                ``None``).
            checkpoint_local_path (str): The local path that the algorithm
                writes its checkpoints to. SageMaker will persist all files
                under this path to `checkpoint_s3_uri` continually during
                training. On job startup the reverse happens - data from the
                s3 location is downloaded to this path before the algorithm is
                started. If the path is unset then SageMaker assumes the
                checkpoints will be provided under `/opt/ml/checkpoints/`.
                (default: ``None``).
            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode (default: ``False``). Network
                isolation mode restricts the container access to outside networks
                (such as the Internet). The container does not make any inbound or
                outbound network calls. Also known as Internet-free mode.
            rules (list[:class:`~sagemaker.debugger.RuleBase`]): A list of
                :class:`~sagemaker.debugger.RuleBase` objects used to define
                SageMaker Debugger rules for real-time analysis
                (default: ``None``). For more information,
                see `Continuous analyses through rules
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html
                #continuous-analyses-through-rules)>`_.
            debugger_hook_config (:class:`~sagemaker.debugger.DebuggerHookConfig` or bool):
                Configuration for how debugging information is emitted with
                SageMaker Debugger. If not specified, a default one is created using
                the estimator's ``output_path``, unless the region does not
                support SageMaker Debugger. To disable SageMaker Debugger,
                set this parameter to ``False``. For more information, see
                `Capture real-time debugging data during model training in Amazon SageMaker
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#
                capture-real-time-debugging-data-during-model-training-in-amazon-sagemaker>`_.
            tensorboard_output_config (:class:`~sagemaker.debugger.TensorBoardOutputConfig`):
                Configuration for customizing debugging visualization using TensorBoard
                (default: ``None``). For more information,
                see `Capture real time tensorboard data
                <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#
                capture-real-time-tensorboard-data-from-the-debugging-hook>`_.
            enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
                Series. For more information, see `AlgorithmSpecification API
                <https://docs.aws.amazon.com/sagemaker/latest/dg/
                API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-
                EnableSageMakerMetricsTimeSeries>`_.
                (default: ``None``).
            profiler_config (:class:`~sagemaker.debugger.ProfilerConfig`):
                Configuration for how SageMaker Debugger collects
                monitoring and profiling information from your training job.
                If not specified, Debugger will be configured with
                a default configuration and will save system and framework metrics
                the estimator's default ``output_path`` in Amazon S3.
                Use :class:`~sagemaker.debugger.ProfilerConfig` to configure this parameter.
                To disable SageMaker Debugger monitoring and profiling, set the
                ``disable_profiler`` parameter to ``True``.
            disable_profiler (bool): Specifies whether Debugger monitoring and profiling
                will be disabled (default: ``False``).
            environment (dict[str, str]) : Environment variables to be set for
                use during training job (default: ``None``)
            max_retry_attempts (int): The number of times to move a job to the STARTING status.
                You can specify between 1 and 30 attempts.
                If the value of attempts is greater than zero,
                the job is retried on InternalServerFailure
                the same number of attempts as the value.
                You can cap the total duration for your job by setting ``max_wait`` and ``max_run``
                (default: ``None``)
        """
        self.image_uri = image_uri
        self.hyperparam_dict = hyperparameters.copy() if hyperparameters else {}
        super(Estimator, self).__init__(
            role,
            instance_count,
            instance_type,
            volume_size,
            volume_kms_key,
            max_run,
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
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            use_spot_instances=use_spot_instances,
            max_wait=max_wait,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
            rules=rules,
            debugger_hook_config=debugger_hook_config,
            tensorboard_output_config=tensorboard_output_config,
            enable_sagemaker_metrics=enable_sagemaker_metrics,
            enable_network_isolation=enable_network_isolation,
            profiler_config=profiler_config,
            disable_profiler=disable_profiler,
            environment=environment,
            max_retry_attempts=max_retry_attempts,
            **kwargs,
        )

    def training_image_uri(self):
        """Returns the docker image to use for training.

        The fit() method, that does the model training, calls this method to
        find the image to use for model training.
        """
        return self.image_uri

    def set_hyperparameters(self, **kwargs):
        """Placeholder docstring"""
        for k, v in kwargs.items():
            self.hyperparam_dict[k] = v

    def hyperparameters(self):
        """Returns the hyperparameters as a dictionary to use for training.

        The fit() method, that does the model training, calls this method to
        find the hyperparameters you specified.
        """
        return self.hyperparam_dict

    def create_model(
        self,
        role=None,
        image_uri=None,
        predictor_cls=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        **kwargs,
    ):
        """Create a model to deploy.

        The serializer and deserializer arguments are only used to define a
        default Predictor. They are ignored if an explicit predictor class is passed in.
        Other arguments are passed through to the Model class.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            image_uri (str): A Docker image URI to use for deploying the model.
                Defaults to the image used for training.
            predictor_cls (Predictor): The predictor class to use when
                deploying the model.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs: Additional parameters passed to :class:`~sagemaker.model.Model`

        .. tip::

            You can find additional parameters for using this method at
            :class:`~sagemaker.model.Model`.

        Returns:
            (sagemaker.model.Model) a Model ready for deployment.
        """
        removed_kwargs("serializer", kwargs)
        removed_kwargs("deserializer", kwargs)
        removed_kwargs("content_type", kwargs)
        removed_kwargs("accept", kwargs)

        if predictor_cls is None:

            def predict_wrapper(endpoint, session):
                return Predictor(endpoint, session)

            predictor_cls = predict_wrapper

        role = role or self.role

        if "enable_network_isolation" not in kwargs:
            kwargs["enable_network_isolation"] = self.enable_network_isolation()

        return Model(
            image_uri or self.training_image_uri(),
            self.model_data,
            role,
            vpc_config=self.get_vpc_config(vpc_config_override),
            sagemaker_session=self.sagemaker_session,
            predictor_cls=predictor_cls,
            **kwargs,
        )


class Framework(EstimatorBase):
    """Base class that cannot be instantiated directly.

    Subclasses define functionality pertaining to specific ML frameworks,
    such as training/deployment images and predictor instances.
    """

    _framework_name = None

    LAUNCH_PS_ENV_NAME = "sagemaker_parameter_server_enabled"
    LAUNCH_MPI_ENV_NAME = "sagemaker_mpi_enabled"
    LAUNCH_SM_DDP_ENV_NAME = "sagemaker_distributed_dataparallel_enabled"
    INSTANCE_TYPE = "sagemaker_instance_type"
    MPI_NUM_PROCESSES_PER_HOST = "sagemaker_mpi_num_of_processes_per_host"
    MPI_CUSTOM_MPI_OPTIONS = "sagemaker_mpi_custom_mpi_options"
    SM_DDP_CUSTOM_MPI_OPTIONS = "sagemaker_distributed_dataparallel_custom_mpi_options"
    CONTAINER_CODE_CHANNEL_SOURCEDIR_PATH = "/opt/ml/input/data/code/sourcedir.tar.gz"

    def __init__(
        self,
        entry_point,
        source_dir=None,
        hyperparameters=None,
        container_log_level=logging.INFO,
        code_location=None,
        image_uri=None,
        dependencies=None,
        enable_network_isolation=False,
        git_config=None,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
        enable_sagemaker_metrics=None,
        **kwargs,
    ):
        """Base class initializer.

        Subclasses which override ``__init__`` should invoke ``super()``.

        Args:
            entry_point (str): Path (absolute or relative) to the local Python
                source file which should be executed as the entry point to
                training. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo.

                Example:
                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- train.py
                    >>>         |----- test.py

                    You can assign entry_point='src/train.py'.
            source_dir (str): Path (absolute, relative or an S3 URI) to a directory
                with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory are preserved
                when training on Amazon SageMaker. If 'git_config' is provided,
                'source_dir' should be a relative location to a directory in the Git
                repo.

                .. admonition:: Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- train.py
                    >>>         |----- test.py

                    and you need 'train.py' as entry point and 'test.py' as
                    training source code as well, you can assign
                    entry_point='train.py', source_dir='src'.
            hyperparameters (dict): Hyperparameters that will be used for
                training (default: None). The hyperparameters are made
                accessible as a dict[str, str] to the training code on
                SageMaker. For convenience, this accepts other types for keys
                and values, but ``str()`` will be called to convert them before
                training.
            container_log_level (int): Log level to use within the container
                (default: logging.INFO). Valid values are defined in the Python
                logging module.
            code_location (str): The S3 prefix URI where custom code will be
                uploaded (default: None) - don't include a trailing slash since
                a string prepended with a "/" is appended to ``code_location``. The code
                file uploaded to S3 is 'code_location/job-name/source/sourcedir.tar.gz'.
                If not specified, the default ``code location`` is s3://output_bucket/job-name/.
            image_uri (str): An alternate image name to use instead of the
                official Sagemaker image for the framework. This is useful to
                run one of the Sagemaker supported frameworks with an image
                containing custom dependencies.
            dependencies (list[str]): A list of paths to directories (absolute
                or relative) with any additional libraries that will be exported
                to the container (default: []). The library folders will be
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo.

                .. admonition:: Example

                    The following call

                    >>> Estimator(entry_point='train.py',
                    ...           dependencies=['my/libs/common', 'virtual-env'])

                    results in the following inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ train.py
                    >>>     |------ common
                    >>>     |------ virtual-env

                This is not supported with "local code" in Local Mode.
            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode. Network isolation mode restricts
                the container access to outside networks (such as the internet).
                The container does not make any inbound or outbound network
                calls. If True, a channel named "code" will be created for any
                user entry script for training. The user entry script, files in
                source_dir (if specified), and dependencies will be uploaded in
                a tar to S3. Also known as internet-free mode (default: `False`).
            git_config (dict[str, str]): Git configurations used for cloning
                files, including ``repo``, ``branch``, ``commit``,
                ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                ``repo`` field is required. All other fields are optional.
                ``repo`` specifies the Git repository where your training script
                is stored. If you don't provide ``branch``, the default value
                'master' is used. If you don't provide ``commit``, the latest
                commit in the specified branch is used. .. admonition:: Example

                    The following config:

                    >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                    >>>               'branch': 'test-branch-git-config',
                    >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

                    results in cloning the repo specified in 'repo', then
                    checkout the 'master' branch, and checkout the specified
                    commit.

                ``2FA_enabled``, ``username``, ``password`` and ``token`` are
                used for authentication. For GitHub (or other Git) accounts, set
                ``2FA_enabled`` to 'True' if two-factor authentication is
                enabled for the account, otherwise set it to 'False'. If you do
                not provide a value for ``2FA_enabled``, a default value of
                'False' is used. CodeCommit does not support two-factor
                authentication, so do not provide "2FA_enabled" with CodeCommit
                repositories.

                For GitHub and other Git repos, when SSH URLs are provided, it
                doesn't matter whether 2FA is enabled or disabled; you should
                either have no passphrase for the SSH key pairs, or have the
                ssh-agent configured so that you will not be prompted for SSH
                passphrase when you do 'git clone' command with SSH URLs. When
                HTTPS URLs are provided: if 2FA is disabled, then either token
                or username+password will be used for authentication if provided
                (token prioritized); if 2FA is enabled, only token will be used
                for authentication if provided. If required authentication info
                is not provided, python SDK will try to use local credentials
                storage to authenticate. If that fails either, an error message
                will be thrown.

                For CodeCommit repos, 2FA is not supported, so '2FA_enabled'
                should not be provided. There is no token in CodeCommit, so
                'token' should not be provided too. When 'repo' is an SSH URL,
                the requirements are the same as GitHub-like repos. When 'repo'
                is an HTTPS URL, username+password will be used for
                authentication if they are provided; otherwise, python SDK will
                try to use either CodeCommit credential helper or local
                credential storage for authentication.
            checkpoint_s3_uri (str): The S3 URI in which to persist checkpoints
                that the algorithm persists (if any) during training. (default:
                ``None``).
            checkpoint_local_path (str): The local path that the algorithm
                writes its checkpoints to. SageMaker will persist all files
                under this path to `checkpoint_s3_uri` continually during
                training. On job startup the reverse happens - data from the
                s3 location is downloaded to this path before the algorithm is
                started. If the path is unset then SageMaker assumes the
                checkpoints will be provided under `/opt/ml/checkpoints/`.
                (default: ``None``).
            enable_sagemaker_metrics (bool): enable SageMaker Metrics Time
                Series. For more information see:
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_AlgorithmSpecification.html#SageMaker-Type-AlgorithmSpecification-EnableSageMakerMetricsTimeSeries
                (default: ``None``).
            **kwargs: Additional kwargs passed to the ``EstimatorBase``
                constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        super(Framework, self).__init__(enable_network_isolation=enable_network_isolation, **kwargs)
        image_uri = renamed_kwargs("image_name", "image_uri", image_uri, kwargs)
        if entry_point.startswith("s3://"):
            raise ValueError(
                "Invalid entry point script: {}. Must be a path to a local file.".format(
                    entry_point
                )
            )
        self.entry_point = entry_point
        self.git_config = git_config
        self.source_dir = source_dir
        self.dependencies = dependencies or []
        self.uploaded_code = None

        self.container_log_level = container_log_level
        self.code_location = code_location
        self.image_uri = image_uri

        self._hyperparameters = hyperparameters or {}
        self.checkpoint_s3_uri = checkpoint_s3_uri
        self.checkpoint_local_path = checkpoint_local_path
        self.enable_sagemaker_metrics = enable_sagemaker_metrics

    def _prepare_for_training(self, job_name=None):
        """Set hyperparameters needed for training. This method will also validate ``source_dir``.

        Args:
           * job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        super(Framework, self)._prepare_for_training(job_name=job_name)

        if self.git_config:
            updated_paths = git_utils.git_clone_repo(
                self.git_config, self.entry_point, self.source_dir, self.dependencies
            )
            self.entry_point = updated_paths["entry_point"]
            self.source_dir = updated_paths["source_dir"]
            self.dependencies = updated_paths["dependencies"]

        # validate source dir will raise a ValueError if there is something wrong with the
        # source directory. We are intentionally not handling it because this is a critical error.
        if self.source_dir and not self.source_dir.lower().startswith("s3://"):
            validate_source_dir(self.entry_point, self.source_dir)

        # if we are in local mode with local_code=True. We want the container to just
        # mount the source dir instead of uploading to S3.
        local_code = get_config_value("local.local_code", self.sagemaker_session.config)
        if self.sagemaker_session.local_mode and local_code:
            # if there is no source dir, use the directory containing the entry point.
            if self.source_dir is None:
                self.source_dir = os.path.dirname(self.entry_point)
            self.entry_point = os.path.basename(self.entry_point)

            code_dir = "file://" + self.source_dir
            script = self.entry_point
        elif self.enable_network_isolation() and self.entry_point:
            self.uploaded_code = self._stage_user_code_in_s3()
            code_dir = self.CONTAINER_CODE_CHANNEL_SOURCEDIR_PATH
            script = self.uploaded_code.script_name
            self.code_uri = self.uploaded_code.s3_prefix
        else:
            self.uploaded_code = self._stage_user_code_in_s3()
            code_dir = self.uploaded_code.s3_prefix
            script = self.uploaded_code.script_name

        # Modify hyperparameters in-place to point to the right code directory and script URIs
        self._hyperparameters[DIR_PARAM_NAME] = code_dir
        self._hyperparameters[SCRIPT_PARAM_NAME] = script
        self._hyperparameters[CONTAINER_LOG_LEVEL_PARAM_NAME] = self.container_log_level
        self._hyperparameters[JOB_NAME_PARAM_NAME] = self._current_job_name
        self._hyperparameters[SAGEMAKER_REGION_PARAM_NAME] = self.sagemaker_session.boto_region_name

        self._validate_and_set_debugger_configs()

    def _validate_and_set_debugger_configs(self):
        """Set defaults for debugging."""
        if self.debugger_hook_config is None and _region_supports_debugger(
            self.sagemaker_session.boto_region_name
        ):
            self.debugger_hook_config = DebuggerHookConfig(s3_output_path=self.output_path)
        elif not self.debugger_hook_config:
            # set hook config to False if _region_supports_debugger is False
            self.debugger_hook_config = False

        # Disable debugger if checkpointing is enabled by the customer
        if self.checkpoint_s3_uri and self.checkpoint_local_path and self.debugger_hook_config:
            if self._framework_name in {"mxnet", "pytorch", "tensorflow"}:
                if self.instance_count > 1 or (
                    hasattr(self, "distribution")
                    and self.distribution is not None  # pylint: disable=no-member
                ):
                    logger.info(
                        "SMDebug Does Not Currently Support \
                        Distributed Training Jobs With Checkpointing Enabled"
                    )
                    self.debugger_hook_config = False

    def _stage_user_code_in_s3(self):
        """Upload the user training script to s3 and return the location.

        Returns: s3 uri
        """
        local_mode = self.output_path.startswith("file://")

        if self.code_location is None and local_mode:
            code_bucket = self.sagemaker_session.default_bucket()
            code_s3_prefix = "{}/{}".format(self._current_job_name, "source")
            kms_key = None
        elif self.code_location is None:
            code_bucket, _ = parse_s3_url(self.output_path)
            code_s3_prefix = "{}/{}".format(self._current_job_name, "source")
            kms_key = self.output_kms_key
        elif local_mode:
            code_bucket, key_prefix = parse_s3_url(self.code_location)
            code_s3_prefix = "/".join(filter(None, [key_prefix, self._current_job_name, "source"]))
            kms_key = None
        else:
            code_bucket, key_prefix = parse_s3_url(self.code_location)
            code_s3_prefix = "/".join(filter(None, [key_prefix, self._current_job_name, "source"]))

            output_bucket, _ = parse_s3_url(self.output_path)
            kms_key = self.output_kms_key if code_bucket == output_bucket else None

        return tar_and_upload_dir(
            session=self.sagemaker_session.boto_session,
            bucket=code_bucket,
            s3_key_prefix=code_s3_prefix,
            script=self.entry_point,
            directory=self.source_dir,
            dependencies=self.dependencies,
            kms_key=kms_key,
            s3_resource=self.sagemaker_session.s3_resource,
        )

    def _model_source_dir(self):
        """Get the appropriate value to pass as ``source_dir`` to a model constructor.

        Returns:
            str: Either a local or an S3 path pointing to the ``source_dir`` to be
                used for code by the model to be deployed
        """
        return (
            self.source_dir if self.sagemaker_session.local_mode else self.uploaded_code.s3_prefix
        )

    def _model_entry_point(self):
        """Get the appropriate value to pass as ``entry_point`` to a model constructor.

        Returns:
            str: The path to the entry point script. This can be either an absolute path or
                a path relative to ``self._model_source_dir()``.
        """
        if self.sagemaker_session.local_mode or (self._model_source_dir() is None):
            return self.entry_point

        return self.uploaded_code.script_name

    def hyperparameters(self):
        """Return the hyperparameters as a dictionary to use for training.

        The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which
        trains the model, calls this method to find the hyperparameters.

        Returns:
            dict[str, str]: The hyperparameters.
        """
        return self._json_encode_hyperparameters(self._hyperparameters)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(Framework, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        init_params["entry_point"] = json.loads(
            init_params["hyperparameters"].get(SCRIPT_PARAM_NAME)
        )
        init_params["source_dir"] = json.loads(init_params["hyperparameters"].get(DIR_PARAM_NAME))
        init_params["container_log_level"] = json.loads(
            init_params["hyperparameters"].get(CONTAINER_LOG_LEVEL_PARAM_NAME)
        )

        hyperparameters = {}
        for k, v in init_params["hyperparameters"].items():
            # Tuning jobs add this special hyperparameter which is not JSON serialized
            if k == "_tuning_objective_metric":
                if v.startswith('"') and v.endswith('"'):
                    v = v.strip('"')
                hyperparameters[k] = v
            else:
                hyperparameters[k] = json.loads(v)

        init_params["hyperparameters"] = hyperparameters

        return init_params

    def training_image_uri(self):
        """Return the Docker image to use for training.

        The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does
        the model training, calls this method to find the image to use for model
        training.

        Returns:
            str: The URI of the Docker image.
        """
        if self.image_uri:
            return self.image_uri
        if hasattr(self, "distribution"):
            distribution = self.distribution  # pylint: disable=no-member
        else:
            distribution = None

        if hasattr(self, "tensorflow_version") or hasattr(self, "pytorch_version"):
            processor = image_uris._processor(self.instance_type, ["cpu", "gpu"])
            container_version = "cu110-ubuntu18.04" if processor == "gpu" else None
            if self.tensorflow_version is not None:  # pylint: disable=no-member
                base_framework_version = (
                    f"tensorflow{self.tensorflow_version}"  # pylint: disable=no-member
                )
            else:
                base_framework_version = (
                    f"pytorch{self.pytorch_version}"  # pylint: disable=no-member
                )
        else:
            container_version = None
            base_framework_version = None

        return image_uris.retrieve(
            self._framework_name,
            self.sagemaker_session.boto_region_name,
            instance_type=self.instance_type,
            version=self.framework_version,  # pylint: disable=no-member
            py_version=self.py_version,  # pylint: disable=no-member
            image_scope="training",
            distribution=distribution,
            base_framework_version=base_framework_version,
            container_version=container_version,
        )

    @classmethod
    def attach(cls, training_job_name, sagemaker_session=None, model_channel_name="model"):
        """Attach to an existing training job.

        Create an Estimator bound to an existing training job, each subclass
        is responsible to implement
        ``_prepare_init_params_from_job_description()`` as this method delegates
        the actual conversion of a training job description to the arguments
        that the class constructor expects. After attaching, if the training job
        has a Complete status, it can be ``deploy()`` ed to create a SageMaker
        Endpoint and return a ``Predictor``.

        If the training job is in progress, attach will block until the training job
        completes, but logs of the training job will not display. To see the logs
        content, please call ``logs()``

        Examples:
            >>> my_estimator.fit(wait=False)
            >>> training_job_name = my_estimator.latest_training_job.name
            Later on:
            >>> attached_estimator = Estimator.attach(training_job_name)
            >>> attached_estimator.logs()
            >>> attached_estimator.deploy()

        Args:
            training_job_name (str): The name of the training job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded (default: 'model'). If no channel
                with the same name exists in the training job, this option will
                be ignored.

        Returns:
            Instance of the calling ``Estimator`` Class with the attached
            training job.
        """
        estimator = super(Framework, cls).attach(
            training_job_name, sagemaker_session, model_channel_name
        )

        # pylint gets confused thinking that estimator is an EstimatorBase instance, but it actually
        # is a Framework or any of its derived classes. We can safely ignore the no-member errors.
        estimator.uploaded_code = UploadedCode(
            estimator.source_dir, estimator.entry_point  # pylint: disable=no-member
        )
        return estimator

    @staticmethod
    def _json_encode_hyperparameters(hyperparameters):
        """Placeholder docstring"""
        current_hyperparameters = hyperparameters
        if current_hyperparameters is not None:
            hyperparameters = {
                str(k): (v if isinstance(v, (Parameter, Expression, Properties)) else json.dumps(v))
                for (k, v) in current_hyperparameters.items()
            }
        return hyperparameters

    @classmethod
    def _update_init_params(cls, hp, tf_arguments):
        """Placeholder docstring"""
        updated_params = {}
        for argument in tf_arguments:
            value = hp.pop(argument, None)
            if value is not None:
                value = json.loads(value)
                updated_params[argument] = value
        return updated_params

    def transformer(
        self,
        instance_count,
        instance_type,
        strategy=None,
        assemble_with=None,
        output_path=None,
        output_kms_key=None,
        accept=None,
        env=None,
        max_concurrent_transforms=None,
        max_payload=None,
        tags=None,
        role=None,
        model_server_workers=None,
        volume_kms_key=None,
        entry_point=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        enable_network_isolation=None,
        model_name=None,
    ):
        """Return a ``Transformer`` that uses a SageMaker Model based on the training job.

        It reuses the SageMaker Session and base job name used by
        the Estimator.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in
                a single request (default: None). Valid values: 'MultiRecord'
                and 'SingleRecord'.
            assemble_with (str): How the output is assembled (default: None).
                Valid values: 'Line' or 'None'.
            output_path (str): S3 location for saving the transform result. If
                not specified, results are stored to a default bucket.
            output_kms_key (str): Optional. KMS key ID for encrypting the
                transform output (default: None).
            accept (str): The accept header passed by the client to
                the inference endpoint. If it is supported by the endpoint,
                it will be the format of the batch transform output.
            env (dict): Environment variables to be set for use during the
                transform job (default: None).
            max_concurrent_transforms (int): The maximum number of HTTP requests
                to be made to each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP
                request to the container in MB.
            tags (list[dict]): List of tags for labeling a transform job. If
                none specified, then the tags used for the training job are used
                for the transform job.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            volume_kms_key (str): Optional. KMS key ID for encrypting the volume
                attached to the ML compute instance (default: None).
            entry_point (str): Path (absolute or relative) to the local Python source file which
                should be executed as the entry point to training. If ``source_dir`` is specified,
                then ``entry_point`` must point to a file located at the root of ``source_dir``.
                If not specified, the training entry point is used.
            vpc_config_override (dict[str, list[str]]): Optional override for
                the VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.

                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode. Network isolation mode restricts
                the container access to outside networks (such as the internet).
                The container does not make any inbound or outbound network
                calls. If True, a channel named "code" will be created for any
                user entry script for inference. Also known as Internet-free mode.
                If not specified, this setting is taken from the estimator's
                current configuration.
            model_name (str): Name to use for creating an Amazon SageMaker
                model. If not specified, the estimator generates a default job name
                based on the training image name and current timestamp.

        Returns:
            sagemaker.transformer.Transformer: a ``Transformer`` object that can be used to start a
                SageMaker Batch Transform job.
        """
        role = role or self.role
        tags = tags or self.tags
        model_name = self._get_or_create_name(model_name)

        if self.latest_training_job is not None:
            if enable_network_isolation is None:
                enable_network_isolation = self.enable_network_isolation()

            model = self.create_model(
                role=role,
                model_server_workers=model_server_workers,
                entry_point=entry_point,
                vpc_config_override=vpc_config_override,
                model_kms_key=self.output_kms_key,
                enable_network_isolation=enable_network_isolation,
                name=model_name,
            )
            model._create_sagemaker_model(instance_type, tags=tags)

            transform_env = model.env.copy()
            if env is not None:
                transform_env.update(env)
        else:
            logger.warning(
                "No finished training job found associated with this estimator. Please make sure "
                "this estimator is only used for building workflow config"
            )
            transform_env = env or {}

        return Transformer(
            model_name,
            instance_count,
            instance_type,
            strategy=strategy,
            assemble_with=assemble_with,
            output_path=output_path,
            output_kms_key=output_kms_key,
            accept=accept,
            max_concurrent_transforms=max_concurrent_transforms,
            max_payload=max_payload,
            env=transform_env,
            tags=tags,
            base_transform_job_name=self.base_job_name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=self.sagemaker_session,
        )

    def _distribution_configuration(self, distribution):
        """Returns a dict of distribution configurations.

        Args:
            distribution (dict): A dictionary with information on how to run distributed training.

        Returns:
            dict that
        """
        distribution_config = {}

        if "parameter_server" in distribution:
            ps_enabled = distribution.get("parameter_server").get("enabled", False)
            distribution_config[self.LAUNCH_PS_ENV_NAME] = ps_enabled

        if "mpi" in distribution:
            mpi_dict = distribution["mpi"]
            mpi_enabled = mpi_dict.get("enabled", False)
            distribution_config[self.LAUNCH_MPI_ENV_NAME] = mpi_enabled

            if mpi_dict.get("processes_per_host"):
                distribution_config[self.MPI_NUM_PROCESSES_PER_HOST] = mpi_dict.get(
                    "processes_per_host"
                )

            distribution_config[self.MPI_CUSTOM_MPI_OPTIONS] = mpi_dict.get(
                "custom_mpi_options", ""
            )

            if get_mp_parameters(distribution):
                distribution_config["mp_parameters"] = get_mp_parameters(distribution)

        elif "modelparallel" in distribution.get("smdistributed", {}):
            raise ValueError("Cannot use Model Parallelism without MPI enabled!")

        if "smdistributed" in distribution:
            # smdistributed strategy selected
            smdistributed = distribution["smdistributed"]
            smdataparallel_enabled = smdistributed.get("dataparallel", {}).get("enabled", False)
            distribution_config[self.LAUNCH_SM_DDP_ENV_NAME] = smdataparallel_enabled
            distribution_config[self.INSTANCE_TYPE] = self.instance_type
            if smdataparallel_enabled:
                distribution_config[self.SM_DDP_CUSTOM_MPI_OPTIONS] = smdistributed[
                    "dataparallel"
                ].get("custom_mpi_options", "")

        return distribution_config


def _s3_uri_prefix(channel_name, s3_data):
    """Placeholder docstring"""
    if isinstance(s3_data, TrainingInput):
        s3_uri = s3_data.config["DataSource"]["S3DataSource"]["S3Uri"]
    else:
        s3_uri = s3_data
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expecting an s3 uri. Got {}".format(s3_uri))
    return {channel_name: s3_uri[5:]}


# E.g. 's3://bucket/data' would return 'bucket/data'.
# Also accepts other valid input types, e.g. dict and TrainingInput.
def _s3_uri_without_prefix_from_input(input_data):
    # Unpack an input_config object from a dict if a dict was passed in.
    """Placeholder docstring"""
    if isinstance(input_data, dict):
        response = {}
        for channel_name, channel_s3_uri in input_data.items():
            response.update(_s3_uri_prefix(channel_name, channel_s3_uri))
        return response
    if isinstance(input_data, str):
        return _s3_uri_prefix("training", input_data)
    if isinstance(input_data, TrainingInput):
        return _s3_uri_prefix("training", input_data)
    raise ValueError(
        "Unrecognized type for S3 input data config - not str or TrainingInput: {}".format(
            input_data
        )
    )
