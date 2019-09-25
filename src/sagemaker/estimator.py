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
from __future__ import print_function, absolute_import

import json
import logging
import os
import warnings
from abc import ABCMeta
from abc import abstractmethod
from six import with_metaclass
from six import string_types
import sagemaker
from sagemaker import git_utils
from sagemaker.analytics import TrainingJobAnalytics

from sagemaker.fw_utils import (
    create_image_uri,
    tar_and_upload_dir,
    parse_s3_url,
    UploadedCode,
    validate_source_dir,
)
from sagemaker.job import _Job
from sagemaker.local import LocalSession
from sagemaker.model import Model, NEO_ALLOWED_TARGET_INSTANCE_FAMILY, NEO_ALLOWED_FRAMEWORKS
from sagemaker.model import (
    SCRIPT_PARAM_NAME,
    DIR_PARAM_NAME,
    CLOUDWATCH_METRICS_PARAM_NAME,
    CONTAINER_LOG_LEVEL_PARAM_NAME,
    JOB_NAME_PARAM_NAME,
    SAGEMAKER_REGION_PARAM_NAME,
)
from sagemaker.predictor import RealTimePredictor
from sagemaker.session import Session
from sagemaker.session import s3_input
from sagemaker.transformer import Transformer
from sagemaker.utils import base_name_from_image, name_from_base, get_config_value
from sagemaker import vpc_utils


class EstimatorBase(with_metaclass(ABCMeta, object)):
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
        train_instance_count,
        train_instance_type,
        train_volume_size=30,
        train_volume_kms_key=None,
        train_max_run=24 * 60 * 60,
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
        train_use_spot_instances=False,
        train_max_wait=None,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
    ):
        """Initialize an ``EstimatorBase`` instance.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use
                for training.
            train_instance_type (str): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            train_volume_size (int): Size in GB of the EBS volume to use for
                storing input data during training (default: 30). Must be large
                enough to store training data if File Mode is used (which is the
                default).
            train_volume_kms_key (str): Optional. KMS key ID for encrypting EBS
                volume attached to the training instance (default: None).
            train_max_run (int): Timeout in seconds for training (default: 24 *
                60 * 60). After this amount of time Amazon SageMaker terminates
                the job regardless of its current status.
            input_mode (str): The input mode that the algorithm supports
                (default: 'File'). Valid modes: 'File' - Amazon SageMaker copies
                the training dataset from the S3 location to a local directory.
                'Pipe' - Amazon SageMaker streams data directly from S3 to the
                container via a Unix-named pipe. This argument can be overriden
                on a per-channel basis using
                ``sagemaker.session.s3_input.input_mode``.
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
            train_use_spot_instances (bool): Specifies whether to use SageMaker
                Managed Spot instances for training. If enabled then the
                `train_max_wait` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (default: ``False``).
            train_max_wait (int): Timeout in seconds waiting for spot training
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
        """
        self.role = role
        self.train_instance_count = train_instance_count
        self.train_instance_type = train_instance_type
        self.train_volume_size = train_volume_size
        self.train_volume_kms_key = train_volume_kms_key
        self.train_max_run = train_max_run
        self.input_mode = input_mode
        self.tags = tags
        self.metric_definitions = metric_definitions
        self.model_uri = model_uri
        self.model_channel_name = model_channel_name
        self.code_uri = None
        self.code_channel_name = "code"

        if self.train_instance_type in ("local", "local_gpu"):
            if self.train_instance_type == "local_gpu" and self.train_instance_count > 1:
                raise RuntimeError("Distributed Training in Local GPU is not supported")
            self.sagemaker_session = sagemaker_session or LocalSession()
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
        self.deploy_instance_type = None

        self._compiled_models = {}

        # VPC configurations
        self.subnets = subnets
        self.security_group_ids = security_group_ids

        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        self.train_use_spot_instances = train_use_spot_instances
        self.train_max_wait = train_max_wait
        self.checkpoint_s3_uri = checkpoint_s3_uri
        self.checkpoint_local_path = checkpoint_local_path

    @abstractmethod
    def train_image(self):
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
        return False

    def prepare_workflow_for_training(self, job_name=None):
        """Calls _prepare_for_training. Used when setting up a workflow.

        Args:
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        self._prepare_for_training(job_name=job_name)

    def _prepare_for_training(self, job_name=None):
        """Set any values in the estimator that need to be set before training.

        Args:
            job_name (str): Name of the training job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.
        """
        if job_name is not None:
            self._current_job_name = job_name
        else:
            # honor supplied base_job_name or generate it
            if self.base_job_name:
                base_name = self.base_job_name
            elif isinstance(self, sagemaker.algorithm.AlgorithmEstimator):
                base_name = self.algorithm_arn.split("/")[-1]  # pylint: disable=no-member
            else:
                base_name = base_name_from_image(self.train_image())

            self._current_job_name = name_from_base(base_name)

        # if output_path was specified we use it otherwise initialize here.
        # For Local Mode with local_code=True we don't need an explicit output_path
        if self.output_path is None:
            local_code = get_config_value("local.local_code", self.sagemaker_session.config)
            if self.sagemaker_session.local_mode and local_code:
                self.output_path = ""
            else:
                self.output_path = "s3://{}/".format(self.sagemaker_session.default_bucket())

    def fit(self, inputs=None, wait=True, logs=True, job_name=None):
        """Train a model using the input training dataset.

        The API calls the Amazon SageMaker CreateTrainingJob API to start
        model training. The API uses configuration you provided to create the
        estimator and the specified input training data to send the
        CreatingTrainingJob request to Amazon SageMaker.

        This is a synchronous operation. After the model training
        successfully completes, you can call the ``deploy()`` method to host the
        model using the Amazon SageMaker hosting services.

        Args:
            inputs (str or dict or sagemaker.session.s3_input): Information
                about the training data. This can be one of three types:

                * (str) the S3 location where training data is saved.
                * (dict[str, str] or dict[str, sagemaker.session.s3_input]) If using multiple
                    channels for training data, you can specify a dict mapping channel names to
                    strings or :func:`~sagemaker.session.s3_input` objects.
                * (sagemaker.session.s3_input) - channel configuration for S3 data sources that can
                    provide additional information as well as the path to the training dataset.
                    See :func:`sagemaker.session.s3_input` for full details.
                * (sagemaker.session.FileSystemInput) - channel configuration for
                    a file system data source that can provide additional information as well as
                    the path to the training dataset.

            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Training job name. If not specified, the estimator generates
                a default job name, based on the training image name and current timestamp.
        """
        self._prepare_for_training(job_name=job_name)

        self.latest_training_job = _TrainingJob.start_new(self, inputs)
        if wait:
            self.latest_training_job.wait(logs=logs)

    def _compilation_job_name(self):
        """Placeholder docstring"""
        base_name = self.base_job_name or base_name_from_image(self.train_image())
        return name_from_base("compilation-" + base_name)

    def compile_model(
        self,
        target_instance_family,
        input_shape,
        output_path,
        framework=None,
        framework_version=None,
        compile_max_run=5 * 60,
        tags=None,
        **kwargs
    ):
        """Compile a Neo model using the input model.

        Args:
            target_instance_family (str): Identifies the device that you want to
                run your model after compilation, for example: ml_c5. Allowed
                strings are: ml_c5, ml_m5, ml_c4, ml_m4, jetsontx1, jetsontx2,
                ml_p2, ml_p3, deeplens, rasp3b
            input_shape (dict): Specifies the name and shape of the expected
                inputs for your trained model in json dictionary form, for
                example: {'data':[1,3,1024,1024]}, or {'var1': [1,1,28,28],
                'var2':[1,1,28,28]}
            output_path (str): Specifies where to store the compiled model
            framework (str): The framework that is used to train the original
                model. Allowed values: 'mxnet', 'tensorflow', 'pytorch', 'onnx',
                'xgboost'
            framework_version (str): The version of the framework
            compile_max_run (int): Timeout in seconds for compilation (default:
                3 * 60). After this amount of time Amazon SageMaker Neo
                terminates the compilation job regardless of its current status.
            tags (list[dict]): List of tags for labeling a compilation job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            **kwargs: Passed to invocation of ``create_model()``.
                Implementations may customize ``create_model()`` to accept
                ``**kwargs`` to customize model creation during deploy. For
                more, see the implementation docs.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """
        if target_instance_family not in NEO_ALLOWED_TARGET_INSTANCE_FAMILY:
            raise ValueError(
                "Please use valid target_instance_family,"
                "allowed values: {}".format(NEO_ALLOWED_TARGET_INSTANCE_FAMILY)
            )
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

        If the training job is in progress, attach will block and display log
        messages from the training job, until the training job completes.

        Examples:
            >>> my_estimator.fit(wait=False)
            >>> training_job_name = my_estimator.latest_training_job.name
            Later on:
            >>> attached_estimator = Estimator.attach(training_job_name)
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
            sagemaker_session=sagemaker_session, job_name=init_params["base_job_name"]
        )
        estimator._current_job_name = estimator.latest_training_job.name
        estimator.latest_training_job.wait()
        return estimator

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        accelerator_type=None,
        endpoint_name=None,
        use_compiled_model=False,
        update_endpoint=False,
        wait=True,
        model_name=None,
        kms_key=None,
        **kwargs
    ):
        """Deploy the trained model to an Amazon SageMaker endpoint and return a
        ``sagemaker.RealTimePredictor`` object.

        More information:
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
            use_compiled_model (bool): Flag to select whether to use compiled
                (optimized) model. Default: False.
            update_endpoint (bool): Flag to update the model in an existing
                Amazon SageMaker endpoint. If True, this will deploy a new
                EndpointConfig to an already existing endpoint and delete
                resources corresponding to the previous EndpointConfig. Default:
                False
            wait (bool): Whether the call should wait until the deployment of
                model completes (default: True).
            model_name (str): Name to use for creating an Amazon SageMaker
                model. If not specified, the name of the training job is used.
            tags(List[dict[str, str]]): Optional. The list of tags to attach to this specific
                endpoint. Example:
                >>> tags = [{'Key': 'tagname', 'Value': 'tagvalue'}]
                For more information about tags, see
                https://boto3.amazonaws.com/v1/documentation\
                /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint.
            **kwargs: Passed to invocation of ``create_model()``.
                Implementations may customize ``create_model()`` to accept
                ``**kwargs`` to customize model creation during deploy.
                For more, see the implementation docs.

        Returns:
            sagemaker.predictor.RealTimePredictor: A predictor that provides a ``predict()`` method,
                which can be used to send requests to the Amazon SageMaker
                endpoint and obtain inferences.
        """
        self._ensure_latest_training_job()
        endpoint_name = endpoint_name or self.latest_training_job.name
        model_name = model_name or self.latest_training_job.name
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
            accelerator_type=accelerator_type,
            endpoint_name=endpoint_name,
            update_endpoint=update_endpoint,
            tags=self.tags,
            wait=wait,
            kms_key=kms_key,
        )

    @property
    def model_data(self):
        """str: The model location in S3. Only set if Estimator has been
        ``fit()``.
        """
        if self.latest_training_job is not None:
            model_uri = self.sagemaker_session.sagemaker_client.describe_training_job(
                TrainingJobName=self.latest_training_job.name
            )["ModelArtifacts"]["S3ModelArtifacts"]
        else:
            logging.warning(
                "No finished training job found associated with this estimator. Please make sure "
                "this estimator is only used for building workflow config"
            )
            model_uri = os.path.join(
                self.output_path, self._current_job_name, "output", "model.tar.gz"
            )

        return model_uri

    @abstractmethod
    def create_model(self, **kwargs):
        """Create a SageMaker ``Model`` object that can be deployed to an
        ``Endpoint``.

        Args:
            **kwargs: Keyword arguments used by the implemented method for
                creating the ``Model``.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the
        class constructor

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
        init_params["train_instance_count"] = job_details["ResourceConfig"]["InstanceCount"]
        init_params["train_instance_type"] = job_details["ResourceConfig"]["InstanceType"]
        init_params["train_volume_size"] = job_details["ResourceConfig"]["VolumeSizeInGB"]
        init_params["train_max_run"] = job_details["StoppingCondition"]["MaxRuntimeInSeconds"]
        init_params["input_mode"] = job_details["AlgorithmSpecification"]["TrainingInputMode"]
        init_params["base_job_name"] = job_details["TrainingJobName"]
        init_params["output_path"] = job_details["OutputDataConfig"]["S3OutputPath"]
        init_params["output_kms_key"] = job_details["OutputDataConfig"]["KmsKeyId"]
        if "EnableNetworkIsolation" in job_details:
            init_params["enable_network_isolation"] = job_details["EnableNetworkIsolation"]

        has_hps = "HyperParameters" in job_details
        init_params["hyperparameters"] = job_details["HyperParameters"] if has_hps else {}

        if "TrainingImage" in job_details["AlgorithmSpecification"]:
            init_params["image"] = job_details["AlgorithmSpecification"]["TrainingImage"]
        elif "AlgorithmName" in job_details["AlgorithmSpecification"]:
            init_params["algorithm_arn"] = job_details["AlgorithmSpecification"]["AlgorithmName"]
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

        return init_params

    def delete_endpoint(self):
        """Delete an Amazon SageMaker ``Endpoint``.

        Raises:
            ValueError: If the endpoint does not exist.
        """
        self._ensure_latest_training_job(error_message="Endpoint was not created yet")
        self.sagemaker_session.delete_endpoint(self.latest_training_job.name)

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
    ):
        """Return a ``Transformer`` that uses a SageMaker Model based on the
        training job. It reuses the SageMaker Session and base job name used by
        the Estimator.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in
                a single request (default: None). Valid values: 'MULTI_RECORD'
                and 'SINGLE_RECORD'.
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
        """
        tags = tags or self.tags

        if self.latest_training_job is None:
            logging.warning(
                "No finished training job found associated with this estimator. Please make sure "
                "this estimator is only used for building workflow config"
            )
            model_name = self._current_job_name
        else:
            model_name = self.latest_training_job.name
            model = self.create_model(
                vpc_config_override=vpc_config_override, model_kms_key=self.output_kms_key
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
        """Return a ``TrainingJobAnalytics`` object for the current training
        job.
        """
        if self._current_job_name is None:
            raise ValueError("Estimator is not associated with a TrainingJob")
        return TrainingJobAnalytics(
            self._current_job_name, sagemaker_session=self.sagemaker_session
        )

    def get_vpc_config(self, vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT):
        """Returns VpcConfig dict either from this Estimator's subnets and
        security groups, or else validate and return an optional override value.

        Args:
            vpc_config_override:
        """
        if vpc_config_override is vpc_utils.VPC_CONFIG_DEFAULT:
            return vpc_utils.to_dict(self.subnets, self.security_group_ids)
        return vpc_utils.sanitize(vpc_config_override)

    def _ensure_latest_training_job(
        self, error_message="Estimator is not associated with a training job"
    ):
        """
        Args:
            error_message:
        """
        if self.latest_training_job is None:
            raise ValueError(error_message)


class _TrainingJob(_Job):
    """Placeholder docstring"""

    @classmethod
    def start_new(cls, estimator, inputs):
        """Create a new Amazon SageMaker training job from the estimator.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): Estimator object
                created by the user.
            inputs (str): Parameters used when called
                :meth:`~sagemaker.estimator.EstimatorBase.fit`.

        Returns:
            sagemaker.estimator._TrainingJob: Constructed object that captures
            all information about the started training job.
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

        if estimator.hyperparameters() is not None:
            hyperparameters = {str(k): str(v) for (k, v) in estimator.hyperparameters().items()}

        train_args = config.copy()
        train_args["input_mode"] = estimator.input_mode
        train_args["job_name"] = estimator._current_job_name
        train_args["hyperparameters"] = hyperparameters
        train_args["tags"] = estimator.tags
        train_args["metric_definitions"] = estimator.metric_definitions

        if isinstance(inputs, s3_input):
            if "InputMode" in inputs.config:
                logging.debug(
                    "Selecting s3_input's input_mode (%s) for TrainingInputMode.",
                    inputs.config["InputMode"],
                )
                train_args["input_mode"] = inputs.config["InputMode"]

        if estimator.enable_network_isolation():
            train_args["enable_network_isolation"] = True

        if estimator.encrypt_inter_container_traffic:
            train_args["encrypt_inter_container_traffic"] = True

        if isinstance(estimator, sagemaker.algorithm.AlgorithmEstimator):
            train_args["algorithm_arn"] = estimator.algorithm_arn
        else:
            train_args["image"] = estimator.train_image()

        cls._add_spot_checkpoint_args(local_mode, estimator, train_args)

        estimator.sagemaker_session.train(**train_args)

        return cls(estimator.sagemaker_session, estimator._current_job_name)

    @classmethod
    def _add_spot_checkpoint_args(cls, local_mode, estimator, train_args):
        """
        Args:
            local_mode:
            estimator:
            train_args:
        """
        if estimator.train_use_spot_instances:
            if local_mode:
                raise ValueError("Spot training is not supported in local mode.")
            train_args["train_use_spot_instances"] = True

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
        """
        Args:
            input_uri:
        """
        return isinstance(input_uri, string_types) and input_uri.startswith("file://")

    def wait(self, logs=True):
        """
        Args:
            logs:
        """
        if logs:
            self.sagemaker_session.logs_for_job(self.job_name, wait=True)
        else:
            self.sagemaker_session.wait_for_job(self.job_name)


class Estimator(EstimatorBase):
    """A generic Estimator to train using any supplied algorithm. This class is
    designed for use with algorithms that don't have their own, custom class.
    """

    def __init__(
        self,
        image_name,
        role,
        train_instance_count,
        train_instance_type,
        train_volume_size=30,
        train_volume_kms_key=None,
        train_max_run=24 * 60 * 60,
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
        train_use_spot_instances=False,
        train_max_wait=None,
        checkpoint_s3_uri=None,
        checkpoint_local_path=None,
        enable_network_isolation=False,
    ):
        """Initialize an ``Estimator`` instance.

        Args:
            image_name (str): The container image to use for training.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use
                for training.
            train_instance_type (str): Type of EC2 instance to use for training,
                for example, 'ml.c4.xlarge'.
            train_volume_size (int): Size in GB of the EBS volume to use for
                storing input data during training (default: 30). Must be large
                enough to store training data if File Mode is used (which is the
                default).
            train_volume_kms_key (str): Optional. KMS key ID for encrypting EBS
                volume attached to the training instance (default: None).
            train_max_run (int): Timeout in seconds for training (default: 24 *
                60 * 60). After this amount of time Amazon SageMaker terminates
                the job regardless of its current status.
            input_mode (str): The input mode that the algorithm supports
                (default: 'File'). Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the
                  S3 location to a local directory.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the
                  container via a Unix-named pipe.

                This argument can be overriden on a per-channel basis using
                ``sagemaker.session.s3_input.input_mode``.
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
            train_use_spot_instances (bool): Specifies whether to use SageMaker
                Managed Spot instances for training. If enabled then the
                `train_max_wait` arg should also be set.

                More information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html
                (default: ``False``).
            train_max_wait (int): Timeout in seconds waiting for spot training
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
                run in network isolation mode. Network isolation mode restricts
                the container access to outside networks (such as the Internet).
                The container does not make any inbound or outbound network
                calls. If ``True``, a channel named "code" will be created for any
                user entry script for training. The user entry script, files in
                source_dir (if specified), and dependencies will be uploaded in
                a tar to S3. Also known as internet-free mode (default: ``False``).
        """
        self.image_name = image_name
        self.hyperparam_dict = hyperparameters.copy() if hyperparameters else {}
        self._enable_network_isolation = enable_network_isolation
        super(Estimator, self).__init__(
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
            encrypt_inter_container_traffic=encrypt_inter_container_traffic,
            train_use_spot_instances=train_use_spot_instances,
            train_max_wait=train_max_wait,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
        )

    def enable_network_isolation(self):
        """If this Estimator can use network isolation when running.

        Returns:
            bool: Whether this Estimator can use network isolation or not.
        """
        return self._enable_network_isolation

    def train_image(self):
        """Returns the docker image to use for training.

        The fit() method, that does the model training, calls this method to
        find the image to use for model training.
        """
        return self.image_name

    def set_hyperparameters(self, **kwargs):
        """
        Args:
            **kwargs:
        """
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
        image=None,
        predictor_cls=None,
        serializer=None,
        deserializer=None,
        content_type=None,
        accept=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        **kwargs
    ):
        """Create a model to deploy.

        The serializer, deserializer, content_type, and accept arguments are only used to define a
        default RealTimePredictor. They are ignored if an explicit predictor class is passed in.
        Other arguments are passed through to the Model class.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            image (str): An container image to use for deploying the model.
                Defaults to the image used for training.
            predictor_cls (RealTimePredictor): The predictor class to use when
                deploying the model.
            serializer (callable): Should accept a single argument, the input
                data, and return a sequence of bytes. May provide a content_type
                attribute that defines the endpoint request content type
            deserializer (callable): Should accept two arguments, the result
                data and the response content type, and return a sequence of
                bytes. May provide a content_type attribute that defines th
                endpoint response Accept content type.
            content_type (str): The invocation ContentType, overriding any
                content_type from the serializer
            accept (str): The invocation Accept, overriding any accept from the
                deserializer.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            **kwargs:

        Returns: a Model ready for deployment.
        """
        if predictor_cls is None:

            def predict_wrapper(endpoint, session):
                return RealTimePredictor(
                    endpoint, session, serializer, deserializer, content_type, accept
                )

            predictor_cls = predict_wrapper

        role = role or self.role

        return Model(
            self.model_data,
            image or self.train_image(),
            role,
            vpc_config=self.get_vpc_config(vpc_config_override),
            sagemaker_session=self.sagemaker_session,
            predictor_cls=predictor_cls,
            enable_network_isolation=self.enable_network_isolation(),
            **kwargs
        )

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the
        class constructor

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(Estimator, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        init_params["image_name"] = init_params.pop("image")
        return init_params


class Framework(EstimatorBase):
    """Base class that cannot be instantiated directly.

    Subclasses define functionality pertaining to specific ML frameworks,
    such as training/deployment images and predictor instances.
    """

    __framework_name__ = None

    LAUNCH_PS_ENV_NAME = "sagemaker_parameter_server_enabled"
    LAUNCH_MPI_ENV_NAME = "sagemaker_mpi_enabled"
    MPI_NUM_PROCESSES_PER_HOST = "sagemaker_mpi_num_of_processes_per_host"
    MPI_CUSTOM_MPI_OPTIONS = "sagemaker_mpi_custom_mpi_options"
    CONTAINER_CODE_CHANNEL_SOURCEDIR_PATH = "/opt/ml/input/data/code/sourcedir.tar.gz"

    def __init__(
        self,
        entry_point,
        source_dir=None,
        hyperparameters=None,
        enable_cloudwatch_metrics=False,
        container_log_level=logging.INFO,
        code_location=None,
        image_name=None,
        dependencies=None,
        enable_network_isolation=False,
        git_config=None,
        **kwargs
    ):
        """Base class initializer. Subclasses which override ``__init__`` should
        invoke ``super()``

        Args:
            entry_point (str): Path (absolute or relative) to the local Python
                source file which should be executed as the entry point to
                training. This should be compatible with either Python 2.7 or
                Python 3.5. If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo.
                Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- train.py
                    >>>         |----- test.py

                    You can assign entry_point='src/train.py'.
            source_dir (str): Path (absolute or relative) to a directory with
                any other training source code dependencies aside from the entry
                point file (default: None). Structure within this directory are
                preserved when training on Amazon SageMaker. If 'git_config' is
                provided, 'source_dir' should be a relative location to a
                directory in the Git repo. .. admonition:: Example

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
            enable_cloudwatch_metrics (bool): [DEPRECATED] Now there are
                cloudwatch metrics emitted by all SageMaker training jobs. This
                will be ignored for now and removed in a further release.
            container_log_level (int): Log level to use within the container
                (default: logging.INFO). Valid values are defined in the Python
                logging module.
            code_location (str): The S3 prefix URI where custom code will be
                uploaded (default: None). The code file uploaded in S3 is
                'code_location/source/sourcedir.tar.gz'. If not specified, the
                default code location is s3://default_bucket/job-name/. And code
                file uploaded to S3 is
                s3://default_bucket/job-name/source/sourcedir.tar.gz
            image_name (str): An alternate image name to use instead of the
                official Sagemaker image for the framework. This is useful to
                run one of the Sagemaker supported frameworks with an image
                containing custom dependencies.
            dependencies (list[str]): A list of paths to directories (absolute
                or relative) with any additional libraries that will be exported
                to the container (default: []). The library folders will be
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo. .. admonition:: Example

                    The following call >>> Estimator(entry_point='train.py',
                    dependencies=['my/libs/common', 'virtual-env']) results in
                    the following inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ train.py
                    >>>     |------ common
                    >>>     |------ virtual-env
            enable_network_isolation (bool): Specifies whether container will
                run in network isolation mode. Network isolation mode restricts
                the container access to outside networks (such as the internet).
                The container does not make any inbound or outbound network
                calls. If True, a channel named "code" will be created for any
                user entry script for training. The user entry script, files in
                source_dir (if specified), and dependencies will be uploaded in
                a tar to S3. Also known as internet-free mode (default: `False`
                ).
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
            **kwargs: Additional kwargs passed to the ``EstimatorBase``
                constructor.
        """
        super(Framework, self).__init__(**kwargs)
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
        if enable_cloudwatch_metrics:
            warnings.warn(
                "enable_cloudwatch_metrics is now deprecated and will be removed in the future.",
                DeprecationWarning,
            )
        self.enable_cloudwatch_metrics = False
        self.container_log_level = container_log_level
        self.code_location = code_location
        self.image_name = image_name
        self._enable_network_isolation = enable_network_isolation

        self.uploaded_code = None

        self._hyperparameters = hyperparameters or {}

    def enable_network_isolation(self):
        """Return True if this Estimator can use network isolation to run.

        Returns:
            bool: Whether this Estimator can use network isolation or not.
        """
        return self._enable_network_isolation

    def _prepare_for_training(self, job_name=None):
        """Set hyperparameters needed for training. This method will also
        validate ``source_dir``.

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
        self._hyperparameters[CLOUDWATCH_METRICS_PARAM_NAME] = self.enable_cloudwatch_metrics
        self._hyperparameters[CONTAINER_LOG_LEVEL_PARAM_NAME] = self.container_log_level
        self._hyperparameters[JOB_NAME_PARAM_NAME] = self._current_job_name
        self._hyperparameters[SAGEMAKER_REGION_PARAM_NAME] = self.sagemaker_session.boto_region_name

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
        )

    def _model_source_dir(self):
        """Get the appropriate value to pass as source_dir to model constructor
        on deploying

        Returns:
            str: Either a local or an S3 path pointing to the source_dir to be
            used for code by the model to be deployed
        """
        return (
            self.source_dir if self.sagemaker_session.local_mode else self.uploaded_code.s3_prefix
        )

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
        """Convert the job description to init params that can be handled by the
        class constructor

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
        init_params["enable_cloudwatch_metrics"] = json.loads(
            init_params["hyperparameters"].get(CLOUDWATCH_METRICS_PARAM_NAME)
        )
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

    def train_image(self):
        """Return the Docker image to use for training.

        The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does
        the model training, calls this method to find the image to use for model
        training.

        Returns:
            str: The URI of the Docker image.
        """
        if self.image_name:
            return self.image_name
        return create_image_uri(
            self.sagemaker_session.boto_region_name,
            self.__framework_name__,
            self.train_instance_type,
            self.framework_version,  # pylint: disable=no-member
            py_version=self.py_version,  # pylint: disable=no-member
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

        If the training job is in progress, attach will block and display log
        messages from the training job, until the training job completes.

        Examples:
            >>> my_estimator.fit(wait=False)
            >>> training_job_name = my_estimator.latest_training_job.name
            Later on:
            >>> attached_estimator = Estimator.attach(training_job_name)
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
        """
        Args:
            hyperparameters:
        """
        return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}

    @classmethod
    def _update_init_params(cls, hp, tf_arguments):
        """
        Args:
            hp:
            tf_arguments:
        """
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
    ):
        """Return a ``Transformer`` that uses a SageMaker Model based on the
        training job. It reuses the SageMaker Session and base job name used by
        the Estimator.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in
                a single request (default: None). Valid values: 'MULTI_RECORD'
                and 'SINGLE_RECORD'.
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
                should be executed as the entry point to training. If not specified, the training
                entry point is used.
            vpc_config_override (dict[str, list[str]]): Optional override for
                the VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

        Returns:
            sagemaker.transformer.Transformer: a ``Transformer`` object that can be used to start a
                SageMaker Batch Transform job.
        """
        role = role or self.role
        tags = tags or self.tags

        if self.latest_training_job is not None:
            model = self.create_model(
                role=role,
                model_server_workers=model_server_workers,
                entry_point=entry_point,
                vpc_config_override=vpc_config_override,
                model_kms_key=self.output_kms_key,
            )
            model._create_sagemaker_model(instance_type, tags=tags)

            model_name = model.name
            transform_env = model.env.copy()
            if env is not None:
                transform_env.update(env)
        else:
            logging.warning(
                "No finished training job found associated with this estimator. Please make sure "
                "this estimator is only used for building workflow config"
            )
            model_name = self._current_job_name
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


def _s3_uri_prefix(channel_name, s3_data):
    """
    Args:
        channel_name:
        s3_data:
    """
    if isinstance(s3_data, s3_input):
        s3_uri = s3_data.config["DataSource"]["S3DataSource"]["S3Uri"]
    else:
        s3_uri = s3_data
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expecting an s3 uri. Got {}".format(s3_uri))
    return {channel_name: s3_uri[5:]}


# E.g. 's3://bucket/data' would return 'bucket/data'.
# Also accepts other valid input types, e.g. dict and s3_input.
def _s3_uri_without_prefix_from_input(input_data):
    # Unpack an input_config object from a dict if a dict was passed in.
    """
    Args:
        input_data:
    """
    if isinstance(input_data, dict):
        response = {}
        for channel_name, channel_s3_uri in input_data.items():
            response.update(_s3_uri_prefix(channel_name, channel_s3_uri))
        return response
    if isinstance(input_data, str):
        return _s3_uri_prefix("training", input_data)
    if isinstance(input_data, s3_input):
        return _s3_uri_prefix("training", input_data)
    raise ValueError(
        "Unrecognized type for S3 input data config - not str or s3_input: {}".format(input_data)
    )
