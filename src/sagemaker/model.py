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

import json
import logging
import os

import sagemaker
from sagemaker import fw_utils, local, session, utils, git_utils
from sagemaker.fw_utils import UploadedCode
from sagemaker.transformer import Transformer

LOGGER = logging.getLogger("sagemaker")

NEO_ALLOWED_FRAMEWORKS = set(
    ["mxnet", "tensorflow", "keras", "pytorch", "onnx", "xgboost", "tflite"]
)

NEO_IMAGE_ACCOUNT = {
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "eu-central-1": "746233611703",
    "eu-north-1": "601324751636",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-east-1": "110948597952",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "ap-south-1": "763008648453",
    "sa-east-1": "756306329178",
    "ca-central-1": "464438896020",
    "me-south-1": "836785723513",
    "cn-north-1": "472730292857",
    "cn-northwest-1": "474822919863",
    "us-gov-west-1": "263933020539",
}

INFERENTIA_INSTANCE_PREFIX = "ml_inf"


class Model(object):
    """A SageMaker ``Model`` that can be deployed to an ``Endpoint``."""

    def __init__(
        self,
        model_data,
        image,
        role=None,
        predictor_cls=None,
        env=None,
        name=None,
        vpc_config=None,
        sagemaker_session=None,
        enable_network_isolation=False,
        model_kms_key=None,
    ):
        """Initialize an SageMaker ``Model``.

        Args:
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            image (str): A Docker image URI.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role if it needs to access some AWS resources.
                It can be null if this is being used to create a Model to pass
                to a ``PipelineModel`` which has its own Role field. (default:
                None)
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name.
            env (dict[str, str]): Environment variables to run with ``image``
                when hosted in SageMaker (default: None).
            name (str): The model name. If None, a default model name will be
                selected on each ``deploy``.
            vpc_config (dict[str, list[str]]): The VpcConfig set on the model
                (default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            enable_network_isolation (Boolean): Default False. if True, enables
                network isolation in the endpoint, isolating the model
                container. No inbound or outbound network calls can be made to
                or from the model container.
            model_kms_key (str): KMS key ARN used to encrypt the repacked
                model archive file if the model is repacked
        """
        self.model_data = model_data
        self.image = image
        self.role = role
        self.predictor_cls = predictor_cls
        self.env = env or {}
        self.name = name
        self.vpc_config = vpc_config
        self.sagemaker_session = sagemaker_session
        self._model_name = None
        self.endpoint_name = None
        self._is_compiled_model = False
        self._enable_network_isolation = enable_network_isolation
        self.model_kms_key = model_kms_key

    def _init_sagemaker_session_if_does_not_exist(self, instance_type):
        """Set ``self.sagemaker_session`` to be a ``LocalSession`` or
        ``Session`` if it is not already. The type of session object is
        determined by the instance type.
        """
        if self.sagemaker_session:
            return

        if instance_type in ("local", "local_gpu"):
            self.sagemaker_session = local.LocalSession()
        else:
            self.sagemaker_session = session.Session()

    def prepare_container_def(
        self, instance_type, accelerator_type=None
    ):  # pylint: disable=unused-argument
        """Return a dict created by ``sagemaker.container_def()`` for deploying
        this model to a specified instance type.

        Subclasses can override this to provide custom container definitions
        for deployment to a specific instance type. Called by ``deploy()``.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. For example, 'ml.eia1.medium'.

        Returns:
            dict: A container definition object usable with the CreateModel API.
        """
        return sagemaker.container_def(self.image, self.model_data, self.env)

    def enable_network_isolation(self):
        """Whether to enable network isolation when creating this Model

        Returns:
            bool: If network isolation should be enabled or not.
        """
        return self._enable_network_isolation

    def _create_sagemaker_model(self, instance_type, accelerator_type=None, tags=None):
        """Create a SageMaker Model Entity

        Args:
            instance_type (str): The EC2 instance type that this Model will be
                used for, this is only used to determine if the image needs GPU
                support or not.
            accelerator_type (str): Type of Elastic Inference accelerator to
                attach to an endpoint for model loading and inference, for
                example, 'ml.eia1.medium'. If not specified, no Elastic
                Inference accelerator will be attached to the endpoint.
            tags (List[dict[str, str]]): Optional. The list of tags to add to
                the model. Example: >>> tags = [{'Key': 'tagname', 'Value':
                'tagvalue'}] For more information about tags, see
                https://boto3.amazonaws.com/v1/documentation
                /api/latest/reference/services/sagemaker.html#SageMaker.Client.add_tags
        """
        container_def = self.prepare_container_def(instance_type, accelerator_type=accelerator_type)
        self.name = self.name or utils.name_from_image(container_def["Image"])
        enable_network_isolation = self.enable_network_isolation()

        self._init_sagemaker_session_if_does_not_exist(instance_type)
        self.sagemaker_session.create_model(
            self.name,
            self.role,
            container_def,
            vpc_config=self.vpc_config,
            enable_network_isolation=enable_network_isolation,
            tags=tags,
        )

    def _framework(self):
        """Placeholder docstring"""
        return getattr(self, "__framework_name__", None)

    def _get_framework_version(self):
        """Placeholder docstring"""
        return getattr(self, "framework_version", None)

    def _compilation_job_config(
        self,
        target_instance_type,
        input_shape,
        output_path,
        role,
        compile_max_run,
        job_name,
        framework,
        tags,
    ):
        """
        Args:
            target_instance_type:
            input_shape:
            output_path:
            role:
            compile_max_run:
            job_name:
            framework:
            tags:
        """
        input_model_config = {
            "S3Uri": self.model_data,
            "DataInputConfig": input_shape
            if not isinstance(input_shape, dict)
            else json.dumps(input_shape),
            "Framework": framework,
        }
        role = self.sagemaker_session.expand_role(role)
        output_model_config = {
            "TargetDevice": target_instance_type,
            "S3OutputLocation": output_path,
        }

        return {
            "input_model_config": input_model_config,
            "output_model_config": output_model_config,
            "role": role,
            "stop_condition": {"MaxRuntimeInSeconds": compile_max_run},
            "tags": tags,
            "job_name": job_name,
        }

    def check_neo_region(self, region):
        """Check if this ``Model`` in the available region where neo support.

        Args:
            region (str): Specifies the region where want to execute compilation

        Returns:
            bool: boolean value whether if neo is available in the specified
            region
        """
        if region in NEO_IMAGE_ACCOUNT:
            return True
        return False

    def _neo_image_account(self, region):
        """
        Args:
            region:
        """
        if region not in NEO_IMAGE_ACCOUNT:
            raise ValueError(
                "Neo is not currently supported in {}, "
                "valid regions: {}".format(region, NEO_IMAGE_ACCOUNT.keys())
            )
        return NEO_IMAGE_ACCOUNT[region]

    def _neo_image(self, region, target_instance_type, framework, framework_version):
        """
        Args:
            region:
            target_instance_type:
            framework:
            framework_version:
        """
        return fw_utils.create_image_uri(
            region,
            "neo-" + framework.lower(),
            target_instance_type.replace("_", "."),
            framework_version,
            py_version="py3",
            account=self._neo_image_account(region),
        )

    def _inferentia_image(self, region, target_instance_type, framework, framework_version):
        """
                Args:
                    region:
                    target_instance_type:
                    framework:
                    framework_version:
                """
        return fw_utils.create_image_uri(
            region,
            "neo-" + framework.lower(),
            target_instance_type.replace("_", "."),
            framework_version,
            py_version="py3",
            account=self._neo_image_account(region),
        )

    def compile(
        self,
        target_instance_family,
        input_shape,
        output_path,
        role,
        tags=None,
        job_name=None,
        compile_max_run=5 * 60,
        framework=None,
        framework_version=None,
    ):
        """Compile this ``Model`` with SageMaker Neo.

        Args:
            target_instance_family (str): Identifies the device that you want to
                run your model after compilation, for example: ml_c5. For allowed
                strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
            input_shape (dict): Specifies the name and shape of the expected
                inputs for your trained model in json dictionary form, for
                example: {'data': [1,3,1024,1024]}, or {'var1': [1,1,28,28],
                'var2': [1,1,28,28]}
            output_path (str): Specifies where to store the compiled model
            role (str): Execution role
            tags (list[dict]): List of tags for labeling a compilation job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            job_name (str): The name of the compilation job
            compile_max_run (int): Timeout in seconds for compilation (default:
                3 * 60). After this amount of time Amazon SageMaker Neo
                terminates the compilation job regardless of its current status.
            framework (str): The framework that is used to train the original
                model. Allowed values: 'mxnet', 'tensorflow', 'keras', 'pytorch',
                'onnx', 'xgboost'
            framework_version (str):

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """
        framework = self._framework() or framework
        if framework is None:
            raise ValueError(
                "You must specify framework, allowed values {}".format(NEO_ALLOWED_FRAMEWORKS)
            )
        if framework not in NEO_ALLOWED_FRAMEWORKS:
            raise ValueError(
                "You must provide valid framework, allowed values {}".format(NEO_ALLOWED_FRAMEWORKS)
            )
        if job_name is None:
            raise ValueError("You must provide a compilation job name")

        framework = framework.upper()
        framework_version = self._get_framework_version() or framework_version

        self._init_sagemaker_session_if_does_not_exist(target_instance_family)
        config = self._compilation_job_config(
            target_instance_family,
            input_shape,
            output_path,
            role,
            compile_max_run,
            job_name,
            framework,
            tags,
        )
        self.sagemaker_session.compile_model(**config)
        job_status = self.sagemaker_session.wait_for_compilation_job(job_name)
        self.model_data = job_status["ModelArtifacts"]["S3ModelArtifacts"]
        if target_instance_family.startswith("ml_"):
            self.image = self._neo_image(
                self.sagemaker_session.boto_region_name,
                target_instance_family,
                framework,
                framework_version,
            )
            self._is_compiled_model = True
        elif target_instance_family.startswith(INFERENTIA_INSTANCE_PREFIX):
            self.image = self._inferentia_image(
                self.sagemaker_session.boto_region_name,
                target_instance_family,
                framework,
                framework_version,
            )
            self._is_compiled_model = True
        else:
            LOGGER.warning(
                "The instance type %s is not supported to deploy via SageMaker,"
                "please deploy the model manually.",
                target_instance_family,
            )
        return self

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        accelerator_type=None,
        endpoint_name=None,
        update_endpoint=False,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config=None,
    ):
        """Deploy this ``Model`` to an ``Endpoint`` and optionally return a
        ``Predictor``.

        Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an
        ``Endpoint`` from this ``Model``. If ``self.predictor_cls`` is not None,
        this method returns a the result of invoking ``self.predictor_cls`` on
        the created endpoint name.

        The name of the created model is accessible in the ``name`` field of
        this ``Model`` after deploy returns

        The name of the created endpoint is accessible in the
        ``endpoint_name`` field of this ``Model`` after deploy returns.

        Args:
            initial_instance_count (int): The initial number of instances to run
                in the ``Endpoint`` created from this ``Model``.
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge', or 'local' for local mode.
            accelerator_type (str): Type of Elastic Inference accelerator to
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
            update_endpoint (bool): Flag to update the model in an existing
                Amazon SageMaker endpoint. If True, this will deploy a new
                EndpointConfig to an already existing endpoint and delete
                resources corresponding to the previous EndpointConfig. If
                False, a new endpoint will be created. Default: False
            tags (List[dict[str, str]]): The list of tags to attach to this
                specific endpoint.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                data on the storage volume attached to the instance hosting the
                endpoint.
            wait (bool): Whether the call should wait until the deployment of
                this model completes (default: True).
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.

        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of
                ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
                is not None. Otherwise, return None.
        """
        self._init_sagemaker_session_if_does_not_exist(instance_type)

        if self.role is None:
            raise ValueError("Role can not be null for deploying a model")

        if instance_type.startswith("ml.inf") and not self._is_compiled_model:
            LOGGER.warning(
                "Your model is not compiled. Please compile your model before using Inferentia."
            )

        compiled_model_suffix = "-".join(instance_type.split(".")[:-1])
        if self._is_compiled_model:
            name_prefix = self.name or utils.name_from_image(self.image)
            self.name = "{}{}".format(name_prefix, compiled_model_suffix)

        self._create_sagemaker_model(instance_type, accelerator_type, tags)
        production_variant = sagemaker.production_variant(
            self.name, instance_type, initial_instance_count, accelerator_type=accelerator_type
        )
        if endpoint_name:
            self.endpoint_name = endpoint_name
        else:
            self.endpoint_name = self.name
            if self._is_compiled_model and not self.endpoint_name.endswith(compiled_model_suffix):
                self.endpoint_name += compiled_model_suffix

        data_capture_config_dict = None
        if data_capture_config is not None:
            data_capture_config_dict = data_capture_config._to_request_dict()

        if update_endpoint:
            endpoint_config_name = self.sagemaker_session.create_endpoint_config(
                name=self.name,
                model_name=self.name,
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                accelerator_type=accelerator_type,
                tags=tags,
                kms_key=kms_key,
                data_capture_config_dict=data_capture_config_dict,
            )
            self.sagemaker_session.update_endpoint(
                self.endpoint_name, endpoint_config_name, wait=wait
            )
        else:
            self.sagemaker_session.endpoint_from_production_variants(
                name=self.endpoint_name,
                production_variants=[production_variant],
                tags=tags,
                kms_key=kms_key,
                wait=wait,
                data_capture_config_dict=data_capture_config_dict,
            )

        if self.predictor_cls:
            return self.predictor_cls(self.endpoint_name, self.sagemaker_session)
        return None

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
        volume_kms_key=None,
    ):
        """Return a ``Transformer`` that uses this Model.

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
            volume_kms_key (str): Optional. KMS key ID for encrypting the volume
                attached to the ML compute instance (default: None).
        """
        self._init_sagemaker_session_if_does_not_exist(instance_type)

        self._create_sagemaker_model(instance_type, tags=tags)
        if self.enable_network_isolation():
            env = None

        return Transformer(
            self.name,
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
            base_transform_job_name=self.name,
            volume_kms_key=volume_kms_key,
            sagemaker_session=self.sagemaker_session,
        )

    def delete_model(self):
        """Delete an Amazon SageMaker Model.

        Raises:
            ValueError: if the model is not created yet.
        """
        if self.name is None:
            raise ValueError(
                "The SageMaker model must be created first before attempting to delete."
            )
        self.sagemaker_session.delete_model(self.name)


SCRIPT_PARAM_NAME = "sagemaker_program"
DIR_PARAM_NAME = "sagemaker_submit_directory"
CLOUDWATCH_METRICS_PARAM_NAME = "sagemaker_enable_cloudwatch_metrics"
CONTAINER_LOG_LEVEL_PARAM_NAME = "sagemaker_container_log_level"
JOB_NAME_PARAM_NAME = "sagemaker_job_name"
MODEL_SERVER_WORKERS_PARAM_NAME = "sagemaker_model_server_workers"
SAGEMAKER_REGION_PARAM_NAME = "sagemaker_region"
SAGEMAKER_OUTPUT_LOCATION = "sagemaker_s3_output"


class FrameworkModel(Model):
    """A Model for working with an SageMaker ``Framework``.

    This class hosts user-defined code in S3 and sets code location and
    configuration in model environment variables.
    """

    def __init__(
        self,
        model_data,
        image,
        role,
        entry_point,
        source_dir=None,
        predictor_cls=None,
        env=None,
        name=None,
        enable_cloudwatch_metrics=False,
        container_log_level=logging.INFO,
        code_location=None,
        sagemaker_session=None,
        dependencies=None,
        git_config=None,
        **kwargs
    ):
        """Initialize a ``FrameworkModel``.

        Args:
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            image (str): A Docker image URI.
            role (str): An IAM role name or ARN for SageMaker to access AWS
                resources on your behalf.
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to model
                hosting. This should be compatible with either Python 2.7 or
                Python 3.5. If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo.
                Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='src/inference.py'.
            source_dir (str): Path (absolute or relative) to a directory with
                any other training source code dependencies aside from the entry
                point file (default: None). Structure within this directory will
                be preserved when training on SageMaker. If 'git_config' is
                provided, 'source_dir' should be a relative location to a
                directory in the Git repo. If the directory points to S3, no
                code will be uploaded and the S3 location will be used instead.
                .. admonition:: Example

                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='inference.py', source_dir='src'.
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name.
            env (dict[str, str]): Environment variables to run with ``image``
                when hosted in SageMaker (default: None).
            name (str): The model name. If None, a default model name will be
                selected on each ``deploy``.
            enable_cloudwatch_metrics (bool): Whether training and hosting
                containers will generate CloudWatch metrics under the
                AWS/SageMakerContainer namespace (default: False).
            container_log_level (int): Log level to use within the container
                (default: logging.INFO). Valid values are defined in the Python
                logging module.
            code_location (str): Name of the S3 bucket where custom code is
                uploaded (default: None). If not specified, default bucket
                created by ``sagemaker.session.Session`` is used.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            dependencies (list[str]): A list of paths to directories (absolute
                or relative) with any additional libraries that will be exported
                to the container (default: []). The library folders will be
                copied to SageMaker in the same folder where the entrypoint is
                copied. If 'git_config' is provided, 'dependencies' should be a
                list of relative locations to directories with any additional
                libraries needed in the Git repo. If the ```source_dir``` points
                to S3, code will be uploaded and the S3 location will be used
                instead. .. admonition:: Example

                    The following call >>> Estimator(entry_point='inference.py',
                    dependencies=['my/libs/common', 'virtual-env']) results in
                    the following inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ inference.py
                    >>>     |------ common
                    >>>     |------ virtual-env
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
            **kwargs: Keyword arguments passed to the ``Model`` initializer.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.Model`.
        """
        super(FrameworkModel, self).__init__(
            model_data,
            image,
            role,
            predictor_cls=predictor_cls,
            env=env,
            name=name,
            sagemaker_session=sagemaker_session,
            **kwargs
        )
        self.entry_point = entry_point
        self.source_dir = source_dir
        self.dependencies = dependencies or []
        self.git_config = git_config
        self.enable_cloudwatch_metrics = enable_cloudwatch_metrics
        self.container_log_level = container_log_level
        if code_location:
            self.bucket, self.key_prefix = fw_utils.parse_s3_url(code_location)
        else:
            self.bucket, self.key_prefix = None, None
        if self.git_config:
            updates = git_utils.git_clone_repo(
                self.git_config, self.entry_point, self.source_dir, self.dependencies
            )
            self.entry_point = updates["entry_point"]
            self.source_dir = updates["source_dir"]
            self.dependencies = updates["dependencies"]
        self.uploaded_code = None
        self.repacked_model_data = None

    def prepare_container_def(
        self, instance_type, accelerator_type=None
    ):  # pylint disable=unused-argument
        """Return a container definition with framework configuration set in
        model environment variables.

        This also uploads user-supplied code to S3.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. For example, 'ml.eia1.medium'.

        Returns:
            dict[str, str]: A container definition object usable with the
            CreateModel API.
        """
        deploy_key_prefix = fw_utils.model_code_key_prefix(self.key_prefix, self.name, self.image)
        self._upload_code(deploy_key_prefix)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())
        return sagemaker.container_def(self.image, self.model_data, deploy_env)

    def _upload_code(self, key_prefix, repack=False):
        """
        Args:
            key_prefix:
            repack:
        """
        local_code = utils.get_config_value("local.local_code", self.sagemaker_session.config)
        if self.sagemaker_session.local_mode and local_code:
            self.uploaded_code = None
        elif not repack:
            bucket = self.bucket or self.sagemaker_session.default_bucket()
            self.uploaded_code = fw_utils.tar_and_upload_dir(
                session=self.sagemaker_session.boto_session,
                bucket=bucket,
                s3_key_prefix=key_prefix,
                script=self.entry_point,
                directory=self.source_dir,
                dependencies=self.dependencies,
            )

        if repack:
            bucket = self.bucket or self.sagemaker_session.default_bucket()
            repacked_model_data = "s3://" + "/".join([bucket, key_prefix, "model.tar.gz"])

            utils.repack_model(
                inference_script=self.entry_point,
                source_directory=self.source_dir,
                dependencies=self.dependencies,
                model_uri=self.model_data,
                repacked_model_uri=repacked_model_data,
                sagemaker_session=self.sagemaker_session,
                kms_key=self.model_kms_key,
            )

            self.repacked_model_data = repacked_model_data
            self.uploaded_code = UploadedCode(
                s3_prefix=self.repacked_model_data, script_name=os.path.basename(self.entry_point)
            )

    def _framework_env_vars(self):
        """Placeholder docstring"""
        if self.uploaded_code:
            script_name = self.uploaded_code.script_name
            if self.enable_network_isolation():
                dir_name = "/opt/ml/model/code"
            else:
                dir_name = self.uploaded_code.s3_prefix
        elif self.entry_point is not None:
            script_name = self.entry_point
            dir_name = "file://" + self.source_dir
        else:
            script_name = None
            dir_name = None

        return {
            SCRIPT_PARAM_NAME.upper(): script_name,
            DIR_PARAM_NAME.upper(): dir_name,
            CLOUDWATCH_METRICS_PARAM_NAME.upper(): str(self.enable_cloudwatch_metrics).lower(),
            CONTAINER_LOG_LEVEL_PARAM_NAME.upper(): str(self.container_log_level),
            SAGEMAKER_REGION_PARAM_NAME.upper(): self.sagemaker_session.boto_region_name,
        }


class ModelPackage(Model):
    """A SageMaker ``Model`` that can be deployed to an ``Endpoint``."""

    def __init__(self, role, model_data=None, algorithm_arn=None, model_package_arn=None, **kwargs):
        """Initialize a SageMaker ModelPackage.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file. Must be provided if algorithm_arn is provided.
            algorithm_arn (str): algorithm arn used to train the model, can be
                just the name if your account owns the algorithm. Must also
                provide ``model_data``.
            model_package_arn (str): An existing SageMaker Model Package arn,
                can be just the name if your account owns the Model Package.
                ``model_data`` is not required.
            **kwargs: Additional kwargs passed to the Model constructor.
        """
        super(ModelPackage, self).__init__(role=role, model_data=model_data, image=None, **kwargs)

        if model_package_arn and algorithm_arn:
            raise ValueError(
                "model_package_arn and algorithm_arn are mutually exclusive."
                "Both were provided: model_package_arn: %s algorithm_arn: %s"
                % (model_package_arn, algorithm_arn)
            )

        if model_package_arn is None and algorithm_arn is None:
            raise ValueError(
                "either model_package_arn or algorithm_arn is required." " None was provided."
            )

        self.algorithm_arn = algorithm_arn
        if self.algorithm_arn is not None:
            if model_data is None:
                raise ValueError("model_data must be provided with algorithm_arn")
            self.model_data = model_data

        self.model_package_arn = model_package_arn
        self._created_model_package_name = None

    def _create_sagemaker_model_package(self):
        """Placeholder docstring"""
        if self.algorithm_arn is None:
            raise ValueError("No algorithm_arn was provided to create a SageMaker Model Pacakge")

        name = self.name or utils.name_from_base(self.algorithm_arn.split("/")[-1])
        description = "Model Package created from training with %s" % self.algorithm_arn
        self.sagemaker_session.create_model_package_from_algorithm(
            name, description, self.algorithm_arn, self.model_data
        )
        return name

    def enable_network_isolation(self):
        """Whether to enable network isolation when creating a model out of this
        ModelPackage

        Returns:
            bool: If network isolation should be enabled or not.
        """
        return self._is_marketplace()

    def _is_marketplace(self):
        """Placeholder docstring"""
        model_package_name = self.model_package_arn or self._created_model_package_name
        if model_package_name is None:
            return True

        # Models can lazy-init sagemaker_session until deploy() is called to support
        # LocalMode so we must make sure we have an actual session to describe the model package.
        sagemaker_session = self.sagemaker_session or sagemaker.Session()

        model_package_desc = sagemaker_session.sagemaker_client.describe_model_package(
            ModelPackageName=model_package_name
        )
        for container in model_package_desc["InferenceSpecification"]["Containers"]:
            if "ProductId" in container:
                return True
        return False

    def _create_sagemaker_model(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Create a SageMaker Model Entity

        Args:
            args: Positional arguments coming from the caller. This class does not require
                any so they are ignored.

            kwargs: Keyword arguments coming from the caller. This class does not require
                any so they are ignored.
        """
        if self.algorithm_arn:
            # When ModelPackage is created using an algorithm_arn we need to first
            # create a ModelPackage. If we had already created one then its fine to re-use it.
            if self._created_model_package_name is None:
                model_package_name = self._create_sagemaker_model_package()
                self.sagemaker_session.wait_for_model_package(model_package_name)
                self._created_model_package_name = model_package_name
            model_package_name = self._created_model_package_name
        else:
            # When a ModelPackageArn is provided we just create the Model
            model_package_name = self.model_package_arn

        container_def = {"ModelPackageName": model_package_name}

        if self.env != {}:
            container_def["Environment"] = self.env

        model_package_short_name = model_package_name.split("/")[-1]
        enable_network_isolation = self.enable_network_isolation()
        self.name = self.name or utils.name_from_base(model_package_short_name)
        self.sagemaker_session.create_model(
            self.name,
            self.role,
            container_def,
            vpc_config=self.vpc_config,
            enable_network_isolation=enable_network_isolation,
        )
