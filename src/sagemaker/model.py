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
import logging

import sagemaker

from sagemaker.fw_utils import tar_and_upload_dir, parse_s3_url
from sagemaker.session import Session
from sagemaker.utils import name_from_image


class Model(object):
    """An SageMaker ``Model`` that can be deployed to an ``Endpoint``."""

    def __init__(self, model_data, image, role, predictor_cls=None, env=None, name=None, sagemaker_session=None):
        """Initialize an SageMaker ``Model``.

        Args:
            model_data (str): The S3 location of a SageMaker model data ``.tar.gz`` file.
            image (str): A Docker image URI.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                After the endpoint is created, the inference code might use the IAM role,
                if it needs to access an AWS resource.
            predictor_cls (callable[string, sagemaker.session.Session]): A function to call to create
               a predictor (default: None). If not None, ``deploy`` will return the result of invoking
               this function on the created endpoint name.
            env (dict[str, str]): Environment variables to run with ``image`` when hosted in SageMaker (default: None).
            name (str): The model name. If None, a default model name will be selected on each ``deploy``.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for SageMaker
               interactions (default: None). If not specified, one is created using the default AWS configuration chain.
        """
        self.model_data = model_data
        self.image = image
        self.role = role
        self.predictor_cls = predictor_cls
        self.env = env or {}
        self.name = name
        self.sagemaker_session = sagemaker_session or Session()
        self._model_name = None

    def prepare_container_def(self, instance_type):
        """Return a dict created by ``sagemaker.container_def()`` for deploying this model to a specified instance type.

        Subclasses can override this to provide custom container definitions for
        deployment to a specific instance type. Called by ``deploy()``.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.

        Returns:
            dict: A container definition object usable with the CreateModel API.
        """
        return sagemaker.container_def(self.image, self.model_data, self.env)

    def deploy(self, initial_instance_count, instance_type, endpoint_name=None):
        """Deploy this ``Model`` to an ``Endpoint`` and optionally return a ``Predictor``.

        Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an ``Endpoint`` from this ``Model``.
        If ``self.predictor_cls`` is not None, this method returns a the result of invoking
        ``self.predictor_cls`` on the created endpoint name.

        The name of the created endpoint is accessible in the ``endpoint_name``
        field of this ``Model`` after deploy returns.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.
            initial_instance_count (int): The initial number of instances to run in the
                ``Endpoint`` created from this ``Model``.
            endpoint_name (str): The name of the endpoint to create (default: None).
                If not specified, a unique endpoint name will be created.

        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of ``self.predictor_cls`` on
                the created endpoint name, if ``self.predictor_cls`` is not None. Otherwise, return None.
        """
        container_def = self.prepare_container_def(instance_type)
        model_name = self.name or name_from_image(container_def['Image'])
        self.sagemaker_session.create_model(model_name, self.role, container_def)
        production_variant = sagemaker.production_variant(model_name, instance_type, initial_instance_count)
        self.endpoint_name = endpoint_name or model_name
        self.sagemaker_session.endpoint_from_production_variants(self.endpoint_name, [production_variant])
        if self.predictor_cls:
            return self.predictor_cls(self.endpoint_name, self.sagemaker_session)


SCRIPT_PARAM_NAME = 'sagemaker_program'
DIR_PARAM_NAME = 'sagemaker_submit_directory'
CLOUDWATCH_METRICS_PARAM_NAME = 'sagemaker_enable_cloudwatch_metrics'
CONTAINER_LOG_LEVEL_PARAM_NAME = 'sagemaker_container_log_level'
JOB_NAME_PARAM_NAME = 'sagemaker_job_name'
MODEL_SERVER_WORKERS_PARAM_NAME = 'sagemaker_model_server_workers'
SAGEMAKER_REGION_PARAM_NAME = 'sagemaker_region'


class FrameworkModel(Model):
    """A Model for working with an SageMaker ``Framework``.

    This class hosts user-defined code in S3 and sets code location and configuration in model environment variables.
    """

    def __init__(self, model_data, image, role, entry_point, source_dir=None, predictor_cls=None, env=None, name=None,
                 enable_cloudwatch_metrics=False, container_log_level=logging.INFO, code_location=None,
                 sagemaker_session=None):
        """Initialize a ``FrameworkModel``.

        Args:
            model_data (str): The S3 location of a SageMaker model data ``.tar.gz`` file.
            image (str): A Docker image URI.
            role (str): An IAM role name or ARN for SageMaker to access AWS resources on your behalf.
            entry_point (str): Path (absolute or relative) to the Python source file which should be executed
                as the entry point to model hosting. This should be compatible with either Python 2.7 or Python 3.5.
            source_dir (str): Path (absolute or relative) to a directory with any other training
                source code dependencies aside from tne entry point file (default: None). Structure within this
                directory will be preserved when training on SageMaker.
            predictor_cls (callable[string, sagemaker.session.Session]): A function to call to create
               a predictor (default: None). If not None, ``deploy`` will return the result of invoking
               this function on the created endpoint name.
            env (dict[str, str]): Environment variables to run with ``image`` when hosted in SageMaker
               (default: None).
            name (str): The model name. If None, a default model name will be selected on each ``deploy``.
            enable_cloudwatch_metrics (bool): Whether training and hosting containers will
               generate CloudWatch metrics under the AWS/SageMakerContainer namespace (default: False).
            container_log_level (int): Log level to use within the container (default: logging.INFO).
                Valid values are defined in the Python logging module.
            code_location (str): Name of the S3 bucket where custom code is uploaded (default: None).
                If not specified, default bucket created by ``sagemaker.session.Session`` is used.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for SageMaker
               interactions (default: None). If not specified, one is created using the default AWS configuration chain.
        """
        super(FrameworkModel, self).__init__(model_data, image, role, predictor_cls=predictor_cls, env=env, name=name,
                                             sagemaker_session=sagemaker_session)
        self.entry_point = entry_point
        self.source_dir = source_dir
        self.enable_cloudwatch_metrics = enable_cloudwatch_metrics
        self.container_log_level = container_log_level
        if code_location:
            self.bucket, self.key_prefix = parse_s3_url(code_location)
        else:
            self.bucket, self.key_prefix = None, None

    def prepare_container_def(self, instance_type):
        """Return a container definition with framework configuration set in model environment variables.

        This also uploads user-supplied code to S3.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.

        Returns:
            dict[str, str]: A container definition object usable with the CreateModel API.
        """
        self._upload_code(self.key_prefix or self.name or name_from_image(self.image))
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())
        return sagemaker.container_def(self.image, self.model_data, deploy_env)

    def _upload_code(self, key_prefix):
        self.uploaded_code = tar_and_upload_dir(session=self.sagemaker_session.boto_session,
                                                bucket=self.bucket or self.sagemaker_session.default_bucket(),
                                                s3_key_prefix=key_prefix,
                                                script=self.entry_point,
                                                directory=self.source_dir)

    def _framework_env_vars(self):
        return {SCRIPT_PARAM_NAME.upper(): self.uploaded_code.script_name,
                DIR_PARAM_NAME.upper(): self.uploaded_code.s3_prefix,
                CLOUDWATCH_METRICS_PARAM_NAME.upper(): str(self.enable_cloudwatch_metrics).lower(),
                CONTAINER_LOG_LEVEL_PARAM_NAME.upper(): str(self.container_log_level),
                SAGEMAKER_REGION_PARAM_NAME.upper(): self.sagemaker_session.boto_session.region_name}
