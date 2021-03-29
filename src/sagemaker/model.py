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
import re

import sagemaker
from sagemaker import (
    fw_utils,
    image_uris,
    local,
    s3,
    session,
    utils,
    git_utils,
)
from sagemaker.deprecations import removed_kwargs
from sagemaker.transformer import Transformer

LOGGER = logging.getLogger("sagemaker")

NEO_ALLOWED_FRAMEWORKS = set(
    ["mxnet", "tensorflow", "keras", "pytorch", "onnx", "xgboost", "tflite"]
)


class Model(object):
    """A SageMaker ``Model`` that can be deployed to an ``Endpoint``."""

    def __init__(
        self,
        image_uri,
        model_data=None,
        role=None,
        predictor_cls=None,
        env=None,
        name=None,
        vpc_config=None,
        sagemaker_session=None,
        enable_network_isolation=False,
        model_kms_key=None,
        image_config=None,
    ):
        """Initialize an SageMaker ``Model``.

        Args:
            image_uri (str): A Docker image URI.
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file (default: None).
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
            env (dict[str, str]): Environment variables to run with ``image_uri``
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
            image_config (dict[str, str]): Specifies whether the image of
                model container is pulled from ECR, or private registry in your
                VPC. By default it is set to pull model container image from
                ECR. (default: None).
        """
        self.model_data = model_data
        self.image_uri = image_uri
        self.role = role
        self.predictor_cls = predictor_cls
        self.env = env or {}
        self.name = name
        self._base_name = None
        self.vpc_config = vpc_config
        self.sagemaker_session = sagemaker_session
        self.endpoint_name = None
        self._is_compiled_model = False
        self._compilation_job_name = None
        self._is_edge_packaged_model = False
        self._enable_network_isolation = enable_network_isolation
        self.model_kms_key = model_kms_key
        self.image_config = image_config

    def register(
        self,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_name=None,
        model_package_group_name=None,
        image_uri=None,
        model_metrics=None,
        metadata_properties=None,
        marketplace_cert=False,
        approval_status=None,
        description=None,
    ):
        """Creates a model package for creating SageMaker models or listing on Marketplace.

        Args:
            content_types (list): The supported MIME types for the input data (default: None).
            response_types (list): The supported MIME types for the output data (default: None).
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time (default: None).
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed (default: None).
            model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
                using `model_package_name` makes the Model Package un-versioned (default: None).
            model_package_group_name (str): Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned (default: None).
            image_uri (str): Inference image uri for the container. Model class' self.image will
                be used if it is None (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties object (default: None).
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace (default: False).
            approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
                or "PendingManualApproval" (default: "PendingManualApproval").
            description (str): Model Package description (default: None).

        Returns:
            A `sagemaker.model.ModelPackage` instance.
        """
        if self.model_data is None:
            raise ValueError("SageMaker Model Package cannot be created without model data.")

        model_pkg_args = self._get_model_package_args(
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
        model_package = self.sagemaker_session.create_model_package_from_containers(
            **model_pkg_args
        )
        return ModelPackage(
            role=self.role,
            model_data=self.model_data,
            model_package_arn=model_package.get("ModelPackageArn"),
        )

    def _get_model_package_args(
        self,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_name=None,
        model_package_group_name=None,
        image_uri=None,
        model_metrics=None,
        metadata_properties=None,
        marketplace_cert=False,
        approval_status=None,
        description=None,
    ):
        """Get arguments for session.create_model_package method.

        Args:
            content_types (list): The supported MIME types for the input data.
            response_types (list): The supported MIME types for the output data.
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time.
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed.
            model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
                using `model_package_name` makes the Model Package un-versioned (default: None).
            model_package_group_name (str): Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned (default: None).
            image_uri (str): Inference image uri for the container. Model class' self.image will
                be used if it is None (default: None).
            model_metrics (ModelMetrics): ModelMetrics object (default: None).
            metadata_properties (MetadataProperties): MetadataProperties object (default: None).
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace (default: False).
            approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
                or "PendingManualApproval" (default: "PendingManualApproval").
            description (str): Model Package description (default: None).
        Returns:
            dict: A dictionary of method argument names and values.
        """
        if image_uri:
            self.image_uri = image_uri
        container = {
            "Image": self.image_uri,
            "ModelDataUrl": self.model_data,
        }

        model_package_args = {
            "containers": [container],
            "content_types": content_types,
            "response_types": response_types,
            "inference_instances": inference_instances,
            "transform_instances": transform_instances,
            "marketplace_cert": marketplace_cert,
        }

        if model_package_name is not None:
            model_package_args["model_package_name"] = model_package_name
        if model_package_group_name is not None:
            model_package_args["model_package_group_name"] = model_package_group_name
        if model_metrics is not None:
            model_package_args["model_metrics"] = model_metrics._to_request_dict()
        if metadata_properties is not None:
            model_package_args["metadata_properties"] = metadata_properties._to_request_dict()
        if approval_status is not None:
            model_package_args["approval_status"] = approval_status
        if description is not None:
            model_package_args["description"] = description
        return model_package_args

    def _init_sagemaker_session_if_does_not_exist(self, instance_type):
        """Set ``self.sagemaker_session`` to ``LocalSession`` or ``Session`` if it's not already.

        The type of session object is determined by the instance type.
        """
        if self.sagemaker_session:
            return

        if instance_type in ("local", "local_gpu"):
            self.sagemaker_session = local.LocalSession()
        else:
            self.sagemaker_session = session.Session()

    def prepare_container_def(
        self, instance_type=None, accelerator_type=None
    ):  # pylint: disable=unused-argument
        """Return a dict created by ``sagemaker.container_def()``.

        It is used for deploying this model to a specified instance type.

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
        return sagemaker.container_def(
            self.image_uri, self.model_data, self.env, image_config=self.image_config
        )

    def enable_network_isolation(self):
        """Whether to enable network isolation when creating this Model

        Returns:
            bool: If network isolation should be enabled or not.
        """
        return self._enable_network_isolation

    def _create_sagemaker_model(self, instance_type=None, accelerator_type=None, tags=None):
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

        self._ensure_base_name_if_needed(container_def["Image"])
        self._set_model_name_if_needed()

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

    def _ensure_base_name_if_needed(self, image_uri):
        """Create a base name from the image URI if there is no model name provided."""
        if self.name is None:
            self._base_name = self._base_name or utils.base_name_from_image(image_uri)

    def _set_model_name_if_needed(self):
        """Generate a new model name if ``self._base_name`` is present."""
        if self._base_name:
            self.name = utils.name_from_base(self._base_name)

    def _framework(self):
        """Placeholder docstring"""
        return getattr(self, "_framework_name", None)

    def _get_framework_version(self):
        """Placeholder docstring"""
        return getattr(self, "framework_version", None)

    def _edge_packaging_job_config(
        self,
        output_path,
        role,
        model_name,
        model_version,
        packaging_job_name,
        compilation_job_name,
        resource_key,
        s3_kms_key,
        tags,
    ):
        """Creates a request object for a packaging job.

        Args:
            output_path (str): where in S3 to store the output of the job
            role (str): what role to use when executing the job
            packaging_job_name (str): what to name the packaging job
            compilation_job_name (str): what compilation job to source the model from
            resource_key (str): the kms key to encrypt the disk with
            s3_kms_key (str): the kms key to encrypt the output with
            tags (list[dict]): List of tags for labeling an edge packaging job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
        Returns:
            dict: the request object to use when creating a packaging job
        """
        output_model_config = {
            "S3OutputLocation": output_path,
        }
        if s3_kms_key is not None:
            output_model_config["KmsKeyId"] = s3_kms_key

        return {
            "output_model_config": output_model_config,
            "role": role,
            "tags": tags,
            "model_name": model_name,
            "model_version": model_version,
            "job_name": packaging_job_name,
            "compilation_job_name": compilation_job_name,
            "resource_key": resource_key,
        }

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
        target_platform_os=None,
        target_platform_arch=None,
        target_platform_accelerator=None,
        compiler_options=None,
        framework_version=None,
    ):
        """Placeholder Docstring"""
        input_model_config = {
            "S3Uri": self.model_data,
            "DataInputConfig": json.dumps(input_shape)
            if isinstance(input_shape, dict)
            else input_shape,
            "Framework": framework.upper(),
        }

        if (
            framework.lower() == "pytorch"
            and re.match("(?=^ml_)(?!ml_inf)", target_instance_type) is not None
            and framework_version is not None
        ):
            input_model_config["FrameworkVersion"] = utils.get_short_version(framework_version)

        role = self.sagemaker_session.expand_role(role)
        output_model_config = {
            "S3OutputLocation": output_path,
        }

        if target_instance_type is not None:
            output_model_config["TargetDevice"] = target_instance_type
        else:
            if target_platform_os is None and target_platform_arch is None:
                raise ValueError(
                    "target_instance_type or (target_platform_os and target_platform_arch) "
                    "should be provided"
                )
            target_platform = {
                "Os": target_platform_os,
                "Arch": target_platform_arch,
            }
            if target_platform_accelerator is not None:
                target_platform["Accelerator"] = target_platform_accelerator
            output_model_config["TargetPlatform"] = target_platform

        if compiler_options is not None:
            output_model_config["CompilerOptions"] = (
                json.dumps(compiler_options)
                if isinstance(compiler_options, dict)
                else compiler_options
            )

        return {
            "input_model_config": input_model_config,
            "output_model_config": output_model_config,
            "role": role,
            "stop_condition": {"MaxRuntimeInSeconds": compile_max_run},
            "tags": tags,
            "job_name": job_name,
        }

    def _compilation_image_uri(self, region, target_instance_type, framework, framework_version):
        """Retrieve the Neo or Inferentia image URI.

        Args:
            region (str): The AWS region.
            target_instance_type (str): Identifies the device on which you want to run
                your model after compilation, for example: ml_c5. For valid values, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
            framework (str): The framework name.
            framework_version (str): The framework version.
        """
        framework_prefix = ""
        framework_suffix = ""

        if framework == "xgboost":
            framework_suffix = "-neo"
        elif target_instance_type.startswith("ml_inf"):
            framework_prefix = "inferentia-"
        else:
            framework_prefix = "neo-"

        return image_uris.retrieve(
            "{}{}{}".format(framework_prefix, framework, framework_suffix),
            region,
            instance_type=target_instance_type,
            version=framework_version,
        )

    def package_for_edge(
        self,
        output_path,
        model_name,
        model_version,
        role=None,
        job_name=None,
        resource_key=None,
        s3_kms_key=None,
        tags=None,
    ):
        """Package this ``Model`` with SageMaker Edge.

        Creates a new EdgePackagingJob and wait for it to finish.
        model_data will now point to the packaged artifacts.

        Args:
            output_path (str): Specifies where to store the packaged model
            role (str): Execution role
            model_name (str): the name to attach to the model metadata
            model_version (str): the version to attach to the model metadata
            job_name (str): The name of the edge packaging job
            resource_key (str): the kms key to encrypt the disk with
            s3_kms_key (str): the kms key to encrypt the output with
            tags (list[dict]): List of tags for labeling an edge packaging job. For
                more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """
        if self._compilation_job_name is None:
            raise ValueError("You must first compile this model")
        if job_name is None:
            job_name = f"packaging{self._compilation_job_name[11:]}"
        if role is None:
            role = self.sagemaker_session.expand_role(role)

        self._init_sagemaker_session_if_does_not_exist(None)
        config = self._edge_packaging_job_config(
            output_path,
            role,
            model_name,
            model_version,
            job_name,
            self._compilation_job_name,
            resource_key,
            s3_kms_key,
            tags,
        )
        self.sagemaker_session.package_model_for_edge(**config)
        job_status = self.sagemaker_session.wait_for_edge_packaging_job(job_name)
        self.model_data = job_status["ModelArtifact"]
        self._is_edge_packaged_model = True

        return self

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
        target_platform_os=None,
        target_platform_arch=None,
        target_platform_accelerator=None,
        compiler_options=None,
    ):
        """Compile this ``Model`` with SageMaker Neo.

        Args:
            target_instance_family (str): Identifies the device that you want to
                run your model after compilation, for example: ml_c5. For allowed
                strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                Alternatively, you can select an OS, Architecture and Accelerator using
                ``target_platform_os``, ``target_platform_arch``,
                and ``target_platform_accelerator``.
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
            framework_version (str): The version of framework, for example:
                '1.5' for PyTorch
            target_platform_os (str): Target Platform OS, for example: 'LINUX'.
                For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family by setting target_instance
                family to None.
            target_platform_arch (str): Target Platform Architecture, for example: 'X86_64'.
                For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family by setting target_instance
                family to None.
            target_platform_accelerator (str, optional): Target Platform Accelerator,
                for example: 'NVIDIA'. For allowed strings see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html.
                It can be used instead of target_instance_family by setting target_instance
                family to None.
            compiler_options (dict, optional): Additional parameters for compiler.
                Compiler Options are TargetPlatform / target_instance_family specific. See
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_OutputConfig.html for details.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See
            :func:`~sagemaker.model.Model` for full details.
        """
        framework = framework or self._framework()
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
        if self.model_data is None:
            raise ValueError("You must provide an S3 path to the compressed model artifacts.")

        framework_version = framework_version or self._get_framework_version()

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
            target_platform_os,
            target_platform_arch,
            target_platform_accelerator,
            compiler_options,
            framework_version,
        )
        self.sagemaker_session.compile_model(**config)
        job_status = self.sagemaker_session.wait_for_compilation_job(job_name)
        self.model_data = job_status["ModelArtifacts"]["S3ModelArtifacts"]
        if target_instance_family is not None:
            if target_instance_family == "ml_eia2":
                pass
            elif target_instance_family.startswith("ml_"):
                self.image_uri = self._compilation_image_uri(
                    self.sagemaker_session.boto_region_name,
                    target_instance_family,
                    framework,
                    framework_version,
                )
                self._is_compiled_model = True
            else:
                LOGGER.warning(
                    "The instance type %s is not supported for deployment via SageMaker."
                    "Please deploy the model manually.",
                    target_instance_family,
                )
        else:
            LOGGER.warning(
                "Devices described by Target Platform OS, Architecture and Accelerator are not"
                "supported for deployment via SageMaker. Please deploy the model manually."
            )

        self._compilation_job_name = job_name

        return self

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        serializer=None,
        deserializer=None,
        accelerator_type=None,
        endpoint_name=None,
        tags=None,
        kms_key=None,
        wait=True,
        data_capture_config=None,
        **kwargs,
    ):
        """Deploy this ``Model`` to an ``Endpoint`` and optionally return a ``Predictor``.

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
                deploy this model for model loading and inference, for example,
                'ml.eia1.medium'. If not specified, no Elastic Inference
                accelerator will be attached to the endpoint. For more
                information:
                https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
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
        removed_kwargs("update_endpoint", kwargs)
        self._init_sagemaker_session_if_does_not_exist(instance_type)

        if self.role is None:
            raise ValueError("Role can not be null for deploying a model")

        if instance_type.startswith("ml.inf") and not self._is_compiled_model:
            LOGGER.warning(
                "Your model is not compiled. Please compile your model before using Inferentia."
            )

        compiled_model_suffix = "-".join(instance_type.split(".")[:-1])
        if self._is_compiled_model:
            self._ensure_base_name_if_needed(self.image_uri)
            if self._base_name is not None:
                self._base_name = "-".join((self._base_name, compiled_model_suffix))

        self._create_sagemaker_model(instance_type, accelerator_type, tags)
        production_variant = sagemaker.production_variant(
            self.name, instance_type, initial_instance_count, accelerator_type=accelerator_type
        )
        if endpoint_name:
            self.endpoint_name = endpoint_name
        else:
            base_endpoint_name = self._base_name or utils.base_from_name(self.name)
            if self._is_compiled_model and not base_endpoint_name.endswith(compiled_model_suffix):
                base_endpoint_name = "-".join((base_endpoint_name, compiled_model_suffix))
            self.endpoint_name = utils.name_from_base(base_endpoint_name)

        data_capture_config_dict = None
        if data_capture_config is not None:
            data_capture_config_dict = data_capture_config._to_request_dict()

        self.sagemaker_session.endpoint_from_production_variants(
            name=self.endpoint_name,
            production_variants=[production_variant],
            tags=tags,
            kms_key=kms_key,
            wait=wait,
            data_capture_config_dict=data_capture_config_dict,
        )

        if self.predictor_cls:
            predictor = self.predictor_cls(self.endpoint_name, self.sagemaker_session)
            if serializer:
                predictor.serializer = serializer
            if deserializer:
                predictor.deserializer = deserializer
            return predictor
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
            base_transform_job_name=self._base_name or self.name,
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
        image_uri,
        role,
        entry_point,
        source_dir=None,
        predictor_cls=None,
        env=None,
        name=None,
        container_log_level=logging.INFO,
        code_location=None,
        sagemaker_session=None,
        dependencies=None,
        git_config=None,
        **kwargs,
    ):
        """Initialize a ``FrameworkModel``.

        Args:
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            image_uri (str): A Docker image URI.
            role (str): An IAM role name or ARN for SageMaker to access AWS
                resources on your behalf.
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to model
                hosting. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                If 'git_config' is provided, 'entry_point' should be
                a relative location to the Python source file in the Git repo.

                Example:
                    With the following GitHub repo directory structure:

                    >>> |----- README.md
                    >>> |----- src
                    >>>         |----- inference.py
                    >>>         |----- test.py

                    You can assign entry_point='src/inference.py'.
            source_dir (str): Path (absolute, relative or an S3 URI) to a directory
                with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must
                point to a tar.gz file. Structure within this directory are preserved
                when training on Amazon SageMaker. If 'git_config' is provided,
                'source_dir' should be a relative location to a directory in the Git repo.
                If the directory points to S3, no code will be uploaded and the S3 location
                will be used instead.

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
            env (dict[str, str]): Environment variables to run with ``image_uri``
                when hosted in SageMaker (default: None).
            name (str): The model name. If None, a default model name will be
                selected on each ``deploy``.
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
                instead.

                .. admonition:: Example

                    The following call

                    >>> Model(entry_point='inference.py',
                    ...       dependencies=['my/libs/common', 'virtual-env'])

                    results in the following inside the container:

                    >>> $ ls

                    >>> opt/ml/code
                    >>>     |------ inference.py
                    >>>     |------ common
                    >>>     |------ virtual-env

                This is not supported with "local code" in Local Mode.
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
            **kwargs: Keyword arguments passed to the superclass
                :class:`~sagemaker.model.Model`.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.Model`.
        """
        super(FrameworkModel, self).__init__(
            image_uri,
            model_data,
            role,
            predictor_cls=predictor_cls,
            env=env,
            name=name,
            sagemaker_session=sagemaker_session,
            **kwargs,
        )
        self.entry_point = entry_point
        self.source_dir = source_dir
        self.dependencies = dependencies or []
        self.git_config = git_config
        self.container_log_level = container_log_level
        if code_location:
            self.bucket, self.key_prefix = s3.parse_s3_url(code_location)
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

    def prepare_container_def(self, instance_type=None, accelerator_type=None):
        """Return a container definition with framework configuration.

        Framework configuration is set in model environment variables.
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
        deploy_key_prefix = fw_utils.model_code_key_prefix(
            self.key_prefix, self.name, self.image_uri
        )
        self._upload_code(deploy_key_prefix)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())
        return sagemaker.container_def(self.image_uri, self.model_data, deploy_env)

    def _upload_code(self, key_prefix, repack=False):
        """Placeholder Docstring"""
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
            self.uploaded_code = fw_utils.UploadedCode(
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
        super(ModelPackage, self).__init__(
            role=role, model_data=model_data, image_uri=None, **kwargs
        )

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
        """Whether to enable network isolation when creating a model out of this ModelPackage

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

        self._ensure_base_name_if_needed(model_package_name.split("/")[-1])
        self._set_model_name_if_needed()

        self.sagemaker_session.create_model(
            self.name,
            self.role,
            container_def,
            vpc_config=self.vpc_config,
            enable_network_isolation=self.enable_network_isolation(),
        )

    def _ensure_base_name_if_needed(self, base_name):
        """Set the base name if there is no model name provided."""
        if self.name is None:
            self._base_name = base_name
