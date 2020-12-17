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

import sagemaker
from sagemaker.session import Session
from sagemaker.utils import name_from_image
from sagemaker.transformer import Transformer


class PipelineModel(object):
    """A pipeline of SageMaker `Model` instances.

    This pipeline can be deployed as an `Endpoint` on SageMaker.
    """

    def __init__(
        self,
        models,
        role,
        predictor_cls=None,
        name=None,
        vpc_config=None,
        sagemaker_session=None,
        enable_network_isolation=False,
    ):
        """Initialize a SageMaker `Model` instance.

        The `Model` can be used to build an Inference Pipeline comprising of
        multiple model containers.

        Args:
            models (list[sagemaker.Model]): For using multiple containers to
                build an inference pipeline, you can pass a list of
                ``sagemaker.Model`` objects in the order you want the inference
                to happen.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name.
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
            enable_network_isolation (bool): Default False. if True, enables
                network isolation in the endpoint, isolating the model
                container. No inbound or outbound network calls can be made to
                or from the model container.Boolean
        """
        self.models = models
        self.role = role
        self.predictor_cls = predictor_cls
        self.name = name
        self.vpc_config = vpc_config
        self.sagemaker_session = sagemaker_session
        self.enable_network_isolation = enable_network_isolation
        self.endpoint_name = None

    def pipeline_container_def(self, instance_type):
        """The pipeline definition for deploying this model.

        This is the dict created by ``sagemaker.pipeline_container_def()``.

        The instance type to be used may be specified.

        Subclasses can override this to provide custom container definitions
        for deployment to a specific instance type. Called by ``deploy()``.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.

        Returns:
            list[dict[str, str]]: A list of container definition objects usable
            with the CreateModel API in the scenario of multiple containers
            (Inference Pipeline).
        """

        return sagemaker.pipeline_container_def(self.models, instance_type)

    def deploy(
        self,
        initial_instance_count,
        instance_type,
        serializer=None,
        deserializer=None,
        endpoint_name=None,
        tags=None,
        wait=True,
        update_endpoint=False,
        data_capture_config=None,
        kms_key=None,
    ):
        """Deploy the ``Model`` to an ``Endpoint``.

        It optionally return a ``Predictor``.

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
                For example, 'ml.p2.xlarge'.
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
            endpoint_name (str): The name of the endpoint to create (default:
                None). If not specified, a unique endpoint name will be created.
            tags (List[dict[str, str]]): The list of tags to attach to this
                specific endpoint.
            wait (bool): Whether the call should wait until the deployment of
                model completes (default: True).
            update_endpoint (bool): Flag to update the model in an existing
                Amazon SageMaker endpoint. If True, this will deploy a new
                EndpointConfig to an already existing endpoint and delete
                resources corresponding to the previous EndpointConfig. If
                False, a new endpoint will be created. Default: False
            data_capture_config (sagemaker.model_monitor.DataCaptureConfig): Specifies
                configuration related to Endpoint data capture for use with
                Amazon SageMaker Model Monitoring. Default: None.
            kms_key (str): The ARN, Key ID or Alias of the KMS key that is used to
                encrypt the data on the storage volume attached to the instance hosting
                the endpoint.

        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of
            ``self.predictor_cls`` on the created endpoint name, if ``self.predictor_cls``
            is not None. Otherwise, return None.
        """
        if not self.sagemaker_session:
            self.sagemaker_session = Session()

        containers = self.pipeline_container_def(instance_type)

        self.name = self.name or name_from_image(containers[0]["Image"])
        self.sagemaker_session.create_model(
            self.name,
            self.role,
            containers,
            vpc_config=self.vpc_config,
            enable_network_isolation=self.enable_network_isolation,
        )

        production_variant = sagemaker.production_variant(
            self.name, instance_type, initial_instance_count
        )
        self.endpoint_name = endpoint_name or self.name

        data_capture_config_dict = None
        if data_capture_config is not None:
            data_capture_config_dict = data_capture_config._to_request_dict()

        if update_endpoint:
            endpoint_config_name = self.sagemaker_session.create_endpoint_config(
                name=self.name,
                model_name=self.name,
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
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
            predictor = self.predictor_cls(self.endpoint_name, self.sagemaker_session)
            if serializer:
                predictor.serializer = serializer
            if deserializer:
                predictor.deserializer = deserializer
            return predictor
        return None

    def _create_sagemaker_pipeline_model(self, instance_type):
        """Create a SageMaker Model Entity

        Args:
            instance_type (str): The EC2 instance type that this Model will be
                used for, this is only used to determine if the image needs GPU
                support or not.
        """
        if not self.sagemaker_session:
            self.sagemaker_session = Session()

        containers = self.pipeline_container_def(instance_type)

        self.name = self.name or name_from_image(containers[0]["Image"])
        self.sagemaker_session.create_model(
            self.name,
            self.role,
            containers,
            vpc_config=self.vpc_config,
            enable_network_isolation=self.enable_network_isolation,
        )

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
        self._create_sagemaker_pipeline_model(instance_type)

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
        """Delete the SageMaker model backing this pipeline model. This does not
        delete the list of SageMaker models used in multiple containers to build
        the inference pipeline.
        """

        if self.name is None:
            raise ValueError("The SageMaker model must be created before attempting to delete.")

        self.sagemaker_session.delete_model(self.name)
