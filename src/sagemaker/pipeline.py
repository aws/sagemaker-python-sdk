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

import sagemaker
from sagemaker.session import Session
from sagemaker.utils import name_from_image


class PipelineModel(object):
    """A pipeline of SageMaker ``Model``s that can be deployed to an ``Endpoint``."""

    def __init__(self, models, role, predictor_cls=None, name=None, vpc_config=None, sagemaker_session=None):
        """Initialize an SageMaker ``Model`` which can be used to build an Inference Pipeline comprising of multiple
        model containers.

        Args:
            models (list[sagemaker.Model]): For using multiple containers to build an inference pipeline,
            you can pass a list of ``sagemaker.Model`` objects in the order you want the inference to happen.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                After the endpoint is created, the inference code might use the IAM role,
                if it needs to access an AWS resource.
            predictor_cls (callable[string, sagemaker.session.Session]): A function to call to create
               a predictor (default: None). If not None, ``deploy`` will return the result of invoking
               this function on the created endpoint name.
            name (str): The model name. If None, a default model name will be selected on each ``deploy``.
            vpc_config (dict[str, list[str]]): The VpcConfig set on the model (default: None)
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for SageMaker
               interactions (default: None). If not specified, one is created using the default AWS configuration chain.
        """
        self.models = models
        self.role = role
        self.predictor_cls = predictor_cls
        self.name = name
        self.vpc_config = vpc_config
        self.sagemaker_session = sagemaker_session
        self._model_name = None

    def pipeline_container_def(self, instance_type):
        """Return a dict created by ``sagemaker.pipeline_container_def()`` for deploying this model to a specified
         instance type.

        Subclasses can override this to provide custom container definitions for
        deployment to a specific instance type. Called by ``deploy()``.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.

        Returns:
            list[dict[str, str]]: A list of container definition objects usable with the CreateModel API in the scenario
            of multiple containers (Inference Pipeline).
        """

        return sagemaker.pipeline_container_def(self.models, instance_type)

    def deploy(self, initial_instance_count, instance_type, endpoint_name=None, tags=None):
        """Deploy this ``Model`` to an ``Endpoint`` and optionally return a ``Predictor``.

        Create a SageMaker ``Model`` and ``EndpointConfig``, and deploy an ``Endpoint`` from this ``Model``.
        If ``self.predictor_cls`` is not None, this method returns a the result of invoking
        ``self.predictor_cls`` on the created endpoint name.

        The name of the created model is accessible in the ``name`` field of this ``Model`` after deploy returns

        The name of the created endpoint is accessible in the ``endpoint_name``
        field of this ``Model`` after deploy returns.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to. For example, 'ml.p2.xlarge'.
            initial_instance_count (int): The initial number of instances to run in the
                ``Endpoint`` created from this ``Model``.
            endpoint_name (str): The name of the endpoint to create (default: None).
                If not specified, a unique endpoint name will be created.
            tags(List[dict[str, str]]): The list of tags to attach to this specific endpoint.

        Returns:
            callable[string, sagemaker.session.Session] or None: Invocation of ``self.predictor_cls`` on
                the created endpoint name, if ``self.predictor_cls`` is not None. Otherwise, return None.
        """
        if not self.sagemaker_session:
            self.sagemaker_session = Session()

        containers = self.pipeline_container_def(instance_type)

        self.name = self.name or name_from_image(containers[0]['Image'])
        self.sagemaker_session.create_model(self.name, self.role, containers, vpc_config=self.vpc_config)

        production_variant = sagemaker.production_variant(self.name, instance_type, initial_instance_count)
        self.endpoint_name = endpoint_name or self.name
        self.sagemaker_session.endpoint_from_production_variants(self.endpoint_name, [production_variant], tags)
        if self.predictor_cls:
            return self.predictor_cls(self.endpoint_name, self.sagemaker_session)
