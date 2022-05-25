# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import logging

import sagemaker
from sagemaker import image_uris
from sagemaker.deserializers import NumpyDeserializer
from sagemaker.fw_utils import model_code_key_prefix, validate_version_or_image_args
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.sklearn import defaults

logger = logging.getLogger("sagemaker")


class SKLearnPredictor(Predictor):
    """A Predictor for inference against Scikit-learn Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to
    multidimensional tensors for Scikit-learn inference.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    ):
        """Initialize an ``SKLearnPredictor``.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to .npy format. Handles lists and numpy
                arrays.
            deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
                Default parses the response from .npy format to numpy array.
        """
        super(SKLearnPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


class SKLearnModel(FrameworkModel):
    """An Scikit-learn SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    _framework_name = defaults.SKLEARN_NAME

    def __init__(
        self,
        model_data,
        role,
        entry_point,
        framework_version=None,
        py_version="py3",
        image_uri=None,
        predictor_cls=SKLearnPredictor,
        model_server_workers=None,
        **kwargs
    ):
        """Initialize an SKLearnModel.

        Args:
            model_data (str): The S3 location of a SageMaker model data
                ``.tar.gz`` file.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            entry_point (str): Path (absolute or relative) to the Python source
                file which should be executed as the entry point to model
                hosting. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            framework_version (str): Scikit-learn version you want to use for
                executing your model training code. Defaults to ``None``. Required
                unless ``image_uri`` is provided.
            py_version (str): Python version you want to use for executing your
                model training code (default: 'py3'). Currently, 'py3' is the only
                supported version. If ``None`` is passed in, ``image_uri`` must be
                provided.
            image_uri (str): A Docker image URI (default: None). If not specified,
                a default image for Scikit-learn will be used.
                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If ``image_uri`` is also ``None``, then a ``ValueError``
                will be raised.
            predictor_cls (callable[str, sagemaker.session.Session]): A function
                to call to create a predictor with an endpoint name and
                SageMaker ``Session``. If specified, ``deploy()`` returns the
                result of invoking this function on the created endpoint name.
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            **kwargs: Keyword arguments passed to the ``FrameworkModel``
                initializer.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.FrameworkModel` and
            :class:`~sagemaker.model.Model`.
        """
        validate_version_or_image_args(framework_version, py_version, image_uri)
        if py_version and py_version != "py3":
            raise AttributeError(
                "Scikit-learn image only supports Python 3. Please use 'py3' for py_version."
            )
        self.framework_version = framework_version
        self.py_version = py_version

        super(SKLearnModel, self).__init__(
            model_data, image_uri, role, entry_point, predictor_cls=predictor_cls, **kwargs
        )

        self.model_server_workers = model_server_workers

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
        drift_check_baselines=None,
        customer_metadata_properties=None,
        domain=None,
    ):
        """Creates a model package for creating SageMaker models or listing on Marketplace.

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
            drift_check_baselines (DriftCheckBaselines): DriftCheckBaselines object (default: None).
            customer_metadata_properties (dict[str, str]): A dictionary of key-value paired
                metadata properties (default: None).
            domain (str): Domain values can be "COMPUTER_VISION", "NATURAL_LANGUAGE_PROCESSING",
                "MACHINE_LEARNING" (default: None).

        Returns:
            A `sagemaker.model.ModelPackage` instance.
        """
        instance_type = inference_instances[0]
        self._init_sagemaker_session_if_does_not_exist(instance_type)

        if image_uri:
            self.image_uri = image_uri
        if not self.image_uri:
            self.image_uri = self.serving_image_uri(
                region_name=self.sagemaker_session.boto_session.region_name,
                instance_type=instance_type,
            )
        return super(SKLearnModel, self).register(
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
            drift_check_baselines=drift_check_baselines,
            customer_metadata_properties=customer_metadata_properties,
            domain=domain,
        )

    def prepare_container_def(
        self, instance_type=None, accelerator_type=None, serverless_inference_config=None
    ):
        """Container definition with framework configuration set in model environment variables.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                This parameter is unused because Scikit-learn supports only CPU.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model. This parameter is unused because accelerator types
                are not supported by SKLearnModel.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to find image URIs.

        Returns:
            dict[str, str]: A container definition object usable with the
            CreateModel API.
        """
        if accelerator_type:
            raise ValueError("Accelerator types are not supported for Scikit-Learn.")

        deploy_image = self.image_uri
        if not deploy_image:
            deploy_image = self.serving_image_uri(
                self.sagemaker_session.boto_region_name, instance_type
            )

        deploy_key_prefix = model_code_key_prefix(self.key_prefix, self.name, deploy_image)
        self._upload_code(key_prefix=deploy_key_prefix, repack=self.enable_network_isolation())
        deploy_env = dict(self.env)
        deploy_env.update(self._script_mode_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(self.model_server_workers)
        model_data_uri = (
            self.repacked_model_data if self.enable_network_isolation() else self.model_data
        )
        return sagemaker.container_def(deploy_image, model_data_uri, deploy_env)

    def serving_image_uri(self, region_name, instance_type, serverless_inference_config=None):
        """Create a URI for the serving image.

        Args:
            region_name (str): AWS region where the image is uploaded.
            instance_type (str): SageMaker instance type.
            serverless_inference_config (sagemaker.serverless.ServerlessInferenceConfig):
                Specifies configuration related to serverless endpoint. Instance type is
                not provided in serverless inference. So this is used to determine device type.


        Returns:
            str: The appropriate image URI based on the given parameters.

        """
        return image_uris.retrieve(
            self._framework_name,
            region_name,
            version=self.framework_version,
            py_version=self.py_version,
            instance_type=instance_type,
            serverless_inference_config=serverless_inference_config,
        )
