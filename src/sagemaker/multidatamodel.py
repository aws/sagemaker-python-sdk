# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
from six.moves.urllib import parse

import sagemaker
from sagemaker.model import Model

MULTI_MODEL_CONTAINER_MODE = "MultiModel"


class MultiDataModel(Model):
    """SageMaker ``MultiDataModel`` used to deploy multiple models to the same ``Endpoint``.

    This class defines methods to create a Model with Multi-Model container, deploy it to an
    endpoint, add more models to make them available to a deployed endpoint, list available
    models and run predictions on a Multi-Model endpoint.
    """

    def __init__(
        self,
        model_name,
        model_data_prefix,
        model=None,
        image=None,
        role=None,
        predictor_cls=None,
        **kwargs
    ):
        """Initialize a ``MultiDataModel``. In addition to these arguments, it supports all
           arguments supported by ``Model`` constructor

        Args:
            model_name (str): The model name.
            model_data_prefix (str): The S3 prefix where all the models artifacts (.tar.gz)
                in a Multi-Model endpoint are located
            model (sagemaker.Model): The Model object that would define the
                SageMaker model attributes like vpc_config, predictors, etc.
                If this is present, the attributes for MultiDataModel are copied from this model
                but are replaced the local arguments if present
            image (str): A Docker image URI. If not provided, `model.image` is used instead.
                (default: None)
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role if it needs to access some AWS resources.
                It can be null if this is being used to create a Model to pass
                to a ``PipelineModel`` which has its own Role field. If not provided,
                `model.image` is used instead. (default: None)
            predictor_cls (callable[string, sagemaker.session.Session]): A
                function to call to create a predictor (default: None). If not
                None, ``deploy`` will return the result of invoking this
                function on the created endpoint name.
            **kwargs: Keyword arguments passed to the ``Model`` initializer.
        """
        self.model_name = model_name
        self._model_data_prefix = model_data_prefix
        self.model_data_prefix = model_data_prefix
        self.model = model
        self.container_mode = MULTI_MODEL_CONTAINER_MODE
        self.image = image
        self.role = role
        self.predictor_cls = predictor_cls

        # Copy the values from the model if these attributes are not provided in the constructor
        if self.model:
            self.image = self.image or self.model.image
            self.role = self.role or self.model.role
            self.predictor_cls = self.predictor_cls or self.model.predictor_cls

        super(MultiDataModel, self).__init__(
            self.model_data_prefix,
            self.image,
            self.role,
            name=self.model_name,
            predictor_cls=self.predictor_cls,
            **kwargs
        )

        if not self.sagemaker_session:
            self.sagemaker_session = sagemaker.session.Session()
        self.s3_client = self.sagemaker_session.boto_session.client("s3")

        # Create the S3 prefix path to ensure model deployment succeeds
        self._create_s3_model_data_path()

    @property
    def model_data_prefix(self):
        """Placeholder docstring"""
        return self._model_data_prefix

    @model_data_prefix.setter
    def model_data_prefix(self, model_data_prefix):
        """
        Args:
            model_data_prefix:
        """
        # Validate path
        if not (model_data_prefix.startswith("s3://") and model_data_prefix.endswith("/")):
            raise ValueError(
                'Expecting S3 model prefix beginning with "s3://" '
                'and ending in "/". Received: "{}"'.format(model_data_prefix)
            )
        self._model_data_prefix = model_data_prefix

    def _create_s3_model_data_path(self):
        """
        Create the S3 prefix path to ensure model deployment succeeds.
        If this path does not exist, calls to CreateModel API fails
        """
        bucket, model_data_path = self._parse_s3_uri(self.model_data_prefix)
        self.s3_client.put_object(Bucket=bucket, Key=os.path.join(model_data_path, "/"))

    def prepare_container_def(self, instance_type, accelerator_type=None):
        """Return a container definition set with MultiModel mode and
        model data and all other parameters from the original model pass
        to the MultiDataModel constructor

        Subclasses can override this to provide custom container definitions
        for deployment to a specific instance type. Called by ``deploy()``.

        Returns:
            dict[str, str]: A complete container definition object usable with the CreateModel API
        """
        # Copy the trained model's image and environment variables if they exist.
        # Models trained with FrameworkEstimator set framework specific environment variables
        # which need to be copied over
        if self.model:
            container_definition = self.model.prepare_container_def(instance_type, accelerator_type)
            self.image = self.image or container_definition["Image"]
            self.env.update(container_definition["Environment"])
        return sagemaker.container_def(
            self.image,
            env=self.env,
            model_data_url=self.model_data_prefix,
            container_mode=self.container_mode,
        )

    def add_model(self, s3_url, model_data_path=None):
        """Adds a model to the `MultiDataModel` by copying
        the s3_url model artifact to the given S3 path relative to model_data_prefix

        Args:
        s3_url: S3 path of the trained model artifact
        model_data_path: S3 path where the trained model artifact
                should be uploaded relative to `self.model_data_prefix`.
                (default: None). If None, then the entire s3_url path
                (after the source bucketname) will be copied to
                `model_data_prefix` location
        """
        # Validate s3_url
        if not s3_url.startswith("s3://"):
            raise ValueError(
                'Expecting S3 model path beginning with "s3://". Received: "{}"'.format(s3_url)
            )

        source_bucket, source_model_data_path = self._parse_s3_uri(s3_url)
        copy_source = {"Bucket": source_bucket, "Key": source_model_data_path}

        if not model_data_path:
            model_data_path = source_model_data_path

        # Construct the destination path
        dst_url = os.path.join(self.model_data_prefix, model_data_path)
        destination_bucket, destination_model_data_path = self._parse_s3_uri(dst_url)

        # Copy the model artifact
        self.s3_client.copy(copy_source, destination_bucket, destination_model_data_path)

    def list_models(self):
        """Generates and returns relative paths to model archives stored at model_data_prefix
        S3 location.

        Yields: Relative paths to model archives stored at model_data_prefix.
        """
        bucket, url_prefix = self._parse_s3_uri(self.model_data_prefix)
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=url_prefix):
            try:
                # Get the metadata about each object returned
                s3_objects_metadata = page["Contents"]
                for s3_object in s3_objects_metadata:
                    # Return the model paths relative to the model_data_prefix
                    # Ex: "a/b/c.tar.gz" -> "b/c.tar.gz" if url_prefix = "a/"
                    yield s3_object["Key"].replace(url_prefix, "")
            except KeyError:
                return

    def _parse_s3_uri(self, s3_url):
        """Parses an s3 uri and returns the bucket and s3 prefix path.

        Args:
            s3_url:
        """
        url = parse.urlparse(s3_url)
        bucket, s3_prefix = url.netloc, url.path.lstrip("/")
        return bucket, s3_prefix
