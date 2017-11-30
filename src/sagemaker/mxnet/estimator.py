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
import sagemaker
from sagemaker.estimator import Framework
from sagemaker.fw_utils import create_image_uri, framework_name_from_image
from sagemaker.mxnet.model import MXNetModel
from sagemaker.session import Session


class MXNet(Framework):
    """Handle end-to-end training and deployment of custom MXNet code."""

    __framework_name__ = "mxnet"

    def __init__(self, entry_point, source_dir=None, hyperparameters=None, py_version='py2', **kwargs):
        """
        This ``Estimator`` executes an MXNet script in a managed MXNet execution environment, within a SageMaker
        Training Job. The managed MXNet environment is an Amazon-built Docker container that executes functions
        defined in the supplied ``entry_point`` Python script.

        Training is started by calling :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a
        hosted SageMaker endpoint and returns an :class:`~sagemaker.amazon.mxnet.model.MXNetPredictor` instance
        that can be used to perform inference against the hosted model.

        Technical documentation on preparing MXNet scripts for SageMaker training and using the MXNet Estimator is
        avaialble on the project home-page: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str): Path (absolute or relative) to the Python source file which should be executed
                as the entry point to training. This should be compatible with either Python 2.7 or Python 3.5.
            source_dir (str): Path (absolute or relative) to a directory with any other training
                source code dependencies aside from tne entry point file (default: None). Structure within this
                directory are preserved when training on Amazon SageMaker.
            hyperparameters (dict): Hyperparameters that will be used for training (default: None).
                The hyperparameters are made accessible as a dict[str, str] to the training code on SageMaker.
                For convenience, this accepts other types for keys and values, but ``str()`` will be called
                to convert them before training.
            py_version (str): Python version you want to use for executing your model training code (default: 'py2').
                              One of 'py2' or 'py3'.
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.estimator.Framework` constructor.
        """
        super(MXNet, self).__init__(entry_point, source_dir, hyperparameters, **kwargs)
        self.py_version = py_version

    def train_image(self):
        """Return the Docker image to use for training.

        The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does the model training, calls this method to
        find the image to use for model training.

        Returns:
            str: The URI of the Docker image.
        """
        return create_image_uri(self.sagemaker_session.boto_session.region_name, self.__framework_name__,
                                self.train_instance_type, py_version=self.py_version, tag=sagemaker.mxnet.DOCKER_TAG)

    def create_model(self, model_server_workers=None):
        """Create a SageMaker ``MXNetModel`` object that can be deployed to an ``Endpoint``.

        Args:
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.

        Returns:
            sagemaker.mxnet.model.MXNetModel: A SageMaker ``MXNetModel`` object.
                See :func:`~sagemaker.mxnet.model.MXNetModel` for full details.
        """
        return MXNetModel(self.model_data, self.role, self.entry_point, source_dir=self.source_dir,
                          enable_cloudwatch_metrics=self.enable_cloudwatch_metrics, name=self._current_job_name,
                          container_log_level=self.container_log_level, code_location=self.code_location,
                          py_version=self.py_version, model_server_workers=model_server_workers,
                          sagemaker_session=self.sagemaker_session)

    @classmethod
    def attach(cls, training_job_name, sagemaker_session=None):
        """Attach to an existing training job.

        Create an ``Estimator`` bound to an existing training job. After attaching, if
        the training job is in a Complete status, it can be ``deploy``ed to create
        a SageMaker ``Endpoint`` and return a ``Predictor``.

        If the training job is in progress, attach will block and display log messages
        from the training job, until the training job completes.

        Args:
            training_job_name (str): The name of the training job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.

        Returns:
            sagemaker.mxnet.estimator.MXNet: ``Estimator`` with the attached training job.

        Raises:
            ValueError: If `training_job_name` is None or the image name does not match the framework.
        """
        sagemaker_session = sagemaker_session or Session()

        if training_job_name is None:
            raise ValueError("must specify training_job name")

        job_details = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        init_params, hp, image = cls._prepare_estimator_params_from_job_description(job_details)

        init_params.update({'hyperparameters': hp})

        framework, py_version = framework_name_from_image(image)
        init_params.update({'py_version': py_version})

        if framework != cls.__framework_name__:
            raise ValueError("Training job: {} didn't use image for requested framework".format(training_job_name))

        return super(MXNet, cls).attach(training_job_name=None, sagemaker_session=sagemaker_session, **init_params)
