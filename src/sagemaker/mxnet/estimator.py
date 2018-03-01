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
from sagemaker.estimator import Framework
from sagemaker.fw_utils import create_image_uri, framework_name_from_image, framework_version_from_tag
from sagemaker.mxnet.defaults import MXNET_VERSION
from sagemaker.mxnet.model import MXNetModel


class MXNet(Framework):
    """Handle end-to-end training and deployment of custom MXNet code."""

    __framework_name__ = "mxnet"

    def __init__(self, entry_point, source_dir=None, hyperparameters=None, py_version='py2',
                 framework_version=MXNET_VERSION, **kwargs):
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
            framework_version (str): MXNet version you want to use for executing your model training code.
                List of supported versions https://github.com/aws/sagemaker-python-sdk#mxnet-sagemaker-estimators
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.estimator.Framework` constructor.
        """
        super(MXNet, self).__init__(entry_point, source_dir, hyperparameters, **kwargs)
        self.py_version = py_version
        self.framework_version = framework_version

    def train_image(self):
        """Return the Docker image to use for training.

        The :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does the model training, calls this method to
        find the image to use for model training.

        Returns:
            str: The URI of the Docker image.
        """
        return create_image_uri(self.sagemaker_session.boto_session.region_name, self.__framework_name__,
                                self.train_instance_type, framework_version=self.framework_version,
                                py_version=self.py_version)

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
                          py_version=self.py_version, framework_version=self.framework_version,
                          model_server_workers=model_server_workers, sagemaker_session=self.sagemaker_session)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(MXNet, cls)._prepare_init_params_from_job_description(job_details)
        framework, py_version, tag = framework_name_from_image(init_params.pop('image'))

        init_params['py_version'] = py_version

        # We switched image tagging scheme from regular image version (e.g. '1.0') to more expressive
        # containing framework version, device type and python version (e.g. '0.12-gpu-py2').
        # For backward compatibility map deprecated image tag '1.0' to a '0.12' framework version
        # otherwise extract framework version from the tag itself.
        init_params['framework_version'] = '0.12' if tag == '1.0' else framework_version_from_tag(tag)

        training_job_name = init_params['base_job_name']

        if framework != cls.__framework_name__:
            raise ValueError("Training job: {} didn't use image for requested framework".format(training_job_name))

        return init_params
