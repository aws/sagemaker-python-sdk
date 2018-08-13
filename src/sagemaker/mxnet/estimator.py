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

from sagemaker.estimator import Framework
from sagemaker.fw_utils import framework_name_from_image, framework_version_from_tag
from sagemaker.mxnet.defaults import MXNET_VERSION
from sagemaker.mxnet.model import MXNetModel

from sagemaker.utils import get_config_value



import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time

logging.basicConfig()
LOGGER = logging.getLogger('sagemaker')

class Tensorboard(threading.Thread):
    def __init__(self, estimator, logdir=None):
        """Initialize ``Tensorboard`` instance.
        Args:
            estimator (sagemaker.estimator.Framework): A SageMaker ``Estimator``.
            logdir (str): Directory for logs (default: None). If not specified, a temporary directory is made.
        """
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.estimator = estimator
        self.logdir = logdir or tempfile.mkdtemp()

    @staticmethod
    def _cmd_exists(cmd):
        return any(
            os.access(os.path.join(path, cmd), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )

    @staticmethod
    def _sync_directories(from_directory, to_directory):
        """Sync to_directory with from_directory by copying each file in
        to_directory with new contents. Files in to_directory will be
        overwritten by files of the same name in from_directory. We need to
        keep two copies of the log directory because otherwise TensorBoard
        picks up temp files from `aws s3 sync` and then stops reading the
        correct tfevent files. We walk the directory and copy each file
        individually because the directory that TensorBoard watches needs to
        always exist.
        Args:
            from_directory (str): The directory with updated files.
            to_directory (str): The directory to be synced.
        """
        if not os.path.exists(to_directory):
            os.mkdir(to_directory)
        for root, dirs, files in os.walk(from_directory):
            to_root = root.replace(from_directory, to_directory)
            for directory in dirs:
                to_child_dir = os.path.join(to_root, directory)
                if not os.path.exists(to_child_dir):
                    os.mkdir(to_child_dir)
            for fname in files:
                from_file = os.path.join(root, fname)
                to_file = os.path.join(to_root, fname)
                with open(from_file, 'rb') as a, open(to_file, 'wb') as b:
                    b.write(a.read())

    @staticmethod
    @contextlib.contextmanager
    def _temporary_directory():
        """Context manager for a temporary directory. This is similar to
        tempfile.TemporaryDirectory in python>=3.2.
        """
        name = tempfile.mkdtemp()
        try:
            yield name
        finally:
            shutil.rmtree(name)

    def validate_requirements(self):
        """Ensure that TensorBoard and the AWS CLI are installed.
        These dependencies are required for using TensorBoard.
        Raises:
            EnvironmentError: If at least one requirement is not installed.
        """
        if not self._cmd_exists('tensorboard'):
            raise EnvironmentError('TensorBoard is not installed in the system. Please install TensorBoard using the'
                                   ' following command: \n pip install tensorboard')

        if not self._cmd_exists('aws'):
            raise EnvironmentError('The AWS CLI is not installed in the system. Please install the AWS CLI using the'
                                   ' following command: \n pip install awscli')

    def create_tensorboard_process(self):
        """Create a TensorBoard process.
        Returns:
            tuple: A tuple containing:
                int: The port number.
                process: The TensorBoard process.
        Raises:
            OSError: If no ports between 6006 and 6105 are available for starting TensorBoard.
        """
        port = 6006


        for i in range(100):
            p = subprocess.Popen(
                ["tensorboard", "--logdir", self.logdir, "--host", "localhost", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.event.wait(5)
            if p.poll():
                port += 1
            else:
                return port, p


        raise OSError('No available ports to start TensorBoard. Attempted all ports between 6006 and 6105')

    def run(self):
        """Run TensorBoard process."""

        port, tensorboard_process = self.create_tensorboard_process()


        LOGGER.info('TensorBoard 0.1.7 at on port {}'.format(port))
        while not self.estimator.checkpoint_path:
            self.event.wait(1)
            print("waiting...")
        with self._temporary_directory() as aws_sync_dir:
            while not self.event.is_set():
                args = ['aws', 's3', 'sync', self.estimator.checkpoint_path, aws_sync_dir]
                subprocess.call(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._sync_directories(aws_sync_dir, self.logdir)
                self.event.wait(10)
        tensorboard_process.terminate()


class MXNet(Framework):
    """Handle end-to-end training and deployment of custom MXNet code."""

    __framework_name__ = "mxnet"

    def __init__(self, entry_point, source_dir=None, hyperparameters=None, py_version='py2', checkpoint_path=None, 
                 framework_version=MXNET_VERSION, image_name=None, **kwargs):
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
            image_name (str): If specified, the estimator will use this image for training and hosting, instead of
                selecting the appropriate SageMaker official image based on framework_version and py_version. It can
                be an ECR url or dockerhub image and tag.
                    Examples:
                        123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                        custom-image:latest.
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.estimator.Framework` constructor.
        """
        super(MXNet, self).__init__(entry_point, source_dir, hyperparameters,
                                    image_name=image_name, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.py_version = py_version
        self.framework_version = framework_version

    def create_model(self, model_server_workers=None, role=None):
        """Create a SageMaker ``MXNetModel`` object that can be deployed to an ``Endpoint``.
        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.
        Returns:
            sagemaker.mxnet.model.MXNetModel: A SageMaker ``MXNetModel`` object.
                See :func:`~sagemaker.mxnet.model.MXNetModel` for full details.
        """
        role = role or self.role
        return MXNetModel(self.model_data, role, self.entry_point, source_dir=self._model_source_dir(),
                          enable_cloudwatch_metrics=self.enable_cloudwatch_metrics, name=self._current_job_name,
                          container_log_level=self.container_log_level, code_location=self.code_location,
                          py_version=self.py_version, framework_version=self.framework_version, image=self.image_name,
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
        image_name = init_params.pop('image')
        framework, py_version, tag = framework_name_from_image(image_name)

        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params['image_name'] = image_name
            return init_params

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

    def fit(self, inputs, wait=True, logs=True, job_name=None, run_tensorboard_locally=False):
        """Train a model using the input training dataset.
        See :func:`~sagemaker.estimator.EstimatorBase.fit` for more details.
        Args:
            inputs (str or dict or sagemaker.session.s3_input): Information about the training data.
                This can be one of three types:
                (str) - the S3 location where training data is saved.
                (dict[str, str] or dict[str, sagemaker.session.s3_input]) - If using multiple channels for
                    training data, you can specify a dict mapping channel names
                    to strings or :func:`~sagemaker.session.s3_input` objects.
                (sagemaker.session.s3_input) - channel configuration for S3 data sources that can provide
                    additional information as well as the path to the training dataset.
                    See :func:`sagemaker.session.s3_input` for full details.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Training job name. If not specified, the estimator generates a default job name,
                based on the training image name and current timestamp.
            run_tensorboard_locally (bool): Whether to execute TensorBoard in a different process with
                downloaded checkpoint information (default: False). This is an experimental feature, and requires
                TensorBoard and AWS CLI to be installed. It terminates TensorBoard when execution ends.
        """
        def fit_super():
            super(MXNet, self).fit(inputs, wait, logs, job_name)

        if run_tensorboard_locally and wait is False:
            raise ValueError("Tensorboard is not supported with async fit")

        if run_tensorboard_locally:
            tensorboard = Tensorboard(self)
            tensorboard.validate_requirements()
            try:
                tensorboard.start()
                fit_super()
            finally:
                # sleep 20 secs for tensorboard start up if fit() quits instantly
                time.sleep(20)
                tensorboard.event.set()
                tensorboard.join()
        else:
            fit_super()        

    def hyperparameters(self):
        """Return hyperparameters used by your custom TensorFlow code during model training."""
        hyperparameters = super(MXNet, self).hyperparameters()

        if not self.checkpoint_path:
            local_code = get_config_value('local.local_code', self.sagemaker_session.config)
            if self.sagemaker_session.local_mode and local_code:
                self.checkpoint_path = '/opt/ml/shared/checkpoints'
            else:
                self.checkpoint_path = os.path.join(self.output_path,
                                                    self._current_job_name, 'checkpoints')


        additional_hyperparameters = {'checkpoint_path': self.checkpoint_path}

        hyperparameters.update(Framework._json_encode_hyperparameters(additional_hyperparameters))
        return hyperparameters            