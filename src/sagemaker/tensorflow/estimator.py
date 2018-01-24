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
import subprocess
import tempfile
import threading

import os

import sagemaker.tensorflow
from sagemaker.estimator import Framework
from sagemaker.fw_utils import create_image_uri, framework_name_from_image
from sagemaker.session import Session
from sagemaker.tensorflow.model import TensorFlowModel

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
        self._aws_sync_dir = tempfile.mkdtemp()
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
        to_directory with new contents. Why do this? Because TensorBoard picks
        up temp files from `aws s3 sync` and then stops reading the correct
        tfevent files. This is probably related to tensorflow/tensorboard#349.

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

        LOGGER.info('TensorBoard 0.1.7 at http://localhost:{}'.format(port))
        while not self.estimator.checkpoint_path:
            self.event.wait(1)
        while not self.event.is_set():
            args = ['aws', 's3', 'sync', self.estimator.checkpoint_path, self._aws_sync_dir]
            subprocess.call(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._sync_directories(self._aws_sync_dir, self.logdir)
            self.event.wait(10)
        tensorboard_process.terminate()


class TensorFlow(Framework):
    """Handle end-to-end training and deployment of user-provided TensorFlow code."""

    __framework_name__ = 'tensorflow'

    def __init__(self, training_steps=None, evaluation_steps=None, checkpoint_path=None, py_version="py2", **kwargs):
        """Initialize an ``TensorFlow`` estimator.
        Args:
            training_steps (int): Perform this many steps of training. `None`, the default means train forever.
            evaluation_steps (int): Perform this many steps of evaluation. `None`, the default means that evaluation
                runs until input from eval_input_fn is exhausted (or another exception is raised).
            checkpoint_path (str): Identifies S3 location where checkpoint data during model training can be
                saved (default: None). For distributed model training, this parameter is required.
            py_version (str): Python version you want to use for executing your model training code (default: 'py2').
            **kwargs: Additional kwargs passed to the Framework constructor.
        """
        super(TensorFlow, self).__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.py_version = py_version
        self.training_steps = training_steps
        self.evaluation_steps = evaluation_steps

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
                    additional information about the training dataset. See :func:`sagemaker.session.s3_input`
                    for full details.
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
            super(TensorFlow, self).fit(inputs, wait, logs, job_name)

        if run_tensorboard_locally:
            tensorboard = Tensorboard(self)
            tensorboard.validate_requirements()

            try:
                tensorboard.start()
                fit_super()
            finally:
                tensorboard.event.set()
        else:
            fit_super()

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
            sagemaker.tensorflow.estimator.TensorFlow: ``Estimator`` with the attached training job.

        Raises:
            ValueError: If `training_job_name` is None or the image name does not match the framework.
        """
        sagemaker_session = sagemaker_session or Session()

        if training_job_name is None:
            raise ValueError("must specify training_job name")

        job_details = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        init_params, hp, image = cls._prepare_estimator_params_from_job_description(job_details)

        updated_params = cls._update_init_params(hp, ['checkpoint_path', 'training_steps', 'evaluation_steps'])
        init_params.update(updated_params)

        init_params.update({'hyperparameters': hp})

        framework, py_version = framework_name_from_image(image)
        init_params.update({'py_version': py_version})

        if framework != cls.__framework_name__:
            raise ValueError("Training job: {} didn't use image for requested framework".format(training_job_name))

        return super(TensorFlow, cls).attach(training_job_name=None, sagemaker_session=sagemaker_session, **init_params)

    def train_image(self):
        """Return the Docker image to use for training.

        The  :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does the model training, calls this method to
        find the image to use for model training.

        Returns:
            str: The URI of the Docker image.
        """
        return create_image_uri(self.sagemaker_session.boto_session.region_name, self.__framework_name__,
                                self.train_instance_type, py_version=self.py_version,
                                tag=sagemaker.tensorflow.DOCKER_TAG)

    def create_model(self, model_server_workers=None):
        """Create a SageMaker ``TensorFlowModel`` object that can be deployed to an ``Endpoint``.

        Args:
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.

        Returns:
            sagemaker.tensorflow.model.TensorFlowModel: A SageMaker ``TensorFlowModel`` object.
                See :func:`~sagemaker.tensorflow.model.TensorFlowModel` for full details.
        """
        return TensorFlowModel(self.model_data, self.role, self.entry_point, source_dir=self.source_dir,
                               enable_cloudwatch_metrics=self.enable_cloudwatch_metrics, name=self._current_job_name,
                               container_log_level=self.container_log_level, code_location=self.code_location,
                               py_version=self.py_version, model_server_workers=model_server_workers,
                               sagemaker_session=self.sagemaker_session)

    def hyperparameters(self):
        """Return hyperparameters used by your custom TensorFlow code during model training."""
        hyperparameters = super(TensorFlow, self).hyperparameters()

        if not self.checkpoint_path:
            self.checkpoint_path = os.path.join(self.output_path, self._current_job_name, 'checkpoints')

        additional_hyperparameters = {'checkpoint_path': self.checkpoint_path,
                                      'training_steps': self.training_steps,
                                      'evaluation_steps': self.evaluation_steps}

        hyperparameters.update(Framework._json_encode_hyperparameters(additional_hyperparameters))
        return hyperparameters
