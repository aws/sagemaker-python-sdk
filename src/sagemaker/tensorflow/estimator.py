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
import logging
import os
import subprocess
import tempfile
import threading

from sagemaker.estimator import Framework
from sagemaker.fw_utils import create_image_uri, framework_name_from_image, framework_version_from_tag

from sagemaker.tensorflow.defaults import TF_VERSION
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
        self.logdir = logdir or tempfile.mkdtemp()

    @staticmethod
    def _cmd_exists(cmd):
        return any(
            os.access(os.path.join(path, cmd), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )

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
            args = ['aws', 's3', 'sync', self.estimator.checkpoint_path, self.logdir]
            subprocess.call(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.event.wait(10)
        tensorboard_process.terminate()


class TensorFlow(Framework):
    """Handle end-to-end training and deployment of user-provided TensorFlow code."""

    __framework_name__ = 'tensorflow'

    def __init__(self, training_steps=None, evaluation_steps=None, checkpoint_path=None, py_version='py2',
                 framework_version=TF_VERSION, requirements_file='', **kwargs):
        """Initialize an ``TensorFlow`` estimator.
        Args:
            training_steps (int): Perform this many steps of training. `None`, the default means train forever.
            evaluation_steps (int): Perform this many steps of evaluation. `None`, the default means that evaluation
                runs until input from eval_input_fn is exhausted (or another exception is raised).
            checkpoint_path (str): Identifies S3 location where checkpoint data during model training can be
                saved (default: None). For distributed model training, this parameter is required.
            py_version (str): Python version you want to use for executing your model training code (default: 'py2').
            framework_version (str): TensorFlow version you want to use for executing your model training code.
                List of supported versions https://github.com/aws/sagemaker-python-sdk#tensorflow-sagemaker-estimators
            requirements_file (str): Path to a ``requirements.txt`` file (default: ''). The path should be within and
                relative to ``source_dir``. Details on the format can be found in the
                `Pip User Guide <https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format>`_.
            **kwargs: Additional kwargs passed to the Framework constructor.
        """
        super(TensorFlow, self).__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.py_version = py_version
        self.framework_version = framework_version
        self.training_steps = training_steps
        self.evaluation_steps = evaluation_steps

        self._validate_requirements_file(requirements_file)
        self.requirements_file = requirements_file

    def _validate_requirements_file(self, requirements_file):
        if not requirements_file:
            return

        if not self.source_dir:
            raise ValueError('Must specify source_dir along with a requirements file.')

        if os.path.isabs(requirements_file):
            raise ValueError('Requirements file {} is not a path relative to source_dir.'.format(requirements_file))

        if not os.path.exists(os.path.join(self.source_dir, requirements_file)):
            raise ValueError('Requirements file {} does not exist.'.format(requirements_file))

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

        if run_tensorboard_locally and wait is False:
            raise ValueError("Tensorboard is not supported with async fit")

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
    def _prepare_init_params_from_job_description(cls, job_details):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(TensorFlow, cls)._prepare_init_params_from_job_description(job_details)

        # Move some of the tensorflow specific init params from hyperparameters into the main init params.
        for argument in ['checkpoint_path', 'training_steps', 'evaluation_steps']:
            value = init_params['hyperparameters'].pop(argument, None)
            if value is not None:
                init_params[argument] = value

        framework, py_version, tag = framework_name_from_image(init_params.pop('image'))
        init_params['py_version'] = py_version

        # We switched image tagging scheme from regular image version (e.g. '1.0') to more expressive
        # containing framework version, device type and python version (e.g. '1.5-gpu-py2').
        # For backward compatibility map deprecated image tag '1.0' to a '1.4' framework version
        # otherwise extract framework version from the tag itself.
        init_params['framework_version'] = '1.4' if tag == '1.0' else framework_version_from_tag(tag)

        training_job_name = init_params['base_job_name']
        if framework != cls.__framework_name__:
            raise ValueError("Training job: {} didn't use image for requested framework".format(training_job_name))

        return init_params

    def train_image(self):
        """Return the Docker image to use for training.

        The  :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does the model training, calls this method to
        find the image to use for model training.

        Returns:
            str: The URI of the Docker image.
        """
        return create_image_uri(self.sagemaker_session.boto_session.region_name, self.__framework_name__,
                                self.train_instance_type, self.framework_version, py_version=self.py_version)

    def create_model(self, model_server_workers=None):
        """Create a SageMaker ``TensorFlowModel`` object that can be deployed to an ``Endpoint``.

        Args:
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.

        Returns:
            sagemaker.tensorflow.model.TensorFlowModel: A SageMaker ``TensorFlowModel`` object.
                See :func:`~sagemaker.tensorflow.model.TensorFlowModel` for full details.
        """
        env = {'SAGEMAKER_REQUIREMENTS': self.requirements_file}
        return TensorFlowModel(self.model_data, self.role, self.entry_point, source_dir=self.source_dir,
                               enable_cloudwatch_metrics=self.enable_cloudwatch_metrics, env=env,
                               name=self._current_job_name, container_log_level=self.container_log_level,
                               code_location=self.code_location, py_version=self.py_version,
                               framework_version=self.framework_version, model_server_workers=model_server_workers,
                               sagemaker_session=self.sagemaker_session)

    def hyperparameters(self):
        """Return hyperparameters used by your custom TensorFlow code during model training."""
        hyperparameters = super(TensorFlow, self).hyperparameters()

        if not self.checkpoint_path:
            self.checkpoint_path = os.path.join(self.output_path, self._current_job_name, 'checkpoints')

        additional_hyperparameters = {'checkpoint_path': self.checkpoint_path,
                                      'training_steps': self.training_steps,
                                      'evaluation_steps': self.evaluation_steps,
                                      'sagemaker_requirements': self.requirements_file}

        hyperparameters.update(Framework._json_encode_hyperparameters(additional_hyperparameters))
        return hyperparameters
