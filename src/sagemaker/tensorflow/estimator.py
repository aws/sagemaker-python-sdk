# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time

from sagemaker.estimator import Framework
import sagemaker.fw_utils as fw
from sagemaker.tensorflow.defaults import TF_VERSION
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.tensorflow.serving import Model
from sagemaker.utils import get_config_value
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

LOGGER = logging.getLogger('sagemaker')


_FRAMEWORK_MODE_ARGS = ('training_steps', 'evaluation_steps', 'requirements_file', 'checkpoint_path')
_SCRIPT_MODE = 'tensorflow-scriptmode'
_SCRIPT_MODE_SERVING_ERROR_MSG = 'Script mode containers does not support serving yet. ' \
                                 'Please use our new tensorflow-serving container by creating the model ' \
                                 'with \'endpoint_type\' set to \'tensorflow-serving\'.'
_SCRIPT_MODE_TENSORBOARD_WARNING = 'Tensorboard is not supported with script mode. You can run the following ' \
                                   'command: tensorboard --logdir {} --host localhost --port 6006 This can be ' \
                                   'run from anywhere with access to the S3 URI used as the logdir.'


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
            raise EnvironmentError(
                'TensorBoard is not installed in the system. Please install TensorBoard using the'
                ' following command: \n pip install tensorboard')

        if not self._cmd_exists('aws'):
            raise EnvironmentError(
                'The AWS CLI is not installed in the system. Please install the AWS CLI using the'
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

        for _ in range(100):
            p = subprocess.Popen(
                ["tensorboard", "--logdir", self.logdir, "--host", "localhost", "--port",
                 str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.event.wait(5)
            if p.poll():
                port += 1
            else:
                return port, p

        raise OSError(
            'No available ports to start TensorBoard. Attempted all ports between 6006 and 6105')

    def run(self):
        """Run TensorBoard process."""
        port, tensorboard_process = self.create_tensorboard_process()

        LOGGER.info('TensorBoard 0.1.7 at http://localhost:{}'.format(port))
        while not self.estimator.checkpoint_path:
            self.event.wait(1)
        with self._temporary_directory() as aws_sync_dir:
            while not self.event.is_set():
                args = ['aws', 's3', 'sync', self.estimator.checkpoint_path, aws_sync_dir]
                subprocess.call(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._sync_directories(aws_sync_dir, self.logdir)
                self.event.wait(10)
        tensorboard_process.terminate()


class TensorFlow(Framework):
    """Handle end-to-end training and deployment of user-provided TensorFlow code."""

    __framework_name__ = 'tensorflow'

    LATEST_VERSION = '1.12'
    """The latest version of TensorFlow included in the SageMaker pre-built Docker images."""

    def __init__(self, training_steps=None, evaluation_steps=None, checkpoint_path=None, py_version='py2',
                 framework_version=None, model_dir=None, requirements_file='', image_name=None,
                 script_mode=False, distributions=None, **kwargs):
        """Initialize a ``TensorFlow`` estimator.

        Args:
            training_steps (int): Perform this many steps of training. `None`, the default means train forever.
            evaluation_steps (int): Perform this many steps of evaluation. `None`, the default means that evaluation
                runs until input from eval_input_fn is exhausted (or another exception is raised).
            checkpoint_path (str): Identifies S3 location where checkpoint data during model training can be
                saved (default: None). For distributed model training, this parameter is required.
            py_version (str): Python version you want to use for executing your model training code (default: 'py2').
            framework_version (str): TensorFlow version you want to use for executing your model training code.
                List of supported versions https://github.com/aws/sagemaker-python-sdk#tensorflow-sagemaker-estimators.
                If not specified, this will default to 1.11.
            model_dir (str): S3 location where the checkpoint data and models can be exported to during training
                (default: None). If not specified a default S3 URI will be generated. It will be passed in the
                training script as one of the command line arguments.
            requirements_file (str): Path to a ``requirements.txt`` file (default: ''). The path should be within and
                relative to ``source_dir``. Details on the format can be found in the
                `Pip User Guide <https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format>`_.
            image_name (str): If specified, the estimator will use this image for training and hosting, instead of
                selecting the appropriate SageMaker official image based on framework_version and py_version. It can
                be an ECR url or dockerhub image and tag.

                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.
            script_mode (bool): If set to True will the estimator will use the Script Mode containers (default: False).
                This will be ignored if py_version is set to 'py3'.
            distributions (dict): A dictionary with information on how to run distributed training
                (default: None). Currently we support distributed training with parameter servers and MPI.
                To enable parameter server use the following setup:

                .. code:: python

                    {
                        'parameter_server':
                        {
                            'enabled': True
                        }
                    }

                To enable MPI:

                .. code:: python

                    {
                        'mpi':
                        {
                            'enabled': True
                        }
                    }

            **kwargs: Additional kwargs passed to the Framework constructor.
        """
        if framework_version is None:
            LOGGER.warning(fw.empty_framework_version_warning(TF_VERSION, self.LATEST_VERSION))
        self.framework_version = framework_version or TF_VERSION

        super(TensorFlow, self).__init__(image_name=image_name, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.py_version = py_version
        self.training_steps = training_steps
        self.evaluation_steps = evaluation_steps
        self.model_dir = model_dir
        self.script_mode = script_mode
        self.distributions = distributions or {}

        self._validate_args(py_version=py_version, script_mode=script_mode, framework_version=framework_version,
                            training_steps=training_steps, evaluation_steps=evaluation_steps,
                            requirements_file=requirements_file, checkpoint_path=checkpoint_path)
        self._validate_requirements_file(requirements_file)
        self.requirements_file = requirements_file

    def _validate_args(self, py_version, script_mode, framework_version, training_steps,
                       evaluation_steps, requirements_file, checkpoint_path):

        if py_version == 'py3' or script_mode:

            if framework_version is None:
                raise AttributeError(fw.EMPTY_FRAMEWORK_VERSION_ERROR)

            found_args = []
            if training_steps:
                found_args.append('training_steps')
            if evaluation_steps:
                found_args.append('evaluation_steps')
            if requirements_file:
                found_args.append('requirements_file')
            if checkpoint_path:
                found_args.append('checkpoint_path')
            if found_args:
                raise AttributeError(
                    '{} are deprecated in script mode. Please do not set {}.'
                    .format(', '.join(_FRAMEWORK_MODE_ARGS), ', '.join(found_args))
                )

    def _validate_requirements_file(self, requirements_file):
        if not requirements_file:
            return

        if not self.source_dir:
            raise ValueError('Must specify source_dir along with a requirements file.')

        if os.path.isabs(requirements_file):
            raise ValueError('Requirements file {} is not a path relative to source_dir.'.format(
                requirements_file))

        if not os.path.exists(os.path.join(self.source_dir, requirements_file)):
            raise ValueError('Requirements file {} does not exist.'.format(requirements_file))

    def fit(self, inputs=None, wait=True, logs=True, job_name=None, run_tensorboard_locally=False):
        """Train a model using the input training dataset.

        See :func:`~sagemaker.estimator.EstimatorBase.fit` for more details.

        Args:
            inputs (str or dict or sagemaker.session.s3_input): Information about the training data.
                This can be one of three types:

                * (str) - the S3 location where training data is saved.
                * (dict[str, str] or dict[str, sagemaker.session.s3_input]) - If using multiple channels for
                    training data, you can specify a dict mapping channel names
                    to strings or :func:`~sagemaker.session.s3_input` objects.
                * (sagemaker.session.s3_input) - channel configuration for S3 data sources that can provide
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
            super(TensorFlow, self).fit(inputs, wait, logs, job_name)

        if run_tensorboard_locally and wait is False:
            raise ValueError("Tensorboard is not supported with async fit")

        if self._script_mode_enabled() and run_tensorboard_locally:
            LOGGER.warning(_SCRIPT_MODE_TENSORBOARD_WARNING.format(self.model_dir))
            fit_super()
        elif run_tensorboard_locally:
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

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(TensorFlow, cls)._prepare_init_params_from_job_description(job_details,
                                                                                       model_channel_name)

        # Move some of the tensorflow specific init params from hyperparameters into the main init params.
        for argument in ('checkpoint_path', 'training_steps', 'evaluation_steps', 'model_dir'):
            value = init_params['hyperparameters'].pop(argument, None)
            if value is not None:
                init_params[argument] = value

        image_name = init_params.pop('image')
        framework, py_version, tag, script_mode = fw.framework_name_from_image(image_name)
        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params['image_name'] = image_name
            return init_params

        if script_mode:
            init_params['script_mode'] = True

        init_params['py_version'] = py_version

        # We switched image tagging scheme from regular image version (e.g. '1.0') to more expressive
        # containing framework version, device type and python version (e.g. '1.5-gpu-py2').
        # For backward compatibility map deprecated image tag '1.0' to a '1.4' framework version
        # otherwise extract framework version from the tag itself.
        init_params['framework_version'] = '1.4' if tag == '1.0' else fw.framework_version_from_tag(
            tag)

        training_job_name = init_params['base_job_name']
        if framework != cls.__framework_name__:
            raise ValueError("Training job: {} didn't use image for requested framework".format(
                training_job_name))

        return init_params

    def create_model(self, model_server_workers=None, role=None,
                     vpc_config_override=VPC_CONFIG_DEFAULT, endpoint_type=None):
        """Create a SageMaker ``TensorFlowModel`` object that can be deployed to an ``Endpoint``.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the model.
                Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            endpoint_type: Optional. Selects the software stack used by the inference server.
                If  not specified, the model will be configured to use the default
                SageMaker model server. If 'tensorflow-serving', the model will be configured to
                use the SageMaker Tensorflow Serving container.

        Returns:
            sagemaker.tensorflow.model.TensorFlowModel: A SageMaker ``TensorFlowModel`` object.
                See :func:`~sagemaker.tensorflow.model.TensorFlowModel` for full details.
        """

        role = role or self.role
        if endpoint_type == 'tensorflow-serving' or self._script_mode_enabled():
            return self._create_tfs_model(role=role, vpc_config_override=vpc_config_override)

        return self._create_default_model(model_server_workers=model_server_workers, role=role,
                                          vpc_config_override=vpc_config_override)

    def _create_tfs_model(self, role=None, vpc_config_override=VPC_CONFIG_DEFAULT):
        return Model(model_data=self.model_data,
                     role=role,
                     image=self.image_name,
                     name=self._current_job_name,
                     container_log_level=self.container_log_level,
                     framework_version=self.framework_version,
                     sagemaker_session=self.sagemaker_session,
                     vpc_config=self.get_vpc_config(vpc_config_override))

    def _create_default_model(self, model_server_workers, role, vpc_config_override):
        return TensorFlowModel(self.model_data, role, self.entry_point,
                               source_dir=self._model_source_dir(),
                               enable_cloudwatch_metrics=self.enable_cloudwatch_metrics,
                               env={'SAGEMAKER_REQUIREMENTS': self.requirements_file},
                               image=self.image_name,
                               name=self._current_job_name,
                               container_log_level=self.container_log_level,
                               code_location=self.code_location, py_version=self.py_version,
                               framework_version=self.framework_version,
                               model_server_workers=model_server_workers,
                               sagemaker_session=self.sagemaker_session,
                               vpc_config=self.get_vpc_config(vpc_config_override),
                               dependencies=self.dependencies)

    def hyperparameters(self):
        """Return hyperparameters used by your custom TensorFlow code during model training."""
        hyperparameters = super(TensorFlow, self).hyperparameters()

        self.checkpoint_path = self.checkpoint_path or self._default_s3_path('checkpoints')
        mpi_enabled = False

        if self._script_mode_enabled():
            additional_hyperparameters = {}

            if 'parameter_server' in self.distributions:
                ps_enabled = self.distributions['parameter_server'].get('enabled', False)
                additional_hyperparameters[self.LAUNCH_PS_ENV_NAME] = ps_enabled

            if 'mpi' in self.distributions:
                mpi_dict = self.distributions['mpi']
                mpi_enabled = mpi_dict.get('enabled', False)
                additional_hyperparameters[self.LAUNCH_MPI_ENV_NAME] = mpi_enabled
                additional_hyperparameters[self.MPI_NUM_PROCESSES_PER_HOST] = mpi_dict.get('processes_per_host', 1)
                additional_hyperparameters[self.MPI_CUSTOM_MPI_OPTIONS] = mpi_dict.get('custom_mpi_options', '')

            self.model_dir = self.model_dir or self._default_s3_path('model', mpi=mpi_enabled)
            additional_hyperparameters['model_dir'] = self.model_dir
        else:
            additional_hyperparameters = {'checkpoint_path': self.checkpoint_path,
                                          'training_steps': self.training_steps,
                                          'evaluation_steps': self.evaluation_steps,
                                          'sagemaker_requirements': self.requirements_file}

        hyperparameters.update(Framework._json_encode_hyperparameters(additional_hyperparameters))
        return hyperparameters

    def _default_s3_path(self, directory, mpi=False):
        local_code = get_config_value('local.local_code', self.sagemaker_session.config)
        if self.sagemaker_session.local_mode and local_code:
            return '/opt/ml/shared/{}'.format(directory)
        elif mpi:
            return '/opt/ml/model'
        elif self._current_job_name:
            return os.path.join(self.output_path, self._current_job_name, directory)
        else:
            return None

    def _script_mode_enabled(self):
        return self.py_version == 'py3' or self.script_mode

    def train_image(self):
        if self.image_name:
            return self.image_name

        if self._script_mode_enabled():
            return fw.create_image_uri(self.sagemaker_session.boto_region_name, _SCRIPT_MODE,
                                       self.train_instance_type, self.framework_version, self.py_version)

        return super(TensorFlow, self).train_image()
