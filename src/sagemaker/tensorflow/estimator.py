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
"""An estimator class for training with TensorFlow on Amazon SageMaker."""
from __future__ import absolute_import

import logging
import os

from packaging import version

from sagemaker import utils
from sagemaker.debugger import DebuggerHookConfig
from sagemaker.estimator import Framework
import sagemaker.fw_utils as fw
from sagemaker.tensorflow import defaults
from sagemaker.tensorflow.serving import Model
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT

logger = logging.getLogger("sagemaker")

# TODO: consider creating a function for generating this command before removing this constant
_SCRIPT_MODE_TENSORBOARD_WARNING = (
    "Tensorboard is not supported with script mode. You can run the following "
    "command: tensorboard --logdir %s --host localhost --port 6006 This can be "
    "run from anywhere with access to the S3 URI used as the logdir."
)


class TensorFlow(Framework):
    """Handle end-to-end training and deployment of user-provided TensorFlow code."""

    __framework_name__ = "tensorflow"
    _SCRIPT_MODE_REPO_NAME = "tensorflow-scriptmode"

    LATEST_VERSION = defaults.LATEST_VERSION

    _LATEST_1X_VERSION = "1.15.2"

    _HIGHEST_LEGACY_MODE_ONLY_VERSION = version.Version("1.10.0")
    _LOWEST_SCRIPT_MODE_ONLY_VERSION = version.Version("1.13.1")

    _HIGHEST_PYTHON_2_VERSION = version.Version("2.1.0")

    def __init__(
        self,
        py_version=None,
        framework_version=None,
        model_dir=None,
        image_name=None,
        distributions=None,
        script_mode=True,
        **kwargs
    ):
        """Initialize a ``TensorFlow`` estimator.

        Args:
            py_version (str): Python version you want to use for executing your model training
                code (default: 'py2').
            framework_version (str): TensorFlow version you want to use for executing your model
                training code. List of supported versions
                https://github.com/aws/sagemaker-python-sdk#tensorflow-sagemaker-estimators.
                If not specified, this will default to 1.11.
            model_dir (str): S3 location where the checkpoint data and models can be exported to
                during training (default: None). It will be passed in the training script as one of
                the command line arguments. If not specified, one is provided based on
                your training configuration:

                * *distributed training with MPI* - ``/opt/ml/model``
                * *single-machine training or distributed training without MPI* - \
                    ``s3://{output_path}/model``
                * *Local Mode with local sources (file:// instead of s3://)* - \
                    ``/opt/ml/shared/model``

            image_name (str): If specified, the estimator will use this image for training and
                hosting, instead of selecting the appropriate SageMaker official image based on
                framework_version and py_version. It can be an ECR url or dockerhub image and tag.

                Examples:
                    123.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0
                    custom-image:latest.
            distributions (dict): A dictionary with information on how to run distributed training
                (default: None). Currently we support distributed training with parameter servers
                and MPI.
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

            script_mode (bool): Whether or not to use the Script Mode TensorFlow images
                (default: True).
            **kwargs: Additional kwargs passed to the Framework constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        if framework_version is None:
            logger.warning(
                fw.empty_framework_version_warning(defaults.TF_VERSION, self.LATEST_VERSION)
            )
        self.framework_version = framework_version or defaults.TF_VERSION

        if not py_version:
            py_version = "py3" if self._only_python_3_supported() else "py2"
        if py_version == "py2":
            logger.warning(
                fw.python_deprecation_warning(self.__framework_name__, defaults.LATEST_PY2_VERSION)
            )

        if distributions is not None:
            train_instance_type = kwargs.get("train_instance_type")
            fw.warn_if_parameter_server_with_multi_gpu(
                training_instance_type=train_instance_type, distributions=distributions
            )

        if "enable_sagemaker_metrics" not in kwargs:
            # enable sagemaker metrics for TF v1.15 or greater:
            if fw.is_version_equal_or_higher([1, 15], self.framework_version):
                kwargs["enable_sagemaker_metrics"] = True

        super(TensorFlow, self).__init__(image_name=image_name, **kwargs)

        self.py_version = py_version
        self.model_dir = model_dir
        self.distributions = distributions or {}

        self._script_mode_enabled = script_mode
        self._validate_args(py_version=py_version, framework_version=self.framework_version)

    def _validate_args(self, py_version, framework_version):
        """Placeholder docstring"""

        if py_version == "py3":
            if framework_version is None:
                raise AttributeError(fw.EMPTY_FRAMEWORK_VERSION_ERROR)

        if py_version == "py2" and self._only_python_3_supported():
            msg = (
                "Python 2 containers are only available with {} and lower versions. "
                "Please use a Python 3 container.".format(defaults.LATEST_PY2_VERSION)
            )
            raise AttributeError(msg)

        if (not self._script_mode_enabled) and self._only_script_mode_supported():
            logger.warning(
                "Legacy mode is deprecated in versions 1.13 and higher. Using script mode instead."
            )
            self._script_mode_enabled = True

        if self._only_legacy_mode_supported():
            # TODO: add link to docs to explain how to use legacy mode with v2
            logger.warning(
                "TF %s supports only legacy mode. If you were using any legacy mode parameters "
                "(training_steps, evaluation_steps, checkpoint_path, requirements_file), "
                "make sure to pass them directly as hyperparameters instead.",
                self.framework_version,
            )
            self._script_mode_enabled = False

    def _only_legacy_mode_supported(self):
        """Placeholder docstring"""
        return version.Version(self.framework_version) <= self._HIGHEST_LEGACY_MODE_ONLY_VERSION

    def _only_script_mode_supported(self):
        """Placeholder docstring"""
        return version.Version(self.framework_version) >= self._LOWEST_SCRIPT_MODE_ONLY_VERSION

    def _only_python_3_supported(self):
        """Placeholder docstring"""
        return version.Version(self.framework_version) > self._HIGHEST_PYTHON_2_VERSION

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(TensorFlow, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )

        model_dir = init_params["hyperparameters"].pop("model_dir", None)
        if model_dir is not None:
            init_params["model_dir"] = model_dir

        image_name = init_params.pop("image")
        framework, py_version, tag, script_mode = fw.framework_name_from_image(image_name)
        if not framework:
            # If we were unable to parse the framework name from the image, it is not one of our
            # officially supported images, so just add the image to the init params.
            init_params["image_name"] = image_name
            return init_params

        if script_mode is None:
            init_params["script_mode"] = False

        init_params["py_version"] = py_version

        # We switched image tagging scheme from regular image version (e.g. '1.0') to more
        # expressive containing framework version, device type and python version
        # (e.g. '1.5-gpu-py2'). For backward compatibility map deprecated image tag '1.0' to a
        # '1.4' framework version otherwise extract framework version from the tag itself.
        init_params["framework_version"] = (
            "1.4" if tag == "1.0" else fw.framework_version_from_tag(tag)
        )

        training_job_name = init_params["base_job_name"]
        if framework != cls.__framework_name__:
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    training_job_name
                )
            )

        return init_params

    def create_model(
        self,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        **kwargs
    ):
        """Create a ``Model`` object that can be used for creating SageMaker model entities,
        deploying to a SageMaker endpoint, or starting SageMaker Batch Transform jobs.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also
                used during transform jobs. If not specified, the role from the Estimator is used.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on the
                model. Default: use subnets and security groups from this Estimator.

                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.

            entry_point (str): Path (absolute or relative) to the local Python source file which
                should be executed as the entry point to training (default: None).
            source_dir (str): Path (absolute or relative) to a directory with any other serving
                source code dependencies aside from the entry point file (default: None).
            dependencies (list[str]): A list of paths to directories (absolute or relative) with
                any additional libraries that will be exported to the container (default: None).
            **kwargs: Additional kwargs passed to :class:`~sagemaker.tensorflow.serving.Model`.

        Returns:
            sagemaker.tensorflow.serving.Model: A ``Model`` object.
                See :class:`~sagemaker.tensorflow.serving.Model` for full details.
        """
        if "image" not in kwargs:
            kwargs["image"] = self.image_name

        if "name" not in kwargs:
            kwargs["name"] = self._current_job_name

        if "enable_network_isolation" not in kwargs:
            kwargs["enable_network_isolation"] = self.enable_network_isolation()

        return Model(
            model_data=self.model_data,
            role=role or self.role,
            container_log_level=self.container_log_level,
            framework_version=self.framework_version,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            entry_point=entry_point,
            source_dir=source_dir,
            dependencies=dependencies,
            **kwargs
        )

    def hyperparameters(self):
        """Return hyperparameters used by your custom TensorFlow code during model training."""
        hyperparameters = super(TensorFlow, self).hyperparameters()
        additional_hyperparameters = {}

        if self._script_mode_enabled:
            mpi_enabled = False

            if "parameter_server" in self.distributions:
                ps_enabled = self.distributions["parameter_server"].get("enabled", False)
                additional_hyperparameters[self.LAUNCH_PS_ENV_NAME] = ps_enabled

            if "mpi" in self.distributions:
                mpi_dict = self.distributions["mpi"]
                mpi_enabled = mpi_dict.get("enabled", False)
                additional_hyperparameters[self.LAUNCH_MPI_ENV_NAME] = mpi_enabled

                if mpi_dict.get("processes_per_host"):
                    additional_hyperparameters[self.MPI_NUM_PROCESSES_PER_HOST] = mpi_dict.get(
                        "processes_per_host"
                    )

                additional_hyperparameters[self.MPI_CUSTOM_MPI_OPTIONS] = mpi_dict.get(
                    "custom_mpi_options", ""
                )

            self.model_dir = self.model_dir or self._default_s3_path("model", mpi=mpi_enabled)
            additional_hyperparameters["model_dir"] = self.model_dir

        hyperparameters.update(Framework._json_encode_hyperparameters(additional_hyperparameters))
        return hyperparameters

    def _default_s3_path(self, directory, mpi=False):
        """Placeholder docstring"""
        local_code = utils.get_config_value("local.local_code", self.sagemaker_session.config)
        if self.sagemaker_session.local_mode and local_code:
            return "/opt/ml/shared/{}".format(directory)
        if mpi:
            return "/opt/ml/model"
        if self._current_job_name:
            return os.path.join(self.output_path, self._current_job_name, directory)
        return None

    def _validate_and_set_debugger_configs(self):
        """Disable Debugger Hook Config for ParameterServer (PS) as it is not
        supported in smdebug.

        Else, set default HookConfig
        """
        ps_enabled = "parameter_server" in self.distributions and self.distributions[
            "parameter_server"
        ].get("enabled", False)
        if ps_enabled:
            if self.debugger_hook_config is not None or self.debugger_rule_configs is not None:
                logger.info(
                    "Amazon SageMaker Debugger does not currently support "
                    "Parameter Server distribution"
                )
            self.debugger_hook_config = None
            self.debugger_rule_configs = None
        elif self.debugger_hook_config is None and fw._region_supports_debugger(
            self.sagemaker_session.boto_session.region_name
        ):
            # Set defaults for debugging.
            self.debugger_hook_config = DebuggerHookConfig(s3_output_path=self.output_path)

    def train_image(self):
        """Placeholder docstring"""
        if self.image_name:
            return self.image_name

        if self._script_mode_enabled:
            return fw.create_image_uri(
                self.sagemaker_session.boto_region_name,
                self._SCRIPT_MODE_REPO_NAME,
                self.train_instance_type,
                self.framework_version,
                self.py_version,
            )

        return super(TensorFlow, self).train_image()
