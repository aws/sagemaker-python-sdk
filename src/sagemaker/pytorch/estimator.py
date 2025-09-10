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

import json
import logging
import math
import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Union, Optional, Dict
from urllib.request import urlretrieve

import omegaconf
from omegaconf import OmegaConf, dictconfig
from packaging.version import Version

from sagemaker.estimator import Framework, EstimatorBase
from sagemaker.inputs import TrainingInput, FileSystemInput
from sagemaker.fw_utils import (
    framework_name_from_image,
    framework_version_from_tag,
    python_deprecation_warning,
    validate_version_or_image_args,
    validate_distribution,
    profiler_config_deprecation_warning,
)
from sagemaker.git_utils import _run_clone_command
from sagemaker.image_uris import retrieve
from sagemaker.pytorch import defaults
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.pytorch.training_compiler.config import TrainingCompilerConfig
from sagemaker.session import Session
from sagemaker.vpc_utils import VPC_CONFIG_DEFAULT
from sagemaker.workflow.entities import PipelineVariable

logger = logging.getLogger("sagemaker")


def _setup_omegaconf_resolvers():
    """Set up omegaconf resolvers for training recipes."""
    if not OmegaConf.has_resolver("multiply"):
        OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    if not OmegaConf.has_resolver("divide_ceil"):
        OmegaConf.register_new_resolver(
            "divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True
        )
    if not OmegaConf.has_resolver("divide_floor"):
        OmegaConf.register_new_resolver(
            "divide_floor", lambda x, y: int(math.floor(x / y)), replace=True
        )
    if not OmegaConf.has_resolver("add"):
        OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))


def _try_resolve_recipe(recipe, key=None):
    """Try to resolve recipe and return resolved recipe."""
    if key is not None:
        recipe = dictconfig.DictConfig({key: recipe})
    try:
        OmegaConf.resolve(recipe)
    except omegaconf.errors.OmegaConfBaseException:
        return None
    if key is None:
        return recipe
    return recipe[key]


def _get_training_recipe_image_uri(image_cfg, region_name):
    """Fetch image uri given image spec and region name to use for training."""
    if isinstance(image_cfg, str):
        return image_cfg
    return retrieve(
        image_cfg.get("framework"),
        region=region_name,
        version=image_cfg.get("version"),
        image_scope="training",
        **image_cfg.get("additional_args"),
    )


def _get_training_recipe_gpu_script(code_dir, recipe, source_dir):
    """Return path to training script (entry point) when running a gpu recipe."""
    model_type_to_script = {
        "llama_v3": ("llama", "llama_pretrain.py"),
        "mistral": ("mistral", "mistral_pretrain.py"),
        "mixtral": ("mixtral", "mixtral_pretrain.py"),
        "deepseek": ("deepseek", "deepseek_pretrain.py"),
        "gpt_oss": ("custom_model", "custom_pretrain.py"),
    }

    if "model" not in recipe:
        raise ValueError("Supplied recipe does not contain required field model.")
    if "model_type" not in recipe["model"]:
        raise ValueError("Supplied recipe does not contain required field model_type.")
    model_type = recipe["model"]["model_type"]

    for key in model_type_to_script:
        if model_type.startswith(key):
            model_type = key
            break

    if model_type not in model_type_to_script:
        raise ValueError(f"Model type {model_type} not supported")

    script_dir = os.path.join(code_dir, "examples", model_type_to_script[model_type][0])
    script = model_type_to_script[model_type][1]
    shutil.copyfile(os.path.join(script_dir, script), os.path.join(source_dir, script))
    return script


def _get_training_recipe_trainium_script(code_dir, source_dir):
    """Return path to training script (entry point) when running a trainium recipe."""
    script_dir = os.path.join(code_dir, "examples")
    script = "training_orchestrator.py"
    shutil.copytree(script_dir, source_dir, dirs_exist_ok=True)
    return script


def _is_nova_recipe(recipe):
    """Check if the recipe is a Nova recipe.

    A Nova recipe is identified by:
    1. Having a run section
    2. The model_type in run has a "amazon.nova" prefix
    3. The run contains model_name_or_path

    OR

    1. Has a training_config section
    2. The training config_section has a distillation_data field

    Args:
        recipe (OmegaConf): The loaded recipe configuration

    Returns:
        bool: True if the recipe is a Nova recipe, False otherwise
    """
    # Check for nova model
    run_config = recipe.get("run", {})
    model_type = run_config.get("model_type", "").lower()
    has_nova_model = (
        model_type and "amazon.nova" in model_type and "model_name_or_path" in run_config
    )

    # Check for distillation data
    training_config = recipe.get("training_config", {})
    has_distillation = training_config.get("distillation_data") is not None

    return bool(has_nova_model) or bool(has_distillation)


def _recipe_initialize_args(source_dir):
    """Initialize the arguments dictionary for recipe setup.

    Args:
        source_dir (str): Path to the source directory.

    Returns:
        dict: Initialized arguments dictionary.

    Raises:
        ValueError: If source_dir is not a local directory.
    """
    args = {"hyperparameters": {}}

    if source_dir is None:
        args["source_dir"] = "."
    else:
        if not os.path.exists(source_dir):
            raise ValueError("When using training_recipe, source_dir must be a local directory.")
        args["source_dir"] = source_dir

    return args


def _recipe_get_region_name(kwargs):
    """Get the AWS region name from session or create a new session.

    Args:
        kwargs (dict): Dictionary of keyword arguments.

    Returns:
        str: AWS region name.
    """
    if kwargs.get("sagemaker_session") is not None:
        return kwargs.get("sagemaker_session").boto_region_name
    return Session().boto_region_name


def _recipe_load_config():
    """Load the training recipes configuration from JSON file.

    Returns:
        dict: Training recipes configuration.
    """
    training_recipes_cfg_filename = os.path.join(os.path.dirname(__file__), "training_recipes.json")
    with open(training_recipes_cfg_filename) as training_recipes_cfg_file:
        return json.load(training_recipes_cfg_file)


def _recipe_load_from_yaml(training_recipe, temp_local_recipe):
    """Load recipe from a YAML file or URL.

    Args:
        training_recipe (str): Path to the training recipe.
        temp_local_recipe (str): Path to the temporary local recipe file.

    Raises:
        ValueError: If the recipe cannot be fetched.
    """
    if os.path.isfile(training_recipe):
        shutil.copy(training_recipe, temp_local_recipe)
    else:
        try:
            urlretrieve(training_recipe, temp_local_recipe)
        except Exception as e:
            raise ValueError(
                f"Could not fetch the provided recipe {training_recipe}: exception {str(e)}"
            )


def _recipe_load_predefined(
    training_recipe, recipe_launcher_dir, temp_local_recipe, training_recipes_cfg
):
    """Load a predefined recipe from the recipe launcher.

    Args:
        training_recipe (str): Name of the predefined recipe.
        recipe_launcher_dir (str): Path to the recipe launcher directory.
        temp_local_recipe (str): Path to the temporary local recipe file.
        training_recipes_cfg (dict): Training recipes configuration.

    Raises:
        ValueError: If the recipe cannot be found.
    """
    launcher_repo = os.environ.get("TRAINING_LAUNCHER_GIT", None) or training_recipes_cfg.get(
        "launcher_repo"
    )
    _run_clone_command(launcher_repo, recipe_launcher_dir)
    recipe_path = os.path.join(
        recipe_launcher_dir,
        "recipes_collection",
        "recipes",
        training_recipe + ".yaml",
    )
    if os.path.isfile(recipe_path):
        shutil.copy(recipe_path, temp_local_recipe)
    else:
        raise ValueError(f"Recipe {training_recipe} not found.")


def _device_get_distribution(device_type):
    """Get the distribution configuration based on device type.

    Args:
        device_type (str): Device type (gpu, trainium, or cpu).

    Returns:
        dict: Distribution configuration.

    Raises:
        ValueError: If the device type is not supported.
    """
    if device_type == "gpu":
        smp_options = {
            "enabled": True,
            "parameters": {
                "placement_strategy": "cluster",
            },
        }
        return {
            "smdistributed": {"modelparallel": smp_options},
            "torch_distributed": {"enabled": True},
        }
    elif device_type == "trainium":
        return {
            "torch_distributed": {"enabled": True},
        }
    else:
        return {}


class PyTorch(Framework):
    """Handle end-to-end training and deployment of custom PyTorch code."""

    _framework_name = "pytorch"
    LAUNCH_PYTORCH_DDP_ENV_NAME = "sagemaker_pytorch_ddp_enabled"
    LAUNCH_TORCH_DISTRIBUTED_ENV_NAME = "sagemaker_torch_distributed_enabled"
    INSTANCE_TYPE_ENV_NAME = "sagemaker_instance_type"

    # [TODO] Add image uris to image_uri_config/_.json and use image_uris.retrieve
    # to retrieve the image uri below before GA.

    def __init__(
        self,
        entry_point: Optional[Union[str, PipelineVariable]] = None,
        framework_version: Optional[str] = None,
        py_version: Optional[str] = None,
        source_dir: Optional[Union[str, PipelineVariable]] = None,
        hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        distribution: Optional[Dict] = None,
        compiler_config: Optional[TrainingCompilerConfig] = None,
        training_recipe: Optional[str] = None,
        recipe_overrides: Optional[Dict] = None,
        **kwargs,
    ):
        """This ``Estimator`` executes a PyTorch script in a managed PyTorch execution environment.

        The managed PyTorch environment is an Amazon-built Docker container that executes functions
        defined in the supplied ``entry_point`` Python script within a SageMaker Training Job.

        Training is started by calling
        :meth:`~sagemaker.amazon.estimator.Framework.fit` on this Estimator.
        After training is complete, calling
        :meth:`~sagemaker.amazon.estimator.Framework.deploy` creates a hosted
        SageMaker endpoint and returns an
        :class:`~sagemaker.amazon.pytorch.model.PyTorchPredictor` instance that
        can be used to perform inference against the hosted model.

        Technical documentation on preparing PyTorch scripts for SageMaker
        training and using the PyTorch Estimator is available on the project
        home-page: https://github.com/aws/sagemaker-python-sdk

        Args:
            entry_point (str or PipelineVariable): Path (absolute or relative) to the
                Python source file which should be executed as the entry point to training.
                If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
            framework_version (str): PyTorch version you want to use for
                executing your model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided. List of supported versions:
                https://github.com/aws/deep-learning-containers/blob/master/available_images.md.
            py_version (str): Python version you want to use for executing your
                model training code. One of 'py2' or 'py3'. Defaults to ``None``. Required
                unless ``image_uri`` is provided.
            source_dir (str or PipelineVariable): Path (absolute, relative or an S3 URI) to
                a directory with any other training source code dependencies aside from the entry
                point file (default: None). If ``source_dir`` is an S3 URI, it must point to a
                file with name ``sourcedir.tar.gz``. Structure within this directory are preserved
                when training on Amazon SageMaker. Must be a local path when using training_recipe.
            hyperparameters (dict[str, str] or dict[str, PipelineVariable]): Hyperparameters
                that will be used for training (default: None). The hyperparameters are made
                accessible as a dict[str, str] to the training code on
                SageMaker. For convenience, this accepts other types for keys
                and values, but ``str()`` will be called to convert them before
                training.
            image_uri (str or PipelineVariable): If specified, the estimator will use this image
                for training and hosting, instead of selecting the appropriate
                SageMaker official image based on framework_version and
                py_version. It can be an ECR url or dockerhub image and tag.
                Examples:
                    * ``123412341234.dkr.ecr.us-west-2.amazonaws.com/my-custom-image:1.0``
                    * ``custom-image:latest``

                If ``framework_version`` or ``py_version`` are ``None``, then
                ``image_uri`` is required. If also ``None``, then a ``ValueError``
                will be raised.
            distribution (dict): A dictionary with information on how to configure and
                run distributed training
                (default: None). The following options are available.

                **To enable the SageMaker distributed data parallelism (SMDDP) library:**

                    .. code:: python

                        { "smdistributed": { "dataparallel": { "enabled": True } } }

                    Beside activating the SMDDP library through this parameter,
                    you also need to add few lines of code in your training script
                    for initializing PyTorch Distributed with the SMDDP setups.
                    To learn how to configure your training job with the SMDDP library v2, see
                    `Run distributed training with the SageMaker distributed data parallelism
                    library
                    <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html>`_
                    in the *Amazon SageMaker User Guide*.

                **To enable the SageMaker distributed model parallelism (SMP) library v2:**

                    .. code:: python

                        {
                            "torch_distributed": { "enabled": True },
                            "smdistributed": {
                                "modelparallel": {
                                    "enabled": True,
                                    "parameters": {
                                        "tensor_parallel_degree": 8,
                                        "hybrid_shard_degree": 1,
                                        ...
                                    },
                                }
                            },
                        }

                    Beside activating the SMP library v2 through this parameter,
                    you also need to add few lines of code in your training script
                    for initializing PyTorch Distributed with the SMP setups.
                    To learn how to configure your training job with the SMP library v2, see
                    `Run distributed training with the SageMaker model parallelism library v2
                    <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-v2.html>`_
                    in the *Amazon SageMaker User Guide*.

                    .. note::

                        The SageMaker distributed model parallel library v2 requires with
                        ``torch_distributed``.

                    .. note::

                        The documentation for the SMP library v1.x is archived and available at
                        `Run distributed training with the SageMaker model parallelism library
                        <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`_
                        in the *Amazon SageMaker User Guide*,
                        and the SMP v1 API reference is available in the
                        `SageMaker Python SDK v2.199.0 documentation
                        <https://sagemaker.readthedocs.io/en/v2.199.0/api/training/distributed.html#the-sagemaker-distributed-model-parallel-library>`_.

                **To enable PyTorch DDP:**

                    .. code:: python

                        {
                            "pytorchddp": {
                                "enabled": True
                            }
                        }

                    To learn more, see `Distributed PyTorch Training
                    <https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#distributed-pytorch-training>`_.

                **To enable Torch Distributed:**

                    This is available for general distributed training on
                    GPU instances from PyTorch v1.13.1 and later.

                    .. code:: python

                        {
                            "torch_distributed": {
                                "enabled": True
                            }
                        }

                    This option also supports distributed training on Trn1.
                    To learn more, see `Distributed PyTorch Training on Trainium
                    <https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#distributed-pytorch-training-on-trainium>`_.

                **To enable MPI:**

                    .. code:: python

                        {
                            "mpi": {
                                "enabled": True
                            }
                        }

                    To learn more, see `Training with Horovod
                    <https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-horovod>`_.

                **To enable parameter server:**

                    .. code:: python

                        {
                            "parameter_server": {
                                "enabled": True
                            }
                        }

                    To learn more, see `Training with parameter servers
                    <https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#training-with-parameter-servers>`_.

                **To enable distributed training with SageMaker Training Compiler:**

                    .. code:: python

                        {
                            "pytorchxla": {
                                "enabled": True
                            }
                        }

                    To learn more, see `SageMaker Training Compiler
                    <https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html>`_
                    in the *Amazon SageMaker Developer Guide*.

                    .. note::

                        When you use this PyTorch XLA option for distributed training strategy,
                        you must add the ``compiler_config`` parameter and activate SageMaker
                        Training Compiler.

                compiler_config (:class:`~sagemaker.pytorch.TrainingCompilerConfig`):
                Configures SageMaker Training Compiler to accelerate training.

            training_recipe (str): Training recipe to use. This is a local file path, a url,
                                   or a recipe provided by Amazon SageMaker HyperPod recipes,
                                   such as training/llama/hf_llama3_70b_seq8k_gpu_p5x64_pretrain.
                                   This is required when using recipes.
            recipe_overrides (Dict): Dictionary specifying key values to override in the
                                     training_recipe. This is optional when using
                                     Amazon SageMaker HyperPod recipes.

            **kwargs: Additional kwargs passed to the :class:`~sagemaker.estimator.Framework`
                constructor.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.estimator.Framework` and
            :class:`~sagemaker.estimator.EstimatorBase`.
        """
        self.is_nova_recipe = False
        if training_recipe is not None:
            if entry_point is not None:
                logger.warning("Argument entry_point will be ignored with training_recipe.")
            if hyperparameters is not None:
                logger.warning("Argument hyperparameters will be ignored with training recipe.")
            if distribution is not None:
                logger.warning("Argument distribution will be ignored with training_recipe.")
            args = self._setup_for_training_recipe(
                training_recipe, recipe_overrides, source_dir, kwargs
            )

            if self.is_nova_recipe and image_uri is None:
                raise ValueError("Must supply image_uri for nova jobs.")

            entry_point = args["entry_point"]
            source_dir = args["source_dir"]
            hyperparameters = args["hyperparameters"]
            if image_uri is None:
                image_uri = args["default_image_uri"]
            distribution = args["distribution"]
        elif entry_point is None:
            raise ValueError(
                "Argument entry_point must be set when training_recipe is not provided"
            )
        validate_version_or_image_args(framework_version, py_version, image_uri)
        if py_version == "py2":
            logger.warning(
                python_deprecation_warning(self._framework_name, defaults.LATEST_PY2_VERSION)
            )
        self.framework_version = framework_version
        self.py_version = py_version

        if "enable_sagemaker_metrics" not in kwargs:
            # enable sagemaker metrics for PT v1.3 or greater:
            if self.framework_version and Version(self.framework_version) >= Version("1.3"):
                kwargs["enable_sagemaker_metrics"] = True

        super(PyTorch, self).__init__(
            entry_point,
            source_dir,
            hyperparameters,
            image_uri=image_uri,
            is_nova_job=self.is_nova_recipe,
            **kwargs,
        )

        if "entry_point" not in kwargs:
            kwargs["entry_point"] = entry_point

        if distribution is not None:
            # rewrite pytorchddp to smdistributed
            if "pytorchddp" in distribution:
                if "smdistributed" in distribution:
                    raise ValueError(
                        "Cannot use both pytorchddp and smdistributed "
                        "distribution options together.",
                        distribution,
                    )

                # convert pytorchddp distribution into smdistributed distribution
                distribution = distribution.copy()
                distribution["smdistributed"] = {"dataparallel": distribution["pytorchddp"]}
                del distribution["pytorchddp"]

            distribution = validate_distribution(
                distribution,
                self.instance_groups,
                self._framework_name,
                framework_version,
                py_version,
                image_uri,
                kwargs,
            )

        self.distribution = distribution or {}

        if compiler_config is not None:
            if not isinstance(compiler_config, TrainingCompilerConfig):
                error_string = (
                    f"Expected instance of type {TrainingCompilerConfig}"
                    f"for argument compiler_config. "
                    f"Instead got {type(compiler_config)}"
                )
                raise ValueError(error_string)
            if compiler_config:
                compiler_config.validate(self)
        elif distribution is not None and "pytorchxla" in distribution:
            raise ValueError(
                "Distributed training through PyTorch XLA is currently only supported "
                "when SageMaker Training Compiler is enabled. To learn more, "
                "see Enable SageMaker Training Compiler at "
                "https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler-enable.html."
            )
        self.compiler_config = compiler_config

        if "profiler_config" in kwargs:
            profiler_config_deprecation_warning(
                kwargs["profiler_config"], image_uri, self._framework_name, framework_version
            )

    def _pytorch_distribution_configuration(self, distribution):
        """Returns a dict of distribution config for PyTorch training

        Args:
            distribution (dict): A dictionary with information on how to run distributed training.
        Returns:
            dict containing Pytorch DDP config
        """
        distribution_config = {}
        pytorch_ddp_enabled = False
        torch_distributed_enabled = False

        if "pytorchddp" in distribution:
            pytorch_ddp_enabled = distribution.get("pytorchddp").get("enabled", False)
        elif "torch_distributed" in distribution:
            torch_distributed_enabled = distribution.get("torch_distributed").get("enabled", False)

        if pytorch_ddp_enabled:
            distribution_config[self.LAUNCH_PYTORCH_DDP_ENV_NAME] = pytorch_ddp_enabled
            if self.instance_type is not None:
                distribution_config[self.INSTANCE_TYPE_ENV_NAME] = self.instance_type
        elif torch_distributed_enabled:
            if "smdistributed" in distribution:
                # Enable torch_distributed for smdistributed.
                distribution_config = self._distribution_configuration(distribution=distribution)
            distribution_config[self.LAUNCH_TORCH_DISTRIBUTED_ENV_NAME] = torch_distributed_enabled
            if self.instance_type is not None:
                distribution_config[self.INSTANCE_TYPE_ENV_NAME] = self.instance_type
        else:
            distribution_config = self._distribution_configuration(distribution=distribution)

        return distribution_config

    def hyperparameters(self):
        """Return hyperparameters used by your custom PyTorch code during model training."""
        hyperparameters = super(PyTorch, self).hyperparameters()
        additional_hyperparameters = self._pytorch_distribution_configuration(
            distribution=self.distribution
        )
        hyperparameters.update(
            EstimatorBase._json_encode_hyperparameters(additional_hyperparameters)
        )
        if self.compiler_config:
            training_compiler_hyperparameters = self.compiler_config._to_hyperparameter_dict()
            hyperparameters.update(
                EstimatorBase._json_encode_hyperparameters(training_compiler_hyperparameters)
            )

        return hyperparameters

    def fit(
        self,
        inputs: Optional[Union[str, Dict, TrainingInput, FileSystemInput]] = None,
        wait: bool = True,
        logs: str = "All",
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
    ):
        """Train a model using the input training dataset.

        Adds the recipe file to the inputs when a training recipe is used.

        Args:
            inputs (str or dict or sagemaker.inputs.TrainingInput or
                sagemaker.inputs.FileSystemInput): Information about the training data.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs ([str]): A list of strings specifying which logs to print.
            job_name (str): Training job name.
            experiment_config (dict[str, str]): Experiment management configuration.

        Returns:
            None or pipeline step arguments
        """
        # Handle recipe upload and input channel creation if we have a recipe
        if (
            self.is_nova_recipe is not None
            and self.is_nova_recipe
            and hasattr(self, "training_recipe_file")
            and self.training_recipe_file
        ):
            # Upload the recipe to S3 if it hasn't been uploaded yet
            if not hasattr(self, "recipe_s3_uri") or not self.recipe_s3_uri:
                self.recipe_s3_uri = self._upload_recipe_to_s3(
                    self.sagemaker_session, self.training_recipe_file.name
                )

            # Prepare inputs dictionary
            from sagemaker.inputs import TrainingInput

            if inputs is None:
                inputs = {}
            elif not isinstance(inputs, dict):
                inputs = {"training": inputs}

            # Add the recipe channel
            recipe_channel_name = "recipe"
            inputs[recipe_channel_name] = TrainingInput(
                s3_data=os.path.dirname(self.recipe_s3_uri), input_mode="File"
            )

            # Update hyperparameters to reference the recipe location in the container
            recipe_filename = os.path.basename(self.training_recipe_file.name)

            self._hyperparameters.update(
                {
                    "sagemaker_recipe_local_path": f"/opt/ml/input/data/{recipe_channel_name}/{recipe_filename}",
                }
            )
        return super(PyTorch, self).fit(
            inputs=inputs,
            wait=wait,
            logs=logs,
            job_name=job_name,
            experiment_config=experiment_config,
        )

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=VPC_CONFIG_DEFAULT,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        **kwargs,
    ):
        """Create a SageMaker ``PyTorchModel`` object that can be deployed to an ``Endpoint``.

        Args:
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``,
                which is also used during transform jobs. If not specified, the
                role from the Estimator will be used.
            vpc_config_override (dict[str, list[str]]): Optional override for VpcConfig set on
                the model. Default: use subnets and security groups from this Estimator.
                * 'Subnets' (list[str]): List of subnet ids.
                * 'SecurityGroupIds' (list[str]): List of security group ids.
            entry_point (str): Path (absolute or relative) to the local Python source file which
                should be executed as the entry point to training. If ``source_dir`` is specified,
                then ``entry_point`` must point to a file located at the root of ``source_dir``.
                If not specified, the training entry point is used.
            source_dir (str): Path (absolute or relative) to a directory with any other serving
                source code dependencies aside from the entry point file.
                If not specified, the model source directory from training is used.
            dependencies (list[str]): A list of paths to directories (absolute or relative) with
                any additional libraries that will be exported to the container.
                If not specified, the dependencies from training are used.
                This is not supported with "local code" in Local Mode.
            **kwargs: Additional kwargs passed to the :class:`~sagemaker.pytorch.model.PyTorchModel`
                constructor.

        Returns:
            sagemaker.pytorch.model.PyTorchModel: A SageMaker ``PyTorchModel``
            object. See :func:`~sagemaker.pytorch.model.PyTorchModel` for full details.
        """
        if "image_uri" not in kwargs:
            kwargs["image_uri"] = self.image_uri

        kwargs["name"] = self._get_or_create_name(kwargs.get("name"))

        return PyTorchModel(
            self.model_data,
            role or self.role,
            entry_point or self._model_entry_point(),
            framework_version=self.framework_version,
            py_version=self.py_version,
            source_dir=(source_dir or self._model_source_dir()),
            container_log_level=self.container_log_level,
            code_location=self.code_location,
            model_server_workers=model_server_workers,
            sagemaker_session=self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            dependencies=(dependencies or self.dependencies),
            **kwargs,
        )

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        """Convert the job description to init params that can be handled by the class constructor.

        Args:
            job_details: the returned job details from a describe_training_job
                API call.
            model_channel_name (str): Name of the channel where pre-trained
                model data will be downloaded.

        Returns:
            dictionary: The transformed init_params
        """
        init_params = super(PyTorch, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
        image_uri = init_params.pop("image_uri")
        framework, py_version, tag, _ = framework_name_from_image(image_uri)
        if framework:
            framework = framework.split("-")[0]

        if tag is None:
            framework_version = None
        else:
            framework_version = framework_version_from_tag(tag)
        init_params["framework_version"] = framework_version
        init_params["py_version"] = py_version

        if not framework:
            # If we were unable to parse the framework name from the image it is not one of our
            # officially supported images, in this case just add the image to the init params.
            init_params["image_uri"] = image_uri
            return init_params

        if framework != cls._framework_name:
            raise ValueError(
                "Training job: {} didn't use image for requested framework".format(
                    job_details["TrainingJobName"]
                )
            )

        return init_params

    # The old class methods have been replaced by static methods and module-level functions

    @staticmethod
    def _recipe_load(training_recipe, recipe_launcher_dir, training_recipes_cfg):
        """Load the recipe from file path, URL, or predefined recipe.

        Args:
            training_recipe (str): Path to the training recipe.
            recipe_launcher_dir (str): Path to the recipe launcher directory.
            training_recipes_cfg (dict): Training recipes configuration.

        Returns:
            tuple: Recipe name and loaded recipe.

        Raises:
            ValueError: If the recipe cannot be fetched or found.
        """
        recipe_name = os.path.splitext(os.path.basename(training_recipe))[0]
        temp_local_recipe = tempfile.NamedTemporaryFile(prefix=recipe_name, suffix=".yaml").name

        try:
            if training_recipe.endswith(".yaml"):
                _recipe_load_from_yaml(training_recipe, temp_local_recipe)
            else:
                _recipe_load_predefined(
                    training_recipe, recipe_launcher_dir, temp_local_recipe, training_recipes_cfg
                )

            recipe = OmegaConf.load(temp_local_recipe)
            os.unlink(temp_local_recipe)
            return recipe_name, recipe
        except Exception as e:
            if os.path.exists(temp_local_recipe):
                os.unlink(temp_local_recipe)
            raise e

    @staticmethod
    def _device_get_image_uri(args, device_type, recipe_config, region_name, recipe):
        """Get the appropriate image URI based on device type.

        Args:
            args (dict): Arguments dictionary.
            device_type (str): Device type (gpu, trainium, or cpu).
            recipe_config (dict): Training recipes configuration.
            region_name (str): AWS region name.
            recipe (OmegaConf): Recipe configuration.

        Returns:
            str: Image URI or None if no image URI was found.
        """
        if "default_image_uri" in args:
            logger.debug("Image URI already exists")
            return args["default_image_uri"]
        elif device_type == "gpu":
            logger.info("Using GPU training image")
            return _get_training_recipe_image_uri(recipe_config.get("gpu_image"), region_name)
        elif device_type == "trainium":
            logger.info("Using Trainium training image")
            return _get_training_recipe_image_uri(recipe_config.get("neuron_image"), region_name)
        else:
            return None

    @staticmethod
    def _recipe_setup_nova(args, recipe):
        """Set up configuration for Nova recipes.

        Args:
            args (dict): Arguments dictionary.
            recipe (OmegaConf): Recipe configuration.
            kwargs (dict): Dictionary of keyword arguments.
        """
        run_config = recipe.get("run", {})
        model_name_or_path = run_config.get("model_name_or_path")

        # Set hyperparameters based on model_name_or_path
        if model_name_or_path:
            if model_name_or_path.startswith("s3://"):
                args["hyperparameters"]["base_model_location"] = model_name_or_path
            else:
                args["hyperparameters"]["base_model"] = model_name_or_path

        args["entry_point"] = None
        args["source_dir"] = None

    @staticmethod
    def _device_validate_and_get_type(kwargs, recipe):
        """Validate instance type and determine device type.

        Args:
            kwargs (dict): Dictionary of keyword arguments.
            recipe (OmegaConf): Recipe configuration.

        Returns:
            str: Device type (gpu, trainium, or cpu).

        Raises:
            ValueError: If instance_type is not provided or recipe is invalid.
        """
        if "instance_type" not in kwargs:
            raise ValueError("Must pass instance type to estimator when using training recipes.")

        if not _is_nova_recipe(recipe) and "trainer" not in recipe:
            raise ValueError("Supplied recipe does not contain required field trainer.")

        instance_type = kwargs["instance_type"].split(".")[1]
        if instance_type.startswith(("p", "g")):
            return "gpu"
        elif instance_type.startswith("trn"):
            return "trainium"
        else:
            return "cpu"

    @staticmethod
    def _device_handle_instance_count(kwargs, recipe):
        """Handle instance count configuration.

        Args:
            kwargs (dict): Dictionary of keyword arguments.
            recipe (OmegaConf): Recipe configuration.

        Raises:
            ValueError: If instance_count is not provided and cannot be found in the recipe.
        """
        # Check if instance_count is already provided in kwargs

        is_nova = _is_nova_recipe(recipe)
        if "instance_count" in kwargs:
            # Warn if there are conflicting configurations in the recipe
            if "num_nodes" in recipe.get("trainer", {}):
                logger.warning(
                    "Using instance_count argument to estimator to set number "
                    "of nodes. Ignoring trainer -> num_nodes in recipe."
                )
            if is_nova and "replicas" in recipe.get("run", {}):
                logger.warning(
                    "Using instance_count argument to estimator to set number "
                    "of nodes. Ignoring run -> replicas in recipe."
                )
            return

        # Try to get instance_count from recipe
        if "trainer" in recipe and "num_nodes" in recipe["trainer"]:
            kwargs["instance_count"] = recipe["trainer"]["num_nodes"]
            return

        if is_nova and "run" in recipe and "replicas" in recipe["run"]:
            kwargs["instance_count"] = recipe["run"]["replicas"]
            return

        # If we get here, we couldn't find instance_count anywhere
        raise ValueError(
            "Must set either instance_count argument for estimator or "
            "set trainer -> num_nodes or run -> replicas in recipe for nova jobs."
        )

    @staticmethod
    def _device_get_entry_point_script(
        device_type, recipe_train_dir, recipe, source_dir, training_recipes_cfg
    ):
        """Get the entry point script based on device type.

        Args:
            device_type (str): Device type (gpu, trainium, or cpu).
            recipe_train_dir (str): Path to the recipe training directory.
            recipe (OmegaConf): Recipe configuration.
            source_dir (str): Path to the source directory.
            training_recipes_cfg (dict): Training recipes configuration.

        Returns:
            str: Path to the entry point script or None if not applicable.
        """
        if device_type == "gpu":
            adapter_repo = os.environ.get("TRAINING_ADAPTER_GIT", None) or training_recipes_cfg.get(
                "adapter_repo"
            )
            _run_clone_command(adapter_repo, recipe_train_dir)
            return _get_training_recipe_gpu_script(recipe_train_dir, recipe, source_dir)
        elif device_type == "trainium":
            _run_clone_command(training_recipes_cfg.get("neuron_dist_repo"), recipe_train_dir)
            return _get_training_recipe_trainium_script(recipe_train_dir, source_dir)
        elif device_type == "cpu":
            raise ValueError(
                f"Devices of type {device_type} are not supported with training recipes."
            )
        return None

    def _recipe_resolve_and_save(self, recipe, recipe_name, source_dir):
        """Resolve and save the final recipe configuration.

        Args:
            recipe (OmegaConf): Recipe configuration.
            recipe_name (str): Recipe name.
            source_dir (str): Path to the source directory.

        Returns:
            OmegaConf: Resolved recipe configuration.

        Raises:
            RuntimeError: If the recipe cannot be resolved.
        """
        _setup_omegaconf_resolvers()

        # Try different resolution strategies
        final_recipe = _try_resolve_recipe(recipe)
        if final_recipe is None:
            final_recipe = _try_resolve_recipe(recipe, "recipes")
        if final_recipe is None:
            final_recipe = _try_resolve_recipe(recipe, "training")
        if final_recipe is None:
            raise RuntimeError("Could not resolve provided recipe.")

        # Save the resolved recipe - this sets an instance attribute
        self.training_recipe_file = tempfile.NamedTemporaryFile(
            dir=source_dir,
            prefix=recipe_name + "_",
            suffix=".yaml",
        )
        OmegaConf.save(config=final_recipe, f=self.training_recipe_file.name)

        return final_recipe

    def _upload_recipe_to_s3(self, session, recipe_file_path):
        """Upload the recipe file to S3.

        Args:
            session (sagemaker.session.Session): SageMaker session.
            recipe_file_path (str): Path to the recipe file.

        Returns:
            str: S3 URI of the uploaded recipe file.
        """
        bucket = session.default_bucket()
        key_prefix = session.default_bucket_prefix

        recipe_filename = os.path.basename(recipe_file_path)

        readable_date = datetime.fromtimestamp(int(time.time()))
        date_format = readable_date.strftime("%Y-%m-%d")

        if key_prefix != "None" and key_prefix is not None:
            s3_key = f"{key_prefix}/recipes/{date_format}_{recipe_filename[:-5]}"
        else:
            s3_key = f"recipes/{date_format}_{recipe_filename[:-5]}"

        # Upload the recipe file to S3
        s3_uri = session.upload_data(
            path=recipe_file_path,
            bucket=bucket,
            key_prefix=os.path.dirname(os.path.join(s3_key, recipe_filename)),
        )

        # Return the full S3 URI to the recipe file
        return f"{s3_uri}"

    def _setup_for_training_recipe(self, training_recipe, recipe_overrides, source_dir, kwargs):
        """Performs training recipe specific setup and returns recipe specific args.

        Updates kwargs and returns a dictionary of args to use for estimator
        initialization and setup when using a training recipe.

        Args:
            training_recipe (str): A recipe which is a local file path, a url or a
                                   sagemaker training recipe.
            recipe_overrides (Dict): Dictionary specifying key values to override in the
                                    training recipe.
            source_dir (str): Path (absolute, or relative) to a directory where to copy
                              the scripts for training recipe.
            kwargs (dict): Dictionary of args used for estimator initialization.

        Returns:
            dict containing arg values for estimator initialization and setup.
        """
        region_name = _recipe_get_region_name(kwargs)
        training_recipes_cfg = _recipe_load_config()
        recipe_overrides = recipe_overrides or {}

        # Create temporary directories for recipe processing
        with (
            tempfile.TemporaryDirectory(prefix="training_") as recipe_train_dir,
            tempfile.TemporaryDirectory(prefix="launcher_") as recipe_launcher_dir,
        ):
            # Load and process the recipe
            recipe_name, recipe = PyTorch._recipe_load(
                training_recipe, recipe_launcher_dir, training_recipes_cfg
            )

            # Merge with overrides
            recipe = OmegaConf.merge(recipe, recipe_overrides)

            self.is_nova_recipe = _is_nova_recipe(recipe)
            if self.is_nova_recipe:
                return self._setup_for_nova_recipe(
                    recipe,
                    recipe_name,
                    source_dir,
                    kwargs,
                )
            else:
                return self._setup_for_standard_recipe(
                    recipe,
                    recipe_name,
                    source_dir,
                    kwargs,
                    recipe_train_dir,
                    training_recipes_cfg,
                    region_name,
                )

    def _setup_for_nova_recipe(
        self,
        recipe,
        recipe_name,
        source_dir,
        kwargs,
    ):
        """Set up configuration specifically for Nova recipes.

        Args:
            recipe (OmegaConf): Recipe configuration.
            recipe_name (str): Recipe name.
            source_dir (str): Path to the source directory.
            kwargs (dict): Dictionary of keyword arguments.

        Returns:
            dict: Arguments dictionary for estimator initialization.
        """
        # Initialize args
        args = _recipe_initialize_args(source_dir)

        # Set up Nova-specific configuration
        run_config = recipe.get("run", {})
        model_name_or_path = run_config.get("model_name_or_path")

        # Set hyperparameters based on model_name_or_path
        if model_name_or_path:
            if model_name_or_path.startswith("s3://"):
                args["hyperparameters"]["base_model_location"] = model_name_or_path
            else:
                args["hyperparameters"]["base_model"] = model_name_or_path

        args["entry_point"] = None
        args["source_dir"] = None
        args["distribution"] = {}

        logger.info("Remote debugging, profiler and debugger hooks are disabled for Nova recipes.")
        kwargs["enable_remote_debug"] = False
        kwargs["disable_profiler"] = True
        kwargs["debugger_hook_config"] = False

        # Handle instance count for Nova recipes
        if "instance_count" in kwargs:
            if "replicas" in recipe.get("run", {}):
                logger.warning(
                    "Using instance_count argument to estimator to set number "
                    "of nodes. Ignoring run -> replicas in recipe."
                )
        elif "run" in recipe and "replicas" in recipe["run"]:
            kwargs["instance_count"] = recipe["run"]["replicas"]
        else:
            raise ValueError(
                "Must set either instance_count argument for estimator or "
                "set run -> replicas in recipe for nova jobs."
            )

        training_config = recipe.get("training_config", {})
        is_distillation = training_config.get("distillation_data", {})
        if bool(is_distillation):
            args["hyperparameters"]["distillation_data"] = is_distillation
            args["hyperparameters"]["role_arn"] = kwargs["role"]
            kms_key = training_config.get("kms_key")
            if kms_key is None:
                ValueError(
                    'Nova distillation job recipe requires "kms_key" field in "training_config"'
                )
            args["hyperparameters"]["kms_key"] = kms_key

        # Resolve and save the final recipe
        self._recipe_resolve_and_save(recipe, recipe_name, args["source_dir"])

        return args

    def _setup_for_standard_recipe(
        self,
        recipe,
        recipe_name,
        source_dir,
        kwargs,
        recipe_train_dir,
        training_recipes_cfg,
        region_name,
    ):
        """Set up configuration for standard (non-Nova) recipes.

        Args:
            recipe (OmegaConf): Recipe configuration.
            recipe_name (str): Recipe name.
            source_dir (str): Path to the source directory.
            kwargs (dict): Dictionary of keyword arguments.
            recipe_train_dir (str): Path to the recipe training directory.
            training_recipes_cfg (dict): Training recipes configuration.
            region_name (str): AWS region name.

        Returns:
            dict: Arguments dictionary for estimator initialization.
        """
        # Initialize args
        args = _recipe_initialize_args(source_dir)

        # Validate recipe structure
        if "trainer" not in recipe:
            raise ValueError("Supplied recipe does not contain required field trainer.")

        # Handle instance count for standard recipes
        if "instance_count" in kwargs:
            if "num_nodes" in recipe.get("trainer", {}):
                logger.warning(
                    "Using instance_count argument to estimator to set number "
                    "of nodes. Ignoring trainer -> num_nodes in recipe."
                )
        elif "trainer" in recipe and "num_nodes" in recipe["trainer"]:
            kwargs["instance_count"] = recipe["trainer"]["num_nodes"]
        else:
            raise ValueError(
                "Must set either instance_count argument for estimator or "
                "set trainer -> num_nodes in recipe."
            )

        # Determine device type
        device_type = PyTorch._device_validate_and_get_type(kwargs, recipe)

        # Get image URI
        image_uri = PyTorch._device_get_image_uri(
            args, device_type, training_recipes_cfg, region_name, recipe
        )
        args["default_image_uri"] = image_uri if image_uri is not None else ""

        # Setup device-specific configuration
        args["distribution"] = _device_get_distribution(device_type)

        # Set entry point if not already set
        if "entry_point" not in args:
            script = PyTorch._device_get_entry_point_script(
                device_type, recipe_train_dir, recipe, args["source_dir"], training_recipes_cfg
            )
            if script:
                args["entry_point"] = os.path.basename(script)

        # Handle container configuration
        if "container" in recipe and not recipe["container"]:
            logger.warning(
                "Ignoring container from training_recipe. Use image_uri arg for estimator."
            )

        # Resolve and save the final recipe
        self._recipe_resolve_and_save(recipe, recipe_name, args["source_dir"])

        # Update hyperparameters with recipe configuration
        args["hyperparameters"].update(
            {
                "config-path": ".",
                "config-name": os.path.basename(self.training_recipe_file.name),
            }
        )

        return args
