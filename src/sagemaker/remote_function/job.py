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
"""Helper classes that interact with SageMaker Training service."""
from __future__ import absolute_import
import dataclasses

import os
import re
import shutil
import sys
import json
import secrets

from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from urllib.parse import urlparse
from io import BytesIO

from sagemaker.config.config_schema import (
    REMOTE_FUNCTION_ENVIRONMENT_VARIABLES,
    REMOTE_FUNCTION_IMAGE_URI,
    REMOTE_FUNCTION_DEPENDENCIES,
    REMOTE_FUNCTION_PRE_EXECUTION_COMMANDS,
    REMOTE_FUNCTION_PRE_EXECUTION_SCRIPT,
    REMOTE_FUNCTION_INCLUDE_LOCAL_WORKDIR,
    REMOTE_FUNCTION_INSTANCE_TYPE,
    REMOTE_FUNCTION_JOB_CONDA_ENV,
    REMOTE_FUNCTION_ROLE_ARN,
    REMOTE_FUNCTION_S3_ROOT_URI,
    REMOTE_FUNCTION_S3_KMS_KEY_ID,
    REMOTE_FUNCTION_VOLUME_KMS_KEY_ID,
    REMOTE_FUNCTION_TAGS,
    REMOTE_FUNCTION_VPC_CONFIG_SUBNETS,
    REMOTE_FUNCTION_VPC_CONFIG_SECURITY_GROUP_IDS,
    REMOTE_FUNCTION_ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION,
)
from sagemaker.experiments._run_context import _RunContext
from sagemaker.experiments.run import Run
from sagemaker.image_uris import get_base_python_image_uri
from sagemaker import image_uris
from sagemaker.remote_function.checkpoint_location import CheckpointLocation
from sagemaker.session import get_execution_role, _logs_for_job, Session
from sagemaker.utils import (
    name_from_base,
    _tmpdir,
    resolve_value_from_config,
    format_tags,
    Tags,
)
from sagemaker.s3 import s3_path_join, S3Uploader
from sagemaker import vpc_utils
from sagemaker.remote_function.core.stored_function import StoredFunction, _SerializedData
from sagemaker.remote_function.core.pipeline_variables import Context

from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentManager,
    _DependencySettings,
)
from sagemaker.remote_function import logging_config
from sagemaker.remote_function.spark_config import SparkConfig
from sagemaker.spark import defaults
from sagemaker.remote_function.custom_file_filter import (
    CustomFileFilter,
    copy_workdir,
    resolve_custom_file_filter_from_config_file,
)
from sagemaker.workflow.function_step import DelayedReturn
from sagemaker.workflow.step_outputs import get_step

if TYPE_CHECKING:
    from sagemaker.workflow.entities import PipelineVariable

# runtime script names
BOOTSTRAP_SCRIPT_NAME = "bootstrap_runtime_environment.py"
ENTRYPOINT_SCRIPT_NAME = "job_driver.sh"
PRE_EXECUTION_SCRIPT_NAME = "pre_exec.sh"
RUNTIME_MANAGER_SCRIPT_NAME = "runtime_environment_manager.py"
SPARK_APP_SCRIPT_NAME = "spark_app.py"

# training channel names
RUNTIME_SCRIPTS_CHANNEL_NAME = "sagemaker_remote_function_bootstrap"
REMOTE_FUNCTION_WORKSPACE = "sm_rf_user_ws"
JOB_REMOTE_FUNCTION_WORKSPACE = "sagemaker_remote_function_workspace"
SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME = "pre_exec_script_and_dependencies"

# Spark config channel and file name
SPARK_CONF_CHANNEL_NAME = "conf"
SPARK_CONF_FILE_NAME = "configuration.json"

# Spark submitted files workspace names on S3
SPARK_SUBMIT_JARS_WORKSPACE = "sm_rf_spark_jars"
SPARK_SUBMIT_PY_FILES_WORKSPACE = "sm_rf_spark_py_files"
SPARK_SUBMIT_FILES_WORKSPACE = "sm_rf_spark_data_files"
SPARK_CONF_WORKSPACE = "sm_rf_spark_conf"

# default spark version
DEFAULT_SPARK_VERSION = "3.3"
DEFAULT_SPARK_CONTAINER_VERSION = "v1"

# run context dictionary keys
KEY_EXPERIMENT_NAME = "experiment_name"
KEY_RUN_NAME = "run_name"

JOBS_CONTAINER_ENTRYPOINT = [
    "/bin/bash",
    f"/opt/ml/input/data/{RUNTIME_SCRIPTS_CHANNEL_NAME}/{ENTRYPOINT_SCRIPT_NAME}",
]

SPARK_APP_SCRIPT_PATH = f"/opt/ml/input/data/{RUNTIME_SCRIPTS_CHANNEL_NAME}/{SPARK_APP_SCRIPT_NAME}"

ENTRYPOINT_SCRIPT = f"""
#!/bin/bash

# Entry point for bootstrapping runtime environment and invoking remote function

set -eu

PERSISTENT_CACHE_DIR=${{SAGEMAKER_MANAGED_WARMPOOL_CACHE_DIRECTORY:-/opt/ml/cache}}
export CONDA_PKGS_DIRS=${{PERSISTENT_CACHE_DIR}}/sm_remotefunction_user_dependencies_cache/conda/pkgs
printf "INFO: CONDA_PKGS_DIRS is set to '$CONDA_PKGS_DIRS'\\n"
export PIP_CACHE_DIR=${{PERSISTENT_CACHE_DIR}}/sm_remotefunction_user_dependencies_cache/pip
printf "INFO: PIP_CACHE_DIR is set to '$PIP_CACHE_DIR'\\n"


printf "INFO: Bootstraping runtime environment.\\n"
python /opt/ml/input/data/{RUNTIME_SCRIPTS_CHANNEL_NAME}/{BOOTSTRAP_SCRIPT_NAME} "$@"

if [ -d {JOB_REMOTE_FUNCTION_WORKSPACE} ]
then
    if [ -f "remote_function_conda_env.txt" ]
    then
        cp remote_function_conda_env.txt {JOB_REMOTE_FUNCTION_WORKSPACE}/remote_function_conda_env.txt
    fi
    printf "INFO: Changing workspace to {JOB_REMOTE_FUNCTION_WORKSPACE}.\\n"
    cd {JOB_REMOTE_FUNCTION_WORKSPACE}
fi

if [ -f "remote_function_conda_env.txt" ]
then
    conda_env=$(cat remote_function_conda_env.txt)

    if which mamba >/dev/null; then
        conda_exe="mamba"
    else
        conda_exe="conda"
    fi

    printf "INFO: Invoking remote function inside conda environment: $conda_env.\\n"
    $conda_exe run -n $conda_env python -m sagemaker.remote_function.invoke_function "$@"
else
    printf "INFO: No conda env provided. Invoking remote function\\n"
    python -m sagemaker.remote_function.invoke_function "$@"
fi
"""

SPARK_ENTRYPOINT_SCRIPT = f"""
#!/bin/bash

# Entry point for bootstrapping runtime environment and invoking remote function for Spark

set -eu

printf "INFO: Bootstraping Spark runtime environment.\\n"

python3 /opt/ml/input/data/{RUNTIME_SCRIPTS_CHANNEL_NAME}/{BOOTSTRAP_SCRIPT_NAME} "$@"

# Spark Container entry point script to initiate the spark application
smspark-submit "$@"
"""

logger = logging_config.get_logger()


class _JobSettings:
    """Helper class that processes the job settings.

    It validates the job settings and provides default values if necessary.
    """

    def __init__(
        self,
        *,
        dependencies: str = None,
        pre_execution_commands: List[str] = None,
        pre_execution_script: str = None,
        environment_variables: Dict[str, Union[str, "PipelineVariable"]] = None,
        image_uri: Union[str, "PipelineVariable"] = None,
        include_local_workdir: bool = None,
        custom_file_filter: Optional[Union[Callable[[str, List], List], CustomFileFilter]] = None,
        instance_count: Union[int, "PipelineVariable"] = 1,
        instance_type: Union[str, "PipelineVariable"] = None,
        job_conda_env: Union[str, "PipelineVariable"] = None,
        job_name_prefix: str = None,
        keep_alive_period_in_seconds: Union[int, "PipelineVariable"] = 0,
        max_retry_attempts: Union[int, "PipelineVariable"] = 1,
        max_runtime_in_seconds: Union[int, "PipelineVariable"] = 24 * 60 * 60,
        role: str = None,
        s3_kms_key: Union[str, "PipelineVariable"] = None,
        s3_root_uri: str = None,
        sagemaker_session: Session = None,
        security_group_ids: List[Union[str, "PipelineVariable"]] = None,
        subnets: List[Union[str, "PipelineVariable"]] = None,
        tags: Optional[Tags] = None,
        volume_kms_key: Union[str, "PipelineVariable"] = None,
        volume_size: Union[int, "PipelineVariable"] = 30,
        encrypt_inter_container_traffic: Union[bool, "PipelineVariable"] = None,
        spark_config: SparkConfig = None,
        use_spot_instances=False,
        max_wait_time_in_seconds=None,
    ):
        """Initialize a _JobSettings instance which configures the remote job.

        Args:
            dependencies (str): Either the path to a dependencies file or the reserved keyword
              ``auto_capture``. Defaults to ``None``.
              If ``dependencies`` is provided, the value must be one of the following:

              * A path to a conda environment.yml file. The following conditions apply.

                * If job_conda_env is set, then the conda environment is updated by installing
                  dependencies from the yaml file and the function is invoked within that
                  conda environment. For this to succeed, the specified conda environment must
                  already exist in the image.
                * If the environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image,
                  then the conda environment is updated by installing dependencies from the
                  yaml file and the function is invoked within that conda environment. For
                  this to succeed, the conda environment name must already be set in
                  ``SAGEMAKER_JOB_CONDA_ENV``, and ``SAGEMAKER_JOB_CONDA_ENV`` must already
                  exist in the image.
                * If none of the previous conditions are met, a new conda environment named
                  ``sagemaker-runtime-env`` is created and the function annotated with the remote
                  decorator is invoked in that conda environment.

              * A path to a requirements.txt file. The following conditions apply.

                * If ``job_conda_env`` is set in the remote decorator, dependencies are installed
                  within that conda environment and the function annotated with the remote decorator
                  is invoked in the same conda environment. For this to succeed, the specified
                  conda environment must already exist in the image.
                * If an environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image,
                  dependencies are installed within that conda environment and the function
                  annotated with the remote decorator is invoked in the same. For this to succeed,
                  the conda environment name must already be set in ``SAGEMAKER_JOB_CONDA_ENV``, and
                  ``SAGEMAKER_JOB_CONDA_ENV`` must already exist in the image.
                * If none of the above conditions are met, conda is not used. Dependencies are
                  installed at the system level, without any virtual environment, and the function
                  annotated with the remote decorator is invoked using the Python runtime available
                  in the system path.

              * The parameter dependencies is set to ``auto_capture``. SageMaker will automatically
                generate an env_snapshot.yml corresponding to the current active conda environment’s
                snapshot. You do not need to provide a dependencies file. The following conditions
                apply:

                * You must run the remote function within an active conda environment.
                * When installing the dependencies on the training job, the same conditions
                  as when dependencies is set to a path to a conda environment file apply.
                  These conditions are as follows:

                  * If job_conda_env is set, then the conda environment is updated by installing
                    dependencies from the yaml file and the function is invoked within that
                    conda environment. For this to succeed, the specified conda environment must
                    already exist in the image.
                  * If the environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image,
                    then the conda environment is updated by installing dependencies from the yaml
                    file and the function is invoked within that conda environment. For this to
                    succeed, the conda environment name must already be set in
                    ``SAGEMAKER_JOB_CONDA_ENV``, and ``SAGEMAKER_JOB_CONDA_ENV`` must already exist
                    in the image.
                  * If none of the previous conditions are met, a new conda environment with name
                    ``sagemaker-runtime-env`` is created and the function annotated with the
                    remote decorator is invoked in that conda environment.

              * ``None``. SageMaker will assume that there are no dependencies to install while
                executing the remote annotated function in the training job.

            pre_execution_commands (List[str]): List of commands to be executed prior to executing
              remote function. Only one of ``pre_execution_commands`` or ``pre_execution_script``
              can be specified at the same time. Defaults to None.

            pre_execution_script (str): Path to script file to be executed prior to executing
              remote function. Only one of ``pre_execution_commands`` or ``pre_execution_script``
              can be specified at the same time. Defaults to None.

            environment_variables (dict[str, str] or dict[str, PipelineVariable]): The environment
              variables used inside the decorator function. Defaults to ``None``.

            image_uri (str, PipelineVariable): The universal resource identifier (URI) location of
              a Docker image on Amazon Elastic Container Registry (ECR). Defaults to the following
              based on where the SDK is running:

                * For users who specify ``spark_config`` and want to run the function in a Spark
                  application, the ``image_uri`` should be ``None``. A SageMaker Spark image will
                  be used for training, otherwise a ``ValueError`` is thrown.
                * For users on SageMaker Studio notebooks, the image used as the kernel image for
                  the notebook is used.
                * For other users, it is resolved to base python image with the same python version
                  as the environment running the local code.

              If no compatible image is found, a ValueError is thrown.

            include_local_workdir (bool): A flag to indicate that the remote function should include
              local directories. Set to ``True`` if the remote function code imports local modules
              and methods that are not available via PyPI or conda. Default value is ``False``.

            custom_file_filter (Callable[[str, List], List], CustomFileFilter): Either a function
              that filters job dependencies to be uploaded to S3 or a ``CustomFileFilter`` object
              that specifies the local directories and files to be included in the remote function.
              If a callable is passed in, that function is passed to the ``ignore``  argument of
              ``shutil.copytree``. Defaults to ``None``, which means only python
              files are accepted and uploaded to S3.

            instance_count (int, PipelineVariable): The number of instances to use. Defaults to 1.

            instance_type (str, PipelineVariable): The Amazon Elastic Compute Cloud (EC2) instance
              type to use to run the SageMaker job. e.g. ml.c4.xlarge. If not provided,
              a ValueError is thrown.

            job_conda_env (str, PipelineVariable): The name of the conda environment to activate
              during job's runtime. Defaults to ``None``.

            job_name_prefix (str, PipelineVariable): The prefix used to create the underlying
              SageMaker job.

            keep_alive_period_in_seconds (int, PipelineVariable): The duration in seconds to retain
              and reuse provisioned infrastructure after the completion of a training job, also
              known as SageMaker managed warm pools. The use of warm pools reduces the latency time
              spent to provision new resources. The default value for
              ``keep_alive_period_in_seconds`` is 0.
              NOTE: Additional charges associated with warm pools may apply. Using this parameter
              also activates a new persistent cache feature, which will further reduce job start up
              latency than over using SageMaker managed warm pools alone by caching the package
              source downloaded in the previous runs.

            max_retry_attempts (int, PipelineVariable): The max number of times the job is retried
              on ``InternalServerFailure`` Error from SageMaker service. Defaults to 1.

            max_runtime_in_seconds (int, PipelineVariable): The upper limit in seconds to be used
              for training. After this specified amount of time, SageMaker terminates the job
              regardless of its current status. Defaults to 1 day or (86400 seconds).

            role (str): The IAM role (either name or full ARN) used to run your SageMaker training
              job. Defaults to:

              * the SageMaker default IAM role if the SDK is running in SageMaker Notebooks or
                SageMaker Studio Notebooks.
              * if not above, a ValueError is thrown.

            s3_kms_key (str): The key used to encrypt the input and output data.
              Default to ``None``.

            s3_root_uri (str): The root S3 folder to which the code archives and data are
              uploaded to. Defaults to ``s3://<sagemaker-default-bucket>``.

            sagemaker_session (sagemaker.session.Session): The underlying SageMaker session to
              which SageMaker service calls are delegated to (default: None). If not provided,
              one is created using a default configuration chain.

            security_group_ids (List[str, PipelineVariable]): A list of security group IDs.
              Defaults to ``None`` and the training job is created without VPC config.

            subnets (List[str, PipelineVariable]): A list of subnet IDs. Defaults to ``None``
              and the job is created without VPC config.

            tags (Optional[Tags]): Tags attached to the job. Defaults to ``None``
                and the training job is created without tags.

            volume_kms_key (str, PipelineVariable): An Amazon Key Management Service (KMS) key
              used to encrypt an Amazon Elastic Block Storage (EBS) volume attached to the
              training instance. Defaults to ``None``.

            volume_size (int, PipelineVariable): The size in GB of the storage volume for storing
              input and output data during training. Defaults to ``30``.

            encrypt_inter_container_traffic (bool, PipelineVariable): A flag that specifies
              whether traffic between training containers is encrypted for the training job.
              Defaults to ``False``.

            spark_config (SparkConfig): Configurations to the Spark application that runs on
              Spark image. If ``spark_config`` is specified, a SageMaker Spark image uri
              will be used for training. Note that ``image_uri`` can not be specified at the
              same time otherwise a ``ValueError`` is thrown. Defaults to ``None``.

            use_spot_instances (bool, PipelineVariable): Specifies whether to use SageMaker
              Managed Spot instances for training. If enabled then the ``max_wait`` arg should
              also be set. Defaults to ``False``.

            max_wait_time_in_seconds (int): Timeout in seconds waiting for spot training job.
              After this amount of time Amazon SageMaker will stop waiting for managed spot
              training job to complete. Defaults to ``None``.
        """
        self.sagemaker_session = sagemaker_session or Session()
        self.environment_variables = resolve_value_from_config(
            direct_input=environment_variables,
            config_path=REMOTE_FUNCTION_ENVIRONMENT_VARIABLES,
            default_value={},
            sagemaker_session=self.sagemaker_session,
        )
        self.environment_variables.update(
            {"AWS_DEFAULT_REGION": self.sagemaker_session.boto_region_name}
        )

        # The following will be overridden by the _Job.compile method.
        # However, it needs to be kept here for feature store SDK.
        # TODO: update the feature store SDK to set the HMAC key there.
        self.environment_variables.update({"REMOTE_FUNCTION_SECRET_KEY": secrets.token_hex(32)})

        if spark_config and image_uri:
            raise ValueError("spark_config and image_uri cannot be specified at the same time!")

        if spark_config and job_conda_env:
            raise ValueError("Remote Spark jobs do not support job_conda_env.")

        if spark_config and dependencies == "auto_capture":
            raise ValueError(
                "Remote Spark jobs do not support automatically capturing dependencies."
            )

        _image_uri = resolve_value_from_config(
            direct_input=image_uri,
            config_path=REMOTE_FUNCTION_IMAGE_URI,
            sagemaker_session=self.sagemaker_session,
        )

        if spark_config:
            self.image_uri = self._get_default_spark_image(self.sagemaker_session)
            logger.info(
                "Set the image uri as %s because value of spark_config is "
                "indicating this is a remote spark job.",
                self.image_uri,
            )
        elif _image_uri:
            self.image_uri = _image_uri
        else:
            self.image_uri = self._get_default_image(self.sagemaker_session)

        self.dependencies = resolve_value_from_config(
            direct_input=dependencies,
            config_path=REMOTE_FUNCTION_DEPENDENCIES,
            sagemaker_session=self.sagemaker_session,
        )

        self.pre_execution_commands = resolve_value_from_config(
            direct_input=pre_execution_commands,
            config_path=REMOTE_FUNCTION_PRE_EXECUTION_COMMANDS,
            sagemaker_session=self.sagemaker_session,
        )

        self.pre_execution_script = resolve_value_from_config(
            direct_input=pre_execution_script,
            config_path=REMOTE_FUNCTION_PRE_EXECUTION_SCRIPT,
            sagemaker_session=self.sagemaker_session,
        )

        if self.pre_execution_commands is not None and self.pre_execution_script is not None:
            raise ValueError(
                "Only one of pre_execution_commands or pre_execution_script can be specified!"
            )

        self.include_local_workdir = resolve_value_from_config(
            direct_input=include_local_workdir,
            config_path=REMOTE_FUNCTION_INCLUDE_LOCAL_WORKDIR,
            default_value=False,
            sagemaker_session=self.sagemaker_session,
        )

        self.custom_file_filter = resolve_custom_file_filter_from_config_file(
            custom_file_filter, self.sagemaker_session
        )

        self.instance_type = resolve_value_from_config(
            direct_input=instance_type,
            config_path=REMOTE_FUNCTION_INSTANCE_TYPE,
            sagemaker_session=self.sagemaker_session,
        )
        if not self.instance_type:
            raise ValueError("instance_type is a required parameter!")

        self.instance_count = instance_count
        self.volume_size = volume_size
        self.max_runtime_in_seconds = max_runtime_in_seconds
        self.max_retry_attempts = max_retry_attempts
        self.keep_alive_period_in_seconds = keep_alive_period_in_seconds
        self.spark_config = spark_config
        self.use_spot_instances = use_spot_instances
        self.max_wait_time_in_seconds = max_wait_time_in_seconds
        self.job_conda_env = resolve_value_from_config(
            direct_input=job_conda_env,
            config_path=REMOTE_FUNCTION_JOB_CONDA_ENV,
            sagemaker_session=self.sagemaker_session,
        )
        self.job_name_prefix = job_name_prefix
        self.encrypt_inter_container_traffic = resolve_value_from_config(
            direct_input=encrypt_inter_container_traffic,
            config_path=REMOTE_FUNCTION_ENABLE_INTER_CONTAINER_TRAFFIC_ENCRYPTION,
            default_value=False,
            sagemaker_session=self.sagemaker_session,
        )
        self.enable_network_isolation = False

        _role = resolve_value_from_config(
            direct_input=role,
            config_path=REMOTE_FUNCTION_ROLE_ARN,
            sagemaker_session=self.sagemaker_session,
        )
        if _role:
            self.role = self.sagemaker_session.expand_role(_role)
        else:
            self.role = get_execution_role(self.sagemaker_session)

        self.s3_root_uri = resolve_value_from_config(
            direct_input=s3_root_uri,
            config_path=REMOTE_FUNCTION_S3_ROOT_URI,
            default_value=s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
            ),
            sagemaker_session=self.sagemaker_session,
        )

        self.s3_kms_key = resolve_value_from_config(
            direct_input=s3_kms_key,
            config_path=REMOTE_FUNCTION_S3_KMS_KEY_ID,
            sagemaker_session=self.sagemaker_session,
        )
        self.volume_kms_key = resolve_value_from_config(
            direct_input=volume_kms_key,
            config_path=REMOTE_FUNCTION_VOLUME_KMS_KEY_ID,
            sagemaker_session=self.sagemaker_session,
        )

        _subnets = resolve_value_from_config(
            direct_input=subnets,
            config_path=REMOTE_FUNCTION_VPC_CONFIG_SUBNETS,
            sagemaker_session=self.sagemaker_session,
        )
        _security_group_ids = resolve_value_from_config(
            direct_input=security_group_ids,
            config_path=REMOTE_FUNCTION_VPC_CONFIG_SECURITY_GROUP_IDS,
            sagemaker_session=self.sagemaker_session,
        )
        vpc_config = vpc_utils.to_dict(subnets=_subnets, security_group_ids=_security_group_ids)
        self.vpc_config = vpc_utils.sanitize(vpc_config)

        tags = format_tags(tags)
        self.tags = self.sagemaker_session._append_sagemaker_config_tags(tags, REMOTE_FUNCTION_TAGS)

    @staticmethod
    def _get_default_image(session):
        """Return Studio notebook image, if in Studio env. Else, base python.

        Args:
            session (Session): Boto session.

        Returns:
            Default SageMaker base python image.
        """

        if (
            "SAGEMAKER_INTERNAL_IMAGE_URI" in os.environ
            and os.environ["SAGEMAKER_INTERNAL_IMAGE_URI"]
        ):
            return os.environ["SAGEMAKER_INTERNAL_IMAGE_URI"]

        py_version = str(sys.version_info[0]) + str(sys.version_info[1])

        if py_version not in ["310", "38"]:
            raise ValueError(
                "Default image is supported only for Python versions 3.8 and 3.10. If you "
                "are using any other python version, you must provide a compatible image_uri."
            )

        region = session.boto_region_name
        image_uri = get_base_python_image_uri(region=region, py_version=py_version)

        return image_uri

    @staticmethod
    def _get_default_spark_image(session):
        """Return the Spark image.

        Args:
            session (Session): Boto session.

        Returns:
            SageMaker Spark container image uri.
        """

        region = session.boto_region_name

        py_version = str(sys.version_info[0]) + str(sys.version_info[1])

        if py_version not in ["39"]:
            raise ValueError(
                "The SageMaker Spark image for remote job only supports Python version 3.9. "
            )

        image_uri = image_uris.retrieve(
            framework=defaults.SPARK_NAME,
            region=region,
            version=DEFAULT_SPARK_VERSION,
            instance_type=None,
            py_version=f"py{py_version}",
            container_version=DEFAULT_SPARK_CONTAINER_VERSION,
        )

        return image_uri


class _Job:
    """Helper class that interacts with the SageMaker training service."""

    def __init__(self, job_name: str, s3_uri: str, sagemaker_session: Session, hmac_key: str):
        """Initialize a _Job object.

        Args:
            job_name (str): The training job name.
            s3_uri (str): The training job output S3 uri.
            sagemaker_session (Session): SageMaker boto session.
            hmac_key (str): Remote function secret key.
        """
        self.job_name = job_name
        self.s3_uri = s3_uri
        self.sagemaker_session = sagemaker_session
        self.hmac_key = hmac_key
        self._last_describe_response = None

    @staticmethod
    def from_describe_response(describe_training_job_response, sagemaker_session):
        """Construct a _Job from a describe_training_job_response object.

        Args:
            describe_training_job_response (Dict): Describe training job response.
            sagemaker_session (Session): SageMaker boto session.

        Returns:
            the _Job object.
        """
        job_name = describe_training_job_response["TrainingJobName"]
        s3_uri = describe_training_job_response["OutputDataConfig"]["S3OutputPath"]
        hmac_key = describe_training_job_response["Environment"]["REMOTE_FUNCTION_SECRET_KEY"]

        job = _Job(job_name, s3_uri, sagemaker_session, hmac_key)
        job._last_describe_response = describe_training_job_response
        return job

    @staticmethod
    def start(job_settings: _JobSettings, func, func_args, func_kwargs, run_info=None):
        """Start a training job.

        Args:
            job_settings (_JobSettings): the job settings.
            func: the function to be executed.
            func_args: the positional arguments to the function.
            func_kwargs: the keyword arguments to the function

        Returns:
            the _Job object.
        """
        job_name = _Job._get_job_name(job_settings, func)
        s3_base_uri = s3_path_join(job_settings.s3_root_uri, job_name)

        training_job_request = _Job.compile(
            job_settings=job_settings,
            job_name=job_name,
            s3_base_uri=s3_base_uri,
            func=func,
            func_args=func_args,
            func_kwargs=func_kwargs,
            run_info=run_info,
        )

        logger.info("Creating job: %s", job_name)
        job_settings.sagemaker_session.sagemaker_client.create_training_job(**training_job_request)

        return _Job(
            job_name,
            s3_base_uri,
            job_settings.sagemaker_session,
            training_job_request["Environment"]["REMOTE_FUNCTION_SECRET_KEY"],
        )

    @staticmethod
    def compile(
        job_settings: _JobSettings,
        job_name: str,
        s3_base_uri: str,
        func: callable,
        func_args: tuple,
        func_kwargs: dict,
        run_info=None,
        serialized_data: _SerializedData = None,
    ) -> dict:
        """Build the artifacts and generate the training job request."""
        from sagemaker.workflow.properties import Properties
        from sagemaker.workflow.parameters import Parameter
        from sagemaker.workflow.functions import Join
        from sagemaker.workflow.execution_variables import ExecutionVariables, ExecutionVariable
        from sagemaker.workflow.utilities import load_step_compilation_context

        step_compilation_context = load_step_compilation_context()

        jobs_container_entrypoint = JOBS_CONTAINER_ENTRYPOINT[:]

        # generate hmac key for integrity check
        if step_compilation_context is None:
            hmac_key = secrets.token_hex(32)
        else:
            hmac_key = step_compilation_context.function_step_secret_token

        # serialize function and arguments
        if step_compilation_context is None:
            stored_function = StoredFunction(
                sagemaker_session=job_settings.sagemaker_session,
                s3_base_uri=s3_base_uri,
                hmac_key=hmac_key,
                s3_kms_key=job_settings.s3_kms_key,
            )
            stored_function.save(func, *func_args, **func_kwargs)
        else:
            stored_function = StoredFunction(
                sagemaker_session=job_settings.sagemaker_session,
                s3_base_uri=s3_base_uri,
                hmac_key=hmac_key,
                s3_kms_key=job_settings.s3_kms_key,
                context=Context(
                    step_name=step_compilation_context.step_name,
                    func_step_s3_dir=step_compilation_context.pipeline_build_time,
                ),
            )

            stored_function.save_pipeline_step_function(serialized_data)

        stopping_condition = {
            "MaxRuntimeInSeconds": job_settings.max_runtime_in_seconds,
        }
        if job_settings.max_wait_time_in_seconds is not None:
            stopping_condition["MaxWaitTimeInSeconds"] = job_settings.max_wait_time_in_seconds

        request_dict = dict(
            TrainingJobName=job_name,
            RoleArn=job_settings.role,
            StoppingCondition=stopping_condition,
            RetryStrategy={"MaximumRetryAttempts": job_settings.max_retry_attempts},
        )

        _update_job_request_with_checkpoint_config(func_args, func_kwargs, request_dict)

        if job_settings.tags:
            request_dict["Tags"] = job_settings.tags

        # generate other build artifacts including workspace, requirements.txt
        request_dict["InputDataConfig"] = _generate_input_data_config(
            job_settings=job_settings, s3_base_uri=s3_base_uri
        )

        if step_compilation_context:
            s3_output_path = Join(
                on="/",
                values=[
                    s3_base_uri,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    step_compilation_context.step_name,
                    "results",
                ],
            )
            output_config = {"S3OutputPath": s3_output_path}
        else:
            output_config = {"S3OutputPath": s3_base_uri}
        if job_settings.s3_kms_key is not None:
            output_config["KmsKeyId"] = job_settings.s3_kms_key
        request_dict["OutputDataConfig"] = output_config

        container_args = ["--s3_base_uri", s3_base_uri]
        container_args.extend(["--region", job_settings.sagemaker_session.boto_region_name])
        container_args.extend(
            ["--client_python_version", RuntimeEnvironmentManager()._current_python_version()]
        )
        container_args.extend(
            [
                "--client_sagemaker_pysdk_version",
                RuntimeEnvironmentManager()._current_sagemaker_pysdk_version(),
            ]
        )
        container_args.extend(
            [
                "--dependency_settings",
                _DependencySettings.from_dependency_file_path(
                    job_settings.dependencies
                ).to_string(),
            ]
        )
        if job_settings.s3_kms_key:
            container_args.extend(["--s3_kms_key", job_settings.s3_kms_key])

        if job_settings.job_conda_env:
            container_args.extend(["--job_conda_env", job_settings.job_conda_env])

        if step_compilation_context:
            # TODO: remove the duplicates in the list
            container_args.extend(["--pipeline_step_name", step_compilation_context.step_name])
            container_args.extend(
                ["--pipeline_execution_id", ExecutionVariables.PIPELINE_EXECUTION_ID]
            )
            container_args.extend(
                ["--func_step_s3_dir", step_compilation_context.pipeline_build_time]
            )
            container_args.extend(["--property_references"])
            container_args.extend(
                [
                    ExecutionVariables.PIPELINE_EXECUTION_ID.expr["Get"],
                    ExecutionVariables.PIPELINE_EXECUTION_ID.to_string(),
                ]
            )
            for arg in func_args + tuple(func_kwargs.values()):
                if isinstance(arg, (Parameter, ExecutionVariable, Properties)):
                    container_args.extend([arg.expr["Get"], arg.to_string()])

                if isinstance(arg, DelayedReturn):
                    # The uri is a Properties object
                    uri = get_step(arg)._properties.OutputDataConfig.S3OutputPath
                    container_args.extend([uri.expr["Get"], uri.to_string()])

        if run_info is not None:
            container_args.extend(["--run_in_context", json.dumps(dataclasses.asdict(run_info))])
        elif _RunContext.get_current_run() is not None:
            container_args.extend(
                ["--run_in_context", _convert_run_to_json(_RunContext.get_current_run())]
            )

        algorithm_spec = dict(
            TrainingImage=job_settings.image_uri,
            TrainingInputMode="File",
            ContainerEntrypoint=jobs_container_entrypoint,
            ContainerArguments=container_args,
        )

        request_dict["AlgorithmSpecification"] = algorithm_spec

        resource_config = dict(
            VolumeSizeInGB=job_settings.volume_size,
            InstanceCount=job_settings.instance_count,
            InstanceType=job_settings.instance_type,
        )
        if job_settings.volume_kms_key is not None:
            resource_config["VolumeKmsKeyId"] = job_settings.volume_kms_key
        if job_settings.keep_alive_period_in_seconds is not None:
            resource_config["KeepAlivePeriodInSeconds"] = job_settings.keep_alive_period_in_seconds

        request_dict["ResourceConfig"] = resource_config

        if job_settings.enable_network_isolation is not None:
            request_dict["EnableNetworkIsolation"] = job_settings.enable_network_isolation

        if job_settings.encrypt_inter_container_traffic is not None:
            request_dict["EnableInterContainerTrafficEncryption"] = (
                job_settings.encrypt_inter_container_traffic
            )

        if job_settings.vpc_config:
            request_dict["VpcConfig"] = job_settings.vpc_config

        request_dict["EnableManagedSpotTraining"] = job_settings.use_spot_instances

        request_dict["Environment"] = job_settings.environment_variables
        request_dict["Environment"].update({"REMOTE_FUNCTION_SECRET_KEY": hmac_key})

        extended_request = _extend_spark_config_to_request(request_dict, job_settings, s3_base_uri)

        return extended_request

    def describe(self):
        """Describe the underlying sagemaker training job.

        Returns:
            Dict: Describe training job response.
        """
        if self._last_describe_response is not None and self._last_describe_response[
            "TrainingJobStatus"
        ] in ["Completed", "Failed", "Stopped"]:
            return self._last_describe_response

        self._last_describe_response = (
            self.sagemaker_session.sagemaker_client.describe_training_job(
                TrainingJobName=self.job_name
            )
        )

        return self._last_describe_response

    def stop(self):
        """Stop the underlying sagemaker training job."""
        self.sagemaker_session.sagemaker_client.stop_training_job(TrainingJobName=self.job_name)

    def wait(self, timeout: int = None):
        """Wait for the underlying sagemaker job to finish and displays its logs .

        This method blocks on the sagemaker job completing for up to the timeout value (if
        specified). If timeout is ``None``, this method will block until the job is completed.

        Args:
            timeout (int): Timeout in seconds to wait until the job is completed. ``None`` by
                default.

        Returns: None
        """

        self._last_describe_response = _logs_for_job(
            sagemaker_session=self.sagemaker_session,
            job_name=self.job_name,
            wait=True,
            timeout=timeout,
        )

    @staticmethod
    def _get_job_name(job_settings, func):
        """Get the underlying SageMaker job name from job_name_prefix or func.

        Args:
            job_settings (_JobSettings): the job settings.
            func: the function to be executed.

        Returns:
            str : the training job name.
        """
        from sagemaker.workflow.utilities import load_step_compilation_context

        step_complication_context = load_step_compilation_context()

        job_name_prefix = job_settings.job_name_prefix
        if not job_name_prefix:
            job_name_prefix = func.__name__
            # remove all special characters in the beginning of function name
            job_name_prefix = re.sub(r"^[^a-zA-Z0-9]+", "", job_name_prefix)
            # convert all remaining special characters to '-'
            job_name_prefix = re.sub(r"[^a-zA-Z0-9-]", "-", job_name_prefix)

        if step_complication_context:
            return job_name_prefix
        return name_from_base(job_name_prefix)


def _prepare_and_upload_runtime_scripts(
    spark_config: SparkConfig, s3_base_uri: str, s3_kms_key: str, sagemaker_session: Session
):
    """Copy runtime scripts to a folder and upload to S3.

    In case of remote function, s3_base_uri is s3_root_uri + function_name.
    In case of pipeline, s3_base_uri is s3_root_uri + pipeline_name. The runtime scripts are
    uploaded only once per pipeline.

    Args:
        spark_config (SparkConfig): remote Spark job configurations.

        s3_base_uri (str): S3 location that the runtime scripts will be uploaded to.

        s3_kms_key (str): kms key used to encrypt the files uploaded to S3.

        sagemaker_session (str): SageMaker boto client session.
    """

    from sagemaker.workflow.utilities import load_step_compilation_context

    step_compilation_context = load_step_compilation_context()

    if step_compilation_context and not step_compilation_context.upload_runtime_scripts:
        return s3_path_join(s3_base_uri, RUNTIME_SCRIPTS_CHANNEL_NAME)

    with _tmpdir() as bootstrap_scripts:

        # write entrypoint script to tmpdir
        entrypoint_script_path = os.path.join(bootstrap_scripts, ENTRYPOINT_SCRIPT_NAME)
        entry_point_script = ENTRYPOINT_SCRIPT
        if spark_config:
            entry_point_script = SPARK_ENTRYPOINT_SCRIPT
            spark_script_path = os.path.join(
                os.path.dirname(__file__), "runtime_environment", SPARK_APP_SCRIPT_NAME
            )
            shutil.copy2(spark_script_path, bootstrap_scripts)

        with open(entrypoint_script_path, "w", newline="\n") as file:
            file.writelines(entry_point_script)

        bootstrap_script_path = os.path.join(
            os.path.dirname(__file__), "runtime_environment", BOOTSTRAP_SCRIPT_NAME
        )
        runtime_manager_script_path = os.path.join(
            os.path.dirname(__file__), "runtime_environment", RUNTIME_MANAGER_SCRIPT_NAME
        )

        # copy runtime scripts to tmpdir
        shutil.copy2(bootstrap_script_path, bootstrap_scripts)
        shutil.copy2(runtime_manager_script_path, bootstrap_scripts)

        upload_path = S3Uploader.upload(
            bootstrap_scripts,
            s3_path_join(s3_base_uri, RUNTIME_SCRIPTS_CHANNEL_NAME),
            s3_kms_key,
            sagemaker_session,
        )

        if step_compilation_context:
            step_compilation_context.upload_runtime_scripts = False
        return upload_path


def _generate_input_data_config(job_settings: _JobSettings, s3_base_uri: str):
    """Generates input data config"""
    from sagemaker.workflow.utilities import load_step_compilation_context

    step_compilation_context = load_step_compilation_context()

    bootstrap_scripts_s3uri = _prepare_and_upload_runtime_scripts(
        spark_config=job_settings.spark_config,
        s3_base_uri=s3_base_uri,
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=job_settings.sagemaker_session,
    )

    input_data_config = [
        dict(
            ChannelName=RUNTIME_SCRIPTS_CHANNEL_NAME,
            DataSource={
                "S3DataSource": {
                    "S3Uri": bootstrap_scripts_s3uri,
                    "S3DataType": "S3Prefix",
                }
            },
        )
    ]

    local_dependencies_path = RuntimeEnvironmentManager().snapshot(job_settings.dependencies)

    if step_compilation_context:
        with _tmpdir() as tmp_dir:
            script_and_dependencies_s3uri = _prepare_dependencies_and_pre_execution_scripts(
                local_dependencies_path=local_dependencies_path,
                pre_execution_commands=job_settings.pre_execution_commands,
                pre_execution_script_local_path=job_settings.pre_execution_script,
                s3_base_uri=s3_base_uri,
                s3_kms_key=job_settings.s3_kms_key,
                sagemaker_session=job_settings.sagemaker_session,
                tmp_dir=tmp_dir,
            )

            if script_and_dependencies_s3uri:
                input_data_config.append(
                    dict(
                        ChannelName=SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME,
                        DataSource={
                            "S3DataSource": {
                                "S3Uri": script_and_dependencies_s3uri,
                                "S3DataType": "S3Prefix",
                            }
                        },
                    )
                )

    user_workspace_s3uri = _prepare_and_upload_workspace(
        local_dependencies_path=local_dependencies_path,
        include_local_workdir=job_settings.include_local_workdir,
        pre_execution_commands=job_settings.pre_execution_commands,
        pre_execution_script_local_path=job_settings.pre_execution_script,
        s3_base_uri=s3_base_uri,
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=job_settings.sagemaker_session,
        custom_file_filter=job_settings.custom_file_filter,
    )

    if user_workspace_s3uri:
        input_data_config.append(
            dict(
                ChannelName=(
                    REMOTE_FUNCTION_WORKSPACE
                    if not step_compilation_context
                    else step_compilation_context.pipeline_build_time
                ),
                DataSource={
                    "S3DataSource": {
                        "S3Uri": user_workspace_s3uri,
                        "S3DataType": "S3Prefix",
                    }
                },
            )
        )

    return input_data_config


def _prepare_dependencies_and_pre_execution_scripts(
    local_dependencies_path: str,
    pre_execution_commands: List[str],
    pre_execution_script_local_path: str,
    s3_base_uri: str,
    s3_kms_key: str,
    sagemaker_session: Session,
    tmp_dir: str,
):
    """Prepare pre-execution scripts and dependencies and upload them to s3.

    If pre execution commands are provided, a new bash file will be created
      with those commands in tmp directory.
    If pre execution script is provided, it copies that file from local file path
      to tmp directory.
    If local dependencies file is provided, it copies that file from local file path
      to tmp directory.
    If under pipeline context, tmp directory with copied dependencies and scripts is
      uploaded to S3.
    """
    from sagemaker.workflow.utilities import load_step_compilation_context

    if not (local_dependencies_path or pre_execution_commands or pre_execution_script_local_path):
        return None

    if local_dependencies_path:
        dst_path = shutil.copy2(local_dependencies_path, tmp_dir)
        logger.info("Copied dependencies file at '%s' to '%s'", local_dependencies_path, dst_path)

    if pre_execution_commands or pre_execution_script_local_path:
        pre_execution_script = os.path.join(tmp_dir, PRE_EXECUTION_SCRIPT_NAME)
        if pre_execution_commands:
            with open(pre_execution_script, "w") as target_script:
                commands = [cmd + "\n" for cmd in pre_execution_commands]
                target_script.writelines(commands)
                logger.info(
                    "Generated pre-execution script from commands to '%s'", pre_execution_script
                )
        else:
            shutil.copy2(pre_execution_script_local_path, pre_execution_script)
            logger.info(
                "Copied pre-execution commands from script at '%s' to '%s'",
                pre_execution_script_local_path,
                pre_execution_script,
            )

    step_compilation_context = load_step_compilation_context()
    if step_compilation_context:
        upload_path = S3Uploader.upload(
            tmp_dir,
            s3_path_join(
                s3_base_uri,
                step_compilation_context.step_name,
                step_compilation_context.pipeline_build_time,
                SCRIPT_AND_DEPENDENCIES_CHANNEL_NAME,
            ),
            s3_kms_key,
            sagemaker_session,
        )
        logger.info(
            "Successfully uploaded dependencies and pre execution scripts to '%s'", upload_path
        )
        return upload_path
    return None


def _prepare_and_upload_workspace(
    local_dependencies_path: str,
    include_local_workdir: bool,
    pre_execution_commands: List[str],
    pre_execution_script_local_path: str,
    s3_base_uri: str,
    s3_kms_key: str,
    sagemaker_session: Session,
    custom_file_filter: Optional[Union[Callable[[str, List], List], CustomFileFilter]] = None,
) -> str:
    """Prepare and upload the workspace to S3.

    Under pipeline context, only workdir is packaged in the workspace folder and uploaded to s3.
    Under remote function context, workdir along with pre execution scripts and dependencies
      are packaged together into the workspace folder and uploaded to S3.
    """
    from sagemaker.workflow.utilities import load_step_compilation_context

    step_compilation_context = load_step_compilation_context()

    if not (
        local_dependencies_path
        or include_local_workdir
        or pre_execution_commands
        or pre_execution_script_local_path
    ):
        return None

    func_step_s3_dir = None
    if step_compilation_context:
        func_step_s3_dir = step_compilation_context.pipeline_build_time
        if not include_local_workdir:
            return None
        if not step_compilation_context.upload_workspace:
            return s3_path_join(s3_base_uri, REMOTE_FUNCTION_WORKSPACE, func_step_s3_dir)

    with _tmpdir() as tmp_dir:
        tmp_workspace_dir = os.path.join(tmp_dir, "temp_workspace/")
        os.mkdir(tmp_workspace_dir)
        # TODO Remove the following hack to avoid dir_exists error in the copy_tree call below.
        tmp_workspace = os.path.join(tmp_workspace_dir, JOB_REMOTE_FUNCTION_WORKSPACE)

        if include_local_workdir:
            copy_workdir(tmp_workspace, custom_file_filter)
            logger.info("Copied user workspace to '%s'", tmp_workspace)

        if not os.path.isdir(tmp_workspace):
            # create the directory if no workdir_path was provided in the input.
            os.mkdir(tmp_workspace)

        if not step_compilation_context:
            _prepare_dependencies_and_pre_execution_scripts(
                local_dependencies_path=local_dependencies_path,
                pre_execution_commands=pre_execution_commands,
                pre_execution_script_local_path=pre_execution_script_local_path,
                s3_base_uri=s3_base_uri,
                s3_kms_key=s3_kms_key,
                sagemaker_session=sagemaker_session,
                tmp_dir=tmp_workspace,
            )

        workspace_archive_path = os.path.join(tmp_dir, "workspace")
        workspace_archive_path = shutil.make_archive(
            workspace_archive_path, "zip", tmp_workspace_dir
        )
        logger.info("Successfully created workdir archive at '%s'", workspace_archive_path)

        upload_path = S3Uploader.upload(
            workspace_archive_path,
            s3_path_join(s3_base_uri, REMOTE_FUNCTION_WORKSPACE, func_step_s3_dir),
            s3_kms_key,
            sagemaker_session,
        )
        logger.info("Successfully uploaded workdir to '%s'", upload_path)
        if step_compilation_context:
            step_compilation_context.upload_workspace = False
        return upload_path


def _convert_run_to_json(run: Run) -> str:
    """Convert current run into json string"""
    run_info = _RunInfo(run.experiment_name, run.run_name)
    return json.dumps(dataclasses.asdict(run_info))


def _prepare_and_upload_spark_dependent_files(
    spark_config: SparkConfig,
    s3_base_uri: str,
    s3_kms_key: str,
    sagemaker_session: Session,
) -> Tuple:
    """Upload the Spark dependencies to S3 if present.

    Args:
        spark_config (SparkConfig): The remote Spark job configurations.
        s3_base_uri (str): The S3 location that the Spark dependencies will be uploaded to.
        s3_kms_key (str): The kms key used to encrypt the files uploaded to S3.
        sagemaker_session (str): SageMaker boto client session.
    """
    if not spark_config:
        return None, None, None, None

    submit_jars_s3_paths = _upload_spark_submit_deps(
        spark_config.submit_jars,
        SPARK_SUBMIT_JARS_WORKSPACE,
        s3_base_uri,
        s3_kms_key,
        sagemaker_session,
    )
    submit_py_files_s3_paths = _upload_spark_submit_deps(
        spark_config.submit_py_files,
        SPARK_SUBMIT_PY_FILES_WORKSPACE,
        s3_base_uri,
        s3_kms_key,
        sagemaker_session,
    )
    submit_files_s3_path = _upload_spark_submit_deps(
        spark_config.submit_files,
        SPARK_SUBMIT_FILES_WORKSPACE,
        s3_base_uri,
        s3_kms_key,
        sagemaker_session,
    )
    config_file_s3_uri = _upload_serialized_spark_configuration(
        s3_base_uri, s3_kms_key, spark_config.configuration, sagemaker_session
    )

    return submit_jars_s3_paths, submit_py_files_s3_paths, submit_files_s3_path, config_file_s3_uri


def _upload_spark_submit_deps(
    submit_deps: List[str],
    workspace_name: str,
    s3_base_uri: str,
    s3_kms_key: str,
    sagemaker_session: Session,
) -> str:
    """Upload the Spark submit dependencies to S3.

    Args:
        submit_deps (List[str]): A list of path which points to the Spark dependency files.
          The path can be either a local path or S3 uri. For example ``/local/deps.jar`` or
          ``s3://<your-bucket>/deps.jar``.

        workspace_name (str): workspace name for Spark dependency.
        s3_base_uri (str): S3 location that the Spark dependencies will be uploaded to.
        s3_kms_key (str): kms key used to encrypt the files uploaded to S3.
        sagemaker_session (str): SageMaker boto client session.

    Returns:
        str : The concatenated path of all dependencies which will be passed to Spark.
    """
    spark_opt_s3_uris = []
    if not submit_deps:
        return None

    if not workspace_name or not s3_base_uri:
        raise ValueError("workspace_name or s3_base_uri may not be empty.")

    for dep_path in submit_deps:
        dep_url = urlparse(dep_path)

        if dep_url.scheme in ["s3", "s3a"]:
            spark_opt_s3_uris.append(dep_path)
        elif not dep_url.scheme or dep_url.scheme == "file":
            if not os.path.isfile(dep_path):
                raise ValueError(f"submit_deps path {dep_path} is not a valid local file.")

            upload_path = S3Uploader.upload(
                local_path=dep_path,
                desired_s3_uri=s3_path_join(s3_base_uri, workspace_name),
                kms_key=s3_kms_key,
                sagemaker_session=sagemaker_session,
            )

            spark_opt_s3_uris.append(upload_path)
            logger.info("Uploaded the local file %s to %s", dep_path, upload_path)
    return str.join(",", spark_opt_s3_uris)


def _upload_serialized_spark_configuration(
    s3_base_uri: str, s3_kms_key: str, configuration: Dict, sagemaker_session: Session
) -> str:
    """Upload the Spark configuration json to S3"""
    if not configuration:
        return None

    serialized_configuration = BytesIO(json.dumps(configuration).encode("utf-8"))
    config_file_s3_uri = s3_path_join(s3_base_uri, SPARK_CONF_WORKSPACE, SPARK_CONF_FILE_NAME)

    S3Uploader.upload_string_as_file_body(
        body=serialized_configuration,
        desired_s3_uri=config_file_s3_uri,
        kms_key=s3_kms_key,
        sagemaker_session=sagemaker_session,
    )

    logger.info("Uploaded spark configuration json %s to %s", configuration, config_file_s3_uri)

    return config_file_s3_uri


def _extend_spark_config_to_request(
    request_dict: Dict,
    job_settings: _JobSettings,
    s3_base_uri: str,
) -> Dict:
    """Extend the create training job request with spark configurations.

    Args:
        request_dict (Dict): create training job request dict.
        job_settings (_JobSettings): the job settings.
        s3_base_uri (str): S3 location that the Spark dependencies will be uploaded to.
    """
    spark_config = job_settings.spark_config

    if not spark_config:
        return request_dict

    extended_request = request_dict.copy()
    container_entrypoint = extended_request["AlgorithmSpecification"]["ContainerEntrypoint"]

    (
        submit_jars_s3_paths,
        submit_py_files_s3_paths,
        submit_files_s3_path,
        config_file_s3_uri,
    ) = _prepare_and_upload_spark_dependent_files(
        spark_config=spark_config,
        s3_base_uri=s3_base_uri,
        s3_kms_key=job_settings.s3_kms_key,
        sagemaker_session=job_settings.sagemaker_session,
    )

    input_data_config = extended_request["InputDataConfig"]

    if config_file_s3_uri:
        input_data_config.append(
            dict(
                ChannelName=SPARK_CONF_CHANNEL_NAME,
                DataSource={
                    "S3DataSource": {
                        "S3Uri": config_file_s3_uri,
                        "S3DataType": "S3Prefix",
                    }
                },
            )
        )

    for input_channel in extended_request["InputDataConfig"]:
        s3_data_source = input_channel["DataSource"].get("S3DataSource", None)
        if s3_data_source:
            s3_data_source["S3DataDistributionType"] = "FullyReplicated"

    if spark_config.spark_event_logs_uri:
        container_entrypoint.extend(
            ["--spark-event-logs-s3-uri", spark_config.spark_event_logs_uri]
        )

    if submit_jars_s3_paths:
        container_entrypoint.extend(["--jars", submit_jars_s3_paths])

    if submit_py_files_s3_paths:
        container_entrypoint.extend(["--py-files", submit_py_files_s3_paths])

    if submit_files_s3_path:
        container_entrypoint.extend(["--files", submit_files_s3_path])

    if spark_config:
        container_entrypoint.extend([SPARK_APP_SCRIPT_PATH])

    return extended_request


def _update_job_request_with_checkpoint_config(args, kwargs, request_dict):
    """Extend job request with checkpoint config based on CheckpointLocation in function args.

    Args:
        args (tuple): The positional arguments of the remote function.
        kwargs (Dict): The keyword arguments of the remote function.
        request_dict (Dict): create training job request dict.
    """
    checkpoint_location_index_in_args = None
    checkpoint_location_key_in_kwargs = None
    checkpoint_location_count = 0

    for index, arg in enumerate(args):
        if isinstance(arg, CheckpointLocation):
            checkpoint_location_index_in_args = index
            checkpoint_location_count += 1

    for key, value in kwargs.items():
        if isinstance(value, CheckpointLocation):
            checkpoint_location_key_in_kwargs = key
            checkpoint_location_count += 1

    if checkpoint_location_count < 1:
        return

    if checkpoint_location_count > 1:
        raise ValueError(
            "Remote function cannot have more than one argument of type CheckpointLocation."
        )

    if checkpoint_location_index_in_args is not None:
        checkpoint_location_arg = args[checkpoint_location_index_in_args]
    else:
        checkpoint_location_arg = kwargs[checkpoint_location_key_in_kwargs]

    checkpoint_s3_uri = checkpoint_location_arg._s3_uri
    checkpoint_local_path = checkpoint_location_arg._local_path

    request_dict["CheckpointConfig"] = {
        "LocalPath": checkpoint_local_path,
        "S3Uri": checkpoint_s3_uri,
    }


@dataclasses.dataclass
class _RunInfo:
    """Data class to hold information of the run object from context."""

    experiment_name: str
    run_name: str
