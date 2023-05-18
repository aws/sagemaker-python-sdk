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
from typing import Dict, List, Tuple

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
from sagemaker.session import get_execution_role, _logs_for_job, Session
from sagemaker.utils import name_from_base, _tmpdir, resolve_value_from_config
from sagemaker.s3 import s3_path_join, S3Uploader
from sagemaker import vpc_utils
from sagemaker.remote_function.core.stored_function import StoredFunction
from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentManager,
)
from sagemaker.remote_function import logging_config

# runtime script names
BOOTSTRAP_SCRIPT_NAME = "bootstrap_runtime_environment.py"
ENTRYPOINT_SCRIPT_NAME = "job_driver.sh"
PRE_EXECUTION_SCRIPT_NAME = "pre_exec.sh"
RUNTIME_MANAGER_SCRIPT_NAME = "runtime_environment_manager.py"

# training channel names
RUNTIME_SCRIPTS_CHANNEL_NAME = "sagemaker_remote_function_bootstrap"
REMOTE_FUNCTION_WORKSPACE = "sm_rf_user_ws"
JOB_REMOTE_FUNCTION_WORKSPACE = "sagemaker_remote_function_workspace"

# run context dictionary keys
KEY_EXPERIMENT_NAME = "experiment_name"
KEY_RUN_NAME = "run_name"

JOBS_CONTAINER_ENTRYPOINT = [
    "/bin/bash",
    f"/opt/ml/input/data/{RUNTIME_SCRIPTS_CHANNEL_NAME}/{ENTRYPOINT_SCRIPT_NAME}",
]

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
        environment_variables: Dict[str, str] = None,
        image_uri: str = None,
        include_local_workdir: bool = None,
        instance_count: int = 1,
        instance_type: str = None,
        job_conda_env: str = None,
        job_name_prefix: str = None,
        keep_alive_period_in_seconds: int = 0,
        max_retry_attempts: int = 1,
        max_runtime_in_seconds: int = 24 * 60 * 60,
        role: str = None,
        s3_kms_key: str = None,
        s3_root_uri: str = None,
        sagemaker_session: Session = None,
        security_group_ids: List[str] = None,
        subnets: List[str] = None,
        tags: List[Tuple[str, str]] = None,
        volume_kms_key: str = None,
        volume_size: int = 30,
        encrypt_inter_container_traffic: bool = None,
    ):

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

        self.environment_variables.update({"REMOTE_FUNCTION_SECRET_KEY": secrets.token_hex(32)})

        _image_uri = resolve_value_from_config(
            direct_input=image_uri,
            config_path=REMOTE_FUNCTION_IMAGE_URI,
            sagemaker_session=self.sagemaker_session,
        )
        if _image_uri:
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

        self.tags = self.sagemaker_session._append_sagemaker_config_tags(
            [{"Key": k, "Value": v} for k, v in tags] if tags else None, REMOTE_FUNCTION_TAGS
        )

    @staticmethod
    def _get_default_image(session):
        """Return Studio notebook image, if in Studio env. Else, base python"""

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


class _Job:
    """Helper class that interacts with the SageMaker training service."""

    def __init__(self, job_name: str, s3_uri: str, sagemaker_session: Session, hmac_key: str):
        """Initialize a _Job object."""
        self.job_name = job_name
        self.s3_uri = s3_uri
        self.sagemaker_session = sagemaker_session
        self.hmac_key = hmac_key
        self._last_describe_response = None

    @staticmethod
    def from_describe_response(describe_training_job_response, sagemaker_session):
        """Construct a _Job from a describe_training_job_response object."""
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

        Returns: the _Job object.
        """
        job_name = _Job._get_job_name(job_settings, func)
        s3_base_uri = s3_path_join(job_settings.s3_root_uri, job_name)
        hmac_key = job_settings.environment_variables["REMOTE_FUNCTION_SECRET_KEY"]

        bootstrap_scripts_s3uri = _prepare_and_upload_runtime_scripts(
            s3_base_uri=s3_base_uri,
            s3_kms_key=job_settings.s3_kms_key,
            sagemaker_session=job_settings.sagemaker_session,
        )

        dependencies_list_path = RuntimeEnvironmentManager().snapshot(job_settings.dependencies)
        user_dependencies_s3uri = _prepare_and_upload_dependencies(
            local_dependencies_path=dependencies_list_path,
            include_local_workdir=job_settings.include_local_workdir,
            pre_execution_commands=job_settings.pre_execution_commands,
            pre_execution_script_local_path=job_settings.pre_execution_script,
            s3_base_uri=s3_base_uri,
            s3_kms_key=job_settings.s3_kms_key,
            sagemaker_session=job_settings.sagemaker_session,
        )

        stored_function = StoredFunction(
            sagemaker_session=job_settings.sagemaker_session,
            s3_base_uri=s3_base_uri,
            hmac_key=hmac_key,
            s3_kms_key=job_settings.s3_kms_key,
        )

        stored_function.save(func, *func_args, **func_kwargs)

        request_dict = dict(
            TrainingJobName=job_name,
            RoleArn=job_settings.role,
            StoppingCondition={
                "MaxRuntimeInSeconds": job_settings.max_runtime_in_seconds,
            },
            RetryStrategy={"MaximumRetryAttempts": job_settings.max_retry_attempts},
        )

        if job_settings.tags:
            request_dict["Tags"] = job_settings.tags

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

        if user_dependencies_s3uri:
            input_data_config.append(
                dict(
                    ChannelName=REMOTE_FUNCTION_WORKSPACE,
                    DataSource={
                        "S3DataSource": {
                            "S3Uri": s3_path_join(s3_base_uri, REMOTE_FUNCTION_WORKSPACE),
                            "S3DataType": "S3Prefix",
                        }
                    },
                )
            )

        request_dict["InputDataConfig"] = input_data_config

        output_config = {"S3OutputPath": s3_base_uri}
        if job_settings.s3_kms_key is not None:
            output_config["KmsKeyId"] = job_settings.s3_kms_key
        request_dict["OutputDataConfig"] = output_config

        container_args = ["--s3_base_uri", s3_base_uri]
        container_args.extend(["--region", job_settings.sagemaker_session.boto_region_name])
        container_args.extend(
            ["--client_python_version", RuntimeEnvironmentManager()._current_python_version()]
        )
        if job_settings.s3_kms_key:
            container_args.extend(["--s3_kms_key", job_settings.s3_kms_key])

        if job_settings.job_conda_env:
            container_args.extend(["--job_conda_env", job_settings.job_conda_env])

        if run_info is not None:
            container_args.extend(["--run_in_context", json.dumps(dataclasses.asdict(run_info))])
        elif _RunContext.get_current_run() is not None:
            container_args.extend(
                ["--run_in_context", _convert_run_to_json(_RunContext.get_current_run())]
            )

        algorithm_spec = dict(
            TrainingImage=job_settings.image_uri,
            TrainingInputMode="File",
            ContainerEntrypoint=JOBS_CONTAINER_ENTRYPOINT,
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
            request_dict[
                "EnableInterContainerTrafficEncryption"
            ] = job_settings.encrypt_inter_container_traffic

        if job_settings.vpc_config:
            request_dict["VpcConfig"] = job_settings.vpc_config

        request_dict["Environment"] = job_settings.environment_variables

        logger.info("Creating job: %s", job_name)
        job_settings.sagemaker_session.sagemaker_client.create_training_job(**request_dict)

        return _Job(job_name, s3_base_uri, job_settings.sagemaker_session, hmac_key)

    def describe(self):
        """Describe the underlying sagemaker training job."""
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
            boto_session=self.sagemaker_session.boto_session,
            job_name=self.job_name,
            wait=True,
            timeout=timeout,
        )

    @staticmethod
    def _get_job_name(job_settings, func):
        """Get the underlying SageMaker job name from job_name_prefix or func."""
        job_name_prefix = job_settings.job_name_prefix
        if not job_name_prefix:
            job_name_prefix = func.__name__
            # remove all special characters in the beginning of function name
            job_name_prefix = re.sub(r"^[^a-zA-Z0-9]+", "", job_name_prefix)
            # convert all remaining special characters to '-'
            job_name_prefix = re.sub(r"[^a-zA-Z0-9-]", "-", job_name_prefix)
        return name_from_base(job_name_prefix)


def _prepare_and_upload_runtime_scripts(
    s3_base_uri: str, s3_kms_key: str, sagemaker_session: Session
):
    """Copy runtime scripts to a folder and upload to S3"""

    with _tmpdir() as bootstrap_scripts:

        # write entrypoint script to tmpdir
        entrypoint_script_path = os.path.join(bootstrap_scripts, ENTRYPOINT_SCRIPT_NAME)
        with open(entrypoint_script_path, "w") as file:
            file.writelines(ENTRYPOINT_SCRIPT)

        bootstrap_script_path = os.path.join(
            os.path.dirname(__file__), "runtime_environment", BOOTSTRAP_SCRIPT_NAME
        )
        runtime_manager_script_path = os.path.join(
            os.path.dirname(__file__), "runtime_environment", RUNTIME_MANAGER_SCRIPT_NAME
        )

        # copy runtime scripts to tmpdir
        shutil.copy2(bootstrap_script_path, bootstrap_scripts)
        shutil.copy2(runtime_manager_script_path, bootstrap_scripts)

        return S3Uploader.upload(
            bootstrap_scripts,
            s3_path_join(s3_base_uri, RUNTIME_SCRIPTS_CHANNEL_NAME),
            s3_kms_key,
            sagemaker_session,
        )


def _prepare_and_upload_dependencies(
    local_dependencies_path: str,
    include_local_workdir: bool,
    pre_execution_commands: List[str],
    pre_execution_script_local_path: str,
    s3_base_uri: str,
    s3_kms_key: str,
    sagemaker_session: Session,
) -> str:
    """Upload the job dependencies to S3 if present"""

    if not (
        local_dependencies_path
        or include_local_workdir
        or pre_execution_commands
        or pre_execution_script_local_path
    ):
        return None

    with _tmpdir() as tmp_dir:
        tmp_workspace_dir = os.path.join(tmp_dir, "temp_workspace/")
        os.mkdir(tmp_workspace_dir)
        # TODO Remove the following hack to avoid dir_exists error in the copy_tree call below.
        tmp_workspace = os.path.join(tmp_workspace_dir, JOB_REMOTE_FUNCTION_WORKSPACE)

        if include_local_workdir:
            shutil.copytree(
                os.getcwd(),
                tmp_workspace,
                ignore=_filter_non_python_files,
            )
            logger.info("Copied user workspace python scripts to '%s'", tmp_workspace)

        if local_dependencies_path:
            if not os.path.isdir(tmp_workspace):
                # create the directory if no workdir_path was provided in the input.
                os.mkdir(tmp_workspace)
            dst_path = shutil.copy2(local_dependencies_path, tmp_workspace)
            logger.info(
                "Copied dependencies file at '%s' to '%s'", local_dependencies_path, dst_path
            )

        if pre_execution_commands or pre_execution_script_local_path:
            if not os.path.isdir(tmp_workspace):
                os.mkdir(tmp_workspace)
            pre_execution_script = os.path.join(tmp_workspace, PRE_EXECUTION_SCRIPT_NAME)
            if pre_execution_commands:
                with open(pre_execution_script, "w") as target_script:
                    commands = [cmd + "\n" for cmd in pre_execution_commands]
                    target_script.writelines(commands)
                    logger.info(
                        "Generated pre-execution script from commands to '%s'", pre_execution_script
                    )
            else:
                shutil.copy(pre_execution_script_local_path, pre_execution_script)
                logger.info(
                    "Copied pre-execution commands from script at '%s' to '%s'",
                    pre_execution_script_local_path,
                    pre_execution_script,
                )

        workspace_archive_path = os.path.join(tmp_dir, "workspace")
        workspace_archive_path = shutil.make_archive(
            workspace_archive_path, "zip", tmp_workspace_dir
        )
        logger.info("Successfully created workdir archive at '%s'", workspace_archive_path)

        upload_path = S3Uploader.upload(
            workspace_archive_path,
            s3_path_join(s3_base_uri, REMOTE_FUNCTION_WORKSPACE),
            s3_kms_key,
            sagemaker_session,
        )
        logger.info("Successfully uploaded workdir to '%s'", upload_path)
        return upload_path


def _convert_run_to_json(run: Run) -> str:
    """Convert current run into json string"""
    run_info = _RunInfo(run.experiment_name, run.run_name)
    return json.dumps(dataclasses.asdict(run_info))


def _filter_non_python_files(path: str, names: List) -> List:
    """Ignore function for filtering out non python files."""
    to_ignore = []
    for name in names:
        full_path = os.path.join(path, name)
        if os.path.isfile(full_path):
            if not name.endswith(".py"):
                to_ignore.append(name)
        elif os.path.isdir(full_path):
            if name == "__pycache__":
                to_ignore.append(name)
        else:
            to_ignore.append(name)

    return to_ignore


@dataclasses.dataclass
class _RunInfo:
    """Data class to hold information of the run object from context."""

    experiment_name: str
    run_name: str
