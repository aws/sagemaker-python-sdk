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
"""The notebook job step definitions for workflow."""
from __future__ import absolute_import

import re
import shutil
import os

from typing import (
    List,
    Optional,
    Union,
    Dict,
)

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.retry import RetryPolicy
from sagemaker.workflow.steps import (
    Step,
    ConfigurableRetryStep,
    StepTypeEnum,
)
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.step_outputs import StepOutput

from sagemaker.workflow.entities import (
    RequestType,
    PipelineVariable,
)
from sagemaker.workflow.utilities import _collect_parameters, load_step_compilation_context
from sagemaker.session import get_execution_role

from sagemaker.s3_utils import s3_path_join
from sagemaker.s3 import S3Uploader
from sagemaker.utils import _tmpdir, name_from_base, resolve_value_from_config, format_tags, Tags
from sagemaker import vpc_utils

from sagemaker.config.config_schema import (
    NOTEBOOK_JOB_ROLE_ARN,
    NOTEBOOK_JOB_S3_ROOT_URI,
    NOTEBOOK_JOB_S3_KMS_KEY_ID,
    NOTEBOOK_JOB_VOLUME_KMS_KEY_ID,
    NOTEBOOK_JOB_VPC_CONFIG_SUBNETS,
    NOTEBOOK_JOB_VPC_CONFIG_SECURITY_GROUP_IDS,
)


# disable E1101 as collect_parameters decorator sets the attributes
# pylint: disable=E1101
class NotebookJobStep(ConfigurableRetryStep):
    """`NotebookJobStep` for SageMaker Pipelines Workflows.

    For more details about SageMaker Notebook Jobs, see `SageMaker Notebook Jobs
    <https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-auto-run.html>`_.
    """

    @_collect_parameters
    def __init__(
        self,
        # Following parameters will set by @collect_parameters
        # pylint: disable=W0613
        input_notebook: str,
        image_uri: str,
        kernel_name: str,
        name: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        notebook_job_name: Optional[str] = None,
        role: Optional[str] = None,
        s3_root_uri: Optional[str] = None,
        parameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        environment_variables: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        initialization_script: Optional[str] = None,
        s3_kms_key: Optional[Union[str, PipelineVariable]] = None,
        instance_type: Optional[Union[str, PipelineVariable]] = "ml.m5.large",
        volume_size: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        encrypt_inter_container_traffic: Optional[Union[bool, PipelineVariable]] = True,
        security_group_ids: Optional[List[Union[str, PipelineVariable]]] = None,
        subnets: Optional[List[Union[str, PipelineVariable]]] = None,
        max_retry_attempts: int = 1,
        max_runtime_in_seconds: int = 2 * 24 * 60 * 60,
        tags: Optional[Tags] = None,
        additional_dependencies: Optional[List[str]] = None,
        # pylint: enable=W0613
        retry_policies: Optional[List[RetryPolicy]] = None,
        depends_on: Optional[List[Union[Step, StepCollection, StepOutput]]] = None,
    ):
        """Constructs a `NotebookJobStep`.

        Args:
            name (Optional[str]): The name of the `NotebookJobStep`. If not provided,
               it is derived from the notebook file name.
            display_name (Optional[str]): The display name of the `NotebookJobStep`.
               Default is ``None``.
            description (Optional[str]): The description of the `NotebookJobStep`.
               Default is ``None``.

            notebook_job_name (Optional[str]): An optional user-specified descriptive name
              for the notebook job. If provided, the sanitized notebook job name
              is used as a prefix for the underlying training job. If not provided, it
              is derived from the notebook file name.
            input_notebook (str): A required local path pointing to
              the notebook that needs to be executed. The notebook file is uploaded to
              ``{s3_root_uri}/{pipeline_name}/{step_name}/input-{timestamp}`` in the job preparation
              step.
            image_uri (str): A required universal resource identifier (URI)
              location of a Docker image on Amazon Elastic Container Registry (ECR). Use the
              following images:

              * SageMaker Distribution Images. See `Available Amazon SageMaker Images
                <https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-images.html>`_
                to get the image ECR URIs.
              * Custom images with required dependencies installed.
                For information about notebook job image requirements, see `Image Constraints
                <https://docs.aws.amazon.com/sagemaker/latest/dg/notebook-auto-run-constraints.html#notebook-auto-run-constraints-image>`_.

            kernel_name (str): A required name of the kernel that is used to run
              the notebook. The kernelspec of the specified kernel needs to be registered in the
              image.
            role (str): An IAM role (either name or full ARN) used to
              run your SageMaker training job. Defaults to one of the following:

                * The SageMaker default IAM role if the SDK is running in SageMaker Notebooks
                  or SageMaker Studio Notebooks.
                * Otherwise, a ``ValueError`` is thrown.

            s3_root_uri (str): The root S3 folder to which the notebook job input and output are
              uploaded. The inputs and outputs are uploaded to the following folders, respectively:

              ``{s3_root_uri}/{pipeline_name}/{step_name}/input-{timestamp}``
              ``{s3_root_uri}/{pipeline_name}/{execution_id}/{step_name}/{job_name}/output``

              Note that ``job_name`` is the name of the underlying SageMaker training job.

            parameters (Dict[str, Union[str, PipelineVariable]]): Key-value pairs passed to the
              notebook execution for parameterization. Defaults to ``None``.
            environment_variables (Dict[str, Union[str, PipelineVariable]]): The environment
              variables used inside the job image container. They could be existing environment
              variables that you want to override, or new environment variables that you want to
              introduce and use in your notebook. Defaults to ``None``.
            initialization_script (str): A path to a local script you can run
              when your notebook starts up. An initialization script is sourced from the same shell
              as the notebook job. This script is uploaded to
              ``{s3_root_uri}/{pipeline_name}/{step_name}/input-{timestamp}``
              in the job preparation step. Defaults to ``None``.
            s3_kms_key (str, PipelineVariable): A KMS key to use if you want to
              customize the encryption key used for your notebook job input and output. If you do
              not specify this field, your notebook job outputs are encrypted with SSE-KMS using the
              default Amazon S3 KMS key. Defaults to ``None``.

            instance_type (str, PipelineVariable): The Amazon Elastic Compute Cloud (EC2) instance
              type to use to run the notebook job. The notebook job uses a SageMaker Training Job
              as a computing layer, so the specified instance type should be a SageMaker Training
              supported instance type. Defaults to ``ml.m5.large``.
            volume_size (int, PipelineVariable): The size in GB of the storage volume for storing
              input and output data during training. Defaults to ``30``.
            volume_kms_key (str, PipelineVariable): An Amazon Key Management Service (KMS) key used
              to encrypt an Amazon Elastic Block Storage (EBS) volume attached to the training
              instance. Defaults to ``None``.
            encrypt_inter_container_traffic (bool, PipelineVariable): A flag that specifies whether
              traffic between training containers is encrypted for the training job. Defaults to
              ``True``.
            security_group_ids (List[str, PipelineVariable]): A list of security group IDs.
              Defaults to ``None`` and the training job is created without a VPC config.
            subnets (List[str, PipelineVariable]): A list of subnet IDs. Defaults to ``None`` and
              the job is created without a VPC config.

            max_retry_attempts (int): The max number of times the job is retried
              after an ``InternalServerFailure`` error configured in the underlying SageMaker
              training job. Defaults to 1.
            max_runtime_in_seconds (int): The maximum length of time, in seconds,
              that a notebook job can run before it is stopped. If you configure both the max run
              time and max retry attempts, the run time applies to each retry. If a job does not
              complete in this time, its status is set to ``Failed``. Defaults to ``172800 seconds(2
              days)``.
            tags (Optional[Tags]): Tags attached to the job. Defaults to ``None`` and the training
                job is created without tags. Your tags control how the Studio UI captures and
                displays the job created by the pipeline in the following ways:

              * If you only attach the domain tag, then the notebook job is displayed to all user
                profiles and spaces.
              * If the domain and user profile/space tags are attached, then the notebook job
                is displayed to those specific user profiles and spaces.
              * If you do not attach any domain or user profile/space tags, the Studio UI does
                not show the notebook job created by pipeline step. You have to use the training
                job console to view the underlying training job.

            additional_dependencies:(List[str]): The list of dependencies for the notebook job. The
              list contains the local files or folder paths. The dependent files or folders are
              uploaded to ``{s3_root_uri}/{pipeline_name}/{step_name}/input-{timestamp}``.
              If a path is pointing to a directory, the subfolders are uploaded recursively.
              Defaults to ``None``.
            sagemaker_session (sagemaker.session.Session): The underlying SageMaker session to
              which SageMaker service calls are delegated. Default is ``None``. If not provided,
              one is created using a default configuration chain.

            retry_policies (List[RetryPolicy]):  A list of retry policies for the notebook job step.
            depends_on (List[Union[Step, StepCollection, StepOutput]]): A list of `Step`/
                `StepCollection`/`StepOutput` instances on which this `NotebookJobStep` depends.
        """

        super(NotebookJobStep, self).__init__(
            name,
            StepTypeEnum.TRAINING,
            display_name,
            description,
            depends_on,
            retry_policies,
        )

        # if notebook job name or step name is not passed,
        # use a name derived from notebook file name
        # Disable E0203 as input_notebook attribute is set by the decorator
        # pylint: disable=E0203
        if self.input_notebook and (not self.notebook_job_name or not self.name):
            derived_name = name_from_base(
                self._get_job_name_prefix(os.path.basename(self.input_notebook))
            )
            if not self.notebook_job_name:
                self.notebook_job_name = derived_name
            if not self.name:
                self.name = derived_name
        # pylint: enable=E0203

        self._scheduler_container_entry_point = ["amazon_sagemaker_scheduler"]
        self._scheduler_container_arguments = []

        self._properties = self._compose_properties(name)

    def _validate_inputs(self):
        """Validation function for the notebook job step inputs."""

        errors = []
        # notebook job name should start with letters and contain only letters, numbers, hyphens,
        # and underscores
        if not self.notebook_job_name or not bool(
            re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", self.notebook_job_name)
        ):
            errors.append(
                f"Notebook Job Name({self.notebook_job_name}) is not valid. Valid name "
                f"should start with letters and contain only letters, numbers, hyphens, "
                f"and underscores."
            )

        # input notebook is required
        if not self.input_notebook or not os.path.isfile(self.input_notebook):
            errors.append(
                f"The required input notebook({self.input_notebook}) is not a valid " f"file."
            )

        # init script is optional
        if self.initialization_script and not os.path.isfile(self.initialization_script):
            errors.append(f"The initialization script({self.input_notebook}) is not a valid file.")

        if self.additional_dependencies:
            for path in self.additional_dependencies:
                if not os.path.exists(path):
                    errors.append(
                        f"The path({path}) specified in additional dependencies does not exist."
                    )
        # image uri is required
        if not self.image_uri or self._region_from_session not in self.image_uri:
            errors.append(
                f"The image uri(specified as {self.image_uri}) is required and "
                f"should be hosted in same region of the session"
                f"({self._region_from_session})."
            )

        if not self.kernel_name:
            errors.append("The kernel name is required.")

        # validate the role after resolving.
        if not self.role:
            errors.append(f"The IAM role is '{self.role}' and no default role can be found.")

        # validate the s3 root uri after resolving
        if not self.s3_root_uri:
            errors.append(
                f"The s3_root_uri is '{self.s3_root_uri}' and no default s3_root_uri can"
                f" be found."
            )

        if errors:
            errors_message = "\n  - ".join(errors)
            raise ValueError(f"Validation Errors: \n  - {errors_message}")

    def _resolve_defaults(self):
        """Resolve intelligent defaults"""

        # resolving logic runs in argument method out of __init__
        # pylint: disable=W0201
        _role = resolve_value_from_config(
            direct_input=self.role,  # pylint: disable=E0203
            config_path=NOTEBOOK_JOB_ROLE_ARN,
            sagemaker_session=self.sagemaker_session,
        )
        if _role:
            self.role = self.sagemaker_session.expand_role(_role)
        else:
            self.role = get_execution_role(self.sagemaker_session)

        self.s3_root_uri = resolve_value_from_config(
            direct_input=self.s3_root_uri,
            config_path=NOTEBOOK_JOB_S3_ROOT_URI,
            default_value=s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix,
            ),
            sagemaker_session=self.sagemaker_session,
        )

        self.s3_kms_key = resolve_value_from_config(
            direct_input=self.s3_kms_key,
            config_path=NOTEBOOK_JOB_S3_KMS_KEY_ID,
            sagemaker_session=self.sagemaker_session,
        )
        self.volume_kms_key = resolve_value_from_config(
            direct_input=self.volume_kms_key,
            config_path=NOTEBOOK_JOB_VOLUME_KMS_KEY_ID,
            sagemaker_session=self.sagemaker_session,
        )

        self.subnets = resolve_value_from_config(
            direct_input=self.subnets,
            config_path=NOTEBOOK_JOB_VPC_CONFIG_SUBNETS,
            sagemaker_session=self.sagemaker_session,
        )
        self.security_group_ids = resolve_value_from_config(
            direct_input=self.security_group_ids,
            config_path=NOTEBOOK_JOB_VPC_CONFIG_SECURITY_GROUP_IDS,
            sagemaker_session=self.sagemaker_session,
        )

        vpc_config = vpc_utils.to_dict(
            subnets=self.subnets, security_group_ids=self.security_group_ids
        )
        self.vpc_config = vpc_utils.sanitize(vpc_config)
        # pylint: disable=W0201

    def _prepare_tags(self):
        """Prepare tags for calling training job API.

        This function converts the custom tags into training API required format and also
        attach the system tags.
        """
        custom_tags = format_tags(self.tags) or []
        system_tags = [
            {"Key": "sagemaker:name", "Value": self.notebook_job_name},
            {"Key": "sagemaker:notebook-name", "Value": os.path.basename(self.input_notebook)},
            {"Key": "sagemaker:notebook-job-origin", "Value": "PIPELINE_STEP"},
            {"Key": "sagemaker:is-studio-archived", "Value": "false"},
        ]
        return custom_tags + system_tags

    def _prepare_env_variables(self):
        """Prepare environment variable for calling training job API.

        Attach the system environments used to pass execution context to the backend
        execution mechanism.
        """

        job_envs = self.environment_variables if self.environment_variables else {}
        system_envs = {
            "AWS_DEFAULT_REGION": self._region_from_session,
            "SM_JOB_DEF_VERSION": "1.0",
            "SM_ENV_NAME": "sagemaker-default-env",
            "SM_SKIP_EFS_SIMULATION": "true",
            "SM_EXECUTION_INPUT_PATH": "/opt/ml/input/data/"
            "sagemaker_headless_execution_pipelinestep",
            "SM_KERNEL_NAME": self.kernel_name,
            "SM_INPUT_NOTEBOOK_NAME": os.path.basename(self.input_notebook),
            "SM_OUTPUT_NOTEBOOK_NAME": f"{self._underlying_job_prefix}.ipynb",
        }
        if self.initialization_script:
            system_envs["SM_INIT_SCRIPT"] = os.path.basename(self.initialization_script)
        job_envs.update(system_envs)
        return job_envs

    @property
    def properties(self):
        """A `Properties` object representing the notebook job step output"""
        return self._properties

    def _compose_properties(self, step_name):
        """Create the properties object for step output"""
        root_prop = Properties(step_name=step_name, step=self)

        root_prop.__dict__["ComputingJobName"] = Properties(
            step_name=step_name, step=self, path="TrainingJobName"
        )

        root_prop.__dict__["ComputingJobStatus"] = Properties(
            step_name=step_name, step=self, path="TrainingJobStatus"
        )

        root_prop.__dict__["NotebookJobInputLocation"] = Properties(
            step_name=step_name, step=self, path="InputDataConfig[0].DataSource.S3DataSource.S3Uri"
        )

        root_prop.__dict__["NotebookJobOutputLocationPrefix"] = Properties(
            step_name=step_name, step=self, path="OutputDataConfig.S3OutputPath"
        )

        root_prop.__dict__["InputNotebookName"] = Properties(
            step_name=step_name, step=self, path="Environment['SM_INPUT_NOTEBOOK_NAME']"
        )

        root_prop.__dict__["OutputNotebookName"] = Properties(
            step_name=step_name, step=self, path="Environment['SM_OUTPUT_NOTEBOOK_NAME']"
        )

        return root_prop

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""

        request_dict = super().to_request()
        return request_dict

    @Step.depends_on.setter
    def depends_on(self, depends_on: List[Union[str, "Step", "StepCollection", StepOutput]]):
        """Set the list of  steps the current step explicitly depends on."""

        raise ValueError(
            "Cannot set depends_on for a NotebookJobStep. "
            "Use add_depends_on instead to extend the list."
        )

    @property
    def arguments(self) -> RequestType:
        """Generates the arguments dictionary that is used to create the job."""

        step_compilation_context = load_step_compilation_context()
        self.sagemaker_session = step_compilation_context.sagemaker_session
        self._region_from_session = self.sagemaker_session.boto_region_name

        self._resolve_defaults()
        # validate the inputs after resolving defaults
        self._validate_inputs()

        # generate training job name which is the key for the underlying training job
        self._underlying_job_prefix = self._get_job_name_prefix(self.notebook_job_name)

        pipeline_name = (
            step_compilation_context.pipeline_name
            if step_compilation_context
            else self._underlying_job_prefix
        )

        # step 1 - prepare for the staged input and upload to s3
        input_staged_folder_s3_uri = s3_path_join(
            self.s3_root_uri,
            pipeline_name,
            self.name,
            name_from_base("input"),
        )

        upload_list = [self.input_notebook]
        if self.initialization_script:
            upload_list.append(self.initialization_script)

        if self.additional_dependencies:
            upload_list = upload_list + self.additional_dependencies

        self._upload_job_files(
            s3_base_uri=input_staged_folder_s3_uri,
            paths_to_upload=upload_list,
            kms_key=self.s3_kms_key,
            sagemaker_session=self.sagemaker_session,
        )

        # step 2 - compose the job request
        request_dict = dict(
            TrainingJobName=self._underlying_job_prefix,
            RoleArn=self.role,
            RetryStrategy={"MaximumRetryAttempts": self.max_retry_attempts},
            StoppingCondition={
                "MaxRuntimeInSeconds": self.max_runtime_in_seconds,
            },
            EnableInterContainerTrafficEncryption=self.encrypt_inter_container_traffic,
        )

        # training algorithm config
        algorithm_spec = dict(
            TrainingImage=self.image_uri,
            TrainingInputMode="File",
            ContainerEntrypoint=self._scheduler_container_entry_point,
        )
        if self._scheduler_container_arguments:
            algorithm_spec["ContainerArguments"] = self._scheduler_container_arguments

        request_dict["AlgorithmSpecification"] = algorithm_spec

        # training input channel
        input_data_config = [
            {
                "ChannelName": "sagemaker_headless_execution_pipelinestep",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_staged_folder_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ]
        request_dict["InputDataConfig"] = input_data_config

        # training output
        if step_compilation_context:
            output_staged_folder_s3_uri = Join(
                "/",
                [
                    self.s3_root_uri,
                    pipeline_name,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    self.name,
                ],
            )
        else:
            output_staged_folder_s3_uri = s3_path_join(
                self.s3_root_uri,
                pipeline_name,
                self.name,
                "output",
            )

        output_config = {"S3OutputPath": output_staged_folder_s3_uri}
        if self.s3_kms_key is not None:
            output_config["KmsKeyId"] = self.s3_kms_key
        request_dict["OutputDataConfig"] = output_config

        # instance config
        resource_config = dict(
            VolumeSizeInGB=self.volume_size,
            InstanceCount=1,
            InstanceType=self.instance_type,
        )
        if self.volume_kms_key is not None:
            resource_config["VolumeKmsKeyId"] = self.volume_kms_key
        request_dict["ResourceConfig"] = resource_config

        # network
        if self.vpc_config:
            request_dict["VpcConfig"] = self.vpc_config

        # tags
        request_dict["Tags"] = self._prepare_tags()

        # env variables
        request_dict["Environment"] = self._prepare_env_variables()

        # notebook job parameter
        if self.parameters:
            request_dict["HyperParameters"] = self.parameters

        return request_dict

    def _get_job_name_prefix(self, notebook_job_name):
        """Get the underlying SageMaker job prefix from notebook job name."""

        # remove all special characters in the beginning of function name
        job_name = re.sub(r"^[^a-zA-Z0-9]+", "", notebook_job_name)
        # convert all remaining special characters to '-'
        job_name = re.sub(r"[^a-zA-Z0-9-]", "-", job_name)

        return job_name

    def _upload_job_files(self, s3_base_uri, paths_to_upload, kms_key, sagemaker_session):
        """Upload the notebook job files/folders to s3 staged folder."""

        # copy all inputs into a local temp folder in order to only upload once.
        with _tmpdir() as temp_input_folder:
            for path in paths_to_upload:

                if os.path.isfile(path):
                    shutil.copy2(path, temp_input_folder)
                elif os.path.isdir(path):
                    shutil.copytree(path, os.path.join(temp_input_folder, os.path.basename(path)))
                else:
                    # for safety to handle edge case e.g. file or dir gets deleted after validation
                    raise ValueError(f"Not supported file type: {path}")
            S3Uploader.upload(temp_input_folder, s3_base_uri, kms_key, sagemaker_session)
