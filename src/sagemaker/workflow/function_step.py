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
"""A proxy to the function returns of arbitrary type."""
from __future__ import absolute_import

import logging
from functools import wraps
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Callable,
    get_origin,
    get_type_hints,
    TYPE_CHECKING,
)

from sagemaker.workflow.functions import JsonGet, Join

from sagemaker.workflow.entities import (
    RequestType,
    PipelineVariable,
)

from sagemaker.workflow.properties import Properties
from sagemaker.workflow.retry import RetryPolicy
from sagemaker.workflow.steps import Step, ConfigurableRetryStep, StepTypeEnum
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.step_outputs import StepOutput, get_step
from sagemaker.workflow.utilities import trim_request_dict, load_step_compilation_context

from sagemaker.s3_utils import s3_path_join
from sagemaker.utils import unique_name_from_base_uuid4, format_tags, Tags

if TYPE_CHECKING:
    from sagemaker.remote_function.spark_config import SparkConfig
    from sagemaker.remote_function.job import _JobSettings

logger = logging.getLogger(__name__)


class _FunctionStep(ConfigurableRetryStep):
    """`_FunctionStep` for SageMaker Pipelines workflows.

    An internal step used with ``@step`` decorator to build TrainingJob arguments and
    build a request dict needed for pipeline definition.
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        retry_policies: Optional[List[RetryPolicy]] = None,
        depends_on: Optional[List[Union[Step, StepCollection, StepOutput]]] = None,
        func: Callable = None,
        func_args: tuple = (),
        func_kwargs: dict = None,
        **kwargs,
    ):
        """Constructs a _FunctionStep

        Args:
            name (str): The name of the `_FunctionStep`.
            display_name (str): The display name of the `_FunctionStep`.
            description (str): The description of the _FunctionStep.
            retry_policies (List[RetryPolicy]):  A list of retry policies.
            depends_on (List[Union[Step, StepCollection, StepOutput]]): A list of `Step`/
                `StepCollection`/`StepOutput` instances that this `_FunctionStep` depends on.
            func (Callable): The python function to run as a pipeline step.
            func_args (tuple): positional arguments to the python function.
            func_kwargs (dict): keyword arguments of the python function.
            **kwargs: Additional arguments to be passed to the `step` decorator.
        """
        from sagemaker.remote_function.core.pipeline_variables import (
            convert_pipeline_variables_to_pickleable,
        )
        from sagemaker.remote_function.core.serialization import CloudpickleSerializer
        from sagemaker.remote_function.core.stored_function import _SerializedData

        super(_FunctionStep, self).__init__(
            name, StepTypeEnum.TRAINING, display_name, description, depends_on, retry_policies
        )

        self._func = func
        self._func_args = func_args
        self._func_kwargs = func_kwargs if func_kwargs is not None else dict()

        self._step_kwargs = kwargs

        self.__job_settings = None

        # It's for internal usage to retrieve execution id from the properties.
        # However, we won't expose the properties of function step to customers.
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeTrainingJobResponse"
        )

        (
            self._converted_func_args,
            self._converted_func_kwargs,
        ) = convert_pipeline_variables_to_pickleable(
            func_args=self._func_args,
            func_kwargs=self._func_kwargs,
        )

        self._serialized_data = _SerializedData(
            func=CloudpickleSerializer.serialize(self._func),
            args=CloudpickleSerializer.serialize(
                (self._converted_func_args, self._converted_func_kwargs)
            ),
        )

    @property
    def func(self):
        """The python function to run as a pipeline step."""
        return self._func

    @property
    def func_args(self):
        """Positional arguments to the python function."""
        return self._func_args

    @property
    def func_kwargs(self):
        """Keyword arguments of the python function."""
        return self._func_kwargs

    # overriding the depends_on by the users is strictly prohibited.
    @Step.depends_on.setter
    def depends_on(self, depends_on: List[Union[str, "Step", "StepCollection", StepOutput]]):
        """Set the list of  steps the current step explicitly depends on."""

        raise ValueError(
            "Cannot set depends_on for a _FunctionStep. "
            "Use add_depends_on instead to extend the list."
        )

    @property
    def _job_settings(self) -> "_JobSettings":
        """Returns the job settings for the step."""

        from sagemaker.remote_function.job import _JobSettings

        context = load_step_compilation_context()

        if self.__job_settings and (
            not context or self.__job_settings.sagemaker_session is context.sagemaker_session
        ):
            return self.__job_settings

        self.__job_settings = _JobSettings(
            dependencies=self._step_kwargs.get("dependencies"),
            pre_execution_commands=self._step_kwargs.get("pre_execution_commands"),
            pre_execution_script=self._step_kwargs.get("pre_execution_script"),
            environment_variables=self._step_kwargs.get("environment_variables"),
            image_uri=self._step_kwargs.get("image_uri"),
            instance_count=self._step_kwargs.get("instance_count"),
            instance_type=self._step_kwargs.get("instance_type"),
            job_conda_env=self._step_kwargs.get("job_conda_env"),
            job_name_prefix=self._step_kwargs.get("job_name_prefix"),
            keep_alive_period_in_seconds=self._step_kwargs.get("keep_alive_period_in_seconds"),
            max_retry_attempts=self._step_kwargs.get("max_retry_attempts"),
            max_runtime_in_seconds=self._step_kwargs.get("max_runtime_in_seconds"),
            role=self._step_kwargs.get("role"),
            security_group_ids=self._step_kwargs.get("security_group_ids"),
            subnets=self._step_kwargs.get("subnets"),
            tags=self._step_kwargs.get("tags"),
            volume_kms_key=self._step_kwargs.get("volume_kms_key"),
            volume_size=self._step_kwargs.get("volume_size"),
            encrypt_inter_container_traffic=self._step_kwargs.get(
                "encrypt_inter_container_traffic"
            ),
            spark_config=self._step_kwargs.get("spark_config"),
            use_spot_instances=self._step_kwargs.get("use_spot_instances"),
            max_wait_time_in_seconds=self._step_kwargs.get("max_wait_time_in_seconds"),
            sagemaker_session=context.sagemaker_session,
        )

        return self.__job_settings

    @property
    def arguments(self) -> RequestType:
        """Generates the arguments dictionary that is used to call `create_training_job`."""
        from sagemaker.remote_function.job import _Job

        step_compilation_context = load_step_compilation_context()

        job_settings = self._job_settings
        base_job_name = _Job._get_job_name(job_settings, self.func)
        s3_base_uri = (
            s3_path_join(job_settings.s3_root_uri, step_compilation_context.pipeline_name)
            if step_compilation_context
            else job_settings.s3_root_uri
        )
        request_dict = _Job.compile(
            job_settings=job_settings,
            job_name=base_job_name,
            s3_base_uri=s3_base_uri,
            func=self.func,
            func_args=self.func_args,
            func_kwargs=self.func_kwargs,
            serialized_data=self._serialized_data,
        )
        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "TrainingJobName", step_compilation_context)
        return request_dict

    @property
    def properties(self):
        """Properties attribute is not supported for _FunctionStep."""
        raise NotImplementedError("Properties attribute is not supported for _FunctionStep.")

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        request_dict = super().to_request()
        return request_dict


class DelayedReturn(StepOutput):
    """A proxy to the function returns of arbitrary type.

    When a function decorated with ``@step`` is invoked, the return of that function
    is of type `DelayedReturn`. If the `DelayedReturn` object represents a Python
    collection, such as a tuple, list, or dict, you can reference the child items
    in the following ways:

      * ``a_member = a_delayed_return[2]``
      * ``a_member = a_delayed_return["a_key"]``
      * ``a_member = a_delayed_return[2]["a_key"]``

    """

    def __init__(self, function_step: _FunctionStep, reference_path: tuple = ()):
        """Initializes a `DelayedReturn` object.

        Args:
            function_step: A `sagemaker.workflow.step._FunctionStep` instance.
            reference_path: A tuple that represents the path to the child member.
        """
        self._reference_path = reference_path
        super().__init__(function_step)

    def __getitem__(self, key):
        """Returns a DelayedReturn object for the key"""
        return DelayedReturn(self._step, self._reference_path + (("__getitem__", key),))

    def __iter__(self):
        """Iterator is not supported for DelayedReturn object."""
        raise NotImplementedError("DelayedReturn object is not iterable.")

    def __deepcopy__(self, memodict=None):
        """Disable deepcopy of DelayedReturn as it is not supposed to be deepcopied."""
        logger.warning(
            "Disabling deepcopy of DelayedReturn as it is not supposed to be deepcopied."
        )
        return self

    @property
    def expr(self) -> RequestType:
        """Get the expression structure for workflow service calls."""
        return self._to_json_get().expr

    def _to_json_get(self) -> JsonGet:
        """Expression structure for workflow service calls using JsonGet resolution."""
        from sagemaker.remote_function.core.stored_function import (
            JSON_SERIALIZED_RESULT_KEY,
            JSON_RESULTS_FILE,
        )

        if not self._step.name:
            raise ValueError("Step name is not defined.")

        # Resolve json path --
        #   Deserializer will be able to resolve a JsonGet using path "Return[1]" to
        #   access value 10 from following serialized JSON:
        #       {
        #           "Return": [10, 20],
        #           "Exception": None
        #       }
        _resolved_reference_path = JSON_SERIALIZED_RESULT_KEY
        if self._reference_path:
            for path in self._reference_path:
                op, key = path
                if not op == "__getitem__":
                    raise RuntimeError(
                        f"Only __getitem__ is supported for DelayedReturn object. "
                        f"Got {op} instead."
                    )
                if isinstance(key, int):
                    _resolved_reference_path = _resolved_reference_path + f"[{key}]"
                else:
                    _resolved_reference_path = _resolved_reference_path + f"['{key}']"

        return JsonGet(
            s3_uri=Join(
                on="/",
                values=[
                    get_step(self)._properties.OutputDataConfig.S3OutputPath,
                    JSON_RESULTS_FILE,
                ],
            ),
            json_path=_resolved_reference_path,
            step=self._step,
        )

    @property
    def _referenced_steps(self) -> List[Step]:
        """Returns a step that generates the StepOutput"""
        return [self._step]


class _DelayedSequence(DelayedReturn):
    """A proxy to the function returns of tuple or list type."""

    def __getitem__(self, key):
        """Returns a DelayedReturn object for the key"""
        if not isinstance(key, int):
            raise TypeError(f"Expected an integer, got {key}")

        return DelayedReturn(self._step, self._reference_path + (("__getitem__", key),))


def _generate_delayed_return(function_step: _FunctionStep, type_hint, reference_path: tuple = ()):
    """Generates a DelayedReturn object based on the type hint.

    Args:
        function_step: A sagemaker.workflow.step._FunctionStep instance.
        type_hint: A type hint of the function return.
        reference_path: A tuple that represents the path to the child member.
    """

    if (
        type_hint is tuple
        or get_origin(type_hint) is tuple
        or type_hint is list
        or get_origin(type_hint) is list
    ):
        return _DelayedSequence(function_step, reference_path)

    return DelayedReturn(function_step, reference_path)


def step(
    _func=None,
    *,
    name: Optional[str] = None,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    retry_policies: Optional[List[RetryPolicy]] = None,
    dependencies: str = None,
    pre_execution_commands: List[str] = None,
    pre_execution_script: str = None,
    environment_variables: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    image_uri: Optional[Union[str, PipelineVariable]] = None,
    instance_count: Union[int, PipelineVariable] = 1,
    instance_type: Optional[Union[str, PipelineVariable]] = None,
    job_conda_env: Optional[Union[str, PipelineVariable]] = None,
    job_name_prefix: Optional[str] = None,
    keep_alive_period_in_seconds: Union[int, PipelineVariable] = 0,
    max_retry_attempts: Union[int, PipelineVariable] = 1,
    max_runtime_in_seconds: Union[int, PipelineVariable] = 24 * 60 * 60,
    role: str = None,
    security_group_ids: Optional[List[Union[str, PipelineVariable]]] = None,
    subnets: Optional[List[Union[str, PipelineVariable]]] = None,
    tags: Optional[Tags] = None,
    volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
    volume_size: Union[int, PipelineVariable] = 30,
    encrypt_inter_container_traffic: Optional[Union[bool, PipelineVariable]] = None,
    spark_config: "SparkConfig" = None,
    use_spot_instances: Union[bool, PipelineVariable] = False,
    max_wait_time_in_seconds: Optional[Union[int, PipelineVariable]] = None,
):
    """Decorator for converting a python function to a pipeline step.

    This decorator wraps the annotated code into a `DelayedReturn` object which can then be passed
    to a pipeline as a step. This creates a new pipeline that proceeds from the step of the
    `DelayedReturn` object.

    If the value for a parameter is not set, the decorator first looks up the value from the
    SageMaker configuration file. If no value is specified in the configuration file or no
    configuration file is found, the decorator selects the default as specified in the following
    list. For more information, see `Configuring and using defaults with the SageMaker Python SDK
    <https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk>`_.

    Args:
        _func: A Python function to run as a SageMaker pipeline step.

        name (str): Name of the pipeline step. Defaults to a generated name using function name
            and uuid4 identifier to avoid duplicates.

        display_name (str): The display name of the pipeline step. Defaults to the function name.

        description (str): The description of the pipeline step. Defaults to the function docstring.
          If there is no docstring, then it defaults to the function file path.

        retry_policies (List[RetryPolicy]): A list of retry policies configured for this step.
          Defaults to ``None``.

        dependencies (str): The path to a dependencies file. Defaults to ``None``.
          If ``dependencies`` is provided, the value must be one of the following:

          * A path to a conda environment.yml file. The following conditions apply:

            * If ``job_conda_env`` is set, then the conda environment is updated by installing
              dependencies from the yaml file and the function is invoked within that
              conda environment. For this to succeed, the specified conda environment must
              already exist in the image.
            * If the environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image, then the
              conda environment is updated by installing dependencies from the yaml file and the
              function is invoked within that conda environment. For this to succeed, the
              conda environment name must already be set with ``SAGEMAKER_JOB_CONDA_ENV``, and
              ``SAGEMAKER_JOB_CONDA_ENV`` must already exist in the image.
            * If none of the previous conditions are met, a new conda environment named
              ``sagemaker-runtime-env`` is created and the function annotated with the remote
              decorator is invoked in that conda environment.

          * A path to a requirements.txt file. The following conditions apply:

            * If ``job_conda_env`` is set in the remote decorator, dependencies are installed
              within that conda environment and the function annotated with the remote decorator
              is invoked in the same conda environment. For this to succeed, the specified
              conda environment must already exist in the image.
            * If an environment variable ``SAGEMAKER_JOB_CONDA_ENV`` is set in the image,
              dependencies are installed within that conda environment and the function annotated
              with the remote decorator is invoked in the environment. For this to succeed, the
              conda environment name must already be set in ``SAGEMAKER_JOB_CONDA_ENV``, and
              ``SAGEMAKER_JOB_CONDA_ENV`` must already exist in the image.
            * If none of the above conditions are met, conda is not used. Dependencies are
              installed at the system level without any virtual environment, and the function
              annotated with the remote decorator is invoked using the Python runtime available
              in the system path.

          * ``None``. SageMaker assumes that there are no dependencies to install while
            executing the remote annotated function in the training job.

        pre_execution_commands (List[str]): A list of commands to be executed prior to executing
          the pipeline step. Only one of ``pre_execution_commands`` or ``pre_execution_script``
          can be specified at the same time. Defaults to ``None``.

        pre_execution_script (str): A path to a script file to be executed prior to executing
          the pipeline step. Only one of ``pre_execution_commands`` or ``pre_execution_script``
          can be specified at the same time. Defaults to ``None``.

        environment_variables (dict[str, str] or dict[str, PipelineVariable]): Environment variables
          to be used inside the step. Defaults to ``None``.

        image_uri (str, PipelineVariable): The universal resource identifier (URI) location of a
          Docker image on Amazon Elastic Container Registry (ECR). Defaults to the following,
          based on where the SDK is running:

            * If you specify ``spark_config`` and want to run the step in a Spark
              application, the ``image_uri`` should be ``None``. A SageMaker Spark image
              is used for training, otherwise a ``ValueError`` is thrown.
            * If you use SageMaker Studio notebooks, the image used as the kernel image for the
              notebook is used.
            * Otherwise, it is resolved to a base python image with the same python version
              as the environment running the local code.

          If no compatible image is found, a ``ValueError`` is thrown.

        instance_count (int, PipelineVariable): The number of instances to use. Defaults to 1.
          Note that pipeline steps do not support values of ``instance_count`` greater than 1
          for non-Spark jobs.

        instance_type (str, PipelineVariable): The Amazon Elastic Compute Cloud (EC2) instance
          type to use to run the SageMaker job. For example, ``ml.c4.xlarge``. If not provided,
          a ``ValueError`` is thrown.

        job_conda_env (str, PipelineVariable): The name of the conda environment to activate during
          the job's runtime. Defaults to ``None``.

        job_name_prefix (str): The prefix used to create the underlying SageMaker job.

        keep_alive_period_in_seconds (int, PipelineVariable): The duration in seconds to retain
          and reuse provisioned infrastructure after the completion of a training job. This
          infrastructure is also known as SageMaker
          managed warm pools. The use of warm pools reduces the latency time spent to
          provision new resources. The default value for ``keep_alive_period_in_seconds`` is 0.
          Note that additional charges associated with warm pools may apply. Using this parameter
          also activates a new persistent cache feature which reduces job start up
          latency more than if you were to use SageMaker managed warm pools alone. This occurs
          because the package source downloaded in the previous runs are cached.

        max_retry_attempts (int, PipelineVariable): The max number of times the job is retried after
          an ``InternalServerFailure`` error from the SageMaker service. Defaults to 1.

        max_runtime_in_seconds (int, PipelineVariable): The upper limit in seconds to be used for
          training. After this specified amount of time, SageMaker terminates the job regardless
          of its current status. Defaults to 1 day or (86400 seconds).

        role (str): The IAM role (either name or full ARN) used to run your SageMaker training
          job. Defaults to one of the following:

          * The SageMaker default IAM role if the SDK is running in SageMaker Notebooks or
            SageMaker Studio Notebooks.
          * Otherwise, a ``ValueError`` is thrown.

        security_group_ids (List[str, PipelineVariable]): A list of security group IDs.
          Defaults to ``None`` and the training job is created without a VPC config.

        subnets (List[str, PipelineVariable]): A list of subnet IDs. Defaults to ``None``
          and the job is created without a VPC config.

        tags (Optional[Tags]): Tags attached to the job. Defaults to ``None``
          and the training job is created without tags.

        volume_kms_key (str, PipelineVariable): An Amazon Key Management Service (KMS) key used to
          encrypt an Amazon Elastic Block Storage (EBS) volume attached to the training instance.
          Defaults to ``None``.

        volume_size (int, PipelineVariable): The size in GB of the storage volume that stores input
          and output data during training. Defaults to ``30``.

        encrypt_inter_container_traffic (bool, PipelineVariable): A flag that specifies whether
          traffic between training containers is encrypted for the training job.
          Defaults to ``False``.

        spark_config (SparkConfig): Configurations of the Spark application that runs on
          the Spark image. If ``spark_config`` is specified, a SageMaker Spark image URI
          is used for training. Note that ``image_uri`` can not be specified at the
          same time, otherwise a ``ValueError`` is thrown. Defaults to ``None``.

        use_spot_instances (bool, PipelineVariable): Specifies whether to use SageMaker
          Managed Spot instances for training. If enabled, then ``max_wait_time_in_seconds``
          argument should also be set. Defaults to ``False``.

        max_wait_time_in_seconds (int, PipelineVariable): Timeout in seconds waiting for
          the spot training job. After this amount of time, Amazon SageMaker stops waiting
          for the managed spot training job to complete. Defaults to ``None``.
    """

    def _step(func):

        if dependencies == "auto_capture":
            raise ValueError("Auto Capture of dependencies is not supported for pipeline steps.")

        # avoid circular import
        from sagemaker.remote_function.client import RemoteExecutor

        @wraps(func)
        def wrapper(*args, **kwargs):

            # TODO: Move _validate_submit_args function out of RemoteExecutor class
            RemoteExecutor._validate_submit_args(func, *args, **kwargs)

            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, (Join, JsonGet)):
                    raise ValueError(f"{type(arg)} is not supported for function arguments.")

            depends_on = {}
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, DelayedReturn):
                    depends_on[id(arg._step)] = arg._step

            # setup default values for name, display_name and description if not provided

            _name = unique_name_from_base_uuid4(func.__name__) if not name else name
            _display_name = (
                f"{func.__module__}.{func.__name__}" if not display_name else display_name
            )
            _description = description
            if not _description:
                _description = func.__doc__ if func.__doc__ else func.__code__.co_filename

            function_step = _FunctionStep(
                name=_name,
                display_name=_display_name,
                description=_description,
                retry_policies=retry_policies,
                func=func,
                func_args=args,
                func_kwargs=kwargs,
                depends_on=list(depends_on.values()),
                dependencies=dependencies,
                pre_execution_commands=pre_execution_commands,
                pre_execution_script=pre_execution_script,
                environment_variables=environment_variables,
                image_uri=image_uri,
                instance_count=instance_count,
                instance_type=instance_type,
                job_conda_env=job_conda_env,
                job_name_prefix=job_name_prefix,
                keep_alive_period_in_seconds=keep_alive_period_in_seconds,
                max_retry_attempts=max_retry_attempts,
                max_runtime_in_seconds=max_runtime_in_seconds,
                role=role,
                security_group_ids=security_group_ids,
                subnets=subnets,
                tags=format_tags(tags),
                volume_kms_key=volume_kms_key,
                volume_size=volume_size,
                encrypt_inter_container_traffic=encrypt_inter_container_traffic,
                spark_config=spark_config,
                use_spot_instances=use_spot_instances,
                max_wait_time_in_seconds=max_wait_time_in_seconds,
            )

            return _generate_delayed_return(
                function_step, type_hint=get_type_hints(func).get("return")
            )

        return wrapper

    if _func is None:
        return _step
    return _step(_func)
