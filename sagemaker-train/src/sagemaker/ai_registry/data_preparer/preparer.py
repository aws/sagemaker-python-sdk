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
"""DataPreparer: submit and track SageMaker data preparation jobs.

Covers SyntheticDataGeneration, DataQualityEvaluation, and DataTransformation.

The bound-vs-unbound dispatch uses a descriptor class so both
`DataPreparer.generate(dataset, ...)` (classmethod-style) and
`preparer.generate(dataset, ...)` (instance-method) work transparently.

Feature.DATA_PREPARATION = 22 is defined in sagemaker-core's Feature enum.
"""

from __future__ import absolute_import

import functools
import json
import logging
from typing import Any, ClassVar, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from sagemaker.ai_registry.data_preparer._job_resource import Job

from pydantic import BaseModel, ConfigDict, PrivateAttr

from sagemaker.ai_registry.data_preparer._api_shape import DqShape
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.defaults import DataPreparerDefaults
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerConfigurationError,
    DataPreparerJobError,
    DataPreparerJobFailedError,
    DataPreparerJobStoppedError,
    DataPreparerJobTimeoutError,
)
from sagemaker.core.jumpstart.hub.utils import get_info_from_hub_resource_arn
from sagemaker.core.shapes import Tag
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.core.workflow.pipeline_capture import capture_create_job_request
from sagemaker.core.workflow.pipeline_context import PipelineSession, runnable_by_pipeline

logger = logging.getLogger(__name__)

# Largest output object the empty-output check will read. Anything bigger is
# treated as non-empty so completed jobs never trigger a large download.
_EMPTY_CHECK_MAX_BYTES = 64 * 1024


def _parse_dataset_arn(arn: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Split a hub-content DataSet ARN into ``(content_name, hub_name, version)``.

    Delegates to sagemaker-core's shared hub-ARN parser rather than splitting on
    ``/`` here, so the accepted shape stays in one place:
    ``arn:{partition}:sagemaker:{region}:{account}:hub-content/{hub}/{type}/{name}/{version}``.

    Anything that is not a well-formed hub-content ARN yields a best-effort name
    with ``(None, None)`` for hub and version, leaving the caller to fall back to
    the derived AI Registry hub and the latest version rather than mis-reading
    some other segment as a hub name. Trailing segments beyond the version are
    treated the same way: the parser's regex would otherwise absorb them into the
    version.
    """
    info = get_info_from_hub_resource_arn(arn)
    if info is not None and info.hub_content_name:
        version = info.hub_content_version
        if not version or "/" in version:
            return info.hub_content_name, None, None
        return info.hub_content_name, info.hub_name, version

    parts = arn.split("/")
    return (parts[-2] if len(parts) >= 2 else arn), None, None


def _get_job_resource():
    """Get the Job resource class.

    Uses the local _job_resource.Job, which adds the DataPreparer user experience
    on top of the Job APIs: Rich progress panels, CloudWatch log streaming,
    S3-based sample counting and the labeling summary. The generated
    sagemaker.core.resources.Job covers the API fields but not this behavior.
    """
    from sagemaker.ai_registry.data_preparer._job_resource import Job

    return Job


class _DualCallable:
    """Descriptor: makes a method work as both instance method and classmethod.

    Satisfies design §2.2.6 / Req 1.4, 1.5 — both call shapes work identically:

    * Instance-method: ``preparer.generate(dataset, job_name=..., ...)``
    * Classmethod:     ``DataPreparer.generate(dataset, role=..., ...)``

    When called from the class, constructs a transient
    ``DataPreparer(sagemaker_session=..., role=..., base_job_name=...)`` and
    delegates to the wrapped instance method. Both paths therefore flow
    through the same ``_run`` / intelligent-defaults resolution pipeline.

    Implementation note: Pydantic v2 does not treat descriptors as fields, so
    this class attribute is safely ignored by the BaseModel field machinery.
    """

    # Kwargs that belong on the DataPreparer constructor (instance state);
    # anything else is a per-call kwarg on the instance method.
    _INSTANCE_KWARGS = ("sagemaker_session", "role")

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self._func = func

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        if instance is None:
            # Class-level access: build a transient instance + delegate.
            func = self._func

            @functools.wraps(func)
            def classmethod_shortcut(
                dataset,
                *,
                sagemaker_session=None,
                role=None,
                base_job_name=None,
                job_name=None,
                wait=True,
                logs=False,
            ):
                tmp = owner(
                    sagemaker_session=sagemaker_session,
                    role=role,
                    base_job_name=base_job_name,
                )
                return func(
                    tmp,
                    dataset,
                    job_name=job_name,
                    base_job_name=None,  # already pinned on the instance
                    wait=wait,
                    logs=logs,
                )

            return classmethod_shortcut

        # Instance-level access: normal bound method.
        return self._func.__get__(instance, owner)


def _parse_job_arn(job_arn: str) -> Dict[str, str]:
    """Parse a Data Preparation job ARN into its components.

    Expected format:
        arn:aws:sagemaker:{region}:{account}:job/{category}/{job_name}

    Returns:
        Dict with keys: region, account, category, job_name.

    Raises:
        ValueError: If the ARN does not match the expected format.
    """
    parts = job_arn.split(":")
    if len(parts) < 6 or not parts[5].startswith("job/"):
        raise ValueError(
            f"Invalid job ARN format: '{job_arn}'. "
            f"Expected 'arn:aws:sagemaker:{{region}}:{{account}}:job/{{category}}/{{job_name}}'."
        )
    resource = parts[5]  # "job/{category}/{job_name}"
    resource_parts = resource.split("/", 2)
    if len(resource_parts) != 3:
        raise ValueError(
            f"Invalid job ARN resource segment: '{resource}'. "
            f"Expected 'job/{{category}}/{{job_name}}'."
        )
    return {
        "region": parts[3],
        "account": parts[4],
        "category": resource_parts[1],
        "job_name": resource_parts[2],
    }


class DataPreparer(BaseModel):
    """Entrypoint for SyntheticDataGeneration, DataQualityEvaluation, and DataTransformation.

    Mirrors ModelTrainer: builds the request, calls Job.create(), job.wait(logs=logs),
    and rehydrates a DataSet from the completed job.

    Three primitives are available:

    * :attr:`generate` — submit a SyntheticDataGeneration job to produce labeled data
      from a task specification.
    * :attr:`evaluate` — submit a DataQualityEvaluation job to score each record in an
      existing dataset and produce per-record quality annotations.
    * :attr:`transform` — submit a DataTransformation job to run operators (dedup, split,
      merge, schema conversion, quality filters, custom code) over an existing dataset.

    Each primitive accepts both an instance-method and a classmethod call style (see
    :attr:`generate` for examples). Both resolve intelligent defaults through the same
    ``_run`` pipeline.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        ignored_types=(_DualCallable,),
    )

    sagemaker_session: Optional[Any] = None
    role: Optional[str] = None
    base_job_name: Optional[str] = None

    _pipeline_caller_name: ClassVar[str] = "create_job"

    _latest_job: Optional[Any] = PrivateAttr(default=None)

    @classmethod
    def attach(
        cls,
        job_arn: str,
        *,
        wait: bool = True,
        logs: bool = False,
        sagemaker_session: Optional[Any] = None,
    ):
        """Re-attach to an existing Data Preparation job by ARN.

        Use this to reconnect to a long-running job after a kernel restart,
        lost connection, or when resuming work in a new process. The job ARN
        is printed at submission time and is available as ``job.job_arn`` on
        the handle returned by any primitive called with ``wait=False``.

        Args:
            job_arn: The full ARN of the job to re-attach to.
                Format: ``arn:aws:sagemaker:{region}:{account}:job/{category}/{job_name}``
            wait: If True (default), block until the job reaches a terminal
                state and return the result (DataSet or Job). If False, return
                the Job handle immediately for manual polling.
            logs: If True, stream CloudWatch logs while waiting. Only applies
                when ``wait=True``.
            sagemaker_session: Optional SageMaker session. If not provided,
                a default session is created.

        Returns:
            The result, depending on ``wait`` and the job state:

            - If ``wait=False``: the ``Job`` handle (call ``.wait()`` or
              ``.refresh()`` to poll).
            - If ``wait=True`` and the job completed: a ``DataSet`` (for SDG,
              DQ) or ``List[DataSet]`` (for DataTransformation).
            - If the job already completed before attach: returns the result
              immediately.

        Raises:
            ValueError: If the ARN format is invalid, or its category is not one
                DataPreparer supports.
            DataPreparerJobFailedError: If the job failed.
            DataPreparerJobStoppedError: If the job was stopped.
            DataPreparerJobTimeoutError: If waiting exceeds the timeout.

        Example::

            # Cell 1 — submit and note the ARN
            job = DataPreparer.generate(dataset, wait=False)
            print(job.job_arn)
            # "arn:aws:sagemaker:us-west-2:123456789012:job/SyntheticDataGeneration/my-ds-sdg-2026-..."

            # --- kernel restart ---

            # Cell 2 — re-attach and wait for results
            dataset = DataPreparer.attach(
                "arn:aws:sagemaker:us-west-2:123456789012:job/SyntheticDataGeneration/my-ds-sdg-2026-..."
            )
        """
        parsed = _parse_job_arn(job_arn)
        job_name = parsed["job_name"]
        job_category = parsed["category"]
        supported = [c.value for c in JobCategory]
        if job_category not in supported:
            raise ValueError(
                f"Unsupported job category '{job_category}' in ARN '{job_arn}'. "
                f"DataPreparer.attach supports: {', '.join(supported)}."
            )

        session = sagemaker_session or DataPreparerDefaults.get_sagemaker_session(None)

        Job = _get_job_resource()
        job = Job.get(job_name=job_name, job_category=job_category, session=session)

        logger.info(
            "Re-attached to %s job '%s' (status: %s).",
            job_category,
            job_name,
            job.job_status,
        )

        if not wait:
            return job

        # If already terminal, skip waiting
        if job.job_status in ("Completed", "Failed", "Stopped"):
            logger.info("Job '%s' already in terminal state: %s.", job_name, job.job_status)
        else:
            try:
                job.wait(logs=logs)
            except Exception as e:
                if "Timeout" in type(e).__name__ or "WaiterError" in type(e).__name__:
                    raise DataPreparerJobTimeoutError(
                        job_arn=job_arn,
                        elapsed_seconds=0,
                        job_name=job_name,
                    ) from e
                raise
            job.refresh()

        # Dispatch on terminal status
        status = job.job_status
        if status == "Completed":
            category_enum = JobCategory(job_category)
            # Rehydrate datasets from completed job
            preparer = cls(sagemaker_session=session)
            datasets = preparer._rehydrate_datasets(job, category_enum)
            for ds in datasets:
                ds.job_arn = job_arn
            preparer._warn_if_empty_output(datasets, category_enum, job)
            if category_enum == JobCategory.DATA_TRANSFORMATION:
                return datasets
            [ds] = datasets
            preparer._preview_output(ds, category_enum, job)
            return ds
        elif status == "Failed":
            raise DataPreparerJobFailedError(
                job_arn=job_arn,
                failure_reason=getattr(job, "failure_reason", None),
                secondary_status=getattr(job, "secondary_status", None),
                job_name=job_name,
                category=JobCategory(job_category),
            )
        elif status == "Stopped":
            raise DataPreparerJobStoppedError(
                job_arn=job_arn,
                job_name=job_name,
                category=JobCategory(job_category),
            )

    @classmethod
    def stop(
        cls,
        job_or_arn: Union[str, "Job"],
        *,
        wait: bool = True,
        logs: bool = False,
        sagemaker_session: Optional[Any] = None,
    ):
        """Stop a running Data Preparation job.

        Use this to cancel a job that was submitted asynchronously (with
        ``wait=False``) and is no longer needed — e.g., a misconfigured
        spec, a wrong input dataset, or a run that has grown too expensive.
        Only jobs still in progress can be stopped; if the job has already
        reached a terminal state (``Completed`` / ``Failed`` / ``Stopped``),
        this is a no-op.

        Args:
            job_or_arn: Either the ``Job`` handle returned by a primitive
                submitted with ``wait=False`` (in-session use), or the full
                job ARN as a string (cross-session use, e.g. after a kernel
                restart). ARN format:
                ``arn:aws:sagemaker:{region}:{account}:job/{category}/{job_name}``.
            wait: If ``True`` (default), block until the job reaches a
                terminal state (typically ``Stopped``). If ``False``, return
                immediately after the stop request is accepted; the
                transition through ``Stopping`` → ``Stopped`` continues
                asynchronously on the service.
            logs: If ``True``, stream CloudWatch logs while waiting. Only
                applies when ``wait=True``.
            sagemaker_session: Optional SageMaker session. Only used when
                ``job_or_arn`` is an ARN; ignored when a live ``Job`` handle
                is passed (the handle carries its own session).

        Returns:
            Job: The ``Job`` handle, refreshed with the latest state. With
            ``wait=True``, always in a terminal state.

        Raises:
            TypeError: If ``job_or_arn`` is neither a ``Job`` handle nor a
                str ARN.
            ValueError: If the ARN format is invalid.
            DataPreparerJobTimeoutError: If ``wait=True`` and the job does
                not reach a terminal state before the waiter expires.

        Example::

            # In-session: pass the handle returned by wait=False.
            job = DataPreparer.generate(dataset, wait=False)
            DataPreparer.stop(job)

            # Cross-session (e.g., after a kernel restart): pass the ARN.
            DataPreparer.stop(
                "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-job",
                wait=False,
            )
        """
        Job = _get_job_resource()

        if isinstance(job_or_arn, Job):
            job = job_or_arn
            job.refresh()
        elif isinstance(job_or_arn, str):
            parsed = _parse_job_arn(job_or_arn)
            session = sagemaker_session or DataPreparerDefaults.get_sagemaker_session(None)
            job = Job.get(
                job_name=parsed["job_name"],
                job_category=parsed["category"],
                session=session,
            )
        else:
            raise TypeError(
                "stop() expects a Job handle or a job ARN string; got "
                f"{type(job_or_arn).__name__}"
            )

        if job.job_status in ("Completed", "Failed", "Stopped"):
            logger.info(
                "Job '%s' is already in terminal state (%s); nothing to stop.",
                job.job_name,
                job.job_status,
            )
            return job

        job.stop()

        if not wait:
            return job

        try:
            job.wait(logs=logs)
        except Exception as e:
            if "Timeout" in type(e).__name__ or "WaiterError" in type(e).__name__:
                raise DataPreparerJobTimeoutError(
                    job_arn=job.job_arn,
                    elapsed_seconds=0,
                    job_name=job.job_name,
                ) from e
            raise
        job.refresh()
        return job

    @_telemetry_emitter(feature=Feature.DATA_PREPARATION, func_name="DataPreparer.generate")
    def _generate_impl(
        self,
        dataset: DatasetJobConfig,
        *,
        job_name: Optional[str] = None,
        base_job_name: Optional[str] = None,
        wait: bool = True,
        logs: bool = False,
        tags: Optional[List[Tag]] = None,
    ):
        """Submit a SyntheticDataGeneration job.

        Generates a new labeled dataset from a task specification and an optional
        seed dataset. Two equivalent call shapes::

            # instance
            preparer.generate(dataset, wait=..., logs=...)

            # class — also accepts role / sagemaker_session / base_job_name
            DataPreparer.generate(dataset, role=..., sagemaker_session=...)

        Both resolve intelligent defaults (``SageMakerConfig`` → well-known
        fallbacks → schema defaults) through the same ``_run`` pipeline.

        The job reads a task specification from ``GenerationConfig.SpecS3Uri`` and
        produces ``NumSamples`` new rows. Use
        ``DatasetJobConfig.print_schema(JobCategory.SYNTHETIC_DATA_GENERATION)``
        to inspect all fields the service currently accepts.

        Args:
            dataset (DatasetJobConfig): Job configuration. For
                SyntheticDataGeneration this carries ``generation_config``
                (including ``SpecS3Uri``, ``NumSamples``, and optional model and
                sampling parameters) and ``output_data_config``
                (``OutputS3Uri`` or ``S3OutputPath``).
            job_name (Optional[str]): Exact name for the job. Must be unique
                within the account and region; jobs are immutable, so a failed
                job cannot be retried under the same name. Mutually exclusive
                with ``base_job_name``.
            base_job_name (Optional[str]): Prefix to which a timestamp is
                appended to derive the job name.
            wait (bool): Whether to block until the job reaches a terminal
                state. Defaults to ``True``.
            logs (bool): Whether to stream job logs while waiting. Requires
                ``wait=True``. Defaults to ``False``.
            tags (Optional[List[Tag]]): Tags to apply to the job.

        Returns:
            DataSet: With ``wait=True``, the output dataset containing the
                generated records. With ``wait=False``, the ``Job`` handle
                instead, for the caller to poll or ``wait()`` on later (or
                persist ``job.job_name`` and re-attach via
                :meth:`attach`).

        Raises:
            DataPreparerConfigurationError: If ``dataset.job_category`` is set to
                a category other than ``SYNTHETIC_DATA_GENERATION``.
            DataPreparerJobFailedError: If the job terminates as ``Failed``
                while ``wait=True``.
            DataPreparerJobStoppedError: If the job terminates as ``Stopped``
                while ``wait=True``.
            DataPreparerJobTimeoutError: If the job does not reach a terminal
                state before the waiter expires.
        """
        return self._run(
            dataset,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            job_name=job_name,
            base_job_name=base_job_name,
            wait=wait,
            logs=logs,
            tags=tags,
        )

    @_telemetry_emitter(feature=Feature.DATA_PREPARATION, func_name="DataPreparer.evaluate")
    def _evaluate_impl(
        self,
        dataset: DatasetJobConfig,
        *,
        job_name: Optional[str] = None,
        base_job_name: Optional[str] = None,
        wait: bool = True,
        logs: bool = False,
        tags: Optional[List[Tag]] = None,
    ):
        """Submit a DataQualityEvaluation job.

        Scores each record in an existing dataset against quality metrics and
        produces per-record quality annotations. Two equivalent call shapes::

            # instance
            preparer.evaluate(dataset, wait=..., logs=...)

            # class — also accepts role / sagemaker_session / base_job_name
            DataPreparer.evaluate(dataset, role=..., sagemaker_session=...)

        Both resolve intelligent defaults (``SageMakerConfig`` → well-known
        fallbacks → schema defaults) through the same ``_run`` pipeline.

        The input dataset is referenced by ARN in ``input_data_config`` (v2+)
        or ``DatasetArn`` (v1). Quality metrics are specified in ``Metrics``
        (v1/v2) or ``EvaluationConfig.Operations`` (v3). Use
        ``DatasetJobConfig.print_schema(JobCategory.DATA_QUALITY_EVALUATION)``
        to inspect the fields and schema version the service currently accepts.

        Args:
            dataset (DatasetJobConfig): Job configuration. For
                DataQualityEvaluation this carries the input dataset reference
                (``input_data_config`` for v2+ or ``DatasetArn`` for v1), the
                metrics or evaluation operations to run, and
                ``output_data_config`` specifying where to write the per-record
                quality annotations.
            job_name (Optional[str]): Exact name for the job. Must be unique
                within the account and region; jobs are immutable, so a failed
                job cannot be retried under the same name. Mutually exclusive
                with ``base_job_name``.
            base_job_name (Optional[str]): Prefix to which a timestamp is
                appended to derive the job name.
            wait (bool): Whether to block until the job reaches a terminal
                state. Defaults to ``True``.
            logs (bool): Whether to stream job logs while waiting. Requires
                ``wait=True``. Defaults to ``False``.
            tags (Optional[List[Tag]]): Tags to apply to the job.

        Returns:
            DataSet: With ``wait=True``, the input dataset rehydrated from the
                completed job. The per-record quality annotations are written
                alongside the dataset's S3 source by the job. With
                ``wait=False``, the ``Job`` handle instead, for the caller to
                poll or ``wait()`` on later (or persist ``job.job_name`` and
                re-attach via :meth:`attach`).

        Raises:
            DataPreparerConfigurationError: If ``dataset.job_category`` is set to
                a category other than ``DATA_QUALITY_EVALUATION``.
            DataPreparerJobFailedError: If the job terminates as ``Failed``
                while ``wait=True``.
            DataPreparerJobStoppedError: If the job terminates as ``Stopped``
                while ``wait=True``.
            DataPreparerJobTimeoutError: If the job does not reach a terminal
                state before the waiter expires.
        """
        return self._run(
            dataset,
            JobCategory.DATA_QUALITY_EVALUATION,
            job_name=job_name,
            base_job_name=base_job_name,
            wait=wait,
            logs=logs,
            tags=tags,
        )

    @_telemetry_emitter(feature=Feature.DATA_PREPARATION, func_name="DataPreparer.transform")
    def _transform_impl(
        self,
        dataset: DatasetJobConfig,
        *,
        job_name: Optional[str] = None,
        base_job_name: Optional[str] = None,
        wait: bool = True,
        logs: bool = False,
        tags: Optional[List[Tag]] = None,
    ):
        """Submit a DataTransformation job.

        Applies a pipeline of operators (dedup, split, merge, schema conversion,
        quality filters, custom code) to an existing dataset. Two equivalent call
        shapes, as for :attr:`generate`::

            # instance
            preparer.transform(dataset, wait=..., logs=...)

            # class — also accepts role / sagemaker_session / base_job_name
            DataPreparer.transform(dataset, role=..., sagemaker_session=...)

        Both resolve intelligent defaults (``SageMakerConfig`` → well-known
        fallbacks → schema defaults) through the same ``_run`` pipeline.

        The input dataset is referenced by ARN in ``input_data_config``, and the
        operators to apply are listed in ``transformation_config``. Each entry is
        an ``Operator`` name plus its ``Config``; both are passed through to the
        service as declared in the Hub schema, so the SDK does not validate
        operator configs field-by-field. Use
        ``DatasetJobConfig.print_schema(JobCategory.DATA_TRANSFORMATION)`` to
        inspect the operators and fields the service currently accepts.

        .. note::

            ``Operations`` accepts exactly **one** entry per job. The published
            schema advertises ``maxItems=10``, but the service enforces a limit of
            1 and rejects jobs with more operations. For multi-step pipelines,
            chain jobs: submit one transformation per job and use the output
            dataset ARN as the next job's input.

        Args:
            dataset (DatasetJobConfig): Job configuration. For
                DataTransformation this carries ``input_data_config`` (a list of
                ``{"DatasetArn": ...}`` entries) and ``transformation_config``
                (an ``Operations`` list).
            job_name (Optional[str]): Exact name for the job. Must be unique
                within the account and region; jobs are immutable, so a failed
                job cannot be retried under the same name. Mutually exclusive
                with ``base_job_name``.
            base_job_name (Optional[str]): Prefix to which a timestamp is
                appended to derive the job name.
            wait (bool): Whether to block until the job reaches a terminal
                state. Defaults to ``True``.
            logs (bool): Whether to stream job logs while waiting. Requires
                ``wait=True``. Defaults to ``False``.
            tags (Optional[List[Tag]]): Tags to apply to the job.

        Returns:
            List[DataSet]: With ``wait=True``, the output datasets — one per
                output, since DataTransformation can emit more than one per job
                (``Split`` yields one per output partition, while single-output
                operators such as ``FuzzyDedup`` yield a list of length 1).
                With ``wait=False``, the ``Job`` handle instead, for the caller
                to poll or ``wait()`` on later (or persist ``job.job_name`` and
                re-attach via ``Job.get``).

        Raises:
            DataPreparerConfigurationError: If ``dataset.job_category`` is set to
                a category other than ``DATA_TRANSFORMATION``.
            DataPreparerJobFailedError: If the job terminates as ``Failed``
                while ``wait=True``.
            DataPreparerJobStoppedError: If the job terminates as ``Stopped``
                while ``wait=True``.
            DataPreparerJobTimeoutError: If the job does not reach a terminal
                state before the waiter expires.
        """
        return self._run(
            dataset,
            JobCategory.DATA_TRANSFORMATION,
            job_name=job_name,
            base_job_name=base_job_name,
            wait=wait,
            logs=logs,
            tags=tags,
        )

    # Public dual-callable methods. Accept the same kwargs as
    # _{generate,evaluate,transform}_impl when called on an instance.
    # When called on the class, also accept ``sagemaker_session=`` / ``role=`` /
    # ``base_job_name=`` for constructing the transient instance.
    generate = _DualCallable(_generate_impl)
    # Docstring lives on _generate_impl: _DualCallable.__get__ returns a
    # functools.wraps-decorated function at class level, so autodoc documents
    # this as a method and reads __doc__ (the impl's) rather than this
    # attribute docstring.

    evaluate = _DualCallable(_evaluate_impl)
    # Docstring lives on _evaluate_impl — same pattern as generate above.

    transform = _DualCallable(_transform_impl)
    # Docstring lives on _transform_impl: _DualCallable.__get__ returns a
    # functools.wraps-decorated function at class level, so autodoc documents
    # this as a method and reads __doc__ (the impl's) rather than this
    # attribute docstring.

    @runnable_by_pipeline
    def _run(
        self,
        dataset: DatasetJobConfig,
        category: JobCategory,
        job_name: Optional[str] = None,
        base_job_name: Optional[str] = None,
        wait: bool = True,
        logs: bool = True,
        tags: Optional[List[Tag]] = None,
    ):
        """Job-lifecycle orchestration."""
        # Category inference / mismatch check
        if dataset.job_category is None:
            dataset.job_category = category
        elif dataset.job_category != category:
            raise DataPreparerConfigurationError(
                f"dataset.job_category={dataset.job_category.value} does not match "
                f"the method-implied category={category.value}. "
                f"Use DataPreparer.generate() for SDG, DataPreparer.evaluate() for DQ "
                f"and DataPreparer.transform() for DT."
            )

        # Resolve session
        self.sagemaker_session = DataPreparerDefaults.get_sagemaker_session(self.sagemaker_session)

        # Resolve envelope-level RoleArn through SageMakerConfig + well-known
        # fallback (Req 4.4/4.5). The per-property schema walk inside
        # DatasetJobConfig._resolve_and_validate does not cover RoleArn because
        # RoleArn lives on the CreateJob envelope, not inside the JobConfigDocument.
        if self.role is None:
            self.role = DataPreparerDefaults.get_role(self.sagemaker_session)

        # Resolve and validate the config
        dataset._resolve_and_validate(
            sagemaker_session=self.sagemaker_session,
        )

        if isinstance(self.sagemaker_session, PipelineSession):
            resolved_name = DataPreparerDefaults.resolve_job_name(
                job_name,
                base_job_name or self.base_job_name,
                category,
                dict(dataset.model_extra or {}),
            )
            return capture_create_job_request(
                self.sagemaker_session,
                job_name=resolved_name,
                role_arn=self.role,
                job_category=category.value,
                schema_version=dataset.resolved_schema_version,
                job_config=dataset.job_config_dict(),
                customer_details=DataPreparerDefaults.get_customer_details(self.sagemaker_session),
                tags=[Tag(**t) if isinstance(t, dict) else t for t in tags] if tags else None,
            )

        # Create the job
        job = self._create_job(dataset, category, job_name, base_job_name, tags=tags)

        if wait and logs:
            logger.info(
                "Tailing logs for job %s. It may take a few minutes for "
                "logs to appear while the job initializes.",
                job.job_name,
            )

        if wait:
            try:
                job.wait(logs=logs)
            except Exception as e:
                if "Timeout" in type(e).__name__:
                    raise DataPreparerJobTimeoutError(
                        job_arn=getattr(job, "job_arn", "unknown"),
                        elapsed_seconds=0,
                        job_name=getattr(job, "job_name", None),
                    ) from e
                # FailedStatusError from sagemaker-core — extract details
                if "FailedStatus" in type(e).__name__:
                    job.refresh()
                # For other errors during wait, refresh and check status
                elif "WaiterError" in type(e).__name__:
                    raise DataPreparerJobTimeoutError(
                        job_arn=getattr(job, "job_arn", "unknown"),
                        elapsed_seconds=0,
                        job_name=getattr(job, "job_name", None),
                    ) from e
                else:
                    raise

            job.refresh()

            # Dispatch on terminal status
            status = getattr(job, "job_status", None)
            if status == "Completed":
                datasets = self._rehydrate_datasets(job, category)
                job_arn = getattr(job, "job_arn", None)
                for ds in datasets:
                    ds.job_arn = job_arn
                self._warn_if_empty_output(datasets, category, job)
                # DataTransformation returns the full list — Split / Merge can
                # yield more than one output dataset per job. SDG / DQ schemas
                # enforce exactly one output dataset service-side; unpack (not
                # [0]) so any future violation fails loudly here instead of
                # silently discarding datasets.
                if category == JobCategory.DATA_TRANSFORMATION:
                    return datasets
                # pylint: disable=unbalanced-tuple-unpacking
                # False positive: pylint infers the empty literal _rehydrate_datasets
                # starts from and reads the length as 0, not the appended results.
                [ds] = datasets
                self._preview_output(ds, category, job)
                return ds
            elif status == "Failed":
                failure_reason = getattr(job, "failure_reason", None)
                if failure_reason == "HUB_FINALIZATION_FAILED":
                    logger.warning(
                        "Job failed with FailureReason='HUB_FINALIZATION_FAILED'. "
                        "Compute succeeded but Hub finalization did not. "
                        "Raising DataPreparerJobFailedError."
                    )
                raise DataPreparerJobFailedError(
                    job_arn=getattr(job, "job_arn", "unknown"),
                    failure_reason=failure_reason,
                    secondary_status=getattr(job, "secondary_status", None),
                    job_name=getattr(job, "job_name", None),
                    category=category,
                )
            elif status == "Stopped":
                raise DataPreparerJobStoppedError(
                    job_arn=getattr(job, "job_arn", "unknown"),
                    job_name=getattr(job, "job_name", None),
                    category=category,
                )
        else:
            if logs:
                logger.warning("Not displaying the job logs as 'wait' is set to False.")
            # Asynchronous (wait=False): return the Job handle so the caller can poll or
            # wait on it later, or persist its name and re-attach by name.
            return job

    def _create_job(
        self,
        dataset: DatasetJobConfig,
        category: JobCategory,
        job_name: Optional[str] = None,
        base_job_name: Optional[str] = None,
        tags: Optional[List[Tag]] = None,
    ):
        """Assemble the CreateJob request and invoke Job.create.

        Per design §2.3.4 / Req 6.1-6.13 the envelope is exactly 7 fields:
        JobName, JobArn, CustomerDetails, RoleArn, JobCategory,
        JobConfigSchemaVersion, JobConfigDocument. Forbidden fields
        (SourceIdentity, CreationTime, IgnoredBySweeper, UseMockAlgo) are
        never set. ``base_job_name`` here is the per-call override per
        Req 1.6; when supplied it takes precedence over ``self.base_job_name``
        for this submission only.
        """
        user_payload = dict(dataset.model_extra or {})
        effective_base = base_job_name or self.base_job_name
        resolved_name = DataPreparerDefaults.resolve_job_name(
            job_name,
            effective_base,
            category,
            user_payload,
        )
        resolved_arn = DataPreparerDefaults.build_job_arn(
            category,
            resolved_name,
            self.sagemaker_session,
        )
        customer_details = DataPreparerDefaults.get_customer_details(self.sagemaker_session)

        logger.info("Submitting %s job: %s", category.value, resolved_name)

        # SDG progress-tracking plumbing: extract the expected sample count and
        # the configured OutputS3Uri from the resolved payload. These are used
        # by Job.wait() to render a "Samples: X/Y" progress line in the Rich
        # status panel by polling S3 Select on the service-written JSONL.
        # DQ jobs track records evaluated via evaluated_data.jsonl.
        expected_samples = None
        progress_output_s3_uri = None
        if category == JobCategory.SYNTHETIC_DATA_GENERATION:
            validated = dataset._validated_payload or {}
            gen_cfg = validated.get("GenerationConfig") or {}
            if isinstance(gen_cfg.get("NumSamples"), int):
                expected_samples = gen_cfg["NumSamples"]
            output_cfg = validated.get("OutputDataConfig") or {}
            if isinstance(output_cfg.get("OutputS3Uri"), str):
                progress_output_s3_uri = output_cfg["OutputS3Uri"]
        elif category == JobCategory.DATA_QUALITY_EVALUATION:
            validated = dataset._validated_payload or {}
            shape = DqShape.for_version(dataset.resolved_schema_version)
            progress_output_s3_uri = shape.extract_output_s3_uri(validated)
            # Count input dataset records for progress display
            dataset_arn = shape.extract_dataset_arn(validated)
            if dataset_arn:
                expected_samples = self._count_input_dataset_records(dataset_arn)

        try:
            Job = _get_job_resource()
            # Build the request dict without forbidden fields. Each entry
            # corresponds to one of the 7 envelope fields in design §1.4.1.
            request = {
                "job_name": resolved_name,
                "job_arn": resolved_arn,
                "customer_details": customer_details,
                "role_arn": self.role,
                "job_category": category.value,
                "job_config_schema_version": dataset.resolved_schema_version,
                "job_config_document": dataset.to_job_config_document(),
                "tags": tags,
            }
            # Strip None values (keeps forbidden-fields invariant trivially true)
            request = {k: v for k, v in request.items() if v is not None}
            # Pass progress-tracking kwargs outside the envelope-filter so
            # they always reach the resource layer (they aren't CreateJob fields).
            job = Job.create(
                session=self.sagemaker_session,
                expected_samples=expected_samples,
                progress_output_s3_uri=progress_output_s3_uri,
                **request,
            )
        except Exception:
            logger.error(
                "Job.create failed for name=%s, category=%s", resolved_name, category.value
            )
            raise

        self._latest_job = job
        job_arn = getattr(job, "job_arn", resolved_name)
        logger.info("Job created: %s", job_arn)
        logger.info('To re-attach after a restart: DataPreparer.attach("%s")', job_arn)
        return job

    def _rehydrate_datasets(self, job, category: JobCategory):
        """Return the completed job's dataset(s) as a list of DataSet objects.

        SDG / DQ echo a scalar ``DatasetArn`` in the JobConfigDocument, so
        the list is always length one. DataTransformation returns
        ``OutputDatasetArns`` (a list) — length depends on the operators
        (e.g. ``Split`` produces N).
        """
        document_str = getattr(job, "job_config_document", None) or "{}"
        document = json.loads(document_str)

        if category == JobCategory.DATA_TRANSFORMATION:
            dataset_arns = list(document.get("OutputDatasetArns") or [])
            missing_field = "OutputDatasetArns"
        elif category == JobCategory.DATA_QUALITY_EVALUATION:
            # DQ v0.2 nests the input under InputDataConfig[0].DatasetArn;
            # DqShape handles both v1 and v0.2 shapes.
            shape = DqShape.for_version(getattr(job, "job_config_schema_version", None))
            scalar = shape.extract_dataset_arn(document)
            dataset_arns = [scalar] if scalar else []
            missing_field = "DatasetArn"
        else:
            scalar = document.get("DatasetArn")
            dataset_arns = [scalar] if scalar else []
            missing_field = "DatasetArn"

        if not dataset_arns:
            raise DataPreparerJobError(
                f"Completed job missing {missing_field} in JobConfigDocument.",
                job_arn=getattr(job, "job_arn", "unknown"),
                job_name=getattr(job, "job_name", None),
                category=category,
            )

        from sagemaker.ai_registry.dataset import DataSet as _DataSet

        # Take the hub and version from the ARN rather than letting the lookup
        # fall back to the derived AI Registry hub and the latest version. The
        # job publishes into the same hub as its inputs, which is the caller's
        # hub and need not be the derived one, and it publishes a specific
        # version — for DataTransformation typically a new version of an
        # existing dataset. Resolving by bare name would raise ResourceNotFound
        # for any other hub, and would otherwise return whichever version is
        # latest at this moment instead of the one this job produced.
        datasets = []
        for arn in dataset_arns:
            content_name, hub_name, version = _parse_dataset_arn(arn)
            logger.info("Retrieving output DataSet from dataset_arn=%s", arn)
            datasets.append(
                _DataSet.get(
                    name=content_name,
                    sagemaker_session=self.sagemaker_session,
                    hub_name=hub_name,
                    version=version,
                )
            )
        return datasets

    def _resolve_output_records_uri(
        self, dataset, category: JobCategory, job=None
    ) -> Optional[str]:
        """Return the S3 URI of the JSONL file holding a completed job's output records.

        Best-effort: returns ``None`` when the location cannot be determined.

        * SyntheticDataGeneration: ``<dataset.source>/output_samples.jsonl``.
        * DataQualityEvaluation: the scored records file
          (``record_level/evaluated_data.jsonl``). When a job is supplied it is the
          one in the job's own output folder, the folder of ``MetadataS3Uri`` in the
          job config document; otherwise the legacy source-based path.
        * DataTransformation: the output dataset's ``source`` object itself, when it
          is a text records file (``.jsonl``, ``.json`` or ``.csv``).
        """
        source = getattr(dataset, "source", None)
        if not source or not isinstance(source, str) or not source.startswith("s3://"):
            return None

        if category == JobCategory.DATA_TRANSFORMATION:
            return source if source.lower().endswith((".jsonl", ".json", ".csv")) else None

        if category == JobCategory.DATA_QUALITY_EVALUATION and job is not None:
            try:
                document = json.loads(getattr(job, "job_config_document", None) or "{}")
            except Exception:  # pylint: disable=broad-except
                document = {}
            metadata_s3_uri = document.get("MetadataS3Uri") if isinstance(document, dict) else None
            if isinstance(metadata_s3_uri, str) and metadata_s3_uri.startswith("s3://"):
                return metadata_s3_uri.rsplit("/", 1)[0] + "/record_level/evaluated_data.jsonl"

        # Legacy / fallback path
        if category == JobCategory.SYNTHETIC_DATA_GENERATION:
            return source.rstrip("/") + "/output_samples.jsonl"
        if category == JobCategory.DATA_QUALITY_EVALUATION:
            base = source.rstrip("/")
            if base.endswith("/output"):
                base = base[: -len("/output")]
            return base + "/record_level/evaluated_data.jsonl"
        return None

    def _output_has_no_records(self, records_uri: str) -> Optional[bool]:
        """Return whether a records file is empty: ``True``, ``False``, or ``None`` if unknown.

        Never downloads large files: a zero-byte object is empty, an object larger
        than ``_EMPTY_CHECK_MAX_BYTES`` is treated as non-empty, and only small
        objects are read and checked for a non-blank line (for CSV, a line beyond
        the header). Any error yields ``None``.
        """
        from urllib.parse import urlparse
        import boto3

        try:
            parsed = urlparse(records_uri)
            bucket, key = parsed.netloc, parsed.path.lstrip("/")
            s3 = boto3.client("s3", region_name=self.sagemaker_session.boto_region_name)
            size = s3.head_object(Bucket=bucket, Key=key).get("ContentLength")
            if size is None:
                return None
            if size == 0:
                return True
            if size > _EMPTY_CHECK_MAX_BYTES:
                return False
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            lines = [line for line in body.splitlines() if line.strip()]
            data_lines = lines[1:] if key.lower().endswith(".csv") else lines
            return not data_lines
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Empty-output check failed (non-blocking): %s", e)
            return None

    def _warn_if_empty_output(self, datasets, category: JobCategory, job=None) -> None:
        """Log a warning for each completed job output that contains no records.

        A job can reach ``Completed`` while producing nothing — for example a
        filter that drops every row. Zero records can be legitimate, so this warns
        rather than raising; the datasets are still returned. Best-effort: never
        raises.
        """
        what = {
            JobCategory.SYNTHETIC_DATA_GENERATION: "generated no records",
            JobCategory.DATA_QUALITY_EVALUATION: "evaluated no records",
            JobCategory.DATA_TRANSFORMATION: "produced an output dataset with no records",
        }.get(category)
        if what is None:
            return
        job_name = getattr(job, "job_name", None) or "<unknown>"
        for dataset in datasets:
            try:
                records_uri = self._resolve_output_records_uri(dataset, category, job)
                if records_uri and self._output_has_no_records(records_uri) is True:
                    logger.warning(
                        "%s job %s completed but %s (%s). The returned dataset is empty; "
                        "check the job configuration (for example, filter thresholds) "
                        "and the input data.",
                        category.value,
                        job_name,
                        what,
                        records_uri,
                    )
            except Exception as e:  # pylint: disable=broad-except
                logger.debug("Empty-output warning skipped (non-blocking): %s", e)

    def _preview_output(self, dataset, category: JobCategory, job=None, n: int = 10) -> None:
        """Display the first n records from the job output JSONL as a pandas DataFrame.

        Best-effort: silently returns on any failure (missing file, permissions, etc.).

        For DataQualityEvaluation, the output file is the one in the job's own
        output folder, the folder of ``MetadataS3Uri``, when a job is supplied;
        falls back to the legacy source-based path when the job is unavailable or
        its config document has no ``MetadataS3Uri``.
        """
        n = min(n, 20)
        try:
            if category not in (
                JobCategory.SYNTHETIC_DATA_GENERATION,
                JobCategory.DATA_QUALITY_EVALUATION,
            ):
                return
            jsonl_uri = self._resolve_output_records_uri(dataset, category, job)
            if jsonl_uri is None:
                return

            from urllib.parse import urlparse
            from io import StringIO
            import boto3
            import pandas as pd

            parsed = urlparse(jsonl_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            s3 = boto3.client("s3", region_name=self.sagemaker_session.boto_region_name)
            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj["Body"].read().decode("utf-8")

            df = pd.read_json(StringIO(content), lines=True, nrows=n)

            pd.set_option("display.max_colwidth", None)
            print(f"\nPreview ({len(df)} records from {jsonl_uri}):\n")
            try:
                from IPython.display import display

                display(df)
            except ImportError:
                print(df.to_string())
        except Exception as e:
            logger.debug("Preview failed (non-blocking): %s", e)

    def _count_input_dataset_records(self, dataset_arn: str) -> Optional[int]:
        """Count records in the input dataset for DQ progress display.

        Resolves the dataset's S3 source from Hub, then counts newlines in the JSONL.
        Returns None on any failure (best-effort, non-blocking).

        Resolves against the hub and version named in ``dataset_arn`` — that ARN is
        the exact input the job was given, so its own hub and version are the right
        ones to read. A bare-name lookup against the derived AI Registry hub returns
        nothing for an input registered in any other hub, which shows up as a missing
        record total in the progress display.
        """
        try:
            from sagemaker.ai_registry.dataset import DataSet as _DataSet

            content_name, hub_name, version = _parse_dataset_arn(dataset_arn)
            ds = _DataSet.get(
                name=content_name,
                sagemaker_session=self.sagemaker_session,
                hub_name=hub_name,
                version=version,
            )

            source = ds.source
            if not source or not source.startswith("s3://") or source == "s3:///":
                return None

            import boto3

            rest = source[len("s3://") :].rstrip("/")
            slash = rest.index("/")
            bucket = rest[:slash]
            prefix = rest[slash + 1 :]

            s3 = boto3.client("s3", region_name=self.sagemaker_session.boto_region_name)

            # List objects under the prefix to find the JSONL file
            listing = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=50)
            objects = listing.get("Contents", []) or []
            target = next((o["Key"] for o in objects if o["Key"].endswith(".jsonl")), None)
            if not target:
                # Prefix itself might be the file
                target = prefix if prefix.endswith(".jsonl") else None
            if not target:
                return None

            resp = s3.get_object(Bucket=bucket, Key=target)
            body = resp["Body"].read()
            if not body:
                return 0
            # Count non-empty lines to handle files without a trailing newline
            # (N records without trailing \n counted as N-1 by b"\n".count()),
            # \r\n line endings, and blank lines that are not valid JSONL records.
            return sum(1 for line in body.splitlines() if line.strip())
        except Exception as e:
            logger.debug("Failed to count input dataset records: %s", e)
            return None
