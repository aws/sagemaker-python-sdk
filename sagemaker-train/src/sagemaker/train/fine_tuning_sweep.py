"""Run serverless fine-tuning sweeps without an AMT training definition."""

import copy
import itertools
import json
import logging
import math
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence

import mlflow
from mlflow.tracking import MlflowClient

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import TrainingJob
from sagemaker.core.s3 import S3Downloader, S3Uploader, parse_s3_url
from sagemaker.train.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

_POLL_SECONDS = 5
_MLFLOW_LOCK = threading.Lock()
_TERMINAL_STATUSES = frozenset(("Completed", "Stopped", "Failed"))
_TRIAL_STATUSES = frozenset(("pending", "running", "completed", "stopped", "failed"))


@dataclass
class SweepTrial:
    """Store a trial's parameters, remote identity, and result."""

    trial_id: int
    hyperparameters: Dict[str, Any]
    training_job_name: Optional[str] = None
    status: str = "pending"
    objective_value: Optional[float] = None
    rung_values: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class SweepConfig:
    """Configure a sweep with finite parameter choices.

    Grid and ASHA use the product order of ``search_space``. The random scheduler
    samples with replacement. ASHA compares each concurrent batch at the given
    steps and retains at least one trial per rung, so a batch of one never stops.
    ``rung_maximize`` ranks the rung metric. It defaults to ``maximize``, so set it
    when a loss gates an accuracy objective. A resumed ASHA sweep batches the
    unfinished trials again in order. With a manifest, every parameter value must
    be JSON serializable.
    """

    search_space: Dict[str, Sequence[Any]]
    max_trials: int
    max_parallel: int = 1
    scheduler: str = "grid"
    maximize: bool = False
    seed: Optional[int] = None
    asha_rungs: Optional[Sequence[int]] = None
    asha_reduction_factor: int = 3
    rung_metric: Optional[str] = None
    rung_maximize: Optional[bool] = None
    manifest_s3_uri: Optional[str] = None

    def __post_init__(self):
        """Reject settings that cannot schedule a trial."""
        for name in ("max_trials", "max_parallel", "asha_reduction_factor"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.scheduler not in ("grid", "random", "asha"):
            raise ValueError("scheduler must be grid, random, or asha")
        self._validate_search_space()
        if self.scheduler == "asha":
            self._validate_asha()
        if self.manifest_s3_uri is not None:
            bucket, key = parse_s3_url(self.manifest_s3_uri)
            if not bucket or not key:
                raise ValueError("manifest_s3_uri must identify an S3 object")

    def _validate_search_space(self):
        if not self.search_space:
            raise ValueError("search_space cannot be empty")
        for name, choices in self.search_space.items():
            if not isinstance(name, str) or not name or not choices:
                raise ValueError("Each parameter must have a name and at least one choice")
            if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
                raise ValueError("Parameter choices must be a sequence, not a string")

    def _validate_asha(self):
        if self.asha_reduction_factor < 2:
            raise ValueError("ASHA reduction factor must be at least 2")
        if not self.rung_metric or not self.asha_rungs:
            raise ValueError("ASHA must have a rung_metric and asha_rungs")
        if any(type(step) is not int or step < 1 for step in self.asha_rungs):
            raise ValueError("ASHA rungs must be positive integer steps")
        if list(self.asha_rungs) != sorted(set(self.asha_rungs)):
            raise ValueError("ASHA rungs must increase without duplicates")


@dataclass(frozen=True)
class _ActiveTrial:
    trial_id: int
    training_job: TrainingJob


def _get_rung_value(training_job: TrainingJob, metric: str, step: int) -> Optional[float]:
    """Return the metric at the largest recorded step at or past the rung."""
    tracking_uri = getattr(
        getattr(training_job, "mlflow_config", None), "mlflow_resource_arn", None
    )
    run_id = getattr(getattr(training_job, "mlflow_details", None), "mlflow_run_id", None)
    if not tracking_uri or not run_id:
        return None
    # sagemaker-mlflow signs each request with the ARN in the global tracking URI,
    # so one lock holds the global setting and the read together.
    with _MLFLOW_LOCK:
        mlflow.set_tracking_uri(tracking_uri)
        readings = MlflowClient(tracking_uri=tracking_uri).get_metric_history(run_id, metric)
    eligible = [reading for reading in readings if reading.step >= step]
    if not eligible:
        return None
    value = float(max(eligible, key=lambda reading: reading.step).value)
    if not math.isfinite(value):
        raise ValueError(f"MLflow metric {metric} has a non-finite value")
    return value


def _refresh_status(training_job: TrainingJob) -> Optional[str]:
    """Return the job status, or None when the describe call fails."""
    try:
        training_job.refresh()
    except Exception:
        logger.exception("Could not describe training job %s", training_job.training_job_name)
        return None
    return training_job.training_job_status


def _stop_quietly(training_job: TrainingJob, reason: str, expected: bool = False) -> bool:
    """Stop the job and log the reason. Return False when the stop call fails."""
    level = logging.INFO if expected else logging.ERROR
    logger.log(level, "Stopping %s: %s", training_job.training_job_name, reason)
    try:
        training_job.stop()
    except Exception:
        logger.exception("Could not stop training job %s", training_job.training_job_name)
        return False
    return True


def _poll_until_terminal(training_job: TrainingJob) -> Optional[str]:
    """Poll the job status without the rich display that ``wait()`` opens.

    Two ``TrainingJob.wait()`` calls in parallel threads share one console, and
    rich 13 raises ``LiveError`` for the second live display.
    """
    while True:
        status = _refresh_status(training_job)
        if status is None or status in _TERMINAL_STATUSES:
            return status
        time.sleep(_POLL_SECONDS)


def _make_trials(config: SweepConfig) -> List[SweepTrial]:
    names = list(config.search_space)
    choices = list(config.search_space.values())
    if config.scheduler == "random":
        generator = random.Random(config.seed)
        points = (
            tuple(generator.choice(values) for values in choices) for _ in range(config.max_trials)
        )
    else:
        points = itertools.islice(itertools.product(*choices), config.max_trials)
    return [SweepTrial(index, dict(zip(names, point))) for index, point in enumerate(points)]


def _validate_trial(trial: SweepTrial) -> None:
    if trial.status not in _TRIAL_STATUSES:
        raise ValueError(f"Unknown trial status: {trial.status}")
    if trial.status == "running" and not trial.training_job_name:
        raise ValueError("A running trial must have a training job name")
    if trial.status == "completed" and trial.objective_value is None:
        raise ValueError("A completed trial must have an objective value")


def _load_trials(records: List[Dict[str, Any]], config: SweepConfig) -> List[SweepTrial]:
    try:
        trials = [SweepTrial(**record) for record in records]
    except TypeError as error:
        raise ValueError(f"The manifest has a malformed trial record: {error}") from error
    if [trial.trial_id for trial in trials] != list(range(len(trials))):
        raise ValueError("The manifest must contain ordered, consecutive trial identifiers")
    if len(trials) > config.max_trials:
        raise ValueError("The manifest has more trials than max_trials")
    for trial in trials:
        _validate_trial(trial)
    return trials


class FineTuningSweep:
    """Run independent trainers with a bounded number of concurrent jobs.

    Args:
        trainer_factory: Build a configured trainer for each parameter dictionary.
        objective: Read a finite score from a completed training job.
        config: Select parameters, concurrency, and optional S3 checkpoints.
        sagemaker_session: Bind S3 calls and resumed jobs to this session.
    """

    def __init__(
        self,
        trainer_factory: Callable[[Dict[str, Any]], BaseTrainer],
        objective: Callable[[TrainingJob], float],
        config: SweepConfig,
        sagemaker_session: Optional[Session] = None,
    ):
        self._trainer_factory = trainer_factory
        self._objective = objective
        self._config = copy.deepcopy(config)
        self._session = sagemaker_session
        self._trials = _make_trials(self._config)
        self._lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._execution: Optional[Future] = None

    @property
    def trials(self) -> List[SweepTrial]:
        """Return snapshots that callers can read without a lock."""
        with self._lock:
            return copy.deepcopy(self._trials)

    @property
    def best_trial(self) -> Optional[SweepTrial]:
        """Return the best completed trial, or None before the first result."""
        completed = [trial for trial in self.trials if trial.status == "completed"]
        if not completed:
            return None
        ordered = sorted(
            completed, key=lambda trial: trial.objective_value, reverse=self._config.maximize
        )
        return ordered[0]

    def _trial(self, trial_id):
        with self._lock:
            return copy.deepcopy(self._trials[trial_id])

    def _checkpoint(self):
        """Write the manifest. The upload lock keeps the newest snapshot last."""
        if self._config.manifest_s3_uri is None:
            return
        with self._upload_lock:
            with self._lock:
                manifest = {
                    "config": asdict(self._config),
                    "trials": [asdict(trial) for trial in self._trials],
                }
            S3Uploader.upload_string_as_file_body(
                body=json.dumps(manifest, allow_nan=False),
                desired_s3_uri=self._config.manifest_s3_uri,
                sagemaker_session=self._session,
            )

    def _update(self, trial_id, **changes):
        """Apply a trial change, then write the manifest. A stopped trial stays stopped."""
        with self._lock:
            current = self._trials[trial_id]
            if current.status == "stopped":
                changes.update(status="stopped", objective_value=None)
            self._trials[trial_id] = replace(current, **changes)
            updated = copy.deepcopy(self._trials[trial_id])
        self._checkpoint()
        return updated

    def _mark_stopped(self, trial_id):
        """Stop a pending or running trial. A finished trial keeps its result."""
        with self._lock:
            current = self._trials[trial_id]
            if current.status not in ("pending", "running"):
                return
            self._trials[trial_id] = replace(current, status="stopped", objective_value=None)
        self._checkpoint()

    def _attach(self, job_name):
        if self._session is None:
            return TrainingJob.get(training_job_name=job_name)
        return TrainingJob.get(
            training_job_name=job_name,
            session=self._session.boto_session,
            region=self._session.boto_region_name,
        )

    def _submit(self, trial):
        """Attach to a trial's recorded job, or submit a new one and return its handle."""
        if trial.training_job_name is not None:
            return self._attach(trial.training_job_name)
        trainer = self._trainer_factory(copy.deepcopy(trial.hyperparameters))
        training_job = trainer.train(wait=False)
        if isinstance(training_job, TrainingJob):
            return training_job
        # A trainer that returns no handle (ModelTrainer before this release) keeps it on
        # latest_training_job.
        training_job = trainer.latest_training_job
        if isinstance(training_job, TrainingJob):
            return training_job
        raise TypeError("The trainer must return a TrainingJob or set latest_training_job")

    def _start_trial(self, trial):
        """Submit or attach the trial. A recorded job that cannot be attached stays running."""
        if self._trial(trial.trial_id).status == "stopped":
            return None
        try:
            training_job = self._submit(trial)
        except Exception:
            logger.exception("Trial %s did not start", trial.trial_id)
            if trial.training_job_name is None:
                self._update(trial.trial_id, status="failed")
            return None
        try:
            current = self._update(
                trial.trial_id, training_job_name=training_job.training_job_name, status="running"
            )
        except Exception:
            if trial.training_job_name is None:
                # The manifest does not name the new job, so a resume would submit it again.
                _stop_quietly(training_job, "the manifest write failed")
            raise
        if current.status == "stopped":
            _stop_quietly(training_job, "the sweep stopped before the job started")
        return _ActiveTrial(trial.trial_id, training_job)

    def _finish_trial(self, active):
        """Score the job. A trial whose job cannot be described stays running."""
        remote_status = _poll_until_terminal(active.training_job)
        if remote_status is None:
            return
        if remote_status == "Stopped" or self._trial(active.trial_id).status == "stopped":
            self._update(active.trial_id, status="stopped")
            return
        if remote_status != "Completed":
            logger.error("Trial %s ended with status %s", active.trial_id, remote_status)
            self._update(active.trial_id, status="failed")
            return
        try:
            objective_value = float(self._objective(active.training_job))
            if not math.isfinite(objective_value):
                raise ValueError("The objective must return a finite value")
        except Exception:
            logger.exception("Trial %s failed", active.trial_id)
            self._update(active.trial_id, status="failed")
            return
        self._update(active.trial_id, status="completed", objective_value=objective_value)

    def _poll_rung(self, active, rung_index):
        """Record the rung value.

        A job that ends before the rung leaves the comparison. A non-finite reading
        stops the trial, because it cannot rank against the others.
        """
        current = self._trial(active.trial_id)
        if current.status == "stopped" or len(current.rung_values) > rung_index:
            return True
        status = _refresh_status(active.training_job)
        if status is None:
            return True
        try:
            value = _get_rung_value(
                active.training_job, self._config.rung_metric, self._config.asha_rungs[rung_index]
            )
        except ValueError as error:
            if _stop_quietly(active.training_job, str(error)):
                self._update(active.trial_id, status="stopped")
            return True
        if value is not None:
            self._update(active.trial_id, rung_values=current.rung_values + [value])
            return True
        return status in _TERMINAL_STATUSES

    def _prune_rung(self, active_trials, rung_index):
        """Rank the jobs still in progress and stop the bottom share.

        A job that cannot be described keeps its last known status.
        """
        snapshots = self.trials
        candidates = []
        for active in active_trials:
            trial = snapshots[active.trial_id]
            if trial.status != "running" or len(trial.rung_values) <= rung_index:
                continue
            status = _refresh_status(active.training_job) or active.training_job.training_job_status
            if status == "InProgress":
                candidates.append(active)
        rung_maximize = self._config.rung_maximize
        if rung_maximize is None:
            rung_maximize = self._config.maximize
        candidates.sort(
            key=lambda active: snapshots[active.trial_id].rung_values[rung_index],
            reverse=rung_maximize,
        )
        retained = max(1, len(candidates) // self._config.asha_reduction_factor)
        for active in candidates[retained:]:
            reason = f"ranked below rung {rung_index}"
            if _stop_quietly(active.training_job, reason, expected=True):
                self._update(active.trial_id, status="stopped")

    def _run_rungs(self, active_trials):
        for rung_index in range(len(self._config.asha_rungs)):
            pending = list(active_trials)
            while pending:
                pending = [active for active in pending if not self._poll_rung(active, rung_index)]
                if pending:
                    time.sleep(_POLL_SECONDS)
            self._prune_rung(active_trials, rung_index)

    def _run_trial(self, trial):
        active = self._start_trial(trial)
        if active is not None:
            self._finish_trial(active)

    def _run_batch(self, trials, executor):
        submissions = [executor.submit(self._start_trial, trial) for trial in trials]
        active_trials = [
            active for future in submissions if (active := future.result()) is not None
        ]
        self._run_rungs(active_trials)
        completions = [executor.submit(self._finish_trial, active) for active in active_trials]
        for future in completions:
            future.result()

    def _execute(self):
        try:
            self._checkpoint()
            self._run_trials()
        except Exception:
            logger.exception("The sweep stopped before every trial ended")
            raise

    def _run_trials(self):
        trials = [trial for trial in self.trials if trial.status in ("pending", "running")]
        batch_size = self._config.max_parallel
        executor = ThreadPoolExecutor(max_workers=batch_size)
        try:
            if self._config.scheduler != "asha":
                for future in [executor.submit(self._run_trial, trial) for trial in trials]:
                    future.result()
                return
            # ASHA compares the trials of one batch at each rung.
            for start in range(0, len(trials), batch_size):
                self._run_batch(trials[start : start + batch_size], executor)
        finally:
            # A sweep error leaves the unsubmitted trials pending for a later resume.
            executor.shutdown(wait=True, cancel_futures=True)

    def run(self, wait: bool = True) -> List[SweepTrial]:
        """Start the sweep once, or wait for an earlier asynchronous run.

        Args:
            wait: Wait for all trials. False returns the current snapshots.

        Returns:
            Trial snapshots. A trial that did not submit or score has status ``failed``.
            A trial whose job the sweep could not describe keeps status ``running`` for
            ``resume()``. A manifest write or MLflow read failure ends the sweep, leaves
            the submitted jobs running, and raises from the next ``run()`` that waits.
        """
        with self._lock:
            if self._execution is None:
                coordinator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweep")
                self._execution = coordinator.submit(self._execute)
                coordinator.shutdown(wait=False)
            execution = self._execution
        if wait:
            execution.result()
        return self.trials

    def stop(self) -> None:
        """Cancel pending trials and stop each running job.

        A trial whose remote stop fails stays running, so ``resume()`` attaches to it.
        """
        failures = []
        for trial in self.trials:
            if trial.status not in ("pending", "running"):
                continue
            if trial.training_job_name is not None:
                try:
                    training_job = self._attach(trial.training_job_name)
                    if training_job.training_job_status not in _TERMINAL_STATUSES:
                        training_job.stop()
                except Exception as error:
                    failures.append(f"{trial.training_job_name}: {error}")
                    continue
            self._mark_stopped(trial.trial_id)
        if failures:
            raise RuntimeError("Could not stop training jobs: " + ", ".join(failures))

    @classmethod
    def resume(cls, manifest_s3_uri, trainer_factory, objective, sagemaker_session=None):
        """Rebuild a sweep from a manifest. The call submits no job.

        Args:
            manifest_s3_uri: Read the saved state from this S3 object.
            trainer_factory: Build trainers for pending trials.
            objective: Read a score from completed training jobs.
            sagemaker_session: Bind S3 calls and resumed jobs to this session.

        Returns:
            A sweep whose ``run()`` skips terminal trials and attaches to running jobs.
        """
        body = S3Downloader.read_file(manifest_s3_uri, sagemaker_session=sagemaker_session)
        manifest = json.loads(body)
        config = SweepConfig(**dict(manifest["config"], manifest_s3_uri=manifest_s3_uri))
        sweep = cls(trainer_factory, objective, config, sagemaker_session)
        sweep._trials = _load_trials(manifest["trials"], config)
        return sweep
