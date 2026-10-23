"""Serverless trainers could not use AMT to schedule and resume parameter sweeps.

``HyperparameterTuner`` builds a ``HyperParameterTrainingJobDefinition`` without a
``serverless_job_config``, so ``SFTTrainer`` and the other serverless trainers had no sweep.
"""

import json
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import pytest

from sagemaker.core.resources import TrainingJob
from sagemaker.train import FineTuningSweep, SweepConfig, SweepTrial
from sagemaker.train.fine_tuning_sweep import _get_rung_value

MANIFEST_URI = "s3://sweep-state/manifest.json"


class FakeTrainer:
    """Return a job that completes after ``in_progress_polls`` refreshes."""

    def __init__(self, hyperparameters, in_progress_polls: Optional[int] = None):
        self.hyperparameters = hyperparameters
        self.in_progress_polls = in_progress_polls or 0
        self.job = Mock(spec=TrainingJob)
        self.job.training_job_name = "trial-" + "-".join(
            str(value) for value in hyperparameters.values()
        )
        self.job.training_job_status = "InProgress"
        self.job.score = sum(hyperparameters.values())
        self.job.refresh.side_effect = self._refresh
        self.job.stop.side_effect = self._stop

    def _refresh(self):
        if self.job.training_job_status != "InProgress":
            return
        if self.in_progress_polls > 0:
            self.in_progress_polls -= 1
            return
        self.job.training_job_status = "Completed"

    def _stop(self):
        self.job.training_job_status = "Stopped"

    def train(self, wait=False, on_submit=None):
        """Return the job without a service call."""
        assert wait is False
        if on_submit is not None:
            on_submit(self.job)
        return self.job


def objective(training_job):
    return training_job.score


def blocking_trainer(entered, release):
    """Build a trainer whose refresh blocks until the test sets ``release``."""

    class BlockingTrainer(FakeTrainer):
        def _refresh(self):
            entered.set()
            assert release.wait(timeout=5)
            super()._refresh()

    return BlockingTrainer


def test_grid_enumerates_product_and_selects_minimum():
    config = SweepConfig({"rate": [2, 1], "epochs": [4, 3]}, max_trials=10)
    sweep = FineTuningSweep(FakeTrainer, objective, config)

    assert sweep.best_trial is None
    trials = sweep.run()

    assert [trial.hyperparameters for trial in trials] == [
        {"rate": 2, "epochs": 4},
        {"rate": 2, "epochs": 3},
        {"rate": 1, "epochs": 4},
        {"rate": 1, "epochs": 3},
    ]
    assert [trial.status for trial in trials] == ["completed"] * 4
    assert [trial.training_job_name for trial in trials] == [
        "trial-2-4",
        "trial-2-3",
        "trial-1-4",
        "trial-1-3",
    ]
    assert sweep.best_trial.hyperparameters == {"rate": 1, "epochs": 3}
    assert sweep.best_trial.objective_value == 4


def test_maximize_selects_largest_objective():
    config = SweepConfig({"rate": [1, 5, 3]}, max_trials=3, maximize=True)
    sweep = FineTuningSweep(FakeTrainer, objective, config)

    sweep.run()

    assert sweep.best_trial.objective_value == 5


def test_grid_respects_max_trials():
    config = SweepConfig({"rate": [1, 2, 3]}, max_trials=2)
    sweep = FineTuningSweep(FakeTrainer, objective, config)

    assert len(sweep.run()) == 2


def test_random_seed_reproduces_samples_and_trial_limit():
    config = SweepConfig(
        {"rate": [1, 2, 3], "epochs": [4, 5]}, max_trials=9, scheduler="random", seed=12
    )
    first = FineTuningSweep(FakeTrainer, objective, config)
    second = FineTuningSweep(FakeTrainer, objective, config)

    assert len(first.run()) == 9
    assert [trial.hyperparameters for trial in first.trials] == [
        trial.hyperparameters for trial in second.run()
    ]
    assert all(trial.hyperparameters["rate"] in (1, 2, 3) for trial in first.trials)


@pytest.mark.parametrize("maximize,kept", [(False, {1, 2}), (True, {5, 6})])
def test_asha_stops_bottom_share_at_rung(maximize, kept):
    config = SweepConfig(
        {"rate": [1, 2, 3, 4, 5, 6]},
        max_trials=6,
        max_parallel=6,
        scheduler="asha",
        maximize=maximize,
        asha_rungs=[10],
        rung_metric="total_loss",
    )
    sweep = FineTuningSweep(
        lambda hyperparameters: FakeTrainer(hyperparameters, in_progress_polls=2),
        objective,
        config,
    )
    with patch(
        "sagemaker.train.fine_tuning_sweep._get_rung_value",
        side_effect=lambda job, metric, step: job.score,
    ) as rung_value:
        trials = sweep.run()

    completed = {trial.hyperparameters["rate"] for trial in trials if trial.status == "completed"}
    assert completed == kept
    assert sum(trial.status == "stopped" for trial in trials) == 4
    assert all(len(trial.rung_values) == 1 for trial in trials)
    assert all(trial.objective_value is None for trial in trials if trial.status == "stopped")
    assert {call.args[1:] for call in rung_value.call_args_list} == {("total_loss", 10)}


def test_asha_rung_direction_can_differ_from_objective_direction():
    config = SweepConfig(
        {"rate": [1, 2, 3, 4, 5, 6]},
        max_trials=6,
        max_parallel=6,
        scheduler="asha",
        maximize=True,
        rung_maximize=False,
        asha_rungs=[10],
        rung_metric="total_loss",
    )
    sweep = FineTuningSweep(
        lambda hyperparameters: FakeTrainer(hyperparameters, in_progress_polls=2),
        objective,
        config,
    )
    with patch(
        "sagemaker.train.fine_tuning_sweep._get_rung_value",
        side_effect=lambda job, metric, step: job.score,
    ):
        trials = sweep.run()

    completed = {trial.hyperparameters["rate"] for trial in trials if trial.status == "completed"}
    assert completed == {1, 2}
    assert sweep.best_trial.hyperparameters == {"rate": 2}


def test_asha_nonfinite_rung_reading_stops_only_that_trial():
    config = SweepConfig(
        {"rate": [1, 2, 3]},
        max_trials=3,
        max_parallel=3,
        scheduler="asha",
        asha_rungs=[10],
        rung_metric="total_loss",
    )
    trainers = {}

    def factory(hyperparameters):
        trainers[hyperparameters["rate"]] = FakeTrainer(hyperparameters, in_progress_polls=2)
        return trainers[hyperparameters["rate"]]

    def rung_value(job, metric, step):
        if job.score == 2:
            raise ValueError("MLflow metric total_loss has a non-finite value")
        return job.score

    sweep = FineTuningSweep(factory, objective, config)
    with patch("sagemaker.train.fine_tuning_sweep._get_rung_value", side_effect=rung_value):
        trials = sweep.run()

    assert [trial.status for trial in trials] == ["completed", "stopped", "stopped"]
    trainers[2].job.stop.assert_called_once_with()
    assert trials[1].rung_values == []


def test_asha_retains_one_trial_at_second_rung():
    config = SweepConfig(
        {"rate": [1, 2, 3, 4, 5, 6]},
        max_trials=6,
        max_parallel=6,
        scheduler="asha",
        asha_rungs=[10, 20],
        rung_metric="total_loss",
    )
    sweep = FineTuningSweep(
        lambda hyperparameters: FakeTrainer(hyperparameters, in_progress_polls=4),
        objective,
        config,
    )
    with patch(
        "sagemaker.train.fine_tuning_sweep._get_rung_value",
        side_effect=lambda job, metric, step: job.score,
    ):
        trials = sweep.run()

    assert sum(trial.status == "completed" for trial in trials) == 1
    assert sweep.best_trial.hyperparameters == {"rate": 1}
    assert sweep.best_trial.rung_values == [1, 1]


def test_max_parallel_runs_two_trainers_at_once():
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    counts = {"active": 0, "maximum": 0}

    class ConcurrentTrainer(FakeTrainer):
        def train(self, wait=False, on_submit=None):
            with lock:
                counts["active"] += 1
                counts["maximum"] = max(counts.values())
            barrier.wait(timeout=5)
            with lock:
                counts["active"] -= 1
            return super().train(wait=wait, on_submit=on_submit)

    config = SweepConfig({"rate": [1, 2, 3, 4]}, max_trials=4, max_parallel=2)
    sweep = FineTuningSweep(ConcurrentTrainer, objective, config)

    assert all(trial.status == "completed" for trial in sweep.run())
    assert counts == {"active": 0, "maximum": 2}


def test_rolling_window_starts_next_trial_when_one_slot_frees():
    entered = threading.Event()
    release = threading.Event()
    started = []
    lock = threading.Lock()

    def factory(hyperparameters):
        with lock:
            started.append(hyperparameters["rate"])
        if hyperparameters["rate"] == 1:
            return blocking_trainer(entered, release)(hyperparameters)
        return FakeTrainer(hyperparameters)

    config = SweepConfig({"rate": [1, 2, 3]}, max_trials=3, max_parallel=2)
    sweep = FineTuningSweep(factory, objective, config)
    try:
        sweep.run(wait=False)
        assert entered.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and sweep.trials[2].status != "completed":
            time.sleep(0.01)
        assert [trial.status for trial in sweep.trials] == ["running", "completed", "completed"]
    finally:
        release.set()

    assert [trial.status for trial in sweep.run()] == ["completed"] * 3
    assert started == [1, 2, 3]


def test_manifest_records_initial_and_each_trial_state():
    config = SweepConfig({"rate": [1, 2]}, max_trials=2, manifest_s3_uri=MANIFEST_URI)
    session = Mock()
    sweep = FineTuningSweep(FakeTrainer, objective, config, session)

    with patch("sagemaker.train.fine_tuning_sweep.S3Uploader.upload_string_as_file_body") as upload:
        sweep.run()

    manifests = [json.loads(call.kwargs["body"]) for call in upload.call_args_list]
    assert [[trial["status"] for trial in manifest["trials"]] for manifest in manifests] == [
        ["pending", "pending"],
        ["running", "pending"],
        ["completed", "pending"],
        ["completed", "running"],
        ["completed", "completed"],
    ]
    assert manifests[-1]["config"] == asdict(config)
    assert manifests[-1]["trials"][1] == asdict(sweep.trials[1])
    assert all(call.kwargs["sagemaker_session"] is session for call in upload.call_args_list)
    assert all(call.kwargs["desired_s3_uri"] == MANIFEST_URI for call in upload.call_args_list)


def test_resume_skips_completed_trial_and_attaches_running_trial():
    config = SweepConfig({"rate": [1, 2, 3]}, max_trials=3)
    trials = [
        SweepTrial(0, {"rate": 1}, "done", "completed", 1),
        SweepTrial(1, {"rate": 2}, "in-progress", "running"),
        SweepTrial(2, {"rate": 3}),
    ]
    manifest = json.dumps({"config": asdict(config), "trials": [asdict(trial) for trial in trials]})
    resumed_job = FakeTrainer({"rate": 2}).job
    resumed_job.training_job_name = "in-progress"
    session = Mock()
    factory = Mock(side_effect=FakeTrainer)
    with (
        patch(
            "sagemaker.train.fine_tuning_sweep.S3Downloader.read_file", return_value=manifest
        ) as read_file,
        patch("sagemaker.train.fine_tuning_sweep.S3Uploader.upload_string_as_file_body") as upload,
        patch.object(TrainingJob, "get", return_value=resumed_job) as get_job,
    ):
        sweep = FineTuningSweep.resume(MANIFEST_URI, factory, objective, session)
        result = sweep.run()

    read_file.assert_called_once_with(MANIFEST_URI, sagemaker_session=session)
    factory.assert_called_once_with({"rate": 3})
    get_job.assert_called_once_with(
        training_job_name="in-progress",
        session=session.boto_session,
        region=session.boto_region_name,
    )
    assert [trial.status for trial in result] == ["completed"] * 3
    assert [trial.training_job_name for trial in result] == ["done", "in-progress", "trial-3"]
    assert upload.call_args.kwargs["desired_s3_uri"] == MANIFEST_URI


def test_rung_helper_uses_largest_step_at_or_past_rung():
    training_job = FakeTrainer({"rate": 1}).job
    training_job.mlflow_config = SimpleNamespace(mlflow_resource_arn="tracking-resource")
    training_job.mlflow_details = SimpleNamespace(mlflow_run_id="run-id")
    readings = [
        SimpleNamespace(step=3, value=6),
        SimpleNamespace(step=20, value=2),
        SimpleNamespace(step=10, value=4),
    ]
    with (
        patch("sagemaker.train.fine_tuning_sweep.mlflow.set_tracking_uri") as set_uri,
        patch("sagemaker.train.fine_tuning_sweep.MlflowClient") as client,
    ):
        client.return_value.get_metric_history.return_value = readings
        assert _get_rung_value(training_job, "loss", 10) == 2
        assert _get_rung_value(training_job, "loss", 30) is None

    set_uri.assert_called_with("tracking-resource")
    client.assert_called_with(tracking_uri="tracking-resource")
    client.return_value.get_metric_history.assert_called_with("run-id", "loss")


def test_rung_helper_waits_for_mlflow_identity():
    training_job = FakeTrainer({"rate": 1}).job

    assert _get_rung_value(training_job, "loss", 10) is None


def test_model_trainer_style_return_falls_back_to_latest_training_job():
    trainer = FakeTrainer({"rate": 1})
    trainer.latest_training_job = trainer.job
    trainer.train = Mock(return_value=None)
    sweep = FineTuningSweep(
        lambda hyperparameters: trainer, objective, SweepConfig({"rate": [1]}, 1)
    )

    assert sweep.run()[0].status == "completed"
    trainer.train.assert_called_once_with(wait=False)


def test_trainer_without_training_job_fails_trial():
    trainer = FakeTrainer({"rate": 1})
    trainer.latest_training_job = None
    trainer.train = Mock(return_value=None)
    sweep = FineTuningSweep(
        lambda hyperparameters: trainer, objective, SweepConfig({"rate": [1]}, 1)
    )

    assert sweep.run()[0].status == "failed"


def test_hyperpod_style_job_name_return_uses_latest_training_job():
    trainer = FakeTrainer({"rate": 1})
    trainer.latest_training_job = trainer.job
    trainer.train = Mock(return_value="trial-1")
    sweep = FineTuningSweep(
        lambda hyperparameters: trainer, objective, SweepConfig({"rate": [1]}, 1)
    )

    trials = sweep.run()

    assert trials[0].status == "completed"
    assert trials[0].training_job_name == "trial-1"


def test_manifest_failure_after_attach_keeps_the_resumed_job():
    config = SweepConfig({"rate": [1]}, max_trials=1, manifest_s3_uri=MANIFEST_URI)
    trials = [SweepTrial(0, {"rate": 1}, "in-progress", "running")]
    manifest = json.dumps({"config": asdict(config), "trials": [asdict(trials[0])]})
    resumed_job = FakeTrainer({"rate": 1}).job
    upload = Mock(side_effect=[None, RuntimeError("S3 denied")])
    with (
        patch("sagemaker.train.fine_tuning_sweep.S3Downloader.read_file", return_value=manifest),
        patch("sagemaker.train.fine_tuning_sweep.S3Uploader.upload_string_as_file_body", upload),
        patch.object(TrainingJob, "get", return_value=resumed_job),
    ):
        sweep = FineTuningSweep.resume(MANIFEST_URI, FakeTrainer, objective)
        with pytest.raises(RuntimeError, match="S3 denied"):
            sweep.run()

    resumed_job.stop.assert_not_called()


def test_manifest_failure_after_submission_stops_the_job():
    config = SweepConfig({"rate": [1]}, max_trials=1, manifest_s3_uri=MANIFEST_URI)
    trainer = FakeTrainer({"rate": 1})
    sweep = FineTuningSweep(lambda hyperparameters: trainer, objective, config)
    upload = Mock(side_effect=[None, RuntimeError("S3 denied")])
    with patch("sagemaker.train.fine_tuning_sweep.S3Uploader.upload_string_as_file_body", upload):
        with pytest.raises(RuntimeError, match="S3 denied"):
            sweep.run()

    trainer.job.stop.assert_called_once_with()
    assert sweep.trials[0].status == "running"


def test_manifest_failure_still_raises_when_the_stop_fails():
    config = SweepConfig({"rate": [1]}, max_trials=1, manifest_s3_uri=MANIFEST_URI)
    trainer = FakeTrainer({"rate": 1})
    trainer.job.stop.side_effect = RuntimeError("AccessDenied")
    sweep = FineTuningSweep(lambda hyperparameters: trainer, objective, config)
    upload = Mock(side_effect=[None, RuntimeError("S3 denied")])
    with patch("sagemaker.train.fine_tuning_sweep.S3Uploader.upload_string_as_file_body", upload):
        with pytest.raises(RuntimeError, match="S3 denied"):
            sweep.run()

    trainer.job.stop.assert_called_once_with()


def test_resume_keeps_running_trial_when_attach_fails():
    config = SweepConfig({"rate": [1]}, max_trials=1)
    trials = [SweepTrial(0, {"rate": 1}, "in-progress", "running")]
    manifest = json.dumps({"config": asdict(config), "trials": [asdict(trials[0])]})
    with (
        patch("sagemaker.train.fine_tuning_sweep.S3Downloader.read_file", return_value=manifest),
        patch("sagemaker.train.fine_tuning_sweep.S3Uploader.upload_string_as_file_body"),
        patch.object(TrainingJob, "get", side_effect=RuntimeError("AccessDenied")),
    ):
        sweep = FineTuningSweep.resume(MANIFEST_URI, FakeTrainer, objective)
        result = sweep.run()

    assert result[0].status == "running"
    assert result[0].training_job_name == "in-progress"


def test_lost_contact_leaves_trial_running_for_resume():
    trainer = FakeTrainer({"rate": 1})
    trainer.job.refresh.side_effect = RuntimeError("ThrottlingException")
    sweep = FineTuningSweep(
        lambda hyperparameters: trainer, objective, SweepConfig({"rate": [1]}, 1)
    )

    trials = sweep.run()

    assert trials[0].status == "running"
    assert trials[0].training_job_name == "trial-1"
    trainer.job.stop.assert_not_called()
    assert sweep.best_trial is None


def test_individual_failure_does_not_block_other_trials():
    def factory(hyperparameters):
        if hyperparameters["rate"] == 1:
            raise RuntimeError("Rejected training job")
        return FakeTrainer(hyperparameters)

    config = SweepConfig({"rate": [1, 2]}, max_trials=2)
    sweep = FineTuningSweep(factory, objective, config)

    assert [trial.status for trial in sweep.run()] == ["failed", "completed"]
    assert sweep.best_trial.hyperparameters == {"rate": 2}


def test_failed_training_job_fails_trial():
    trainer = FakeTrainer({"rate": 1})
    trainer.job.refresh.side_effect = lambda: setattr(trainer.job, "training_job_status", "Failed")
    sweep = FineTuningSweep(
        lambda hyperparameters: trainer, objective, SweepConfig({"rate": [1]}, 1)
    )

    assert sweep.run()[0].status == "failed"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_objective_fails_trial(value):
    config = SweepConfig({"rate": [1]}, max_trials=1)
    sweep = FineTuningSweep(FakeTrainer, lambda job: value, config)

    assert sweep.run()[0].status == "failed"
    assert sweep.best_trial is None


def test_trial_snapshots_do_not_mutate_state():
    config = SweepConfig({"rate": [1]}, max_trials=1)
    sweep = FineTuningSweep(FakeTrainer, objective, config)
    snapshot = sweep.trials[0]
    snapshot.hyperparameters["rate"] = 99
    snapshot.status = "completed"

    assert sweep.trials[0].hyperparameters == {"rate": 1}
    assert sweep.trials[0].status == "pending"


def test_async_run_returns_before_training_and_can_wait_again():
    entered = threading.Event()
    release = threading.Event()
    config = SweepConfig({"rate": [1]}, max_trials=1)
    sweep = FineTuningSweep(blocking_trainer(entered, release), objective, config)
    try:
        sweep.run(wait=False)
        assert entered.wait(timeout=5)
        assert sweep.trials[0].status == "running"
    finally:
        release.set()

    assert sweep.run()[0].status == "completed"


def test_stop_ends_running_trial():
    entered = threading.Event()
    release = threading.Event()
    trainers = []

    def factory(hyperparameters):
        trainers.append(blocking_trainer(entered, release)(hyperparameters))
        return trainers[-1]

    sweep = FineTuningSweep(factory, objective, SweepConfig({"rate": [1]}, max_trials=1))
    try:
        sweep.run(wait=False)
        assert entered.wait(timeout=5)
        with patch.object(TrainingJob, "get", side_effect=lambda **kwargs: trainers[0].job):
            sweep.stop()
    finally:
        release.set()

    assert sweep.run()[0].status == "stopped"
    trainers[0].job.stop.assert_called_once_with()


def test_stop_keeps_trial_running_when_remote_stop_fails():
    entered = threading.Event()
    release = threading.Event()
    trainers = []

    def factory(hyperparameters):
        trainers.append(blocking_trainer(entered, release)(hyperparameters))
        return trainers[-1]

    sweep = FineTuningSweep(factory, objective, SweepConfig({"rate": [1]}, max_trials=1))
    try:
        sweep.run(wait=False)
        assert entered.wait(timeout=5)
        trainers[0].job.stop.side_effect = RuntimeError("AccessDenied")
        with patch.object(TrainingJob, "get", side_effect=lambda **kwargs: trainers[0].job):
            with pytest.raises(RuntimeError, match="trial-1: AccessDenied"):
                sweep.stop()
        assert sweep.trials[0].status == "running"
    finally:
        release.set()

    assert sweep.run()[0].status == "completed"


def test_stop_wins_over_a_later_refresh_failure():
    entered = threading.Event()
    release = threading.Event()
    trainers = []

    def factory(hyperparameters):
        trainers.append(blocking_trainer(entered, release)(hyperparameters))
        return trainers[-1]

    sweep = FineTuningSweep(factory, objective, SweepConfig({"rate": [1]}, max_trials=1))
    try:
        sweep.run(wait=False)
        assert entered.wait(timeout=5)
        with patch.object(TrainingJob, "get", side_effect=lambda **kwargs: trainers[0].job):
            sweep.stop()
        trainers[0].job.refresh.side_effect = RuntimeError("ThrottlingException")
    finally:
        release.set()

    assert sweep.run()[0].status == "stopped"


def test_stop_cancels_pending_trials_without_starting_trainers():
    factory = Mock(side_effect=FakeTrainer)
    sweep = FineTuningSweep(factory, objective, SweepConfig({"rate": [1, 2]}, max_trials=2))

    sweep.stop()

    assert [trial.status for trial in sweep.run()] == ["stopped", "stopped"]
    factory.assert_not_called()


def test_checkpoint_failure_does_not_start_training():
    config = SweepConfig({"rate": [1]}, max_trials=1, manifest_s3_uri=MANIFEST_URI)
    factory = Mock(side_effect=FakeTrainer)
    sweep = FineTuningSweep(factory, objective, config)
    with patch(
        "sagemaker.train.fine_tuning_sweep.S3Uploader.upload_string_as_file_body",
        side_effect=RuntimeError("S3 denied"),
    ):
        with pytest.raises(RuntimeError, match="S3 denied"):
            sweep.run()

    factory.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        {"max_trials": 0},
        {"max_parallel": 0},
        {"scheduler": "unknown"},
        {"search_space": {}},
        {"search_space": {"rate": []}},
        {"search_space": {"rate": "123"}},
        {"scheduler": "asha"},
        {"scheduler": "asha", "rung_metric": "loss", "asha_rungs": [2, 1]},
        {"scheduler": "asha", "rung_metric": "loss", "asha_rungs": [1, 1]},
        {"scheduler": "asha", "rung_metric": "loss", "asha_rungs": [1], "asha_reduction_factor": 1},
        {"manifest_s3_uri": "s3://bucket-only"},
    ],
)
def test_invalid_config_rejects_before_submission(settings):
    arguments = {"search_space": {"rate": [1]}, "max_trials": 1}
    arguments.update(settings)

    with pytest.raises(ValueError):
        SweepConfig(**arguments)


@pytest.mark.parametrize(
    "records",
    [
        [{"trial_id": 1, "hyperparameters": {"rate": 1}}],
        [{"trial_id": 0, "hyperparameters": {"rate": 1}, "status": "running"}],
        [{"trial_id": 0, "hyperparameters": {"rate": 1}, "status": "completed"}],
        [{"trial_id": 0, "hyperparameters": {"rate": 1}, "status": "unknown"}],
        [{"trial_id": 0}],
        [{"trial_id": 0, "hyperparameters": {"rate": 1}, "job": "extra-key"}],
    ],
)
def test_resume_rejects_inconsistent_manifest(records):
    config = SweepConfig({"rate": [1]}, max_trials=1)
    manifest = json.dumps({"config": asdict(config), "trials": records})

    with (
        patch("sagemaker.train.fine_tuning_sweep.S3Downloader.read_file", return_value=manifest),
        pytest.raises(ValueError),
    ):
        FineTuningSweep.resume(MANIFEST_URI, FakeTrainer, objective)
