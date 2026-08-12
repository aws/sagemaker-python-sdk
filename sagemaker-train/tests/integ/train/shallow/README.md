# Shallow (submit-then-stop) integration tests

These tests replace the full `sagemaker-train` integ suite **on the PR gate only**.
The deep suites still run on the scheduled CI-health workflows.

## What a passing test proves

Each test submits a real `CreateTrainingJob`, asserts the service returned a
`TrainingJobArn`, then immediately stops the job.

The ARN is returned synchronously, and only after the request has cleared every
synchronous server-side gate:

| Layer | Checks |
|---|---|
| Public API front end | Coral model/shape validation, required-member checks, SigV4 |
| IAM | `sagemaker:CreateTrainingJob` incl. condition keys, `iam:PassRole` on the execution role, training-plan ARN authorization |
| Interceptors | marketplace entitlement, resource reservation, tag governance, experiment config, IdC |
| Training backend — sync validators | ~56 validators: instance type/count, volume, KMS, stopping condition, channels, output config, VPC, debug/profiler, HPO params, environment, payload size, ARN partition/region, unlaunched-feature gating |
| Training backend — mutating validators | recipe resolution / hub content fetch |
| Training backend — role-assuming validators | real S3, ECR, FSx, algorithm, VPC dry-run calls **as the customer** |
| Post-validator business logic | training-plan capacity, per-preference plan matching, state-machine routing, SDC lookups, recipe filtering |
| Entity write | duplicate job name → `ResourceInUse` |

So "the ARN came back" means: **the payload the SDK produced was accepted by the
service exactly as sent, and the caller held the permissions needed to submit it.**

## What these tests deliberately do NOT cover

Nothing about training *behaviour*: no model artifacts, no metrics, no container
logs, no convergence, no output-model-package creation. Those require a job to
actually run and remain the responsibility of the deep suites.

Concretely, a regression that makes training itself fail — a broken entry script,
a bad container command, a distributed-launch bug — **will still pass here.** That
is the accepted trade for the runtime and cost reduction.

## Layout

One file per trainer, mirroring the existing deep suite so the shallow counterpart
of any deep test is easy to find:

| Shallow file | Deep counterpart |
|---|---|
| `test_model_trainer.py` | `test_model_trainer.py` |
| `test_sft_trainer.py` | `test_sft_trainer_integration.py` |
| `test_dpo_trainer.py` | `test_dpo_trainer_integration.py` |
| `test_rlvr_trainer.py` | `test_rlvr_trainer_integration.py` |
| `test_rlaif_trainer.py` | `test_rlaif_trainer_integration.py` |
| `test_cpt_trainer.py` | `test_cpt_hyperpod.py` |
| `test_multi_turn_rl_trainer.py` | `test_multi_turn_rl_trainer_integration.py` |
| `test_tuner.py` | `test_tuner_distributed.py` |
| `test_nova_data_mixing.py` | `test_sft_trainer_data_mixing_integration.py` |

`recipe_cases.py` holds the cases every recipe trainer shares (minimal submit,
validation dataset, dataset override, output path, serverful compute, and the two
negative cases). Each per-trainer class subclasses `RecipeTrainerCases` and sets
`TRAINER`, so a new trainer is a two-line file. Override the class attributes only
where the trainer genuinely differs:

* `EXTRA_KWARGS` — required constructor args (RLAIF's reward model/prompt)
* `SUPPORTS_SERVERFUL = False` — trainer takes no `compute` (RLAIF)
* `SUPPORTS_TRAINING_TYPE = False` — no LoRA/full distinction (CPT)

It is deliberately not named `test_*` so pytest does not collect the base class.

## What was marked `gpu_intensive`, and why only those

A deep test is only marked `gpu_intensive` (i.e. moved off the PR gate) when this
suite has a shallow test covering the same code path. 10 tests met that bar:

| Deep test (now marked) | Shallow equivalent |
|---|---|
| `test_model_trainer.py::test_source_dir_local_tar_file` | `TestSourceCodePackaging::test_local_tar_file_source_dir` |
| `::test_hp_contract_basic_py_script` | `TestMinimalSubmission::test_minimal_request_is_accepted` |
| `::test_hp_contract_basic_sh_script` | `TestSourceCodePackaging::test_shell_entry_script` |
| `::test_hp_contract_mpi_script` | `TestComputeConfiguration::test_mpi_distributed` |
| `::test_hp_contract_torchrun_script` | `TestComputeConfiguration::test_torchrun_distributed` |
| `::test_hp_contract_hyperparameter_json` | `TestPayloadShaping::test_hyperparameters_from_json_file` |
| `::test_hp_contract_hyperparameter_yaml` | `TestPayloadShaping::test_hyperparameters_from_yaml_file` |
| `::test_custom_distributed_driver` | `TestSourceCodePackaging::test_custom_distributed_driver` |
| `test_sft_trainer_integration.py::test_sft_trainer_lora_with_sequence_length` | `test_sft_trainer.py::test_sequence_length_is_accepted` |
| `test_tuner_distributed.py::test_tuner_includes_sm_drivers_channel` | `test_tuner.py::test_distributed_tuning_job_is_accepted` |

**Deliberately NOT marked**, because this suite does not cover them — marking them
would remove coverage with nothing replacing it:

* every evaluator test (`test_benchmark_evaluator.py`, `test_custom_scorer_evaluator.py`,
  `test_inspect_ai_evaluator.py`, `test_llm_as_judge_*`, `test_llmaj_custom_model.py`)
  — `evaluate()` is a different API surface returning pipeline executions, and there
  is no shallow coverage for it yet
* `test_notifications.py` — asserts EventBridge/SNS side effects, not submission
* `test_local_model_trainer.py` — local container mode makes no service call

**The rule to preserve:** do not add `gpu_intensive` to a deep test unless a shallow
test covers the same path. Otherwise the PR gate silently loses coverage.

## Relationship to `dry_run=True`

`tests/integ/train/test_dry_run_integration.py` covers `trainer.train(dry_run=True)`,
which returns *before* submitting. It therefore validates only client-side logic
(config assembly, S3 path existence checks, hyperparameter constraints) and
exercises **none** of the table above.

These suites are complementary and both are cheap:

* `dry_run` — catches SDK-side problems with no service call at all.
* shallow — catches problems only the service can detect.

## Cost and capacity

Stopping is not free and not instantaneous. `StopTrainingJob` marks the job
`Stopping` and returns; the compute layer reacts asynchronously. Meanwhile the
create call has already handed the job to a state machine and queued it, so
capacity acquisition has begun.

In practice a job stopped within seconds is torn down while still in
`Starting`/`Pending`, before instances become billable — but that is a timing
property, **not a guarantee**. Expect a small, non-deterministic cost per test,
and transient capacity consumption.

Two design rules follow, and should be preserved:

1. **Use the smallest instance that exercises the path.** `ModelTrainer` tests use
   `ml.m5.large`; payload and permission validation is instance-type agnostic.
   Only the recipe trainers pin an accelerator type (`ml.g5.12xlarge`), because
   their recipes will not resolve onto CPU.
2. **Never set `keep_alive_period_in_seconds`.** A warm pool would outlive the stop
   and keep instances provisioned after the test finished.

## Writing a new test

Use the harness; do not call `trainer.train()` directly.

```python
from .harness import assert_submitted, submitted, unique_name

def test_my_feature_is_accepted(sagemaker_session, train_data_uri):
    trainer = _trainer(sagemaker_session, unique_name("shallow-my-feature"), ...)
    with submitted(trainer) as job:
        assert_submitted(job)
```

`submitted()` forces `wait=False`, resolves the submitted job across the
inconsistent trainer attributes (`_latest_training_job` vs `latest_training_job`),
and stops the job in a `finally` so a failed assertion still cleans up. Passing
`wait=` is rejected with a `TypeError` so a copy-pasted `wait=True` cannot
silently reintroduce a full training run.

For negative cases use `assert_rejected`, which also stops the job if the request
is unexpectedly *accepted*:

```python
assert_rejected(trainer, ("does not exist", "ValidationException"))
```

Keep at least one negative test per feature area. Without them the suite
degenerates into "any ARN is fine" and would stay green even if the SDK started
sending a permissive-but-wrong payload.
