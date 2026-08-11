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

## Coverage vs. the suite this replaces

`tests/integ/train` has 181 pre-existing tests, but only ~50 actually submit a
job — the rest are client-side (recipe resolution, data utils, log streaming,
docker-compose detection). Mapping the *submitting* ones against this suite:

| Existing area | Ported here | Notes |
|---|---|---|
| `test_model_trainer.py` (8) | yes | hyperparameter contract (dict/JSON/YAML), MPI, Torchrun, local tar source, `.sh` entry script, custom distributed driver |
| `test_sft_trainer_integration.py` (4) | partly | LoRA/FULL, validation dataset, `sequence_length`. **Nova workflow not ported** (us-east-1 + gated model) |
| `test_dpo_trainer_integration.py` (2) | yes | via `RECIPE_TRAINERS` parametrization |
| `test_rlvr_trainer_integration.py` (7) | partly | base + recipe/overrides + direct hyperparameter mutation. **Custom reward function / evaluator objects not ported** |
| `test_rlaif_trainer_integration.py` (3) | yes | RLAIF is in `RECIPE_TRAINERS`; its reward model/prompt come from `_TRAINER_EXTRA_KWARGS` |
| `test_cpt_hyperpod.py`, `test_nova_sft_hyperpod.py`, `test_sft_data_mixing_hyperpod.py` (3) | no | HyperPod submits to a pre-provisioned cluster, not `CreateTrainingJob` — the pattern does not apply |
| `test_sft_trainer_data_mixing_integration.py` (1) | yes | `DataMixingConfig`, both explicit and recipe-default |
| `test_tuner_distributed.py` (1) | yes | `HyperParameterTuningJob`; also asserts the `sm_drivers` channel survived submission |
| `test_multi_turn_rl_trainer_integration.py` (7) | partly | AgentRFT `Job` submission, marked `gpu_intensive` — see below |
| `test_recipe_override_integration.py` (35) | n/a | client-side `get_resolved_recipe`; keep as-is, cheap already |
| Evaluators (`test_benchmark_evaluator.py`, `test_llm_as_judge_*`, `test_mtrl_*`, ~20) | no | `evaluate()` not `train()`; the same pattern applies and is the clearest next extension |
| `test_notifications.py`, `test_local_model_trainer.py` | no | EventBridge/SNS side effects and local-container mode (no service call) |

Note that not every trainer creates a `TrainingJob`. `HyperparameterTuner` creates
a `HyperParameterTuningJob` and `MultiTurnRLTrainer` creates an AgentRFT `Job`, so
`assert_submitted` takes a `resource=` argument for the expected ARN segment and
the harness resolves the submitted job across four different attribute names.

**Deliberately out of scope for this pattern:** HyperPod (different submission
API), local container mode (no service call), and anything asserting a job's
*outcome*.

**Requires prerequisites, so marked `gpu_intensive` and skipped on the PR gate:**
the MTRL tests. Unlike everything else here they cannot be made self-contained —
they need a pre-provisioned agent runtime and MLflow app. They read those from
`SHALLOW_MTRL_AGENT_ENV` / `SHALLOW_MTRL_MLFLOW_APP_ARN` / `SHALLOW_MTRL_DATASET`
and skip when unset, so once the PR account has them, dropping one marker makes
them PR-gate-eligible.

**Genuine remaining gap:** evaluator `evaluate()` submissions (~20 existing
tests). Same pattern, distinct API surface; not yet written.

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
