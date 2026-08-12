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
| `test_nova_trainers.py` | `::test_sft_trainer_nova_workflow`, `::test_rlvr_trainer_nova_workflow`, `test_sft_trainer_serverful_smtj.py` |

`recipe_cases.py` holds the cases every recipe trainer shares (minimal submit,
validation dataset, dataset override, output path, serverful compute, and the two
negative cases). Each per-trainer class subclasses `RecipeTrainerCases` and sets
`TRAINER`, so a new trainer is a two-line file. Override the class attributes only
where the trainer genuinely differs:

* `EXTRA_KWARGS` — required constructor args (RLAIF's reward model/prompt)
* `SUPPORTS_SERVERFUL = False` — trainer takes no `compute` (RLAIF)
* `SUPPORTS_TRAINING_TYPE = False` — no LoRA/full distinction (CPT)

It is deliberately not named `test_*` so pytest does not collect the base class.

## Coverage of every `gpu_intensive` test

The rule: **a deep test belongs off the PR gate only if this suite covers the same
code path.** There are 46 `gpu_intensive` tests in `tests/integ/train`; the table
below accounts for all of them.

### Covered by this suite

| Deep test | Shallow equivalent |
|---|---|
| `test_model_trainer.py` — 8 tests (tar source, py/sh entry, MPI, torchrun, HP json/yaml, custom driver) | `test_model_trainer.py` — `TestSourceCodePackaging`, `TestPayloadShaping`, `TestComputeConfiguration` |
| `test_sft_trainer_integration.py::test_sft_trainer_lora_complete_workflow` | `test_minimal_request_is_accepted` + `test_mlflow_resource_arn` |
| `::test_sft_trainer_with_validation_dataset` | `test_with_validation_dataset` |
| `::test_sft_trainer_lora_with_sequence_length` | `test_sft_trainer.py::test_sequence_length_is_accepted` |
| `::test_sft_trainer_nova_workflow` | `test_nova_trainers.py::test_nova_sft_is_accepted` |
| `test_dpo_trainer_integration.py` — both tests | `test_dpo_trainer.py` (inherits the shared cases) |
| `test_rlaif_trainer_integration.py::test_rlaif_trainer_lora_complete_workflow` | `test_minimal_request_is_accepted` |
| `::test_rlaif_trainer_with_custom_reward_settings` | `test_rlaif_trainer.py::test_reward_prompt_as_arn` |
| `::test_rlaif_trainer_continued_finetuning` | `::test_continued_finetuning_from_model_package` |
| `test_rlvr_trainer_integration.py::test_rlvr_trainer_lora_complete_workflow` | `test_minimal_request_is_accepted` |
| `::test_rlvr_trainer_with_custom_reward_function` | `test_rlvr_trainer.py::test_custom_reward_function_arn` |
| `::test_rlvr_trainer_with_lambda_arn_auto_creates_evaluator` | `::test_custom_reward_function_lambda_arn` |
| `::test_rlvr_trainer_with_evaluator_object` | `::test_custom_reward_function_evaluator_object` |
| `::test_rlvr_trainer_nemotron_with_kl_and_recipe` | `::test_explicit_recipe_file`, `::test_recipe_and_overrides_together`, `::test_kl_and_clipping_hyperparameters` |
| `::test_rlvr_trainer_lora_with_sequence_length` | `test_sft_trainer.py::test_sequence_length_is_accepted` (same code path) |
| `::test_rlvr_trainer_nova_workflow` | `test_nova_trainers.py::test_nova_rlvr_is_accepted` |
| `test_sft_trainer_serverful_smtj.py` | `test_explicit_compute_is_accepted` (OSS/us-west-2), `test_sft_trainer.py::test_recipe_overrides_are_accepted` (the override half), `test_nova_trainers.py::TestNovaServerfulSubmission` (Nova/us-east-1) |
| `test_sft_trainer_data_mixing_integration.py` | `test_nova_data_mixing.py` |
| `test_tuner_distributed.py::test_tuner_includes_sm_drivers_channel` | `test_tuner.py::test_distributed_tuning_job_is_accepted` |
| `test_multi_turn_rl_trainer_integration.py` — 3 submit tests | `test_multi_turn_rl_trainer.py` (needs prerequisites) |
| `test_cpt_hyperpod.py` | `test_cpt_trainer.py` (needs a HyperPod cluster) |

MLflow is worth calling out: every `*_complete_workflow` deep test configures it,
so `RecipeTrainerCases` covers both forms — `test_mlflow_experiment_tracking`
(experiment/run names, always runs) and `test_mlflow_resource_arn` (tracking-server
ARN, skips when the account has no app).

### Not covered, and why

**Evaluator tests (11)** — `test_benchmark_evaluator.py`, `test_custom_scorer_evaluator.py`,
`test_mtrl_evaluator_3p_agent.py`, `test_mtrl_trainer_integration.py`. `evaluate()`
is a different API surface returning pipeline executions rather than jobs, so it
needs its own harness support. **These were already `gpu_intensive` on master, so
this PR loses no coverage** — but closing this gap is the clearest follow-up.

Worth knowing before that follow-up: five evaluator tests are **not** marked
`gpu_intensive` and each blocks on `execution.wait(..., timeout=14400)` — a 4-hour
ceiling, and a measured ~33 minutes per execution in practice:

| Test | Marks |
|---|---|
| `test_benchmark_evaluator.py::test_benchmark_evaluation_full_flow` | none |
| `test_custom_scorer_evaluator.py::test_custom_scorer_evaluation_full_flow` | `xdist_group` |
| `test_llm_as_judge_evaluator.py::test_llm_as_judge_evaluation_full_flow` | none |
| `test_llm_as_judge_base_model_fix.py::test_base_model_evaluation_uses_correct_weights` | `serial` |
| `test_llm_as_judge_base_model_fix.py::test_base_model_false_still_works` | `serial` |

They run on master's gate too, so this PR does not add them — but they now dominate
its wall clock. Measured on a full gate run: **201 of 204 tests finished in ~7
minutes, and these held the run open for another 40+** before it was killed. The
whole shallow suite costs less than any one of them.

Marking them is not a call this PR makes, because unlike every other
`gpu_intensive` test they have no shallow counterpart yet — marking them would
remove coverage, which is exactly what the rule above forbids. The right order is:
add evaluator support to the harness, then mark them. Until then the gate is
bounded by evaluation-pipeline latency rather than by anything in this suite.

**HyperPod (3)** — `test_nova_sft_hyperpod.py`, `test_sft_data_mixing_hyperpod.py`,
`test_cpt_data_mixing_hyperpod.py`. HyperPod submits to a pre-provisioned cluster
rather than through `CreateTrainingJob`, so the pattern does not apply.
`test_cpt_trainer.py` is written in the shallow style and activates when
`SHALLOW_HYPERPOD_CLUSTER` is set.

### Tests this PR newly marks

Only these 10 gained `gpu_intensive` here — the 8 in `test_model_trainer.py`,
`test_sft_trainer_lora_with_sequence_length`, and
`test_tuner_includes_sm_drivers_channel`. Everything else in the table above was
already marked on master.

**Do not add `gpu_intensive` to a deep test unless a shallow test covers the same
path**, or the PR gate silently loses coverage.

### Fixtures that skip rather than create

`mlflow_arn`, `reward_lambda_arn`, `reward_evaluator` and `nova_reward_function_arn`
only *look up* their resources and skip when absent. The deep suite's equivalents
create them (IAM roles, Lambdas, MLflow apps, registry entries) — durable side
effects that a fast PR-gate suite should not perform.

### Fixtures that derive rather than hardcode

The Nova tests (`us_east_1`) build every S3 path from `default_bucket()` and
resolve the reward function from the calling account's own hub, rather than naming
the resources the deep Nova tests use.

This is not stylistic. The deep tests hardcode
`s3://sagemaker-us-east-1-784379639078/...`, which other accounts cannot read —
verified: `AccessDenied` on `ListObjectsV2` from 729646638167. A hardcoded path
means the test only runs in one account and fails everywhere else, which is how
these five ended up never having been executed. `test_sft_trainer_serverful_smtj.py`
already takes the derived approach (`training_resources`); these follow it, and
upload the Nova-shaped sample data the deep suite already ships
(`tests/data/train/sft_smtj_sample_data.jsonl`) rather than adding a second copy.

Two region constraints are worth knowing before adding a Nova test, both verified
against the service:

* the model package group must be in the **job's** region — passing the us-west-2
  ARN from `MODEL_PACKAGE_GROUP` is rejected with `Model package group ARN region
  'us-west-2' does not match expected region 'us-east-1'`, so Nova files use
  `NOVA_MODEL_PACKAGE_GROUP` (a bare name, which resolves per-session);
* an S3 input must be in the job's region, so `nova_rlvr_data_uri` copies the
  us-west-2 RLVR dataset into the us-east-1 bucket rather than referencing it.

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
