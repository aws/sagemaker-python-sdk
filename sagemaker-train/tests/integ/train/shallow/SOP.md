# SOP — adding or updating a fast (shallow) integ test

The step-by-step procedure for changing anything under
`sagemaker-train/tests/integ/train/shallow`. [`README.md`](./README.md) explains *why*
the suite is shaped the way it is; this file is what to do, in what order, and how to
verify each step.

## 0. Does your test belong here?

This suite submits a real `CreateTrainingJob`, asserts the service returned an ARN,
and immediately stops the job. A test belongs here if and only if what it checks is
decided **synchronously, at submit time** — payload shape, validation, permissions,
image/recipe resolution.

| You want to assert… | Put it in |
|---|---|
| the service accepted (or rejected) a payload | **here** |
| client-side config assembly, with no service call | `tests/integ/train/test_dry_run_integration.py` |
| artifacts, metrics, logs, convergence — anything needing the job to *run* | the deep suite, marked `gpu_intensive` |

A regression that makes training itself fail will still pass here. That is deliberate;
see *What these tests deliberately do NOT cover* in the README.

## 1. Decide where the test goes

| Case | Action | Cost |
|---|---|---|
| New behaviour on one trainer | Add to the matching `test_<trainer>.py` | 1 job |
| Behaviour shared by **all** recipe trainers | Add a case to `recipe_cases.py` | **1 job × every subclass** — currently 5 (SFT, DPO, RLVR, RLAIF, CPT) |
| A brand-new trainer | New `test_<trainer>.py` subclassing `RecipeTrainerCases` | the 9 shared cases |

Adding to `recipe_cases.py` multiplies. Only put a case there if it is genuinely
trainer-independent; otherwise it belongs in one file.

A new trainer file is two lines plus opt-outs:

```python
class TestMyTrainer(RecipeTrainerCases):
    TRAINER = MyTrainer
    EXTRA_KWARGS = {...}            # required constructor args, if any
    SUPPORTS_SERVERFUL = False      # trainer takes no `compute`
    SUPPORTS_TRAINING_TYPE = False  # no LoRA/full distinction
```

**Verify:** `pytest tests/integ/train/shallow --collect-only` lists your test(s) and
the total moved by the number you expect.

## 2. Write it with the harness

Never call `trainer.train()` directly — the harness is what forces `wait=False`,
stops the job in a `finally`, and holds the concurrency slot.

```python
from .harness import assert_submitted, assert_rejected, submitted, unique_name

def test_my_feature_is_accepted(sagemaker_session, train_data_uri):
    trainer = _trainer(sagemaker_session, unique_name("shallow-my-feature"), ...)
    with submitted(trainer) as job:
        assert_submitted(job)

def test_bad_input_is_rejected(sagemaker_session):
    assert_rejected(trainer, ("does not exist", "ValidationException"))
```

Rules that are not negotiable, each with the failure it prevents:

| Rule | Why |
|---|---|
| Use `submitted()` / `assert_rejected()`, never bare `train()` | Skips the stop, the slot, and the terminal-wait |
| Never pass `wait=` | Rejected with `TypeError`, so a copy-pasted `wait=True` cannot reintroduce a real training run |
| Never set `keep_alive_period_in_seconds` | A warm pool outlives the stop and keeps instances provisioned |
| Smallest instance that exercises the path (`ml.m5.large` unless a recipe needs an accelerator) | Validation is instance-type agnostic; big instances cost real money on a non-deterministic teardown |
| Resolve images with `harness.cpu_image(sagemaker_session)` | The DLC registry account differs by partition; a hardcoded URI is unusable outside one |
| Derive S3 paths from `default_bucket()`, never hardcode a bucket | A hardcoded bucket means the test only passes in one account — how five deep Nova tests ended up never running |
| Look resources up and `skip` when absent; never create them | Creating IAM roles/Lambdas/MLflow apps is a durable side effect a PR-gate suite must not have |
| Keep at least one **negative** test per feature area | Without them the suite degenerates into "any ARN is fine" and stays green on a permissive-but-wrong payload |

**Verify:** `grep -n "wait=\|keep_alive" <your file>` returns nothing.

## 3. Markers

| Marker | Effect on your shallow test |
|---|---|
| *(none)* | Runs on the PR gate in `fast-integ-tests`. **This is what you want.** |
| `us_east_1` | Removed from `fast-integ-tests`; runs in the `integ-tests-us-east-1` job instead. Needs us-east-1 test-account credentials. Use only for Nova. |
| `serial` | No effect here (the fast project runs one pytest command), but it *does* split the deep and us-east-1 projects. Don't add it. |
| `gpu_intensive` | **Never on a shallow test.** It marks deep tests *off* the gate. |

Two rules that cut both ways:

* **Do not add `gpu_intensive` to a deep test unless a shallow test covers the same
  path** — the gate silently loses coverage. Update the coverage table in the README
  when you do.
* **Register any new marker in `pyproject.toml`, not `tox.ini`.** pytest reads its
  config from `pyproject.toml` and prints
  `WARNING: ignoring pytest config in tox.ini`, so a marker declared only in
  `tox.ini` is unregistered at runtime. That matters because the gate selects with
  `-m "not gpu_intensive and not us_east_1"`: a typo'd name would put an expensive
  deep test back on the gate instead of erroring.

**Verify:** run with `-W error::pytest.PytestUnknownMarkWarning`; an unregistered
marker fails instead of warning.

## 4. Adding a new submission path? Take slots yourself

The concurrency cap is enforced inside `submitted()` and `assert_rejected()`, so a
normal test is capped automatically. If you add a path that submits **without** going
through those two — as `_tuning()` in `test_tuner.py` does, because a tuning job is
stopped via `tuner.stop_tuning_job()` — you must acquire `job_slots()` yourself and
hold them until the job is **terminal**, not until `stop()` returns.

Getting this wrong is the bug that made an earlier version peak at ~37 concurrent jobs
against a limit of 20. See *Concurrency cap* in the README.

## 5. Run it locally

**Prerequisites**

* AWS credentials for an SDK test account. The suite uses ambient credentials — no
  profile logic — and resolves the execution role through the real discovery path
  (`TrainDefaults.get_role(role=None, ...)`), so the account needs a discoverable
  SageMaker execution role.
* Region defaults to **us-west-2**; an autouse fixture pins `AWS_DEFAULT_REGION`
  unless you set it yourself.
* Recipe resolution goes through a private hub named `sdktest` (an autouse fixture
  sets `SAGEMAKER_HUB_NAME`), so the account needs it.
* `us_east_1` tests need credentials in the us-east-1 test account; their fixture
  pins the region regardless of `AWS_DEFAULT_REGION`.

**Install, the same way CI does**

```bash
cd sagemaker-core  && pip install -e '.[test]'
cd ../sagemaker-train && pip install -e '.[test]'
```

**Run**

```bash
# one test, no cap, no xdist — for debugging
SHALLOW_MAX_CONCURRENT_JOBS=0 python -m pytest \
  tests/integ/train/shallow/test_sft_trainer.py -k my_feature

# one file, gate selection
python -m pytest tests/integ/train/shallow/test_sft_trainer.py \
  -m "not gpu_intensive and not us_east_1"

# the whole suite exactly as the PR gate runs it
python -m pytest tests/integ/train/shallow -v -n 8 \
  -m "not gpu_intensive and not us_east_1" --durations 15
```

Expect ~7 minutes for the full suite at `-n 8`, and ~1–2 minutes for a single test —
dominated by the wait for the job to reach a terminal state, not by the SDK.

**Optional env vars.** Tests whose inputs are not derivable read them from the
environment and **skip** when absent, so they never fail for a missing resource:

| Variable | Gates | Default |
|---|---|---|
| `SHALLOW_MAX_CONCURRENT_JOBS` | Concurrency cap; `0` disables gating | 10 |
| `SHALLOW_HYPERPOD_CLUSTER` | `test_cpt_trainer.py` HyperPod cases | skip |
| `SHALLOW_MTRL_AGENT_ENV`, `SHALLOW_MTRL_MLFLOW_APP_ARN`, `SHALLOW_MTRL_DATASET` | `test_multi_turn_rl_trainer.py` | skip |
| `SHALLOW_MTRL_MODEL` | Multi-turn RL model id | `mock-oss-test` |

## 6. Pre-submit checklist

```bash
tox -e black-format && tox -e flake8 && tox -e pylint   # from sagemaker-train/
```

- [ ] Collection count moved by exactly what you expect (step 1).
- [ ] At least one negative test for the feature area (step 2).
- [ ] No `wait=`, no `keep_alive_period_in_seconds`, no hardcoded bucket/URI/ARN.
- [ ] Your test is unmarked unless it is genuinely us-east-1-only (step 3).
- [ ] If you marked a deep test `gpu_intensive`, the README coverage table accounts
      for it.
- [ ] The suite still passes locally at `-n 8`.

## 7. What CI will do

| Job | Project | Runs |
|---|---|---|
| `fast-integ-tests` | `…-ci-sagemaker-train-fast-integ-tests` | `python3.10 -m pytest tests/integ/train/shallow -v -n 8 -m "not gpu_intensive and not us_east_1" --durations 15`, `SHALLOW_MAX_CONCURRENT_JOBS=10`, 30-min timeout |
| `integ-tests-us-east-1` | `…-ci-integ-tests-us-east-1` | `pytest tests/integ -m "us_east_1 and not gpu_intensive …"` — where your `us_east_1` shallow tests run |
| `integ-tests` | `…-ci-sagemaker-train-integ-tests` | The deep suite. Goes through tox, and `tox.ini` passes `--ignore=tests/integ/train/shallow`, so it does **not** rerun this suite |

Two things worth knowing:

* `fast-integ-tests` only fires when `sagemaker-train` is in the change set. A
  docs-only or CDK-only change will not exercise your test.
* Neither job checks out PR code onto the GitHub runner — both start CodeBuild with
  `source-version-override`. Adding a *file* under `shallow/` is picked up
  automatically and needs no CI change.

## 8. Changing *how* the suite is invoked

The worker count, marker selection, Python version, compute size and timeout live in
`createCIShallowIntegBuildSpec` in the internal `SageMakerMLFPySDKInfraCDK` package —
**not in this repo**. Changing any of them is a CR against that package that deploys
through a pipeline, not something that merges with your PR. Two constraints there,
both learned the hard way:

* The `cpu-integ` image the project runs on has **only Python 3.10.13** under pyenv.
* Do not add `--dist loadfile`. It pins a file's tests to one xdist worker, and since
  each test holds a slot until its job is terminal (~75s), it serialized the largest
  file to 19m45s against a 30-minute timeout.

## 9. Troubleshooting

| Symptom | Cause |
|---|---|
| `ResourceLimitExceeded` / utilization above the quota | A submission path that doesn't hold a slot to terminal (step 4), or `SHALLOW_MAX_CONCURRENT_JOBS` raised |
| Test hangs ~15 min then warns | Slot-acquire timeout (900s). Something is holding slots — usually a leaked job |
| `PytestUnknownMarkWarning` | Marker registered in `tox.ini` instead of `pyproject.toml` (step 3) |
| Build passes but ran nothing | pytest exit code 5, "no tests collected" — a moved directory or mistyped marker. The fast project deliberately does **not** tolerate exit 5, so this fails the build |
| `AccessDenied` on S3 from CI but not locally | A hardcoded bucket belonging to your account (step 2) |
| `ImportError: cannot import name …` locally | Stale editable install; re-run the step-5 installs |
