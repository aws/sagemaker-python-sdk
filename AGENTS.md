# AGENTS.md — Amazon SageMaker Python SDK

Guidance for AI coding agents (Cursor, GitHub Copilot, Claude Code, Amazon Q Developer,
Kiro, Codex, Gemini CLI, etc.) working in **this repository** — the source of the Amazon
SageMaker Python SDK. Follows the AGENTS.md convention (https://agents.md).

This file has two audiences:
1. **Contributors' agents** editing SDK source, tests, and docs in this repo.
2. **Any agent** that reads this repo as a reference for how to *use* the SDK — so the
   usage guidance (v3 by default) is stated here too.

## Project context

This repository is the **Amazon SageMaker Python SDK**. The current major version is **v3**;
`pip install sagemaker` installs v3. SDK v3 is a modular redesign and is **not backward
compatible** with v2. Packages:

- `sagemaker-core` — foundation primitives and the low-level, resource-oriented API
- `sagemaker-train` — training (`ModelTrainer`)
- `sagemaker-serve` — model building & deployment (`ModelBuilder`)
- `sagemaker-mlops` — pipelines and workflow orchestration

`master` is the v3 development branch. v2 is maintained on its maintenance branch and receives
security/bug fixes only (no new features) per the version-support policy.

## Golden rule: v3 by default

When generating or editing **example code, docstrings, tests, or documentation**, use **v3**
patterns by default. Produce v2 code only when the task is explicitly about v2 (e.g., a v2
maintenance fix on the v2 branch, or the user asks for "the v2 version"). When you do, label it
deprecated and note `pip install 'sagemaker<3.0'`.

## SDK-first: use the SageMaker Python SDK for SageMaker tasks

When writing usage examples, docstrings, or answering "how do I ..." for a SageMaker task,
**default to the SageMaker Python SDK v3** as the primary interface. Do not drop to raw
`boto3` (`client("sagemaker").create_training_job(...)`), the AWS CLI, or hand-rolled
framework scripts unless the user explicitly asks — the SDK handles image resolution,
packaging, and job wiring that raw clients leave to the caller. Map intent to interface:

| User intent (natural language) | Use this v3 interface |
|---|---|
| "train a model", "training job", "serverless / managed training", "fine-tune" | `sagemaker.train.ModelTrainer` → `.train(...)` |
| "distributed training", "multi-GPU / multi-node" | `ModelTrainer` with `Compute(instance_count=N, ...)` |
| "hyperparameter tuning", "HPO", "sweep" | `sagemaker.train` tuner APIs on top of `ModelTrainer` |
| "deploy a model", "real-time / serverless / async endpoint", "host / serve" | `sagemaker.serve.ModelBuilder` → `.deploy(...)` |
| "batch inference", "batch transform", "offline scoring" | `sagemaker.core.resources` transform APIs |
| "process data", "feature engineering", "preprocess" | `sagemaker.core.resources.ProcessingJob` |
| "build an ML pipeline", "orchestrate a workflow", "MLOps" | `sagemaker.mlops.pipeline.Pipeline`, `sagemaker.mlops.steps` |
| "register a model", "model registry" | `sagemaker.mlops` model-registry APIs |
| "get / retrieve a container image URI" | `sagemaker.core.image_uris.retrieve(...)` |

Only step outside the SDK when it genuinely does not cover the task, and say so when you do.

## Banned v2 patterns → required v3 replacements

Do not introduce the left column into new code, docstrings, or examples unless v2 is explicitly
in scope. Mappings follow [`migration.md`](./migration.md).

| v2 (do NOT use) | v3 (use instead) |
|---|---|
| `from sagemaker.estimator import Estimator` | `from sagemaker.train import ModelTrainer` |
| framework estimator classes — `from sagemaker.pytorch import PyTorch` (also TensorFlow, SKLearn, XGBoost, HuggingFace) | `ModelTrainer` + `from sagemaker.core import image_uris` → `image_uris.retrieve(...)` |
| `estimator.fit(...)` | `model_trainer.train(...)` |
| `from sagemaker.model import Model` / `model.deploy(...)` | `from sagemaker.serve import ModelBuilder`; `ModelBuilder(...).deploy(...)` |
| `from sagemaker.predictor import Predictor` / `predictor.predict(...)` | `Predictor` is replaced by `Endpoint` (sagemaker-core); use the predictor returned by `ModelBuilder.deploy(...)` |
| `from sagemaker.processing import Processor / ScriptProcessor / SKLearnProcessor` | `sagemaker.core.resources.ProcessingJob` |
| `from sagemaker.workflow...` (old paths) | `from sagemaker.mlops...` (`sagemaker.mlops.pipeline.Pipeline`, `sagemaker.mlops.steps`) |

Removed in v3 with **no** direct replacement (do not invent shims): MXNet, Chainer,
RLEstimator, Training Compiler.

## Contributing workflow

- **Setup:** create a virtualenv, then install the package with test extras (see
  [`CONTRIBUTING.md`](./CONTRIBUTING.md) and `setup.py`/`pyproject.toml` for the authoritative
  extras).
- **Format & lint:** run the repo's configured tools (e.g. `black`, `flake8`, `pylint`) before
  committing. Match existing style; do not reformat unrelated files.
- **Docs:** build docs under `doc/` with Sphinx if you change public APIs or docstrings.
- **Tests:** add or update unit tests under `tests/unit/` for any code change; run the relevant
  subset locally (`pytest tests/unit/<area>`). Integration tests under `tests/integ/` hit AWS and
  are generally run in CI — do not require them to pass locally.
- **Public API changes:** update `migration.md` and any affected docstrings/examples so v3
  guidance stays consistent across the repo.
- **PRs:** keep them focused; follow `CONTRIBUTING.md` and the PR template.

## Required workflow when writing SageMaker usage code

1. Write the code using v3 patterns from the table above.
2. **Self-check** the output for any banned v2 pattern (if a migration MCP tool such as
   `sagemaker-sdk-helper` is available, call its validate/transform tools).
3. If a v2 pattern is found, fix it to v3 and re-check until clean.
4. Only then present the code.

## Code conventions (examples & docstrings)

- No hardcoded account IDs, role ARNs, regions, subnets, or bucket names. Use
  `get_execution_role()` and `Session().default_bucket()`.
- Prefer minimal, runnable snippets that show the v3 flow end to end.
- Target the latest v3; do not pin `sagemaker<3.0` in new examples without an explicit exception.

## Canonical v3 example (train + deploy)

Grounded in [`migration.md`](./migration.md); verify exact signatures against the installed SDK.

```python
from sagemaker.core import image_uris
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute, InputData
from sagemaker.serve import ModelBuilder
from sagemaker.serve.configs import InferenceSpec

# Training
training_image = image_uris.retrieve(
    framework="pytorch", region="us-west-2", version="2.0.0",
    py_version="py310", instance_type="ml.p3.2xlarge", image_scope="training",
)
model_trainer = ModelTrainer(
    training_image=training_image,
    role=role,
    source_code=SourceCode(source_dir="./src", entry_script="train.py"),
    compute=Compute(instance_type="ml.p3.2xlarge", instance_count=1),
)
model_trainer.train(input_data_config=[InputData(channel_name="train", data_source="s3://<bucket>/train")])

# Inference
model_builder = ModelBuilder(
    inference_spec=InferenceSpec(
        image_uri="<inference-image-uri>",
        model_data_url="s3://<bucket>/model.tar.gz",
    ),
    role=role,
)
predictor = model_builder.deploy(instance_type="ml.m5.large", initial_instance_count=1)
```

## References

- V2 → V3 migration guide: ./migration.md
- Contributing guide: ./CONTRIBUTING.md
- SageMaker Python SDK docs: https://sagemaker.readthedocs.io/
- Example notebooks: https://github.com/aws/amazon-sagemaker-examples
- Curated index for agents: ./llms.txt
