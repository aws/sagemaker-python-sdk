# GitHub Copilot instructions — Amazon SageMaker Python SDK

Guidance for [GitHub Copilot](https://docs.github.com/en/copilot) working in
**this repository** — the source of the Amazon SageMaker Python SDK.

This repository keeps its AI-agent guidance in a single source of truth,
[`AGENTS.md`](../AGENTS.md), following the [AGENTS.md](https://agents.md) convention.
Copilot instructions do not support file imports, so **read
[`AGENTS.md`](../AGENTS.md) at the repository root and follow it**.

Key points from that file (see `AGENTS.md` for the authoritative, complete version):

- **v3 by default.** The current major version is v3 (`pip install sagemaker`). SDK v3 is a
  modular redesign and is **not** backward compatible with v2. Generate v3 patterns in all
  example code, docstrings, tests, and docs unless v2 is explicitly in scope.
- **SDK-first.** Use the SageMaker Python SDK v3 as the primary interface (e.g.
  `sagemaker.train.ModelTrainer`, `sagemaker.serve.ModelBuilder`); do not drop to raw
  `boto3`, the AWS CLI, or hand-rolled scripts unless the SDK genuinely does not cover the
  task.
- **No banned v2 patterns** (e.g. `sagemaker.estimator.Estimator`, `estimator.fit(...)`,
  framework estimator classes, `sagemaker.model.Model`) in new code. See the v2 → v3
  mapping table in `AGENTS.md` and [`migration.md`](../migration.md).
- **Contributing:** add/update unit tests under `tests/unit/` for code changes, run the
  configured formatters/linters, and keep `migration.md` and docstrings consistent for
  public API changes. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).
