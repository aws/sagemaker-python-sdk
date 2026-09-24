---
description: Reproduce, diagnose, fix, and test a GitHub issue on a reviewable local branch
argument-hint: <issue-number | issue-URL> [extra context]
allowed-tools: Bash(gh issue view:*), Bash(git checkout:*), Bash(git switch:*), Bash(git branch:*), Bash(git status:*), Bash(git diff:*), Bash(git add:*), Bash(python -m pytest:*), Bash(python:*), Bash(export:*), Read, Grep, Glob, Edit, Write, Agent, Task, TodoWrite
---

# Fix a SageMaker Python SDK issue

Drive an issue from `aws/sagemaker-python-sdk` to a **reviewable local branch**: reproduce it,
find the root cause, write a v3-correct fix, and back it with tests. **Stop before pushing or
opening a PR** — leave the branch and a written summary for the human to review.

## Input

`$ARGUMENTS` is the GitHub issue number or URL, optionally followed by extra context/hints.
If it's empty, ask for an issue number or URL and stop.

## Repo facts (this monorepo)

- Four namespace packages: `sagemaker-core`, `sagemaker-train`, `sagemaker-serve`,
  `sagemaker-mlops`, each with `src/` and `tests/unit/`.
- **v3 is the golden rule.** Follow `AGENTS.md` — never introduce banned v2 patterns
  (`Estimator`, framework estimators, `estimator.fit`, `sagemaker.model.Model`,
  `sagemaker.workflow.*`, etc.) into fix code, tests, or examples.
- **Running unit tests** (v2 install shadows the v3 namespace, so PYTHONPATH is required).
  From the package dir, e.g. for `sagemaker-train`:
  ```bash
  cd sagemaker-train
  export PYTHONPATH="$PWD/../sagemaker-core/src:$PWD/src:$PWD/../sagemaker-serve/src:$PWD/../sagemaker-mlops/src"
  python -m pytest tests/unit/<path> -q
  ```
  Adjust the first `$PWD/src` to whichever package you're testing. Pydantic v1
  `@validator`/`@root_validator` deprecation warnings are expected noise.

## Steps

Track progress with TodoWrite. Do the steps in order; do not skip reproduction.

### 1. Fetch the issue — including the full comment thread
Run `gh issue view <n> --repo aws/sagemaker-python-sdk --comments` to pull the title, body,
labels, **and the entire discussion**. Extract: the failing API/module, the reported behavior
vs. expected, any repro snippet, versions, and stack traces. State which of the four packages
it most likely lives in.

**Read the comments before writing any code** and check the maintainer stance. STOP and report
back (do not fix) if the thread shows any of:
- a maintainer **reclassified** it (e.g. bug → feature request) or **declined / won't-fix** it;
- it targets an **unsupported configuration** (e.g. Windows — the SDK supports Unix/Linux/Mac
  only) where maintainers have said they won't support it;
- it's a **duplicate**, already fixed, or there's an **existing open PR** for it;
- it's a pure question or docs request.

In those cases, summarize the maintainer position and ask how to proceed rather than assuming
the triage label (e.g. a spreadsheet "bug" tag) is authoritative — the issue thread wins over
external triage.

### 1b. Scope gate — is this a large feature request?
Before starting work, classify the issue as **in-scope** (a bug fix, a type/annotation or docs
correction, or a small self-contained enhancement — the kind of change that lands in roughly one
focused PR touching a bounded set of files) vs. a **large feature request**.

Treat it as a **large feature request** — and STOP — when any of these hold:
- it asks for a **new public API, class, or capability** rather than fixing existing behavior
  (e.g. "add support for X", "make the SDK able to Y");
- delivering it would need **design decisions, a new module, cross-package changes, or new
  external dependencies**, or would plausibly span multiple PRs;
- it's labeled `type: feature` / `feature request` (or was reclassified as one) and is not a
  trivial addition;
- the effort is clearly beyond "quick win" (e.g. the triage doc marks it larger than S), or you
  can't scope it to a concrete, bounded change after reading the code.

When it's a large feature request, **do NOT fix it. Backlog it**: report a one-paragraph summary
(what it asks for, rough scope, why it's out of scope for this quick-fix flow) and stop. Do not
create a branch. If a tracking sink is in use (e.g. the Pippin backlog doc), note it there as
"backlogged — large feature request" rather than as a completed fix. When unsure whether
something is a small enhancement or a large feature, ask rather than starting to build it.

### 1c. v2-vs-v3 relevance check (older issues)
Many older issues were filed against **v2** (they cite a v2 version like `1.x`/`2.x`, or use v2
imports such as `sagemaker.estimator`, `sagemaker.tuner`, `sagemaker.pytorch`, `sagemaker.model`,
`sagemaker.workflow.*`). This repo is **v3**. Before fixing such an issue, confirm the bug is
still relevant here:
- **Locate the equivalent v3 code** for the reported behavior (map the v2 symbol to its v3 home
  via `AGENTS.md` / `migration.md`; grep the four `src/` trees). The reporter's v2 snippet is a
  description of intent — translate it to the v3 API, don't run it verbatim.
- **STOP and report (do not fix)** if:
  - the feature was **removed in v3 with no replacement** (e.g. MXNet, Chainer, RLEstimator,
    Training Compiler) — nothing to fix here;
  - the v3 code path **does not have the bug** — the defect was already fixed or the v3 redesign
    doesn't exhibit it. Show the v3 code that proves it's fine.
- **Only proceed** once you've confirmed the defect actually reproduces (or is clearly present by
  inspection) in the **v3** code — not merely that it existed in v2. Reproduce against v3 in
  step 3, and write the fix and tests against the v3 module.

### 2. Create a working branch
From a clean tree, `git switch -c fix/issue-<n>-<slug>`. If the tree is dirty, stop and report
it rather than mixing unrelated changes. Never work on `master`.

### 3. Reproduce (fork subagent)
Dispatch a **fork** subagent to reproduce the bug. It should write a minimal repro (a script or
a temporary failing test), run it with the PYTHONPATH setup above, and report the exact
observed failure (traceback, wrong value) — or report that it could NOT reproduce, with what it
tried. Do not proceed to a fix until repro is confirmed **or** you've explicitly decided the
issue is valid-by-inspection (e.g. an obvious logic error) and said why.

### 4. Explore the code (Explore subagent)
Dispatch an `Explore` (or fork) subagent to map the code around the failure: the entry point,
call chain to the fault, related tests in `tests/unit/`, and any nearby patterns/conventions the
fix must match. Ask it for `file:line` anchors and the conclusion, not file dumps.

### 5. Write the fix
Edit the source to address the **root cause**, not the symptom. Keep the change focused and
match surrounding style. No hardcoded account IDs, ARNs, regions, or bucket names. Keep it v3.

### 6. Add / update tests
Add or update unit tests under the owning package's `tests/unit/` that **fail without the fix
and pass with it**. Run the relevant subset with the PYTHONPATH setup and paste the real result.
Only consider an integ test (`tests/integ/`) when the bug genuinely needs AWS to exercise; if
so, write it but note it's CI-only and don't require it to pass locally.

### 6b. Validate at the strongest feasible level (before AND after)
A passing mocked unit test is the floor, not proof. Validate the fix as close to the real failure
as you can, and demonstrate the defect **before** the fix and its absence **after**:
- **Reproduce the actual reported error, deterministically and offline, when possible.** If the
  bug is about an external contract, exercise that contract directly rather than asserting on
  mocks: e.g. validate params against the real boto **service model** with
  `botocore.validate.ParamValidator` (offline, no network/credentials); reproduce a type-checker
  error by running the real tool (`mypy`) against a translated repro; run the real function for
  in-memory logic. State which level you reached and what you could NOT cover.
- **Prove "fails without the fix" against a real baseline.** `git stash push <file>` only stashes
  **uncommitted** changes — once the fix is committed on the branch it stashes nothing and you'll
  falsely "confirm" a fail. Use `git checkout master -- <file>` as the negative control, then
  restore. BUT: if you haven't committed the fix yet, the branch `HEAD` still equals `master`, so
  `git checkout HEAD -- <file>` will restore the **unfixed** file and silently wipe your edit.
  Safe recipe: **commit the fix first**, then `git checkout master -- <file>` (run the test → it
  fails), then `git checkout HEAD -- <file>` (now HEAD has the fix → it passes). After any such
  dance, re-grep the file to confirm your fix is actually present before continuing.
- **Never let `|| echo`, `|| true`, or a grep-with-no-match masquerade as success.** Check the
  tool's real exit code and read its actual output; a command that errored out is not a pass.
- **Live AWS calls** only when no offline reproduction exists. Prefer ReadOnly/least-privilege,
  create only resources you clean up, confirm the account is non-production, and never run
  destructive operations. Most SDK bugs (serialization, typing, arg wiring, in-memory logic) are
  fully provable offline — reach for a live call last, not first.
- Some things genuinely can't be validated here (e.g. Windows-only behavior with no Windows host,
  a real distributed training run). Say so plainly rather than implying coverage you don't have.

### 7. Verify
Run the affected package's relevant unit subset once more and confirm green.

### 8. Independent review (fix-reviewer subagent)
Dispatch the **`fix-reviewer`** subagent (fresh agent, not a fork — it must judge
independently). Give it the issue number/URL and the branch name. It re-derives correctness
from the issue and diff, hunts backwards-incompatibilities (callers/readers, serialized
formats, semantics for existing inputs), and independently reproduces + runs adversarial
checks. Relay its verdict. If it returns `REQUEST CHANGES` with a BLOCKER/MAJOR finding,
address it and re-run the reviewer before summarizing.

### 9. Summarize
Report:
- **Root cause** — what was actually wrong, with `file:line`.
- **Fix** — what changed and why it's correct.
- **Tests** — what you added and the pass/fail output (verbatim if anything failed).
- **Review** — the `fix-reviewer` verdict and any findings you actioned.
- **Branch** — the branch name and `git diff --stat`.
- **Next step** — that it's ready for human review; do NOT push or open a PR.

## Constraints
- You MUST stop at the local branch. Do NOT `git push`, `gh pr create`, or otherwise publish.
- You MUST report test failures faithfully, with output — never claim green without running.
- You MUST NOT introduce v2 patterns; self-check the diff against `AGENTS.md` before summarizing.
- If reproduction fails, say so plainly and ask how to proceed instead of guessing a fix.
