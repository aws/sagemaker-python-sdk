# Add Telemetry to a SageMaker Python SDK Feature

## Overview

This SOP instruments a SageMaker Python SDK V3 feature with usage telemetry using the
SDK's standard `@_telemetry_emitter` decorator from `sagemaker.core.telemetry`. It is
intended for partner teams who contribute features and need a clear, consistent,
low-overhead way to add telemetry to their own public APIs.

Use this SOP when a new (or existing) public method or module-level function in one of the
V3 modules (`sagemaker-core`, `sagemaker-train`, `sagemaker-serve`, `sagemaker-mlops`)
should report usage. The outcome is: the feature's public entry points carry the
`@_telemetry_emitter(Feature.<X>, "<name>")` decorator, a valid `Feature` is used (creating
one if none fits), matching unit tests exist, and the module builds green — without changing
the feature's behavior.

Do NOT use this SOP to design new telemetry infrastructure or to emit metrics to a custom
backend; it only applies the existing SDK telemetry contract to feature code.

## Parameters

- **target_path** (required): Repo-relative path to the feature's Python file or package
  directory to instrument (e.g., `sagemaker-train/src/sagemaker/train/ai_registry/dataset.py`).
- **feature_name** (required): The telemetry feature to attribute usage to. Either the name
  of an existing `Feature` enum member (e.g., `MODEL_CUSTOMIZATION`, `FEATURE_STORE`,
  `MODEL_TRAINER`) or a new `UPPER_SNAKE_CASE` name to create if no existing member fits.
- **entry_points** (optional): Explicit list of public entry points to instrument, each as
  `Class.method` or `module.function`. If omitted, the SOP detects the public entry points
  in `target_path`.
- **telemetry_params** (optional): Granular parameters to capture per entry point, each as
  a `(name, type)` pair where `type` is one of `ATTR_VALUE`, `ATTR_EXISTS`, `ATTR_CALL`,
  `KWARG_VALUE`, `KWARG_EXISTS` (from `TelemetryParamType`).
- **workspace_root** (optional, default: current directory): Absolute path to the
  `sagemaker-python-sdk` repository root that contains the `sagemaker-core` module.

**Constraints for parameter acquisition:**
- If all required parameters are already provided, You MUST proceed to the Steps
- If any required parameters are missing, You MUST ask for them before proceeding
- When asking for parameters, You MUST request all parameters in a single prompt
- When asking for parameters, You MUST use the exact parameter names as defined

## Steps

### 1. Locate the feature and its public entry points
Open `target_path` under `workspace_root` and determine the set of public entry points to
instrument. A public entry point is a user-facing method or module-level function that a
customer calls directly.

**Constraints:**
- You MUST resolve `target_path` relative to `workspace_root` and read the file(s) before deciding anything, because instrumenting the wrong functions produces misleading telemetry.
- If `entry_points` is provided, You MUST instrument exactly those and no others.
- If `entry_points` is not provided, You MUST select only public entry points and You MUST NOT select names beginning with an underscore (private/internal helpers) because decorating internal helpers inflates and double-counts usage when public methods call them.
- You MUST exclude `__init__`, simple property getters/setters, and trivial pass-through wrappers, because they do not represent a feature invocation.
- You MUST produce an explicit list of fully-qualified targets in the form `Class.method` or `module.function` and confirm it is non-empty before advancing.

### 2. Resolve the Feature enum value
Decide which `Feature` enum member attributes this usage. Use an existing member when one
fits; otherwise create a new one.

**Constraints:**
- You MUST read `sagemaker-core/src/sagemaker/core/telemetry/constants.py` (the `Feature` enum) and the `FEATURE_TO_CODE` dictionary in `sagemaker-core/src/sagemaker/core/telemetry/telemetry_logging.py` before deciding.
- If `feature_name` matches an existing `Feature` member, You MUST reuse it and You MUST NOT create a duplicate, because duplicate members fragment the usage data for one feature across two codes.
- If `feature_name` does not match an existing member, You MUST add it in TWO places using the same next unused integer: (a) the `Feature` enum in `constants.py`, and (b) the `FEATURE_TO_CODE` dictionary in `telemetry_logging.py`.
- You MUST NOT add the new member to only one of those two locations, because the emitter looks up `FEATURE_TO_CODE[str(feature)]` and a missing entry raises `KeyError` that the non-blocking wrapper swallows, so telemetry silently never emits.
- You MUST NOT reuse or renumber an existing integer code, because codes are a stable contract consumed by downstream telemetry analysis.
- Validation: You MUST confirm the chosen `Feature` member exists in both the enum and `FEATURE_TO_CODE` (for a new feature, that both edits are present with matching integers) before advancing.

### 3. Apply the `@_telemetry_emitter` decorator to each entry point
Add the required import(s) and decorate every target from Step 1.

**Constraints:**
- You MUST import the decorator with `from sagemaker.core.telemetry import Feature, _telemetry_emitter`.
- If `telemetry_params` is provided, You MUST additionally import `from sagemaker.core.telemetry.telemetry_logging import TelemetryParamType`.
- You MUST decorate each target as `@_telemetry_emitter(Feature.<FEATURE_NAME>, "<func_name>")`, where `<func_name>` is `"<Class>.<method>"` for methods and `"<function>"` or `"<module>.<function>"` for module-level functions, matching the existing convention in `sagemaker-mlops/src/sagemaker/mlops/feature_store/` and `sagemaker-train/src/sagemaker/ai_registry/`.
- If `telemetry_params` is provided, You MUST pass it as the `telemetry_params=` keyword argument using the `(name, TelemetryParamType.<TYPE>)` tuple form.
- You MUST place the `@_telemetry_emitter` decorator as the outermost decorator unless another decorator must wrap it, because ordering changes which call is measured.
- You MUST NOT change the decorated function's signature, arguments, return value, or logic, because this SOP adds observability only and must not alter feature behavior.
- You MUST NOT add any code that can raise outside the decorator, and You MUST NOT catch or re-raise telemetry errors yourself, because the emitter already guarantees telemetry never blocks the SDK call and must return the underlying result unchanged.
- You MUST NOT decorate the same function more than once, because duplicate emitters double-count usage.
- Validation: You MUST confirm every target now carries exactly one `@_telemetry_emitter` decorator and the imports resolve.

### 4. Add or extend unit tests
Create or update unit tests that assert the telemetry decorator is applied and, when
`telemetry_params` is used, that the expected parameters are emitted.

**Constraints:**
- You MUST mirror the patterns in `sagemaker-core/tests/unit/telemetry/test_granular_telemetry.py` (use `unittest.mock` `Mock`/`patch`, assert on the emitted `&x-...` query parameters and the `Feature` used).
- You MUST place tests in the same module's `tests/unit/` telemetry test location as the code you changed.
- If you created a new `Feature` in Step 2, You MUST add an assertion that the new member is present in both the `Feature` enum and `FEATURE_TO_CODE`, because that mapping is the silent-failure point.
- You MUST NOT weaken or delete existing telemetry assertions to make new tests pass, because that erases regression coverage.
- Validation: the new/updated tests MUST exist and MUST be runnable by the module's test runner.

### 5. Build and verify the module
Run the affected module's build and tests to confirm the change compiles, the tests pass,
and telemetry does not block the instrumented calls.

**Constraints:**
- You MUST run the module's unit tests (for `sagemaker-core`, `python -m pytest sagemaker-core/tests/unit/telemetry/`; for a partner module, the equivalent unit-test target) and You MUST NOT proceed while any telemetry-related test fails, because shipping unverified telemetry can emit incorrect usage data.
- You MUST confirm the instrumented entry points still return their original results when invoked in the tests, because a broken non-blocking guarantee would surface as an altered or missing return value.
- If verification fails, You MUST diagnose and fix the root cause and re-run, and You MUST NOT mark the SOP complete with failing tests, because a green telemetry test suite is the completion criterion for this SOP.
- Validation: the build/test run MUST complete with the telemetry tests green.

### 6. Summarize the changes
Produce a concise summary of what was instrumented for the requesting partner team.

**Constraints:**
- You MUST list each instrumented entry point with its `Feature` and `func_name`, whether a new `Feature` was created (with its integer code), any `telemetry_params` added, and the files changed.
- You MUST state the test command that was run and its result.
- You MUST NOT run `git commit`, `git push`, or create a code review unless the user explicitly asks, because publishing changes is the author's decision.

## Examples

### Example 1: Instrument an existing feature with an existing Feature
**Input:**
- target_path: `sagemaker-mlops/src/sagemaker/mlops/feature_store/feature_utils.py`
- feature_name: `FEATURE_STORE`

**Expected Behavior:**
The agent detects the public functions (e.g., `ingest_dataframe`, `get_feature_group_as_dataframe`),
decorates each with `@_telemetry_emitter(Feature.FEATURE_STORE, "<function>")`, reuses the
existing `FEATURE_STORE` member, adds/extends telemetry unit tests, runs them green, and
summarizes the changes.

### Example 2: New feature with granular parameters
**Input:**
- target_path: `sagemaker-train/src/sagemaker/train/my_feature/client.py`
- feature_name: `MY_FEATURE`
- entry_points: ["MyClient.run", "MyClient.submit"]
- telemetry_params: [("instance_type", "KWARG_VALUE"), ("networking", "ATTR_EXISTS")]

**Expected Behavior:**
The agent adds `Feature.MY_FEATURE = 22` to `constants.py` and `str(Feature.MY_FEATURE): 22`
to `FEATURE_TO_CODE`, decorates `MyClient.run` and `MyClient.submit` with the granular
`telemetry_params`, adds unit tests asserting `x-instanceType` and `x-hasNetworking` are
emitted plus the new-member/mapping assertions, runs the tests green, and summarizes.

## Troubleshooting

### Telemetry never emits at runtime
If the decorator is applied but no telemetry appears:
- Confirm the `Feature` member exists in BOTH the `Feature` enum and `FEATURE_TO_CODE` with the same integer; a missing `FEATURE_TO_CODE` entry raises a swallowed `KeyError`.
- Confirm a `sagemaker_session` is resolvable in the call path, because the emitter only fires when a session is available.

### ImportError on `_telemetry_emitter` or `TelemetryParamType`
- Import `Feature` and `_telemetry_emitter` from `sagemaker.core.telemetry`.
- Import `TelemetryParamType` from `sagemaker.core.telemetry.telemetry_logging` (it is not re-exported from the package root).

### Tests fail with an unexpected return value
- Verify no logic was changed and the `@_telemetry_emitter` decorator is the only addition; the emitter must return the wrapped function's result unchanged.

### Cannot determine which functions to instrument
- Ask the partner team for the `entry_points`, or instrument only the documented public API surface and exclude underscore-prefixed helpers.
