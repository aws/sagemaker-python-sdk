"""Run the rollout entrypoint (main.run_rollout) end to end against your own model server.

Exercises everything the deployed agent does except the container and the SageMaker AI runtime:
payload parsing, benchmark loading, the agent loop (through the LOCAL_TEST client), and scoring.
The payload has the same shape the runtime sends: the prompt is a JSON string, encoded exactly
as scripts/preprocess.py writes it (not-applicable ground-truth values as JSON null).

Run:
  LOCAL_TEST=1 LOCAL_ENDPOINT=http://<host>:8080/v1 MODEL_NAME=<served-model-name> \
  PYTHONPATH=src python3 scripts/test_invoke_local.py [--debug]
"""

import json
import math
import os
import sys


def _nan_to_null(value):
    if isinstance(value, dict):
        return {k: _nan_to_null(v) for k, v in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main():
    if "--debug" in sys.argv:
        os.environ["DEBUG"] = "1"  # before importing main, so AgentConfig picks it up
    if not all(os.environ.get(k) for k in ("LOCAL_TEST", "LOCAL_ENDPOINT", "MODEL_NAME")):
        print("Set LOCAL_TEST=1, LOCAL_ENDPOINT, and MODEL_NAME to run against your server.")
        return 1

    from sopbench_agentcore.main import run_rollout
    from sopbench_agentcore.samples import load_instance

    instance = _nan_to_null(load_instance())
    payload = {
        "prompt": json.dumps(instance, allow_nan=False),
        "inferenceParams": {
            "model": os.environ.get("MODEL_NAME", ""),
            "temperature": float(os.environ.get("TEMPERATURE", "1.0")),
            "topP": float(os.environ.get("TOP_P", "1.0")),
            "maxTokens": int(os.environ.get("MAX_TOKENS", "4096")),
        },
        "metadata": {},
    }
    print(f"endpoint={os.environ['LOCAL_ENDPOINT']}  model={payload['inferenceParams']['model']}")
    result = run_rollout(payload)
    print(json.dumps(result, indent=2, default=str))

    if result.get("status") != "success":
        print(f"FAIL  rollout failed: {result.get('error')}")
        return 3
    print(f"PASS  reward={result['reward']} dense={result['dense']} TSR={result['binary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
