"""SOP-Bench rollout agent on Amazon Bedrock AgentCore, wired to SageMaker AI MTRL.

`invoke` is the AgentCore entrypoint. `@sagemaker_rft_handler` reports the result to the
SageMaker AI Job Runtime:
  * {"reward": ...}            -> CompleteRollout + UpdateReward
  * {"status": "error", ...} or an exception
                               -> CompleteRollout(status="failed") + UpdateReward(0.0).
    The trainer retries failed rollouts up to the `rollout_max_retries` hyperparameter.
"""

import functools
import json
import logging
import os
import re
import time

from bedrock_agentcore import BedrockAgentCoreApp
from sagemaker.train.rft import sagemaker_rft_handler

from .config import AgentConfig
from .reward import compute_reward, field_vocabulary
from .tools import create_openai_tools, make_tool_executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()


@functools.lru_cache(maxsize=None)
def _load_benchmark(name: str):
    """Load a benchmark once per process. SOP-Bench's tools are stateless lookups."""
    from amazon_sop_bench.benchmarks import load_benchmark

    return load_benchmark(name)


def _parse_instance(payload: dict) -> dict:
    """Return the task instance {benchmark, task_input, ground_truth}; raise if malformed.

    The training prompt is a JSON string (see scripts/preprocess.py). The instance may also be
    nested under "instance", either inside the prompt or at the top level of the payload.
    """
    raw = payload.get("prompt")
    instance = payload.get("instance")
    if raw is None and isinstance(instance, dict):
        # SDK RolloutRequest shape: the dataset row, e.g. {"prompt": "<json>"}.
        raw = instance.get("prompt", instance)
    if raw is None:
        raise ValueError("payload has no 'prompt'")
    prompt = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(prompt, dict) and "task_input" not in prompt and "instance" in prompt:
        prompt = prompt["instance"]
    if not isinstance(prompt, dict):
        raise ValueError(f"prompt must be a JSON object, got {type(prompt).__name__}")

    task_input = prompt.get("task_input")
    if isinstance(task_input, str):
        task_input = json.loads(task_input)
    ground_truth = prompt.get("ground_truth")
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    if not isinstance(task_input, dict) or not task_input:
        raise ValueError("prompt has no 'task_input' object")
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError("prompt has no 'ground_truth' object")
    return {
        "benchmark": prompt.get("benchmark"),
        "task_input": task_input,
        "ground_truth": ground_truth,
    }


def _inference_params(payload: dict, config: AgentConfig) -> dict:
    """Per-rollout sampling params from the runtime (camelCase or snake_case), else config."""
    ip = payload.get("inferenceParams") or payload.get("inference_params") or {}

    def pick(*keys, default):
        for k in keys:
            if ip.get(k) not in (None, ""):
                return ip[k]
        return default

    return {
        "model": pick("model", default=config.model_name or "default"),
        "temperature": pick("temperature", default=config.temperature),
        "top_p": pick("topP", "top_p", default=config.top_p),
        "max_tokens": pick("maxTokens", "max_tokens", default=config.max_tokens),
    }


def _resolve_inference_endpoint(payload: dict, config: AgentConfig) -> str:
    """The runtime injects a per-rollout endpoint in metadata.endpoint; prefer it."""
    meta = payload.get("metadata") or {}
    return (meta.get("endpoint") or config.rft_runtime_endpoint).rstrip("/")


def run_rollout(payload: dict) -> dict:
    """Run one rollout and score it. Returns {"reward", ...} or {"status": "error", ...}.

    Decorator-free so it can be tested locally (LOCAL_TEST) without AgentCore or the runtime.
    """
    # [rollout-trace] stage markers: the platform fails a trajectory if no sampling call arrives
    # within a time window, and these show the last stage reached before a stall.
    t0 = time.monotonic()
    meta = (payload.get("metadata") if isinstance(payload, dict) else None) or {}
    # Same key lookup as the SDK's RolloutFeedbackClient; the runtime sends rolloutId.
    trajectory_id = meta.get("trajectory_id") or meta.get("trajectoryId") or meta.get("rolloutId")
    logger.info("[rollout-trace] RECEIVED trajectory_id=%s", trajectory_id)
    try:
        config = AgentConfig.from_env()
        instance = _parse_instance(payload)
        benchmark_name = instance["benchmark"] or config.benchmark_name
        if benchmark_name != config.benchmark_name:
            raise ValueError(
                f"prompt benchmark {benchmark_name!r} != BENCHMARK_NAME {config.benchmark_name!r}"
            )
        params = _inference_params(payload, config)
        # LOCAL_TEST calls LOCAL_ENDPOINT instead (see _runtime_client).
        endpoint = None if config.local_test else _resolve_inference_endpoint(payload, config)
        benchmark = _load_benchmark(benchmark_name)
        logger.info(
            "[rollout-trace] stage=ready benchmark=%s endpoint=%s t=%.2fs",
            benchmark_name,
            endpoint,
            time.monotonic() - t0,
        )
        final_output, messages = _run_openai(
            benchmark, instance["task_input"], params, config, endpoint, trajectory_id
        )
        result = compute_reward(
            messages, instance["ground_truth"], field_vocabulary(benchmark_name)
        )
        result["status"] = "success"
        if config.debug:
            logger.info(
                "rollout trajectory_id=%s reward=%s dense=%s binary=%s assistant_turns=%s",
                trajectory_id,
                result["reward"],
                result["dense"],
                result["binary"],
                sum(1 for m in messages if m.get("role") == "assistant"),
            )
        return result
    except Exception as e:  # noqa: BLE001 - every failure is reported as a failed rollout
        logger.exception("rollout failed trajectory_id=%s", trajectory_id)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def _runtime_client(config: AgentConfig, endpoint: str):
    """OpenAI-compatible client for the SageMaker AI Job Runtime (or LOCAL_ENDPOINT)."""
    from openai import OpenAI

    if config.local_test:
        base_url = os.environ.get("LOCAL_ENDPOINT", "http://localhost:8080/v1")
        # Placeholder key: local OpenAI-compatible servers ignore it.
        return OpenAI(base_url=base_url, api_key="local")  # pragma: allowlist secret
    from sagemaker.core.token_generator import generate_token
    from sagemaker.train.rft import get_inference_headers

    # Token region: AWS_REGION, else the region in the runtime endpoint host
    # (job-runtime.sagemaker.<region>.api.aws).
    match = re.search(r"\.sagemaker\.([a-z0-9-]+)\.", endpoint)
    region = config.region or (match.group(1) if match else "")
    if not region:
        raise ValueError("cannot determine the AWS region for the Job Runtime token")
    t0 = time.monotonic()
    logger.info("[rollout-trace] stage=generating_token region=%s", region)
    api_key = generate_token(region=region)
    logger.info("[rollout-trace] stage=token_generated t=%.2fs", time.monotonic() - t0)
    return OpenAI(
        base_url=endpoint + "/v1",
        api_key=api_key,
        default_headers=get_inference_headers(),
        max_retries=5,  # ride out HTTP 429 throttling under high rollout concurrency
        timeout=300.0,
    )


def _run_openai(benchmark, task_input, params, config, endpoint, trajectory_id):
    from .openai_agent import run_openai_agent

    client = _runtime_client(config, endpoint)
    tools = create_openai_tools(benchmark)
    logger.info("[rollout-trace] stage=agent_start n_tools=%s", len(tools))
    return run_openai_agent(
        task_input,
        benchmark.sop_text,
        tools,
        make_tool_executor(benchmark),
        client,
        params["model"],
        max_turns=config.max_turns,
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        top_p=params["top_p"],
        debug=config.debug,
        trajectory_id=trajectory_id,
        stream=config.stream,
        verbose_trace=config.verbose_trace,
    )


@app.entrypoint
@sagemaker_rft_handler
def invoke(payload: dict) -> dict:
    """AgentCore entrypoint; @sagemaker_rft_handler reports the result to the trainer."""
    return run_rollout(payload)


if __name__ == "__main__":
    app.run()
