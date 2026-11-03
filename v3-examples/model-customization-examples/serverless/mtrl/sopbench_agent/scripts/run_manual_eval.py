"""Manual evaluation runner: score SOPBench rollouts against YOUR vLLM endpoint.

Runs the real openai_agent loop against an OpenAI-compatible vLLM server (no AgentCore, no RFT
token path) and scores each instance with the same reward.compute_reward used in training. Use
this to sanity-check the agent + reward + prompt before deploying to AgentCore.

USAGE
-----
  PYTHONPATH=src python3 scripts/run_manual_eval.py \
      --base-url http://<your-vllm-host>:8080/v1 \
      --model <model-name-as-served-by-vllm> \
      --max-turns 10

  # Multi-instance: point --data at a parquet or jsonl of instances
  # (same shape as sopbench_agentcore.samples.load_instance(): instance_id, benchmark, task_input,
  # ground_truth).
  PYTHONPATH=src python3 scripts/run_manual_eval.py \
      --base-url http://<your-vllm-host>:8080/v1 --model <model> \
      --data data/aircraft/aircraft_inspection_val.parquet \
      --num-samples 23

NOTES
-----
- api_key is sent as "EMPTY" (vLLM ignores it). Nothing is logged that contains credentials.
- Model name: pass the bare vLLM name (no "openai/" prefix).
- Scores reported: dense (per-field partial credit), binary/TSR (all fields correct), and the
  training reward (dense * 5, or -1 when the rollout never answers).
"""

import argparse
import json
import sys


def load_instances(path: str, limit: int):
    """Load instances from .json (single or list), .jsonl, or .parquet. Returns list of dicts with
    keys: instance_id, benchmark, task_input (str or dict), ground_truth (dict)."""
    instances = []
    if path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(path)
        rows = df.to_dict(orient="records")
        for row in rows:
            # A row may itself be the instance, or carry a JSON 'prompt' column.
            if "prompt" in row and isinstance(row["prompt"], str):
                try:
                    inst = json.loads(row["prompt"])
                except json.JSONDecodeError:
                    inst = row
            else:
                inst = row
            instances.append(inst)
    elif path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    instances.append(json.loads(line))
    else:  # .json: single object or a list
        with open(path) as f:
            data = json.load(f)
        instances = data if isinstance(data, list) else [data]
    if limit and limit > 0:
        instances = instances[:limit]
    return instances


def main():
    ap = argparse.ArgumentParser(description="Manual SOPBench eval against a vLLM endpoint")
    ap.add_argument("--base-url", required=True, help="vLLM OpenAI-compatible base URL (.../v1)")
    ap.add_argument(
        "--model", required=True, help="Model name as served by vLLM (no openai/ prefix)"
    )
    ap.add_argument(
        "--data",
        default=None,
        help="Instances: .json / .jsonl / .parquet (default: one built-in SOP-Bench instance)",
    )
    ap.add_argument("--num-samples", type=int, default=0, help="Limit instances (0 = all)")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print raw per-turn response (content, reasoning_content, tool_calls)",
    )
    args = ap.parse_args()

    model = args.model

    from openai import OpenAI
    from amazon_sop_bench.benchmarks import load_benchmark
    from sopbench_agentcore.openai_agent import run_openai_agent
    from sopbench_agentcore.tools import create_openai_tools, make_tool_executor
    from sopbench_agentcore.reward import compute_reward

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=2, timeout=180.0)

    if args.data:
        instances = load_instances(args.data, args.num_samples)
    else:
        from sopbench_agentcore.samples import load_instance

        instances = [load_instance()]
    print(f"Loaded {len(instances)} instance(s) from {args.data}")
    print(f"Endpoint: {args.base_url}  Model: {model}\n")

    # Cache loaded benchmarks (load_benchmark can be slow).
    bench_cache = {}

    dense_scores, binary_scores = [], []
    for i, inst in enumerate(instances, 1):
        bname = inst.get("benchmark", "aircraft_inspection")
        task_input_raw = inst.get("task_input", "{}")
        task_input = (
            json.loads(task_input_raw) if isinstance(task_input_raw, str) else task_input_raw
        )
        ground_truth = inst.get("ground_truth", {}) or {}
        instance_id = inst.get("instance_id", f"inst_{i}")

        if bname not in bench_cache:
            bench_cache[bname] = load_benchmark(bname)
        benchmark = bench_cache[bname]

        tools = create_openai_tools(benchmark)
        tool_executor = make_tool_executor(benchmark)
        try:
            final_output, messages = run_openai_agent(
                task_input,
                benchmark.sop_text,
                tools,
                tool_executor,
                client,
                model,
                max_turns=args.max_turns,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                debug=args.debug,
            )
            scored = compute_reward(messages, ground_truth)
            if args.debug:
                print(f"  [final_output] {final_output!r}")
                last_asst = next(
                    (m for m in reversed(messages) if m.get("role") == "assistant"), None
                )
                print(f"  [last assistant msg] {(last_asst or {}).get('content', '')!r}")
                print(f"  [ground_truth] {ground_truth}")
            dense_scores.append(scored["dense"])
            binary_scores.append(scored["binary"])
            n_turns = sum(1 for m in messages if m.get("role") == "assistant")
            print(
                f"[{i:>3}/{len(instances)}] {instance_id:<28} "
                f"dense={scored['dense']:.3f} TSR={scored['binary']:.0f} "
                f"turns={n_turns} reward={scored['reward']}"
            )
        except Exception as e:
            print(f"[{i:>3}/{len(instances)}] {instance_id:<28} ERROR: {e}")
            dense_scores.append(0.0)
            binary_scores.append(0.0)

    n = len(dense_scores) or 1
    print("\n=== Aggregate ===")
    print(f"instances:      {len(dense_scores)}")
    print(f"mean dense:     {sum(dense_scores) / n:.4f}")
    print(f"mean TSR (bin): {sum(binary_scores) / n:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
