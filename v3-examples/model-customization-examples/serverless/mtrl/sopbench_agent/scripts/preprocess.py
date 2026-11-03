"""Preprocess SOP-Bench tasks into RFT prompt parquets for an MTRL training job.

Emits one parquet row per task with a single `prompt` column. The MTRL RFT runtime forwards
that `prompt` string verbatim to the deployed AgentCore agent, which deserializes it in
main._parse_instance (json.loads), reading benchmark / instance_id / task_input / ground_truth.
So this schema is the contract between the training data and the rollout agent: do NOT switch
to the verl chat/reward_spec schema.

Strict-JSON guarantees (must hold or the rollout breaks):
  - Upstream tasks carry float NaN for inapplicable ground-truth fields. NaN/Inf are converted
    to JSON null and json.dumps uses allow_nan=False, so a bare `NaN` literal can never reach
    the parquet (it would parse only because every Python hop is lenient, then fail elsewhere).
    The reward comparator treats null and NaN as equivalent, so scoring is unchanged.
  - `cross_check_response` is excluded: the SOP output template does not list it, so the model
    cannot know to emit it. Keeping it would pin TSR at 0 for those tasks.

Deterministic 80/20 train/val split (seed 42).

Usage:
  # Local only (writes ./data/aircraft/aircraft_inspection_{train,val}.parquet)
  PYTHONPATH=src python3 scripts/preprocess.py

  # Also upload to S3 in the layout the launch notebook expects
  #   <prefix>/train/aircraft_inspection_train.parquet
  #   <prefix>/validation/aircraft_inspection_val.parquet
  PYTHONPATH=src python3 scripts/preprocess.py \
    --s3-prefix s3://<your-bucket>/sopbench-mtrl/aircraft

Requires amazon_sop_bench (in requirements.txt). S3 upload uses boto3 (pulled in transitively).
"""

import argparse
import json
import math
import os
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from amazon_sop_bench.benchmarks import load_benchmark

from sopbench_agentcore.config import SUPPORTED_BENCHMARKS
from sopbench_agentcore.samples import EXCLUDED_FIELDS


def _clean_nan(obj):
    """Recursively replace float NaN/Inf with None so the result serializes to valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _dumps_strict(value):
    """json.dumps with NaN cleaned first and allow_nan=False as a hard guard (fails loudly)."""
    return json.dumps(_clean_nan(value), allow_nan=False, default=str)


def split_tasks(tasks, seed=42, train_ratio=0.8):
    """Deterministic train/val split."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(tasks))
    n_train = int(len(tasks) * train_ratio)
    return indices[:n_train], indices[n_train:]


def serialize_task(benchmark_name, task_id, task):
    """Serialize one task to the {"prompt": <json string>} row format. None if invalid."""
    expected = task.expected_output
    if not isinstance(expected, dict):
        return None  # this example supports multi-field benchmarks only
    ground_truth = {k: v for k, v in expected.items() if k not in EXCLUDED_FIELDS}
    if not ground_truth:
        return None

    instance = {
        "data_source": "sopbench",
        "instance_id": f"{benchmark_name}_{task_id}",
        "benchmark": benchmark_name,
        "task_input": _dumps_strict(task.inputs),
        "ground_truth": _clean_nan(ground_truth),
    }
    return {"prompt": _dumps_strict(instance)}


def write_prompts_parquet(rows, out_path):
    """Write the `prompt` column as Arrow `string`.

    The MTRL service rejects `large_string`, which pandas >= 3.0 writes by default for str columns,
    so the type is pinned explicitly instead of going through DataFrame.to_parquet.
    """
    table = pa.table({"prompt": pa.array([r["prompt"] for r in rows], type=pa.string())})
    pq.write_table(table, out_path)
    assert pq.read_schema(out_path).field("prompt").type == pa.string(), out_path


def _upload_to_s3(local_path, s3_uri):
    """Upload a single file to an s3://bucket/key URI using boto3."""
    import boto3

    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    boto3.client("s3").upload_file(local_path, bucket, key)
    print(f"  uploaded {local_path} -> {s3_uri}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SOP-Bench to RFT prompt parquet (strict JSON, NaN->null)"
    )
    parser.add_argument("--output-dir", default="./data/aircraft", help="Local output directory")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=list(SUPPORTED_BENCHMARKS),
        choices=SUPPORTED_BENCHMARKS,
        help=f"Benchmarks to process (supported: {', '.join(SUPPORTED_BENCHMARKS)})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train fraction (default: 0.8)"
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Optional s3://bucket/prefix; uploads <prefix>/train/<name>_train.parquet and "
        "<prefix>/validation/<name>_val.parquet (the layout the launch notebook expects)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for name in sorted(args.benchmarks):
        benchmark = load_benchmark(name)
        tasks = list(benchmark.tasks)
        train_idx, val_idx = split_tasks(tasks, seed=args.seed, train_ratio=args.train_ratio)

        written = {}
        for split_name, indices, suffix in [
            ("train", train_idx, "train"),
            ("val", val_idx, "val"),
        ]:
            rows = []
            for i in indices:
                row = serialize_task(name, tasks[i].task_id, tasks[i])
                if row is not None:
                    rows.append(row)
            if not rows:
                continue
            out_path = os.path.join(args.output_dir, f"{name}_{suffix}.parquet")
            write_prompts_parquet(rows, out_path)
            written[split_name] = (out_path, len(rows))

        train_n = written.get("train", ("", 0))[1]
        val_n = written.get("val", ("", 0))[1]
        filtered = len(tasks) - train_n - val_n
        print(
            f"  {name}: {len(tasks)} tasks -> train={train_n} val={val_n}"
            + (f" (filtered {filtered} invalid)" if filtered else "")
        )

        if args.s3_prefix:
            prefix = args.s3_prefix.rstrip("/")
            # train -> train/ ; val -> validation/ (matches the 02_train.ipynb dataset layout).
            s3_subdir = {"train": "train", "val": "validation"}
            for split_name, (local_path, _) in written.items():
                fname = os.path.basename(local_path)
                _upload_to_s3(local_path, f"{prefix}/{s3_subdir[split_name]}/{fname}")

    print(
        f"\nDone. Local parquets in {args.output_dir}/"
        + (f"; uploaded under {args.s3_prefix.rstrip('/')}/" if args.s3_prefix else "")
    )


if __name__ == "__main__":
    main()
