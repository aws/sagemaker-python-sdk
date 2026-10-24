"""Load a single SOP-Bench task instance for offline tests and reward inspection.

Instances come from the installed `amazon-sop-bench` package (its benchmark data ships in-package),
so no benchmark data is vendored into this example. The shape matches one training prompt
(see scripts/preprocess.py), except that ground-truth NaN values are kept as float NaN so tests
can exercise the "not applicable" path.
"""

import json

# Not in the SOP output template, so the model cannot know to emit it (see scripts/preprocess.py).
EXCLUDED_FIELDS = {"cross_check_response"}


def load_instance(benchmark_name: str = "aircraft_inspection", task_id: int = 110) -> dict:
    """Return {data_source, instance_id, benchmark, task_input (JSON str), ground_truth (dict)}."""
    from amazon_sop_bench.benchmarks import load_benchmark

    benchmark = load_benchmark(benchmark_name)
    task = next((t for t in benchmark.tasks if str(t.task_id) == str(task_id)), None)
    if task is None:
        raise ValueError(f"task_id {task_id!r} not found in {benchmark_name}")
    ground_truth = {k: v for k, v in task.expected_output.items() if k not in EXCLUDED_FIELDS}
    return {
        "data_source": "sopbench",
        "instance_id": f"{benchmark_name}_{task.task_id}",
        "benchmark": benchmark_name,
        "task_input": json.dumps(task.inputs, default=str),
        "ground_truth": ground_truth,
    }
