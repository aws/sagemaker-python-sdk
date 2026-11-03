"""Inspect the reward on canned answers - no model endpoint needed.

Scores a few hand-written answers with the same reward.compute_reward used in training, against
one real SOP-Bench instance, so you can see how dense, TSR, and the training reward respond to
the output contract. The live, model-in-the-loop evaluation is 03_evaluate_and_deploy.ipynb.

Usage:
  PYTHONPATH=src python3 scripts/inspect_reward.py
"""

import json
import sys


def tagged(answer: dict) -> str:
    return "<task_result><answer_json>" + json.dumps(answer) + "</answer_json></task_result>"


def main() -> int:
    from sopbench_agentcore.reward import compute_reward
    from sopbench_agentcore.samples import load_instance

    gt = load_instance()["ground_truth"]  # aircraft_inspection_110
    not_applicable = [k for k, v in gt.items() if isinstance(v, float) and v != v]
    correct = {k: ("NA" if k in not_applicable else v) for k, v in gt.items()}
    applicable = {k: v for k, v in correct.items() if k not in not_applicable}
    first = next(k for k in applicable if isinstance(gt[k], str))

    cases = [
        ("1. every field, NA for not-applicable", tagged(correct)),
        ("2. not-applicable fields omitted", tagged(applicable)),
        ("3. hedged value in one field", tagged(dict(correct, **{first: "success or failed"}))),
        ("4. correct fields, no answer tags", json.dumps(correct)),
    ]
    print(f"Ground truth: {len(gt)} fields; not applicable: {not_applicable}\n")
    print(f"{'case':<42} {'dense':>6} {'TSR':>4} {'reward':>7}")
    for label, answer in cases:
        r = compute_reward([{"role": "assistant", "content": answer}], gt)
        print(f"{label:<42} {r['dense']:>6.2f} {r['binary']:>4.0f} {r['reward']:>7.2f}")
    print(
        '\n* Case 2: every field must be emitted ("NA" when not applicable) or it scores a miss.\n'
        "* Case 3: a value that matches several of the field's allowed values never counts.\n"
        "* Case 4: without the answer tags the rollout has not answered, so it scores -1."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
