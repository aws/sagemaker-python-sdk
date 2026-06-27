import json
import logging
import re
import uuid
from typing import Any, Dict, List

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# GSM8k specific constants and functions
_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    """Extract numerical solution from solution string."""
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break

    return final_answer


def compute_gsm8k_score(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning."
    Proceedings of the 62nd Annual Meeting of the Association for Computational
    Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    print("gsm8k solutionstr", solution_str)
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score


# Lambda utility functions
def _ok(body: Any, code: int = 200) -> Dict[str, Any]:
    return {
        "statusCode": code,
        "headers": {
            "content-type": "application/json",
            "access-control-allow-origin": "*",
            "access-control-allow-methods": "POST,OPTIONS",
            "access-control-allow-headers": "content-type",
        },
        "body": json.dumps(body),
    }


def _assistant_text(sample: Dict[str, Any]) -> str:
    """Extract assistant text from sample messages."""
    for m in reversed(sample.get("messages", [])):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def _sample_id(sample: Dict[str, Any]) -> str:
    """Generate or extract sample ID."""
    if isinstance(sample.get("id"), str) and sample["id"]:
        return sample["id"]
    md = sample.get("metadata") or {}
    if isinstance(md.get("key"), str) and md["key"]:
        return md["key"]
    return str(uuid.uuid4())


def _score_and_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single sample using GSM8k scoring logic."""
    sid = _sample_id(sample)
    solution_text = _assistant_text(sample)
    md = sample.get("metadata") or {}
    extra = sample.get("extra_info") or {}

    # Extract ground truth
    gt_raw = md.get("reference_answer") or md.get("ground_truth")
    if gt_raw is None:
        gt = ""
    else:
        gt = str(gt_raw).strip()

    metrics_list: List[Dict[str, Any]] = []

    # GSM8k specific scoring
    if solution_text and gt:
        # Get scoring parameters from extra_info or use defaults
        method = extra.get("extraction_method", "strict")
        format_score = float(extra.get("format_score", 0.0))
        full_score = float(extra.get("full_score", 1.0))

        # Compute GSM8k score
        gsm8k_score = compute_gsm8k_score(
            solution_str=solution_text,
            ground_truth=gt,
            method=method,
            format_score=format_score,
            score=full_score,
        )

        # Extract the answer for debugging/metrics
        extracted_answer = extract_solution(solution_text, method=method)

        # Add detailed metrics
        metrics_list.append(
            {"name": "gsm8k_score", "value": float(gsm8k_score), "type": "Reward"}
        )
        metrics_list.append(
            {
                "name": "extracted_answer",
                "value": extracted_answer or "None",
                "type": "Metric",
            }
        )
        metrics_list.append(
            {"name": "ground_truth", "value": gt, "type": "Metric"}
        )
        metrics_list.append(
            {"name": "extraction_method", "value": method, "type": "Metric"}
        )

        # The aggregate reward score is the GSM8k score
        aggregate_score = gsm8k_score
    else:
        # No solution text or ground truth - default to 0
        aggregate_score = 0.0
        metrics_list.append(
            {"name": "default_zero", "value": 0.0, "type": "Reward"}
        )

    print(
        "detected score",
        {
            "id": sid,
            "aggregate_reward_score": float(aggregate_score),
            "metrics_list": metrics_list,
        },
    )

    return {
        "id": sid,
        "aggregate_reward_score": float(aggregate_score),
        "metrics_list": metrics_list,
    }


def lambda_handler(event, context):
    """AWS Lambda handler for GSM8k grading."""

    # If event is already a list (direct lambda.invoke with batch payload), process directly
    if isinstance(event, list):
        samples = event
        try:
            results = [_score_and_metrics(s) for s in samples]
        except Exception as e:
            return _ok({"error": f"GSM8k scoring failed: {e}"}, 500)
        return _ok(results)

    # CORS preflight
    try:
        if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
            return _ok({"ok": True})
    except Exception as e:
        logger.error("Error during CORS preflight check: %s", e)
        raise

    # Body may be a JSON string (API GW/Function URL) or already a dict (Invoke)
    try:
        raw = event.get("body") or "{}"
        body = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.error("Error parsing request body: %s", e)
        raise

    # Accept top-level list, {"batch":[...]}, or single sample object
    if isinstance(body, dict) and isinstance(body.get("batch"), list):
        samples = body["batch"]
    elif isinstance(body, list):
        samples = body
    else:
        return _ok(
            {
                "error": "Send a sample object, or {'batch':[...]} , or a top-level list of samples."
            },
            400,
        )

    try:
        results = [_score_and_metrics(s) for s in samples]
    except Exception as e:
        return _ok({"error": f"GSM8k scoring failed: {e}"}, 500)

    return _ok(results)
