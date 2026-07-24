"""Simple processing script for S3 source_dir integ test.

This script validates that:
1. It can be executed from an S3-based source_dir
2. It can import from a sibling module in the same source bundle
"""
import os
import json

from helpers import get_greeting


if __name__ == "__main__":
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "status": "success",
        "greeting": get_greeting("integration-test"),
        "source_dir_type": "s3",
    }

    output_path = os.path.join(output_dir, "result.json")
    with open(output_path, "w") as f:
        json.dump(result, f)

    print(f"Processing complete. Output written to {output_path}")
