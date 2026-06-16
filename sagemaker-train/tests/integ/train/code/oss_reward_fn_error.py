"""OSS reward function that returns a non-200 status code (for error handling tests)."""

import json


def lambda_handler(event, context):
    """Simulates an OSS reward function that encounters an internal error."""
    return {
        "statusCode": 500,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": "Internal server error"}),
    }
