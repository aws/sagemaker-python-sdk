# Copyright Amazon.com, Inc. or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reward function verification utility for RLVR training."""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError

from sagemaker.core.training.configs import TrainingJobCompute, HyperPodCompute


logger = logging.getLogger(__name__)

LAMBDA_ARN_REGEX = re.compile(
    r"^arn:aws[a-zA-Z-]*:lambda:[a-z0-9-]+:\d{12}:function:[A-Za-z0-9-_]+$"
)

class RewardMetric(BaseModel):
    """A single metric or reward entry from a reward function output."""

    name: str
    value: Union[int, float]
    type: Literal["Metric", "Reward"]


class RewardFunctionOutput(BaseModel):
    """Validated output format for a reward function result."""

    id: Any
    aggregate_reward_score: Union[int, float]
    metrics_list: Optional[List[RewardMetric]] = None


def _validate_output_format(result: Dict[str, Any], idx: int) -> List[str]:
    """Validate the output format of a single reward function result."""
    if not isinstance(result, dict):
        return [f"Output {idx}: Expected dict, got {type(result).__name__}"]
    try:
        RewardFunctionOutput.model_validate(result)
        return []
    except ValidationError as e:
        return [
            f"Output {idx}: {'.'.join(str(p) for p in err['loc'])} - {err['msg']}"
            for err in e.errors()
        ]


def _unwrap_response(payload: Any, is_nova: bool) -> Any:
    """Unwrap reward function response based on output format.

    Nova format: returns results list directly (List[Dict]).
    OSS format: returns {statusCode: 200, headers: {...}, body: json.dumps(results)}.
    """
    if is_nova:
        return payload

    # OSS format: unwrap from HTTP-style envelope
    if not isinstance(payload, dict):
        raise ValueError(
            f"OSS reward function must return a dict with 'statusCode' and 'body', "
            f"got {type(payload).__name__}"
        )

    status_code = payload.get("statusCode")
    if status_code != 200:
        error_body = payload.get("body", "No error details")
        raise ValueError(
            f"OSS reward function returned non-200 status code: {status_code}. "
            f"Body: {error_body}"
        )

    body = payload.get("body")
    if body is None:
        raise ValueError("OSS reward function response missing 'body' field")

    # body is JSON-encoded string in OSS format
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"OSS reward function 'body' is not valid JSON: {e}")

    # body is already parsed (e.g., local handler returned dict directly)
    return body

def verify_reward_function(
    reward_function: str,
    sample_data: List[Dict[str, Any]],
    validate_format: bool = True,
    compute: Optional[Union[TrainingJobCompute, HyperPodCompute]] = None,
    is_nova: bool = True,
) -> Dict[str, Any]:
    """
    Verify a reward function with sample data before using it in RLVR training or evaluation.

    This function allows you to test your reward function implementation with sample
    conversation data to ensure it works correctly before submitting a training or evaluation job.

    Args:
        reward_function: Either a Lambda ARN (string starting with 'arn:aws:lambda:')
                        or a path to a local Python file containing the reward function.
        sample_data: List of conversation samples to test. Each sample should be a dict
                    with 'id', 'messages', and optionally 'reference_answer' keys.
        validate_format: If True, validates that the output matches expected format
                        (default: True).
        compute: Compute configuration (TrainingJobCompute or HyperPodCompute). Required
                 when using Lambda ARN with Nova models. When set to HyperPodCompute,
                 validates that Lambda ARN contains 'SageMaker' in the function name as
                 required by SageMaker HyperPod. Optional for local files.
        is_nova: If True (default), expects Nova format where the reward function returns
                 a list of results directly. If False, expects OSS format where the function
                 returns {statusCode: 200, headers: {...}, body: json.dumps(results)}.

    Returns:
        Dict containing:
            - success: bool (always True if no exception raised)
            - results: list of individual test results
            - total_samples: total number of samples tested
            - successful_samples: number of successful tests
            - warnings: list of warning messages (e.g., missing reference_answer)

    Raises:
        ValueError: If any validation errors are encountered, with a detailed error message
                   listing all issues found.

    Example:
        >>> # Test training reward function with Lambda ARN
        >>> result = verify_reward_function(
        ...     reward_function="arn:aws:lambda:us-east-1:123456789012:function:my-reward",
        ...     sample_data=[
        ...         {
        ...             "id": "sample_1",
        ...             "reference_answer": "correct answer",
        ...             "messages": [
        ...                 {"role": "user", "content": "question"},
        ...                 {"role": "assistant", "content": "response"}
        ...             ]
        ...         }
        ...     ],
        ...     compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge")
        ... )

        >>> # Test reward function with local Python file
        >>> result = verify_reward_function(
        ...     reward_function="./my_reward.py",
        ...     sample_data=[
        ...         {
        ...             "id": "sample_1",
        ...             "reference_answer": "correct answer",
        ...             "messages": [
        ...                 {"role": "user", "content": "question"},
        ...                 {"role": "assistant", "content": "response"}
        ...             ]
        ...         }
        ...     ]
        ... )

        >>> # Test with HyperPod compute validation
        >>> result = verify_reward_function(
        ...     reward_function="arn:aws:lambda:us-east-1:123456789012:function:MySageMakerReward",
        ...     sample_data=[...],
        ...     compute=HyperPodCompute(cluster_name="my-cluster")
        ... )
    """
    # Determine if it's a Lambda ARN or local file
    is_lambda = bool(LAMBDA_ARN_REGEX.match(reward_function))

    # Compute is required for Lambda ARNs with Nova models
    if is_lambda and is_nova and compute is None:
        raise ValueError(
            "The 'compute' parameter is required for Nova models when using a Lambda ARN. "
            "Please specify compute as a TrainingJobCompute or HyperPodCompute instance."
        )

    # Validate Lambda ARN format for HyperPod compute
    if is_nova and isinstance(compute, HyperPodCompute) and is_lambda:
        # Extract function name from ARN: arn:aws[-*]:lambda:region:account:function:function-name
        function_name_match = re.search(
            r"arn:aws[a-zA-Z-]*:lambda:[^:]+:[^:]+:function:([^:]+)", reward_function
        )
        if function_name_match:
            function_name = function_name_match.group(1)
            # Check if function name contains 'SageMaker' (case-insensitive)
            if not re.search(r"sagemaker", function_name, re.IGNORECASE):
                raise ValueError(
                    f"Lambda ARN for HyperPod compute must contain 'SageMaker' in the function name for Nova models. "
                    f"Current function name: '{function_name}'. "
                    f"Expected format: 'arn:aws:lambda:*:*:function:*SageMaker*'"
                )
        else:
            raise ValueError(
                f"Invalid Lambda ARN format: {reward_function}. "
                f"Expected format: 'arn:aws:lambda:region:account:function:function-name'"
            )

    results = []
    warnings = []

    if is_lambda:
        # Extract region from ARN (arn:partition:lambda:REGION:account:function:name)
        region = reward_function.split(":")[3]

        # Test with Lambda
        lambda_client = boto3.client("lambda", region_name=region)

        try:
            # Log Lambda invocation details for debugging
            logger.info(f"Invoking Lambda: {reward_function}")
            logger.info(f"Number of samples: {len(sample_data)}")

            # Invoke Lambda with all samples at once (Lambda expects array)
            response = lambda_client.invoke(
                FunctionName=reward_function,
                InvocationType="RequestResponse",
                Payload=json.dumps(sample_data),
            )

            # Parse response
            payload = json.loads(response["Payload"].read())

            # Unwrap based on output format (Nova vs OSS)
            try:
                payload = _unwrap_response(payload, is_nova)
            except ValueError as e:
                # Unwrap failure means the response format is wrong
                error_msg = str(e)
                for idx in range(len(sample_data)):
                    results.append(
                        {
                            "sample_index": idx,
                            "input": sample_data[idx],
                            "status": "error",
                            "errors": [error_msg] if idx == 0 else [],
                        }
                    )
                payload = None  # Signal to skip further processing

            # Process results
            if payload is not None and isinstance(payload, list):
                logger.info(f"Lambda returned list with {len(payload)} result(s)")
                for idx, result in enumerate(payload):
                    # Log input for this sample
                    if idx < len(sample_data):
                        input_str = json.dumps(sample_data[idx], indent=2)
                        logger.info(f"Sample {idx} INPUT:\n{input_str}")

                    # Validate output format
                    sample_errors = []
                    if validate_format:
                        sample_errors = _validate_output_format(result, idx)

                    # Log output for this sample
                    result_str = json.dumps(result, indent=2)
                    status = "PASS" if not sample_errors else "FAIL"
                    logger.info(f"Sample {idx} OUTPUT [{status}]:\n{result_str}")
                    if sample_errors:
                        logger.warning(
                            f"Sample {idx} validation errors: {', '.join(sample_errors)}"
                        )

                    results.append(
                        {
                            "sample_index": idx,
                            "input": sample_data[idx] if idx < len(sample_data) else {},
                            "output": result,
                            "status": "error" if sample_errors else "success",
                            "errors": sample_errors,
                        }
                    )
            else:
                if payload is None:
                    pass  # Already handled above during unwrap
                # Single result format (not a list) - could be Lambda error or single dict response
                elif isinstance(payload, dict) and (
                    "errorMessage" in payload or "errorType" in payload
                ):
                    # Lambda execution error - treat all samples as failed with a single shared error
                    error_msg = payload.get("errorMessage", "Unknown error")
                    error_type = payload.get("errorType", "Unknown")
                    error_text = f"Lambda execution error - {error_type}: {error_msg}"

                    # Mark all samples as failed, but only include error text in first result
                    # to avoid repeating the same error 200 times
                    for idx in range(len(sample_data)):
                        results.append(
                            {
                                "sample_index": idx,
                                "input": sample_data[idx],
                                "output": payload,
                                "status": "error",
                                "errors": [error_text] if idx == 0 else [],
                            }
                        )
                else:
                    # Single successful result
                    logger.info("Lambda returned single result (non-list format)")
                    result_str = (
                        json.dumps(payload, indent=2) if isinstance(payload, dict) else str(payload)
                    )
                    logger.info(f"Result:\n{result_str}")
                    
                    results.append(
                        {
                            "sample_index": 0,
                            "input": sample_data,
                            "output": payload,
                            "status": "success",
                            "errors": [],
                        }
                    )

        except (ClientError, Exception) as e:
            error_msg = f"Lambda invocation failed: {str(e)}"
            # Mark all samples as failed
            for idx in range(len(sample_data)):
                results.append(
                    {
                        "sample_index": idx,
                        "input": sample_data[idx],
                        "status": "error",
                        "errors": [error_msg],
                    }
                )

    else:
        # Test with local Python file
        logger.info(f"Testing local Python file: {reward_function}")
        logger.info(f"Number of samples: {len(sample_data)}")

        try:
            # Read and execute the Python file
            with open(reward_function, "r") as f:
                code = f.read()

            # Create a namespace for execution
            namespace: Dict[str, Any] = {}
            exec(code, namespace)

            # Find the lambda_handler function
            if "lambda_handler" not in namespace:
                raise ValueError("Local Python file must contain a 'lambda_handler' function")

            handler = namespace["lambda_handler"]

            # Call handler with all samples at once (matches Lambda behavior)
            try:
                result = handler(sample_data, {})

                # Unwrap based on output format (Nova vs OSS)
                try:
                    result = _unwrap_response(result, is_nova)
                except ValueError as e:
                    error_msg = str(e)
                    for idx in range(len(sample_data)):
                        results.append(
                            {
                                "sample_index": idx,
                                "input": sample_data[idx] if idx < len(sample_data) else {},
                                "status": "error",
                                "errors": [error_msg] if idx == 0 else [],
                            }
                        )
                    result = None  # Signal to skip further processing

                # Log handler response
                result_str = (
                    json.dumps(result, indent=2)
                    if isinstance(result, (dict, list))
                    else str(result)
                )
                logger.info(f"Handler response:\n{result_str}")

                # Process results
                if result is not None and isinstance(result, list):
                    logger.info(f"Lambda returned list with {len(result)} result(s)")
                    for idx, item in enumerate(result):
                        # Log input for this sample
                        if idx < len(sample_data):
                            input_str = json.dumps(sample_data[idx], indent=2)
                            logger.info(f"Sample {idx} INPUT:\n{input_str}")

                        # Validate output format
                        sample_errors = []
                        if validate_format:
                            sample_errors = _validate_output_format(item, idx)

                        # Log output for this sample
                        item_str = json.dumps(item, indent=2)
                        status = "PASS" if not sample_errors else "FAIL"
                        logger.info(f"Sample {idx} OUTPUT [{status}]:\n{item_str}")
                        if sample_errors:
                            logger.warning(
                                f"Sample {idx} validation errors: {', '.join(sample_errors)}"
                            )

                        results.append(
                            {
                                "sample_index": idx,
                                "input": sample_data[idx] if idx < len(sample_data) else {},
                                "output": item,
                                "status": "error" if sample_errors else "success",
                                "errors": sample_errors,
                            }
                        )
                else:
                    # Single result format or already handled by unwrap
                    if result is not None:
                        logger.info("Lambda returned single result (non-list format)")
                        results.append(
                            {
                                "sample_index": 0,
                                "input": sample_data,
                                "output": result,
                                "status": "success",
                                "errors": [],
                            }
                        )

            except Exception as e:
                error_msg = f"Handler execution failed: {str(e)}"
                # Mark all samples as failed
                for idx in range(len(sample_data)):
                    results.append(
                        {
                            "sample_index": idx,
                            "input": sample_data[idx] if idx < len(sample_data) else {},
                            "status": "error",
                            "errors": [error_msg],
                        }
                    )

        except FileNotFoundError:
            error_msg = f"Python file not found: {reward_function}"
            # Mark all samples as failed (no results to process)
            for idx in range(len(sample_data)):
                results.append(
                    {
                        "sample_index": idx,
                        "input": sample_data[idx],
                        "status": "error",
                        "errors": [error_msg],
                    }
                )

        except Exception as e:
            error_msg = f"Failed to load Python file: {str(e)}"
            # Mark all samples as failed
            for idx in range(len(sample_data)):
                results.append(
                    {
                        "sample_index": idx,
                        "input": sample_data[idx],
                        "status": "error",
                        "errors": [error_msg],
                    }
                )

    # Log warnings if any
    if warnings:
        logger.warning(f"Reward function verification completed with {len(warnings)} warning(s):")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Count successful samples
    successful_samples = len([r for r in results if r.get("status") == "success"])
    total_samples = len(sample_data)
    failed_samples = total_samples - successful_samples

    # Log verification summary
    if failed_samples == 0:
        logger.info(f"All {total_samples} sample(s) passed validation")
    else:
        logger.warning(f"{failed_samples}/{total_samples} sample(s) failed validation")

    # Raise error if any samples failed
    if failed_samples > 0:
        # Collect all errors from results
        all_errors = []
        for r in results:
            if r.get("status") == "error" and "errors" in r:
                all_errors.extend(r["errors"])

        # Build simplified error message
        error_parts = [
            f"Reward function verification failed: {failed_samples}/{total_samples} sample(s) failed validation.",
            f"Only {successful_samples}/{total_samples} sample(s) passed.",
            "",
        ]

        # Include all validation errors
        if all_errors:
            error_parts.append("Validation errors:")
            for err in all_errors:
                error_parts.append(f"  - {err}")
            error_parts.append("")

        # Add helpful guidance
        error_parts.extend(
            [
                "Please check your reward function output format. Each output must include:",
                "  - 'id': string identifier",
                "  - 'aggregate_reward_score': numeric reward value",
                "",
            ]
        )

        error_message = "\n".join(error_parts)
        raise ValueError(error_message)

    return {
        "success": True,
        "results": results,
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "warnings": warnings,
    }
