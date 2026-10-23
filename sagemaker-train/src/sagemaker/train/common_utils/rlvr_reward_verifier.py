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
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError

from sagemaker.core.training.configs import TrainingJobCompute, HyperPodCompute

logger = logging.getLogger(__name__)

LAMBDA_ARN_REGEX = re.compile(
    r"^arn:aws[a-zA-Z-]*:lambda:[a-z0-9-]+:\d{12}:function:[A-Za-z0-9-_]+$"
)

RewardEnvelope = Literal["oss", "nova", "eval_oss", "eval_nova"]


class RewardSample(BaseModel):
    """One training sample sent to an RLVR reward function.

    The ``oss`` and ``nova`` envelopes send a list of these in one invocation.
    ``reference_answer`` is a plain string or a ``{"text": ...}`` object, as the
    dataset provides it.
    """

    id: str
    messages: List[Dict[str, Any]]
    reference_answer: Optional[Union[str, Dict[str, Any]]] = None


class EvalRewardSample(BaseModel):
    """One evaluation sample sent to a custom-scorer reward function.

    The ``eval_oss`` and ``eval_nova`` envelopes send one invocation per sample, each
    with a list that holds that one sample.
    """

    id: str
    model_response: str
    query: Optional[str] = None
    response: Optional[str] = None
    reference_answer: Optional[Union[str, Dict[str, Any]]] = None


@dataclass(frozen=True)
class RewardVerificationOptions:
    """Options of ``verify_reward_function``.

    Attributes:
        validate_format: Validate each result against ``RewardFunctionOutput``.
        compute: Compute of the training job. A ``HyperPodCompute`` with a Nova model
            makes the Lambda function name contain ``SageMaker``.
        is_nova: Response format when ``envelope`` is None. True expects a bare list,
            False expects ``{statusCode: 200, body: json.dumps(results)}``.
        envelope: Request and response contract. ``oss`` and ``nova`` send every sample
            in one invocation and validate the samples as ``RewardSample``. ``eval_oss``
            and ``eval_nova`` send one invocation per sample and validate the samples as
            ``EvalRewardSample``. ``nova`` expects a bare list back, the other three
            expect the ``statusCode`` and ``body`` envelope. None keeps the ``is_nova``
            behavior and does not validate the samples.
    """

    validate_format: bool = True
    compute: Optional[Union[TrainingJobCompute, HyperPodCompute]] = None
    is_nova: bool = True
    envelope: Optional[RewardEnvelope] = None


@dataclass(frozen=True)
class _EnvelopeSpec:
    """How one envelope shapes the request, the response, and the sample model."""

    sample_model: Optional[Type[BaseModel]]
    per_sample: bool
    nova_model: bool
    bare_list_response: bool


_ENVELOPE_SPECS: Dict[str, _EnvelopeSpec] = {
    "oss": _EnvelopeSpec(
        sample_model=RewardSample, per_sample=False, nova_model=False, bare_list_response=False
    ),
    "nova": _EnvelopeSpec(
        sample_model=RewardSample, per_sample=False, nova_model=True, bare_list_response=True
    ),
    "eval_oss": _EnvelopeSpec(
        sample_model=EvalRewardSample, per_sample=True, nova_model=False, bare_list_response=False
    ),
    "eval_nova": _EnvelopeSpec(
        sample_model=EvalRewardSample, per_sample=True, nova_model=True, bare_list_response=False
    ),
}


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


def _resolve_options(
    options: Optional[RewardVerificationOptions], option_fields: Dict[str, Any]
) -> RewardVerificationOptions:
    """Build the options from the ``options`` object or from its fields passed as keywords."""
    if options is not None and option_fields:
        known = ", ".join(field.name for field in fields(RewardVerificationOptions))
        raise ValueError(f"Pass 'options' or the keyword arguments {known}, not the two together.")
    if options is not None:
        return options
    return RewardVerificationOptions(**option_fields)


def _resolve_envelope(options: RewardVerificationOptions) -> _EnvelopeSpec:
    """Return the envelope spec, with ``is_nova`` as the fallback when no envelope is set."""
    if options.envelope is None:
        return _EnvelopeSpec(
            sample_model=None,
            per_sample=False,
            nova_model=options.is_nova,
            bare_list_response=options.is_nova,
        )
    return _ENVELOPE_SPECS[options.envelope]


def _validate_samples(sample_data: List[Dict[str, Any]], sample_model: Type[BaseModel]) -> None:
    """Validate every sample against the envelope's sample model.

    Raises:
        ValueError: If a sample does not match, with one line per field error.
    """
    errors: List[str] = []
    for idx, sample in enumerate(sample_data):
        try:
            sample_model.model_validate(sample)
        except ValidationError as e:
            errors.extend(
                f"Sample {idx}: {'.'.join(str(p) for p in err['loc'])} - {err['msg']}"
                for err in e.errors()
            )
    if errors:
        raise ValueError(
            f"Sample data does not match the {sample_model.__name__} contract:\n  - "
            + "\n  - ".join(errors)
        )


def _invoke_batches(
    invoke: Callable[[List[Dict[str, Any]]], Any],
    sample_data: List[Dict[str, Any]],
    spec: _EnvelopeSpec,
) -> List[Any]:
    """Return one raw response per invocation.

    A batch envelope sends every sample in one invocation. A per-sample envelope sends
    one invocation per sample.
    """
    if not spec.per_sample:
        return [invoke(sample_data)]
    return [invoke([sample]) for sample in sample_data]


def _unwrap_batches(raw_responses: List[Any], spec: _EnvelopeSpec) -> Any:
    """Unwrap each raw response. A per-sample envelope joins the per-sample lists.

    Raises:
        ValueError: If a response does not match the envelope.
    """
    unwrapped = [_unwrap_response(raw, spec.bare_list_response) for raw in raw_responses]
    if not spec.per_sample:
        return unwrapped[0]
    results: List[Any] = []
    for item in unwrapped:
        if not isinstance(item, list):
            raise ValueError(
                "Reward function must return a list with one result for each sample, "
                f"got {type(item).__name__}"
            )
        results.extend(item)
    return results


def verify_reward_function(
    reward_function: str,
    sample_data: List[Dict[str, Any]],
    options: Optional[RewardVerificationOptions] = None,
    **option_fields: Any,
) -> Dict[str, Any]:
    """Verify a reward function with sample data before using it in RLVR training or evaluation.

    This function allows you to test your reward function implementation with sample
    conversation data to ensure it works correctly before submitting a training or evaluation job.

    Args:
        reward_function: Either a Lambda ARN (string starting with 'arn:aws:lambda:')
                        or a path to a local Python file containing the reward function.
        sample_data: List of conversation samples to test. Each sample should be a dict
                    with 'id', 'messages', and optionally 'reference_answer' keys.
        options: ``RewardVerificationOptions`` with ``validate_format``, ``compute``,
                 ``is_nova``, and ``envelope``.
        **option_fields: The fields of ``RewardVerificationOptions`` as keyword arguments,
                 as an alternative to ``options``. ``validate_format`` (default True)
                 validates each result against ``RewardFunctionOutput``. ``compute``
                 (``TrainingJobCompute`` or ``HyperPodCompute``) with a Nova model makes
                 the Lambda function name contain 'SageMaker', as SageMaker HyperPod
                 demands. ``is_nova`` (default True) expects a bare list back, False
                 expects {statusCode: 200, headers: {...}, body: json.dumps(results)}.
                 ``envelope`` selects the request and response contract, see
                 ``RewardVerificationOptions``.

    Returns:
        Dict containing:
            - success: bool (always True if no exception raised)
            - results: list of individual test results
            - total_samples: total number of samples tested
            - successful_samples: number of successful tests
            - warnings: list of warning messages (e.g., missing reference_answer)

    Raises:
        ValueError: If any validation errors are encountered, with a detailed error message
                   listing all issues found, or if a sample does not match the envelope's
                   sample model.

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

        >>> # Test an evaluation custom scorer for an OSS model, one invocation per sample
        >>> result = verify_reward_function(
        ...     reward_function="./my_scorer.py",
        ...     sample_data=[{"id": "s1", "model_response": "4", "query": "2+2?"}],
        ...     envelope="eval_oss",
        ... )
    """
    resolved_options = _resolve_options(options, option_fields)
    validate_format = resolved_options.validate_format
    compute = resolved_options.compute
    spec = _resolve_envelope(resolved_options)
    if spec.sample_model is not None:
        _validate_samples(sample_data, spec.sample_model)

    # Determine if it's a Lambda ARN or local file
    is_lambda = bool(LAMBDA_ARN_REGEX.match(reward_function))

    # Validate Lambda ARN format for HyperPod compute
    if spec.nova_model and isinstance(compute, HyperPodCompute):
        if is_lambda:
            # Extract function name from ARN: arn:aws[-*]:lambda:region:account:function:function-name
            function_name_match = re.search(
                r"arn:aws[a-zA-Z-]*:lambda:[^:]+:[^:]+:function:([^:]+)", reward_function
            )
            if function_name_match:
                function_name = function_name_match.group(1)
                # Check if function name contains 'SageMaker' (case-insensitive)
                if not re.search(r"sagemaker", function_name, re.IGNORECASE):
                    raise ValueError(
                        f"Lambda ARN for HyperPod compute must contain 'SageMaker' "
                        f"in the function name for Nova models. "
                        f"Current function name: '{function_name}'. "
                        f"Expected format: 'arn:aws:lambda:*:*:function:*SageMaker*'"
                    )
            else:
                raise ValueError(
                    f"Invalid Lambda ARN format: {reward_function}. "
                    f"Expected format: 'arn:aws:lambda:region:account:function:function-name'"
                )
        else:
            logger.warning(
                f"Skipping Lambda function name validation because a local Python file is being used: "
                f"'{reward_function}'. "
                "For Nova RLVR jobs on HyperPod, ensure your reward function Lambda name "
                "contains 'SageMaker' (case-insensitive). "
                "Expected format: 'arn:aws:lambda:*:*:function:*SageMaker*'"
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

            def invoke_lambda(batch: List[Dict[str, Any]]) -> Any:
                response = lambda_client.invoke(
                    FunctionName=reward_function,
                    InvocationType="RequestResponse",
                    Payload=json.dumps(batch),
                )
                return json.loads(response["Payload"].read())

            raw_responses = _invoke_batches(invoke_lambda, sample_data, spec)

            # Unwrap based on output format (Nova vs OSS)
            try:
                payload = _unwrap_batches(raw_responses, spec)
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

            # Call handler per the envelope (matches Lambda behavior)
            try:
                raw_responses = _invoke_batches(lambda batch: handler(batch, {}), sample_data, spec)

                # Unwrap based on output format (Nova vs OSS)
                try:
                    result = _unwrap_batches(raw_responses, spec)
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
