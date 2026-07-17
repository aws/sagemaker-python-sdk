# Data utility functions for inspecting and processing datasets
import re
import json
import boto3
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from botocore.exceptions import ClientError

from sagemaker.core.s3 import parse_s3_url
from sagemaker.ai_registry.dataset import DataSet

logger = logging.getLogger(__name__)

S3_URI_REGEX = re.compile(r"^s3://([a-zA-Z0-9.\-_]+)/(.+)$")


class FileLoadError(Exception):
    """Custom exception for file loading errors."""

    pass


def _parse_s3_uri(uri: str) -> Optional[Tuple[str, str]]:
    """Parse S3 URI into (bucket, key) tuple, or None if the URI is invalid."""
    if not S3_URI_REGEX.match(uri):
        return None

    return parse_s3_url(uri)


def _validate_extension(path: str, extension: str) -> None:
    """
    Validate that the given path has the required file extension.

    Args:
        path: File path or S3 URI
        extension: Extension (e.g., '.yaml')

    Raises:
        FileLoadError: If extension doesn't match
    """
    if not path.lower().endswith(extension.lower()):
        raise FileLoadError(f"File must have {extension} extension: {path}")


def load_file_content(
    file_path: str,
    extension: Optional[str] = None,
    encoding: Optional[str] = "utf-8",
    region: Optional[str] = None,
):
    """
    Stream file content line by line from S3 or local filesystem.
    This is a generator that yields lines lazily without loading the entire file into memory.

    Args:
        file_path: Path to file (either local path or S3 URI)
        extension: Optional file extension to validate
        encoding: Optional encoding format (defaults to utf-8)

    Yields:
        Lines from the file

    Raises:
        FileLoadError: If file cannot be loaded
    """
    # Validate extension
    if extension is not None:
        _validate_extension(file_path, extension)

    # Try S3 first
    s3_parts = _parse_s3_uri(file_path)
    if s3_parts:
        bucket, key = s3_parts
        try:
            s3 = boto3.client("s3", region_name=region)
            response = s3.get_object(Bucket=bucket, Key=key)
            # Stream from S3 using iter_lines
            for line in response["Body"].iter_lines():
                yield line.decode(encoding)
        except ClientError as e:
            raise FileLoadError(f"Failed to load S3 file {file_path}: {e}")
    else:
        # Try local filesystem
        try:
            path = Path(file_path)
            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    yield line.rstrip("\n\r")
        except FileNotFoundError:
            raise FileLoadError(f"File not found: {file_path}")
        except OSError as e:
            raise FileLoadError(f"Failed to read file {file_path}: {e}")


def validate_data_path_exists(
    data_path: Union[str, "DataSet"],
    sagemaker_session,
    label: str = "data",
) -> None:
    """Validate that a data path (S3 URI, dataset ARN, or DataSet object) exists and is accessible.

    Called inline during dry_run to catch bad paths before job submission.

    Args:
        data_path: S3 URI, SageMaker hub-content DataSet ARN, or DataSet object to validate.
        sagemaker_session: SageMaker session (provides boto_session).
        label: Human-readable label for error messages.

    Raises:
        ValueError: If the path does not exist or is inaccessible.
    """
    # Handle DataSet objects — extract the ARN for validation
    if isinstance(data_path, DataSet):
        data_path = data_path.arn

    # Handle SageMaker hub-content DataSet ARNs
    if data_path.startswith("arn:aws:sagemaker:") and "/DataSet/" in data_path:
        _validate_dataset_arn_exists(data_path, sagemaker_session, label=label)
        return

    # Handle S3 URIs
    parts = _parse_s3_uri(data_path)
    if parts is None:
        raise ValueError(
            f"Invalid {label} path format: {data_path}. "
            f"Expected an S3 URI (s3://bucket/key) or a DataSet ARN."
        )

    bucket, key = parts
    s3 = sagemaker_session.boto_session.client("s3")

    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        if resp.get("KeyCount", 0) == 0:
            raise ValueError(
                f"S3 {label} path does not exist: {data_path}"
            )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "403" or "AccessDenied" in str(e):
            # Caller may not have access but the execution role might —
            # log a warning and allow the job to proceed.
            logger.warning(
                "Cannot verify S3 %s path %s from caller identity "
                "(AccessDenied). The execution role may still have access.",
                label, data_path,
            )
        else:
            raise ValueError(f"Error accessing S3 {label} path {data_path}: {e}")


def _validate_dataset_arn_exists(
    dataset_arn: str,
    sagemaker_session,
    label: str = "data",
) -> None:
    """Validate that a SageMaker hub-content DataSet ARN exists.

    Args:
        dataset_arn: ARN like arn:aws:sagemaker:<region>:<account>:hub-content/<hub>/DataSet/<name>/<version>
        sagemaker_session: SageMaker session (provides boto_session).
        label: Human-readable label for error messages.

    Raises:
        ValueError: If the dataset ARN cannot be described.
    """

    pattern = (
        r"^arn:aws:sagemaker:([^:]+):(\d+):hub-content/"
        r"([^/]+)/DataSet/([^/]+)/([\d\.]+)$"
    )
    match = re.match(pattern, dataset_arn)
    if not match:
        raise ValueError(
            f"Invalid {label} DataSet ARN format: {dataset_arn}"
        )

    region, _, hub_name, content_name, content_version = match.groups()
    sm_client = sagemaker_session.sagemaker_client

    try:
        sm_client.describe_hub_content(
            HubName=hub_name,
            HubContentType="DataSet",
            HubContentName=content_name,
            HubContentVersion=content_version,
        )
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceNotFound" or "does not exist" in str(e).lower():
            raise ValueError(
                f"{label.capitalize()} DataSet does not exist: {dataset_arn}"
            )
        elif code == "AccessDeniedException" or "AccessDenied" in str(e):
            logger.warning(
                "Cannot verify %s DataSet %s from caller identity "
                "(AccessDenied). The execution role may still have access.",
                label, dataset_arn,
            )
        else:
            raise ValueError(
                f"Error validating {label} DataSet {dataset_arn}: {e}"
            )


def _has_multimodal_content(record: dict) -> bool:
    """Check if a single record contains multimodal content."""
    if "messages" not in record:
        return False

    for msg in record["messages"] or []:
        if not isinstance(msg, dict):
            continue
        # Use `or []` to handle both missing key and explicit null value
        for content_item in msg.get("content") or []:
            if isinstance(content_item, dict) and any(
                key in content_item for key in ["image", "video", "document"]
            ):
                return True
    return False


def _check_records(records) -> bool:
    """Return True if any record in the iterable contains multimodal content."""
    for record in records:
        if _has_multimodal_content(record):
            return True
    return False


def is_multimodal_data(dataset: Union[str, "DataSet"]) -> bool:
    """
    Check if dataset contains multimodal data by scanning records.

    Supports .jsonl (line-delimited JSON, streamed) and .json (full JSON array/object,
    loaded into memory). Returns True as soon as a multimodal record is found.

    Uses the same file loading approach as the dataset loader (load_file_content),
    including UTF-8 BOM handling via utf-8-sig encoding for JSONL files.

    Args:
        dataset: S3 URI or local dataset (.jsonl or .json), or a DataSet object.
            For DataSet objects, the .source attribute is used to get the S3 URI.

    Returns:
        True if multimodal fields detected, False otherwise
    """

    logger.info(f"Auto-detecting whether dataset is multimodal: {dataset}")

    if isinstance(dataset, DataSet):
        data_s3_path = dataset.source
    else:
        data_s3_path = dataset

    try:
        if data_s3_path.endswith(".json"):
            # .json files are a single JSON object or array — read all lines then parse.
            lines = list(load_file_content(data_s3_path, extension=".json", encoding="utf-8"))
            content = "\n".join(lines)
            data = json.loads(content)
            records = data if isinstance(data, list) else [data]
            return _check_records(records)
        else:
            # .jsonl (default): stream line by line for memory efficiency.
            # utf-8-sig strips the UTF-8 BOM that some tools add, matching dataset loader behavior.
            for line in load_file_content(data_s3_path, extension=".jsonl", encoding="utf-8-sig"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if _has_multimodal_content(record):
                        return True
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    continue
            return False
    except Exception as e:
        logger.warning(
            f"Failed to check multimodal data from {data_s3_path}: {e}. Defaulting to text-only."
        )
        return False
