from __future__ import absolute_import

import json
from dataclasses import dataclass
from typing import Dict, Any, Set, List, Optional

import boto3
from botocore.client import BaseClient

from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

STUDIO_METADATA_FILENAME = "metadata-modelzoo_v6.json"


def get_studio_model_metadata_map_from_region(region: str) -> Dict[str, Dict[str, Any]]:
    """Pulls Studio modelzoo metadata from JS prod bucket in region."""
    bucket = get_jumpstart_content_bucket(region=region)
    s3 = boto3.client("s3", region_name=region)
    metadata_file = s3.get_object(Bucket=bucket, Key=STUDIO_METADATA_FILENAME)
    metadata_json = json.loads(metadata_file["Body"].read().decode("utf-8"))
    model_metadata_map: Dict[str, Dict[str, Any]] = {}
    for metadata_entry in metadata_json:
        model_metadata_map[metadata_entry["id"]] = metadata_entry

    return model_metadata_map


def find_objects_under_prefix(bucket: str, prefix: str, s3_client: BaseClient) -> Set[str]:
    """Returns a set of object keys in the bucket with the provided S3 prefix."""
    s3_objects_to_deploy: Set[str] = set()
    src_prefix = to_s3_folder_prefix(prefix=prefix)
    s3_object_list: List[Any] = list_objects_by_prefix(
        bucket_name=bucket,
        prefix=src_prefix,
        s3_client=s3_client
    )
    for s3_object in s3_object_list:
        src_key = s3_object["Key"]
        if not src_key.startswith(src_prefix):
            raise ValueError(f"{src_key} does not have the prefix used to list objects! ({src_prefix})")
        s3_objects_to_deploy.add(src_key)
    return s3_objects_to_deploy


def list_objects_by_prefix(
    bucket_name: str, prefix: str, s3_client: BaseClient, paginate: bool = True
) -> List[Any]:
    """Call s3.list_objects_v2 for objects in the specified bucket that match a prefix.

    Raise:
        ValueError if the bucket name or the prefix are invalid.
    """
    if not bucket_name:
        raise ValueError(f"list objects: invalid bucket name: {bucket_name or 'not-available'}")
    if not prefix:
        raise ValueError(f"list objects: invalid prefix: {prefix or 'not-available'}")

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    contents = response.get("Contents", [])
    if not paginate:
        return contents

    all_content = []
    all_content.extend(contents)
    token: Optional[str] = response.get("NextContinuationToken")
    while token:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=token)
        contents = response.get("Contents", [])
        all_content.extend(contents)
        token = response.get("NextContinuationToken")

    return all_content


def to_s3_folder_prefix(prefix: str) -> str:
    """Removes any leading forward slashes and appends a forward slash to the prefix to match s3 folder conventions."""
    if prefix == "":
        return prefix
    prefix = prefix.lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def convert_s3_key_to_new_prefix(src_key: str, src_prefix: str, dst_prefix: str) -> str:
    """Remove a prefix from the S3 key and prepend a new one."""
    if not src_key.startswith(src_prefix):  # no conversion if src_prefix not matched
        return src_key
    key_without_prefix: str = src_key[len(src_prefix) :]
    return dst_prefix + key_without_prefix


@dataclass
class PublicModelId:
    id: str
    version: str


def construct_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def base_framework(model_specs: JumpStartModelSpecs) -> Optional[str]:
    if model_specs.hosting_ecr_specs.framework == "huggingface":
        return f"pytorch{model_specs.hosting_ecr_specs.framework_version}"
    return None


def get_model_framework(model_specs: JumpStartModelSpecs) -> str:
    return model_specs.model_id.split("-")[0]
