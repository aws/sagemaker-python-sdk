import json
from typing import Dict, Any

import boto3

from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

STUDIO_METADATA_FILENAME = "metadata-modelzoo_v6.json"


def get_studio_model_metadata_map_from_region(region: str) -> Dict[str, Dict[str, Any]]:
    """Pulls Studio modelzoo metadata from JS prod bucket in region."""
    bucket = get_jumpstart_content_bucket(region=region)
    s3 = boto3.client("s3", region_name=region)
    metadata_file = s3.get_object(bucket=bucket, key=STUDIO_METADATA_FILENAME)
    metadata_json = json.loads(metadata_file["Body"].read().decode("utf-8"))
    model_metadata_map: Dict[str, Dict[str, Any]] = {}
    for metadata_entry in metadata_json:
        model_metadata_map[metadata_entry["id"]] = metadata_entry

    return model_metadata_map
