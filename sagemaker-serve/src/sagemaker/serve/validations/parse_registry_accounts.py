"""Script to scrape account IDs for 1p image URIs defined in src/sagemaker/image_uri_config."""

from __future__ import absolute_import
import os

import json

image_uri_config_dir_path = "../../image_uri_config"

account_ids = set()


def extract_account_ids(json_obj):
    """Traverses JSON object until account_ids are found under 'registries'."""
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == "registries" and isinstance(value, dict):
                account_ids.update(value.values())
            else:
                extract_account_ids(value)
    elif isinstance(json_obj, list):
        for item in json_obj:
            extract_account_ids(item)


for filename in os.listdir(image_uri_config_dir_path):
    with open(os.path.join(image_uri_config_dir_path, filename), "r") as f:
        try:
            data = json.load(f)
            for version in data["versions"]:
                extract_account_ids(data["versions"][version])
        except KeyError:
            # JSON objects in image_uri_config/ don't have consistent formatting.
            # Some include job types i.e. 'eia', 'inference', 'training', etc.
            # see tensorflow.json for an example.
            for job_type in data:
                for version in data[job_type]["versions"]:
                    extract_account_ids(data[job_type]["versions"][version])
        except json.JSONDecodeError:
            print(f"Error processing file: {filename}")

print(account_ids)
