# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Utility functions for fetching model information from HuggingFace Hub"""

from __future__ import absolute_import
import json
import urllib.request
from json import JSONDecodeError
from urllib.error import HTTPError, URLError
import logging

logger = logging.getLogger(__name__)


def _get_model_config_properties_from_hf(model_id: str, hf_hub_token: str = None):
    """Placeholder docstring"""

    config_files = ["config.json", "model_index.json", "adapter_config.json"]

    model_config = None
    for config_file in config_files:
        config_url = f"https://huggingface.co/{model_id}/raw/main/{config_file}"
        request = config_url

        try:
            if hf_hub_token:
                request = urllib.request.Request(
                    config_url, headers={"Authorization": "Bearer " + hf_hub_token}
                )

            with urllib.request.urlopen(request) as response:
                model_config = json.load(response)
                break
        except (HTTPError, URLError, TimeoutError, JSONDecodeError) as e:
            if "HTTP Error 401: Unauthorized" in str(e):
                raise ValueError(
                    "Trying to access a gated/private HuggingFace model without valid credentials. "
                    "Please provide a HUGGING_FACE_HUB_TOKEN in env_vars"
                )

            logger.warning(
                "Exception encountered while trying to read config file %s. Details: %s",
                config_url,
                e,
            )

    if not model_config:
        allowed_files = ", ".join(config_files)
        raise ValueError(
            f"Did not find any supported model config file in Hugging Face Hub for {model_id}. "
            f"Expected one of: {allowed_files}"
        )
    return model_config
