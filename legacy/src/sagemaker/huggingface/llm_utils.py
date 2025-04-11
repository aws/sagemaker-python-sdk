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
"""Functions for generating ECR image URIs for pre-built SageMaker Docker images."""
from __future__ import absolute_import

import os
from typing import Optional
import importlib.util

import urllib.request
from urllib.error import HTTPError, URLError
import json
from json import JSONDecodeError
import logging
from sagemaker import image_uris
from sagemaker.session import Session

logger = logging.getLogger(__name__)


def get_huggingface_llm_image_uri(
    backend: str,
    session: Optional[Session] = None,
    region: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    """Retrieves the image URI for inference.

    Args:
        backend (str): The backend to use. Valid values include "huggingface" and "lmi".
        session (Session): The SageMaker Session to use. (Default: None).
        region (str): The AWS region to use for image URI. (default: None).
        version (str): The framework version for which to retrieve an
            image URI. If no version is set, defaults to latest version. (default: None).

    Returns:
        str: The image URI string.
    """

    if region is None:
        if session is None:
            region = Session().boto_session.region_name
        else:
            region = session.boto_session.region_name
    if backend == "huggingface":
        return image_uris.retrieve(
            "huggingface-llm",
            region=region,
            version=version,
            image_scope="inference",
        )
    if backend == "huggingface-neuronx":
        return image_uris.retrieve(
            "huggingface-llm-neuronx",
            region=region,
            version=version,
            image_scope="inference",
            inference_tool="neuronx",
        )
    if backend == "huggingface-tei":
        return image_uris.retrieve(
            "huggingface-tei",
            region=region,
            version=version,
            image_scope="inference",
        )
    if backend == "huggingface-tei-cpu":
        return image_uris.retrieve(
            "huggingface-tei-cpu",
            region=region,
            version=version,
            image_scope="inference",
        )
    if backend == "lmi":
        version = version or "0.24.0"
        return image_uris.retrieve(framework="djl-deepspeed", region=region, version=version)
    raise ValueError("Unsupported backend: %s" % backend)


def get_huggingface_model_metadata(model_id: str, hf_hub_token: Optional[str] = None) -> dict:
    """Retrieves the json metadata of the HuggingFace Model via HuggingFace API.

    Args:
        model_id (str): The HuggingFace Model ID
        hf_hub_token (str): The HuggingFace Hub Token needed for Private/Gated HuggingFace Models

    Returns:
        dict: The model metadata retrieved with the HuggingFace API
    """
    if not model_id:
        raise ValueError("Model ID is empty. Please provide a valid Model ID.")
    hf_model_metadata_url = f"https://huggingface.co/api/models/{model_id}"
    hf_model_metadata_json = None
    try:
        if hf_hub_token:
            hf_model_metadata_url = urllib.request.Request(
                hf_model_metadata_url, None, {"Authorization": "Bearer " + hf_hub_token}
            )
        with urllib.request.urlopen(hf_model_metadata_url) as response:
            hf_model_metadata_json = json.load(response)
    except (HTTPError, URLError, TimeoutError, JSONDecodeError) as e:
        if "HTTP Error 401: Unauthorized" in str(e):
            raise ValueError(
                "Trying to access a gated/private HuggingFace model without valid credentials. "
                "Please provide a HUGGING_FACE_HUB_TOKEN in env_vars"
            )
        logger.warning(
            "Exception encountered while trying to retrieve HuggingFace model metadata %s. "
            "Details: %s",
            hf_model_metadata_url,
            e,
        )
    if not hf_model_metadata_json:
        raise ValueError(
            "Did not find model metadata for the following HuggingFace Model ID %s" % model_id
        )
    return hf_model_metadata_json


def download_huggingface_model_metadata(
    model_id: str, model_local_path: str, hf_hub_token: Optional[str] = None
) -> None:
    """Downloads the HuggingFace Model snapshot via HuggingFace API.

    Args:
        model_id (str): The HuggingFace Model ID
        model_local_path (str): The local path to save the HuggingFace Model snapshot.
        hf_hub_token (str): The HuggingFace Hub Token

    Raises:
        ImportError: If huggingface_hub is not installed.
    """
    if not importlib.util.find_spec("huggingface_hub"):
        raise ImportError("Unable to import huggingface_hub, check if huggingface_hub is installed")

    from huggingface_hub import snapshot_download

    os.makedirs(model_local_path, exist_ok=True)
    logger.info("Downloading model %s from Hugging Face Hub to %s", model_id, model_local_path)
    snapshot_download(repo_id=model_id, local_dir=model_local_path, token=hf_hub_token)
