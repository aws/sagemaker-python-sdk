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
import os
import tempfile
import urllib.request
from json import JSONDecodeError
from typing import Optional
from urllib.error import HTTPError, URLError
import logging

logger = logging.getLogger(__name__)


def download_huggingface_model(
    model_id: str,
    *,
    local_dir: Optional[str] = None,
    s3_uri: Optional[str] = None,
    hf_hub_token: Optional[str] = None,
    revision: Optional[str] = None,
    allow_patterns=None,
    ignore_patterns=None,
    sagemaker_session=None,
) -> str:
    """Download a HuggingFace Hub model snapshot, optionally staging it to S3.

    A supported, importable helper so notebooks and scripts don't hand-roll
    ``huggingface_hub.snapshot_download`` + ``S3Uploader``. Downloads the full
    model snapshot from the Hub, then either leaves it on local disk or uploads
    it to S3 and returns the resulting location.

    Args:
        model_id: The HuggingFace Hub model id (e.g. ``"gpt2"``).
        local_dir: Local directory to download into. When ``s3_uri`` is given
            and ``local_dir`` is omitted, the snapshot is downloaded into a
            temporary directory that is removed after the upload. Defaults to
            ``None``.
        s3_uri: Optional ``s3://bucket/prefix`` destination. When set, the
            snapshot is uploaded there and the returned value is the S3 URI.
        hf_hub_token: Optional HuggingFace Hub token for gated/private models.
        revision: Optional Hub revision (branch, tag, or commit) to pin; passed
            through to ``snapshot_download``. Defaults to the repo's default
            branch.
        allow_patterns: Optional glob(s) of files to include, passed through to
            ``snapshot_download`` (e.g. ``"*.safetensors"`` to skip duplicate
            ``.bin`` weights).
        ignore_patterns: Optional glob(s) of files to exclude, passed through to
            ``snapshot_download``.
        sagemaker_session: Optional session used for the S3 upload. Defaults to
            a new session built from the ambient AWS configuration.

    Returns:
        The S3 URI the snapshot was uploaded to when ``s3_uri`` is given,
        otherwise the local directory path the snapshot was downloaded into.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        ValueError: If neither ``local_dir`` nor ``s3_uri`` is given.
    """
    if local_dir is None and s3_uri is None:
        raise ValueError("Provide local_dir, s3_uri, or both.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "download_huggingface_model requires huggingface_hub, which is not "
            "installed. Install it with `pip install huggingface_hub`."
        ) from exc

    def _download(target: str) -> None:
        os.makedirs(target, exist_ok=True)
        logger.info("Downloading model %s from Hugging Face Hub to %s", model_id, target)
        snapshot_download(
            repo_id=model_id,
            local_dir=target,
            token=hf_hub_token,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    if s3_uri is None:
        _download(local_dir)
        return local_dir

    from sagemaker.core.s3 import S3Uploader

    def _upload(source: str) -> str:
        logger.info("Uploading model %s snapshot to %s", model_id, s3_uri)
        return S3Uploader.upload(
            local_path=source,
            desired_s3_uri=s3_uri,
            sagemaker_session=sagemaker_session,
        )

    if local_dir is not None:
        _download(local_dir)
        return _upload(local_dir)
    with tempfile.TemporaryDirectory(prefix="hf-model-") as staging_dir:
        _download(staging_dir)
        return _upload(staging_dir)


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
