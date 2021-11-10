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
import datetime
from typing import List, Optional
from sagemaker.jumpstart.types import (
    JumpStartCachedS3ContentKey,
    JumpStartCachedS3ContentValue,
    JumpStartModelHeader,
    JumpStartModelSpecs,
    JumpStartModelSpecs,
    JumpStartS3FileType,
    JumpStartVersionedModelId,
)
from sagemaker.jumpstart import utils
from sagemaker.utilities.cache import LRUCache
import boto3
import json
import semantic_version


DEFAULT_REGION_NAME = boto3.session.Session().region_name

DEFAULT_MAX_S3_CACHE_ITEMS = 20
DEFAULT_S3_CACHE_EXPIRATION_TIME = datetime.timedelta(hours=6)

DEFAULT_MAX_SEMANTIC_VERSION_CACHE_ITEMS = 20
DEFAULT_SEMANTIC_VERSION_CACHE_EXPIRATION_TIME = datetime.timedelta(hours=6)

DEFAULT_MANIFEST_FILE_S3_KEY = "models_manifest.json"


class JumpStartModelsCache:
    """Class that implements a cache for JumpStart models manifests and specs.
    The manifest and specs associated with JumpStart models provide the information necessary
    for launching JumpStart models from the SageMaker SDK.
    """

    def __init__(
        self,
        region: Optional[str] = DEFAULT_REGION_NAME,
        max_s3_cache_items: Optional[int] = DEFAULT_MAX_S3_CACHE_ITEMS,
        s3_cache_expiration_time: Optional[datetime.timedelta] = DEFAULT_S3_CACHE_EXPIRATION_TIME,
        max_semantic_version_cache_items: Optional[int] = DEFAULT_MAX_SEMANTIC_VERSION_CACHE_ITEMS,
        semantic_version_cache_expiration_time: Optional[
            datetime.timedelta
        ] = DEFAULT_SEMANTIC_VERSION_CACHE_EXPIRATION_TIME,
        manifest_file_s3_key: Optional[str] = DEFAULT_MANIFEST_FILE_S3_KEY,
        bucket: Optional[str] = None,
    ) -> None:
        """Initialize a ``JumpStartModelsCache`` instance.

        Args:
            region (Optional[str]): AWS region to associate with cache. Default: region associated
                with botocore session.
            max_s3_cache_items (Optional[int]): Maximum number of files to store in s3 cache. Default: 20.
            s3_cache_expiration_time (Optional[datetime.timedelta]): Maximum time to hold items in s3
                cache before invalidation. Default: 6 hours.
            max_semantic_version_cache_items (Optional[int]): Maximum number of files to store in
                semantic version cache. Default: 20.
            semantic_version_cache_expiration_time (Optional[datetime.timedelta]): Maximum time to hold
                items in semantic version cache before invalidation. Default: 6 hours.
            bucket (Optional[str]): S3 bucket to associate with cache. Default: JumpStart-hosted content
                bucket for region.
        """

        self._region = region
        self._s3_cache = LRUCache[JumpStartCachedS3ContentKey, JumpStartCachedS3ContentValue](
            max_cache_items=max_s3_cache_items,
            expiration_time=s3_cache_expiration_time,
            retrieval_function=self._get_file_from_s3,
        )
        self._model_id_semantic_version_manifest_key_cache = LRUCache[
            JumpStartVersionedModelId, JumpStartVersionedModelId
        ](
            max_cache_items=max_semantic_version_cache_items,
            expiration_time=semantic_version_cache_expiration_time,
            retrieval_function=self._get_manifest_key_from_model_id_semantic_version,
        )
        self._manifest_file_s3_key = manifest_file_s3_key
        self._bucket = (
            utils.get_jumpstart_content_bucket(self._region) if bucket is None else bucket
        )
        self._has_retried_cache_refresh = False

    def set_region(self, region: str) -> None:
        """Set region for cache. Clears cache after new region is set."""
        self._region = region
        self.clear()

    def get_region(self) -> str:
        """Return region for cache."""
        return self._region

    def set_manifest_file_s3_key(self, key: str) -> None:
        """Set manifest file s3 key. Clears cache after new key is set."""
        self._manifest_file_s3_key = key
        self.clear()

    def get_manifest_file_s3_key(self) -> None:
        """Return manifest file s3 key for cache."""
        return self._manifest_file_s3_key

    def set_bucket(self, bucket: str) -> None:
        """Set s3 bucket used for cache."""
        self._bucket = bucket
        self.clear()

    def get_bucket(self) -> None:
        """Return bucket used for cache."""
        return self._bucket

    def _get_manifest_key_from_model_id_semantic_version(
        self, key: JumpStartVersionedModelId, value: Optional[JumpStartVersionedModelId]
    ) -> JumpStartVersionedModelId:
        """Return model id and version in manifest that matches semantic version/id
        from customer request.

        Args:
            key (JumpStartVersionedModelId): Key for which to fetch versioned model id.
            value (Optional[JumpStartVersionedModelId]): Unused variable for current value of old cached
                model id/version.

        Raises:
            KeyError: If the semantic version is not found in the manifest.
        """

        model_id, version = key.model_id, key.version

        manifest = self._s3_cache.get(
            JumpStartCachedS3ContentKey(JumpStartS3FileType.MANIFEST, self._manifest_file_s3_key)
        ).formatted_file_content

        sm_version = utils.get_sagemaker_version()

        versions_compatible_with_sagemaker = [
            semantic_version.Version(header.version)
            for _, header in manifest.items()
            if header.model_id == model_id
            and semantic_version.Version(header.min_version) <= semantic_version.Version(sm_version)
        ]

        spec = (
            semantic_version.SimpleSpec("*")
            if version is None
            else semantic_version.SimpleSpec(version)
        )

        sm_compatible_model_version = spec.select(versions_compatible_with_sagemaker)
        if sm_compatible_model_version is not None:
            return JumpStartVersionedModelId(model_id, str(sm_compatible_model_version))
        else:
            versions_incompatible_with_sagemaker = [
                semantic_version.Version(header.version)
                for _, header in manifest.items()
                if header.model_id == model_id
            ]
            sm_incompatible_model_version = spec.select(versions_incompatible_with_sagemaker)
            if sm_incompatible_model_version is not None:
                model_version_to_use_incompatible_with_sagemaker = str(
                    sm_incompatible_model_version
                )
                sm_version_to_use = [
                    header.min_version
                    for _, header in manifest.items()
                    if header.model_id == model_id
                    and header.version == model_version_to_use_incompatible_with_sagemaker
                ]
                assert len(sm_version_to_use) == 1
                sm_version_to_use = sm_version_to_use[0]

                error_msg = (
                    f"Unable to find model manifest for {model_id} with version {version} compatible with your SageMaker version ({sm_version}). "
                    f"Consider upgrading your SageMaker library to at least version {sm_version_to_use} so you can use version "
                    f"{model_version_to_use_incompatible_with_sagemaker} of {model_id}."
                )
                raise KeyError(error_msg)
            else:
                error_msg = f"Unable to find model manifest for {model_id} with version {version}"
                raise KeyError(error_msg)

    def _get_file_from_s3(
        self,
        key: JumpStartCachedS3ContentKey,
        value: Optional[JumpStartCachedS3ContentValue],
    ) -> JumpStartCachedS3ContentValue:
        """Return s3 content given a file type and s3_key in ``JumpStartCachedS3ContentKey``.
        If a manifest file is being fetched, we only download the object if the md5 hash in
        ``head_object`` does not match the current md5 hash for the stored value. This prevents
        unnecessarily downloading the full manifest when it hasn't changed.

        Args:
            key (JumpStartCachedS3ContentKey): key for which to fetch s3 content.
            value (Optional[JumpStartVersionedModelId]): Current value of old cached
                s3 content. This is used for the manifest file, so that it is only
                downloaded when its content changes.
        """

        file_type, s3_key = key.file_type, key.s3_key

        s3_client = boto3.client("s3", region_name=self._region)

        if file_type == JumpStartS3FileType.MANIFEST:
            etag = s3_client.head_object(Bucket=self._bucket, Key=s3_key)["ETag"]
            if value is not None and etag == value.md5_hash:
                return value
            response = s3_client.get_object(Bucket=self._bucket, Key=s3_key)
            formatted_body = json.loads(response["Body"].read().decode("utf-8"))
            return JumpStartCachedS3ContentValue(
                formatted_file_content=utils.get_formatted_manifest(formatted_body),
                md5_hash=etag,
            )
        if file_type == JumpStartS3FileType.SPECS:
            response = s3_client.get_object(Bucket=self._bucket, Key=s3_key)
            formatted_body = json.loads(response["Body"].read().decode("utf-8"))
            return JumpStartCachedS3ContentValue(
                formatted_file_content=JumpStartModelSpecs(formatted_body)
            )
        raise RuntimeError(f"Bad value for key: {key}")

    def get_header(
        self, model_id: str, semantic_version: Optional[str] = None
    ) -> List[JumpStartModelHeader]:
        """Return list of headers for a given JumpStart model id and semantic version.

        Args:
            model_id (str): model id for which to get a header.
            semantic_version (Optional[str]): The semantic version for which to get a header.
                If None, the highest compatible version is returned.
        """

        versioned_model_id = self._model_id_semantic_version_manifest_key_cache.get(
            JumpStartVersionedModelId(model_id, semantic_version)
        )
        manifest = self._s3_cache.get(
            JumpStartCachedS3ContentKey(JumpStartS3FileType.MANIFEST, self._manifest_file_s3_key)
        ).formatted_file_content
        try:
            header = manifest[versioned_model_id]
            if self._has_retried_cache_refresh:
                self._has_retried_cache_refresh = False
            return header
        except KeyError:
            if self._has_retried_cache_refresh:
                self._has_retried_cache_refresh = False
                raise
            self.clear()
            self._has_retried_cache_refresh = True
            return self.get_header(model_id, semantic_version)

    def get_specs(
        self, model_id: str, semantic_version: Optional[str] = None
    ) -> JumpStartModelSpecs:
        """Return specs for a given JumpStart model id and semantic version.

        Args:
            model_id (str): model id for which to get specs.
            semantic_version (Optional[str]): The semantic version for which to get specs.
                If None, the highest compatible version is returned.
        """
        header = self.get_header(model_id, semantic_version)
        spec_key = header.spec_key
        return self._s3_cache.get(
            JumpStartCachedS3ContentKey(JumpStartS3FileType.SPECS, spec_key)
        ).formatted_file_content

    def clear(self) -> None:
        """Clears the model id/version and s3 cache and resets ``_has_retried_cache_refresh``."""
        self._s3_cache.clear()
        self._model_id_semantic_version_manifest_key_cache.clear()
        self._has_retried_cache_refresh = False
