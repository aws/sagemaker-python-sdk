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
# pylint: skip-file
"""This module contains accessors related to SageMaker JumpStart."""
from __future__ import absolute_import
import functools
import logging
from typing import Any, Dict, List, Optional
import boto3

from sagemaker.deprecations import deprecated
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartModelSpecs, HubContentType
from sagemaker.jumpstart.enums import JumpStartModelType
from sagemaker.jumpstart import cache
from sagemaker.jumpstart.hub.utils import (
    construct_hub_model_arn_from_inputs,
    construct_hub_model_reference_arn_from_inputs,
)
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.session import Session
from sagemaker.jumpstart import constants


class SageMakerSettings(object):
    """Static class for storing the SageMaker settings."""

    _parsed_sagemaker_version = ""

    @staticmethod
    def set_sagemaker_version(version: str) -> None:
        """Set SageMaker version."""
        SageMakerSettings._parsed_sagemaker_version = version

    @staticmethod
    def get_sagemaker_version() -> str:
        """Return SageMaker version."""
        return SageMakerSettings._parsed_sagemaker_version


class JumpStartS3PayloadAccessor(object):
    """Static class for storing and retrieving S3 payload artifacts."""

    MAX_CACHE_SIZE_BYTES = int(100 * 1e6)
    MAX_PAYLOAD_SIZE_BYTES = int(6 * 1e6)

    CACHE_SIZE = MAX_CACHE_SIZE_BYTES // MAX_PAYLOAD_SIZE_BYTES

    @staticmethod
    def clear_cache() -> None:
        """Clears LRU caches associated with S3 client and retrieved objects."""

        JumpStartS3PayloadAccessor._get_default_s3_client.cache_clear()
        JumpStartS3PayloadAccessor.get_object_cached.cache_clear()

    @staticmethod
    @functools.lru_cache()
    def _get_default_s3_client(region: str = JUMPSTART_DEFAULT_REGION_NAME) -> boto3.client:
        """Returns default S3 client associated with the region.

        Result is cached so multiple clients in memory are not created.
        """
        return boto3.client("s3", region_name=region)

    @staticmethod
    @functools.lru_cache(maxsize=CACHE_SIZE)
    def get_object_cached(
        bucket: str,
        key: str,
        region: str = JUMPSTART_DEFAULT_REGION_NAME,
        s3_client: Optional[boto3.client] = None,
    ) -> bytes:
        """Returns S3 object located at the bucket and key.

        Requests are cached so that the same S3 request is never made more
        than once, unless a different region or client is used.
        """
        return JumpStartS3PayloadAccessor.get_object(
            bucket=bucket, key=key, region=region, s3_client=s3_client
        )

    @staticmethod
    def _get_object_size_bytes(
        bucket: str,
        key: str,
        region: str = JUMPSTART_DEFAULT_REGION_NAME,
        s3_client: Optional[boto3.client] = None,
    ) -> bytes:
        """Returns size in bytes of S3 object using S3.HeadObject operation."""
        if s3_client is None:
            s3_client = JumpStartS3PayloadAccessor._get_default_s3_client(region)

        return s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"]

    @staticmethod
    def get_object(
        bucket: str,
        key: str,
        region: str = JUMPSTART_DEFAULT_REGION_NAME,
        s3_client: Optional[boto3.client] = None,
    ) -> bytes:
        """Returns S3 object located at the bucket and key.

        Raises:
            ValueError: The object size is too large.
        """
        if s3_client is None:
            s3_client = JumpStartS3PayloadAccessor._get_default_s3_client(region)

        object_size_bytes = JumpStartS3PayloadAccessor._get_object_size_bytes(
            bucket=bucket, key=key, region=region, s3_client=s3_client
        )
        if object_size_bytes > JumpStartS3PayloadAccessor.MAX_PAYLOAD_SIZE_BYTES:
            raise ValueError(
                f"s3://{bucket}/{key} has size of {object_size_bytes} bytes, "
                "which exceeds maximum allowed size of "
                f"{JumpStartS3PayloadAccessor.MAX_PAYLOAD_SIZE_BYTES} bytes."
            )

        return s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()


class JumpStartModelsAccessor(object):
    """Static class for storing the JumpStart models cache."""

    _cache: Optional[cache.JumpStartModelsCache] = None
    _curr_region = JUMPSTART_DEFAULT_REGION_NAME

    _content_bucket: Optional[str] = None
    _gated_content_bucket: Optional[str] = None

    _cache_kwargs: Dict[str, Any] = {}

    @staticmethod
    def set_jumpstart_content_bucket(content_bucket: str) -> None:
        """Sets JumpStart content bucket."""
        JumpStartModelsAccessor._content_bucket = content_bucket

    @staticmethod
    def get_jumpstart_content_bucket() -> Optional[str]:
        """Returns JumpStart content bucket."""
        return JumpStartModelsAccessor._content_bucket

    @staticmethod
    def set_jumpstart_gated_content_bucket(gated_content_bucket: str) -> None:
        """Sets JumpStart gated content bucket."""
        JumpStartModelsAccessor._gated_content_bucket = gated_content_bucket

    @staticmethod
    def get_jumpstart_gated_content_bucket() -> Optional[str]:
        """Returns JumpStart gated content bucket."""
        return JumpStartModelsAccessor._gated_content_bucket

    @staticmethod
    def _validate_and_mutate_region_cache_kwargs(
        cache_kwargs: Optional[Dict[str, Any]] = None, region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Returns cache_kwargs with region argument removed if present.

        Raises:
            ValueError: If region in `cache_kwargs` is inconsistent with `region` argument.

        Args:
            cache_kwargs (Optional[Dict[str, Any]]): cache kwargs to validate.
            region (str): The region to validate along with the kwargs.
        """
        cache_kwargs_dict = {} if cache_kwargs is None else cache_kwargs
        if region is not None and "region" in cache_kwargs_dict:
            if region != cache_kwargs_dict["region"]:
                raise ValueError(
                    f"Inconsistent region definitions: {region}, {cache_kwargs_dict['region']}"
                )
            del cache_kwargs_dict["region"]
        return cache_kwargs_dict

    @staticmethod
    def _set_cache_and_region(region: str, cache_kwargs: dict) -> None:
        """Sets ``JumpStartModelsAccessor._cache`` and ``JumpStartModelsAccessor._curr_region``.

        Args:
            region (str): region for which to retrieve header/spec.
            cache_kwargs (dict): kwargs to pass to ``JumpStartModelsCache``.
        """
        new_cache_kwargs = JumpStartModelsAccessor._validate_and_mutate_region_cache_kwargs(
            cache_kwargs, region
        )
        if (
            JumpStartModelsAccessor._cache is None
            or region != JumpStartModelsAccessor._curr_region
            or new_cache_kwargs != JumpStartModelsAccessor._cache_kwargs
        ):
            JumpStartModelsAccessor._cache = cache.JumpStartModelsCache(
                region=region, **cache_kwargs
            )
            JumpStartModelsAccessor._curr_region = region
            JumpStartModelsAccessor._cache_kwargs = new_cache_kwargs

    @staticmethod
    def _get_manifest(
        region: str = JUMPSTART_DEFAULT_REGION_NAME,
        s3_client: Optional[boto3.client] = None,
        model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    ) -> List[JumpStartModelHeader]:
        """Return entire JumpStart models manifest.

        Raises:
            ValueError: If region in `cache_kwargs` is inconsistent with `region` argument.

        Args:
            region (str): Optional. The region to use for the cache.
            s3_client (boto3.client): Optional. Boto3 client to use for accessing JumpStart models
                s3 cache. If not set, a default client will be made.
        """

        additional_kwargs = {}
        if s3_client is not None:
            additional_kwargs.update({"s3_client": s3_client})

        cache_kwargs = JumpStartModelsAccessor._validate_and_mutate_region_cache_kwargs(
            {**JumpStartModelsAccessor._cache_kwargs, **additional_kwargs},
            region,
        )
        JumpStartModelsAccessor._set_cache_and_region(region, cache_kwargs)
        return JumpStartModelsAccessor._cache.get_manifest(model_type)  # type: ignore

    @staticmethod
    def get_model_header(
        region: str,
        model_id: str,
        version: str,
        model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    ) -> JumpStartModelHeader:
        """Returns model header from JumpStart models cache.

        Args:
            region (str): region for which to retrieve header.
            model_id (str): model ID to retrieve.
            version (str): semantic version to retrieve for the model ID.
        """
        cache_kwargs = JumpStartModelsAccessor._validate_and_mutate_region_cache_kwargs(
            JumpStartModelsAccessor._cache_kwargs, region
        )
        JumpStartModelsAccessor._set_cache_and_region(region, cache_kwargs)
        return JumpStartModelsAccessor._cache.get_header(  # type: ignore
            model_id=model_id,
            semantic_version_str=version,
            model_type=model_type,
        )

    @staticmethod
    def get_model_specs(
        region: str,
        model_id: str,
        version: str,
        hub_arn: Optional[str] = None,
        s3_client: Optional[boto3.client] = None,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        sagemaker_session: Session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    ) -> JumpStartModelSpecs:
        """Returns model specs from JumpStart models cache.

        Args:
            region (str): region for which to retrieve header.
            model_id (str): model ID to retrieve.
            version (str): semantic version to retrieve for the model ID.
            s3_client (boto3.client): boto3 client to use for accessing JumpStart models s3 cache.
                If not set, a default client will be made.
        """

        additional_kwargs = {}
        if s3_client is not None:
            additional_kwargs.update({"s3_client": s3_client})

        if hub_arn:
            additional_kwargs.update({"sagemaker_session": sagemaker_session})

        cache_kwargs = JumpStartModelsAccessor._validate_and_mutate_region_cache_kwargs(
            {**JumpStartModelsAccessor._cache_kwargs, **additional_kwargs}
        )
        JumpStartModelsAccessor._set_cache_and_region(region, cache_kwargs)

        if hub_arn:
            try:
                hub_model_arn = construct_hub_model_reference_arn_from_inputs(
                    hub_arn=hub_arn, model_name=model_id, version=version
                )
                model_specs = JumpStartModelsAccessor._cache.get_hub_model_reference(
                    hub_model_reference_arn=hub_model_arn
                )
                model_specs.set_hub_content_type(HubContentType.MODEL_REFERENCE)
                return model_specs

            except Exception as ex:
                logging.info(
                    "Received exeption while calling APIs for ContentType ModelReference, \
                        retrying with ContentType Model: "
                    + str(ex)
                )
                hub_model_arn = construct_hub_model_arn_from_inputs(
                    hub_arn=hub_arn, model_name=model_id, version=version
                )
                model_specs = JumpStartModelsAccessor._cache.get_hub_model(
                    hub_model_arn=hub_model_arn
                )
                model_specs.set_hub_content_type(HubContentType.MODEL)
                return model_specs

        return JumpStartModelsAccessor._cache.get_specs(  # type: ignore
            model_id=model_id, version_str=version, model_type=model_type
        )

    @staticmethod
    def set_cache_kwargs(cache_kwargs: Dict[str, Any], region: str = None) -> None:
        """Sets cache kwargs, clears the cache.

        Raises:
            ValueError: If region in `cache_kwargs` is inconsistent with `region` argument.

        Args:
            cache_kwargs (str): cache kwargs to validate.
            region (str): Optional. The region to validate along with the kwargs.
        """
        cache_kwargs = JumpStartModelsAccessor._validate_and_mutate_region_cache_kwargs(
            cache_kwargs, region
        )
        JumpStartModelsAccessor._cache_kwargs = cache_kwargs
        if region is None:
            JumpStartModelsAccessor._cache = cache.JumpStartModelsCache(
                **JumpStartModelsAccessor._cache_kwargs
            )
        else:
            JumpStartModelsAccessor._curr_region = region
            JumpStartModelsAccessor._cache = cache.JumpStartModelsCache(
                region=region, **JumpStartModelsAccessor._cache_kwargs
            )

    @staticmethod
    def reset_cache(cache_kwargs: Dict[str, Any] = None, region: Optional[str] = None) -> None:
        """Resets cache, optionally allowing cache kwargs to be passed to the new cache.

        Raises:
            ValueError: If region in `cache_kwargs` is inconsistent with `region` argument.

        Args:
            cache_kwargs (str): cache kwargs to validate.
            region (str): The region to validate along with the kwargs.
        """
        cache_kwargs_dict = {} if cache_kwargs is None else cache_kwargs
        JumpStartModelsAccessor.set_cache_kwargs(cache_kwargs_dict, region)

    @staticmethod
    @deprecated()
    def get_manifest(
        cache_kwargs: Optional[Dict[str, Any]] = None, region: Optional[str] = None
    ) -> List[JumpStartModelHeader]:
        """Return entire JumpStart models manifest.

        Raises:
            ValueError: If region in `cache_kwargs` is inconsistent with `region` argument.

        Args:
            cache_kwargs (Dict[str, Any]): Optional. Cache kwargs to use.
                (Default: None).
            region (str): Optional. The region to use for the cache.
                (Default: None).
        """
        cache_kwargs_dict: Dict[str, Any] = {} if cache_kwargs is None else cache_kwargs
        JumpStartModelsAccessor.set_cache_kwargs(cache_kwargs_dict, region)
        return JumpStartModelsAccessor._cache.get_manifest()  # type: ignore
