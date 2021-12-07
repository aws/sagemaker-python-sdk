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
"""This module contains accessors related to SageMaker JumpStart."""
from __future__ import absolute_import
from typing import Any, Dict, Optional
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartModelSpecs
from sagemaker.jumpstart import cache
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME


class SageMakerSettings(object):
    """Static class for storing the SageMaker settings."""

    _PARSED_SAGEMAKER_VERSION = ""

    @staticmethod
    def set_sagemaker_version(version: str) -> None:
        """Set SageMaker version."""
        SageMakerSettings._PARSED_SAGEMAKER_VERSION = version

    @staticmethod
    def get_sagemaker_version() -> str:
        """Return SageMaker version."""
        return SageMakerSettings._PARSED_SAGEMAKER_VERSION


class JumpStartModelsCache(object):
    """Static class for storing the JumpStart models cache."""

    _cache: Optional[cache.JumpStartModelsCache] = None
    _curr_region = JUMPSTART_DEFAULT_REGION_NAME

    _cache_kwargs = {}

    def _validate_region_cache_kwargs(
        cache_kwargs: Dict[str, Any] = {}, region: Optional[str] = None
    ):
        if region is not None and "region" in cache_kwargs:
            if region != cache_kwargs["region"]:
                raise ValueError(
                    f"Inconsistent region definitions: {region}, {cache_kwargs['region']}"
                )
            del cache_kwargs["region"]
        return cache_kwargs

    @staticmethod
    def get_model_header(region: str, model_id: str, version: str) -> JumpStartModelHeader:
        cache_kwargs = JumpStartModelsCache._validate_region_cache_kwargs(
            JumpStartModelsCache._cache_kwargs, region
        )
        if JumpStartModelsCache._cache == None or region != JumpStartModelsCache._curr_region:
            JumpStartModelsCache._cache = cache.JumpStartModelsCache(region=region, **cache_kwargs)
            JumpStartModelsCache._curr_region = region
        return JumpStartModelsCache._cache.get_header(model_id, version)

    @staticmethod
    def get_model_specs(region: str, model_id: str, version: str) -> JumpStartModelSpecs:
        cache_kwargs = JumpStartModelsCache._validate_region_cache_kwargs(
            JumpStartModelsCache._cache_kwargs, region
        )
        if JumpStartModelsCache._cache == None or region != JumpStartModelsCache._curr_region:
            JumpStartModelsCache._cache = cache.JumpStartModelsCache(region=region, **cache_kwargs)
            JumpStartModelsCache._curr_region = region
        return JumpStartModelsCache._cache.get_specs(model_id, version)

    @staticmethod
    def set_cache_kwargs(cache_kwargs: Dict[str, Any], region: str = None) -> None:
        cache_kwargs = JumpStartModelsCache._validate_region_cache_kwargs(cache_kwargs, region)
        JumpStartModelsCache._cache_kwargs = cache_kwargs
        if region is None:
            JumpStartModelsCache._cache = cache.JumpStartModelsCache(
                **JumpStartModelsCache._cache_kwargs
            )
        else:
            JumpStartModelsCache._curr_region = region
            JumpStartModelsCache._cache = cache.JumpStartModelsCache(
                region=region, **JumpStartModelsCache._cache_kwargs
            )

    @staticmethod
    def reset_cache(cache_kwargs: Dict[str, Any] = {}, region: str = None) -> None:
        cache_kwargs = JumpStartModelsCache._validate_region_cache_kwargs(cache_kwargs, region)
        JumpStartModelsCache._cache_kwargs = cache_kwargs
        if region is None:
            JumpStartModelsCache._cache = cache.JumpStartModelsCache(
                **JumpStartModelsCache._cache_kwargs
            )
        else:
            JumpStartModelsCache._curr_region = region
            JumpStartModelsCache._cache = cache.JumpStartModelsCache(
                region=region, **JumpStartModelsCache._cache_kwargs
            )
