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
"""This module contains utilities related to SageMaker JumpStart."""
from __future__ import absolute_import
from copy import copy
import logging
import os
from typing import Any, Dict, List, Set, Optional, Tuple, Union
from urllib.parse import urlparse
import boto3
from packaging.version import Version
import botocore
import sagemaker
from sagemaker.config.config_schema import (
    MODEL_ENABLE_NETWORK_ISOLATION_PATH,
    MODEL_EXECUTION_ROLE_ARN_PATH,
    TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
    TRAINING_JOB_ROLE_ARN_PATH,
)

from sagemaker.jumpstart import constants, enums
from sagemaker.jumpstart import accessors
from sagemaker.s3 import parse_s3_url
from sagemaker.jumpstart.exceptions import (
    DeprecatedJumpStartModelError,
    VulnerableJumpStartModelError,
    get_old_model_version_msg,
)
from sagemaker.jumpstart.types import (
    JumpStartBenchmarkStat,
    JumpStartMetadataConfig,
    JumpStartModelHeader,
    JumpStartModelSpecs,
    JumpStartVersionedModelId,
)
from sagemaker.session import Session
from sagemaker.config import load_sagemaker_config
from sagemaker.utils import resolve_value_from_config, TagsDict
from sagemaker.workflow import is_pipeline_variable
from sagemaker.user_agent import get_user_agent_extra_suffix


def get_jumpstart_launched_regions_message() -> str:
    """Returns formatted string indicating where JumpStart is launched."""
    if len(constants.JUMPSTART_REGION_NAME_SET) == 0:
        return "JumpStart is not available in any region."
    if len(constants.JUMPSTART_REGION_NAME_SET) == 1:
        region = list(constants.JUMPSTART_REGION_NAME_SET)[0]
        return f"JumpStart is available in {region} region."

    sorted_regions = sorted(list(constants.JUMPSTART_REGION_NAME_SET))
    if len(constants.JUMPSTART_REGION_NAME_SET) == 2:
        return f"JumpStart is available in {sorted_regions[0]} and {sorted_regions[1]} regions."

    formatted_launched_regions_list = []
    for i, region in enumerate(sorted_regions):
        region_prefix = "" if i < len(sorted_regions) - 1 else "and "
        formatted_launched_regions_list.append(region_prefix + region)
    formatted_launched_regions_str = ", ".join(formatted_launched_regions_list)
    return f"JumpStart is available in {formatted_launched_regions_str} regions."


def get_jumpstart_gated_content_bucket(
    region: str = constants.JUMPSTART_DEFAULT_REGION_NAME,
) -> str:
    """Returns regionalized private content bucket name for JumpStart.

    Raises:
        ValueError: If JumpStart is not launched in ``region`` or private content
            unavailable in that region.
    """

    old_gated_content_bucket: Optional[str] = (
        accessors.JumpStartModelsAccessor.get_jumpstart_gated_content_bucket()
    )

    info_logs: List[str] = []

    gated_bucket_to_return: Optional[str] = None
    if (
        constants.ENV_VARIABLE_JUMPSTART_GATED_CONTENT_BUCKET_OVERRIDE in os.environ
        and len(os.environ[constants.ENV_VARIABLE_JUMPSTART_GATED_CONTENT_BUCKET_OVERRIDE]) > 0
    ):
        gated_bucket_to_return = os.environ[
            constants.ENV_VARIABLE_JUMPSTART_GATED_CONTENT_BUCKET_OVERRIDE
        ]
        info_logs.append(f"Using JumpStart gated bucket override: '{gated_bucket_to_return}'")
    else:
        try:
            gated_bucket_to_return = constants.JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT[
                region
            ].gated_content_bucket
            if gated_bucket_to_return is None:
                raise ValueError(
                    f"No private content bucket for JumpStart exists in {region} region."
                )
        except KeyError:
            formatted_launched_regions_str = get_jumpstart_launched_regions_message()
            raise ValueError(
                f"Unable to get private content bucket for JumpStart in {region} region. "
                f"{formatted_launched_regions_str}"
            )

    accessors.JumpStartModelsAccessor.set_jumpstart_gated_content_bucket(gated_bucket_to_return)

    if gated_bucket_to_return != old_gated_content_bucket:
        if old_gated_content_bucket is not None:
            accessors.JumpStartModelsAccessor.reset_cache()
        for info_log in info_logs:
            constants.JUMPSTART_LOGGER.info(info_log)

    return gated_bucket_to_return


def get_jumpstart_content_bucket(
    region: str = constants.JUMPSTART_DEFAULT_REGION_NAME,
) -> str:
    """Returns the regionalized content bucket name for JumpStart.

    Raises:
        ValueError: If JumpStart is not launched in ``region``.
    """

    old_content_bucket: Optional[str] = (
        accessors.JumpStartModelsAccessor.get_jumpstart_content_bucket()
    )

    info_logs: List[str] = []

    bucket_to_return: Optional[str] = None
    if (
        constants.ENV_VARIABLE_JUMPSTART_CONTENT_BUCKET_OVERRIDE in os.environ
        and len(os.environ[constants.ENV_VARIABLE_JUMPSTART_CONTENT_BUCKET_OVERRIDE]) > 0
    ):
        bucket_to_return = os.environ[constants.ENV_VARIABLE_JUMPSTART_CONTENT_BUCKET_OVERRIDE]
        info_logs.append(f"Using JumpStart bucket override: '{bucket_to_return}'")
    else:
        try:
            bucket_to_return = constants.JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT[
                region
            ].content_bucket
        except KeyError:
            formatted_launched_regions_str = get_jumpstart_launched_regions_message()
            raise ValueError(
                f"Unable to get content bucket for JumpStart in {region} region. "
                f"{formatted_launched_regions_str}"
            )

    accessors.JumpStartModelsAccessor.set_jumpstart_content_bucket(bucket_to_return)

    if bucket_to_return != old_content_bucket:
        if old_content_bucket is not None:
            accessors.JumpStartModelsAccessor.reset_cache()
        for info_log in info_logs:
            constants.JUMPSTART_LOGGER.info(info_log)
    return bucket_to_return


def get_formatted_manifest(
    manifest: List[Dict],
) -> Dict[JumpStartVersionedModelId, JumpStartModelHeader]:
    """Returns formatted manifest dictionary from raw manifest.

    Keys are JumpStartVersionedModelId objects, values are
    ``JumpStartModelHeader`` objects.
    """
    manifest_dict = {}
    for header in manifest:
        header_obj = JumpStartModelHeader(header)
        manifest_dict[JumpStartVersionedModelId(header_obj.model_id, header_obj.version)] = (
            header_obj
        )
    return manifest_dict


def get_sagemaker_version() -> str:
    """Returns sagemaker library version.

    If the sagemaker library version has not been set, this function
    calls ``parse_sagemaker_version`` to retrieve the version and set
    the constant.
    """
    if accessors.SageMakerSettings.get_sagemaker_version() == "":
        accessors.SageMakerSettings.set_sagemaker_version(parse_sagemaker_version())
    return accessors.SageMakerSettings.get_sagemaker_version()


def parse_sagemaker_version() -> str:
    """Returns sagemaker library version. This should only be called once.

    Function reads ``__version__`` variable in ``sagemaker`` module.
    In order to maintain compatibility with the ``packaging.version``
    library, versions with fewer than 2, or more than 3, periods are rejected.
    All versions that cannot be parsed with ``packaging.version`` are also
    rejected.

    Raises:
        RuntimeError: If the SageMaker version is not readable. An exception is also raised if
        the version cannot be parsed by ``packaging.version``.
    """
    version = sagemaker.__version__
    parsed_version = None

    num_periods = version.count(".")
    if num_periods == 2:
        parsed_version = version
    elif num_periods == 3:
        trailing_period_index = version.rfind(".")
        parsed_version = version[:trailing_period_index]
    else:
        raise RuntimeError(f"Bad value for SageMaker version: {sagemaker.__version__}")

    Version(parsed_version)

    return parsed_version


def is_jumpstart_model_input(model_id: Optional[str], version: Optional[str]) -> bool:
    """Determines if `model_id` and `version` input are for JumpStart.

    This method returns True if both arguments are not None, false if both arguments
    are None, and raises an exception if one argument is None but the other isn't.

    Args:
        model_id (str): Optional. Model ID of the JumpStart model.
        version (str): Optional. Version of the JumpStart model.

    Raises:
        ValueError: If only one of the two arguments is None.
    """
    if model_id is not None or version is not None:
        if model_id is None or version is None:
            raise ValueError(
                "Must specify JumpStart `model_id` and `model_version` when getting specs for "
                "JumpStart models."
            )
        return True
    return False


def is_jumpstart_model_uri(uri: Optional[str]) -> bool:
    """Returns True if URI corresponds to a JumpStart-hosted model.

    Args:
        uri (Optional[str]): uri for inference/training job.
    """

    bucket = None
    if urlparse(uri).scheme == "s3":
        bucket, _ = parse_s3_url(uri)

    return bucket in constants.JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET


def tag_key_in_array(tag_key: str, tag_array: List[Dict[str, str]]) -> bool:
    """Returns True if ``tag_key`` is in the ``tag_array``.

    Args:
        tag_key (str): the tag key to check if it's already in the ``tag_array``.
        tag_array (List[Dict[str, str]]): array of tags to check for ``tag_key``.
    """
    for tag in tag_array:
        if tag_key == tag["Key"]:
            return True
    return False


def get_tag_value(tag_key: str, tag_array: List[Dict[str, str]]) -> str:
    """Return the value of a tag whose key matches the given ``tag_key``.

    Args:
        tag_key (str): AWS tag for which to search.
        tag_array (List[Dict[str, str]]): List of AWS tags, each formatted as dicts.

    Raises:
        KeyError: If the number of matches for the ``tag_key`` is not equal to 1.
    """
    tag_values = [tag["Value"] for tag in tag_array if tag_key == tag["Key"]]
    if len(tag_values) != 1:
        raise KeyError(
            f"Cannot get value of tag for tag key '{tag_key}' -- found {len(tag_values)} "
            f"number of matches in the tag list."
        )

    return tag_values[0]


def add_single_jumpstart_tag(
    tag_value: str,
    tag_key: enums.JumpStartTag,
    curr_tags: Optional[List[Dict[str, str]]],
    is_uri=False,
) -> Optional[List]:
    """Adds ``tag_key`` to ``curr_tags`` if ``uri`` corresponds to a JumpStart model.

    Args:
        uri (str): URI which may correspond to a JumpStart model.
        tag_key (enums.JumpStartTag): Custom tag to apply to current tags if the URI
            corresponds to a JumpStart model.
        curr_tags (Optional[List]): Current tags associated with ``Estimator`` or ``Model``.
        is_uri (boolean): Set to True to indicate a s3 uri is to be tagged. Set to False to indicate
            tags for JumpStart model id / version are being added. (Default: False).
    """
    if not is_uri or is_jumpstart_model_uri(tag_value):
        if curr_tags is None:
            curr_tags = []
        if not tag_key_in_array(tag_key, curr_tags):
            skip_adding_tag = (
                (
                    tag_key_in_array(enums.JumpStartTag.MODEL_ID, curr_tags)
                    or tag_key_in_array(enums.JumpStartTag.MODEL_VERSION, curr_tags)
                    or tag_key_in_array(enums.JumpStartTag.MODEL_TYPE, curr_tags)
                )
                if is_uri
                else False
            )
            if not skip_adding_tag:
                curr_tags.append(
                    {
                        "Key": tag_key,
                        "Value": tag_value,
                    }
                )
    return curr_tags


def get_jumpstart_base_name_if_jumpstart_model(
    *uris: Optional[str],
) -> Optional[str]:
    """Return default JumpStart base name if a URI belongs to JumpStart.

    If no URIs belong to JumpStart, return None.

    Args:
        *uris (Optional[str]): URI to test for association with JumpStart.
    """
    for uri in uris:
        if is_jumpstart_model_uri(uri):
            return constants.JUMPSTART_RESOURCE_BASE_NAME
    return None


def add_jumpstart_model_id_version_tags(
    tags: Optional[List[TagsDict]],
    model_id: str,
    model_version: str,
    model_type: Optional[enums.JumpStartModelType] = None,
) -> List[TagsDict]:
    """Add custom model ID and version tags to JumpStart related resources."""
    if model_id is None or model_version is None:
        return tags
    tags = add_single_jumpstart_tag(
        model_id,
        enums.JumpStartTag.MODEL_ID,
        tags,
        is_uri=False,
    )
    tags = add_single_jumpstart_tag(
        model_version,
        enums.JumpStartTag.MODEL_VERSION,
        tags,
        is_uri=False,
    )
    if model_type == enums.JumpStartModelType.PROPRIETARY:
        tags = add_single_jumpstart_tag(
            enums.JumpStartModelType.PROPRIETARY.value,
            enums.JumpStartTag.MODEL_TYPE,
            tags,
            is_uri=False,
        )
    return tags


def add_hub_content_arn_tags(
    tags: Optional[List[TagsDict]],
    hub_arn: str,
) -> Optional[List[TagsDict]]:
    """Adds custom Hub arn tag to JumpStart related resources."""

    tags = add_single_jumpstart_tag(
        hub_arn,
        enums.JumpStartTag.HUB_CONTENT_ARN,
        tags,
        is_uri=False,
    )
    return tags


def add_jumpstart_uri_tags(
    tags: Optional[List[TagsDict]] = None,
    inference_model_uri: Optional[Union[str, dict]] = None,
    inference_script_uri: Optional[str] = None,
    training_model_uri: Optional[str] = None,
    training_script_uri: Optional[str] = None,
) -> Optional[List[TagsDict]]:
    """Add custom uri tags to JumpStart models, return the updated tags.

    No-op if this is not a JumpStart model related resource.

    Args:
        tags (Optional[List[Dict[str,str]]): Current tags for JumpStart inference
            or training job. (Default: None).
        inference_model_uri (Optional[Union[dict, str]]): S3 URI for inference model artifact.
            (Default: None).
        inference_script_uri (Optional[str]): S3 URI for inference script tarball.
            (Default: None).
        training_model_uri (Optional[str]): S3 URI for training model artifact.
            (Default: None).
        training_script_uri (Optional[str]): S3 URI for training script tarball.
            (Default: None).
    """
    warn_msg = (
        "The URI (%s) is a pipeline variable which is only interpreted at execution time. "
        "As a result, the JumpStart resources will not be tagged."
    )

    if isinstance(inference_model_uri, dict):
        inference_model_uri = inference_model_uri.get("S3DataSource", {}).get("S3Uri", None)

    if inference_model_uri:
        if is_pipeline_variable(inference_model_uri):
            logging.warning(warn_msg, "inference_model_uri")
        else:
            tags = add_single_jumpstart_tag(
                inference_model_uri,
                enums.JumpStartTag.INFERENCE_MODEL_URI,
                tags,
                is_uri=True,
            )

    if inference_script_uri:
        if is_pipeline_variable(inference_script_uri):
            logging.warning(warn_msg, "inference_script_uri")
        else:
            tags = add_single_jumpstart_tag(
                inference_script_uri,
                enums.JumpStartTag.INFERENCE_SCRIPT_URI,
                tags,
                is_uri=True,
            )

    if training_model_uri:
        if is_pipeline_variable(training_model_uri):
            logging.warning(warn_msg, "training_model_uri")
        else:
            tags = add_single_jumpstart_tag(
                training_model_uri,
                enums.JumpStartTag.TRAINING_MODEL_URI,
                tags,
                is_uri=True,
            )

    if training_script_uri:
        if is_pipeline_variable(training_script_uri):
            logging.warning(warn_msg, "training_script_uri")
        else:
            tags = add_single_jumpstart_tag(
                training_script_uri,
                enums.JumpStartTag.TRAINING_SCRIPT_URI,
                tags,
                is_uri=True,
            )

    return tags


def update_inference_tags_with_jumpstart_training_tags(
    inference_tags: Optional[List[Dict[str, str]]], training_tags: Optional[List[Dict[str, str]]]
) -> Optional[List[Dict[str, str]]]:
    """Updates the tags for the ``sagemaker.model.Model.deploy`` command with any JumpStart tags.

    Args:
        inference_tags (Optional[List[Dict[str, str]]]): Custom tags to appy to inference job.
        training_tags (Optional[List[Dict[str, str]]]): Tags from training job.
    """
    if training_tags:
        for tag_key in enums.JumpStartTag:
            if tag_key_in_array(tag_key, training_tags):
                tag_value = get_tag_value(tag_key, training_tags)
                if inference_tags is None:
                    inference_tags = []
                if not tag_key_in_array(tag_key, inference_tags):
                    inference_tags.append({"Key": tag_key, "Value": tag_value})

    return inference_tags


def get_eula_message(model_specs: JumpStartModelSpecs, region: str) -> str:
    """Returns EULA message to display if one is available, else empty string."""
    if model_specs.hosting_eula_key is None:
        return ""
    return (
        f"Model '{model_specs.model_id}' requires accepting end-user license agreement (EULA). "
        f"See https://{get_jumpstart_content_bucket(region=region)}.s3.{region}."
        f"amazonaws.com{'.cn' if region.startswith('cn-') else ''}"
        f"/{model_specs.hosting_eula_key} for terms of use."
    )


def emit_logs_based_on_model_specs(
    model_specs: JumpStartModelSpecs, region: str, s3_client: boto3.client
) -> None:
    """Emits logs based on model specs and region."""

    if model_specs.hosting_eula_key:
        constants.JUMPSTART_LOGGER.info(get_eula_message(model_specs, region))

    full_version: str = model_specs.version

    models_manifest_list = accessors.JumpStartModelsAccessor._get_manifest(
        region=region, s3_client=s3_client
    )
    max_version_for_model_id: Optional[str] = None
    for header in models_manifest_list:
        if header.model_id == model_specs.model_id:
            if max_version_for_model_id is None or Version(header.version) > Version(
                max_version_for_model_id
            ):
                max_version_for_model_id = header.version

    if full_version != max_version_for_model_id:
        constants.JUMPSTART_LOGGER.info(
            get_old_model_version_msg(model_specs.model_id, full_version, max_version_for_model_id)
        )

    if model_specs.deprecated:
        deprecated_message = model_specs.deprecated_message or (
            "Using deprecated JumpStart model "
            f"'{model_specs.model_id}' and version '{model_specs.version}'."
        )

        constants.JUMPSTART_LOGGER.warning(deprecated_message)

    if model_specs.deprecate_warn_message:
        constants.JUMPSTART_LOGGER.warning(model_specs.deprecate_warn_message)

    if model_specs.usage_info_message:
        constants.JUMPSTART_LOGGER.info(model_specs.usage_info_message)

    if model_specs.inference_vulnerable or model_specs.training_vulnerable:
        constants.JUMPSTART_LOGGER.warning(
            "Using vulnerable JumpStart model '%s' and version '%s'.",
            model_specs.model_id,
            model_specs.version,
        )


def verify_model_region_and_return_specs(
    model_id: Optional[str],
    version: Optional[str],
    scope: Optional[str],
    region: Optional[str] = None,
    hub_arn: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: enums.JumpStartModelType = enums.JumpStartModelType.OPEN_WEIGHTS,
) -> JumpStartModelSpecs:
    """Verifies that an acceptable model_id, version, scope, and region combination is provided.

    Args:
        model_id (Optional[str]): model ID of the JumpStart model to verify and
            obtains specs.
        version (Optional[str]): version of the JumpStart model to verify and
            obtains specs.
        scope (Optional[str]): scope of the JumpStart model to verify.
        region (Optional[str]): region of the JumpStart model to verify and
            obtains specs.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated models should be tolerated
            (exception not raised). False if these models should raise an exception.
            (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).

    Raises:
        NotImplementedError: If the scope is not supported.
        ValueError: If the combination of arguments specified is not supported.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    if scope is None:
        raise ValueError(
            "Must specify `model_scope` argument to retrieve model "
            "artifact uri for JumpStart models."
        )

    if scope not in constants.SUPPORTED_JUMPSTART_SCOPES:
        raise NotImplementedError(
            "JumpStart models only support scopes: "
            f"{', '.join(constants.SUPPORTED_JUMPSTART_SCOPES)}."
        )

    model_specs = accessors.JumpStartModelsAccessor.get_model_specs(  # type: ignore
        region=region,
        model_id=model_id,
        hub_arn=hub_arn,
        version=version,
        s3_client=sagemaker_session.s3_client,
        model_type=model_type,
        sagemaker_session=sagemaker_session,
    )

    if (
        scope == constants.JumpStartScriptScope.TRAINING.value
        and not model_specs.training_supported
    ):
        raise ValueError(
            f"JumpStart model ID '{model_id}' and version '{version}' " "does not support training."
        )

    if model_specs.deprecated:
        if not tolerate_deprecated_model:
            raise DeprecatedJumpStartModelError(
                model_id=model_id, version=version, message=model_specs.deprecated_message
            )

    if scope == constants.JumpStartScriptScope.INFERENCE.value and model_specs.inference_vulnerable:
        if not tolerate_vulnerable_model:
            raise VulnerableJumpStartModelError(
                model_id=model_id,
                version=version,
                vulnerabilities=model_specs.inference_vulnerabilities,
                scope=constants.JumpStartScriptScope.INFERENCE,
            )

    if scope == constants.JumpStartScriptScope.TRAINING.value and model_specs.training_vulnerable:
        if not tolerate_vulnerable_model:
            raise VulnerableJumpStartModelError(
                model_id=model_id,
                version=version,
                vulnerabilities=model_specs.training_vulnerabilities,
                scope=constants.JumpStartScriptScope.TRAINING,
            )

    return model_specs


def update_dict_if_key_not_present(
    dict_to_update: Optional[dict], key_to_add: Any, value_to_add: Any
) -> Optional[dict]:
    """If a key is not present in the dict, add the new (key, value) pair, and return dict.

    If dict is empty, return None.
    """
    if dict_to_update is None:
        dict_to_update = {}
    if key_to_add not in dict_to_update:
        dict_to_update[key_to_add] = value_to_add
    if dict_to_update == {}:
        dict_to_update = None

    return dict_to_update


def resolve_model_sagemaker_config_field(
    field_name: str,
    field_val: Optional[Any],
    sagemaker_session: Session,
    default_value: Optional[str] = None,
) -> Any:
    """Given a field name, checks if there is a sagemaker config value to set.

    For the role field, which is customer-supplied, we allow ``field_val`` to take precedence
    over sagemaker config values. For all other fields, sagemaker config values take precedence
    over the JumpStart default fields.
    """
    # In case, sagemaker_session is None, get sagemaker_config from load_sagemaker_config()
    # to resolve value from config for the respective field_name parameter
    _sagemaker_config = load_sagemaker_config() if (sagemaker_session is None) else None

    # We allow customers to define a role which takes precedence
    # over the one defined in sagemaker config
    if field_name == "role":
        return resolve_value_from_config(
            direct_input=field_val,
            config_path=MODEL_EXECUTION_ROLE_ARN_PATH,
            default_value=default_value or sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=sagemaker_session,
            sagemaker_config=_sagemaker_config,
        )

    # JumpStart Models have certain default field values. We want
    # sagemaker config values to take priority over the model-specific defaults.
    if field_name == "enable_network_isolation":
        resolved_val = resolve_value_from_config(
            direct_input=None,
            config_path=MODEL_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=sagemaker_session,
            default_value=default_value,
            sagemaker_config=_sagemaker_config,
        )
        return resolved_val if resolved_val is not None else field_val

    # field is not covered by sagemaker config so return as is
    return field_val


def resolve_estimator_sagemaker_config_field(
    field_name: str,
    field_val: Optional[Any],
    sagemaker_session: Session,
    default_value: Optional[str] = None,
) -> Any:
    """Given a field name, checks if there is a sagemaker config value to set.

    For the role field, which is customer-supplied, we allow ``field_val`` to take precedence
    over sagemaker config values. For all other fields, sagemaker config values take precedence
    over the JumpStart default fields.
    """

    # Workaround for config injection if sagemaker_session is None, since in
    # that case sagemaker_session will not be initialized until
    # `_init_sagemaker_session_if_does_not_exist` is called later
    _sagemaker_config = load_sagemaker_config() if (sagemaker_session is None) else None

    # We allow customers to define a role which takes precedence
    # over the one defined in sagemaker config
    if field_name == "role":
        return resolve_value_from_config(
            direct_input=field_val,
            config_path=TRAINING_JOB_ROLE_ARN_PATH,
            default_value=default_value or sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=sagemaker_session,
            sagemaker_config=_sagemaker_config,
        )

    # JumpStart Estimators have certain default field values. We want
    # sagemaker config values to take priority over the model-specific defaults.
    if field_name == "enable_network_isolation":

        resolved_val = resolve_value_from_config(
            direct_input=None,
            config_path=TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=sagemaker_session,
            default_value=default_value,
            sagemaker_config=_sagemaker_config,
        )
        return resolved_val if resolved_val is not None else field_val

    if field_name == "encrypt_inter_container_traffic":

        resolved_val = resolve_value_from_config(
            direct_input=None,
            config_path=TRAINING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
            sagemaker_session=sagemaker_session,
            default_value=default_value,
            sagemaker_config=_sagemaker_config,
        )
        return resolved_val if resolved_val is not None else field_val

    # field is not covered by sagemaker config so return as is
    return field_val


def validate_model_id_and_get_type(
    model_id: Optional[str],
    region: Optional[str] = None,
    model_version: Optional[str] = None,
    script: enums.JumpStartScriptScope = enums.JumpStartScriptScope.INFERENCE,
    sagemaker_session: Optional[Session] = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    hub_arn: Optional[str] = None,
) -> Optional[enums.JumpStartModelType]:
    """Returns model type if the model ID is supported for the given script.

    Raises:
        ValueError: If the script is not supported by JumpStart.
    """

    if model_id in {None, ""}:
        return None
    if not isinstance(model_id, str):
        return None
    if hub_arn:
        return None

    s3_client = sagemaker_session.s3_client if sagemaker_session else None
    region = region or constants.JUMPSTART_DEFAULT_REGION_NAME
    model_version = model_version or "*"
    models_manifest_list = accessors.JumpStartModelsAccessor._get_manifest(
        region=region, s3_client=s3_client, model_type=enums.JumpStartModelType.OPEN_WEIGHTS
    )
    open_weight_model_id_set = {model.model_id for model in models_manifest_list}

    if model_id in open_weight_model_id_set:
        return enums.JumpStartModelType.OPEN_WEIGHTS

    proprietary_manifest_list = accessors.JumpStartModelsAccessor._get_manifest(
        region=region, s3_client=s3_client, model_type=enums.JumpStartModelType.PROPRIETARY
    )

    proprietary_model_id_set = {model.model_id for model in proprietary_manifest_list}
    if model_id in proprietary_model_id_set:
        if script == enums.JumpStartScriptScope.INFERENCE:
            return enums.JumpStartModelType.PROPRIETARY
        raise ValueError(f"Unsupported script for Proprietary models: {script}")
    return None


def get_jumpstart_model_id_version_from_resource_arn(
    resource_arn: str,
    sagemaker_session: Session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Tuple[Optional[str], Optional[str]]:
    """Returns the JumpStart model ID and version if in resource tags.

    Returns 'None' if model ID or version cannot be inferred from tags.
    """

    list_tags_result = sagemaker_session.list_tags(resource_arn)

    model_id: Optional[str] = None
    model_version: Optional[str] = None

    model_id_keys = [enums.JumpStartTag.MODEL_ID, *constants.EXTRA_MODEL_ID_TAGS]
    model_version_keys = [enums.JumpStartTag.MODEL_VERSION, *constants.EXTRA_MODEL_VERSION_TAGS]

    for model_id_key in model_id_keys:
        try:
            model_id_from_tag = get_tag_value(model_id_key, list_tags_result)
        except KeyError:
            continue
        if model_id_from_tag is not None:
            if model_id is not None and model_id_from_tag != model_id:
                constants.JUMPSTART_LOGGER.warning(
                    "Found multiple model ID tags on the following resource: %s", resource_arn
                )
                model_id = None
                break
            model_id = model_id_from_tag

    for model_version_key in model_version_keys:
        try:
            model_version_from_tag = get_tag_value(model_version_key, list_tags_result)
        except KeyError:
            continue
        if model_version_from_tag is not None:
            if model_version is not None and model_version_from_tag != model_version:
                constants.JUMPSTART_LOGGER.warning(
                    "Found multiple model version tags on the following resource: %s", resource_arn
                )
                model_version = None
                break
            model_version = model_version_from_tag

    return model_id, model_version


def get_region_fallback(
    s3_bucket_name: Optional[str] = None,
    s3_client: Optional[boto3.client] = None,
    sagemaker_session: Optional[Session] = None,
) -> str:
    """Returns region to use for JumpStart functionality implicitly via session objects."""
    regions_in_s3_bucket_name: Set[str] = {
        region
        for region in constants.JUMPSTART_REGION_NAME_SET
        if s3_bucket_name is not None
        if region in s3_bucket_name
    }
    regions_in_s3_client_endpoint_url: Set[str] = {
        region
        for region in constants.JUMPSTART_REGION_NAME_SET
        if s3_client is not None
        if region in s3_client._endpoint.host
    }

    regions_in_sagemaker_session: Set[str] = {
        region
        for region in constants.JUMPSTART_REGION_NAME_SET
        if sagemaker_session
        if region == sagemaker_session.boto_region_name
    }

    combined_regions = regions_in_s3_client_endpoint_url.union(
        regions_in_s3_bucket_name, regions_in_sagemaker_session
    )

    if len(combined_regions) > 1:
        raise ValueError("Unable to resolve a region name from the s3 bucket and client provided.")

    if len(combined_regions) == 0:
        return constants.JUMPSTART_DEFAULT_REGION_NAME

    return list(combined_regions)[0]


def get_config_names(
    region: str,
    model_id: str,
    model_version: str,
    sagemaker_session: Optional[Session] = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    scope: enums.JumpStartScriptScope = enums.JumpStartScriptScope.INFERENCE,
    model_type: enums.JumpStartModelType = enums.JumpStartModelType.OPEN_WEIGHTS,
) -> List[str]:
    """Returns a list of config names for the given model ID and region."""
    model_specs = verify_model_region_and_return_specs(
        region=region,
        model_id=model_id,
        version=model_version,
        sagemaker_session=sagemaker_session,
        scope=scope,
        model_type=model_type,
    )

    if scope == enums.JumpStartScriptScope.INFERENCE:
        metadata_configs = model_specs.inference_configs
    elif scope == enums.JumpStartScriptScope.TRAINING:
        metadata_configs = model_specs.training_configs
    else:
        raise ValueError(f"Unknown script scope {scope}.")

    return list(metadata_configs.configs.keys()) if metadata_configs else []


def get_benchmark_stats(
    region: str,
    model_id: str,
    model_version: str,
    config_names: Optional[List[str]] = None,
    hub_arn: Optional[str] = None,
    sagemaker_session: Optional[Session] = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    scope: enums.JumpStartScriptScope = enums.JumpStartScriptScope.INFERENCE,
    model_type: enums.JumpStartModelType = enums.JumpStartModelType.OPEN_WEIGHTS,
) -> Dict[str, List[JumpStartBenchmarkStat]]:
    """Returns benchmark stats for the given model ID and region."""
    model_specs = verify_model_region_and_return_specs(
        region=region,
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        sagemaker_session=sagemaker_session,
        scope=scope,
        model_type=model_type,
    )

    if scope == enums.JumpStartScriptScope.INFERENCE:
        metadata_configs = model_specs.inference_configs
    elif scope == enums.JumpStartScriptScope.TRAINING:
        metadata_configs = model_specs.training_configs
    else:
        raise ValueError(f"Unknown script scope {scope}.")

    if not config_names:
        config_names = metadata_configs.configs.keys() if metadata_configs else []

    benchmark_stats = {}
    for config_name in config_names:
        if config_name not in metadata_configs.configs:
            raise ValueError(f"Unknown config name: '{config_name}'")
        benchmark_stats[config_name] = metadata_configs.configs.get(config_name).benchmark_metrics

    return benchmark_stats


def get_jumpstart_configs(
    region: str,
    model_id: str,
    model_version: str,
    config_names: Optional[List[str]] = None,
    sagemaker_session: Optional[Session] = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    scope: enums.JumpStartScriptScope = enums.JumpStartScriptScope.INFERENCE,
    model_type: enums.JumpStartModelType = enums.JumpStartModelType.OPEN_WEIGHTS,
) -> Dict[str, List[JumpStartMetadataConfig]]:
    """Returns metadata configs for the given model ID and region."""
    model_specs = verify_model_region_and_return_specs(
        region=region,
        model_id=model_id,
        version=model_version,
        sagemaker_session=sagemaker_session,
        scope=scope,
        model_type=model_type,
    )

    if scope == enums.JumpStartScriptScope.INFERENCE:
        metadata_configs = model_specs.inference_configs
    elif scope == enums.JumpStartScriptScope.TRAINING:
        metadata_configs = model_specs.training_configs
    else:
        raise ValueError(f"Unknown script scope {scope}.")

    if not config_names:
        config_names = metadata_configs.configs.keys() if metadata_configs else []

    return (
        {config_name: metadata_configs.configs[config_name] for config_name in config_names}
        if metadata_configs
        else {}
    )


def get_jumpstart_user_agent_extra_suffix(model_id: str, model_version: str) -> str:
    """Returns the model-specific user agent string to be added to requests."""
    sagemaker_python_sdk_headers = get_user_agent_extra_suffix()
    jumpstart_specific_suffix = f"md/js_model_id#{model_id} md/js_model_ver#{model_version}"
    return (
        sagemaker_python_sdk_headers
        if os.getenv(constants.ENV_VARIABLE_DISABLE_JUMPSTART_TELEMETRY, None)
        else f"{sagemaker_python_sdk_headers} {jumpstart_specific_suffix}"
    )


def get_default_jumpstart_session_with_user_agent_suffix(
    model_id: str, model_version: str
) -> Session:
    """Returns default JumpStart SageMaker Session with model-specific user agent suffix."""
    botocore_session = botocore.session.get_session()
    botocore_config = botocore.config.Config(
        user_agent_extra=get_jumpstart_user_agent_extra_suffix(model_id, model_version),
    )
    botocore_session.set_default_client_config(botocore_config)
    # shallow copy to not affect default session constant
    session = copy(constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION)
    session.boto_session = boto3.Session(
        region_name=constants.JUMPSTART_DEFAULT_REGION_NAME, botocore_session=botocore_session
    )
    session.sagemaker_client = boto3.client(
        "sagemaker", region_name=constants.JUMPSTART_DEFAULT_REGION_NAME, config=botocore_config
    )
    session.sagemaker_runtime_client = boto3.client(
        "sagemaker-runtime",
        region_name=constants.JUMPSTART_DEFAULT_REGION_NAME,
        config=botocore_config,
    )
    return session
