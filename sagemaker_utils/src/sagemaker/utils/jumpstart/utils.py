from typing import Any, Dict, List, Set, Optional, Tuple, Union
import boto3
from sagemaker_core.helper.session_helper import Session
from sagemaker.utils.jumpstart import constants
from sagemaker.utils.jumpstart import enums
from sagemaker.utils.jumpstart.exceptions import (
    DeprecatedJumpStartModelError,
    VulnerableJumpStartModelError,
    get_old_model_version_msg,
)

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
    config_name: Optional[str] = None,
):
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
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).

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

    if model_specs and config_name:
        model_specs.set_config(config_name, scope)

    return model_specs
