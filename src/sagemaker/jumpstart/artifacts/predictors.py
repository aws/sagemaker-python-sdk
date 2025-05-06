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
"""This module contains functions for obtaining JumpStart predictors."""
from __future__ import absolute_import
from typing import List, Optional, Set, Type
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.jumpstart.constants import (
    ACCEPT_TYPE_TO_DESERIALIZER_TYPE_MAP,
    CONTENT_TYPE_TO_SERIALIZER_TYPE_MAP,
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    DESERIALIZER_TYPE_TO_CLASS_MAP,
    SERIALIZER_TYPE_TO_CLASS_MAP,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    MIMEType,
    JumpStartModelType,
)
from sagemaker.jumpstart.utils import (
    get_region_fallback,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


def _retrieve_serializer_from_content_type(
    content_type: MIMEType,
) -> BaseDeserializer:
    """Returns serializer object to use for content type."""

    serializer_type = CONTENT_TYPE_TO_SERIALIZER_TYPE_MAP.get(content_type)

    if serializer_type is None:
        raise RuntimeError(f"Unrecognized content type: {content_type}")

    serializer_handle = SERIALIZER_TYPE_TO_CLASS_MAP.get(serializer_type)

    if serializer_handle is None:
        raise RuntimeError(f"Unrecognized serializer type: {serializer_type}")

    return serializer_handle.__call__()


def _retrieve_deserializer_from_accept_type(
    accept_type: MIMEType,
) -> BaseDeserializer:
    """Returns deserializer object to use for accept type."""

    deserializer_type = ACCEPT_TYPE_TO_DESERIALIZER_TYPE_MAP.get(accept_type)

    if deserializer_type is None:
        raise RuntimeError(f"Unrecognized accept type: {accept_type}")

    deserializer_handle = DESERIALIZER_TYPE_TO_CLASS_MAP.get(deserializer_type)

    if deserializer_handle is None:
        raise RuntimeError(f"Unrecognized deserializer type: {deserializer_type}")

    return deserializer_handle.__call__()


def _retrieve_default_deserializer(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str],
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    config_name: Optional[str] = None,
) -> BaseDeserializer:
    """Retrieves the default deserializer for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default deserializer.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default deserializer.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve default deserializer.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).

    Returns:
        BaseDeserializer: the default deserializer to use for the model.
    """

    default_accept_type = _retrieve_default_accept_type(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
        config_name=config_name,
    )

    return _retrieve_deserializer_from_accept_type(MIMEType.from_suffixed_type(default_accept_type))


def _retrieve_default_serializer(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str],
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    config_name: Optional[str] = None,
) -> BaseSerializer:
    """Retrieves the default serializer for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default serializer.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default serializer.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve default serializer.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        BaseSerializer: the default serializer to use for the model.
    """

    default_content_type = _retrieve_default_content_type(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
        config_name=config_name,
    )

    return _retrieve_serializer_from_content_type(MIMEType.from_suffixed_type(default_content_type))


def _retrieve_deserializer_options(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str],
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    config_name: Optional[str] = None,
) -> List[BaseDeserializer]:
    """Retrieves the supported deserializers for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported deserializers.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported deserializers.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve deserializer options.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        List[BaseDeserializer]: the supported deserializers to use for the model.
    """

    supported_accept_types = _retrieve_supported_accept_types(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
        config_name=config_name,
    )

    seen_classes: Set[Type] = set()

    deserializers_with_duplicates: List[BaseDeserializer] = [
        _retrieve_deserializer_from_accept_type(MIMEType.from_suffixed_type(accept_type))
        for accept_type in supported_accept_types
    ]

    deserializers: List[BaseDeserializer] = []

    for deserializer in deserializers_with_duplicates:
        if type(deserializer) not in seen_classes:
            seen_classes.add(type(deserializer))
            deserializers.append(deserializer)

    return deserializers


def _retrieve_serializer_options(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str],
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    config_name: Optional[str] = None,
) -> List[BaseSerializer]:
    """Retrieves the supported serializers for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported serializers.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported serializers.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve serializer options.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        List[BaseSerializer]: the supported serializers to use for the model.
    """

    supported_content_types = _retrieve_supported_content_types(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        config_name=config_name,
    )

    seen_classes: Set[Type] = set()

    serializers_with_duplicates: List[BaseSerializer] = [
        _retrieve_serializer_from_content_type(MIMEType.from_suffixed_type(content_type))
        for content_type in supported_content_types
    ]

    serializers: List[BaseSerializer] = []

    for serializer in serializers_with_duplicates:
        if type(serializer) not in seen_classes:
            seen_classes.add(type(serializer))
            serializers.append(serializer)

    return serializers


def _retrieve_default_content_type(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str],
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    config_name: Optional[str] = None,
) -> str:
    """Retrieves the default content type for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default content type.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default content type.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve default content type.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        str: the default content type to use for the model.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
        config_name=config_name,
    )

    default_content_type = model_specs.predictor_specs.default_content_type
    return default_content_type


def _retrieve_default_accept_type(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str],
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    config_name: Optional[str] = None,
) -> str:
    """Retrieves the default accept type for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default accept type.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default accept type.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve default accept type.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        str: the default accept type to use for the model.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
        config_name=config_name,
    )

    default_accept_type = model_specs.predictor_specs.default_accept_type

    return default_accept_type


def _retrieve_supported_accept_types(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str],
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    config_name: Optional[str] = None,
) -> List[str]:
    """Retrieves the supported accept types for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported accept types.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported accept types.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve accept type options.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        list: the supported accept types to use for the model.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
        config_name=config_name,
    )

    supported_accept_types = model_specs.predictor_specs.supported_accept_types

    return supported_accept_types


def _retrieve_supported_content_types(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str],
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    config_name: Optional[str] = None,
) -> List[str]:
    """Retrieves the supported content types for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported content types.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported content types.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve content type options.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        list: the supported content types to use for the model.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
        config_name=config_name,
    )

    supported_content_types = model_specs.predictor_specs.supported_content_types

    return supported_content_types
