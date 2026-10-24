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
"""Accessors to retrieve instance types.

``retrieve`` and ``retrieve_default`` read the instance types a JumpStart model supports.
``list_instance_types``, ``describe_instance_type``, and ``describe_instance_types`` read the
SageMaker hosting instance-type names from the installed botocore model and their hardware
from EC2 ``DescribeInstanceTypes``.
"""

from __future__ import absolute_import

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import boto3
from botocore.exceptions import ClientError

from sagemaker.core.jumpstart import utils as jumpstart_utils
from sagemaker.core.jumpstart import artifacts
from sagemaker.core.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.core.jumpstart.enums import JumpStartModelType
from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)

SAGEMAKER_INSTANCE_TYPE_PREFIX = "ml."
PRODUCTION_VARIANT_INSTANCE_TYPE_SHAPE = "ProductionVariantInstanceType"
# ``RequestInstanceTypeList`` in the EC2 service model allows 100 names per request.
DESCRIBE_INSTANCE_TYPES_BATCH_SIZE = 100
INVALID_INSTANCE_TYPE_ERROR_CODE = "InvalidInstanceType"


@dataclass(frozen=True)
class InstanceSpec:
    """Hardware of one SageMaker instance type as EC2 ``DescribeInstanceTypes`` reports it.

    Attributes:
        name: The SageMaker instance type, for example ``ml.g5.xlarge``.
        vcpus: The default vCPU count, or ``None`` when EC2 does not know the type.
        memory_mib: The instance memory in MiB, or ``None`` when EC2 does not know the type.
        gpu_count: The number of GPUs. ``0`` for a CPU instance and for an unknown type.
        gpu_memory_mib: The GPU memory in MiB summed over all GPUs, or ``None`` without GPUs.
        gpu_name: The GPU model name, for example ``A10G``, or ``None`` without GPUs.
    """

    name: str
    vcpus: Optional[int]
    memory_mib: Optional[int]
    gpu_count: int
    gpu_memory_mib: Optional[int]
    gpu_name: Optional[str]


def retrieve_default(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    scope: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    training_instance_type: Optional[str] = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    config_name: Optional[str] = None,
) -> str:
    """Retrieves the default instance type for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the default instance type.
            Defaults to ``None``.
        model_id (str): The model ID of the model for which to
            retrieve the default instance type. (Default: None).
        model_version (str): The version of the model for which to retrieve the
            default instance type. (Default: None).
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (default: None).
        scope (str): The model type, i.e. what it is used for.
            Valid values: "training" and "inference".
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
        training_instance_type (str): In the case of a model fine-tuned on SageMaker, the training
            instance type used for the training job that produced the fine-tuned weights.
            Optionally supply this to get a inference instance type conditioned
            on the training instance, to ensure compatability of training artifact to inference
            instance. (Default: None).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        str: The default instance type to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` when retrieving instance types."
        )

    if scope is None:
        raise ValueError("Must specify scope for instance types.")

    return artifacts._retrieve_default_instance_type(
        model_id,
        model_version,
        scope,
        hub_arn,
        region,
        tolerate_vulnerable_model,
        tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        training_instance_type=training_instance_type,
        model_type=model_type,
        config_name=config_name,
    )


def retrieve(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    scope: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    training_instance_type: Optional[str] = None,
) -> List[str]:
    """Retrieves the supported training instance types for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the supported instance types.
            Defaults to ``None``.
        model_id (str): The model ID of the model for which to
            retrieve the supported instance types. (Default: None).
        model_version (str): The version of the model for which to retrieve the
            supported instance types. (Default: None).
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
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
        training_instance_type (str): In the case of a model fine-tuned on SageMaker, the training
            instance type used for the training job that produced the fine-tuned weights.
            Optionally supply this to get a inference instance type conditioned
            on the training instance, to ensure compatability of training artifact to inference
            instance. (Default: None).

    Returns:
        list: The supported instance types to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` when retrieving instance types."
        )

    if scope is None:
        raise ValueError("Must specify scope for instance types.")

    return artifacts._retrieve_instance_types(
        model_id=model_id,
        model_version=model_version,
        scope=scope,
        hub_arn=hub_arn,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        training_instance_type=training_instance_type,
    )


def _boto_session(session: Optional[boto3.Session]) -> boto3.Session:
    """Returns the given session, or a new default session."""
    if session is not None:
        return session
    return boto3.Session()


def _ec2_instance_type(instance_type: str) -> str:
    """Returns the EC2 name of a SageMaker instance type (``ml.g5.xlarge`` to ``g5.xlarge``)."""
    if instance_type.startswith(SAGEMAKER_INSTANCE_TYPE_PREFIX):
        return instance_type[len(SAGEMAKER_INSTANCE_TYPE_PREFIX) :]
    return instance_type


def _unknown_spec(instance_type: str) -> InstanceSpec:
    """Returns the spec of an instance type that EC2 does not know, and logs a warning."""
    logger.warning(
        "EC2 DescribeInstanceTypes does not know instance type %s. "
        "Its InstanceSpec has gpu_count=0 and no other hardware.",
        instance_type,
    )
    return InstanceSpec(
        name=instance_type,
        vcpus=None,
        memory_mib=None,
        gpu_count=0,
        gpu_memory_mib=None,
        gpu_name=None,
    )


def _spec_from_ec2(instance_type: str, instance_info: Dict[str, Any]) -> InstanceSpec:
    """Builds an ``InstanceSpec`` from one EC2 ``InstanceTypeInfo`` element."""
    gpus = instance_info.get("GpuInfo", {}).get("Gpus", [])
    gpu_count = sum(gpu.get("Count", 0) for gpu in gpus)
    first_gpu = gpus[0] if gpus else {}
    memory_per_gpu_mib = first_gpu.get("MemoryInfo", {}).get("SizeInMiB")
    gpu_memory_mib = None
    if gpu_count > 0 and memory_per_gpu_mib is not None:
        gpu_memory_mib = memory_per_gpu_mib * gpu_count
    return InstanceSpec(
        name=instance_type,
        vcpus=instance_info.get("VCpuInfo", {}).get("DefaultVCpus"),
        memory_mib=instance_info.get("MemoryInfo", {}).get("SizeInMiB"),
        gpu_count=gpu_count,
        gpu_memory_mib=gpu_memory_mib,
        gpu_name=first_gpu.get("Name"),
    )


def _describe_ec2_instance_types(
    ec2_client: Any, instance_types: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Calls EC2 ``DescribeInstanceTypes`` for the names and reads every page of the response."""
    request: Dict[str, Any] = {
        "InstanceTypes": [_ec2_instance_type(instance_type) for instance_type in instance_types]
    }
    described: Dict[str, Dict[str, Any]] = {}
    while True:
        response = ec2_client.describe_instance_types(**request)
        for instance_info in response.get("InstanceTypes", []):
            described[instance_info["InstanceType"]] = instance_info
        next_token = response.get("NextToken")
        # A real token is a non-empty string. The type check also stops a client stub
        # whose ``get`` returns a truthy Mock from looping forever.
        if not isinstance(next_token, str) or not next_token:
            return described
        request["NextToken"] = next_token


def _describe_known(ec2_client: Any, instance_types: List[str]) -> Dict[str, Dict[str, Any]]:
    """Returns the EC2 ``InstanceTypeInfo`` of each name that EC2 knows, keyed by EC2 name.

    EC2 answers ``InvalidInstanceType`` for the whole request when one name does not exist.
    The function then asks for each name alone, so one unknown name hides no other.
    """
    try:
        return _describe_ec2_instance_types(ec2_client, instance_types)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") != INVALID_INSTANCE_TYPE_ERROR_CODE:
            raise
    if len(instance_types) == 1:
        return {}
    described: Dict[str, Dict[str, Any]] = {}
    for instance_type in instance_types:
        described.update(_describe_known(ec2_client, [instance_type]))
    return described


def _describe_batch(ec2_client: Any, instance_types: List[str]) -> Dict[str, InstanceSpec]:
    """Describes one request of at most 100 instance types and maps each name to a spec."""
    described = _describe_known(ec2_client, instance_types)
    specs: Dict[str, InstanceSpec] = {}
    for instance_type in instance_types:
        instance_info = described.get(_ec2_instance_type(instance_type))
        if instance_info is None:
            specs[instance_type] = _unknown_spec(instance_type)
        else:
            specs[instance_type] = _spec_from_ec2(instance_type, instance_info)
    return specs


def list_instance_types(
    region: Optional[str] = None, session: Optional[boto3.Session] = None
) -> List[str]:
    """Lists the SageMaker hosting instance types that the loaded service model knows.

    The names come from the ``ProductionVariantInstanceType`` enum of the SageMaker service
    model that the session loads, so the call makes no request. The list holds the
    service-wide set and keeps types the region does not offer.

    Args:
        region (str): The region of the SageMaker client. Defaults to the session region.
        session (boto3.Session): The session that creates the client. Defaults to a new session.

    Returns:
        list[str]: The instance type names in model order, for example ``ml.t2.medium`` first.

    Raises:
        botocore.exceptions.NoRegionError: If neither ``region`` nor the session has a region.
    """
    sagemaker_client = _boto_session(session).client("sagemaker", region_name=region)
    shape = sagemaker_client.meta.service_model.shape_for(PRODUCTION_VARIANT_INSTANCE_TYPE_SHAPE)
    return list(shape.enum)


def describe_instance_types(
    instance_types: Sequence[str], session: Optional[boto3.Session] = None
) -> Dict[str, InstanceSpec]:
    """Describes the hardware of SageMaker instance types through EC2 ``DescribeInstanceTypes``.

    The function strips the ``ml.`` prefix before the EC2 call and sends 100 names per request,
    the EC2 limit. EC2 rejects a whole request when one name does not exist, so the function
    then describes each name of that request alone. An instance type that EC2 does not know
    (a retired type, or a type the region does not offer) gets an ``InstanceSpec`` with
    ``gpu_count=0`` and ``None`` in every other field. The function logs a warning for it.

    Args:
        instance_types (Sequence[str]): The SageMaker instance types, for example
            ``["ml.g5.xlarge", "ml.m5.large"]``.
        session (boto3.Session): The session that creates the EC2 client. Defaults to a new
            session.

    Returns:
        dict[str, InstanceSpec]: One ``InstanceSpec`` per requested name, keyed by that name.

    Raises:
        botocore.exceptions.ClientError: If EC2 rejects a request for a reason other than an
            unknown instance type, for example missing ``ec2:DescribeInstanceTypes`` permission.
    """
    ec2_client = _boto_session(session).client("ec2")
    specs: Dict[str, InstanceSpec] = {}
    for start in range(0, len(instance_types), DESCRIBE_INSTANCE_TYPES_BATCH_SIZE):
        batch = list(instance_types[start : start + DESCRIBE_INSTANCE_TYPES_BATCH_SIZE])
        specs.update(_describe_batch(ec2_client, batch))
    return specs


def describe_instance_type(
    instance_type: str, session: Optional[boto3.Session] = None
) -> InstanceSpec:
    """Describes the hardware of one SageMaker instance type through EC2.

    Args:
        instance_type (str): The SageMaker instance type, for example ``ml.g5.xlarge``.
        session (boto3.Session): The session that creates the EC2 client. Defaults to a new
            session.

    Returns:
        InstanceSpec: The hardware. An instance type that EC2 does not know gets
        ``gpu_count=0`` and ``None`` in every other field. See :func:`describe_instance_types`.
    """
    return describe_instance_types([instance_type], session=session)[instance_type]
