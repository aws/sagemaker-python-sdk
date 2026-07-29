from typing import Optional

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import HyperPodCompute


def validate_hyperpod_compute(
    compute: HyperPodCompute,
    sagemaker_session: Optional[Session] = None,
    is_nova: bool = False,
) -> None:
    """Validate that a HyperPod cluster has the required instance capacity for training.

    Checks that:
    - The cluster exists and is accessible
    - The requested instance type is available in the appropriate instance groups
    - Sufficient capacity exists for the requested node count

    For Nova models, validation is performed against restricted instance groups.
    For non-Nova models, validation is performed against normal instance groups.

    Args:
        compute: HyperPodCompute configuration with cluster_name, instance_type, and node_count.
        sagemaker_session: SageMaker session used to obtain boto clients and region.
            If None, a default Session will be created.
        is_nova: If True, validates against restricted instance groups; otherwise validates
            against normal instance groups.

    Raises:
        ValueError: If cluster_name is not specified in compute config.
        PermissionError: If the caller lacks sagemaker:DescribeCluster permission.
        RuntimeError: If the cluster cannot be described for non-permission reasons.
        ValueError: If instance type is not available or capacity is insufficient.
    """
    if sagemaker_session is None:
        sagemaker_session = Session()

    cluster_name = compute.cluster_name
    if not cluster_name:
        raise ValueError("HyperPod cluster name is required in compute configuration.")

    region_name = sagemaker_session.boto_session.region_name
    sagemaker_client = sagemaker_session.boto_session.client("sagemaker", region_name=region_name)

    try:
        response = sagemaker_client.describe_cluster(ClusterName=cluster_name)
    except Exception as e:
        if "AccessDenied" in str(e) or "UnauthorizedOperation" in str(e):
            raise PermissionError(
                "Missing SageMaker permissions: sagemaker:DescribeCluster required"
            ) from e
        raise RuntimeError(
            f"Failed to describe cluster '{cluster_name}': {str(e)}"
        ) from e

    # Gather instance groups from cluster response
    if is_nova:
        response_key = "RestrictedInstanceGroups"
        group_label = "restricted instance groups"
    else:
        response_key = "InstanceGroups"
        group_label = "instance groups"

    instance_groups = []
    for group in response.get(response_key, []):
        instance_groups.append({
            "instance_group_name": group["InstanceGroupName"],
            "instance_type": group["InstanceType"],
            "current_count": group["CurrentCount"],
            "target_count": group["TargetCount"],
            "status": group["Status"],
        })

    # Check if requested instance type exists in any instance group
    compatible_groups = [
        group for group in instance_groups
        if group["instance_type"] == compute.instance_type
    ]

    if not compatible_groups:
        available_types = sorted(set(
            group["instance_type"] for group in instance_groups
        ))
        raise ValueError(
            f"Instance type '{compute.instance_type}' not available in {group_label} "
            f"in cluster '{cluster_name}'. Available types: {available_types}"
        )

    # Check if any compatible group has sufficient capacity
    required_count = compute.node_count or 1
    sufficient_capacity = any(
        group["current_count"] >= required_count for group in compatible_groups
    )

    if not sufficient_capacity:
        max_available = max(group["current_count"] for group in compatible_groups)
        raise ValueError(
            f"Insufficient capacity for instance type '{compute.instance_type}' in cluster "
            f"'{cluster_name}'. Required: {required_count}, Maximum available: {max_available}"
        )
